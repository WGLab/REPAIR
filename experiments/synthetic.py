"""
Reproduce Figure 2: Synthetic experiments (unified DGP).
=========================================================
Runs 4 experiments on a synthetic long-tailed classification setup:

  Exp 1 (Fig 2A/B): Two-regime comparison
    - Class-separable: CW ~= REPAIR (pairwise adds little)
    - Contradictory: REPAIR > CW (pairwise helps significantly)

  Exp 2 (Fig 3a): Ablation bar chart (Base / PW-only / CW-only / REPAIR)

  Exp 3 (Fig 3b): Contradiction quintile analysis (5 seeds)
    - Classes binned by contradiction score D_yj
    - Pairwise gain increases monotonically Q1 -> Q5

  Exp 4 (Fig 3c): Shrinkage validation (10 seeds)
    - MLE offset variance vs. shrunk offset variance
    - Shrinkage reduces variance, improves tail accuracy

DGP:
  g_y(x) = log p(y|x) + bias_y + noise
  K=20, d=10, k=5, Zipf class frequencies

Output: results/toy/toy_unified_results.json

Usage:
    python -m experiments.synthetic
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from collections import defaultdict
import json
import time
from pathlib import Path

SEED = 42
K = 20
D = 10
K_SHORT = 5
N_TRAIN = 5000
N_TEST = 2000
PHI_DIM = 3  # [score_gap, rank_gap, cosine_similarity]

CORRUPTION_FACTOR = 0.8
SIGMA_NOISE = 0.3
CONFUSER_PAIRS = [(5 + i, i % 5) for i in range(15)]
CONFUSION_STRENGTH = 3.0

OUT = Path(__file__).resolve().parent.parent / "results" / "toy"
OUT.mkdir(parents=True, exist_ok=True)


# ── Shared DGP ──

def make_population(seed=SEED):
    rng = np.random.RandomState(seed)
    raw = np.array([1.0 / (y + 1) for y in range(K)])
    pi = raw / raw.sum()
    mu = rng.multivariate_normal(np.zeros(D), 0.5 * np.eye(D), size=K)
    counts = (pi * N_TRAIN).astype(int)
    counts = np.maximum(counts, 1)
    return pi, mu, counts


def generate_data(n, pi, mu, corruption_factor, confuser_pairs=None,
                  confusion_strength=0.0, rng=None):
    rng = rng or np.random.RandomState(SEED)
    tau = 1.0
    labels = rng.choice(K, size=n, p=pi)
    X = np.zeros((n, D))
    logits = np.zeros((n, K))

    for i in range(n):
        X[i] = rng.multivariate_normal(mu[labels[i]], np.eye(D))
        for c in range(K):
            logits[i, c] = (np.log(pi[c] + 1e-15)
                            - 0.5 * np.sum((X[i] - mu[c]) ** 2)
                            - tau * np.log(pi[c] + 1e-15) * corruption_factor)
        logits[i] += rng.normal(0, SIGMA_NOISE, size=K)

    if confuser_pairs and confusion_strength > 0:
        S = np.argsort(logits, axis=1)[:, -K_SHORT:][:, ::-1]
        for i in range(n):
            Si = set(S[i].tolist())
            for (a, b) in confuser_pairs:
                if a in Si and b in Si:
                    sign = rng.choice([-1, 1])
                    logits[i, a] += sign * confusion_strength
                    logits[i, b] -= sign * confusion_strength
    return X, labels, logits


# ── Phi features + fitting + evaluation ──

def get_shortlists(logits, k):
    return np.argsort(logits, axis=1)[:, -k:][:, ::-1]


def precompute_phi(logits, shortlists, mu):
    N, k = shortlists.shape
    g_S = np.take_along_axis(logits, shortlists, axis=1)
    mu_S = mu[shortlists]
    mu_norms = np.linalg.norm(mu, axis=1)
    mu_norms_S = mu_norms[shortlists]
    phi = np.zeros((N, k, k, PHI_DIM))
    for y in range(k):
        for j in range(k):
            if y == j:
                continue
            phi[:, y, j, 0] = g_S[:, y] - g_S[:, j]
            phi[:, y, j, 1] = float(y - j)
            dot = np.sum(mu_S[:, y, :] * mu_S[:, j, :], axis=1)
            denom = mu_norms_S[:, y] * mu_norms_S[:, j] + 1e-10
            phi[:, y, j, 2] = dot / denom
    return phi


def compute_scores(g_S, a_S, phi_all, theta):
    N, k = g_S.shape
    pw = np.einsum("nyjd,d->nyj", phi_all, theta)
    np.einsum("nii->ni", pw)[:] = 0
    return g_S + a_S + pw.sum(axis=2) / max(k - 1, 1)


def fit_repair(logits, labels, shortlists, phi_all,
               fit_cw=True, fit_pw=True,
               lambda_a=0.01, lambda_theta=0.001):
    N, k = shortlists.shape
    g_S = np.take_along_axis(logits, shortlists, axis=1)
    cov = (shortlists == labels[:, None]).any(axis=1)
    yp = np.zeros(N, dtype=int)
    for i in np.where(cov)[0]:
        yp[i] = np.where(shortlists[i] == labels[i])[0][0]
    ci = np.where(cov)[0]; nc = len(ci)
    if nc < 5:
        return np.zeros(K), np.zeros(PHI_DIM)
    gc = g_S[ci]; pc = phi_all[ci]; Sc = shortlists[ci]; yc = yp[ci]
    na = K if fit_cw else 0; nt = PHI_DIM if fit_pw else 0

    def lg(params):
        a = np.zeros(K); t = np.zeros(PHI_DIM)
        if fit_cw: a = params[:K]
        if fit_pw: t = params[na:na + nt]
        pw = np.einsum("nyjd,d->nyj", pc, t)
        np.einsum("nii->ni", pw)[:] = 0
        sc = gc + a[Sc] + pw.sum(axis=2) / max(k - 1, 1)
        lp = sc - logsumexp(sc, axis=1)[:, None]
        nll = -lp[np.arange(nc), yc].mean()
        pr = np.exp(lp)
        oh = np.zeros_like(pr); oh[np.arange(nc), yc] = 1.0
        ds = (pr - oh) / nc
        g = np.zeros(na + nt)
        if fit_cw:
            ga = np.zeros(K); np.add.at(ga, Sc.ravel(), ds.ravel())
            g[:K] = ga + 2 * lambda_a * a
        if fit_pw:
            g[na:na + nt] = (np.einsum("ny,nyd->d", ds,
                              pc.sum(axis=2) / max(k - 1, 1))
                             + 2 * lambda_theta * t)
        loss = nll
        if fit_cw: loss += lambda_a * (a ** 2).sum()
        if fit_pw: loss += lambda_theta * (t ** 2).sum()
        return loss, g

    res = minimize(lg, np.zeros(na + nt), jac=True, method="L-BFGS-B",
                   options={"maxiter": 200})
    a_out = np.zeros(K); t_out = np.zeros(PHI_DIM)
    if fit_cw: a_out = res.x[:K]
    if fit_pw: t_out = res.x[na:na + nt]
    return a_out, t_out


def syn_evaluate(logits, labels, shortlists, a, theta, phi_all):
    N, k = shortlists.shape
    g_S = np.take_along_axis(logits, shortlists, axis=1)
    scores = compute_scores(g_S, a[shortlists], phi_all, theta)
    preds = shortlists[np.arange(N), scores.argmax(axis=1)]
    top1 = (preds == labels).mean()
    covered = np.array([labels[i] in shortlists[i] for i in range(N)])
    recall_k = covered.mean()
    base_preds = shortlists[np.arange(N), g_S.argmax(axis=1)]
    base_acc = (base_preds == labels).mean()
    rho_k = (top1 - base_acc) / max(recall_k - base_acc, 1e-10)
    # HFR
    hfr_s, hfr_n = 0, 0
    for i in range(N):
        y = labels[i]; Si = shortlists[i].tolist()
        if y not in Si: continue
        yi = Si.index(y)
        if Si[g_S[i].argmax()] != y:
            hi = max((g_S[i, j2], j2) for j2 in range(k) if j2 != yi)[1]
            if scores[i, yi] > scores[i, hi]: hfr_s += 1
            hfr_n += 1
    return {"top1": top1, "recall_k": recall_k, "rho_k": rho_k,
            "hfr": hfr_s / max(hfr_n, 1)}


# ══════════════════════════════════════════════════════════
# Experiments
# ══════════════════════════════════════════════════════════

def exp_regimes():
    """Exp 1: Two-regime comparison (Fig 2 A/B)."""
    print("=" * 60)
    print("Exp 1: Two-regime comparison")
    print("=" * 60)
    pi, mu, counts = make_population()
    results = {}

    for regime, cp, cs in [("A_class_separable", None, 0.0),
                            ("B_contradictory", CONFUSER_PAIRS, CONFUSION_STRENGTH)]:
        _, trn_y, trn_l = generate_data(N_TRAIN, pi, mu, CORRUPTION_FACTOR,
                                         cp, cs, np.random.RandomState(SEED + 1))
        _, tst_y, tst_l = generate_data(N_TEST, pi, mu, CORRUPTION_FACTOR,
                                         cp, cs, np.random.RandomState(SEED + 2))
        S_trn = get_shortlists(trn_l, K_SHORT)
        S_tst = get_shortlists(tst_l, K_SHORT)
        phi_trn = precompute_phi(trn_l, S_trn, mu)
        phi_tst = precompute_phi(tst_l, S_tst, mu)
        a0, t0 = np.zeros(K), np.zeros(PHI_DIM)

        r_base = syn_evaluate(tst_l, tst_y, S_tst, a0, t0, phi_tst)
        a_cw, _ = fit_repair(trn_l, trn_y, S_trn, phi_trn, True, False)
        r_cw = syn_evaluate(tst_l, tst_y, S_tst, a_cw, t0, phi_tst)
        a_f, t_f = fit_repair(trn_l, trn_y, S_trn, phi_trn, True, True)
        r_full = syn_evaluate(tst_l, tst_y, S_tst, a_f, t_f, phi_tst)

        results[regime] = {
            "base": {k: float(v) for k, v in r_base.items()},
            "classwise": {k: float(v) for k, v in r_cw.items()},
            "repair": {k: float(v) for k, v in r_full.items()},
            "recall": float(r_base["recall_k"]),
        }
        print(f"\n  {regime}:")
        for nm, r in [("Base", r_base), ("CW", r_cw), ("REPAIR", r_full)]:
            print(f"    {nm:<8} Top1={r['top1']:.3f}  rho={r['rho_k']:.3f}")
    return results


def exp_ablation():
    """Exp 2: Ablation (Fig 3a)."""
    print("\n" + "=" * 60)
    print("Exp 2: Ablation (both regimes)")
    print("=" * 60)
    pi, mu, counts = make_population()
    results_both = {}

    for regime, cp, cs in [("class_separable", None, 0.0),
                            ("contradictory", CONFUSER_PAIRS, CONFUSION_STRENGTH)]:
        _, trn_y, trn_l = generate_data(N_TRAIN, pi, mu, CORRUPTION_FACTOR,
                                         cp, cs, np.random.RandomState(SEED + 1))
        _, tst_y, tst_l = generate_data(N_TEST, pi, mu, CORRUPTION_FACTOR,
                                         cp, cs, np.random.RandomState(SEED + 2))
        S_trn = get_shortlists(trn_l, K_SHORT)
        S_tst = get_shortlists(tst_l, K_SHORT)
        phi_trn = precompute_phi(trn_l, S_trn, mu)
        phi_tst = precompute_phi(tst_l, S_tst, mu)
        a0, t0 = np.zeros(K), np.zeros(PHI_DIM)
        pw_reg = 0.5 if cp is None else 0.001

        r_base = syn_evaluate(tst_l, tst_y, S_tst, a0, t0, phi_tst)
        _, t_pw = fit_repair(trn_l, trn_y, S_trn, phi_trn, False, True,
                             lambda_theta=pw_reg)
        r_pw = syn_evaluate(tst_l, tst_y, S_tst, a0, t_pw, phi_tst)
        a_cw, _ = fit_repair(trn_l, trn_y, S_trn, phi_trn, True, False)
        r_cw = syn_evaluate(tst_l, tst_y, S_tst, a_cw, t0, phi_tst)
        a_f, t_f = fit_repair(trn_l, trn_y, S_trn, phi_trn, True, True,
                              lambda_theta=pw_reg)
        r_full = syn_evaluate(tst_l, tst_y, S_tst, a_f, t_f, phi_tst)

        results_both[regime] = {
            "base": float(r_base["top1"]),
            "pairwise_only": float(r_pw["top1"]),
            "classwise_only": float(r_cw["top1"]),
            "full_repair": float(r_full["top1"]),
        }
        print(f"\n  {regime}:")
        for k, v in results_both[regime].items():
            print(f"    {k:<20} {v * 100:.1f}%")
    return results_both


def exp_quintile():
    """Exp 3: Contradiction quintile (Fig 3b) -- 5 seeds."""
    print("\n" + "=" * 60)
    print("Exp 3: Contradiction quintile (5 seeds)")
    print("=" * 60)
    all_gains = [[] for _ in range(5)]

    for s in range(5):
        pi, mu, _ = make_population(seed=SEED + s * 77)
        _, trn_y, trn_l = generate_data(N_TRAIN, pi, mu, CORRUPTION_FACTOR,
                                         CONFUSER_PAIRS, CONFUSION_STRENGTH,
                                         np.random.RandomState(SEED + 10 + s))
        _, tst_y, tst_l = generate_data(N_TEST, pi, mu, CORRUPTION_FACTOR,
                                         CONFUSER_PAIRS, CONFUSION_STRENGTH,
                                         np.random.RandomState(SEED + 20 + s))
        S_trn = get_shortlists(trn_l, K_SHORT)
        S_tst = get_shortlists(tst_l, K_SHORT)
        phi_trn = precompute_phi(trn_l, S_trn, mu)
        phi_tst = precompute_phi(tst_l, S_tst, mu)
        N = len(tst_l)

        # Contradiction scores
        pt = defaultdict(list)
        for i in range(N):
            Si = S_tst[i].tolist()
            y = tst_y[i]
            if y not in Si: continue
            for j in Si:
                if j == y: continue
                pt[(y, j)].append(tst_l[i, j] - tst_l[i, y])
        D_yj = {p: np.std(ts) for p, ts in pt.items() if len(ts) >= 3}
        ec = np.zeros(N)
        for i in range(N):
            Si = S_tst[i].tolist()
            y = tst_y[i]
            if y not in Si: continue
            mx = 0.0
            for j in Si:
                if j == y: continue
                mx = max(mx, D_yj.get((y, j), 0))
            ec[i] = mx

        a0, t0 = np.zeros(K), np.zeros(PHI_DIM)
        ac, _ = fit_repair(trn_l, trn_y, S_trn, phi_trn, True, False)
        af, tf = fit_repair(trn_l, trn_y, S_trn, phi_trn, True, True,
                            lambda_theta=0.01)

        qs = np.percentile(ec, [20, 40, 60, 80])
        bins = np.digitize(ec, qs)
        g_S = np.take_along_axis(tst_l, S_tst, axis=1)
        scw = compute_scores(g_S, ac[S_tst], phi_tst, t0)
        sfu = compute_scores(g_S, af[S_tst], phi_tst, tf)
        pcw = S_tst[np.arange(N), scw.argmax(axis=1)]
        pfu = S_tst[np.arange(N), sfu.argmax(axis=1)]

        for q in range(5):
            idx = np.where(bins == q)[0]
            if len(idx) == 0:
                all_gains[q].append(0); continue
            cc, cf, tot = 0, 0, 0
            for ii in idx:
                y = tst_y[ii]
                if y in S_tst[ii]:
                    tot += 1; cc += (pcw[ii] == y); cf += (pfu[ii] == y)
            all_gains[q].append((cf - cc) / max(tot, 1) * 100)

    results = {
        "labels": ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"],
        "pw_gain_mean": [float(np.mean(g)) for g in all_gains],
        "pw_gain_std": [float(np.std(g)) for g in all_gains],
    }
    print(f"\n  {'Q':<12} {'Mean':>8} {'Std':>8}")
    for q in range(5):
        print(f"  {results['labels'][q]:<12} {results['pw_gain_mean'][q]:>+8.1f} "
              f"{results['pw_gain_std'][q]:>8.1f}")
    return results


def exp_shrinkage():
    """Exp 4: Shrinkage validation (Fig 3c) -- 10 seeds."""
    print("\n" + "=" * 60)
    print("Exp 4: Shrinkage validation (10 seeds)")
    print("=" * 60)
    pi, mu, counts = make_population()
    tail = list(np.argsort(counts)[:K // 2])

    # Fixed test set
    _, tst_y, tst_l = generate_data(N_TEST, pi, mu, CORRUPTION_FACTOR,
                                     rng=np.random.RandomState(SEED + 20))
    S_tst = get_shortlists(tst_l, K_SHORT)
    g_S_tst = np.take_along_axis(tst_l, S_tst, axis=1)
    phi_tst = precompute_phi(tst_l, S_tst, mu)
    t0 = np.zeros(PHI_DIM)

    cal_sizes = [2, 5, 10, 20, 50]
    results = {"cal_sizes": cal_sizes,
               "mle_var_mean": [], "mle_var_std": [],
               "shrunk_var_mean": [], "shrunk_var_std": [],
               "mle_tail_mean": [], "mle_tail_std": [],
               "shrunk_tail_mean": [], "shrunk_tail_std": []}

    for n_cal in cal_sizes:
        seed_mv, seed_sv, seed_mt, seed_st = [], [], [], []
        for s in range(10):
            mle_offs_list, shrunk_offs_list = [], []
            tail_m_list, tail_s_list = [], []
            for b in range(15):
                rng_b = np.random.RandomState(SEED + 1000 * s + b)
                X_cal, y_cal = [], []
                for c in range(K):
                    xc = rng_b.multivariate_normal(mu[c], np.eye(D), size=n_cal)
                    X_cal.append(xc)
                    y_cal.append(np.full(n_cal, c))
                X_cal = np.vstack(X_cal)
                y_cal = np.concatenate(y_cal)
                cal_logits = np.zeros((len(y_cal), K))
                for i in range(len(y_cal)):
                    for c in range(K):
                        cal_logits[i, c] = (np.log(pi[c] + 1e-15)
                                            - 0.5 * np.sum((X_cal[i] - mu[c]) ** 2)
                                            - 1.0 * np.log(pi[c] + 1e-15) * CORRUPTION_FACTOR)
                    cal_logits[i] += rng_b.normal(0, SIGMA_NOISE, size=K)

                S_cal = get_shortlists(cal_logits, K_SHORT)
                phi_cal = precompute_phi(cal_logits, S_cal, mu)
                a_mle, _ = fit_repair(cal_logits, y_cal, S_cal, phi_cal,
                                      True, False, lambda_a=0.001)
                mle_offs_list.append(a_mle)

                # Shrinkage
                a_bar = a_mle.mean()
                tau_sq = np.var(a_mle)
                se = 1.0 / np.sqrt(np.maximum(
                    np.array([np.sum(y_cal == c) for c in range(K)]), 1))
                B = np.maximum(0, 1 - se ** 2 / (tau_sq + 1e-10))
                a_shrunk = a_bar + B * (a_mle - a_bar)
                shrunk_offs_list.append(a_shrunk)

                # Tail accuracy
                scores_m = compute_scores(g_S_tst, a_mle[S_tst], phi_tst, t0)
                scores_s = compute_scores(g_S_tst, a_shrunk[S_tst], phi_tst, t0)
                preds_m = S_tst[np.arange(N_TEST), scores_m.argmax(axis=1)]
                preds_s = S_tst[np.arange(N_TEST), scores_s.argmax(axis=1)]
                mask = np.isin(tst_y, tail)
                tail_m_list.append((preds_m[mask] == tst_y[mask]).mean() * 100)
                tail_s_list.append((preds_s[mask] == tst_y[mask]).mean() * 100)

            mle_var = np.mean([np.var(a) for a in mle_offs_list])
            shrunk_var = np.mean([np.var(a) for a in shrunk_offs_list])
            seed_mv.append(mle_var); seed_sv.append(shrunk_var)
            seed_mt.append(np.mean(tail_m_list)); seed_st.append(np.mean(tail_s_list))

        results["mle_var_mean"].append(float(np.mean(seed_mv)))
        results["mle_var_std"].append(float(np.std(seed_mv)))
        results["shrunk_var_mean"].append(float(np.mean(seed_sv)))
        results["shrunk_var_std"].append(float(np.std(seed_sv)))
        results["mle_tail_mean"].append(float(np.mean(seed_mt)))
        results["mle_tail_std"].append(float(np.std(seed_mt)))
        results["shrunk_tail_mean"].append(float(np.mean(seed_st)))
        results["shrunk_tail_std"].append(float(np.std(seed_st)))
        print(f"  n_cal={n_cal:>3}: MLE var={np.mean(seed_mv):.3f}  "
              f"Shrunk var={np.mean(seed_sv):.3f}  "
              f"Tail MLE={np.mean(seed_mt):.1f}  Shrunk={np.mean(seed_st):.1f}")

    return results


# ── Main ──

if __name__ == "__main__":
    t0 = time.time()
    all_results = {}
    all_results["regimes"] = exp_regimes()
    all_results["ablation"] = exp_ablation()
    all_results["quintile"] = exp_quintile()
    all_results["shrinkage"] = exp_shrinkage()

    with open(str(OUT / "toy_unified_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUT / 'toy_unified_results.json'}")
    print(f"Total time: {time.time() - t0:.0f}s")
