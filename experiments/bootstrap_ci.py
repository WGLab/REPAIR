"""
Reproduce Table 9: Bootstrap 95% CI on RareBench (k=10).
=========================================================
Computes per-example correctness for Base, LogitAdj, Classwise, REPAIR,
then bootstrap-resamples 10,000 times to obtain 95% confidence intervals.

Also reports P(Delta > 0) for the REPAIR - Classwise difference.

Output: results/rarebench_ci.json

Usage:
    python -m experiments.bootstrap_ci
"""

import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repair.core import (
    split_rare, get_shortlist, compute_phi, fit, apply_scores, evaluate,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 10000


def rarebench_bootstrap_ci(ds, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Bootstrap 95% CI on RareBench k=10 covered test examples."""
    K = ds["K"]; tc = ds["tc"]; bLA = ds["bLA"]; sim = ds["sim"]
    tau = ds["tau_la"]; la_ = ds["la"]; lt_ = ds["lt"]; alpha = ds["alpha"]
    rare_set, _, _ = split_rare(tc, K)
    k = 10
    has_base_separate = ds.get("test_l_base") is not None

    # Shortlists
    if has_base_separate:
        St = get_shortlist(ds["test_l_base"], k)
        gt_base = np.take_along_axis(ds["test_l_base"], St, axis=1)
    else:
        St = get_shortlist(ds["test_l"], k)
        gt_base = np.take_along_axis(ds["test_l"], St, axis=1)

    gt = np.take_along_axis(ds["test_l"], St, axis=1)

    if "cal_l_raw" in ds:
        Sc = get_shortlist(ds["cal_l_raw"], k)
    else:
        Sc = get_shortlist(ds["cal_l"], k)

    la_c = np.take_along_axis(ds["cal_l"], Sc, axis=1) + tau * bLA[Sc]
    la_t = gt + tau * bLA[St]
    phi_c = compute_phi(la_c, Sc, k, sim, tc)
    phi_t = compute_phi(la_t, St, k, sim, tc)

    # Compute method scores
    methods_scores = {}
    methods_scores["Base"] = gt_base.copy()

    # LogitAdj
    la_base = gt_base if has_base_separate else gt
    best_la_scores, best_la_h1 = None, -1
    for t_ in [0.5, 1.0, 1.5, 2.0]:
        s = la_base + t_ * bLA[St]
        m, _ = evaluate(s, gt_base, St, ds["test_y"], k, rare_set)
        if m["hit1"] > best_la_h1:
            best_la_h1 = m["hit1"]
            best_la_scores = s.copy()
    methods_scores["LogitAdj"] = best_la_scores

    # Classwise
    a_cw, _ = fit(la_c, phi_c, Sc, ds["cal_y"], K, k,
                  lam_a=la_, lam_t=lt_, fit_cw=True, fit_pw=False,
                  alpha=alpha, train_counts=tc)
    methods_scores["Classwise"] = la_t + a_cw[St]

    # REPAIR
    a_r, t_r = fit(la_c, phi_c, Sc, ds["cal_y"], K, k,
                   lam_a=la_, lam_t=lt_, fit_cw=True, fit_pw=True,
                   alpha=alpha, train_counts=tc)
    methods_scores["REPAIR"] = apply_scores(la_t, phi_t, St, a_r, t_r, k)

    # Covered test examples
    test_y = ds["test_y"]; N = len(test_y)
    covered_idx = np.array([i for i in range(N)
                            if test_y[i] in St[i]])
    n_cov = len(covered_idx)
    print(f"  RareBench CI: n_test={N}, n_covered={n_cov}")

    # Per-example correctness
    method_correct = {}
    for method, scores in methods_scores.items():
        correct = np.zeros(n_cov, dtype=bool)
        for idx_i, i in enumerate(covered_idx):
            y = test_y[i]; sl = list(St[i]); yi = sl.index(y)
            ry = scores[i, yi]
            rank = 1 + sum(1 for j in range(k) if j != yi and scores[i, j] > ry)
            correct[idx_i] = (rank == 1)
        method_correct[method] = correct

    # Bootstrap
    rng = np.random.RandomState(seed)
    boot_results = {m: np.zeros(n_bootstrap) for m in methods_scores}
    boot_delta = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        bi = rng.choice(n_cov, size=n_cov, replace=True)
        for method in methods_scores:
            boot_results[method][b] = method_correct[method][bi].mean() * 100
        boot_delta[b] = boot_results["REPAIR"][b] - boot_results["Classwise"][b]

    ci_results = {}
    for method in methods_scores:
        vals = boot_results[method]
        ci_results[method] = {
            "mean": float(np.mean(vals)),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)),
        }
    ci_results["delta_REPAIR_minus_CW"] = {
        "mean": float(np.mean(boot_delta)),
        "ci_lo": float(np.percentile(boot_delta, 2.5)),
        "ci_hi": float(np.percentile(boot_delta, 97.5)),
        "p_positive": float((boot_delta > 0).mean()),
    }
    return ci_results, n_cov


if __name__ == "__main__":
    from experiments.run_main_table import load_gmdb, load_rarebench

    print("Loading GMDB (for shared config)...")
    ds_gmdb = load_gmdb()

    print("Loading RareBench...")
    ds_rb = load_rarebench(ds_gmdb)
    ds_rb.update({"la": 0.005, "lt": 5e-5, "alpha": 0.0})

    ci_results, n_cov = rarebench_bootstrap_ci(ds_rb)

    print(f"\nRareBench Bootstrap CI (n_covered={n_cov}, {N_BOOTSTRAP} resamples):")
    print(f"{'Method':<12} {'Mean':>7} {'95% CI':>18}")
    print("-" * 40)
    for method in ["Base", "LogitAdj", "Classwise", "REPAIR"]:
        m = ci_results[method]
        print(f"{method:<12} {m['mean']:>7.1f}  [{m['ci_lo']:.1f}, {m['ci_hi']:.1f}]")

    d = ci_results["delta_REPAIR_minus_CW"]
    print(f"\nDelta (REPAIR - CW): {d['mean']:+.1f}  "
          f"[{d['ci_lo']:+.1f}, {d['ci_hi']:+.1f}]  "
          f"P(>0) = {d['p_positive']:.3f}")

    with open(str(OUT / "rarebench_ci.json"), "w") as f:
        json.dump(ci_results, f, indent=2)
    print(f"\nSaved: {OUT / 'rarebench_ci.json'}")
