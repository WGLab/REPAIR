"""
Reproduce Table 1: Main results across 5 datasets.
====================================================
Runs Base, LogitAdj, Classwise, and REPAIR on:
  - ImageNet-LT  (1000 classes, ResNet-50)
  - iNaturalist  (8142 classes, ResNet-50)
  - Places-LT    (365 classes, ResNet-152)
  - GMDB         (508 diseases, Qwen3.5-0.8B)
  - RareBench    (508 diseases, HPO-based scoring)

Phase 1: Point estimates (deterministic)
Phase 2: 5-seed sweep (80% cal subsampling) + sign test

Output: results/final/main_results.csv

Usage:
    python -m experiments.run_main_table
"""

import numpy as np
import json
import csv
import re
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Add parent to path so repair module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repair.core import (
    split_rare, get_shortlist, compute_phi, fit, apply_scores, evaluate,
)

# ── Configuration ──
ROOT = Path(__file__).resolve().parent.parent  # submission_code/
DATA_ROOT = Path(os.environ.get("REPAIR_DATA_ROOT", ROOT.parent))
OUT = ROOT / "results" / "final"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
CAL_SUBSAMPLE = 0.8

# Hyperparameters per dataset (tuned on calibration set)
HP_GRID = {
    "ImageNet-LT":  {"lam_a": 0.001,  "lam_t": 0.001,  "alpha": 0.0},
    "iNaturalist":  {"lam_a": 0.01,   "lam_t": 0.005,  "alpha": 0.3},
    "Places-LT":    {"lam_a": 0.001,  "lam_t": 0.001,  "alpha": 0.3},
    "GMDB":         {"lam_a": 0.5,    "lam_t": 0.01,   "alpha": 0.3},
    "RareBench":    {"lam_a": 0.005,  "lam_t": 5e-5,   "alpha": 0.0},
}


# ══════════════════════════════════════════════════════════════
# Dataset loaders
# ══════════════════════════════════════════════════════════════

def load_imagenet_lt():
    d = np.load(str(DATA_ROOT / "results/imagenet_lt/imagenet_lt_logits.npz"))
    logits = d["logits"].astype(np.float64)
    labels = d["labels"]
    K = 1000
    tc = np.zeros(K)
    with open(str(DATA_ROOT / "data/imagenet_lt/ImageNet_LT_train.txt")) as f:
        for line in f:
            tc[int(line.strip().split()[-1])] += 1
    bLA = -np.log(np.clip(tc / tc.sum(), 1e-8, 1))
    wn_path = DATA_ROOT / "results/imagenet_lt/wn_sim.npy"
    sim = np.load(str(wn_path)) if wn_path.exists() else None
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(labels))
    return {
        "name": "ImageNet-LT", "K": K, "k": 10,
        "cal_l": logits[idx[:20000]], "cal_y": labels[idx[:20000]],
        "test_l": logits[idx[20000:]], "test_y": labels[idx[20000:]],
        "tc": tc, "bLA": bLA, "sim": sim, "tau_la": 1.0,
        "test_l_base": None,
    }


def load_inat():
    d = np.load(str(DATA_ROOT / "results/inat/logits.npz"), allow_pickle=True)
    tc = d["counts"].astype(np.float64)
    K = d["cal_logits"].shape[1]
    bLA = -np.log(np.clip(tc / tc.sum(), 1e-8, 1))
    return {
        "name": "iNaturalist", "K": K, "k": 10,
        "cal_l": d["cal_logits"].astype(np.float64),
        "cal_y": d["cal_labels"],
        "test_l": d["test_logits"].astype(np.float64),
        "test_y": d["test_labels"],
        "tc": tc, "bLA": bLA, "sim": d["tax_dist"], "tau_la": 1.0,
        "test_l_base": None,
    }


def load_places_lt():
    dp = np.load(str(DATA_ROOT / "results/backbone/resnet152_places_lt_logits.npz"),
                 allow_pickle=True)
    all_l = dp["test_logits"].astype(np.float64)
    all_y = dp["test_labels"]
    tc = dp["train_counts"]
    K = all_l.shape[1]
    bLA = -np.log(np.clip(tc / tc.sum(), 1e-8, 1))
    rng = np.random.RandomState(42)
    cal_idx, test_idx = [], []
    for c in range(K):
        c_idx = np.where(all_y == c)[0]
        rng.shuffle(c_idx)
        n_cal = int(len(c_idx) * 0.6)
        cal_idx.extend(c_idx[:n_cal])
        test_idx.extend(c_idx[n_cal:])
    cal_idx = np.array(cal_idx)
    test_idx = np.array(test_idx)
    return {
        "name": "Places-LT", "K": K, "k": 10,
        "cal_l": all_l[cal_idx], "cal_y": all_y[cal_idx],
        "test_l": all_l[test_idx], "test_y": all_y[test_idx],
        "tc": tc, "bLA": bLA, "sim": None, "tau_la": 0.0,
        "test_l_base": None,
    }


def load_gmdb():
    td = np.load(str(DATA_ROOT / "qwen-vl-inference/results/inference_output/"
                      "v2_kway_08b_final_test.logits.npz"), allow_pickle=True)
    cd = np.load(str(DATA_ROOT / "qwen-vl-inference/results/inference_output/"
                      "v2_kway_08b_final_cal.logits.npz"), allow_pickle=True)
    dn = list(cd["disease_names"])
    K = len(dn)
    tc = np.zeros(K)
    nti = {n: i for i, n in enumerate(dn)}
    with open(str(DATA_ROOT / "qwen-vl-finetune/qwenvl/data/v2_manifest.csv")) as f:
        for row in csv.DictReader(f):
            if row["split"] == "train" and row["disease_name"] in nti:
                tc[nti[row["disease_name"]]] += 1
    bLA = -np.log(np.clip(tc / tc.sum(), 1e-8, 1))
    # HPO Jaccard similarity
    with open(str(DATA_ROOT / "qwen-vl-finetune/qwenvl/data/v2_disease_cards.json")) as f:
        cards = json.load(f)
    hs = [set(re.findall(r"HP:\d+", cards.get(n, ""))) for n in dn]
    hsim = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(i + 1, K):
            inter = len(hs[i] & hs[j])
            union = len(hs[i] | hs[j])
            if union > 0:
                hsim[i, j] = hsim[j, i] = inter / union
    # Classifier weight norms for tau-normalization
    norms_cache = DATA_ROOT / "results" / "gmdb_classifier_norms.npy"
    if norms_cache.exists():
        w_norms = np.load(str(norms_cache))
    else:
        import torch
        ckpt = DATA_ROOT / "qwen-vl-finetune/output_v2_kway_08b_20260319_174611"
        sd = torch.load(str(ckpt / "pytorch_model.bin"), map_location="cpu")
        W = sd["classifier.weight"].float().numpy()
        w_norms = np.linalg.norm(W, axis=1)
        np.save(str(norms_cache), w_norms)
    test_raw = td["logits"].astype(np.float64)
    cal_raw = cd["logits"].astype(np.float64)
    return {
        "name": "GMDB", "K": K, "k": 10,
        "cal_l": cal_raw / (w_norms[None, :] ** 0.5),
        "cal_y": cd["true_labels"],
        "test_l": test_raw / (w_norms[None, :] ** 0.5),
        "test_y": td["true_labels"],
        "tc": tc, "bLA": bLA, "sim": hsim, "tau_la": 0.0,
        "test_l_base": test_raw,
    }


def load_rarebench(gmdb_ds):
    rb = np.load(str(DATA_ROOT / "results/rarebench/rarebench_qwen_logits.npz"),
                 allow_pickle=True)
    rb_raw = rb["logits"].astype(np.float64)
    rb_labels = rb["labels"].astype(int)
    norms_cache = DATA_ROOT / "results" / "gmdb_classifier_norms.npy"
    w_norms = np.load(str(norms_cache))
    rb_norm = rb_raw / (w_norms[None, :] ** 0.5)
    N = len(rb_labels)
    K = gmdb_ds["K"]
    tc = np.load(str(DATA_ROOT / "results/rarebench/disease_train_counts.npy"))
    bLA = -np.log(np.clip(tc / tc.sum(), 1e-8, 1))
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    nc_s = int(N * 0.6)
    return {
        "name": "RareBench", "K": K, "k": 10,
        "cal_l": rb_norm[idx[:nc_s]], "cal_y": rb_labels[idx[:nc_s]],
        "test_l": rb_norm[idx[nc_s:]], "test_y": rb_labels[idx[nc_s:]],
        "tc": tc, "bLA": bLA, "sim": gmdb_ds["sim"], "tau_la": 0.0,
        "test_l_base": rb_raw[idx[nc_s:]],
        "cal_l_raw": rb_raw[idx[:nc_s]],
    }


LOADERS = {
    "ImageNet-LT": load_imagenet_lt,
    "iNaturalist": load_inat,
    "Places-LT": load_places_lt,
}


# ══════════════════════════════════════════════════════════════
# Run one dataset
# ══════════════════════════════════════════════════════════════

def run_dataset(ds, seed=None, lam_a=0.01, lam_t=0.001, alpha=0.0):
    """Run Base, LogitAdj, Classwise, REPAIR on one dataset."""
    K = ds["K"]; k = ds["k"]
    tc = ds["tc"]; bLA = ds["bLA"]
    sim = ds["sim"]; tau = ds["tau_la"]
    has_base_separate = ds.get("test_l_base") is not None
    rare_set, freq_set, cutoff = split_rare(tc, K)

    cal_l = ds["cal_l"]; cal_y = ds["cal_y"]
    test_l = ds["test_l"]; test_y = ds["test_y"]

    # Optional 80% cal subsampling
    if seed is not None:
        rng = np.random.RandomState(seed)
        n_sub = int(len(cal_l) * CAL_SUBSAMPLE)
        sub_idx = rng.choice(len(cal_l), size=n_sub, replace=False)
        cal_l = cal_l[sub_idx]
        cal_y = cal_y[sub_idx]

    # Shortlists
    if has_base_separate:
        St = get_shortlist(ds["test_l_base"], k)
        gt_base = np.take_along_axis(ds["test_l_base"], St, axis=1)
    else:
        St = get_shortlist(test_l, k)
        gt_base = np.take_along_axis(test_l, St, axis=1)

    gt = np.take_along_axis(test_l, St, axis=1)

    if "cal_l_raw" in ds:
        Sc = get_shortlist(ds["cal_l_raw"] if seed is None else ds["cal_l_raw"][sub_idx if seed is not None else slice(None)], k)
        # After subsampling we need to re-slice
        if seed is not None:
            Sc = get_shortlist(ds["cal_l_raw"][sub_idx], k)
    else:
        Sc = get_shortlist(cal_l, k)

    # LA-adjusted scores for fitting
    la_c = np.take_along_axis(cal_l, Sc, axis=1) + tau * bLA[Sc]
    la_t = gt + tau * bLA[St]

    # Pairwise features
    phi_c = compute_phi(la_c, Sc, k, sim, tc)
    phi_t = compute_phi(la_t, St, k, sim, tc)

    results = {}

    # 1. Base
    m_base, _ = evaluate(gt_base, gt_base, St, test_y, k, rare_set)
    results["Base"] = m_base

    # 2. LogitAdj (sweep tau)
    la_base = gt_base if has_base_separate else gt
    best_la = None
    for t_ in [0.5, 1.0, 1.5, 2.0]:
        s = la_base + t_ * bLA[St]
        m_la, _ = evaluate(s, gt_base, St, test_y, k, rare_set)
        if best_la is None or m_la["hit1"] > best_la["hit1"]:
            best_la = m_la
    results["LogitAdj"] = best_la

    # 3. Classwise
    a_cw, _ = fit(la_c, phi_c, Sc, cal_y, K, k,
                  lam_a=lam_a, lam_t=lam_t,
                  fit_cw=True, fit_pw=False,
                  alpha=alpha, train_counts=tc)
    s_cw = la_t + a_cw[St]
    m_cw, cw_correct = evaluate(s_cw, gt_base, St, test_y, k, rare_set)
    results["Classwise"] = m_cw

    # 4. REPAIR (full)
    a_r, t_r = fit(la_c, phi_c, Sc, cal_y, K, k,
                   lam_a=lam_a, lam_t=lam_t,
                   fit_cw=True, fit_pw=True,
                   alpha=alpha, train_counts=tc)
    s_rep = apply_scores(la_t, phi_t, St, a_r, t_r, k)
    m_rep, rep_correct = evaluate(s_rep, gt_base, St, test_y, k, rare_set)
    results["REPAIR"] = m_rep

    # Sign test (REPAIR vs Classwise)
    wins = sum(1 for c, r in zip(cw_correct, rep_correct)
               if c is not None and r is not None and r and not c)
    losses = sum(1 for c, r in zip(cw_correct, rep_correct)
                 if c is not None and r is not None and c and not r)
    results["_sign"] = {"wins": wins, "losses": losses}

    return results, rare_set, freq_set, cutoff


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_results = []
    split_info = []

    # Load datasets
    datasets = {}
    for name in ["ImageNet-LT", "iNaturalist", "Places-LT"]:
        print(f"Loading {name}...")
        datasets[name] = LOADERS[name]()

    print("Loading GMDB...")
    datasets["GMDB"] = load_gmdb()

    print("Loading RareBench...")
    datasets["RareBench"] = load_rarebench(datasets["GMDB"])

    for ds_name in ["ImageNet-LT", "iNaturalist", "Places-LT", "GMDB", "RareBench"]:
        ds = datasets[ds_name]
        hp = HP_GRID[ds_name]
        print(f"\n{'='*60}\n  {ds_name}\n{'='*60}")

        # Phase 1: Point estimate
        res, rare_set, freq_set, cutoff = run_dataset(ds, seed=None, **hp)
        split_info.append({"dataset": ds_name, "K": ds["K"],
                           "n_rare": len(rare_set), "n_freq": len(freq_set),
                           "cutoff": cutoff})

        for method in ["Base", "LogitAdj", "Classwise", "REPAIR"]:
            m = res[method]
            print(f"  {method:12s} H@1={m['hit1']:.1f}  Rare={m['rare_hit1']:.1f}  "
                  f"HFR={m['hfr']:.3f}")
            all_results.append({"dataset": ds_name, "method": method,
                                "seed": "det", **m})

        # Phase 2: 5-seed sweep
        print(f"  5-seed sweep...")
        seed_results = {"Classwise": [], "REPAIR": []}
        for seed in SEEDS:
            res_s, _, _, _ = run_dataset(ds, seed=seed, **hp)
            for method in ["Classwise", "REPAIR"]:
                seed_results[method].append(res_s[method])
                all_results.append({"dataset": ds_name, "method": method,
                                    "seed": seed, **res_s[method]})

        for method in ["Classwise", "REPAIR"]:
            sr = seed_results[method]
            for metric in ["hit1", "rare_hit1", "hfr"]:
                vals = [r[metric] for r in sr]
                print(f"    {method:12s} {metric:12s} "
                      f"{np.mean(vals):.1f} +/- {np.std(vals):.1f}")

    # Save
    with open(str(OUT / "main_results.csv"), "w") as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)
    print(f"\nSaved: {OUT / 'main_results.csv'}")

    with open(str(OUT / "split_info.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "K", "n_rare", "n_freq", "cutoff"])
        w.writeheader()
        w.writerows(split_info)
    print(f"Saved: {OUT / 'split_info.csv'}")
    print("\nDone!")
