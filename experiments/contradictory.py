"""
Contradictory pair analysis for all 5 datasets.
=================================================
For each pair (u,v) co-occurring in shortlists of covered examples:
  1. t(x; u,v) = g_v(x) - g_u(x) for each context (x,S)
  2. Contradictory if: exists high-t context where u wins AND low-t context
     where v wins (the score gap does not consistently predict the winner).
  3. Reports mean D_y = number of distinct confusers per class.

This analysis motivates the pairwise component of REPAIR (Section 4.3).

Output: results/contradictory_{dataset}.json

Usage:
    python -m experiments.contradictory inat
    python -m experiments.contradictory imagenet
    python -m experiments.contradictory places
    python -m experiments.contradictory gmdb
    python -m experiments.contradictory rarebench
"""

import sys
import numpy as np
import json
import csv
import os
import time
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repair.core import split_rare, get_shortlist

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("REPAIR_DATA_ROOT", ROOT.parent))
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

k = 10


def load_dataset(name):
    """Load test logits and labels for a dataset."""
    if name == "inat":
        d = np.load(str(DATA_ROOT / "results/inat/logits.npz"), allow_pickle=True)
        return (d["test_logits"].astype(np.float64), d["test_labels"],
                d["counts"], d["test_logits"].shape[1], "iNaturalist")

    elif name == "imagenet":
        di = np.load(str(DATA_ROOT / "results/imagenet_lt/imagenet_lt_logits.npz"))
        all_l = di["logits"].astype(np.float64); all_y = di["labels"]; K = 1000
        tc = np.zeros(K)
        with open(str(DATA_ROOT / "data/imagenet_lt/ImageNet_LT_train.txt")) as f:
            for line in f:
                tc[int(line.strip().split()[-1])] += 1
        rng = np.random.RandomState(42); perm = rng.permutation(len(all_l))
        n_cal = int(len(all_l) * 0.4)
        return (all_l[perm[n_cal:]], all_y[perm[n_cal:]], tc, K, "ImageNet-LT")

    elif name == "places":
        dp = np.load(str(DATA_ROOT / "results/backbone/resnet152_places_lt_logits.npz"),
                     allow_pickle=True)
        all_l = dp["test_logits"].astype(np.float64); all_y = dp["test_labels"]
        tc = dp["train_counts"]; K = all_l.shape[1]
        rng = np.random.RandomState(42); perm = rng.permutation(len(all_l))
        n_cal = int(len(all_l) * 0.6)
        return (all_l[perm[n_cal:]], all_y[perm[n_cal:]], tc, K, "Places-LT")

    elif name == "gmdb":
        td = np.load(str(DATA_ROOT / "qwen-vl-inference/results/inference_output/"
                          "v2_kway_08b_final_test.logits.npz"), allow_pickle=True)
        cd = np.load(str(DATA_ROOT / "qwen-vl-inference/results/inference_output/"
                          "v2_kway_08b_final_cal.logits.npz"), allow_pickle=True)
        dn = list(cd["disease_names"]); K = len(dn)
        tc = np.zeros(K); nti = {n: i for i, n in enumerate(dn)}
        with open(str(DATA_ROOT / "qwen-vl-finetune/qwenvl/data/v2_manifest.csv")) as f:
            for row in csv.DictReader(f):
                if row["split"] == "train" and row["disease_name"] in nti:
                    tc[nti[row["disease_name"]]] += 1
        wn = np.sqrt(tc + 1.0)
        test_l = td["logits"].astype(np.float64) / (wn[None, :] ** 0.5)
        return (test_l, td["true_labels"], tc, K, "GMDB")

    elif name == "rarebench":
        dr = np.load(str(DATA_ROOT / "results/rarebench/rarebench_qwen_logits.npz"),
                     allow_pickle=True)
        all_l = dr["logits"].astype(np.float64); all_y = dr["labels"]
        cd = np.load(str(DATA_ROOT / "qwen-vl-inference/results/inference_output/"
                          "v2_kway_08b_final_cal.logits.npz"), allow_pickle=True)
        dn = list(cd["disease_names"]); K = len(dn)
        tc = np.zeros(K); nti = {n: i for i, n in enumerate(dn)}
        with open(str(DATA_ROOT / "qwen-vl-finetune/qwenvl/data/v2_manifest.csv")) as f:
            for row in csv.DictReader(f):
                if row["split"] == "train" and row["disease_name"] in nti:
                    tc[nti[row["disease_name"]]] += 1
        wn = np.sqrt(tc + 1.0)
        all_l = all_l / (wn[None, :] ** 0.5)
        rng = np.random.RandomState(42); perm = rng.permutation(len(all_l))
        n_cal = int(len(all_l) * 0.6)
        return (all_l[perm[n_cal:]], all_y[perm[n_cal:]], tc, K, "RareBench")
    else:
        raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m experiments.contradictory <dataset>")
        print("  dataset: inat, imagenet, places, gmdb, rarebench")
        sys.exit(1)

    DATASET = sys.argv[1]
    print(f"Loading {DATASET}...", flush=True)
    test_l, test_y, tc, K, ds_name = load_dataset(DATASET)
    rare_set, _, _ = split_rare(tc, K)

    St = get_shortlist(test_l, k)
    gt = np.take_along_axis(test_l, St, axis=1)
    N = len(test_y)

    covered = np.array([i for i in range(N) if test_y[i] in St[i]])
    print(f"  {ds_name}: N={N}, covered={len(covered)}, K={K}", flush=True)

    # Build pair contexts
    print("Building pair contexts...", flush=True)
    t0 = time.time()
    pair_data = defaultdict(list)

    for idx in covered:
        sl = list(St[idx])
        y = test_y[idx]
        scores = gt[idx]

        for a_pos in range(k):
            for b_pos in range(a_pos + 1, k):
                u, v = sl[a_pos], sl[b_pos]
                if u > v:
                    u, v = v, u
                    t_val = scores[a_pos] - scores[b_pos]
                    label = 1 if y == v else (-1 if y == u else 0)
                else:
                    t_val = scores[b_pos] - scores[a_pos]
                    label = 1 if y == u else (-1 if y == v else 0)
                pair_data[(u, v)].append((t_val, label))

    print(f"  {len(pair_data)} unique pairs ({time.time()-t0:.1f}s)", flush=True)

    # Check contradictory pairs
    print("Checking contradictory pairs...", flush=True)
    t0 = time.time()
    total_pairs = 0
    contradictory_pairs = 0

    for (u, v), contexts in pair_data.items():
        labeled = [(t_val, lab) for t_val, lab in contexts if lab != 0]
        total_pairs += 1
        if len(labeled) < 2:
            continue

        has_u_win = any(lab == 1 for _, lab in labeled)
        has_v_win = any(lab == -1 for _, lab in labeled)

        if has_u_win and has_v_win:
            max_t_u = max(tv for tv, lab in labeled if lab == 1)
            min_t_v = min(tv for tv, lab in labeled if lab == -1)
            if max_t_u >= min_t_v:
                contradictory_pairs += 1

    print(f"  Total pairs: {total_pairs}", flush=True)
    print(f"  Contradictory: {contradictory_pairs}", flush=True)
    print(f"  Fraction: {contradictory_pairs/max(total_pairs,1):.4f}", flush=True)

    # Mean D_y
    confusers_per_class = defaultdict(set)
    for idx in covered:
        y = test_y[idx]
        for c in St[idx]:
            if c != y:
                confusers_per_class[y].add(c)

    Dy_values = [len(confusers_per_class[y]) for y in confusers_per_class]
    print(f"\n  Mean D_y: {np.mean(Dy_values):.1f}")
    print(f"  Median D_y: {np.median(Dy_values):.1f}")

    result = {
        "dataset": ds_name, "K": K,
        "N_test": N, "N_covered": len(covered),
        "total_pairs": total_pairs,
        "contradictory_pairs": contradictory_pairs,
        "fraction": contradictory_pairs / max(total_pairs, 1),
        "mean_Dy": float(np.mean(Dy_values)),
        "median_Dy": float(np.median(Dy_values)),
    }
    with open(str(OUT / f"contradictory_{DATASET}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT / f'contradictory_{DATASET}.json'}")
