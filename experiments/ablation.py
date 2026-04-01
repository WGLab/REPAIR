"""
Reproduce Table 3 / Figure 7: Component ablation across 5 datasets.
====================================================================
Variants:
  1. Base           -- raw logits
  2. PW-only        -- a=0, fit theta
  3. CW-only (no a) -- fit a with alpha=0, theta=0
  4. Classwise      -- fit a with alpha, theta=0
  5. REPAIR (no a)  -- fit a+theta with alpha=0
  6. REPAIR full    -- fit a+theta with alpha

Metrics: Hit@1, Rare Hit@1, HFR. Single deterministic run per dataset.

Output: results/ablation_all.json

Usage:
    python -m experiments.ablation
"""

import numpy as np
import json
import csv
import re
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repair.core import (
    split_rare, get_shortlist, compute_phi, fit, apply_scores, evaluate,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("REPAIR_DATA_ROOT", ROOT.parent))
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


def run_ablation(name, cal_l, cal_y, test_l, test_y, K, tc, bLA, sim,
                 la_hp, lt_hp, alpha_hp, tau_la=1.0, test_l_base=None):
    """Run all 6 ablation variants on one dataset."""
    k = 10
    rare_set, _, _ = split_rare(tc, K)

    St = get_shortlist(test_l if test_l_base is None else test_l_base, k)
    gt_base = np.take_along_axis(
        test_l if test_l_base is None else test_l_base, St, axis=1)
    gt = np.take_along_axis(test_l, St, axis=1)
    Sc = get_shortlist(cal_l, k)

    la_c = np.take_along_axis(cal_l, Sc, axis=1) + tau_la * bLA[Sc]
    la_t = gt + tau_la * bLA[St]
    phi_c = compute_phi(la_c, Sc, k, sim, tc)
    phi_t = compute_phi(la_t, St, k, sim, tc)

    results = {}

    # 1. Base
    m, _ = evaluate(gt_base, gt_base, St, test_y, k, rare_set)
    results["Base"] = m

    # 2. PW-only (a=0, fit theta, alpha=alpha_hp)
    _, t2 = fit(la_c, phi_c, Sc, cal_y, K, k, lam_a=la_hp, lam_t=lt_hp,
                fit_cw=False, fit_pw=True, alpha=alpha_hp, train_counts=tc)
    s2 = la_t.copy()
    pw = np.einsum("nyjd,d->nyj", phi_t, t2)
    np.einsum("nii->ni", pw)[:] = 0
    s2 += pw.sum(axis=2) / max(k - 1, 1)
    m2, _ = evaluate(s2, gt_base, St, test_y, k, rare_set)
    results["PW-only"] = m2

    # 3. CW-only, no alpha (alpha=0, theta=0)
    a3, _ = fit(la_c, phi_c, Sc, cal_y, K, k, lam_a=la_hp, lam_t=lt_hp,
                fit_cw=True, fit_pw=False, alpha=0.0, train_counts=tc)
    s3 = la_t + a3[St]
    m3, _ = evaluate(s3, gt_base, St, test_y, k, rare_set)
    results["CW (no alpha)"] = m3

    # 4. Classwise (fit a with alpha, theta=0)
    a4, _ = fit(la_c, phi_c, Sc, cal_y, K, k, lam_a=la_hp, lam_t=lt_hp,
                fit_cw=True, fit_pw=False, alpha=alpha_hp, train_counts=tc)
    s4 = la_t + a4[St]
    m4, _ = evaluate(s4, gt_base, St, test_y, k, rare_set)
    results["Classwise"] = m4

    # 5. REPAIR, no alpha (alpha=0)
    a5, t5 = fit(la_c, phi_c, Sc, cal_y, K, k, lam_a=la_hp, lam_t=lt_hp,
                 fit_cw=True, fit_pw=True, alpha=0.0, train_counts=tc)
    s5 = apply_scores(la_t, phi_t, St, a5, t5, k)
    m5, _ = evaluate(s5, gt_base, St, test_y, k, rare_set)
    results["REPAIR (no alpha)"] = m5

    # 6. REPAIR full
    a6, t6 = fit(la_c, phi_c, Sc, cal_y, K, k, lam_a=la_hp, lam_t=lt_hp,
                 fit_cw=True, fit_pw=True, alpha=alpha_hp, train_counts=tc)
    s6 = apply_scores(la_t, phi_t, St, a6, t6, k)
    m6, _ = evaluate(s6, gt_base, St, test_y, k, rare_set)
    results["REPAIR"] = m6

    # Print
    print(f"\n{'='*60}")
    print(f"  {name} Ablation (k={k})")
    print(f"{'='*60}")
    print(f"  {'Variant':<18} {'Hit@1':>7} {'Rare':>7} {'HFR':>7}")
    print(f"  {'-'*44}")
    for v in ["Base", "PW-only", "CW (no alpha)", "Classwise",
              "REPAIR (no alpha)", "REPAIR"]:
        r = results[v]
        hfr = f"{r['hfr']:.3f}" if v != "Base" else "--"
        print(f"  {v:<18} {r['hit1']:>7.1f} {r['rare_hit1']:>7.1f} {hfr:>7}")

    return results


# ── Main ──

if __name__ == "__main__":
    # This script loads datasets inline. In practice, adapt the data loaders
    # from run_main_table.py. The structure below demonstrates the pattern.
    from experiments.run_main_table import (
        load_imagenet_lt, load_inat, load_places_lt, load_gmdb, load_rarebench,
    )

    all_results = {}

    # iNaturalist
    print("Loading iNaturalist...")
    ds = load_inat()
    all_results["iNaturalist"] = run_ablation(
        "iNaturalist", ds["cal_l"], ds["cal_y"],
        ds["test_l"], ds["test_y"], ds["K"], ds["tc"], ds["bLA"], ds["sim"],
        la_hp=0.01, lt_hp=0.005, alpha_hp=0.3, tau_la=1.0)

    # ImageNet-LT
    print("\nLoading ImageNet-LT...")
    ds = load_imagenet_lt()
    all_results["ImageNet-LT"] = run_ablation(
        "ImageNet-LT", ds["cal_l"], ds["cal_y"],
        ds["test_l"], ds["test_y"], ds["K"], ds["tc"], ds["bLA"], ds["sim"],
        la_hp=0.001, lt_hp=0.001, alpha_hp=0.0, tau_la=1.0)

    # Places-LT
    print("\nLoading Places-LT...")
    ds = load_places_lt()
    all_results["Places-LT"] = run_ablation(
        "Places-LT", ds["cal_l"], ds["cal_y"],
        ds["test_l"], ds["test_y"], ds["K"], ds["tc"], ds["bLA"], ds["sim"],
        la_hp=0.001, lt_hp=0.001, alpha_hp=0.3, tau_la=0.0)

    # GMDB
    print("\nLoading GMDB...")
    ds = load_gmdb()
    all_results["GMDB"] = run_ablation(
        "GMDB", ds["cal_l"], ds["cal_y"],
        ds["test_l"], ds["test_y"], ds["K"], ds["tc"], ds["bLA"], ds["sim"],
        la_hp=0.5, lt_hp=0.01, alpha_hp=0.3, tau_la=0.0,
        test_l_base=ds["test_l_base"])

    # RareBench
    print("\nLoading RareBench...")
    ds_rb = load_rarebench(ds)
    all_results["RareBench"] = run_ablation(
        "RareBench", ds_rb["cal_l"], ds_rb["cal_y"],
        ds_rb["test_l"], ds_rb["test_y"], ds_rb["K"], ds_rb["tc"],
        ds_rb["bLA"], ds_rb["sim"],
        la_hp=0.005, lt_hp=5e-5, alpha_hp=0.0, tau_la=0.0,
        test_l_base=ds_rb["test_l_base"])

    # Save
    with open(str(OUT / "ablation_all.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {OUT / 'ablation_all.json'}")
