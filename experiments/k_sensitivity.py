"""
Reproduce Table 6: k-sensitivity (k=5, 10, 20, 50) for all 5 datasets.
========================================================================
Runs Base, LogitAdj, Classwise, and REPAIR at each shortlist size k.
For small datasets (GMDB, RareBench), uses 5-seed 80% cal subsampling
to match Table 1 means.

Output: results/k_sensitivity.json

Usage:
    python -m experiments.k_sensitivity
"""

import numpy as np
import json
import time
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

K_VALUES = [5, 10, 20, 50]


def run_k_sensitivity(ds):
    """Run all methods at all k values for one dataset."""
    name = ds["name"]
    K = ds["K"]; tc = ds["tc"]; bLA = ds["bLA"]
    sim = ds["sim"]; tau = ds["tau_la"]
    la_ = ds["la"]; lt_ = ds["lt"]; alpha = ds["alpha"]
    rare_set, _, _ = split_rare(tc, K)
    has_base_separate = ds.get("test_l_base") is not None
    use_5seed = name in ("GMDB", "RareBench")

    results = {}
    for k_val in K_VALUES:
        t0 = time.time()

        # Shortlists
        if has_base_separate:
            St = get_shortlist(ds["test_l_base"], k_val)
            gt_base = np.take_along_axis(ds["test_l_base"], St, axis=1)
        else:
            St = get_shortlist(ds["test_l"], k_val)
            gt_base = np.take_along_axis(ds["test_l"], St, axis=1)

        gt = np.take_along_axis(ds["test_l"], St, axis=1)

        if "cal_l_raw" in ds:
            Sc = get_shortlist(ds["cal_l_raw"], k_val)
        else:
            Sc = get_shortlist(ds["cal_l"], k_val)

        la_c = np.take_along_axis(ds["cal_l"], Sc, axis=1) + tau * bLA[Sc]
        la_t = gt + tau * bLA[St]
        phi_c = compute_phi(la_c, Sc, k_val, sim, tc)
        phi_t = compute_phi(la_t, St, k_val, sim, tc)

        # Base
        m_base, _ = evaluate(gt_base, gt_base, St, ds["test_y"], k_val, rare_set)

        # LogitAdj (sweep tau)
        la_base_scores = gt_base if has_base_separate else gt
        best_la = None
        for t_ in [0.5, 1.0, 1.5, 2.0]:
            s = la_base_scores + t_ * bLA[St]
            m, _ = evaluate(s, gt_base, St, ds["test_y"], k_val, rare_set)
            if best_la is None or m["hit1"] > best_la["hit1"]:
                best_la = m

        # CW and REPAIR (with optional 5-seed subsampling)
        seeds = [0, 1, 2, 3, 4] if use_5seed else [None]
        all_cw, all_rep = [], []

        for seed in seeds:
            if seed is not None:
                rng_s = np.random.RandomState(seed)
                n_sub = int(len(ds["cal_y"]) * 0.8)
                sub = rng_s.choice(len(ds["cal_y"]), size=n_sub, replace=False)
                if "cal_l_raw" in ds:
                    Sc_s = get_shortlist(ds["cal_l_raw"][sub], k_val)
                else:
                    Sc_s = get_shortlist(ds["cal_l"][sub], k_val)
                la_cs = (np.take_along_axis(ds["cal_l"][sub], Sc_s, axis=1)
                         + tau * bLA[Sc_s])
                phi_cs = compute_phi(la_cs, Sc_s, k_val, sim, tc)
                cal_y_s = ds["cal_y"][sub]
            else:
                Sc_s, la_cs, phi_cs, cal_y_s = Sc, la_c, phi_c, ds["cal_y"]

            a_cw, _ = fit(la_cs, phi_cs, Sc_s, cal_y_s, K, k_val,
                          lam_a=la_, lam_t=lt_, fit_cw=True, fit_pw=False,
                          alpha=alpha, train_counts=tc)
            s_cw = la_t + a_cw[St]
            m_cw, _ = evaluate(s_cw, gt_base, St, ds["test_y"], k_val, rare_set)
            all_cw.append(m_cw)

            a_r, t_r = fit(la_cs, phi_cs, Sc_s, cal_y_s, K, k_val,
                           lam_a=la_, lam_t=lt_, fit_cw=True, fit_pw=True,
                           alpha=alpha, train_counts=tc)
            s_r = apply_scores(la_t, phi_t, St, a_r, t_r, k_val)
            m_r, _ = evaluate(s_r, gt_base, St, ds["test_y"], k_val, rare_set)
            all_rep.append(m_r)

        # Average over seeds
        avg_keys = ["hit1", "rare_hit1", "freq_hit1", "hfr", "recall"]
        m_cw_avg = {m: np.mean([r[m] for r in all_cw]) for m in avg_keys
                    if m in all_cw[0]}
        m_rep_avg = {m: np.mean([r[m] for r in all_rep]) for m in avg_keys
                     if m in all_rep[0]}

        results[k_val] = {
            "Base": m_base, "LogitAdj": best_la,
            "Classwise": m_cw_avg, "REPAIR": m_rep_avg,
        }
        dt = time.time() - t0
        print(f"    k={k_val:>2}: Base={m_base['hit1']:.1f} "
              f"REP={m_rep_avg['hit1']:.1f} [{dt:.0f}s]")

    return results


if __name__ == "__main__":
    from experiments.run_main_table import (
        load_imagenet_lt, load_inat, load_places_lt, load_gmdb, load_rarebench,
    )

    all_ksens = {}

    # Load and add per-dataset HP
    loaders = [
        ("iNaturalist", load_inat,
         {"la": 0.01, "lt": 0.005, "alpha": 0.3}),
        ("ImageNet-LT", load_imagenet_lt,
         {"la": 0.001, "lt": 0.001, "alpha": 0.0}),
        ("Places-LT", load_places_lt,
         {"la": 0.001, "lt": 0.001, "alpha": 0.3}),
    ]

    for name, loader, hp in loaders:
        print(f"\n{name}")
        ds = loader()
        ds.update(hp)
        all_ksens[name] = run_k_sensitivity(ds)

    print("\nGMDB")
    ds_gmdb = load_gmdb()
    ds_gmdb.update({"la": 0.5, "lt": 0.01, "alpha": 0.3})
    all_ksens["GMDB"] = run_k_sensitivity(ds_gmdb)

    print("\nRareBench")
    ds_rb = load_rarebench(ds_gmdb)
    ds_rb.update({"la": 0.005, "lt": 5e-5, "alpha": 0.0})
    all_ksens["RareBench"] = run_k_sensitivity(ds_rb)

    with open(str(OUT / "k_sensitivity.json"), "w") as f:
        json.dump(all_ksens, f, indent=2, default=str)
    print(f"\nSaved: {OUT / 'k_sensitivity.json'}")
