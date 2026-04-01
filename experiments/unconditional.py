"""
Reproduce Table 7: Unconditional results (k=10).
=================================================
Converts conditioned metrics (Hit@1 | Y in S) to unconditional
(Hit@1 over all test examples) by multiplying by Recall@k.

This table demonstrates that REPAIR's gains are not an artifact
of conditioning on coverage.

Output: results/unconditional.json

Usage:
    python -m experiments.unconditional
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"


def compute_unconditional(ksens_results):
    """Compute unconditional metrics from k=10 conditioned results."""
    uncond = {}
    for ds_name, ds_res in ksens_results.items():
        k10 = ds_res["10"] if "10" in ds_res else ds_res[10]
        recall = k10["Base"]["recall"]
        ds_uncond = {}
        for method in ["Base", "LogitAdj", "Classwise", "REPAIR"]:
            m = k10[method]
            ds_uncond[method] = {
                "recall": recall,
                "cond_hit1": m["hit1"],
                "uncond_hit1": m["hit1"] * recall / 100,
                "cond_rare": m.get("rare_hit1", m.get("rare", 0)),
                "uncond_rare": m.get("rare_hit1", m.get("rare", 0)) * recall / 100,
            }
        uncond[ds_name] = ds_uncond
    return uncond


if __name__ == "__main__":
    # Requires k_sensitivity.json from k_sensitivity.py
    ksens_path = OUT / "k_sensitivity.json"
    if not ksens_path.exists():
        print(f"Error: {ksens_path} not found. Run k_sensitivity.py first.")
        sys.exit(1)

    with open(str(ksens_path)) as f:
        ksens = json.load(f)

    uncond = compute_unconditional(ksens)

    print(f"\n{'Dataset':<14} {'Method':<10} {'Cond H@1':>9} {'Recall':>8} "
          f"{'Uncond H@1':>11}")
    print("-" * 56)
    for ds_name in ["iNaturalist", "ImageNet-LT", "Places-LT", "GMDB", "RareBench"]:
        if ds_name not in uncond:
            continue
        for method in ["Base", "LogitAdj", "Classwise", "REPAIR"]:
            m = uncond[ds_name][method]
            label = ds_name if method == "Base" else ""
            print(f"{label:<14} {method:<10} {m['cond_hit1']:>9.1f} "
                  f"{m['recall']:>8.1f} {m['uncond_hit1']:>11.1f}")

    with open(str(OUT / "unconditional.json"), "w") as f:
        json.dump(uncond, f, indent=2)
    print(f"\nSaved: {OUT / 'unconditional.json'}")
