"""
Generate k-sensitivity line plot.
==================================
Delta(Hit@1) = REPAIR - Classwise vs. shortlist size k,
split into public benchmarks and medical datasets.

Reads: results/k_sensitivity.json (from experiments/k_sensitivity.py)
Output: results/figures/fig_k_sensitivity.pdf

Usage:
    python -m figures.gen_k_sensitivity
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from figures.plot_style import *

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = str(ROOT / "results" / "figures")

with open(str(ROOT / "results" / "k_sensitivity.json")) as f:
    data = json.load(f)

ks = [5, 10, 20, 50]
datasets = ["iNaturalist", "ImageNet-LT", "Places-LT", "GMDB", "RareBench"]

# Compute deltas
deltas = {}
for ds in datasets:
    deltas[ds] = []
    for k in ks:
        rep = data[ds][str(k)]["REPAIR"]["hit1"]
        cw = data[ds][str(k)]["Classwise"]["hit1"]
        deltas[ds].append(rep - cw)

ds_colors = {
    "iNaturalist": "#4e79a7", "ImageNet-LT": "#59a14f", "Places-LT": "#f28e2b",
    "GMDB": "#c25a6e", "RareBench": "#8a3348",
}
ds_markers = {
    "iNaturalist": "o", "ImageNet-LT": "s", "Places-LT": "^",
    "GMDB": "D", "RareBench": "v",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_2, 3.0), sharey=False)

# Left: public benchmarks
for ds in ["iNaturalist", "ImageNet-LT", "Places-LT"]:
    ax1.plot(ks, deltas[ds], marker=ds_markers[ds], color=ds_colors[ds],
             label=ds, markeredgecolor="white", markeredgewidth=0.6, zorder=5)
ax1.axhline(0, color="#888", ls="--", lw=0.6, zorder=0)
ax1.set_xlabel("Shortlist size $k$")
ax1.set_ylabel(r"$\Delta$Hit@1 (REPAIR $-$ Classwise)")
ax1.set_xticks(ks)
ax1.set_title("Public benchmarks", fontsize=8, fontweight="bold")
ax1.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#ccc")

# Right: medical
for ds in ["GMDB", "RareBench"]:
    ax2.plot(ks, deltas[ds], marker=ds_markers[ds], color=ds_colors[ds],
             label=ds, markeredgecolor="white", markeredgewidth=0.6, zorder=5)
ax2.axhline(0, color="#888", ls="--", lw=0.6, zorder=0)
ax2.set_xlabel("Shortlist size $k$")
ax2.set_xticks(ks)
ax2.set_title("Rare disease diagnosis", fontsize=8, fontweight="bold")
ax2.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#ccc")

plt.tight_layout(w_pad=2.5)
save_fig(fig, "fig_k_sensitivity", FIG_DIR)
plt.close(fig)
print("Done.")
