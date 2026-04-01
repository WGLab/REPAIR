"""
Generate Figure 7: Ablation bar chart for real datasets.
=========================================================
Reads ablation results and produces grouped bar chart of
Hit@1 across ablation variants for all 5 datasets.

Reads: results/ablation_all.json (from experiments/ablation.py)
Output: results/figures/fig_ablation_bars.pdf

Usage:
    python -m figures.gen_ablation
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from figures.plot_style import *

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = str(ROOT / "results" / "figures")

with open(str(ROOT / "results" / "ablation_all.json")) as f:
    DATA = json.load(f)

datasets = ["iNaturalist", "ImageNet-LT", "Places-LT", "GMDB", "RareBench"]
variants = ["Base", "PW-only", "CW (no alpha)", "Classwise",
            "REPAIR (no alpha)", "REPAIR"]
short_labels = ["Base", "PW", "CW\n(no $\\alpha$)", "CW", "REPAIR\n(no $\\alpha$)", "REPAIR"]

colors = [PAL["gray"], PAL["teal"], PAL["blue"], PAL["blue"],
          PAL["rose_mid"], PAL["rose_deep"]]
alphas = [0.7, 0.7, 0.45, 0.85, 0.6, 0.95]

fig, axes = plt.subplots(1, len(datasets), figsize=(COL_2, 2.5), sharey=False)

for di, ds_name in enumerate(datasets):
    ax = axes[di]
    vals = []
    for v in variants:
        m = DATA[ds_name].get(v, {})
        vals.append(m.get("hit1", 0))

    x = np.arange(len(variants))
    bars = ax.bar(x, vals, 0.7, color=colors, alpha=alphas,
                  edgecolor="white", lw=0.3, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=5, rotation=45, ha="right")
    ax.set_title(ds_name, fontsize=7, fontweight="bold")
    if di == 0:
        ax.set_ylabel("Hit@1 (%)")

    # Annotate best
    best_i = np.argmax(vals)
    ax.text(best_i, vals[best_i] + 0.5, f"{vals[best_i]:.1f}",
            ha="center", va="bottom", fontsize=5, fontweight="bold",
            color=PAL["rose_deep"])

plt.tight_layout(w_pad=0.8)
save_fig(fig, "fig_ablation_bars", FIG_DIR)
plt.close(fig)
print("Done.")
