"""
Generate Figure 3: Quintile stratification figures.
====================================================
  - Synthetic quintile (5 seeds, SE bars)
  - GMDB quintile (rare-class, binomial SE)

Output: results/figures/fig_quintile_synthetic.pdf
        results/figures/fig_quintile_gmdb.pdf

Usage:
    python -m figures.gen_quintile
"""

import sys
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from figures.plot_style import *
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = str(ROOT / "results" / "figures")

# Rose gradient (Q1 light -> Q5 deep)
EDGE = "#8a3348"
ANNOT_COLOR = "#8a3348"

quintile_labels = ["Q1\n(low)", "Q2", "Q3", "Q4", "Q5\n(high)"]

# Synthetic data (5 seeds)
syn_delta = [-9.4, 0.6, 3.9, 7.2, 16.4]
syn_std_raw = [4.9, 2.1, 1.8, 3.7, 4.6]
syn_se = [s / np.sqrt(5) for s in syn_std_raw]

# GMDB data (k=5, rare-class) with binomial SE
gmdb_delta = [0.0, 0.0, 0.0, 7.1, 20.0]
se_q4 = math.sqrt(0.929 * 0.071 / 65 + 1.0 * 0.0 / 65) * 100
se_q5 = math.sqrt(0.50 * 0.50 / 37 + 0.70 * 0.30 / 37) * 100
gmdb_std = [0.0, 0.0, 0.0, se_q4, se_q5]

x = np.arange(5)
bw = 0.56


def gradient_colors(light, deep, n=5):
    lr = mcolors.to_rgb(light)
    dr = mcolors.to_rgb(deep)
    return [tuple(lr[j] * (1 - i / (n - 1)) + dr[j] * i / (n - 1) for j in range(3))
            for i in range(n)]


GRAD = gradient_colors(PAL["rose_light"], PAL["rose_deep"], 5)


def make_plot(ax, deltas, labels, errs=None):
    bars = ax.bar(x, deltas, bw, color=GRAD, edgecolor=EDGE, linewidth=0.4,
                  zorder=3)
    if errs is not None:
        ax.errorbar(x, deltas, yerr=errs, fmt="none",
                    ecolor="#444444", capsize=3.5, capthick=0.7,
                    elinewidth=0.7, zorder=5)
    ax.axhline(0, color="#AAAAAA", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Contradiction quintile")
    ax.set_ylabel(r"$\Delta$Hit@1 (%)")

    q5 = deltas[-1]
    y_top = q5 + (errs[-1] if errs else 0) + 1.0
    ax.annotate(f"+{q5:.1f}", xy=(4, y_top),
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", color=ANNOT_COLOR)


if __name__ == "__main__":
    # Synthetic
    fig1, ax1 = plt.subplots(figsize=(3.3, 2.5))
    make_plot(ax1, syn_delta, quintile_labels, errs=syn_se)
    ax1.set_ylim(-16, 24)
    ax1.set_title("Synthetic ($K{=}20,\\; k{=}5$)", fontsize=9, pad=6)
    save_fig(fig1, "fig_quintile_synthetic", FIG_DIR)
    plt.close(fig1)

    # GMDB
    fig2, ax2 = plt.subplots(figsize=(3.3, 2.5))
    make_plot(ax2, gmdb_delta, quintile_labels, errs=gmdb_std)
    ax2.set_ylim(-5, 36)
    ax2.set_title("GMDB ($K{=}508,\\; k{=}5$, rare)", fontsize=9, pad=6)
    save_fig(fig2, "fig_quintile_gmdb", FIG_DIR)
    plt.close(fig2)

    print("Done.")
