"""
Generate Figure 2: Synthetic validation panels.
================================================
  Fig 2A: Class-separable regime (CW ~= REPAIR)
  Fig 2B: Contradictory regime (REPAIR >> CW)
  Fig 3a: Ablation bar chart (both regimes)
  Fig 3b: Contradiction quintile
  Fig 3c: Shrinkage validation

Reads results from: results/toy/toy_unified_results.json
  (produced by experiments/synthetic.py)

Output: results/figures/fig_validation_{a,b}.pdf
        results/figures/fig_panel_{a,b,c}.pdf

Usage:
    python -m figures.gen_validation
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from figures.plot_style import *
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "results" / "toy" / "toy_unified_results.json"
FIG_DIR = str(ROOT / "results" / "figures")

with open(str(DATA_FILE)) as f:
    DATA = json.load(f)

C_BASE = PAL["gray"]
C_CW = PAL["blue"]
C_OURS = PAL["rose_deep"]


def fig_validation_a():
    """Class-separable regime bar chart."""
    d = DATA["regimes"]["A_class_separable"]
    methods = ["Base", "Classwise", "REPAIR\n(Ours)"]
    vals = [d["base"]["top1"] * 100, d["classwise"]["top1"] * 100,
            d["repair"]["top1"] * 100]
    rhos = [d["base"]["rho_k"], d["classwise"]["rho_k"], d["repair"]["rho_k"]]
    colors = [C_BASE, C_CW, C_OURS]
    recall = d["recall"] * 100

    fig, ax = plt.subplots(figsize=(COL_1, 2.3))
    x = np.arange(len(methods)); w = 0.55

    ax.fill_between([-0.6, len(methods) - 0.2], vals[0], recall,
                    facecolor="none", edgecolor="#b0b0b0", lw=0.0,
                    hatch="///", alpha=0.25, zorder=0)
    bars = ax.bar(x, vals, w, color=colors, edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(recall, color="#555", ls=":", lw=0.6, zorder=4)
    ax.text(x[-1] + w / 2, recall + 0.8, f"Recall@5 = {recall:.1f}%",
            fontsize=6, ha="right", va="bottom", color="#555", style="italic")

    for i, (bar, rho, val) in enumerate(zip(bars, rhos, vals)):
        c = colors[i] if i > 0 else "#888"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                f"$\\rho$={rho:.2f}", ha="center", va="bottom", fontsize=7, color=c)

    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel("Hit@1 (%)"); ax.set_ylim(25, 98)
    save_fig(fig, "fig_validation_a", FIG_DIR)
    plt.close(fig)


def fig_validation_b():
    """Contradictory regime bar chart with delta bracket."""
    d = DATA["regimes"]["B_contradictory"]
    methods = ["Base", "Classwise", "REPAIR\n(Ours)"]
    vals = [d["base"]["top1"] * 100, d["classwise"]["top1"] * 100,
            d["repair"]["top1"] * 100]
    rhos = [d["base"]["rho_k"], d["classwise"]["rho_k"], d["repair"]["rho_k"]]
    colors = [C_BASE, C_CW, C_OURS]
    recall = d["recall"] * 100

    fig, ax = plt.subplots(figsize=(COL_1, 2.3))
    x = np.arange(len(methods)); w = 0.55

    ax.fill_between([-0.6, len(methods) - 0.2], vals[0], recall,
                    facecolor="none", edgecolor="#b0b0b0", lw=0.0,
                    hatch="///", alpha=0.25, zorder=0)
    bars = ax.bar(x, vals, w, color=colors, edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(recall, color="#555", ls=":", lw=0.6, zorder=4)

    for i, (bar, rho, val) in enumerate(zip(bars, rhos, vals)):
        fw = "bold" if i == 2 else "normal"
        c = C_OURS if i == 2 else (colors[i] if i > 0 else "#888")
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                f"$\\rho$={rho:.2f}", ha="center", va="bottom",
                fontsize=7, fontweight=fw, color=c)

    # Delta bracket
    delta = vals[2] - vals[1]
    bx = x[2] + w / 2 + 0.12
    ax.plot([bx, bx], [vals[1] + 0.5, vals[2] - 0.5], color="#555", lw=0.9)
    ax.text(bx + 0.05, (vals[1] + vals[2]) / 2,
            r"$\Delta$Hit@1=" + f"{delta:.1f}", fontsize=7, color="#333",
            fontweight="bold", va="center", ha="left")

    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel("Hit@1 (%)"); ax.set_ylim(25, 98)
    save_fig(fig, "fig_validation_b", FIG_DIR)
    plt.close(fig)


def fig_panel_a():
    """Ablation bar chart (class-separable vs contradictory)."""
    da = DATA["ablation"]
    methods = ["Base", "PW Only", "CW Only", "REPAIR\n(Ours)"]
    keys = ["base", "pairwise_only", "classwise_only", "full_repair"]
    v_cs = [da["class_separable"][k] * 100 for k in keys]
    v_ct = [da["contradictory"][k] * 100 for k in keys]

    fig, ax = plt.subplots(figsize=(COL_1 * 1.2, 2.3))
    x = np.arange(len(methods)); w = 0.35

    ax.bar(x - w / 2, v_cs, w, color=C_CW, alpha=0.45,
           edgecolor=C_CW, lw=0.4, zorder=3, label="Class-separable")
    ax.bar(x + w / 2, v_ct, w, color=PAL["rose_mid"], alpha=0.85,
           edgecolor=C_OURS, lw=0.4, zorder=3, label="Non-class-separable")

    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=6.5)
    ax.set_ylabel("Hit@1 (%)")
    ax.legend(fontsize=5.5, loc="upper left")
    save_fig(fig, "fig_panel_a", FIG_DIR)
    plt.close(fig)


def fig_panel_b():
    """Contradiction quintile bar chart."""
    import matplotlib.colors as mcolors
    qa = DATA["quintile"]
    gains = qa["pw_gain_mean"]
    stds = qa["pw_gain_std"]
    labels = qa["labels"]

    fig, ax = plt.subplots(figsize=(COL_1, 2.3))
    x = np.arange(len(gains)); w = 0.58
    lr = mcolors.to_rgb(PAL["rose_light"])
    dr = mcolors.to_rgb(PAL["rose_deep"])
    bar_colors = [tuple(lr[j] * (1 - i / 4) + dr[j] * i / 4 for j in range(3))
                  for i in range(5)]

    ax.bar(x, gains, w, color=bar_colors, edgecolor=PAL["rose_deep"], lw=0.3,
           zorder=3, yerr=stds, capsize=2,
           error_kw=dict(lw=0.5, color="#888", capthick=0.4))
    ax.axhline(0, color="#ccc", lw=0.4, zorder=1)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_xlabel("Contradiction quintile")
    ax.set_ylabel(r"$\Delta$Hit@1 (%)")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    save_fig(fig, "fig_panel_b", FIG_DIR)
    plt.close(fig)


def fig_panel_c():
    """Shrinkage: offset variance + tail accuracy vs. cal size."""
    d = DATA["shrinkage"]
    cs = np.array(d["cal_sizes"])
    mv = np.array(d["mle_var_mean"]); mvs = np.array(d["mle_var_std"])
    sv = np.array(d["shrunk_var_mean"]); svs = np.array(d["shrunk_var_std"])
    mt = np.array(d["mle_tail_mean"]); mts = np.array(d["mle_tail_std"])
    st = np.array(d["shrunk_tail_mean"]); sts = np.array(d["shrunk_tail_std"])

    fig, ax = plt.subplots(figsize=(COL_1, 2.2))
    ax.fill_between(cs, mv - mvs, mv + mvs, color=PAL["blue"], alpha=0.12)
    ax.fill_between(cs, sv - svs, sv + svs, color=PAL["red"], alpha=0.12)
    l1, = ax.plot(cs, mv, "s-", color=PAL["blue"], ms=4, lw=1.2,
                  markeredgecolor="white", markeredgewidth=0.4, label="MLE offset var")
    l2, = ax.plot(cs, sv, "^-", color=PAL["red"], ms=4, lw=1.2,
                  markeredgecolor="white", markeredgewidth=0.4, label="Shrunk offset var")
    ax.set_xlabel("Cal. examples per class")
    ax.set_ylabel("Mean offset variance", color=PAL["blue"])
    ax.set_xscale("log"); ax.set_xticks(cs); ax.set_xticklabels(cs)
    ax.set_xlim(1.6, 65); ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.fill_between(cs, mt - mts, mt + mts, color=PAL["blue"], alpha=0.06)
    ax2.fill_between(cs, st - sts, st + sts, color=PAL["red"], alpha=0.06)
    l3, = ax2.plot(cs, mt, "o--", color=PAL["blue"], ms=3, lw=0.8, alpha=0.5,
                   label="MLE tail Hit@1")
    l4, = ax2.plot(cs, st, "D--", color=PAL["red"], ms=3, lw=0.8, alpha=0.5,
                   label="Shrunk tail Hit@1")
    ax2.set_ylabel("Tail Hit@1 (%)", color=PAL["red"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(PAL["red"])

    ax.legend([l1, l2, l3, l4],
              ["MLE offset var", "Shrunk offset var",
               "MLE tail Hit@1", "Shrunk tail Hit@1"],
              loc="center right", fontsize=5, ncol=1,
              frameon=True, framealpha=0.95, edgecolor="#ddd")
    save_fig(fig, "fig_panel_c", FIG_DIR)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating all synthetic validation panels...")
    fig_validation_a()
    fig_validation_b()
    fig_panel_a()
    fig_panel_b()
    fig_panel_c()
    print("Done -- all 5 figures saved.")
