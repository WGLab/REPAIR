"""
Shared Nature-level plot style for REPAIR paper figures.
Import this in every figure generation script.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Nature house style ──
matplotlib.rcParams.update({
    # Typography
    "font.size": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.title_fontsize": 7,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold",

    # Lines and edges
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "patch.linewidth": 0.4,

    # Layout
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.facecolor": "white",
    "axes.facecolor": "white",

    # Legend
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
})

# ── Unified palette (warm rose scheme) ──
PAL = {
    "gray":       "#a0a0a0",
    "blue":       "#8faabe",
    "red":        "#c25a6e",
    "red_light":  "#dda0a8",
    "rose_light": "#f0d0d4",
    "rose_mid":   "#d47a86",
    "rose_deep":  "#a83c50",
    "green":      "#59a14f",
    "gold":       "#edc948",
    "purple":     "#b07aa1",
    "teal":       "#76b7b2",
    "orange":     "#f28e2b",
}

# Nature column widths (inches)
COL_1 = 3.5    # single column (89 mm)
COL_1_5 = 5.0  # 1.5 columns (120 mm)
COL_2 = 7.2    # full width (183 mm)


def save_fig(fig, name, out_dir="results/figures"):
    """Save PDF + PNG."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p / f"{name}.pdf"))
    fig.savefig(str(p / f"{name}.png"), dpi=200)
    print(f"Saved: {p / name}.pdf")
