#!/usr/bin/env python3
"""Figure 3 (fig:stepcurve) of the BRACE paper, self-contained.

Writes stepcurve.pdf. Needs only matplotlib -- no paper_numbers, no records, no cluster.

    python make_figure3.py                 # -> ./stepcurve.pdf
    python make_figure3.py out/            # -> out/stepcurve.pdf

WHAT IS PLOTTED. Each y value is the arm's ABSOLUTE solve rate (%) on the held-out MCP tasks at
that checkpoint step, with no base subtraction. The dashed line in each panel is that scale's
untrained base policy is the step-0 point every line starts from, so the effect a table reports is
the rise from step 0. Rates are computed on the same
arm-vs-base task intersection the effects were, so the two readings reconcile exactly. "Held out"
means held-out MCP tasks, disjoint from every training pool but the same substrate the arms trained
on; it is not transfer. Figure 4 is the transfer figure and is measured on BFCL and NESTFUL.

Each line is ONE ARM'S BEST SEED by window mean -- the same run Table 1's cell reports, for every
multi-seed rule including the control -- not a seed average.

PROVENANCE OF THE NUMBERS. Extracted 2026-09-09 from the live emitter
(surface/verl_rl/paper_figures.py emit_figures, via paper_numbers.full_family/agg, which
since 2026-09-09 carries the absolute rate as `absr` and its anchor as `basr`) and frozen
below. They are a snapshot: if the records change, regenerate with the emitter and re-extract.
This file is for reading, tweaking and re-plotting the figure, not for producing publication
numbers.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------- constants from the paper
WINDOW = [15, 20, 25, 30]          # the four pre-registered checkpoint steps
SCALES = ["2B", "4B", "8B"]
# The 6.0pp resolution floor is no longer DRAWN (the gauge was removed 2026-09-09); it is stated
# in the caption and in the protocol section instead. Kept here as documentation of the number.
FLOOR_PP = 6.0                     # resolution floor between two single-seed arms at 80% power

# name, colour, linestyle, marker, linewidth, markersize.  EVERY series carries the same
# weight (PI 2026-09-09): drawing our own method heavier than the baselines it is compared
# against is a thumb on the scale.  BRACE stays identifiable as the only SOLID line.
SERIES = [
    ("BRACE",         "#0072B2", "-",                  "o", 0.95, 2.5),
    ("PLR",           "#6E7B8B", "--",                 "s", 0.95, 2.5),
    ("RAG-MCP",       "#8C8C8C", "-.",                 "^", 0.95, 2.5),
    ("DAPO",          "#B07AA1", ":",                  "D", 0.95, 2.2),
    ("TSCL",          "#5F9E93", (0, (3, 1, 1, 1)),    "v", 0.95, 2.5),
    ("TRACE",         "#A6893C", (0, (5, 2)),          "P", 0.95, 2.6),
    ("VIP",           "#9C6B5E", (0, (1, 1.4)),        "X", 0.95, 2.6),
    ("uniform GRPO",  "#4D4D4D", (0, (2, 1, 4, 1)),    "*", 0.95, 3.4),
]

# ABSOLUTE solve rate (%) on the held-out tasks, at steps 15 / 20 / 25 / 30
DATA = {
    "2B": {
        "BRACE":         [10.9029, 12.0339, 11.1864, 10.8475],
        "PLR":           [ 9.6610,  9.1525,  8.8136,  9.7603],
        "RAG-MCP":       [ 9.8305, 10.6780, 11.5254, 11.6949],
        "DAPO":          [ 7.2881,  8.1356,  8.6441,  8.3051],
        "TSCL":          [ 8.9831,  9.1525,  9.4915,  9.6610],
        "TRACE":         [ 8.6441,  9.8305, 10.5085, 11.6949],
        "VIP":           [ 7.9661, 11.0169, 11.6949, 10.6780],
        "uniform GRPO":  [ 8.9831,  9.6610, 11.0169,  9.8305],
    },
    "4B": {
        "BRACE":         [12.2034, 13.0508, 13.7288, 14.2373],
        "PLR":           [11.6949, 12.3729, 11.6438, 13.7288],
        "RAG-MCP":       [12.3939, 13.5593, 13.3898, 13.7288],
        "DAPO":          [11.6949, 10.1695, 10.5085, 10.0000],
        "TSCL":          [12.3729, 13.7288, 12.3729, 10.6780],
        "TRACE":         [12.2034, 11.3559, 11.6949, 11.6949],
        "VIP":           [11.1864, 11.1864, 12.2558, 11.8280],
        "uniform GRPO":  [12.0339, 12.5424, 13.1399, 11.7148],
    },
    "8B": {
        "BRACE":         [15.7627, 15.0847, 16.3559, 16.5254],
        "PLR":           [14.2373, 14.5763, 13.6286, 13.9932],
        "RAG-MCP":       [14.2373, 13.1399, 13.3898, 13.0508],
        "DAPO":          [11.6949, 12.2034, 11.0476, 11.0169],
        "TSCL":          [10.3390,  9.8305,  7.4576,  7.4576],
        "TRACE":         [15.5932, 14.2373, 14.0678, 14.9153],
        "VIP":           [ 8.1356,  8.6441,  7.4576,  7.6271],
        "uniform GRPO":  [14.0678, 15.7627, 15.3390, 15.5932],
    },
}

# untrained base policy, mean over the same intersections (spread < 0.3 pp within a scale)
BASE = {"2B": 8.648, "4B": 9.677, "8B": 11.359}


def make(outdir="."):
    os.makedirs(outdir, exist_ok=True)
    # Sizes are set for the PAGE. The PDF is placed at \textwidth (397.5pt); 5.52in matches that
    # to a rounding, so the placement scale is ~1.0 and a point size set here is the size that
    # prints. Nothing prints below 7pt.
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "axes.titleweight": "normal",
        "legend.fontsize": 8.4, "xtick.labelsize": 8.6, "ytick.labelsize": 8.6,
        "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.22, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "serif",
    })

    fig, axes = plt.subplots(1, 3, figsize=(5.52, 2.10), sharey=True)
    # Explicit bottom margin, not cosmetic: at default margins the tick labels of a figure this
    # short fall below the figure box, into where the legend is placed, and the legend then prints
    # on top of "checkpoint step".
    fig.subplots_adjust(bottom=0.17, top=0.91, left=0.085, right=0.995, wspace=0.13)

    for ax, model in zip(axes, SCALES):
        # Every line starts at step 0 from the untrained base policy: the one point every arm
        # genuinely shares, because at step 0 no arm has trained and they are the same checkpoint.
        for i, (name, col, ls, mk, lw, ms) in enumerate(SERIES):
            ys = DATA[model].get(name)
            if ys is None:
                continue
            ax.plot([0] + WINDOW, [BASE[model]] + ys, color=col, ls=ls, marker=mk, ms=ms, lw=lw,
                    zorder=4 if i == 0 else 3)
        ax.set_title("Qwen3-VL-%s" % model)
        ax.set_xticks([0] + WINDOW)
        ax.set_xlim(-1.8, 31.6)

    axes[0].set_ylabel("held-out solve rate (%)")
    # Limits set from the drawn data, not by taste: with absolute rates the extremes are 7.46
    # (TSCL/VIP at 8B) and 16.53 (the method at 8B), with margin so the floor gauge and the base
    # policy label stay inside the panel.
    axes[0].set_ylim(6.0, 18.0)
    # Ticks every 2pp so a reader has a rule to measure a gap against.
    axes[0].yaxis.set_major_locator(MultipleLocator(2))

    axes[1].set_xlabel("checkpoint step")

    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=ms, lw=lw, label=n)
               for n, c, ls, mk, lw, ms in SERIES]
    # ONE row, kept narrower than the axes: the legend row sets this figure's tight bbox, and a
    # wider bbox drops the placement scale and with it every printed point size.
    fig.legend(handles=handles, loc="lower center", ncol=len(SERIES), frameon=False,
               bbox_to_anchor=(0.5, -0.155), handlelength=1.05, handletextpad=0.24,
               columnspacing=0.52, labelspacing=0.2)

    path = os.path.join(outdir, "stepcurve.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote %s" % path)
    return path


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else ".")
