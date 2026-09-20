#!/usr/bin/env python3
"""Figure 4 (fig:transfermain) of the BRACE paper, self-contained.

Writes transferfig.pdf. Needs only matplotlib -- no paper_numbers, no records, no cluster.

    python make_figure4.py                 # -> ./transferfig.pdf
    python make_figure4.py out/            # -> out/transferfig.pdf

WHAT IS PLOTTED. Two panels, one per transfer benchmark, neither of which any arm trained on. Each
bar is a SIGNED CHANGE in percentage points against the untrained base policy of that scale:
bar = cell - base. The zero rule is therefore the base policy itself, and a bar below it is a rule
that made the policy worse than not training at all. Eight bars per scale group, the same eight in
every group -- not "the best other baseline at each scale", which would be a per-panel maximum and
exactly the selection this paper refuses.

CELLS ARE ABSOLUTE SCORES and the subtraction happens here, so BASE below is not decoration: change
it and every bar moves. BFCL is a pass rate over n=800; NESTFUL a win rate over n=1,861.

SCALE DIVIDERS (added 2026-09-09 at the PI's request) separate the 2B, 4B and 8B groups. They carry
no data: they are there because eight adjacent bars read as one run of sixteen without them, and a
reader has to count to find the group boundary.

REVERTED 2026-09-15 at the PI's request to this base-subtracted form, after a period drawing
absolute rates (kept as make_figure4.py.bak-absolute). The later non-metric fixes are retained:
solid scale dividers, bar width 0.105, and the regenerated TSCL 8B BFCL cell (24.125 -> 24.625,
appendix "Published values that changed", case (v)).

PROVENANCE. Extracted 2026-09-09 from the live emitter (surface/verl_rl/paper_figures.py
emit_transfer_figure, via _transfer_data) and frozen below. A snapshot: if the records change,
regenerate with the emitter and re-extract. For reading and re-plotting, not for publication
numbers.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------- palette (the paper's)
TRBLUE, TRGREY, TRGREEN, TRINK = "#0072B2", "#8C8C8C", "#00674C", "#262626"
TRGREEN_TINT = "#73AB9C"      # trGreen blended 45% into white; still >= 3:1 against the page

SCALES = ["2B", "4B", "8B"]

# name, facecolour, edgecolour, hatch.  \methodname{} first, then the control, then the published
# rules.  The three published entries share one hue and are separated by fill and hatch, so the
# block reads as a block and the figure still survives greyscale.
BARS = [
    ("BRACE",                  TRBLUE,       TRBLUE,       None),
    ("uniform GRPO (control)", TRGREY,       TRGREY,       None),
    ("PLR",                    TRGREEN,      TRGREEN,      None),
    ("RAG-MCP",                TRGREEN_TINT, TRGREEN,      "///"),
    ("DAPO",                   "white",      TRGREY,       "..."),
    ("TSCL",                   TRGREEN_TINT, TRGREEN,      "\\\\\\"),
    ("TRACE",                  TRGREEN_TINT, TRGREEN_TINT, None),
    ("VIP",                    "white",      TRGREEN,      None),
]

PANELS = [("bfcl", "BFCL v4 multi-turn: pass rate, $n=800$"),
          ("nest", "NESTFUL: win rate, $n=1{,}861$")]
YSTEP = {"bfcl": 5, "nest": 5}   # pp between y ticks; EVERY tick carries its value (PI 2026-09-15)

# untrained base policy, per benchmark and scale -- the zero rule of each panel
BASE = {
    "bfcl": {"2B": 15.125,    "4B": 29.000,    "8B": 35.875},
    "nest": {"2B": 21.601290, "4B": 30.789898, "8B": 38.259001},
}

# absolute cell scores; the bar drawn is CELL - BASE
CELLS = {
    "BRACE":                  {"bfcl": {"2B": 16.625, "4B": 31.125, "8B": 38.500},
                               "nest": {"2B": 25.470177, "4B": 31.166040, "8B": 40.300913}},
    "uniform GRPO (control)": {"bfcl": {"2B":  8.500, "4B": 19.375, "8B": 34.000},
                               "nest": {"2B": 19.398173, "4B": 29.177861, "8B": 39.602364}},
    "PLR":                    {"bfcl": {"2B":  8.250, "4B": 33.625, "8B": 37.375},
                               "nest": {"2B": 31.757120, "4B": 33.799033, "8B": 41.859215}},
    "RAG-MCP":                {"bfcl": {"2B": 14.625, "4B": 32.625, "8B": 37.375},
                               "nest": {"2B": 25.362708, "4B": 31.757120, "8B": 38.205266}},
    "DAPO":                   {"bfcl": {"2B":  6.500, "4B": 30.375, "8B": 37.875},
                               "nest": {"2B": 24.556690, "4B": 29.124127, "8B": 39.065019}},
    "TSCL":                   {"bfcl": {"2B": 12.750, "4B": 29.125, "8B": 24.625},
                               "nest": {"2B": 26.383665, "4B": 30.843632, "8B": 32.993015}},
    "TRACE":                  {"bfcl": {"2B": 16.750, "4B": 33.375, "8B": 36.875},
                               "nest": {"2B": 27.189683, "4B": 29.016658, "8B": 39.065019}},
    "VIP":                    {"bfcl": {"2B": 15.000, "4B": 25.500, "8B": 21.625},
                               "nest": {"2B": 29.339065, "4B":  7.254164, "8B": 15.368082}},
}


def make(outdir="."):
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({
        # Set for the PAGE. The tight bbox trims the saved width to the content, so the placement
        # scale is \textwidth over that width; figsize keeps it near 1.0 and nothing prints under 7pt.
        "font.size": 7.4, "axes.titlesize": 7.4, "axes.labelsize": 7.4,
        "axes.titleweight": "normal", "legend.fontsize": 7.4,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "figure.dpi": 200, "savefig.bbox": "tight",
        # A y GRID AND NOT A DECORATION: the panels carry bars an order of magnitude apart
        # (NESTFUL runs to -23.5), so a small bar is read against the tick it sits under rather
        # than against its own height. Faint enough never to compete with the zero rule.
        "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.20,
        "grid.linewidth": 0.4, "grid.color": TRGREY,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
        "axes.edgecolor": TRINK, "xtick.color": TRINK, "ytick.color": TRINK,
        "text.color": TRINK, "axes.labelcolor": TRINK, "hatch.linewidth": 0.55,
    })

    fig, axs = plt.subplots(1, 2, figsize=(6.30, 1.14))
    # The bottom band is reserved explicitly: in a figure this short the tick labels otherwise fall
    # below the figure's own bottom edge, into the coordinates where fig.legend places the legend.
    fig.subplots_adjust(bottom=0.20, top=0.88, wspace=0.26)

    # W WAS 0.132, CHOSEN WHEN THIS FIGURE DREW SIX BARS. With eight, 8 x 0.132 = 1.056 per
    # group against a unit pitch, so adjacent scale groups OVERLAPPED and the 0.5 divider fell
    # inside the last bar of each group. 0.105 gives a half-width of 0.415, clear of it.
    W = 0.105                      # eight bars and a real gap inside a unit pitch
    for ax, (bm, title) in zip(axs, PANELS):
        vals = []
        for gi, sc in enumerate(SCALES):
            for bi, (nm, fc, ec, ht) in enumerate(BARS):
                v = CELLS[nm][bm].get(sc)
                x = gi + (bi - 3.5) * W
                if not isinstance(v, (int, float)):
                    # A GAP AND A MARK, never a zero-height bar: "not measured" and "no effect"
                    # are different statements and this figure may not confuse them.
                    ax.text(x, 0.0, "n/a", fontsize=6.2, color=TRGREY, rotation=90,
                            ha="center", va="bottom")
                    continue
                dv = v - BASE[bm][sc]
                vals.append(dv)
                ax.bar(x, dv, width=W * 0.90, facecolor=fc, edgecolor=ec, hatch=ht,
                       linewidth=0.5, zorder=3)

        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        # 5% and not 10%: at this figure height NESTFUL's labels every 5 pp need the whole axis
        # height; with 10% padding they printed closer together than the type is tall.
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        # Y UNITS (PI 2026-09-15): a tick every 5 pp in both panels, every one labelled with its value,
        # each with a faint grid line, so a bar's height can be read off the axis. No unlabelled ticks.
        ax.yaxis.set_major_locator(MultipleLocator(YSTEP[bm]))
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.axhline(0.0, color=TRINK, lw=0.8, zorder=2)

        # SCALE DIVIDERS. Drawn BETWEEN groups, at the midpoints, faint and behind the bars so they
        # separate without competing with the zero rule (which is the figure's one datum line).
        # They carry no value; they exist because eight adjacent bars otherwise read as one run.
        for gi in range(len(SCALES) - 1):
            ax.axvline(gi + 0.5, color=TRGREY, lw=0.7, alpha=0.85, zorder=1)

        ax.set_xticks(range(len(SCALES)))
        ax.set_xticklabels(SCALES)
        ax.set_xlim(-0.5, len(SCALES) - 0.5)
        ax.tick_params(axis="x", length=0, pad=2.0)
        ax.tick_params(axis="y", length=2, pad=1.5)
        ax.set_title(title, pad=3.0)
        ax.set_ylabel("$\\Delta$ vs untrained base (pp)", labelpad=1.5)
        # The zero rule IS the base policy, so the bottom spine may not also run through the bars.
        ax.spines["bottom"].set_visible(False)

    handles = [Patch(facecolor=fc, edgecolor=ec, hatch=ht, linewidth=0.5, label=nm)
               for nm, fc, ec, ht in BARS]
    # Kept NARROWER THAN THE AXES: with a tight bbox the saved width is the wider of the two, and a
    # legend that set it would shrink the whole float and drive every label under the 7pt floor.
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, -0.44), handlelength=1.25, handleheight=0.85,
                     handletextpad=0.40, columnspacing=1.1, labelspacing=0.30)
    for h in leg.get_patches():
        h.set_linewidth(0.5)

    path = os.path.join(outdir, "transferfig.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote %s" % path)
    return path


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else ".")
