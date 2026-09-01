"""The paper's learning-curve figure: held-out effect vs training step, one line per arm.

WHY NOT TRAINING REWARD. Training reward cannot carry this figure, and the reason is measurable
rather than stylistic. Arms train on DIFFERENT surfaces -- mean advertised tools ranges from 19.9
(a8Ae0) to 34.1 (a8A), a 70% spread -- so an arm whose gate advertised fewer, better-targeted tools
faces an easier per-episode search. Two arms (a8C, a8Cu) additionally optimise a SHAPED objective,
so their gradients follow a different signal entirely. And arms sit at wildly different steps, where
every measured arm peaks in 15-30 and decays after, so a late arm is read on its downslope while a
young one is still climbing. Ranking arms by training reward would rank them by a mixture of surface
richness, objective, and horizon.

WHAT THIS PLOTS INSTEAD. Each point is a held-out eval cell: the arm's checkpoint at that step,
evaluated on the SAME fixed surface (advertised_init.txt, byte-identical across arms) over the same
task pool, scored on the BINARY solve. Every point is then PAIRED against one base-policy anchor
(cell_A, n=1180) on the tasks both attempted, so the y value is a within-task difference and not an
accuracy difference between two differently-covered samples.

THE ERROR BAND is the paired binomial SE, sqrt(b+c)/n over the McNemar discordant counts -- the same
statistic the main table tests with. Concordant pairs carry no information about the difference, so
a band built from the marginal rates would be far too narrow and would make every arm look separated.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# THE METHOD'S PRINTED NAME, ONCE (review P0-1, 2026-08-29). Every legend, label and console line
# in this file reads METHOD_NAME rather than a literal, because the 2026-08-29 rename left nine
# occurrences of the old name inside already-rendered figure artwork while the tables and prose had
# moved on. It is the plain-text twin of head.tex's \methodname{}; the LaTeX macro cannot be used
# here because matplotlib renders these strings, not TeX.
METHOD_NAME = "BRACE"

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, f"{R}/surface/verl_rl")
from main_table import ABSORBED, CONTINUATION, cell_path  # one arm, one trajectory, one line

# Okabe-Ito: the standard 8-hue categorical set designed for deuteranopia/protanopia/tritanopia.
# Assigned in FIXED order and never cycled -- a 9th arm folds into "other" rather than reusing a hue,
# because a repeated colour in a figure with a legend reads as a repeated identity.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# Marker shapes carry identity ALONGSIDE colour, in the same fixed never-cycled order. This is not
# only an accessibility hedge. Arms genuinely coincide: VIP and the TRACE-style plug-in each have one
# evaluated checkpoint and both measure +0.0339 at step 15, so with filled same-shape markers the
# second one drawn erases the first and the figure silently shows five series while the legend
# promises six. Markers are drawn UNFILLED so a stack reads as a stack -- two open shapes at one
# coordinate are both legible, and no point has to be nudged off its true value to be seen.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load_cell(path):
    """{(scenario, task, seed): solved}, first occurrence wins.

    Duplicates and torn records both occur -- two cards briefly shared a cell and appended
    concurrently, leaving 410 lines holding 162 distinct tasks plus 40 unparsable rows. Keying
    the dict is what makes that harmless.
    """
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        k = (d.get("scenario"), d.get("task_idx"), d.get("seed"))
        if None in k or k in out:
            continue
        try:
            out[k] = int(d["reward"] > 0)
        except Exception:
            continue
    return out


def paired(cur, base):
    """(diff, se, n) for one cell against the base anchor, on the tasks both attempted."""
    keys = sorted(set(cur) & set(base))
    n = len(keys)
    if not n:
        return None
    b = sum(1 for k in keys if base[k] and not cur[k])      # base solved, arm did not
    c = sum(1 for k in keys if cur[k] and not base[k])      # arm solved, base did not
    diff = (c - b) / n
    se = math.sqrt(b + c) / n if (b + c) else 0.0
    return diff, se, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=f"{R}/work/coadapt_eval_coadapt/cell_A.jsonl")
    ap.add_argument("--min-pairs", type=int, default=300,
                    help="a cell below this is not plotted: at n=40 a single discordant pair "
                         "moves the estimate by 0.025, which is half the effect being claimed")
    ap.add_argument("--arms", default="", help="space/comma separated; default = every arm with data")
    ap.add_argument("--label", default="", help="arms to name on the plot (default: top 4 at the end)")
    ap.add_argument("--max-step", type=int, default=0, help="clip the x axis")
    ap.add_argument("--out", default=f"{R}/work/fig_learning_curve.png")
    ap.add_argument("--mode", default="arms",
                    choices=["arms", "efficiency", "efficiency8b", "forest", "transfer",
                             "valsolve", "failuretax"])
    ap.add_argument("--compact", action="store_true",
                    help="forest: keep only the most informative arms and set at half width, "
                         "for a main-text figure that sits beside its table rather than under it")
    a = ap.parse_args()
    if a.mode == "efficiency":
        return efficiency(a)
    if a.mode == "efficiency8b":
        return efficiency(a, scales=("8B",), figsize=(3.9, 3.2))
    if a.mode == "forest":
        return forest(a)
    if a.mode == "transfer":
        return transfer(a)
    if a.mode == "valsolve":
        return valsolve(a)
    if a.mode == "failuretax":
        return failuretax(a)

    base = load_cell(a.base)
    if not base:
        print("no base cell; the figure has no anchor and every point would be unpaired")
        return 1

    want = {t for t in re.split(r"[ ,]+", a.arms) if t}
    series = {}
    for d in sorted(glob.glob(f"{R}/work/coadapt_eval_*")):
        arm = os.path.basename(d).replace("coadapt_eval_", "")
        if arm == "coadapt" or arm in ABSORBED or (want and arm not in want):
            continue
        # An arm that was preempted and relaunched is ONE trajectory: its later cells live under the
        # continuation tag and are plotted at their equivalent step, not at the relaunch's own count.
        files = [(st, cell_path(arm, st)) for st in CONTINUATION.get(arm, {})]
        files += [(int(m.group(1)), f) for f in glob.glob(f"{d}/cell_STEP*.jsonl")
                  if (m := re.search(r"cell_STEP(\d+)\.jsonl$", f))
                  and int(m.group(1)) not in CONTINUATION.get(arm, {})]
        pts = []
        for st, f in files:
            if a.max_step and st > a.max_step:
                continue
            r = paired(load_cell(f), base)
            if r and r[2] >= a.min_pairs:
                pts.append((st, *r))
        if pts:
            series[arm] = sorted(pts)

    if not series:
        print("no arm has a cell above --min-pairs yet")
        return 0

    # order legend by final effect so the reading order matches the visual order
    order = sorted(series, key=lambda k: -series[k][-1][1])

    # Internal arm tags are not readable in a paper figure, and a legend the reader has to decode
    # against a table is a legend that does not work. Anything not in this map keeps its tag.
    NAMES = {
        "a8T3g": METHOD_NAME, "a8T3gr": METHOD_NAME, "a8T3g2": METHOD_NAME,
        "a8T5k": METHOD_NAME + ", sharpened", "a8T5kr": METHOD_NAME + ", sharpened",
        "a8F": "uniform GRPO", "a8Fr": "uniform GRPO",
        "b8plr": "PLR-style replay", "b8ret": "retrieval surface",
        "dapo": "DAPO", "a8C": "ACCORD (dense shaping)", "a8A3": "ELSA", "a8A": "adaptive surface",
        "a8Tvip": "VIP", "a8Tlp": "learning progress", "a8T2n": "band filter",
        "a8Tnr": "band filter", "a8T2nr": "band filter",
        "a8Tpe": "plug-in estimator",
    }
    SEEDNO = {"a8T3g": 1, "a8T3gr": 2, "a8T3g2": 3, "a8T5k": 1, "a8T5kr": 2,
              "a8F": 1, "a8Fr": 2, "a8T2n": 1, "a8T2nr": 2}

    def label_of(arm, plotted):
        """Method name, plus a seed number ONLY when a sibling seed is in the same figure."""
        base = NAMES.get(arm, arm)
        if sum(1 for x in plotted if NAMES.get(x, x) == base) > 1 and arm in SEEDNO:
            return f"{base} (seed {SEEDNO[arm]})"
        return base
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    steps_all = sorted({p[0] for v in series.values() for p in v})

    # The matched window the main table reports. Drawn first so it sits UNDER the data: a band
    # over the lines would mute exactly the region the reader is being asked to look at. Skipped
    # entirely when every plotted point already lies inside it -- shading the whole axis and then
    # labelling it "matched window" tells the reader nothing and just greys the figure.
    show_window = bool(steps_all) and (steps_all[0] < 15 or steps_all[-1] > 30)
    if show_window:
        ax.axvspan(15, 30, color="#000000", alpha=0.045, lw=0, zorder=0)
    ax.axhline(0, color="#666666", lw=1.0, ls=(0, (4, 3)), zorder=1)

    for i, arm in enumerate(order):
        col = OKABE_ITO[i % len(OKABE_ITO)]
        mk = MARKERS[i % len(MARKERS)]
        xs = [p[0] for p in series[arm]]
        ys = [100 * p[1] for p in series[arm]]      # percentage points, as everywhere else
        es = [100 * p[2] for p in series[arm]]
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                        color=col, alpha=0.13, lw=0, zorder=2)
        ax.plot(xs, ys, color=col, lw=1.8, marker=mk, ms=5.6, mew=1.5, mfc="none", mec=col,
                label=label_of(arm, order), zorder=3, solid_capstyle="round")

    # The x grid the table reports on. Auto ticks put labels at 16/18/22 -- steps nothing is
    # measured at -- while hiding that every arm is read at 15/20/25/30.
    grid = [s for s in steps_all if s % 5 == 0]
    if grid:
        ax.set_xticks(grid)

    # Annotate AFTER the data is drawn, in axes-fraction y. Anchoring this to ax.get_ylim() before
    # plotting read the default (0,1) limits and forced the axes to span the whole unit interval --
    # a 7298px-tall figure with the data compressed into a sliver at the bottom.
    if show_window:
        ax.text(22.5, 0.985, "matched window", ha="center", va="top", fontsize=7.5,
                color="#555555", transform=ax.get_xaxis_transform())

    ax.set_xlabel("training step (GRPO)")
    ax.set_ylabel("held-out effect vs base policy (pp)\n(paired difference in solve rate)")
    ax.set_title("Held-out transfer over training, fixed surface for every arm", fontsize=11, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.grid(axis="y", color="#000000", alpha=0.07, lw=0.8)
    ax.set_axisbelow(True)
    # A legend is always present for >=2 series, so identity is never carried by colour alone.
    ax.legend(frameon=False, fontsize=8.5, ncol=min(3, len(order)), loc="lower center",
              bbox_to_anchor=(0.5, -0.40), columnspacing=1.4, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    fig.savefig(a.out.replace(".png", ".pdf"), bbox_inches="tight")   # vector for the camera-ready
    print(f"[plot] {len(series)} arms, {sum(len(v) for v in series.values())} points -> {a.out}")
    for arm in order:
        p = series[arm]
        print(f"  {arm:<10} steps {p[0][0]:>3}-{p[-1][0]:<3} n={len(p):<3} "
              f"final {p[-1][1]:+.4f} +/- {p[-1][2]:.4f}")
    return 0


# =================================================================================================
# EFFICIENCY PANEL (--mode efficiency). Seed-MEAN effect vs step for the method and its uniform
# control, with the seed range as a band, and a horizontal guide at uniform's step-30 seed mean.
# Why a dedicated panel and not a guide line on the main figure: the main figure plots individual
# arms, and the time-to-threshold claim is about SEED MEANS. Drawing the guide there would invite
# the reader to compare it against one lucky seed, which is the comparison this paper argues
# against everywhere else.
def efficiency(a, scales=("2B", "4B", "8B"), figsize=None):
    """Method vs its matched uniform control, seed means, PP units, one panel per model scale.

    TWO CALLERS, ONE FUNCTION. `--mode efficiency8b` draws the 8B panel alone, small, for the main
    text: the time-to-threshold claim -- the control's step-30 level reached at step 15 -- is an 8B
    result and is the only thing this exhibit is load-bearing for. `--mode efficiency` draws all
    three for the appendix, where "does that hold as the policy shrinks" can be asked without the
    main text appearing to lead on an in-distribution head-to-head it says is unresolvable.
    """
    ALL = [("2B", f"{R}/work/coadapt_eval_probe2b/cell_PROBE2B.jsonl",
            ["q2bT", "q2bT3"], ["q2bF5", "q2bF6", "q2bF7"]),
           ("4B", f"{R}/work/coadapt_eval_probe4b/cell_PROBE4B.jsonl",
            ["q4bT"], ["q4bF"]),
           ("8B", f"{R}/work/coadapt_eval_coadapt/cell_A.jsonl",
            ["a8T3g", "a8T3gr", "a8T3g2"], ["a8F", "a8Fr"])]
    SCALES = [s for s in ALL if s[0] in scales]
    steps = [15, 20, 25, 30]
    if figsize is None:
        # Sized to sit at \textwidth (397.5pt = 5.51in) with NO downscaling, so the 8-9pt type
        # here is 8-9pt on the page. The previous 10.6in-wide version was scaled to 52% and set
        # its labels at ~4.8pt.
        figsize = (5.5, 2.5) if len(SCALES) == 3 else (3.6 * len(SCALES), 3.5)
    fig, axes = plt.subplots(1, len(SCALES), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (name, basep, tri, uni) in zip(axes, SCALES):
        base = load_cell(basep)
        def series(arms):
            per = {}
            for st in steps:
                v = []
                for arm in arms:
                    r = paired(load_cell(cell_path(arm, st)), base)
                    if r and r[2] >= 560:
                        v.append(100 * r[0])
                if v:
                    per[st] = v
            return per
        t, u = series(tri), series(uni)
        ax.axhline(0, color="#666666", lw=1.0, ls=(0, (4, 3)), zorder=1)
        for per, col, mk, lab in ((u, OKABE_ITO[2], "^", "uniform GRPO"),
                                  (t, OKABE_ITO[0], "o", METHOD_NAME)):
            xs = sorted(per)
            if not xs:
                continue
            ys = [sum(per[x]) / len(per[x]) for x in xs]
            ax.fill_between(xs, [min(per[x]) for x in xs], [max(per[x]) for x in xs],
                            color=col, alpha=0.15, lw=0, zorder=2)
            ax.plot(xs, ys, color=col, lw=2.2, marker=mk, ms=7, mew=1.8, mfc="none", mec=col,
                    label=lab, zorder=3)
        # Panel title is the scale and nothing else; the seed counts are a caption clause.
        ax.set_title(name, fontsize=10)
        ax.set_xticks(steps)
        ax.tick_params(labelsize=8.5, colors="#444444")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#999999")
        ax.grid(axis="y", color="#000000", alpha=0.07, lw=0.8)
        ax.set_axisbelow(True)
    # The time-to-threshold guide belongs on the 8B panel and nowhere else: it is the one scale
    # where the claim was made, and drawing the same guide under 2B/4B would imply three results.
    t15 = None
    if "8B" in scales:
        ax8 = axes[[s[0] for s in SCALES].index("8B")]
        base8 = load_cell(f"{R}/work/coadapt_eval_coadapt/cell_A.jsonl")
        u30 = 100 * sum(paired(load_cell(cell_path(x, 30)), base8)[0] for x in ["a8F", "a8Fr"]) / 2
        t15 = 100 * sum(paired(load_cell(cell_path(x, 15)), base8)[0]
                        for x in ["a8T3g", "a8T3gr", "a8T3g2"]) / 3
        ax8.axhline(u30, color=OKABE_ITO[2], lw=1.1, ls=(0, (2, 2)), zorder=2)
        if len(axes) == 1:
            ax8.text(0.03, 0.035,
                     f"dotted: uniform at step 30 ({u30:+.1f} pp);\n{METHOD_NAME} is there at step 15",
                     transform=ax8.transAxes, fontsize=8.0, color="#333333", va="bottom",
                     ha="left", path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    axes[0].set_ylabel("effect over base policy (pp)", fontsize=9)
    axes[len(axes) // 2].set_xlabel(
        "training step (GRPO); per-step generation cost identical across arms"
        if len(axes) > 1 else "training step (GRPO)", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.5,
                   loc="lower left" if len(axes) > 1 else "upper left",
                   handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[eff] {len(SCALES)} scale(s) -> {a.out}"
          + (f";  8B: {METHOD_NAME}@15 {t15:+.1f}pp vs uniform@30 {u30:+.1f}pp" if t15 is not None else ""))
    return 0


# =================================================================================================
# TRANSFER (--mode transfer). THE RESULTS FIGURE.
#
# Why this replaced the three-panel in-distribution exhibit as the main-text figure: the axis that
# exhibit puts forward -- method against control ON THE TRAINING DISTRIBUTION -- is the axis this
# paper spends a subsection showing it cannot resolve, because the within-configuration seed spread
# is as large as the between-method spread. A figure whose visual claim is "strong at 2B, nothing
# at 8B, a tie at 4B" therefore leads on noise. Off the training distribution the same checkpoints
# separate at every scale and by margins several times the seed spread, so that is what the reader
# should see first.
#
# FORM. Magnitude across a small set of named groups -> grouped bars, one group per model scale.
# Three bars: the untrained policy, the uniform control's seed mean, the method's seed mean.
#
# COLOUR. Three fixed hues, never cycled, and the same assignment as every other figure here:
# method blue, control green (Okabe-Ito, deuteranopia/protanopia-safe), untrained policy neutral
# grey. Grey is deliberate -- the untrained policy is the REFERENCE the other two are read against,
# not a third competitor -- and because a neutral carries no hue to be told apart by, it also gets
# a hatch and a dashed rule across its group, so its identity never rests on colour alone. Checked:
# worst all-pairs separation is dE 18.0 under protanopia, 18.7 normal.
def transfer(a):
    """Held-out transfer at three scales: base vs uniform vs method, seed means, gap annotated."""
    sys.path.insert(0, f"{R}/surface/verl_rl")
    import bfcl_records as BR

    rows = BR.scale_block()
    if not rows:
        print("no transfer records")
        return 1
    C_BASE, C_UNI, C_TRI = "#333333", OKABE_ITO[2], OKABE_ITO[0]
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    W, GAP = 0.24, 0.035          # bar width and the surface gap that keeps fills from touching
    for i, r in enumerate(rows):
        xs = [i - (W + GAP), i, i + (W + GAP)]
        vals = [r["base"], r["uni_mean"], r["tri_mean"]]
        cols = [C_BASE, C_UNI, C_TRI]
        labs = ["untrained base policy", "uniform GRPO", METHOD_NAME]
        # the top a value label must clear: the bar, or the seed whisker where there is one
        tops = [r["base"],
                max(r["uni"]) if len(r["uni"]) > 1 else r["uni_mean"],
                max(r["tri"]) if len(r["tri"]) > 1 else r["tri_mean"]]
        for x, v, c, lab, tp in zip(xs, vals, cols, labs, tops):
            base_bar = c == C_BASE
            ax.bar(x, v, width=W, color=c,
                   edgecolor=c if not base_bar else "#6f6f6f", lw=0 if not base_bar else 0.7,
                   hatch="////" if base_bar else None,
                   alpha=0.28 if base_bar else 1.0,
                   label=lab if i == 0 else None, zorder=3)
            # white halo: the untrained-policy rule runs through this row of labels at 8B,
            # where the control's bar top and the reference level are 2.8 pp apart
            ax.text(x, tp + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5,
                    color="#333333", zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
        # the untrained policy as an explicit rule across its own group: above it the training
        # helped, below it the training hurt, which is the whole reading of this figure
        ax.plot([i - 0.42, i + 0.42], [r["base"]] * 2, color=C_BASE, lw=1.0, ls=(0, (3, 2.5)),
                zorder=5)
        # seed range as a thin whisker; a bar that is a mean of seeds must show the spread
        for x, v, lo, hi, c in ((xs[1], r["uni_mean"], min(r["uni"]), max(r["uni"]), C_UNI),
                                (xs[2], r["tri_mean"], min(r["tri"]), max(r["tri"]), C_TRI)):
            if hi > lo:
                ax.plot([x, x], [lo, hi], color="white", lw=3.0, zorder=5)
                ax.plot([x, x], [lo, hi], color=c, lw=1.4, zorder=6)
        # the gap the paper claims, annotated between the two bars it is a difference of
        top = max(max(r["uni"]), max(r["tri"])) + 5.4
        ax.annotate("", xy=(xs[2], top), xytext=(xs[1], top),
                    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0, shrinkA=0,
                                    shrinkB=0), zorder=4)
        ax.text(i + W / 2, top + 0.8, f"{r['tu_mean']:+.1f} pp", ha="center", va="bottom",
                fontsize=9.5, color="#333333", fontweight="bold", zorder=4)
    ax.set_xticks(range(len(rows)))
    # NO seed annotation on the canvas (2026-08-17 PI directive): the counts and the fact that a
    # bar is a seed mean and a whisker a seed range are one clause of the caption.
    ax.set_xticklabels([r["scale"] for r in rows], fontsize=11)
    ax.set_ylabel("BFCL v4 multi-turn pass rate (%)", fontsize=10)
    ax.set_ylim(0, max(max(r["tri"]) for r in rows) + 14)
    ax.tick_params(axis="y", labelsize=9, colors="#444444")
    ax.tick_params(axis="x", length=0, colors="#222222")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")
    ax.grid(axis="y", color="#000000", alpha=0.07, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3, columnspacing=1.3,
              handlelength=1.3, borderaxespad=0.1)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[transfer] {len(rows)} scales -> {a.out}")
    for r in rows:
        print(f"  {r['scale']}: base {r['base']:.2f}  uniform {r['uni_mean']:.2f} "
              f"({len(r['uni'])} seeds)  {METHOD_NAME} {r['tri_mean']:.2f} ({len(r['tri'])} seeds)  "
              f"gap {r['tu_mean']:+.2f}pp over {r['tu_npair']} pairings, "
              f"weakest p={r['tu_pmax']:.2g}")
    return 0


# =================================================================================================
# VALIDATION SOLVE RATE (--mode valsolve). THE MAIN-TEXT TRAINING-TIME FIGURE.
#
# WHAT IT PLOTS AND HOW IT DIFFERS FROM EVERY OTHER NUMBER IN THE PAPER. This is the TRAINING-TIME
# measurement: each arm's own held-out validation pool (pool_coadapt_heldout, 0% overlap with the
# training pool), one FIXED tool surface for every arm, greedy decoding, binary solve. It is not
# the post-hoc held-out eval cells the tables report, and it is not paired against a base anchor --
# it is an absolute solve rate, which is why the two arms are compared against each other on the
# panel rather than against zero. Curves come from val_curve.solve_curve(), which recovers each
# validation pass from episodes.jsonl and only keeps a pass whose own mean reproduces verl's logged
# val-core value for the step it was aligned to.
#
# WHY THE ARM SETS ARE IMPORTED. They are the rows of tab:termination, and they must not drift from
# it: the table's val-solve columns and this figure are the same episodes of the same arms, so one
# definition serves both. The 8B control is a8Fr alone -- a8F's supervisor line never set
# VAL_BEFORE=True, so that arm logged zero val episodes and cannot appear here at any price.
#
# THE FIXED-SEED-SET RULE, which is the whole reason this function is longer than it looks. Seeds
# have unequal horizons (q4bT2 stops at step 25 while q4bT runs to 48). A seed mean taken over
# "whichever seeds have a point here" MOVES WHEN A SEED DROPS OUT, so a curve drawn that way shows
# composition change as if it were learning -- at 2B the uniform mean would fall 0.4pp at step 30
# purely because its best seed's run ended at 25. So the solid line and its band are drawn ONLY
# over the span where every seed of that set has a measurement, and the seeds that run longer are
# continued as thin individual lines, which is what they are.
def _place_labels(fig, wanted, ink, legend, fontsize=8.5):
    """Put each endpoint-change label in the first candidate slot that lands on no drawn line.

    Hand-picked offsets do not survive this figure. The two 4B arms sit 0.2pp apart at the step the
    change is measured to, and every fixed offset tried put one label's white halo across either
    the control's descending line or the method's single-seed continuation -- which does not just
    look untidy, it ERASES a segment and makes a continuous line read as a broken one. So the label
    is placed against the rendered geometry: candidate offsets are tried outward in both directions
    and the first one clear of every polyline in that panel, of the legend, and of the labels
    already placed, wins. Deterministic, and it re-solves itself when a cell lands and a curve
    moves.
    """
    fig.canvas.draw()                       # transData is only meaningful once the layout is fixed
    # transData is in PIXELS and font sizes are in POINTS. Conflating the two silently shrank every
    # collision box by 28% at dpi 100, which is how the first render put a label across the 4B
    # control's descending line and reported no collision.
    px = fig.dpi / 72.0
    placed = {}
    for ax, x, y, text, prefer_up, col in wanted:
        anchor = ax.transData.transform((x, y))
        w = (0.62 * fontsize * len(text) + 4.0) * px   # a digit is ~0.62 em in DejaVu Sans
        h = (fontsize + 4.0) * px
        # transform() takes and returns (N, 2). Wrapping the RESULT in column_stack again reads it
        # as N points of length 2 and hands back a (2, N) array -- two "segments" spanning the whole
        # panel, which is why the first version of this check passed everything.
        lines = [ax.transData.transform(np.column_stack(p)) for p in ink[ax]]
        blockers = list(placed.get(ax, []))
        if legend.axes is ax:
            b = legend.get_window_extent(fig.canvas.get_renderer())
            blockers.append((b.x0, b.y0, b.x1, b.y1))
        signs = (1, -1) if prefer_up else (-1, 1)
        best = None
        for d in (10 * px, 16 * px, 22 * px, 28 * px, 34 * px, 40 * px):
            for s in signs:
                for dx in (4.0 * px, -4.0 * px - w):
                    box = (anchor[0] + dx, anchor[1] + s * d - h / 2,
                           anchor[0] + dx + w, anchor[1] + s * d + h / 2)
                    ab = ax.get_window_extent(fig.canvas.get_renderer())
                    if box[0] < ab.x0 or box[2] > ab.x1 or box[1] < ab.y0 or box[3] > ab.y1:
                        continue
                    hit = any(b[0] < box[2] and box[0] < b[2] and b[1] < box[3] and box[1] < b[3]
                              for b in blockers)
                    for pts in lines:
                        if hit:
                            break
                        # Sample each segment: a label can straddle a steep line whose vertices
                        # both fall outside the box.
                        for i in range(len(pts) - 1):
                            t = np.linspace(0, 1, 24)[:, None]
                            q = pts[i] + t * (pts[i + 1] - pts[i])
                            if np.any((q[:, 0] >= box[0]) & (q[:, 0] <= box[2])
                                      & (q[:, 1] >= box[1]) & (q[:, 1] <= box[3])):
                                hit = True
                                break
                    if not hit:
                        best = box
                        break
                if best:
                    break
            if best:
                break
        if best is None:                    # nothing clear: fall back to the preferred side
            s, d = (1 if prefer_up else -1), 10 * px
            best = (anchor[0] + 4.0 * px, anchor[1] + s * d - h / 2,
                    anchor[0] + 4.0 * px + w, anchor[1] + s * d + h / 2)
            print(f"  ! no clear slot for label {text!r}; placed at the preferred offset")
        placed.setdefault(ax, []).append(best)
        # Text in text ink; the LEADER carries identity in the series colour. Without it the 4B
        # panel is unreadable -- the two arms are within 0.2pp of each other at the measuring step,
        # so neither label is nearer to its own line than to the other one.
        ax.annotate(text, xy=(x, y), xycoords="data", textcoords="data",
                    xytext=ax.transData.inverted().transform(
                        ((best[0] + best[2]) / 2, (best[1] + best[3]) / 2)),
                    fontsize=fontsize, color="#222222", va="center", ha="center", zorder=6,
                    # Finely dotted and lighter than any series line, so a leader that happens to
                    # run beside a curve is not read as a curve.
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.75,
                                    linestyle=(0, (1, 2)), shrinkA=2.0, shrinkB=4.5),
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])


def valsolve(a, figsize=None):
    """Validation solve rate vs training step, method vs uniform control, one panel per scale."""
    sys.path.insert(0, f"{R}/surface/verl_rl")
    import val_curve as VC
    from paper_numbers import TERM              # same rows as tab:termination, one definition

    if figsize is None:
        # \textwidth is 397.5pt = 5.51in. Drawn at that size so the figure is included at 1:1 and
        # its 8.5-10pt type is 8.5-10pt on the page rather than whatever downscaling leaves.
        figsize = (5.5, 2.55)
    fig, axes = plt.subplots(1, len(TERM), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    log, ink, wanted = [], {}, []
    for ax, (scale, rows) in zip(axes, TERM):
        # rows[0] is the method, rows[1] the matched uniform control -- the order tab:termination
        # prints them in.
        series = []
        for (lab, arms), col, mk, key in ((rows[1], OKABE_ITO[2], "^", "uniform GRPO"),
                                          (rows[0], OKABE_ITO[0], "o", METHOD_NAME)):
            per = {arm: {r["step"]: 100 * r["solve"] for r in VC.solve_curve(arm)} for arm in arms}
            per = {k: v for k, v in per.items() if v}
            if per:
                series.append((key, col, mk, per,
                               sorted(set.intersection(*(set(v) for v in per.values())))))
        # EVERY PANEL IS EXACTLY THE FULLY-SEEDED WINDOW: the steps at which both arms still have
        # all of their seeds. Nothing else is drawn. Two reasons, and the second was found by
        # looking at the render.
        #   1. A seed mean over "whichever seeds reach this step" moves when a seed drops out, so
        #      a longer panel would draw composition change as learning.
        #   2. With the seed bands removed, the single-seed continuations that used to sit inside a
        #      band became free-floating fragments -- each one starts at ITS OWN value at the last
        #      fully-seeded step, which is not where the mean line ends, so they rendered as broken
        #      pieces of the mean rather than as separate seeds. One of them (a 2B control seed
        #      falling to 5.4%) also dragged the shared y-axis down by two points, flattening the
        #      gaps in all three panels.
        # What lies past the window is real and is stated in the caption and the text instead: at
        # 4B it is where the table's separation comes from, and it is one seed per arm.
        xmin = max(s[4][0] for s in series)
        xmax = min(s[4][-1] for s in series)
        drawn = []
        for key, col, mk, per, full in series:
            full = [x for x in full if xmin <= x <= xmax]
            ys = [sum(v[x] for v in per.values()) / len(per) for x in full]
            ink.setdefault(ax, []).append((full, ys))
            # NO SEED BAND. The gaps this figure exists to show are 1-5pp and the seed ranges are
            # 2-7pp wide, so a tinted range band covered the very thing it was drawn beside and the
            # panels read as two overlapping clouds. The spread is not dropped, it is moved into
            # the caption as a number -- printed below so the caption can be checked against it.
            spread = max((max(v[x] for v in per.values()) - min(v[x] for v in per.values())
                          for x in full), default=0.0) if len(per) > 1 else None
            ax.plot(full, ys, color=col, lw=2.2, marker=mk, ms=4.4, mew=1.3, mfc="white", mec=col,
                    label=key, zorder=4)
            drawn.append((key, col, full, per, spread))
        # The change is measured across the whole drawn window, at the same two steps for both
        # arms -- a change read off two different windows is not a comparison. With the panel now
        # ending at that window, the guide rule that used to mark it is redundant and is gone.
        x_lo, x_hi = xmin, xmax
        # A gutter on the right, because the labels attach to the line ends and the line ends are
        # now always at the panel edge; without it the only slot left is on top of the curve.
        ax.set_xlim(xmin - 1.5, xmax + 0.30 * (xmax - xmin) + 4.0)
        # The labels are placed after the layout is settled (see _place_labels): where they can go
        # depends on where the lines actually are on the page, and at 4B the two arms are 0.2pp
        # apart at the measuring step, so every hand-chosen offset put one label on a line.
        ends = {k: sum(v[x_hi] for v in p.values()) / len(p) for k, _, _, p, _ in drawn}
        for key, col, full, per, spread in drawn:
            y0 = sum(v[x_lo] for v in per.values()) / len(per)
            y1 = ends[key]
            wanted.append((ax, x_hi, y1, "%+.1f" % (round(y1 - y0, 1) + 0.0),
                           y1 >= max(ends.values()), col))
            log.append((scale, key, len(per), x_lo, x_hi, y0, y1,
                        max(x for v in per.values() for x in v), full[-1], spread))
        ax.set_title(scale, fontsize=10)
        ax.set_xticks([x for x in (0, 10, 20, 30, 40) if xmin - 1 <= x <= xmax + 1])
        ax.tick_params(labelsize=8.5, colors="#444444")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#999999")
        # Stop the x spine at the last measured step. The gutter to its right exists only to hold
        # the two change labels, and an axis line ruled across it reads as range that was measured
        # and came back empty.
        ax.spines["bottom"].set_bounds(xmin, xmax)
        ax.grid(axis="y", color="#000000", alpha=0.07, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("validation solve rate (%)", fontsize=9)
    axes[len(axes) // 2].set_xlabel("training step (GRPO)", fontsize=9)
    leg = axes[0].legend(frameon=False, fontsize=8.5, loc="upper left",
                         handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    _place_labels(fig, wanted, ink, leg)
    # NOT bbox_inches="tight": that trims to the ink and then adds a 0.1in pad on every side, so
    # the saved page is narrower than 5.5in, \includegraphics scales it back up to \textwidth, and
    # the type no longer prints at the size it was set at. Saved at exactly the figure size, the
    # scale factor is 5.5 -> 5.51in and the panels get the whole measure.
    fig.savefig(a.out)
    print(f"[valsolve] {len(TERM)} scales -> {a.out}")
    for scale, key, nseed, x0, x1, y0, y1, xlast, xfull, spread in log:
        print(f"  {scale:>3} {key:<13} {nseed} seed(s)  paired window {x0}->{x1}: "
              f"{y0:.2f} -> {y1:.2f} ({y1 - y0:+.2f} pp);  all seeds to step {xfull}, "
              f"longest seed to {xlast}")
    for i in range(0, len(log), 2):
        u, t = log[i], log[i + 1]
        print(f"  {u[0]:>3} gap at step {u[4]}: {METHOD_NAME} {t[6]:.2f} - uniform {u[6]:.2f} = "
              f"{t[6] - u[6]:+.2f} pp")
    # The bands are gone, so this is the only place the spread is stated. The caption must carry
    # it; a caption that disagrees with this block is a caption that is wrong.
    print("  -- widest seed spread at any plotted step (the band the caption replaces) --")
    for scale, key, nseed, _, _, _, _, _, _, spread in log:
        print(f"     {scale:>3} {key:<13} {nseed} seed(s)  "
              + ("single seed, no spread" if spread is None else f"{spread:.2f} pp"))
    return 0


def forest(a):
    """Effect-size forest, ONE ROW PER METHOD (seeds aggregated), PERCENTAGE POINTS.
    Per-seed traces are gone: a chart whose rows are seeds invites ranking seeds, which is what
    this paper says the benchmark cannot support. Where a method has replicates the bar is the
    SEED range -- the honest width to compare any between-method gap against -- and where it has
    one seed the bar is the range over its evaluated steps. The two are distinguished in the key."""
    base = load_cell(a.base)
    GROUPS = [("Method", "#0072B2", [
                  (METHOD_NAME, ["a8T3g", "a8T3gr", "a8T3g2"]),
                  (METHOD_NAME + ", sharpened", ["a8T5k", "a8T5kr"])]),
              ("Our ablations", "#56B4E9", [
                  ("difficulty band filter", ["a8T2n", "a8T2nr"]),
                  ("plug-in estimator", ["a8Tpe"])]),
              ("Control", "#009E73", [
                  ("uniform GRPO", ["a8F", "a8Fr"])]),
              ("Published", "#8a8a8a", [
                  # 2026-08-24: PLR is two seeds from this pass (b8plrr, offset 1500), so its bar
                  # becomes a SEED range like VIP's, the method's and the control's instead of a
                  # step range. Kept identical to paper_numbers.MAIN's own arm list so the figure
                  # and tab:main cannot drift.
                  ("PLR-style replay", ["b8plr", "b8plrr"]), ("retrieval surface", ["b8ret"]),
                  # 2026-08-22: VIP is two seeds from this pass (a8Tvipr is the clean rerun of the
                  # optimizer-restarted first seed), so its bar becomes a SEED range like the
                  # method's and the control's instead of a step range. Kept identical to
                  # paper_numbers.MAIN's own arm list so the figure and tab:main cannot drift.
                  ("VIP", ["a8Tvip", "a8Tvipr"]), ("DAPO", ["dapo"]),
                  ("learning progress", ["a8Tlp"])])]
    HOLM = {"a8T3g2", "a8T5k", "a8Tpe"}
    rows = []
    for gname, col, methods in GROUPS:
        for lab, arms in methods:
            seeds = []
            for arm in arms:
                v = [paired(load_cell(cell_path(arm, st)), base) for st in [15, 20, 25, 30]]
                v = [100 * r[0] for r in v if r and r[2] >= a.min_pairs]
                if v:
                    seeds.append((arm, sum(v) / len(v), min(v), max(v), len(v)))
            if not seeds:
                continue
            ms = [x[1] for x in seeds]
            mean = sum(ms) / len(ms)
            if len(ms) > 1:
                lo, hi, kind = min(ms), max(ms), "seed"
            else:
                lo, hi, kind = seeds[0][2], seeds[0][3], "step"
            rows.append(dict(lab=lab, col=col, mean=mean, lo=lo, hi=hi, kind=kind,
                             n=len(ms), grp=gname,
                             holm=any(x[0] in HOLM for x in seeds),
                             cells=min(x[4] for x in seeds)))
    rows.sort(key=lambda r: r["mean"])
    # COMPACT: the half-width main-text cut. Same arms, no per-row seed-count annotations (the
    # key already says what a bar is), tighter type, so the figure can sit beside its table
    # instead of costing a full column of its own.
    C = a.compact
    fig, ax = plt.subplots(figsize=(3.45, 3.5) if C else (7.2, 4.2))
    ax.axvline(0, color="#444444", lw=1.1, ls=(0, (4, 3)), zorder=1)
    for i, r in enumerate(rows):
        ls = "-" if r["kind"] == "seed" else (0, (1.6, 1.6))
        ax.plot([r["lo"], r["hi"]], [i, i], color=r["col"], lw=2.6, alpha=0.6, ls=ls,
                solid_capstyle="round", zorder=2)
        ax.plot([r["mean"]], [i], marker="o", ms=9,
                mfc=r["col"] if r["holm"] else "white", mec=r["col"], mew=2.4, zorder=3)
        if not C:
            note = f"{r['n']} seeds" if r["n"] > 1 else ("1 seed, %d/4 steps" % r["cells"]
                                                         if r["cells"] < 4 else "1 seed")
            ax.text(r["hi"] + 0.18, i, note, fontsize=8, color="#777777", va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["lab"] for r in rows], fontsize=7.6 if C else 10)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(r["col"] if r["grp"] != "Published" else "#555555")
        if r["grp"] == "Method":
            tick.set_fontweight("bold")
    ax.set_xlabel("effect over the base policy (pp)" if C else
                  "effect over the base policy (percentage points); higher is better",
                  fontsize=8.5 if C else 10)
    ax.tick_params(axis="x", labelsize=8 if C else 10, colors="#444444")
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.grid(axis="x", color="#000000", alpha=0.07, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(right=max(r["hi"] for r in rows) + (0.6 if C else 2.6))
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=c, lw=2.6, marker="o", mfc="white", mec=c, mew=2.0, ms=7,
                   label=n) for n, c, _ in GROUPS]
    keys += [Line2D([], [], color="#555555", lw=2.4, ls="-", label="bar: seed range"),
             Line2D([], [], color="#555555", lw=2.4, ls=(0, (1.6, 1.6)), label="bar: step range"),
             Line2D([], [], color="#555555", lw=0, marker="o", mfc="#555555", mec="#555555",
                    ms=7, label="clears Holm")]
    ax.legend(handles=keys, frameon=False, fontsize=6.6 if C else 8.8, loc="upper center",
              bbox_to_anchor=(0.5, 1.24 if C else 1.15), ncol=2 if C else 4,
              columnspacing=1.0 if C else 1.2, handletextpad=0.4 if C else 0.5)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[forest] {len(rows)} methods -> {a.out}")
    for r in rows:
        print(f"  {r['lab']:24s} {r['mean']:+.1f}pp  [{r['lo']:+.1f},{r['hi']:+.1f}]  "
              f"{r['n']} seed(s), {r['kind']} range")
    return 0




# =================================================================================================
# FAILURE-MODE TAXONOMY (--mode failuretax). THE MAIN-TEXT PROTECTION-MECHANISM FIGURE.
#
# Counts are per-instance labels over all 800 BFCL v4 multi-turn instances/arm, priority-ordered
# and mutually exclusive (they sum to 800 for every arm, asserted at measurement time), from
# work/analysis/protection_mechanism.md Section 1. That document's reproduction scripts
# (taxonomy.py, drift2.py) read only the read-only archive paths named in its Section 0 and are not
# committed to this repo (session scratchpad); the counts are hard-coded here for the same reason
# paper_numbers.FROZEN_MECH hard-codes the DAPO row -- the number is measured and cited, not the
# live recomputation path. Every row here was independently spot-checked against
# $BRACE_WORK/bfcl/run_<arm>/records.jsonl (pass rate) and
# .../bfcl_root/score_s0/.../*_score.json (error_type) before being written in.
_TAXONOMY_ARMS = {
    # arm:     correct, runaway, silent, wrongFN, wrongARG, crash
    "q2bTnw":  (18, 666, 0, 28, 88, 0),
    "q2bF5":   (36, 30, 153, 443, 138, 0),
    "q2bF7":   (51, 0, 165, 339, 36, 209),
    "q2bF6":   (68, 5, 279, 361, 87, 0),
    "q2bTiid": (109, 10, 106, 364, 211, 0),
    "base2b":  (121, 196, 113, 209, 160, 1),
    "q2bT":    (133, 92, 134, 183, 251, 7),
}
# THE THREE UNIFORM CONTROLS ARE AVERAGED INTO ONE BAR, not shown as three (2026-08-18 PI
# directive: no seed rows/columns/legends in main-text floats, averages only, counts in the
# caption). q2bF5/q2bF6/q2bF7 are the matched-rate uniform seeds of Table~scalemain's 2B row;
# averaging them here is the same operation that table already performs on their held-out effect.
_UNI3 = ["q2bF5", "q2bF6", "q2bF7"]
_TAXONOMY = dict(_TAXONOMY_ARMS)
_TAXONOMY["uniform"] = tuple(sum(_TAXONOMY_ARMS[a][i] for a in _UNI3) / len(_UNI3)
                             for i in range(6))
_TAXONOMY_LABEL = {"q2bTnw": METHOD_NAME + "\n$-$ warm bank", "uniform": "uniform GRPO\n(mean of 3)",
                   "q2bTiid": METHOD_NAME + "\n$-$ calibration", "base2b": "base policy,\nuntrained",
                   "q2bT": METHOD_NAME + "\n(full method)"}
_TAXONOMY_ORDER = ["q2bTnw", "uniform", "q2bTiid", "base2b", "q2bT"]  # correct count, ascending
_SEG_NAMES = ["correct", "runaway\n(never stops)", "silent turn\n(never acts)",
              "wrong function", "wrong arguments", "crash"]
_SEG_COLORS = ["#009E73", "#D55E00", "#0072B2", "#BBBBBB", "#777777", "#222222"]


def failuretax(a):
    """Stacked failure-mode composition, 2B, BFCL v4 multi-turn, 800 instances/arm.

    THE POINT OF THE FIGURE. Removing the warm bank does not move the policy along the SAME axis
    uniform allocation fails on (under-acting: silent turns, wrong-function selection); it inverts
    it (over-acting: 83.2% runaway, exactly 0 silent turns in 800 instances -- the only arm in the
    whole grid that never once answers a turn without calling something). One sentence of caption
    is meant to carry that, so the bars are ordered by `correct` ascending and `runaway` /
    `silent turn` are drawn in opposed hues, the one contrast this figure exists to show.
    """
    for k, v in _TAXONOMY.items():
        assert abs(sum(v) - 800) < 1e-6, "taxonomy row for %r does not sum to 800 (%s)" % (k, sum(v))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    xs = list(range(len(_TAXONOMY_ORDER)))
    bottoms = [0.0] * len(xs)
    for seg_i, (seg_name, col) in enumerate(zip(_SEG_NAMES, _SEG_COLORS)):
        vals = [100.0 * _TAXONOMY[arm][seg_i] / 800.0 for arm in _TAXONOMY_ORDER]
        ax.bar(xs, vals, bottom=bottoms, width=0.68, color=col,
               label=seg_name.replace("\n", " "), zorder=3,
               edgecolor="white", linewidth=0.4)
        for x, v, b in zip(xs, vals, bottoms):
            if v >= 6.0:      # label only segments wide enough to hold text legibly
                txt_col = "white" if col in ("#D55E00", "#0072B2", "#777777", "#222222") else "#222222"
                ax.text(x, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=7.6,
                        color=txt_col, zorder=4)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(xs)
    ax.set_xticklabels([_TAXONOMY_LABEL[arm] for arm in _TAXONOMY_ORDER], fontsize=9.2)
    ax.set_ylabel("share of 800 instances (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=9, colors="#444444")
    ax.tick_params(axis="x", length=0, colors="#222222")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")
    ax.grid(axis="y", color="#000000", alpha=0.07, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, 1.28),
              ncol=3, columnspacing=1.1, handlelength=1.3, borderaxespad=0.1)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[failuretax] {len(_TAXONOMY_ORDER)} arms -> {a.out}")
    for arm in _TAXONOMY_ORDER:
        c, r, s, fn, ar, cr = _TAXONOMY[arm]
        print(f"  {arm:<10} correct {c:6.1f} ({100*c/800:.1f}%)  runaway {r:6.1f} ({100*r/800:.1f}%)  "
              f"silent {s:6.1f} ({100*s/800:.1f}%)  wrongFN {fn:6.1f}  wrongARG {ar:6.1f}  "
              f"crash {cr:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
