"""Figures for the paper, generated from banked receipts and logs.

The exemplar venue papers we examined are figure-driven: Agarwal et al. carry 12 figures against
1 table in 28 pages; Patterson et al. 15 figures and no tables. Our draft was the inverse -- six
tables, no figures -- which reads as a lab report rather than a paper. Each figure here replaces
or supplements a table that was carrying the argument alone.

Every number is read from the receipts, never hard-coded, so the figures cannot drift from the
banked results.
"""
from __future__ import annotations

import collections
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = f"{R}/surface/paper/tex/fig"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif",
})
C_PAIR, C_UNPAIR, C_NAIVE = "#1b4965", "#bc4749", "#8a8a8a"


def rd(name):
    p = f"{R}/surface/receipts/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- Fig 0: the identity
# Last verified reproduction of every identity cell, as (caller, substrate, variant, tie,
# predicted, observed). This list is a FALLBACK: fig_identity() reads
# surface/receipts/identity_cells.json, which now exists, so these literals are no longer what the
# figure draws. They are kept as a mirror of the receipt, regenerated from it, so that a lost or
# truncated receipt degrades to the published numbers instead of to a stale ten-cell figure.
#
# Provenance: banked 2026-08-17 by surface/gate_caller/bank_identity_cells.py, which runs the two
# producers itself -- checks/g1_chance_rate.py for MCP (12 ordered interface pairs x 500 draws) and
# substrate_report.py for BFCL and API-Bank (groups of 4, 1,500 trials). The ten cells that existed
# before the API-Bank fill were recomputed from their episode files and every one reproduced to
# three decimals, including the two API-Bank tie rates the appendix table used to round to 0.186
# and 0.136 (the producers print 0.187 and 0.137; the producers' values are used).
#
# API-Bank now carries all four callers on each of two tool surfaces, so all three substrates are
# indexed the same way and panel (b) labels every bar by model. Before the fill it was one caller
# on two surfaces, which forced that group to be labelled by surface instead.
IDENTITY_FALLBACK = [
    ("qwen", "MCP", "4 interfaces", 0.088, 0.456, 0.464),
    ("granite", "MCP", "4 interfaces", 0.082, 0.459, 0.455),
    ("mistral", "MCP", "4 interfaces", 0.149, 0.426, 0.427),
    ("falcon3", "MCP", "4 interfaces", 0.107, 0.447, 0.446),
    ("qwen", "BFCL", "live_multiple", 0.223, 0.389, 0.377),
    ("granite", "BFCL", "live_multiple", 0.118, 0.441, 0.451),
    ("mistral", "BFCL", "live_multiple", 0.067, 0.466, 0.460),
    ("falcon3", "BFCL", "live_multiple", 0.110, 0.445, 0.442),
    ("qwen", "API-Bank", "gold", 0.187, 0.407, 0.383),
    ("granite", "API-Bank", "gold", 0.069, 0.466, 0.479),
    ("mistral", "API-Bank", "gold", 0.058, 0.471, 0.477),
    ("falcon3", "API-Bank", "gold", 0.068, 0.466, 0.473),
    ("qwen", "API-Bank", "49 tools", 0.137, 0.432, 0.423),
    ("granite", "API-Bank", "49 tools", 0.085, 0.457, 0.465),
    ("mistral", "API-Bank", "49 tools", 0.051, 0.474, 0.459),
    ("falcon3", "API-Bank", "49 tools", 0.083, 0.459, 0.465),
]

# Group-label names for tool-surface variants. These sit UNDER the axis with four bar-widths of
# room, not in a tick slot with one, so they use the appendix's wording ("published", "49 tools")
# rather than the runner's flag values (gold, all) -- the reader of the figure has the appendix,
# not the command line. Bars themselves are always labelled by model.
VARIANT_LABEL = {"49 tools": "49 tools", "gold": "published"}


def fig_identity():
    """Predicted vs observed admission rate. The paper's central claim, currently table-only."""
    banked = rd("identity_cells")
    rows = ([(c["caller"], c["substrate"], c["variant"], c["tie"], c["predicted"], c["observed"])
             for c in banked["cells"]] if banked else IDENTITY_FALLBACK)
    # Panel (b) carries 16 bars against panel (a)'s 16 points, so it is given the wider column and
    # the figure a little more height. At ten cells the two panels could be equal; at sixteen, equal
    # columns printed the four model codes of each group as one word ("QwGrMiFa") at 0.86\textwidth.
    # 2026-08-28 PRESENTATION PASS II. LEGIBILITY IS A SIZE PROBLEM AND IT IS FIXED HERE, NOT IN
    # THE FLOAT. This figure was drawn at 6.6in and placed at 0.495\textwidth = 2.73in -- a scale
    # factor of 0.41, so its 9pt labels printed at about 3.7pt and its 7pt tick labels at 2.9pt,
    # which is not readable at any print size. It is now drawn at 5.5in, which is \textwidth
    # (397.5pt = 5.52in) to within a rounding, and the float stacks it at \textwidth: the scale
    # factor is 1.0 and every label prints at exactly the size it is set at below. Both panels
    # and every number in them are unchanged; only the canvas is.
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.68),
                             gridspec_kw={"width_ratios": [1.0, 1.5], "wspace": 0.26})
    ax = axes[0]
    for sub, mk, col in (("MCP", "o", "#2a78d6"), ("BFCL", "^", "#eb6834"),
                         ("API-Bank", "s", "#1baf7a")):
        xs = [r[4] for r in rows if r[1] == sub]
        ys = [r[5] for r in rows if r[1] == sub]
        ax.scatter(xs, ys, marker=mk, s=42, color=col, edgecolor="white", lw=0.6,
                   label=sub, zorder=3)
    # Bounds follow the data. They were the literals 0.36 and 0.49, fitted to ten cells; adding
    # six more would have pushed points off the axis, and a point silently outside the frame of a
    # scatter whose whole argument is "they lie on the line" is the worst failure this figure has.
    vals = [v for r in rows for v in (r[4], r[5])]
    pad = max(0.012, 0.10 * (max(vals) - min(vals)))
    lo, hi = min(vals) - pad, max(vals) + pad
    # The line is named in the LEGEND, not by rotated text laid along it. At ten cells there was a
    # clear stretch to write on; at sixteen there is not -- eight of them have a predicted rate in
    # [0.455, 0.475], so the text landed on the API-Bank cluster wherever it was placed near the
    # line, and anywhere clear of the points was too far from the line to be labelling it.
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, zorder=1, label="identity")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # Equal aspect, so the dashed line is drawn at a true 45 degrees. The panel's whole claim is
    # judged by eye against that line, and on a non-square box a point's distance from it reads
    # differently in x than in y. Matching tick steps on both axes for the same reason.
    ax.set_aspect("equal", adjustable="box")
    # 0.04, not 0.02: at 5.5in this panel's axes box is about 1.9in wide, and a 0.02 step put
    # seven x labels into it that set as one run of digits ("0.380.400.42...").
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    ax.set_xlabel(r"predicted $(1-P(\mathrm{tie}))/2$")
    ax.set_ylabel("observed admission rate")
    ax.set_title("prediction uses the tie rate alone")
    ax.legend(frameon=False, fontsize=7, loc="upper left", handletextpad=0.4,
              borderpad=0.2, labelspacing=0.25)
    ax = axes[1]
    sub_col = {"MCP": "#2a78d6", "BFCL": "#eb6834", "API-Bank": "#1baf7a"}
    # ONE letter, not two. Widening the panel was not enough: sixteen bars share a column ~3.0in
    # wide DRAWN, which is ~2.2in PLACED at 0.86\textwidth, so a bar's tick slot is about 8pt and a
    # two-letter code at 7pt sets ~5.5pt wide -- "QwGrMiFa" as one word, the same collision the
    # surface labels had. Widening the step cannot fix it, because the data range is rescaled to the
    # axes width and the slot stays the size it was. The callers appear in the SAME ORDER in every
    # group, and the caption says which order, so the letter is a reminder and not the only key.
    short = {"qwen": "Q", "granite": "G", "mistral": "M", "falcon3": "F"}
    # EVERY bar is labelled by MODEL and every group is one (substrate, tool surface). Until
    # API-Bank had cross-model coverage it was one caller on two surfaces, so that group had to be
    # indexed by surface instead, and panel (b) mixed two indexings. It no longer does: API-Bank is
    # now four callers on each of two surfaces, which is two groups of four, each indexed the same
    # way as MCP and BFCL. The two API-Bank groups keep one colour, because they are one substrate,
    # and sit closer to each other than to their neighbours, because the appendix says to read
    # them as a pair.
    nvar = collections.defaultdict(set)
    for r in rows:
        nvar[r[1]].add(r[2])
    xs, err, cols, ticks = [], [], [], []
    groups, order = collections.defaultdict(list), []
    x = 0.0
    for i, r in enumerate(rows):
        if i:
            prev = rows[i - 1]
            if r[1] != prev[1]:
                x += 1.5                  # gap between substrates
            elif r[2] != prev[2]:
                x += 0.9                  # gap between tool surfaces of one substrate
        xs.append(x); err.append(r[5] - r[4]); cols.append(sub_col[r[1]])
        # Tick labels get one bar-width of room. This figure is DRAWN at 6.6in and PLACED at
        # 0.86\textwidth = 4.74in, so its 7pt labels print at ~5pt and the spacing shrinks with
        # them; a two-letter model code is what reliably fits, which is the other reason every
        # group is now model-indexed. The surface name moved to the group label below the axis,
        # where it has four bar-widths instead of one -- that is where "goldall" used to collide.
        ticks.append(short[r[0]])
        key = (r[1], r[2])
        if key not in groups:
            order.append(key)
        groups[key].append(x)
        x += 1.0
    ax.bar(xs, err, width=0.82, color=cols, edgecolor="white", lw=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, fontsize=7)
    ax.tick_params(axis="x", length=0, pad=2)
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    # Two rows, because the substrate and the tool surface are different things and stacking
    # "API-Bank" over each surface printed that word twice, side by side, nearly touching -- which
    # reads as two substrates. The substrate is named ONCE, centred over all of its bars; the
    # surface is named under each group, and only where a substrate carries more than one, so MCP
    # and BFCL do not acquire a second line saying nothing.
    for sub in dict.fromkeys(s for s, _ in order):
        allpos = [p for (s, v), ps in groups.items() if s == sub for p in ps]
        ax.text(float(np.mean(allpos)), -0.115, sub, transform=tr,
                ha="center", va="top", fontsize=7, color=sub_col[sub])
    for (sub, var) in order:
        if len(nvar[sub]) < 2:
            continue
        ax.text(float(np.mean(groups[(sub, var)])), -0.245, VARIANT_LABEL.get(var, var),
                transform=tr, ha="center", va="top", fontsize=7, color=sub_col[sub])
    ax.set_ylabel("observed $-$ predicted")
    lim = max(abs(e) for e in err)
    ax.set_ylim(-1.2 * lim, 1.2 * lim)
    # Computed, not written down: the bound moves when a cell is added, and a title asserting a
    # tolerance the bars no longer respect is the kind of error nobody re-reads a figure to catch.
    ax.set_title("residuals within $\\pm%.3f$" % lim)
    fig.savefig(f"{OUT}/identity.pdf")
    plt.close(fig)
    print("[identity] %d cells (%s); worst residual %+.3f"
          % (len(rows), "banked" if banked else "fallback literals",
             max(err, key=abs)))


# ---------------------------------------------------------------- Fig 1: sign inversion
def fig_sign():
    d = rd("sign_flip_scale")
    if not d or "pooled" not in d:
        return
    ks = sorted(d["pooled"], key=float)
    x = [float(k) for k in ks]
    cross = [d["pooled"][k]["crossover"] for k in ks]
    single = [d["pooled"][k]["single_round"] for k in ks]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.axhline(0.5, color="k", lw=0.6, ls=":", zorder=1)
    ax.text(0.203, 0.505, "coin flip", fontsize=7, color="k", ha="right", va="bottom")
    ax.plot(x, single, "o-", color=C_UNPAIR, lw=1.6, ms=4, label="single-round", zorder=3)
    ax.plot(x, cross, "s-", color=C_PAIR, lw=1.6, ms=4, label="crossover", zorder=3)
    ax.set_xlabel("realised effect size")
    ax.set_ylabel("sign-inversion rate")
    ax.set_ylim(-0.03, 0.55)
    ax.legend(frameon=False, loc="center right")
    fig.savefig(f"{OUT}/sign_inversion.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2: calibration + power
def fig_calib():
    d = rd("null_power")
    if not d:
        return
    callers = [c for c in ("qwen", "granite", "mistral", "falcon3") if c in d]
    arms = [("paired_t", "paired $t$", C_PAIR, "//"), ("unpaired_t", "unpaired $t$", C_UNPAIR, "//"),
            ("paired_perm", "paired perm.", C_PAIR, ""), ("unpaired_perm", "unpaired perm.", C_UNPAIR, "")]
    # 2026-08-28: redrawn at \textwidth for the same reason fig_identity was -- see that
    # function's comment. Placed at \textwidth the scale factor is 1.0 and nothing is shrunk.
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.45))
    ax = axes[0]
    w = 0.2
    for i, (k, lab, col, hs) in enumerate(arms):
        v = [d[c][k]["0.00"] for c in callers]
        ax.bar(np.arange(len(callers)) + (i - 1.5) * w, v, w, color=col, hatch=hs,
               edgecolor="white", lw=0.5, label=lab)
    ax.axhline(0.025, color="k", lw=0.9, ls="--")
    # The label sits at the LEFT end of the line: at the right end it landed on panel (b)'s
    # y-axis label once both panels were drawn at half of \textwidth each.
    ax.text(-0.48, 0.0265, "nominal", fontsize=7, ha="left", va="bottom")
    ax.set_ylim(0, 1.45 * max(d[c][k]["0.00"] for c in callers for k, _, _, _ in arms))
    ax.set_xticks(range(len(callers))); ax.set_xticklabels(callers)
    ax.set_ylabel("false-positive rate"); ax.set_title("calibration under a true null")
    ax.legend(frameon=False, ncol=2, fontsize=7, handletextpad=0.4, borderpad=0.2,
              labelspacing=0.25, columnspacing=1.0)
    ax = axes[1]
    # 2026-09-08 (PI: "the bottom right subfigure legend is unclear"). TWO DEFECTS, both fixed here.
    # (1) The default handlelength is about 2.0 font units, which is SHORTER THAN ONE DASH CYCLE at
    #     this figure size, so every handle rendered as a plain solid line and the two curves of a
    #     colour -- the t-test and the permutation test -- were indistinguishable in the legend even
    #     though they are drawn differently in the axes. handlelength is now long enough to show a
    #     full dash cycle, and the dash is set explicitly rather than left to the "--" shorthand.
    # (2) The legend was ordered paired-t / unpaired-t / paired-perm / unpaired-perm, which
    #     interleaves the colours and hides that COLOUR is the pairing and DASH is the test family.
    #     It is now grouped by colour, so the two encodings read off the legend directly.
    ds = ["0.00", "0.05", "0.10", "0.20"]
    handles = {}
    for k, lab, col, hs in arms:
        y = [float(np.mean([d[c][k][t] for c in callers])) for t in ds]
        (ln,) = ax.plot([float(t) for t in ds], y, marker="o", color=col, lw=1.5, ms=3.5,
                        ls=(0, (3.5, 1.5)) if hs else "-", label=lab)
        handles[k] = ln
    ax.set_xlabel("injected effect size"); ax.set_ylabel("power")
    ax.set_title("power, mean over callers")
    grouped = ["paired_t", "paired_perm", "unpaired_t", "unpaired_perm"]
    ax.legend([handles[k] for k in grouped if k in handles],
              [dict((a[0], a[1]) for a in arms)[k] for k in grouped if k in handles],
              frameon=False, fontsize=7, handletextpad=0.5, borderpad=0.2,
              labelspacing=0.28, handlelength=2.9)
    fig.savefig(f"{OUT}/calibration_power.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3: required budget
def fig_budget():
    d = rd("required_budget")
    if not d:
        return
    callers = [c for c in ("qwen", "granite", "mistral", "falcon3") if c in d]
    CAP = 2560
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    w = 0.35
    for i, (arm, col) in enumerate((("paired", C_PAIR), ("unpaired", C_UNPAIR))):
        vals, hatch = [], []
        for c in callers:
            v = d[c].get("0.20", {}).get(arm)
            vals.append(CAP * 2 if v in (None, "None") else int(v))
            hatch.append(v in (None, "None"))
        b = ax.bar(np.arange(len(callers)) + (i - 0.5) * w, vals, w, color=col,
                   edgecolor="white", lw=0.5, label=arm)
        for r, h in zip(b, hatch):
            if h:
                r.set_hatch("xx"); r.set_alpha(0.55)
    ax.axhline(40, color="k", lw=1.0, ls="--")
    ax.text(-0.45, 46, "budget used in practice", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(range(len(callers))); ax.set_xticklabels(callers)
    ax.set_ylabel("tasks for 80% power")
    ax.set_title(r"detecting $\delta{=}0.20$; hatched $=$ beyond grid")
    ax.legend(frameon=False)
    fig.savefig(f"{OUT}/budget.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- evolution histories
def load_ev():
    ev = collections.defaultdict(dict)
    for f in sorted(os.listdir(f"{R}/work/bfcl_evolve")):
        m = re.match(r"history_(paired|unpaired|naive)(?:_h(\d+))?"
                     r"(?:_(granite|mistral|falcon3))?(?:_(full_ps\d+))?\.json", f)
        if not m:
            continue
        gate, h, caller, full = m.group(1), m.group(2) or "555", m.group(3) or "qwen", m.group(4)
        try:
            d = json.load(open(f"{R}/work/bfcl_evolve/{f}"))["history"]
        except Exception:
            continue
        ev[("full" if full else "hcap", caller, full or h)][gate] = d[-1]["obj"] - d[0]["obj"]
    return ev


# ---------------------------------------------------------------- Fig 4: gate divergence
def fig_divergence(ev):
    pts = [(k[1], ev[k]["unpaired"], ev[k]["paired"]) for k in ev
           if k[0] == "hcap" and "paired" in ev[k] and "unpaired" in ev[k]]
    if not pts:
        return
    # categorical identity: one validated hue per caller (dataviz reference order),
    # marker shape as secondary encoding so identity survives grayscale printing
    STYLE = {"qwen":    ("#2a78d6", "o"), "granite": ("#eb6834", "s"),
             "mistral": ("#1baf7a", "^"), "falcon3": ("#eda100", "D")}
    fig, ax = plt.subplots(figsize=(3.25, 2.10))
    xs_all = [p[1] for p in pts]; ys_all = [p[2] for p in pts]
    lo = min(min(xs_all), min(ys_all)); hi = max(max(xs_all), max(ys_all))
    pad = 0.06 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="#8a8a8a", lw=0.9, ls="--", zorder=1)
    for c, (col, mk) in STYLE.items():
        xs = [p[1] for p in pts if p[0] == c]; ys = [p[2] for p in pts if p[0] == c]
        if xs:
            ax.scatter(xs, ys, marker=mk, s=34, color=col,
                       edgecolor="white", lw=0.6, label=c, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("recovery, single-round gate")
    ax.set_ylabel("recovery, crossover gate")
    fig.subplots_adjust(top=0.97)
    ax.legend(frameon=False, fontsize=7, loc="lower right", handletextpad=0.15,
              borderaxespad=0.1)
    ax.text(hi - 0.004, hi - 0.017, "equal", fontsize=7, color="#8a8a8a",
            ha="right", rotation=45)
    fig.savefig(f"{OUT}/divergence.pdf")
    fig.savefig(f"{OUT}/divergence.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5: full-surface regime
def fig_regime(ev):
    full = [k for k in ev if k[0] == "full"]
    if not full:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    order = [("naive", "untested", C_NAIVE), ("unpaired", "unpaired", C_UNPAIR),
             ("paired", "crossover", C_PAIR)]
    for i, (g, lab, col) in enumerate(order):
        v = [ev[k][g] * 100 for k in full if g in ev[k]]
        if not v:
            continue
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).normal(0, 0.055, len(v)),
                   v, s=20, color=col, alpha=0.75, edgecolor="white", lw=0.4, zorder=3)
        ax.hlines(np.mean(v), i - 0.25, i + 0.25, color="k", lw=1.6, zorder=4)
        deg = sum(1 for x in v if x < 0)
        ax.text(i, ax.get_ylim()[0], f"{deg}/{len(v)}", ha="center", va="bottom",
                fontsize=7.5, color="k")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([o[1] for o in order])
    ax.set_ylabel("objective change (points)")
    ax.set_title("full surface: proposals mostly harmful\n(labels: runs that degraded)")
    fig.savefig(f"{OUT}/regime.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6: RL learning curves
def fig_rl():
    def traj(p, tail=None):
        v = [float(x) for x in re.findall(r"critic/score/mean:([0-9.eE+-]+)",
                                          open(p, errors="ignore").read())]
        return np.array(v[-tail:] if tail else v)
    runs = [(f"{R}/logs/rl_bandlong.log", None, "selected, run 1", C_PAIR, "-"),
            (f"{R}/logs/rl_bandlong2.log", 150, "selected, run 2", C_PAIR, "--"),
            (f"{R}/logs/rl_control2.log", None, "unselected, run 2", C_UNPAIR, "-"),
            (f"{R}/logs/rl_controllong.log", None, "unselected, run 1", C_UNPAIR, "--")]
    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    for p, tail, lab, col, ls in runs:
        if not os.path.exists(p):
            continue
        v = traj(p, tail)
        if len(v) < 20:
            continue
        k = 15
        sm = np.convolve(v, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(sm)) + k // 2, sm, ls, color=col, lw=1.7, label=lab)
    ax.set_xlabel("GRPO step"); ax.set_ylabel("reward (15-step mean)")
    ax.set_title("matched compute, 150 steps")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.savefig(f"{OUT}/rl_learning.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7: sequential gate
def fig_sequential():
    d = rd("sequential_gate")
    if not d:
        return
    c = next((k for k in ("granite", "mistral", "falcon3") if k in d and isinstance(d[k], dict)
              and d[k]), None)
    if not c:
        return
    ks = sorted(d[c], key=float)
    x = [float(k) for k in ks]
    sp = [d[c][k]["seq_rate"] for k in ks]
    fp = [d[c][k]["fixed_rate"] for k in ks]
    st = [d[c][k]["seq_tasks"] for k in ks]
    ft = [d[c][k]["fixed_tasks"] for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    ax = axes[0]
    ax.plot(x, fp, "o--", color=C_UNPAIR, lw=1.5, ms=4, label="fixed-$n$")
    ax.plot(x, sp, "s-", color=C_PAIR, lw=1.6, ms=4, label="sequential")
    ax.axhline(0.025, color="k", lw=0.8, ls=":")
    ax.text(0.2, 0.045, "nominal", fontsize=7, ha="right")
    ax.set_xlabel("effect size"); ax.set_ylabel("retention rate")
    ax.set_title("(a) calibration at $\\delta{=}0$, power beyond")
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax = axes[1]
    w = 0.35
    ax.bar(np.arange(len(x)) - w / 2, ft, w, color=C_UNPAIR, edgecolor="white", lw=0.5,
           label="fixed-$n$")
    ax.bar(np.arange(len(x)) + w / 2, st, w, color=C_PAIR, edgecolor="white", lw=0.5,
           label="sequential")
    for i, (a_, b_) in enumerate(zip(ft, st)):
        ax.text(i, max(a_, b_) + 6, f"$-${100*(1-b_/a_):.0f}\\%", ha="center", fontsize=7)
    ax.set_xticks(range(len(x))); ax.set_xticklabels([f"{v:.2f}" for v in x])
    ax.set_xlabel("effect size"); ax.set_ylabel("tasks per decision")
    ax.set_ylim(0, max(ft) * 1.22)
    ax.set_title("(b) evaluation spent per decision")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(f"{OUT}/sequential.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 8: locality
def fig_locality():
    d = rd("locality")
    if not d or not d.get("affected"):
        return
    A = np.array(d["affected"]); U = np.array(d["unaffected"]); N = np.array(d["null_unaffected"])
    sh = np.array(d["shares"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    ax = axes[0]
    xs = np.arange(len(A))
    ax.axhline(0, color="k", lw=0.8)
    ax.bar(xs - 0.22, A, 0.44, color=C_PAIR, edgecolor="white", lw=0.5, label="tasks needing the edited tools")
    ax.bar(xs + 0.22, U, 0.44, color=C_UNPAIR, edgecolor="white", lw=0.5, label="all other tasks")
    ax.plot(xs + 0.22, N, "k_", ms=9, mew=1.4, label="no-edit noise floor")
    ax.set_xticks(xs); ax.set_xticklabels([f"{i+1}" for i in xs])
    ax.set_xlabel("withheld block"); ax.set_ylabel("mean change in task outcome")
    ax.set_title("(a) the effect is confined to relevant tasks")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax = axes[1]
    ks = [1, 3, 5, 10, 20, 51]
    share = [0.006, 0.018, 0.029, 0.059, 0.119, 0.299]
    ax.plot(ks, [1/s for s in share], "o-", color=C_PAIR, lw=1.6, ms=4)
    ax.set_yscale("log"); ax.set_xlabel("function names edited")
    ax.set_ylabel("evaluation saved ($\\times$)")
    ax.set_title("(b) saving from evaluating only relevant tasks")
    for k, s_ in zip(ks, share):
        if k in (3, 51):
            ax.annotate(f"{1/s_:.0f}x", (k, 1/s_), textcoords="offset points",
                        xytext=(6, 4), fontsize=7)
    fig.savefig(f"{OUT}/locality.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ev = load_ev()
    fig_identity(); fig_sign(); fig_calib(); fig_budget()
    fig_divergence(ev); fig_regime(ev); fig_rl(); fig_sequential(); fig_locality()
    made = sorted(os.listdir(OUT))
    print(f"  wrote {len(made)} figures to tex/fig/: {', '.join(made)}")
