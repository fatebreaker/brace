"""Main-text figures whose data is the SAME data the tables are emitted from.

WHY THIS FILE EXISTS AND WHY IT IS NOT IN make_figures.py. surface/paper/make_figures.py reads the
receipts under surface/receipts/ and the gate-half logs; the training-half floats are built from
the paired evaluation cells that paper_numbers.py owns (load_cell / full_family / agg), and there
is exactly one definition of those in this project. So this figure IMPORTS paper_numbers and calls
the same agg() the tables call, rather than re-deriving anything: if tab:main's window column
moves, this figure's lines move with it, and if it cannot, the figure cannot be drawn at all.

    python paper_numbers.py --emit-figures surface/paper/tex/fig

Every value plotted here is ALSO printed as a digit in tab:mainsteps (the four per-step effects)
and tab:main (the window means), so the drawing is checkable against the tables cell by cell.

SEED CONVENTION (PI decision 2026-08-28, applied here and in tab:main / tab:mainsteps /
tab:transfermain): a multi-seed rule's line is that arm's BEST SEED by window mean -- ours, the
control and every multi-seed published baseline under the identical rule -- and the caption says
so. The lines are therefore four cells of ONE run rather than a seed average, which is also why
the panels' curves are rougher than they were: a seed mean smooths a trajectory that a single run
does not have. Every seed of every line, and each configuration's seed mean, are in tab:perseed.
"""
from __future__ import annotations

import os

# THE RESOLUTION FLOOR IS NOT RE-DERIVED HERE. 6.0 pp is the smallest difference two single-seed
# arms on this benchmark could resolve at 80% power; it is computed and stated in the Generality
# subsection from the measured within-configuration seed sd (1.5 pp), and quoted -- as this one
# number -- throughout the paper and in Figure 1's own floor gauge. This figure DRAWS it as a
# length; it does not compute it, and its caption says where the number comes from.
FLOOR_PP = 6.0

# Okabe-Ito, the colour-blind-safe eight. Four series, four hues that survive deuteranopia,
# protanopia and greyscale -- and each also carries its own dash pattern and marker, so the figure
# does not depend on colour at all. The row labels are tab:main's, verbatim, because a figure that
# names its lines differently from the table it is read beside is a figure a reader has to decode.
# THE FIRST THREE HEXES ARE head.tex's SHARED FIGURE PALETTE (trBlue / trVerm / trGreen), so the
# method, its control and the published rule are the same colour in Figure 1 and in this figure.
# The fourth series has no counterpart in Figure 1 and takes Okabe-Ito's reddish purple; the gauge
# takes the palette's grey.
# 2026-08-29 (PI 21:20): THE CONTROL LINE LEAVES THIS FIGURE AND THE SIX PUBLISHED RULES ENTER.
# The panels now show what the paper is asking a reader to compare: \methodname{} at its selected
# configuration against every published allocation rule, which is exactly Table~\ref{tab:main}'s
# row set minus the untrained base policy. The uniform control's per-step cells are NOT deleted
# and did not move: they are in tab:mainsteps, tab:configs and tab:mainfull, where they always
# were, and the caption points there.
#
# SEVEN KEYS, LOOKED UP BY tab:main's OWN ROW LABELS. The keys below are matched against
# PN.T1ROWS, so a row renamed or dropped in Table 1 fails here loudly instead of silently drawing
# something else or one line fewer. That guard is the reason this list is labels and not arms.
#
# ONE STRONG LINE AND SIX QUIET ONES, which is a claim about what the figure is for and not a
# decoration: \methodname{} is trBlue at full weight with filled markers, and the six published
# rules are thin and muted. Each of the six still carries its OWN dash pattern and marker, so the
# figure survives greyscale and deuteranopia and a reader can follow any single baseline; none of
# them is tinted to be illegible. The names in the legend are the formal ones the papers give
# their own methods, identical to Table~\ref{tab:main}'s labels, with no descriptors.
SERIES = [
    # 2026-09-01 (PI 18:40): THE SERIES IS LABELLED "BRACE" AND NOTHING ELSE. It was
    # "BRACE (selected)" from 2026-08-31; the parenthetical is removed from both main-text
    # figures, so a reader meets one name for one thing here, in Figure 4 and in the tables.
    # WHICH configuration the line is (fixed-rho at 2B, per-scale rho at 4B, sharpened at 8B) and
    # the fact that it was chosen after the measurements existed are UNCHANGED and still
    # disclosed, in the caption, in Section 7.2's reading paragraph and in the appendix; only the
    # legend text is shorter. Shortening it also returns the saved width the longer label cost
    # (the legend row, not the axes, sets this figure's width -- see the fig.legend comment).
    ("\\quad \\methodname{}", "BRACE", "#0072B2", "-", "o", 1.9, 3.6),
    ("\\quad \\textsc{Plr}", "PLR", "#6E7B8B", "--", "s", 0.95, 2.5),
    ("\\quad \\textsc{Rag}-\\textsc{Mcp}", "RAG-MCP", "#8C8C8C", "-.", "^", 0.95, 2.5),
    ("\\quad \\textsc{Dapo}", "DAPO", "#B07AA1", ":", "D", 0.95, 2.2),
    ("\\quad \\textsc{Tscl}", "TSCL", "#5F9E93", (0, (3, 1, 1, 1)), "v", 0.95, 2.5),
    ("\\quad \\textsc{Trace}", "TRACE", "#A6893C", (0, (5, 2)), "P", 0.95, 2.6),
    ("\\quad \\textsc{Vip}", "VIP", "#9C6B5E", (0, (1, 1.4)), "X", 0.95, 2.6),
]
GAUGE = "#8C8C8C"

# --- POST-WINDOW: NO PANEL OF ITS OWN, AND HOW IT COMES BACK ----------------------------------
# PI decision 2026-08-29 16:25: the separate "8B, post-window" FOURTH PANEL is removed. This
# figure is the three scale panels over the pre-registered window steps 15-30 and nothing else.
# A fourth panel that exists at one scale reads as a result about that scale; it was drawn when
# 8B was the only scale with post-window cells, and one-scale-only is exactly the asymmetry the
# rest of this paper refuses.
#
# WHAT REPLACES IT, AT A LATER FOLD, ONCE CELLS EXIST AT ALL THREE SCALES. The post-window cells
# MERGE INTO the three panels rather than standing beside them:
#   * the x axis of every panel extends past step 30 (the window ticks {15,20,25,30} keep their
#     positions; the continuation ticks are that scale's own post-window steps);
#   * each rule's post-window points are drawn as a DASHED CONTINUATION OF THE SAME LINE, in the
#     same colour and marker as its window segment, so one rule is one line across the figure;
#   * the legend gains ONE entry, "post-window (one seed)", describing the dashed segment as a
#     region and not as a further rule, so the four rule entries keep meaning what they mean in
#     the window part;
#   * the window/post-window boundary at step 30 stays visible (a rule at the boundary, or the
#     light shading the fourth panel used), because the reporting rule is unchanged: post-window
#     cells are reported separately and are NEVER folded into a window mean, a Holm q or any
#     arm's window statistic. Table~\ref{tab:durability} stays their record.
#
# THE PRECONDITION IS DATA AT ALL THREE SCALES, NOT LAYOUT. Today only 8B has post-window cells
# (arms a8T3g and a8F over steps 30/35/40/45/50, read through PN.DUR, PN.load_cell and the same
# base anchor tab:durability uses). The 2B post-window shards (q2bTs/q2bFs at 40/50/60) were
# pruned at every step and produced zero cells (PLAN 2026-08-29 02:40), and no 4B post-window cell
# has ever been banked; the continuation arms registered on 2026-08-29 are what fills them.
#
# ONE CAVEAT THE MERGE MUST CARRY, AND IT IS NOT A LAYOUT DETAIL. At 8B the two sides are
# DIFFERENT RUNS on the method side: the window line is a8T3g2 (seed 3, the best seed by window
# mean, +4.6) and the post-window cells are a8T3g (seed 1, +3.9 at the same step 30), so a dashed
# continuation there is a continuation of the RULE and not of one trajectory. The control side is
# the same run on both sides (a8F, +4.6 at step 30 in both). Wherever that holds at a scale, the
# caption must say which side is which before the segments are joined.


def emit_figures(outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import paper_numbers as PN

    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({
        # SIZES ARE SET FOR THE PAGE, NOT FOR THE SCREEN. The saved PDF is placed at \textwidth
        # (397.5pt); with a tight bbox its own width is set by the wider of the axes and the legend
        # row, so the legend below is kept narrower than the axes and the placement scale stays at
        # ~1.0 -- which is the only way a point size set here is the point size that prints. At the
        # previous settings the figure saved 435pt wide, so its 7pt ticks and legend printed at
        # 6.4pt. Nothing here now prints below 7pt.
        "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "axes.titleweight": "normal",
        # 2026-08-29 (PI 22:10: "axis type 8-9 pt"). At the placement scale this float actually
        # gets (0.968) an 8pt setting prints 7.74pt, so the setting is 8.6 and the print is 8.3.
        "legend.fontsize": 8.4, "xtick.labelsize": 8.6, "ytick.labelsize": 8.6,
        "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.22, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "serif",
    })
    fams = {m: PN.full_family(m) for m in PN.MAIN_SCALES}

    # The four rules drawn are a FIXED SET and they are the same four in every panel -- not "the
    # best other baseline at each scale", which would be a per-panel maximum over rows and exactly
    # the selection this paper refuses everywhere else. They are looked up BY tab:main's own row
    # label, so a row that is renamed or removed there fails here loudly instead of silently
    # drawing something else.
    spec = {}
    for kind, lab, sp in PN.T1ROWS:
        if kind == "@HDR" or sp is PN.BASE_ROW:
            continue
        for i, (want, _, _, _, _, _, _) in enumerate(SERIES):
            if lab == want:
                spec[i] = sp
    if sorted(spec) != list(range(len(SERIES))):
        missing = [SERIES[i][0] for i in range(len(SERIES)) if i not in spec]
        raise SystemExit("fig:stepcurve: tab:main no longer carries all %d rows this figure draws "
                         "(missing %s) -- the figure and the table must name the same rows"
                         % (len(SERIES), missing))

    # 5.5in is \textwidth (397.5pt) to within a rounding, so the figure is PLACED at scale 1.0 and
    # every label prints at the size it is set at here. Nothing in it is shrunk on the page.
    # THREE panels, one per scale, at the same x scale, so a reader compares them across panels.
    # 2026-08-29 (PI 21:50): "figure 3 a bit small, the y scale too short". Both are fixed here
    # rather than by scaling the placement: the panels are TALLER (the axes get about 4.4cm of the
    # 6.5cm float instead of 2.4cm of 4.4cm) and the y axis is given real extent with ticks every
    # 2pp, so a 5pp difference is about 40% of the panel height instead of a band near the middle.
    fig, axes = plt.subplots(1, 3, figsize=(5.52, 2.10), sharey=True)
    fig.subplots_adjust(bottom=0.17, top=0.91, left=0.085, right=0.995, wspace=0.13)
    # THE BOTTOM MARGIN IS SET EXPLICITLY, and it is not cosmetic. This float is short, and at the
    # default subplot margins the tick labels and the x label of a 1.1in figure fall BELOW the
    # figure's own bottom edge, into the negative figure coordinates where fig.legend places the
    # legend: the legend then prints on top of "checkpoint step". Reserving the band here keeps the
    # labels inside the figure, so the legend offset below only has to clear the figure box.
    log = []
    for ax, model in zip(axes, PN.MAIN_SCALES):
        fam = fams[model][0]
        ax.axhline(0.0, color="k", lw=0.7, zorder=1)
        for i, (_, name, col, ls, mk, lw, ms) in enumerate(SERIES):
            arms = spec[i].get(model)
            if arms is PN.NOT_RUN or arms is PN.IN_FLIGHT:
                continue
            # best=True, 2026-08-28: a line is one arm's BEST SEED by window mean, the same run
            # tab:main's cell and tab:mainsteps' row are, for every multi-seed rule in the panel --
            # ours and the control alike. A panel that averaged one rule's seeds and printed
            # another's single run would not be comparable across its own curves.
            a = PN.agg(fam, arms or [], best=True)
            if a is None:
                continue
            xs = [s for s in PN.WINDOW if s in a["per"]]
            ys = [100.0 * a["per"][s] for s in xs]
            ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=ms, lw=lw,
                    zorder=4 if i == 0 else 3)
            log.append((model, name, ["%+.1f" % y for y in ys], "%+.1f" % (100 * a["mean"]),
                        a["sel"], a["n"]))
        ax.set_title("Qwen3-VL-%s" % model)
        ax.set_xticks(PN.WINDOW)
        ax.set_xlim(13.0, 35.0)
    # THE FLOOR AS A LENGTH, NOT AS A BAND. Drawn as a capped gauge of exactly FLOOR_PP in the
    # panels' own units, so the reader compares it against the vertical gaps between the curves by
    # eye: no gap in any panel reaches it. A shaded +-FLOOR_PP band about the method's window mean
    # was tried first and covered every panel almost entirely, which reads as a printing fault
    # rather than as the finding it is.
    axes[0].set_ylabel("held-out effect (pp)")
    # 2026-08-29: WIDENED for the six published rules. The control line this figure used to draw
    # never went below -0.1, so -2.4 held; TSCL and VIP at 8B run to -3.9, and an axis that
    # clipped them would hide exactly the baselines the figure now exists to show. Set from the
    # drawn data, not by taste: the extremes are -3.9 (TSCL/VIP at 8B) and +5.8 (the method at
    # 4B), with a margin that keeps the floor gauge and its label inside the panel.
    axes[0].set_ylim(-5.0, 7.0)
    # TICKS EVERY 2pp, so the reader has a rule to measure a gap against rather than a top and a
    # bottom. MultipleLocator and not MaxNLocator: the step is the thing being fixed here.
    from matplotlib.ticker import MultipleLocator
    axes[0].yaxis.set_major_locator(MultipleLocator(2))
    for k, ax in enumerate(axes):
        # The gauge sits to the RIGHT of the last step, in a strip of the panel no curve reaches,
        # and its label is set vertically beside it: laid horizontally inside the panel the label
        # printed on top of the 2B curves, which is the one place a legibility fix must not land.
        # Raised 2026-08-29 with the widened axis: at the old offset the gauge's label hung
        # into the tick-label band at the foot of the panel. The gauge is a LENGTH and its
        # vertical position carries no meaning, so it is free to sit where it is legible.
        y0 = -3.0
        ax.annotate("", xy=(33.1, y0), xytext=(33.1, y0 + FLOOR_PP),
                    arrowprops=dict(arrowstyle="|-|,widthA=0.5,widthB=0.5",
                                    lw=0.9, color=GAUGE, shrinkA=0, shrinkB=0))
        # HORIZONTAL, and short. Set vertically beside the gauge it printed cramped at page size;
        # what the gauge IS is one clause of the caption, so the label only has to carry its length.
        ax.text(33.1, y0 - 0.28, "%.1f pp" % FLOOR_PP, fontsize=8, color=GAUGE,
                ha="center", va="top")
    axes[1].set_xlabel("checkpoint step")

    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=ms, lw=lw, label=n)
               for _, n, c, ls, mk, lw, ms in SERIES]
    # Kept to ONE row and narrower than the axes, so the tight bbox is set by the axes and the
    # figure places at scale ~1.0 (see the rcParams comment above).
    # SEVEN entries in one shared legend below the panels, kept narrower than the axes so the
    # tight bbox is set by the axes and the float still places at scale ~1.0. Four columns puts
    # \methodname{} first on the top row and the six published rules after it.
    # 2026-08-31: the legend row, not the axes, sets this figure's saved width, so lengthening
    # the first label to "BRACE (selected)" widened the PDF from 411.7pt to 415.8pt and cost the
    # placement scale (and with it the printed height, which has a 6.2 cm floor the PI set).
    # Paid for inside the legend rather than by shortening the label back: handles and column
    # gaps trimmed, which costs nothing legible and returns the width.
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.155), handlelength=1.25, handletextpad=0.28,
               columnspacing=0.75, labelspacing=0.2)
    fig.savefig(os.path.join(outdir, "stepcurve.pdf"))
    plt.close(fig)
    print("[stepcurve] wrote %s/stepcurve.pdf" % outdir)
    for model, name, ys, mean, sel, n in log:
        print("   %-3s %-30s %s  window %s  seed %s (best of %d)"
              % (model, name, " ".join(ys), mean, sel, n))
    # fig:transfer is emitted by the SAME --emit-figures run, from the cells --emit-tables
    # captured; see emit_transfer_figure for why it is not re-derived here.
    emit_transfer_figure(outdir)


# ==============================================================================================
# fig:transfer -- THE MAIN-TEXT TRANSFER FLOAT, WHICH USED TO BE A TABLE
# ==============================================================================================
# PI decision 2026-08-29 16:40: Tables 1 and 2 looked alike, so the transfer float becomes a
# FIGURE and the full grid of digits moves to the appendix as tab:transfermain (label unchanged,
# every \ref still resolves). This figure and that table therefore have to carry THE SAME cells.
#
# HOW THAT IS GUARANTEED, AND WHY IT IS NOT "the same loaders, called twice". The values below
# are not re-derived here at all: paper_numbers.emit_tables captures every value AT THE MOMENT IT
# FORMATS IT INTO A CELL of transfermain.tex and writes them beside the table as transferfig.json,
# together with the sha256 of the exact transfermain.tex bytes it wrote. This emitter reads that
# file, re-hashes the transfermain.tex sitting next to it, and REFUSES TO DRAW if the two differ.
# So the figure cannot show a number the shipped table does not print, and a figure built against
# a stale emission cannot be produced silently. Run --emit-tables before --emit-figures.
#
# WHAT IS DRAWN, AND WHAT IS DELIBERATELY NOT.
#  * 2 x 3 panels: benchmark (BFCL v4 multi-turn / NESTFUL) x model scale, one horizontal dot
#    plot per panel, one y position per allocation rule in tab:main's own block order.
#  * The rule names are set ONCE, on the left column, and shared across the three scales.
#  * The vertical grey rule in each panel is that benchmark and scale's UNTRAINED BASE POLICY,
#    labelled with its value, because every delta in the table is read against it.
#  * NO NUMERIC LABELS on the markers. The digits are the appendix table's job; a dot plot that
#    prints its own values twice is a table with extra ink.
#  * NESTFUL step subscripts are NOT drawn. The column is genuinely not step-uniform and the
#    appendix table keeps every subscript; the caption points there rather than implying
#    uniformity by silence.
#  * A cell that was never measured (or is not measurable) gets NO MARKER and a small "n/a" at
#    the panel's left edge, so an absence reads as an absence and never as a low score.
#  * A configuration that COINCIDES with fixed-rho at a scale is drawn as "=" at the fixed-rho
#    position, the same statement the table's "$=$ fixed-$\rho$" cell makes, and not as a second
#    measurement.
#  * The RING is the "(column best)" selection the table's summary row carries: per benchmark and
#    scale, the framework configuration with the largest displayed cell. It is a selection made
#    after the measurements existed, it is drawn on the row it actually came from, and the
#    caption says so in one clause.
# Palette: head.tex's trBlue (ours) / trGrey (the uniform control) / trGreen (published) / trInk.
TRBLUE, TRGREY, TRGREEN, TRINK = "#0072B2", "#8C8C8C", "#00674C", "#262626"

# LaTeX row label -> the plain name printed on the axis, and the marker family. The keys are the
# labels transferfig.json carries, verbatim, so a row renamed in paper_numbers fails here loudly
# instead of silently losing its marker.
#   "cfg"     = one of this framework's configurations   (trBlue, open)
#   "method"  = the fixed-rho method row                 (trBlue, filled)
#   "control" = uniform GRPO                             (trGrey square)
#   "pub"     = a published rule we reimplemented        (trGreen diamond, filled)
#   "faith"   = a published allocator run as published   (trGreen diamond, open)
TRANSFER_ROWS = [
    ("\\quad \\methodname{} (fixed-$\\rho$)", "BRACE (fixed-ρ)", "method"),
    ("\\quad scale-refit $\\rho$", "scale-refit ρ", "cfg"),
    ("\\quad plug-in estimator", "plug-in estimator", "cfg"),
    ("\\quad \\textsc{Vip} rule $+$ bank", "VIP rule + bank", "cfg"),
    ("\\quad sharpened ($\\tau{=}0.3$)", "sharpened (τ=0.3)", "cfg"),
    ("\\quad uniform \\textsc{Grpo}", "uniform GRPO", "control"),
    ("\\quad \\textsc{Plr}", "PLR", "pub"),
    ("\\quad \\textsc{Rag}-\\textsc{Mcp}", "RAG-MCP", "pub"),
    ("\\quad \\textsc{Dapo}", "DAPO", "pub"),
    ("\\quad \\textsc{Tscl}", "TSCL", "pub"),
    ("\\quad \\textsc{Trace}", "TRACE", "faith"),
    ("\\quad \\textsc{Vip}", "VIP", "faith"),
]
# The block boundaries, as y positions to rule a hairline above. tab:main's blocks: Method /
# this framework's other configurations / Control / Published baselines / run as published.
TRANSFER_SEPS = [1, 5, 6, 10]
MARKSTYLE = {
    "method":  dict(marker="o", ms=3.4, color=TRBLUE,  mfc=TRBLUE,   mew=0.9),
    "cfg":     dict(marker="o", ms=3.4, color=TRBLUE,  mfc="none",   mew=0.9),
    "control": dict(marker="s", ms=3.2, color=TRGREY,  mfc=TRGREY,   mew=0.9),
    "pub":     dict(marker="D", ms=2.9, color=TRGREEN, mfc=TRGREEN,  mew=0.9),
    "faith":   dict(marker="D", ms=2.9, color=TRGREEN, mfc="none",   mew=0.9),
}
BENCH = [("bfcl", "     ", "%.1f"), ("nest", "", "%.1f")]


def _transfer_data(outdir):
    """transferfig.json, verified against the transfermain.tex it was written beside."""
    import hashlib
    import json
    texdir = os.path.dirname(os.path.abspath(outdir))
    jp = os.path.join(texdir, "transferfig.json")
    tp = os.path.join(texdir, "transfermain.tex")
    if not os.path.exists(jp):
        raise SystemExit("fig:transfer: %s is missing -- run `paper_numbers.py --emit-tables %s` "
                         "before --emit-figures; this figure is never drawn from re-derived "
                         "numbers" % (jp, texdir))
    d = json.load(open(jp))
    got = hashlib.sha256(open(tp, "rb").read()).hexdigest()
    if got != d["transfermain_sha256"]:
        raise SystemExit("fig:transfer: %s does not match %s (json %s, file %s) -- re-emit the "
                         "tables; the figure must draw the SHIPPED table's cells"
                         % (jp, tp, d["transfermain_sha256"][:12], got[:12]))
    return d


# ==============================================================================================
# RETIRED 2026-08-29 (PI 17:30): "Figure 4 hard to observe, redesign it."
# ==============================================================================================
# The six-panel dot plot below is NOT DELETED and NOT EDITED. It is kept here verbatim, one
# character for one character, and it is simply no longer called: emit_figures() now calls
# emit_transfer_figure(), the two-panel grouped-bar emitter written beneath it. Keeping it live
# but uncalled rather than commenting it out is deliberate -- a commented block is not verbatim
# (every line gains a "# ") and it stops being checked by the interpreter, so a constant it
# depends on could be renamed with nothing failing. This function still parses, still names
# TRANSFER_ROWS / TRANSFER_SEPS / MARKSTYLE / _h, and can be restored by changing one call.
#
# WHY IT WAS RETIRED, IN ONE SENTENCE THAT IS ABOUT LEGIBILITY AND NOT ABOUT TASTE. Twelve rules
# x six panels put 72 markers into 196pt of page at a 7.3pt floor, with each panel on its own
# x range and the reader asked to compare a marker's distance from a dashed line across panels
# whose axes differ; the quantity the paper actually claims -- the DIFFERENCE from the untrained
# base policy -- was never drawn, only implied by that distance. The replacement draws that
# difference directly, as a signed length from a common zero.
def emit_transfer_figure_v1_dotplot(outdir):
    """The six panels of fig:transfer: two benchmark groups x three model scales.

    WHY SIX PANELS IN ONE ROW AND NOT A 2x3 GRID, WHICH IS WHAT WAS ASKED FOR. Measured, not
    preferred. The float has to carry TWELVE allocation rules as named rows, print nothing below
    7pt at page size, and stay inside the height the page-9 budget allows. Twelve labels at a
    7.3pt print size need about 8.2pt of pitch each, so ONE stack of them is ~98pt tall. Stacked
    two deep -- a benchmark per row -- the labels are printed twice and the float measures ~283pt
    (10.0cm) before its caption, which is TALLER than the 245pt table it replaces: the layout
    would cost page space rather than free it. Laid in one row the twelve labels are set ONCE, on
    the left, shared by all six panels, and the float measures about half that. The grid the ask
    describes is preserved as GROUPS: benchmark left-to-right in two blocks of three, scale inside
    each block, which is the same two-factor reading in the other orientation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    d = _transfer_data(outdir)
    scales = d["scales"]
    byrow = {r["label"]: r for r in d["rows"]}
    best = next(r for r in d["rows"] if r["kind"] == "@BEST")["best_cfg"]
    cfglab = d["cfg_row_label"]
    for lab, _, _ in TRANSFER_ROWS:
        if lab not in byrow:
            raise SystemExit("fig:transfer: tab:transfermain no longer carries the row %r -- the "
                             "figure and the table must name the same rules" % lab)

    plt.rcParams.update({
        # Set for the PAGE. The tight bbox trims the saved width to the content, so the placement
        # scale is \textwidth (397.5pt) over the saved width; the figsize below puts that within
        # a couple of percent of 1.0, and NOTHING here prints below 7pt on the page.
        "font.size": 7.3, "axes.titlesize": 7.3, "axes.labelsize": 7.3,
        "axes.titleweight": "normal", "legend.fontsize": 7.3,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.3,
        "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
        "axes.edgecolor": TRINK, "xtick.color": TRINK, "ytick.color": TRINK,
        "text.color": TRINK, "axes.labelcolor": TRINK,
    })
    # Seven columns, the fourth an invisible spacer: it is what separates the two BENCHMARK
    # groups, and a gap is the only thing in this figure that says the left three panels and the
    # right three are different measurements rather than six scales of one.
    fig, axg = plt.subplots(1, 7, figsize=(5.10, 2.00), sharey=True,
                            gridspec_kw={"width_ratios": [1, 1, 1, 0.30, 1, 1, 1],
                                         "wspace": 0.22})
    axg[3].set_visible(False)
    panels = [(0, axg[0]), (1, axg[1]), (2, axg[2]), (0, axg[4]), (1, axg[5]), (2, axg[6])]
    log = []
    for pi, (si, ax) in enumerate(panels):
        bm = "bfcl" if pi < 3 else "nest"
        sc = scales[si]
        base = d["base"][bm][sc]
        vals = [base]
        for lab, _, _ in TRANSFER_ROWS:
            v = byrow[lab][bm].get(sc)
            if isinstance(v, (int, float)):
                vals.append(v)
        lo, hi = min(vals), max(vals)
        pad = max(1.2, 0.09 * (hi - lo))
        x0, x1 = lo - pad * 2.1, hi + pad
        ax.set_xlim(x0, x1)
        # the extra 0.9 at the foot is a BLANK STRIP, not padding: it is where the base
        # policy value prints, clear of the last rule and clear of the tick labels.
        ax.set_ylim(len(TRANSFER_ROWS) + 0.75, -0.6)
        ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
        # THE UNTRAINED BASE POLICY, as a line and as a number. Every Delta in the appendix table
        # is read against it, so a reader has to see which side of it a marker sits on without
        # doing arithmetic. The number is set at the FOOT of the line, below the last rule, which
        # is the one strip of every panel that no marker occupies.
        ax.axvline(base, color=TRGREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(base, len(TRANSFER_ROWS) + 0.15, "%.1f" % base, fontsize=7.0,
                color=TRGREY, ha="center", va="center")
        for y, (lab, _, kindm) in enumerate(TRANSFER_ROWS):
            if y in TRANSFER_SEPS:
                ax.axhline(y - 0.5, color=TRINK, lw=0.35, alpha=0.20, zorder=0)
            v = byrow[lab][bm].get(sc)
            if v is None:
                ax.text(x0 + 0.03 * (x1 - x0), y, "n/a", fontsize=7.0, color=TRGREY,
                        ha="left", va="center")
                continue
            if v == "@SAMEFIXED":
                ax.text(byrow[cfglab["fixed"]][bm][sc], y, "=", fontsize=7.3, color=TRGREY,
                        ha="center", va="center")
                continue
            st = MARKSTYLE[kindm]
            ax.plot([v], [y], marker=st["marker"], ms=st["ms"], color=st["color"],
                    markerfacecolor=st["mfc"], markeredgewidth=st["mew"], ls="none", zorder=3)
            if kindm in ("cfg", "method") and cfglab[best[bm][sc]] == lab:
                ax.plot([v], [y], marker="o", ms=st["ms"] + 3.2, color=TRBLUE,
                        markerfacecolor="none", markeredgewidth=0.7, ls="none", zorder=4)
                log.append((bm, sc, lab, v))
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2, pad=1.5)
        ax.set_title(sc, pad=2.5)
    axg[0].set_yticks(range(len(TRANSFER_ROWS)))
    axg[0].set_yticklabels([n for _, n, _ in TRANSFER_ROWS])
    # The benchmark, its metric, its unit and its n, once per group and never in a panel title.
    fig.canvas.draw()
    for group, txt in (((axg[0], axg[2]), "BFCL v4 multi-turn: pass rate (%), n = 800"),
                       ((axg[4], axg[6]), "NESTFUL: win rate (%), n = 1,861")):
        p0 = group[0].get_position()
        p1 = group[1].get_position()
        fig.text(0.5 * (p0.x0 + p1.x1), p0.y0 - 0.118, txt, ha="center", va="top", fontsize=7.3)

    handles = [
        Line2D([], [], ls="none", label="BRACE (fixed-\u03c1)", **_h(MARKSTYLE["method"])),
        Line2D([], [], ls="none", label="other configurations", **_h(MARKSTYLE["cfg"])),
        Line2D([], [], ls="none", marker="o", ms=6.6, color=TRBLUE, markerfacecolor="none",
               markeredgewidth=0.7, label="column best (selected)"),
        Line2D([], [], ls="none", label="uniform GRPO (control)", **_h(MARKSTYLE["control"])),
        Line2D([], [], ls="none", label="published baselines", **_h(MARKSTYLE["pub"])),
        Line2D([], [], ls="none", label="run as published", **_h(MARKSTYLE["faith"])),
    ]
    # SIX MARKER ENTRIES ONLY. The two in-panel conventions are glyphs and not markers, so they
    # are said in words on the line below rather than given a legend handle that would have to
    # draw the glyph twice at two sizes.
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, -0.232), handlelength=0.8, handletextpad=0.4,
                     columnspacing=1.6, labelspacing=0.30)
    fig.text(0.5, -0.256, "\u201c=\u201d the same configuration as fixed-\u03c1 at that scale"
             "\u2003\u2003\u201cn/a\u201d not measured, or not measurable",
             ha="center", va="top", fontsize=7.0, color=TRGREY)
    fig.savefig(os.path.join(outdir, "transferfig.pdf"))
    plt.close(fig)
    print("[transferfig] wrote %s/transferfig.pdf" % outdir)
    for bm, sc, lab, v in log:
        print("   column best  %-5s %-3s %-32s %.2f"
              % (bm, sc, lab.replace("\\quad ", ""), v))


def _h(st):
    return dict(marker=st["marker"], ms=st["ms"], color=st["color"],
                markerfacecolor=st["mfc"], markeredgewidth=st["mew"])


# ==============================================================================================
# fig:transfer, SECOND DESIGN -- SIGNED BARS AGAINST THE UNTRAINED BASE POLICY
# ==============================================================================================
# PI decision 2026-08-29 17:30. The float's job is one comparison: how far each allocation rule
# moves a checkpoint OFF the training distribution, relative to the policy nobody trained. The
# dot plot made the reader measure that distance by eye against a dashed line whose position
# changed from panel to panel. Here it IS the drawn quantity: every bar is (cell - base) in
# percentage points, signed, from a shared zero rule, so "above the base" and "below the base"
# are directions rather than arithmetic.
#
# THE SAME GUARANTEE AS BEFORE, AND IT IS THE REASON THIS FILE MAY NOT COMPUTE A RATE. The cells
# come from transferfig.json, which paper_numbers.emit_tables writes at the moment it formats
# each cell into transfermain.tex, together with the sha256 of the exact table bytes. _transfer_data
# re-hashes the table and REFUSES TO DRAW on a mismatch. The subtraction below is the only
# arithmetic this emitter does, and its subtrahend is the base cell that same capture recorded --
# which is the number the table's own Delta columns are computed from, so a bar here and a Delta
# there are the same quantity to the last bit.
#
# SIX BARS, A FIXED SET, THE SAME SIX IN EVERY GROUP. Not "the best other baseline at each scale":
# that would be a per-panel maximum over rows, which is the selection this paper refuses. They are
# looked up BY tab:transfermain's own row labels, so a row renamed there fails here loudly.
#   1 \methodname{} (fixed-rho)      trBlue, filled     the row every paired transfer contrast uses
#   2 \methodname{} (column best)    trBlue, hatched    A SELECTION, and the legend says so
#   3 uniform GRPO                   trGrey             the control
#   4 PLR-style replay               trGreen, filled    the strongest published rule
#   5 TRACE run as published         trGreen, tint
#   6 VIP run as published           trGreen, outline
# The three published entries share one hue and descend in emphasis, so the block reads as a block
# and the eye still separates its members; ours share one hue and are separated by fill, which is
# the same convention Figure 1 uses.
#
# INDEPENDENT y PER PANEL, DELIBERATELY. The two panels are different benchmarks with different
# metrics (pass rate, win rate) and different spreads: BFCL spans about 22pp, NESTFUL about 34pp
# because two faithful-VIP cells sit more than 22pp below the base policy. Forcing one scale would
# print the BFCL block at a third of its height to accommodate two bars in the other panel. Each
# panel therefore carries its own tick labels and its own axis label, which is what tells a reader
# the scales differ; nothing is compared ACROSS the two panels anywhere in the paper.
#
# NO NUMERIC LABELS ON BARS, per the ask and for the same reason the dot plot carried none: the
# digits are Table~\ref{tab:transfermain}'s job and a bar chart that prints its own values is a
# table with extra ink. A cell that is missing or not measurable draws NO BAR and a small "n/a"
# at the zero line, so an absence reads as an absence and never as a zero effect.
TRGREEN_TINT = "#73AB9C"   # trGreen blended 45% into white; still >= 3:1 against the page

# (row label in transferfig.json, legend text, facecolor, edgecolor, hatch)
# 2026-08-31 (PI 14:45): ONE BRACE SERIES. This figure drew two of them, the full method and the
# per-column best configuration, and by construction the second is never below the first -- it is
# a maximum over a set that contains it. Two bars of one colour where one is always at least the
# other is not a comparison, it is the same claim printed twice, and a reader has to work out
# which is which before reading anything else. The full method's cells are not withdrawn from the
# paper: they are a row of Table 2 and of tab:transfermain, where the selection can be checked
# against everything it ranged over. What the figure now draws is the selected configuration,
# solid, with the selection disclosed in the caption exactly as before.
# 2026-09-01 (PI 18:40): the legend text is "BRACE", not "BRACE (best configuration, selected)".
# The KEY into transferfig.json is unchanged, so the bar still plots the per-column best
# configuration's cells and the guard above still checks that tab:transfermain carries that row;
# what changed is the six words a reader sees. The selection is disclosed where a reader can act
# on it -- the caption, Section 7.3's transfer paragraph and Table~\ref{tab:transfermain} -- and
# not inside a legend entry that has to be parsed before any bar can be read.
BARS = [
    ("\\quad \\methodname{} (column best)\\textsuperscript{\\S}",
     "BRACE", TRBLUE, TRBLUE, None),
    ("\\quad uniform \\textsc{Grpo}", "uniform GRPO (control)", TRGREY, TRGREY, None),
    ("\\quad \\textsc{Plr}", "PLR", TRGREEN, TRGREEN, None),
    ("\\quad \\textsc{Trace}", "TRACE", TRGREEN_TINT,
     TRGREEN_TINT, None),
    ("\\quad \\textsc{Vip}", "VIP", "white", TRGREEN, None),
]
# 2026-08-29 (PI directive, 19:05): the six bars are named as their own papers name them, keys
# and legend alike -- see the note under SERIES. Every acronym is SHORTER than the descriptor it
# replaces ("PLR" for "PLR-style replay", "TRACE"/"VIP" for "TRACE/VIP, run as published"), so
# the saved figure narrows rather than widens and nothing is driven under the 7pt floor. The two
# \methodname{} entries keep their parenthetical because those name WHICH CONFIGURATION a bar is,
# which is a fact about our own rows and not a descriptor of somebody else's method. The
# "run as published, without our warm bank" disclosure is in tab:transfermain's block header and
# in its note.
PANELS = [("bfcl", "\\textsc{Bfcl} v4 multi-turn", "BFCL v4 multi-turn: pass rate, $n=800$"),
          ("nest", "\\textsc{Nestful}", "NESTFUL: win rate, $n=1{,}861$")]


def emit_transfer_figure(outdir):
    """fig:transfer: two panels, six signed bars per scale group, all against the base policy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = _transfer_data(outdir)
    scales = d["scales"]
    byrow = {r["label"]: r for r in d["rows"]}
    for lab, _, _, _, _ in BARS:
        if lab not in byrow:
            raise SystemExit("fig:transfer: tab:transfermain no longer carries the row %r -- the "
                             "figure and the table must name the same rules" % lab)

    plt.rcParams.update({
        # Set for the PAGE. The saved width is trimmed to the content by the tight bbox, so the
        # placement scale is \textwidth (397.5pt) over the saved width; figsize below keeps that
        # within a couple of percent of 1.0 and NOTHING here prints below 7pt on the page.
        "font.size": 7.4, "axes.titlesize": 7.4, "axes.labelsize": 7.4,
        "axes.titleweight": "normal", "legend.fontsize": 7.4,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "figure.dpi": 200, "savefig.bbox": "tight",
        # A y GRID AND NOT A DECORATION. Both panels carry bars an order of magnitude apart in
        # size (NESTFUL runs from -23.5 to +10.2), so the small ones are read against the tick
        # they sit under rather than against their own height. Set behind the bars and faint
        # enough that it never competes with the zero rule, which is the figure's one datum line.
        "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.20,
        "grid.linewidth": 0.4, "grid.color": TRGREY,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
        "axes.edgecolor": TRINK, "xtick.color": TRINK, "ytick.color": TRINK,
        "text.color": TRINK, "axes.labelcolor": TRINK, "hatch.linewidth": 0.55,
    })
    fig, axs = plt.subplots(1, 2, figsize=(6.30, 1.14))
    # THE BOTTOM BAND IS RESERVED EXPLICITLY, for the reason Figure 3's emitter records: in a
    # short figure the tick labels fall below the figure's own bottom edge, into the negative
    # coordinates where fig.legend places the legend, and the legend then prints on top of the
    # scale labels. Caught by rendering the page, not by any measurement of the file.
    fig.subplots_adjust(bottom=0.20, top=0.88, wspace=0.26)
    W = 0.132                       # six bars and a group gap inside a unit pitch
    log = []
    for ax, (bm, _, title) in zip(axs, PANELS):
        base = d["base"][bm]
        vals = []
        for gi, sc in enumerate(scales):
            for bi, (lab, _, fc, ec, ht) in enumerate(BARS):
                v = byrow[lab][bm].get(sc)
                x = gi + (bi - 2.5) * W
                if not isinstance(v, (int, float)):
                    # A GAP AND A MARK, never a zero-height bar: "not measured" and "no effect"
                    # are different statements and this figure may not confuse them.
                    ax.text(x, 0.0, "n/a", fontsize=6.2, color=TRGREY, rotation=90,
                            ha="center", va="bottom")
                    continue
                dv = v - base[sc]
                vals.append(dv)
                ax.bar(x, dv, width=W * 0.90, facecolor=fc, edgecolor=ec, hatch=ht,
                       linewidth=0.5, zorder=3)
                log.append((bm, sc, lab, v, dv))
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = 0.10 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.axhline(0.0, color=TRINK, lw=0.8, zorder=2)
        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels(scales)
        ax.set_xlim(-0.5, len(scales) - 0.5)
        ax.tick_params(axis="x", length=0, pad=2.0)
        ax.tick_params(axis="y", length=2, pad=1.5)
        ax.set_title(title, pad=3.0)
        ax.set_ylabel("$\\Delta$ vs untrained base (pp)", labelpad=1.5)
        # The zero rule IS the base policy, so the left spine may not also run through the bars.
        ax.spines["bottom"].set_visible(False)

    handles = [Patch(facecolor=fc, edgecolor=ec, hatch=ht, linewidth=0.5, label=nm)
               for _, nm, fc, ec, ht in BARS]
    # Kept NARROWER THAN THE AXES on purpose: with a tight bbox the saved width is the wider of
    # the two, and a legend that sets it would shrink the whole float on the page and print every
    # label below the 7pt floor this venue pass fixed.
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, -0.44), handlelength=1.25, handleheight=0.85,
                     handletextpad=0.40, columnspacing=1.1, labelspacing=0.30)
    for h in leg.get_patches():
        h.set_linewidth(0.5)
    fig.savefig(os.path.join(outdir, "transferfig.pdf"))
    plt.close(fig)
    print("[transferfig] wrote %s/transferfig.pdf  (two panels, signed bars)" % outdir)
    for bm, sc, lab, v, dv in log:
        print("   %-5s %-3s %-40s cell %7.2f  delta %+7.2f"
              % (bm, sc, lab.replace("\\quad ", ""), v, dv))
