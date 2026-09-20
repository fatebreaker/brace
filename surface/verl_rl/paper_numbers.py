#!/usr/bin/env python
"""Every number the paper's TRIAGE half quotes that is NOT already printed by main_table.py,
triage_report.py or val_curve.py, regenerated from banked records in one pass.

Sections, in the order the paper uses them:

  DIAG   difficulty bimodality, pooled degenerate-group fraction, wasted budget
  CERT   the certified-exclusion set: budget, live groups, false positives
  RHO    within-group correlation: observed group histogram vs the independent-Bernoulli one
  CALIB  each arm's own recorded per-cycle degeneracy prediction against what the next eight
         optimizer steps measured -- iid model (v2 arms) vs Beta-Binomial group model (v3 arms)
  SEED   seed-level comparison of the method against uniform GRPO on the matched window
  H2H    per-step paired head-to-heads used in the results text
  EST    the estimator gap that separates the posterior weight from TRACE's plug-in

Nothing here touches a GPU and nothing here writes outside stdout.

    paper_numbers.py                # everything
    paper_numbers.py --only CALIB   # one section
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict, Counter

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BSZ, NROLL, STEPS_PER_CYCLE = 32, 5, 8
WINDOW = [15, 20, 25, 30]
FULL_N = 590   # paired cells reach this when complete; below ~560 a cell is still filling

# THE METHOD'S ROW LABEL, EXPORTED SO NOBODY STRING-MATCHES ITS NAME AGAIN.
# Every registry below writes the method's rows as this exact token, and the token expands to the
# printed name in ONE place, tex/head.tex's \methodname (PI 2026-08-29: TRIAGE -> BRACE = Bank-
# calibrated Rollout Allocation under Correlated Episodes). Consumers that need to pick the method
# row out of a registry -- work/analysis/w14_reconcile.py does, in three places -- must test
# against METHOD_TEX and not against a name: those three tests read `"Triage" in lab` and every one
# of them broke silently at the rename, which is why this constant exists.
METHOD_TEX = "\\methodname{}"

sys.path.insert(0, f"{R}/surface/verl_rl")
from main_table import (load_cell, mcnemar, arm_models, model_of,   # noqa: E402
                        BASE_CELLS, ABSORBED, cell_path, SUPERVISOR)
                                           # same loaders, statistics, model registry AND
                                           # continuation registry as the main table -- one
                                           # definition of the family, not two


# ---------------------------------------------------------------------------------------------
def train_rows(tag):
    """Training rollouts of one arm, in generation order, val dropped, `.arm_since` honoured.

    A MARKER AT OR PAST EOF IS NOT A SEGMENT BOUNDARY, IT IS A DESTROYED ONE (fixed 2026-08-19).
    `.arm_since` records the line at which THIS arm's segment of a shared episode log begins, so a
    marker equal to the file's own line count selects the empty segment and the arm vanishes from
    every statistic computed here -- silently, because a row with zero groups is skipped rather
    than reported. That is what had happened to `run_a8Tvip/.arm_since = 4336`, which equals that
    log's line count exactly: VIP was dropped from this file's mechanism table and, by the same
    marker through a different route, from the 8B correlation refit. Both were found in
    work/analysis/reviewer_response_analyses.md (its sections 4.2 and 5.5) and both are this one
    defect. An arm whose log exists but whose marker consumes all of it is read WHOLE, with the
    substitution announced on stderr rather than absorbed: a segment boundary that cannot be
    honoured is a fact about the run directory that a reader of the table is entitled to.
    Markers that select a genuine sub-segment are untouched (`run_dapo` at 73,678 of a still-
    growing file, `run_q2bN` at 13,622 of 14,509), and `dapo` is frozen in FROZEN_MECH anyway.
    """
    p = f"{R}/work/verl/run_{tag}/episodes.jsonl"
    try:
        since = int(open(f"{R}/work/verl/run_{tag}/.arm_since").read().strip() or 0)
    except Exception:
        since = 0
    if since and os.path.exists(p):
        with open(p, errors="ignore") as fh:
            n_lines = sum(1 for _ in fh)
        if since >= n_lines:
            print("[paper_numbers] %s: .arm_since=%d >= %d lines -- marker at EOF, so it is a "
                  "destroyed boundary rather than a segment; reading the whole log."
                  % (tag, since, n_lines), file=sys.stderr)
            since = 0
    return _read(p, since)


def _read(p, since=0):
    rows = []
    if not os.path.exists(p):
        return rows
    for i, line in enumerate(open(p, errors="ignore")):
        if i < since:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("split") == "val" or "scenario" not in d or "reward" not in d:
            continue
        rows.append(((d["scenario"], d.get("task_idx")), 1 if float(d["reward"]) > 0 else 0))
    return rows


def groups_of(rows):
    """Reconstruct GRPO groups by chunking the log into optimizer steps, as triage_report does."""
    per_step = BSZ * NROLL
    out = []
    for s in range(len(rows) // per_step):
        g = defaultdict(list)
        for k, r in rows[s * per_step:(s + 1) * per_step]:
            g[k].append(r)
        out.append([(k, v) for k, v in g.items() if len(v) >= 2])
    return out


def pool_max():
    p = json.load(open(f"{R}/surface/gate_caller/pools/pool_max.json"))
    tasks = p if isinstance(p, list) else p.get("tasks", [])
    return {(t.get("scenario"), t.get("task_idx")) for t in tasks}


def beta_moment(a, b, k):
    """E[p^k] for p ~ Beta(a,b), the closed form the allocator uses."""
    r = 1.0
    for j in range(k):
        r *= (a + j) / (a + b + j)
    return r


# ---------------------------------------------------------------------------------------------
def sec_diag_cert_rho():
    solve = defaultdict(lambda: [0, 0])
    tot_g = tot_d = 0
    ex = [json.loads(l) for l in open(f"{R}/work/analysis/excluded_tasks.jsonl")]
    EX = {(r["scenario"], r["task_idx"]): r for r in ex}
    EXPM = {k for k, r in EX.items() if r.get("in_pool_max")}
    UNSAT = {k for k, r in EX.items() if r["reason"] == "verifier-unsatisfiable" and r.get("in_pool_max")}
    PRES = {k for k, r in EX.items() if r["reason"] == "pre-solved" and r.get("in_pool_max")}
    PM = pool_max()
    KEPT = PM - EXPM
    bucket = defaultdict(lambda: [0, 0])
    n_runs = 0

    for p in sorted(glob.glob(f"{R}/work/verl/run_*/episodes.jsonl")):
        rows = _read(p)
        if not rows:
            continue
        n_runs += 1
        for k, r in rows:
            solve[k][0] += r
            solve[k][1] += 1
        for step in groups_of(rows):
            for k, v in step:
                tot_g += 1
                deg = 1 if len(set(v)) == 1 else 0
                tot_d += deg
                for name, S in (("verifier-unsat", UNSAT), ("pre-solved", PRES), ("kept pool", KEPT)):
                    if k in S:
                        bucket[name][0] += 1
                        bucket[name][1] += deg

    n = len(solve)
    roll = sum(v[1] for v in solve.values())
    never = [k for k, v in solve.items() if v[0] == 0]
    always = [k for k, v in solve.items() if v[0] == v[1]]
    mid = [k for k, v in solve.items() if 0 < v[0] < v[1]]
    band = [k for k in mid if 0.2 <= solve[k][0] / solve[k][1] <= 0.8]
    r_never = sum(solve[k][1] for k in never)
    r_always = sum(solve[k][1] for k in always)

    print("== DIAG  (all banked training rollouts, every arm ever run) ==")
    print(f"  runs {n_runs}   distinct tasks {n}   rollouts {roll}   groups {tot_g}")
    print(f"  degenerate group fraction, pooled          {tot_d/tot_g:.4f}")
    print(f"  never solved   {len(never):5d} tasks {100*len(never)/n:5.1f}%   "
          f"{r_never:7d} rollouts {100*r_never/roll:5.1f}% of all generation")
    print(f"  always solved  {len(always):5d} tasks {100*len(always)/n:5.1f}%   "
          f"{r_always:7d} rollouts {100*r_always/roll:5.1f}%")
    print(f"  0<p<1          {len(mid):5d} tasks {100*len(mid)/n:5.1f}%     "
          f"of which 0.2<=p<=0.8: {len(band)} = {100*len(band)/n:.1f}%")
    print(f"  budget on tasks that never produced a gradient (never or always solved): "
          f"{100*(r_never+r_always)/roll:.1f}%")
    sub = {k: v for k, v in solve.items() if k in PM}
    print(f"  pool_max {len(PM)} tasks: never solved {sum(1 for v in sub.values() if v[0]==0)}, "
          f"always solved {sum(1 for v in sub.values() if v[0]==v[1])}")

    print("\n== CERT  (certified exclusion) ==")
    print(f"  file {os.path.relpath(f'{R}/work/analysis/excluded_tasks.jsonl', R)}: {len(ex)} tasks, "
          f"{len(EXPM)} in pool_max ({len(UNSAT)} verifier-unsatisfiable, {len(PRES)} pre-solved)")
    print(f"  sub-rules: {dict(Counter(r['subrule'] for r in ex))}")
    ex_roll = sum(solve[k][1] for k in EX if k in solve)
    pm_roll = sum(solve[k][1] for k in PM if k in solve)
    pm_ex = sum(solve[k][1] for k in EXPM if k in solve)
    print(f"  rollouts spent on them: {ex_roll} = {100*ex_roll/roll:.2f}% of all generation; "
          f"{pm_ex} = {100*pm_ex/pm_roll:.2f}% of pool_max generation")
    for name in ("verifier-unsat", "pre-solved", "kept pool"):
        g, d = bucket[name]
        print(f"  {name:<15} groups {g:>6}  degenerate {d/g:.4f}  live groups {g-d}")
    print(f"  verifier-unsatisfiable: {sum(solve[k][0] for k in UNSAT)} solves in "
          f"{sum(solve[k][1] for k in UNSAT)} rollouts, ever")
    print(f"  pre-solved mean solve rate {sum(solve[k][0] for k in PRES)/sum(solve[k][1] for k in PRES):.4f}")
    ever = {k for k, v in solve.items() if v[0] > 0}
    prov = {(r['scenario'], r['task_idx']) for r in ex
            if r['subrule'] in ("init-only-guard-fires", "complete-before-action")}
    print(f"  provable sub-rules fire on {len(prov)} tasks and on "
          f"{len(prov & ever)} of the {len(ever)} ever-solved tasks")

    print("\n== RHO  (within-group correlation, a8T2 cycles 1-2, k=5) ==")
    rows = train_rows("a8T2")
    obs, gs = Counter(), []
    for step in groups_of(rows)[:2 * STEPS_PER_CYCLE]:
        for k, v in step:
            if len(v) == NROLL:
                obs[sum(v)] += 1
                gs.append(sum(v))
    ng = len(gs)
    pbar = sum(gs) / (ng * NROLL)
    binom = [math.comb(NROLL, i) * pbar ** i * (1 - pbar) ** (NROLL - i) * ng for i in range(NROLL + 1)]
    ssq = sum((s / NROLL - pbar) ** 2 for s in gs) / ng
    rho = (ssq - pbar * (1 - pbar) / NROLL) / (pbar * (1 - pbar) * (1 - 1.0 / NROLL))
    print(f"  {ng} complete groups, batch mean solve rate {pbar:.4f}")
    print(f"  observed n_solved histogram      {[obs[i] for i in range(NROLL+1)]}")
    print(f"  independent Bernoulli at same p  {[round(x) for x in binom]}")
    print(f"  all-agree: observed {100*(obs[0]+obs[NROLL])/ng:.1f}%  vs "
          f"independent {100*(binom[0]+binom[NROLL])/ng:.1f}%")
    print(f"  pooled-mean moment estimate of the intraclass correlation rho = {rho:.4f}")
    print("  (the allocator runs rho=0.78, nu=(1-rho)/rho=0.285, fitted against the per-task")
    print("   posterior means over the four warm arms' first two cycles: triage.py header, sec. 4)")


IID_ARMS = ["a8T2", "a8T2r", "a8T2g1", "a8T2w"]
GRP_ARMS = ["a8T3g", "a8T3gr", "a8T3g2", "a8T5k", "a8Tpe"]


def sec_calib():
    print("== CALIB  (predicted vs measured degenerate fraction, per cycle) ==")
    res = {}
    for fam, tags in (("independent Bernoulli (v2 arms)", IID_ARMS),
                      ("Beta-Binomial group model (v3 arms)", GRP_ARMS)):
        errs = []
        print(f"  -- {fam}")
        for tag in tags:
            sp = f"{R}/work/coadapt_{tag}/triage_stats.jsonl"
            if not os.path.exists(sp):
                continue
            steps = groups_of(train_rows(tag))
            for line in open(sp):
                rec = json.loads(line)
                cyc, pred = rec["cycle"], rec.get("pred_degenerate_frac")
                lo, hi = (cyc - 1) * STEPS_PER_CYCLE, cyc * STEPS_PER_CYCLE
                chunk = steps[lo:hi]
                if pred is None or not chunk:
                    continue
                g = sum(len(s) for s in chunk)
                d = sum(1 for s in chunk for _, v in s if len(set(v)) == 1)
                meas = d / g
                errs.append(abs(pred - meas))
                print(f"     {tag:<8} cycle {cyc}  predicted {pred:.4f}  measured {meas:.4f}  "
                      f"error {pred-meas:+.4f}")
        res[fam] = (sum(errs) / len(errs), len(errs))
        print(f"     mean |error| {res[fam][0]:.4f} over {res[fam][1]} cycles")
    a = list(res.values())
    print(f"  the group model closes {100*(1-a[1][0]/a[0][0]):.1f}% of the independent model's bias")
    _calib_trivial()


def _calib_trivial():
    """The cheapest predictor that could work, on fig:calibration's own 83 arm-cycles.

    Review of 2026-09-02, P0-7: Figure~\ref{fig:calibration} compares the correlated model against
    the independent one, which the paper has already shown is wrong by 0.59, and against nothing
    else. The comparison a reader wants is against a predictor that carries no correlation model at
    all -- the running mean of the degenerate fractions this family's earlier cycles realised. It is
    computed here on the SAME rows the figure plots (work/analysis/calibration_matched.jsonl, the
    file calibration_matched.py writes beside the figure) and under the SAME error metric this
    section uses everywhere else, mean |predicted - measured| over cycles.

    A family's first cycle has no earlier cycle and so has no trivial prediction; it is dropped,
    and the calibrated model is re-scored on exactly the rows that survive so the two numbers are
    paired rather than taken over different sets.
    """
    fp = f"{R}/work/analysis/calibration_matched.jsonl"
    if not os.path.exists(fp):
        print("  (no calibration_matched.jsonl: trivial-predictor baseline not computed)")
        return
    rows = [json.loads(l) for l in open(fp)]
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    triv, cal, iid = [], [], []
    for f in sorted(fams):
        hist = []
        for r in sorted(fams[f], key=lambda x: (x["arm"], x["cycle"])):
            if hist:
                triv.append(abs(sum(hist) / len(hist) - r["realised"]))
                cal.append(abs(r["pred_cal"] - r["realised"]))
                iid.append(abs(r["pred_iid"] - r["realised"]))
            hist.append(r["realised"])
    n = len(triv)
    mt, mc = sum(triv) / n, sum(cal) / n
    d = [x - y for x, y in zip(triv, cal)]
    mu = sum(d) / n
    se = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1) / n)
    print("  -- trivial predictor (running mean of the family's earlier realised fractions) --")
    print(f"     over the same {n} of {len(rows)} arm-cycles: trivial {mt:.4f}, "
          f"correlated {mc:.4f}, independent {sum(iid)/n:.4f}")
    print(f"     paired difference trivial - correlated {mu:+.4f} +/- {se:.4f}; "
          f"the correlated model is closer on {sum(1 for x in d if x > 0)} of {n} cycles")


SEEDS = {"TRIAGE (v3g)": ["a8T3g", "a8T3gr", "a8T3g2"],
         "uniform GRPO": ["a8F", "a8Fr"],
         "sharpened (v5k)": ["a8T5k", "a8T5kr"]}


def arm_window(arm, base):
    """(per-step effect dict, mean over the steps that reach 300 paired tasks)."""
    per = {}
    for st in WINDOW:
        cur = load_cell(f"{R}/work/coadapt_eval_{arm}/cell_STEP{st}.jsonl")
        keys = sorted(set(cur) & set(base))
        if len(keys) < 300:
            continue
        per[st] = (sum(cur[k] for k in keys) - sum(base[k] for k in keys)) / len(keys)
    return per, (sum(per.values()) / len(per) if per else None)


def sec_seed():
    base = load_cell(f"{R}/work/coadapt_eval_coadapt/cell_A.jsonl")
    print(f"== SEED  (matched window {WINDOW}, base anchor n={len(base)}) ==")
    means = {}
    for name, arms in SEEDS.items():
        print(f"  -- {name}")
        ms = []
        for arm in arms:
            per, m = arm_window(arm, base)
            if m is None:
                continue
            ms.append(m)
            print(f"     {arm:<8} window {m:+.4f}   " +
                  "  ".join(f"{s}:{d:+.4f}" for s, d in sorted(per.items())))
        means[name] = ms
        mu = sum(ms) / len(ms)
        sd = math.sqrt(sum((x - mu) ** 2 for x in ms) / (len(ms) - 1)) if len(ms) > 1 else float("nan")
        print(f"     seed mean {mu:+.4f}  sd {sd:.4f}  n_seeds {len(ms)}")
    a, b = means["TRIAGE (v3g)"], means["uniform GRPO"]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    print(f"  method minus uniform, seed as the unit: {ma-mb:+.4f}, se {se:.4f}, "
          f"t {(ma-mb)/se:.2f} on {len(a)}v{len(b)} seeds -- NOT resolvable at this seed count")
    print("  per-step seed means:")
    for st in WINDOW:
        line = []
        for name, arms in SEEDS.items():
            ds = []
            for arm in arms:
                cur = load_cell(f"{R}/work/coadapt_eval_{arm}/cell_STEP{st}.jsonl")
                keys = sorted(set(cur) & set(base))
                if len(keys) >= 300:
                    ds.append((sum(cur[k] for k in keys) - sum(base[k] for k in keys)) / len(keys))
            if ds:
                line.append(f"{name} {sum(ds)/len(ds):+.4f} (n={len(ds)})")
        print(f"     step {st:>2}: " + "   ".join(line))


def sec_h2h():
    print("== H2H  (arm vs arm, paired, same held-out records) ==")
    # 2026-08-24: b8plrr added. The prose sentence this section feeds names PLR as "the
    # strongest published method with a cell at every matched step", and PLR is a two-seed
    # row from this pass, so the head-to-head has to be readable against BOTH of its seeds
    # rather than against the one that happens to be weaker.
    for a, b in (("a8T3g2", "a8F"), ("a8T3g2", "b8plr"), ("a8T3g2", "b8plrr"),
                 ("a8T3g", "b8plr"),
                 ("a8T3gr", "b8plr"), ("a8T5k", "b8plr"), ("a8F", "a8Fr")):
        ds = []
        out = []
        for st in WINDOW:
            A = load_cell(f"{R}/work/coadapt_eval_{a}/cell_STEP{st}.jsonl")
            B = load_cell(f"{R}/work/coadapt_eval_{b}/cell_STEP{st}.jsonl")
            keys = sorted(set(A) & set(B))
            if len(keys) < 300:
                continue
            bb = sum(1 for k in keys if A[k] and not B[k])
            cc = sum(1 for k in keys if B[k] and not A[k])
            d = (sum(A[k] for k in keys) - sum(B[k] for k in keys)) / len(keys)
            ds.append(d)
            out.append(f"{st}:{d:+.4f}(p={mcnemar(bb,cc):.3f})")
        if ds:
            print(f"  {a:<8} vs {b:<8} mean {sum(ds)/len(ds):+.4f}   " + " ".join(out))


def sec_est():
    print("== EST  (posterior weight vs TRACE-style plug-in, k=5) ==")
    k = NROLL
    for lab, (a, b) in (("unexplored task, Beta(1,1)", (1, 1)),
                        ("p=1/2 measured on 10 rollouts", (6, 6)),
                        ("p=1/2 measured on 20 rollouts", (11, 11)),
                        ("20 failures, no solve", (1, 21))):
        post = 1 - beta_moment(a, b, k) - beta_moment(b, a, k)
        m = a / (a + b)
        plug = 1 - m ** k - (1 - m) ** k
        print(f"  {lab:<32} posterior E[g] {post:.4f}   plug-in g(p_hat) {plug:.4f}")
    print("  the plug-in cannot separate an unexplored task from a measured coin flip: both 0.9375")
    ok = True
    import numpy as np
    ps = np.linspace(1e-9, 0.5, 200001)
    for kk in range(2, 17):
        g = 1 - ps ** kk - (1 - ps) ** kk
        ok &= bool((np.diff(g) > 0).all())
    print(f"  g(p,k) strictly increasing in p on (0,1/2] for k=2..16: {ok}  "
          "-> strictly increasing in u=p(1-p), so g and VIP's p(1-p) induce the SAME ranking")


# =============================================================================================
# The paper's three TRIAGE tables, written from the same records rather than transcribed.
# =============================================================================================

def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, run = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


# =============================================================================================
# HOLM FAMILY MEMBERSHIP PIN (added 2026-08-18, second presentation pass).
#
# full_family() discovers its members by globbing work/coadapt_eval_*. That is the right
# membership RULE -- every arm of this model with a scorable cell against this model's anchor, no
# arm ever removed -- but it is not a stable SET, because the fleet keeps training. An arm that
# banks its first cell after a table has been typeset silently raises m for every arm already in
# that table, so a regeneration that changes nothing about the paper still moves published q
# values. That is exactly what happened between the 2026-08-18 fold-in and this pass: the 2B
# family grew 15 -> 17 as q2bTk8 and q2bTnw2 each banked ONE cell of the four-step window, which
# alone moved q2bT's published q from 0.705 to 0.752.
#
# The pin below freezes each model's family to the membership the shipped tables were computed
# from. It is a presentation pin and not a statistical one: the two excluded arms have one cell
# each of a four-cell window, appear in no table, and are recorded as out of scope in
# CHANGES.md's 2026-08-18 entry (section 4) and PLAN_TRIAGE.md; the 4B and 8B lists are the live
# sets unchanged and are pinned only so the same drift cannot happen to them mid-revision.
#
# HOW TO CHANGE IT: when an arm's window is complete and the paper is ready to absorb the
# multiplicity, add its tag here and re-emit every table in one pass, narrating the q movement in
# CHANGES.md. Never drop an arm from this list -- the membership rule is that no arm leaves.
#
# ---------------------------------------------------------------------------------------------
# GROWN 2026-08-22, FINAL REGENERATION. This is the deliberate, narrated growth the paragraph
# above prescribes, not drift: six arms whose four-step windows completed since the last emission
# now have printed rows, so they pay -- and charge -- the multiplicity of the search that produced
# them. Every q value in this pass moved for that reason and the movement is tabulated in
# CHANGES.md (2026-08-22, section "What the family growth moved").
#   2B  15 -> 19   + q2bV (VIP), q2bLp (TSCL)          -- new rows of the scale-blocked tab:main
#                  + q2bF3e5 (uniform at LR 3e-5)      -- the LR-sweep rung, tab:lrsweep
#                  + q2bTk8 (k=8)                      -- the k-sensitivity rung, tab:ksens
#   4B   6 ->  7   + q4bTnw (no warm bank at 4B)       -- the mechanism-boundary arm
#   8B  36 -> 37   + a8Tvipr (clean VIP seed 2)        -- second seed of the VIP baseline row
#
# GROWN AGAIN 2026-08-24, PRE-FINAL FOLD -- 4B ONLY, 7 -> 10, AND THIS IS A DELIBERATE PIN CHANGE.
# The three arms the 2026-08-22 pass listed as the paper's own open 4B gaps have completed their
# four-step windows, so all three acquire printed rows and all three must therefore pay the
# multiplicity of the search that produced them:
#   4B   7 -> 10   + q4bT2 (TRIAGE seed 2 at 4B)   -- makes tab:main's 4B TRIAGE row a 2-seed row,
#                                                     the same convention the 2B and 8B rows use
#                  + q4bV  (VIP at 4B)             -- replaces a `not run at this scale` row
#                  + q4bLp (TSCL at 4B)            -- replaces a `not run at this scale` row
# 2B AND 8B ARE NOT TOUCHED BY THAT PASS: no arm joined or left either family, so no 2B or 8B q
# moved, and every 2B/8B table re-emitted byte-identical.
#
# GROWN AGAIN 2026-08-24, FINAL FOLD -- 8B ONLY, 37 -> 38. b8plrr, the PLR baseline's second seed
# (offset 1500, the fleet's standard second-seed rung), banked all four window cells at n=590 and
# acquires printed rows in tab:main, tab:mech, the per-seed appendix and the forest figure. This is
# the SAME act as the a8Tvipr admission of 2026-08-22 and is narrated the same way. It is also the
# review's item 2 half that was still open ("one more seed of VIP and of PLR"): VIP was done then,
# PLR is done now.
#   8B  37 -> 38   + b8plrr (PLR-style prompt replay, seed 2)
# 2B AND 4B ARE NOT TOUCHED BY THIS PASS.
#
# EVERY PRE-EXISTING 8B q MOVES BECAUSE m MOVED, AND NONE FALLS. Measured, not asserted; only arms
# whose q actually moved are listed, and the largest movement is 0.0385:
#     arm       q at m=37   ->   q at m=38
#     a8T5        0.8850    ->     0.9235   (largest movement, +0.0385)
#     b8plr       0.8420    ->     0.8771
#     a8T6d       0.5018    ->     0.5219
#     b8van       0.4335    ->     0.4502
#     a8T3g       0.3653    ->     0.3789
#     a8T2r       0.2989    ->     0.3095
#     a8T2nr      0.2712    ->     0.2803
#     a8T3gr      0.2712    ->     0.2803
#     a8A         0.1031    ->     0.1065
#     a8T2n       0.0673    ->     0.0694
#     a8Tpe       0.0271    ->     0.0279
#     a8T5k       0.0020    ->     0.0021
#     a8F         0.0015    ->     0.0015
#     a8T3g2    < 0.0005    ->   < 0.0005
# NO VERDICT CHANGES: the same five arms clear 0.05 (a8F, a8T3g2, a8T4, a8T5k, a8Tpe) and the same
# thirty-two do not. b8plrr itself enters at window +2.63 pp, raw p = 0.0436, q = 1.000.
# WHAT THE ARM COSTS US, PRINTED BECAUSE IT RUNS AGAINST US: on the transfer benchmark b8plrr
# scores 38.625 -- a numerical tie with the method's best seed (a8T3gr, 309/800 each) and the joint
# top row of tab:bfcl -- so PLR's TWO-SEED transfer mean is 38.00 against the method's three-seed
# 37.79. A published baseline's replicated mean is now ABOVE ours on 8B transfer. That is exactly
# the outcome the paper already says it cannot rule out ("what we do not claim, anywhere, is a lead
# over the allocation literature"), and it is printed at full strength rather than absorbed.
#
# EVERY PRE-EXISTING 4B q MOVES, BECAUSE m MOVED, AND THE FULL ENUMERATION IS THE POINT. m rises
# from 7 to 10, so every 4B q rises; none falls. Measured, not asserted (CHANGES.md carries the
# same table):
#     arm      window     p (unchanged)     q at m=7   ->   q at m=10
#     q4bR     +3.60      0.00700           0.0490 *   ->   0.0630
#     q4bTnw   +3.14      0.00904           0.0542     ->   0.0650
#     q4bF     +2.68      0.03570           0.1785     ->   0.2142
#     q4bP     +2.67      0.08953           0.3581     ->   0.4477
#     q4bT     +2.46      0.22952           0.6886     ->   0.9181
#     q4bF2    +1.78      0.52347           1.0000     ->   1.0000
#     q4bD     +0.93      0.83882           1.0000     ->   1.0000
# No arm's EFFECT and no arm's raw p moved -- only the correction. ONE VERDICT CHANGES AND IT
# CHANGES AGAINST A PUBLISHED BASELINE WE HAD STARRED: the retrieval-shaped surface loses its star
# (0.0490 -> 0.0630), and the star it loses is the one the 2026-08-22 draft built a caption title
# and a numbered observation on. The newly admitted q4bV clears at 0.0012, so the 4B block still
# has exactly one starred row -- a different published baseline, for a different reason, and both
# the title and the observation are rewritten to say what is now true. Growing a family is allowed
# to move a star. Suppressing the growth to keep one is not, which is why the enumeration above
# exists rather than a sentence saying nothing important moved.
#
# THREE 8B ARMS HOLD SCORABLE CELLS AND ARE DELIBERATELY NOT IN THIS FAMILY. Recorded here because
# an exclusion that shrinks m moves every published q downward, which is the same failure as the
# growth this pin exists to stop, and it must therefore be visible rather than inferred:
#   * a8Tme, a8Tme2 -- the matched-ESS outcome controls. They are read against their own
#     same-offset TRIAGE parents and against nothing else (the pairing rule registered with them),
#     they have no row in any float, and the paper reports them in Limitations as an unresolved
#     control whose registered ESS confound was recorded before either outcome existed. Held out
#     by PI directive at this pass. The sensitivity is measured, not assumed: admitting both
#     raises m from 37 to 39 and is reported in CHANGES.md with the largest q movement it causes.
#   * one arm whose window is still filling (two of four cells). It has no row, no number and no
#     mention anywhere in the paper; an arm with half a window is not a member of a family whose
#     statistic is the four-step window mean.
# q2bTnw2 (one cell of four, the registered 2B no-warm-bank replication) stays out for the reason
# it always has: one cell is not a window, and it appears in no table. Its tag is named here and
# not in any .tex file, which is the standing rule for an unshipped arm.
FAMILY_PIN = {
    # 20 arms -- grown from 19 on 2026-08-28 by ONE arm:
    #   + q2bTr   the scale-corrected-rho configuration at 2B (nu from the 2B refit). Its four
    #             window cells are banked and scorable; its transfer evals are still running, so it
    #             appears in the window tables and is absent from the transfer ones, which is a
    #             coverage fact stated in those captions rather than a null.
    # 2026-08-29 FOLD: 20 -> 22, by the TWO FAITHFUL PUBLISHED ARMS at this scale, which is what
    # this fold was waiting for. Both are "run as published": the published rule with NONE of our
    # components -- no warm bank, no certified exclusion, no shrinkage.
    #   + q2bVf   FAITHFUL published VIP at 2B, the 2B counterpart of q4bTvip and a8Tvipf. Window
    #             +1.69 (-0.7/+2.4/+3.1/+2.0, n=590 at every step). It does NOT collapse in
    #             distribution at 2B, which is the opposite of the 4B and 8B readings and is
    #             printed as measured.
    #   + t2bTf   FAITHFUL published TRACE at 2B, completing that baseline at all three scales.
    #             Window +1.53 (+0.0/+1.2/+1.9/+3.1). This row replaces the IN_FLIGHT state that
    #             tab:main and fig:stepcurve both printed as "registered, window incomplete".
    # 2026-08-31 FOLD v: 22 -> 25, by THREE arms whose windows completed on 08-30/08-31. Every
    # one is a scorable four-cell window of a 2B arm, which is this family's whole membership
    # rule, so admitting them is not a choice and holding them out would be one:
    #   + q2bTpe  the PLUG-IN ESTIMATOR configuration at 2B, the cell Table 2's plug-in row was
    #             missing. Window +2.03 (+1.5/+2.7/+2.0/+1.9, n=590 at every step; solve 10.7),
    #             between the full method (+2.59) and the uniform control, as at 4B and 8B.
    #   + q2bV2   a PRE-REGISTERED RERUN of the VIP-rule-plus-bank arm at the PARENT'S OWN seed
    #             offset (500, the same one),
    #             run because q2bV's step-30 adapter was pruned off scratch and its NESTFUL cell
    #             could not otherwise exist. Window +1.27 (solve 9.9) against q2bV's +1.95. It
    #             is one seed run twice, NOT a second seed, and tab:reruns says so where it is
    #             printed; it is a family member because it is an arm with a scorable window,
    #             which is the whole membership rule.
    #   + q2bLp2  the same for TSCL at 2B. Window +0.55 (solve 9.2) against q2bLp's +0.68.
    # The two reruns are family members and NOT row members: tab:main and tab:mainsteps keep the
    # parent arms, so no window cell in any table moves. What they supply is the transfer pair
    # their parent cannot (see TRANSFER_RERUN).
    # 2026-09-01 _fold6: 25 -> 26 by ONE arm, and it is admitted under the same membership rule as
    # every other member -- an arm of this model with a scorable four-cell window -- not because it
    # is ours or because it is convenient:
    #   + q2bT3e  the method at LEARNING RATE 3e-5 (q2bT with that token alone changed), the rung
    #             tab:lrsweep's caption previously disclosed as not run. Window +1.44
    #             (+1.36/+2.03/+1.36/+1.02, n=590 at every step) against the 1e-4 parent's +2.59;
    #             transfer BFCL 14.00 (n=800) and NESTFUL 27.51 (n=1861). It carries a printed row
    #             and therefore a claim, so it is pinned rather than read outside the correction.
    # THE GROWTH IS MEASURED AND IT MOVES NOTHING, which is worth stating rather than assuming:
    # every 2B q at m=25 was already 1.0000 -- no 2B arm clears the correction at any m -- so
    # 25 -> 26 leaves all twenty-six at 1.0000 and no verdict, star or printed q anywhere in the
    # 2B block changes. The 2B family is saturated; that is a property of the 2B evidence, not a
    # convenience of this pass, and it is the reason the enumeration here is one line rather than
    # the arm-by-arm q table the 4B and 8B growths needed.
    # 2026-09-01 _fold6, second growth of this pass: 26 -> 27 by ONE arm, admitted under the same
    # membership rule as every other member (an arm of this model with a scorable four-cell
    # window):
    #   + t2bTbk  FAITHFUL PUBLISHED TRACE AT 2B PLUS OUR WARM BANK, and nothing else: t2bTf with
    #             --warm-bank at its default and no other change (diff verified at launch), one
    #             seed at the same offset 500 its parent ran. It is the reviewer's P0 experiment,
    #             the one that separates the bank's contribution from the allocation rule at a
    #             scale where the bank is priced, and it carries a printed row in
    #             tab:ladder2b, so it is pinned rather than read outside the correction.
    # The growth moves nothing, measured rather than assumed: every 2B q was already 1.000 at
    # m=26 and all twenty-seven are 1.000 at m=27, because no 2B arm clears the correction at any
    # family size. The 2B family is saturated.
    # 2026-09-03 _fold7h: 27 -> 28 by ONE arm, admitted under the same membership rule as every
    # other member -- an arm of this model with a scorable four-cell window -- and not because of
    # what it measures, which is a negative:
    #   + q2bTrt  the PER-SCALE-$\rho$ CONFIGURATION AT THE NATURAL SAMPLING TEMPERATURE: q2bTr
    #             with --temp 0.3 -> 1.12 and no other token changed, one seed at the parent's own
    #             offset (500). Window +0.55 (-1.02/-0.17/+1.36/+2.03, n=590 at every step), the
    #             weakest window of the 2B method family, against the parent's +2.67 and the full
    #             recipe's +2.59. It carries a printed row in tab:negatives, so it is pinned
    #             rather than read outside the correction.
    # The growth moves nothing, measured rather than assumed: every 2B q was already 1.000 at
    # m=27 and all twenty-eight are 1.000 at m=28, because no 2B arm clears the correction at any
    # family size. The 2B family is saturated, as it has been since m=25.
    # 2026-09-06 _fold12: 28 -> 29 by ONE arm, the 2B rung of the VARIABLE-k PILOT, admitted under
    # the same membership rule as every other member -- an arm of this model with a scorable
    # four-cell window and a printed row -- and not because of what it measures, which is the worst
    # transfer outcome in this project:
    #   + q2bK2   THE GROUP SIZE FREED UNDER ONE BUDGET: q2bT with VARK=1 and no other token
    #             changed, one seed at the parent's own offset (500), the per-step rollout total
    #             pinned to the uniform k=5 spend (sum k = 1280 = 8 x 160 at every cycle, read off
    #             the run log). Window +1.27 (+1.86/+1.36/+0.85/+1.02, n=590 at every step)
    #             against the fixed-k parent's +2.59; transfer BFCL 0.50 (base 15.12) and NESTFUL
    #             17.89 (base 21.60). The allocator, free to act, moved fewer than 8% of rows off
    #             k=5. It carries a printed row in tab:negatives, so it is pinned rather than read
    #             outside the correction.
    # The growth moves nothing, measured rather than assumed: every 2B q was already 1.000 at
    # m=28 and all twenty-nine are 1.000 at m=29, because no 2B arm clears the correction at any
    # family size. The 2B family is saturated, as it has been since m=25, and q2bK2 itself enters
    # at raw p = 0.522, q = 1.000, so what it costs is m and nothing else.
    "2B": ["q2bD", "q2bF2", "q2bF3e5", "q2bF5", "q2bF6", "q2bF7", "q2bFs", "q2bK2", "q2bLp",
           "q2bLp2",
           "q2bN", "q2bP", "q2bR", "q2bT", "q2bT3", "q2bT3e", "q2bTd", "q2bTiid", "q2bTk8",
           "q2bTnw", "q2bTpe", "q2bTr", "q2bTrt", "q2bTs", "q2bV", "q2bV2", "q2bVf", "t2bTbk",
           "t2bTf"],
    # 15 arms -- grown from 10 on 2026-08-27 (the comprehensive-grid pass) by FIVE measured arms,
    # each enumerated here because growing a family moves every q in it and that must never be a
    # silent side effect:
    #   + q4bV2   the SECOND SEED of the VIP-rule arm. Its absence was a real defect: tab:main
    #             printed VIP's 4B window as one seed (+4.53) while the prose quoted the two-seed
    #             mean (+4.11). One arm, one family slot, and the table now prints what the prose
    #             says. This is the row relabelled "VIP + warm bank" -- both its seeds run OUR warm
    #             bank by default (boot log: 1127/1127 primed), which is why the faithful arm below
    #             had to exist at all.
    #   + q4bTvip FAITHFUL published VIP: q4bV minus --warm-bank, i.e. the published bank-less
    #             configuration. The row labelled "VIP (published, no bank)". Window +1.88.
    #   + q4bTr   the scale-corrected-rho revision (nu from the 4B refit rho=0.7407), seed 1.
    #   + q4bTr2  its second seed. Two-seed window +4.51, the highest in the 4B block; its BFCL
    #             and NESTFUL are in the transfer emitters, and its NESTFUL is seed-unstable
    #             (11.34/31.00) -- disclosed in prose, not smoothed.
    #   + q4bTpe  the framework's PLUG-IN ESTIMATOR configuration at 4B (our bank + our certified
    #             exclusion + our harness, plug-in point estimate = our posterior's no-prior
    #             limit). One seed.
    # DELIBERATELY NOT ADMITTED, because their windows are incomplete as this pass runs: q4bTf
    # (faithful TRACE at 4B, 2 of 4 cells), q4bTpe2, q4bTr3. They enter the next refresh. Admitting
    # a mid-flight arm would raise m for everyone on cells that are still moving.
    # 2026-08-28 micro-fold: 15 -> 18, three arms whose windows completed overnight:
    #   + q4bTpe2 second seed of the plug-in configuration. Its BFCL (28.38) is 7.7 pp BELOW seed
    #             1's (36.12), so the two-seed mean lands at method parity and the earlier
    #             "balanced configuration leads BFCL" reading was seed luck. Printed as such.
    #   + q4bTr3  third seed of the revision, which arbitrates its NESTFUL instability 2-of-3.
    #   + q4bTf   FAITHFUL published TRACE at 4B, completing that baseline at both scales.
    # 2026-08-31 FOLD v: 18 -> 20.
    #   + q4bTiid the shrinkage-calibration ablation at 4B (--warm-shrink group -> none, the only
    #             flag that differs from q4bT2). Window +0.59 (solve 10.3) against the full
    #             method's +3.64: removing the calibration costs 3.0 pp at 4B and drops the arm
    #             BELOW the uniform control, which is printed rather than softened.
    #   + q4bLp2  the pre-registered TSCL rerun at 4B, the 4B counterpart of q2bLp2 and for the
    #             same reason. Window +2.54 (solve 12.2) against q4bLp's +2.63.
    # 2026-09-02 _fold7c: 20 -> 21 by ONE arm, admitted under the same membership rule every other
    # member is admitted under -- an arm of this model with a scorable four-cell window -- and not
    # because of what it measures:
    #   + t4bTbk  FAITHFUL PUBLISHED TRACE AT 4B PLUS OUR WARM BANK: q4bTf with --warm-bank at its
    #             default and no other flag changed, one seed at the same offset its parent ran.
    #             It is the 4B counterpart of t2bTbk and the second rung of the reviewer's
    #             bank-confound construction, and it carries a printed row in tab:ladder2b, so it
    #             is pinned rather than read outside the correction.
    # WHAT THE GROWTH MOVES, measured arm by arm rather than assumed: 20 -> 21 raises every 4B q,
    # and the emitter's own audit block prints the before/after for each. No verdict crosses 0.05
    # in either direction -- the two 4B arms that clear (per-scale rho, VIP rule with our bank)
    # still clear and everything outside stays outside -- which is what makes this growth a cost
    # paid rather than a result bought.
    # 2026-09-03 _fold7h: 21 -> 22 by ONE arm, the 4B counterpart of the 2B admission above and
    # admitted under the same rule:
    #   + q4bTrt  q4bTr with --temp 0.3 -> 1.12 and no other token changed, one seed at the
    #             parent's own offset (500). Window +1.74 (+0.68/+2.71/+2.03/+1.53, n=590 at
    #             every step) against the parent's +5.21 and the full recipe's +3.64. Printed
    #             row in tab:negatives.
    # WHAT THE GROWTH MOVES, measured arm by arm rather than assumed: 21 -> 22 raises every 4B q
    # and none falls. NO VERDICT CROSSES 0.05 in either direction -- the same two 4B arms clear
    # (per-scale rho q4bTr, VIP's rule on our bank q4bV) and everything outside stays outside; the
    # largest movement is q4bTr2 0.8025 -> 0.8754 and the closest pair to the line, q4bV2, goes
    # 0.0747 -> 0.0787. The arm itself enters at raw p = 0.557, q = 1.000, so what it costs is m
    # and nothing else.
    # 2026-09-04 _fold10: 22 -> 24 by TWO arms, THE 4B LEARNING-RATE RUNG. They are the 4B
    # counterparts of q2bF3e5 and q2bT3e, admitted under the same membership rule as every other
    # member -- an arm of this model with a scorable four-cell window -- and they carry printed
    # rows in tab:lrsweep, so they are pinned rather than read outside the correction:
    #   + q4bF3e5 the UNIFORM control at LEARNING RATE 3e-5 (q4bF with that token alone changed),
    #             one seed at the parent's own offset 500. Window +2.08
    #             (+2.03/+2.20/+1.69/+2.37); transfer BFCL 28.50 and NESTFUL 29.61.
    #   + q4bT3e5 the METHOD at the same rate (q4bT with that token alone changed), same offset.
    #             Window +1.99 (+0.85/+2.20/+1.86/+3.05); BFCL 31.00, NESTFUL 29.98.
    # WHAT THE PAIR MEASURES RUNS PARTLY AGAINST US AND IS PRINTED THAT WAY: at 3e-5 the control's
    # held-out tool use RECOVERS (BFCL 19.38 -> 28.50 against an untrained 29.00), so the 4B
    # transfer contrast is partly an optimisation-stability result, exactly as it is at 2B. What
    # survives at both scales is that the aggressive rate is where the in-distribution gain lives
    # and only the allocator keeps transfer positive there.
    # 2026-09-06 _fold12: 24 -> 27 by THREE arms, and this is the largest single 4B growth since
    # the comprehensive grid. Two are the interior points of the SAMPLING-TEMPERATURE DIAL and one
    # is the 4B rung of the VARIABLE-k PILOT. All three are one-flag arms of a configuration this
    # paper prints, all three have scorable four-cell windows, and all three acquire printed rows
    # in tab:negatives, so all three are pinned rather than read outside the correction:
    #   + q4bTm   q4bTr with --temp 0.3 -> 0.6 (the geometric midpoint of the dial) and no other
    #             token changed, one seed at the parent's own offset (500). Window +4.28
    #             (+3.90/+3.90/+5.08/+4.24); BFCL 23.62, NESTFUL 32.13.
    #   + q4bTm8  the same with --temp 0.85. Window +3.86 (+4.07/+2.37/+4.07/+4.92); BFCL 28.00,
    #             NESTFUL 32.89.
    #   + q4bK    q4bT with VARK=1 and no other token changed, one seed at the parent's own offset
    #             (500), the per-step rollout total pinned to the uniform k=5 spend. Window +4.11
    #             (+2.88/+4.07/+3.73/+5.76); BFCL 33.12, NESTFUL 29.18.
    # WHAT THE GROWTH MOVES, measured arm by arm rather than assumed. m 24 -> 27 raises every
    # pre-existing 4B q and none falls; the movements are
    #     q4bTr  0.0010 -> 0.0011 (*)   q4bV   0.0027 -> 0.0029 (*)
    #     q4bV2  0.0865 -> 0.0944       q4bR   0.1470 -> 0.1540
    #     q4bT2  0.1626 -> 0.1707       q4bTnw 0.1718 -> 0.1808
    #     q4bTr3 0.2436 -> 0.2571       t4bTbk 0.3413 -> 0.3613
    # and every other 4B arm was already at 1.000 and stays there. NO PRE-EXISTING VERDICT CROSSES
    # 0.05 in either direction: the two 4B arms that cleared (per-scale rho q4bTr, VIP's rule on
    # our bank q4bV) still clear and everything outside stays outside, with q4bV2 the closest at
    # 0.0944. ONE VERDICT IS NEW AND IT IS AGAINST THE PAPER'S OWN CONVENIENCE: q4bTm ENTERS
    # STARRED at raw p = 0.000066, q = 0.0017, so the 4B block goes from two starred arms to
    # three -- and the arm that acquires the star is an arm the pre-registered ship test REJECTS,
    # on BFCL, at 23.62 against a base of 29.00. An arm can clear this benchmark's correction and
    # still fail the shipping rule, because the shipping rule is not this benchmark; printing the
    # star and the rejection together is the only honest way to carry both.
    # q4bK enters at q = 0.1047 and q4bTm8 at q = 0.4132, so neither of those costs anything but m.
    "4B": ["q4bD", "q4bF", "q4bF2", "q4bF3e5", "q4bK", "q4bLp", "q4bLp2", "q4bP", "q4bR", "q4bT",
           "q4bT2",
           "q4bT3e5", "q4bTiid", "q4bTm", "q4bTm8", "q4bTnw", "q4bV",
           "q4bV2", "q4bTvip", "q4bTr", "q4bTr2", "q4bTr3", "q4bTrt", "q4bTpe", "q4bTpe2",
           "q4bTf", "t4bTbk"],
    # 39 arms -- grown from 38 on 2026-08-27 (the comprehensive-grid pass) by ONE arm:
    #   + t8Tf    FAITHFUL published TRACE at 8B: the plug-in rule with NONE of our components
    #             (no warm bank, no certified exclusion, no shrinkage, default temperature, cold
    #             start), boot-verified at cycle 1 with 0/1127 tasks carrying evidence and a
    #             constant weight vector, i.e. literally uniform allocation. It is the published
    #             baseline that a8Tpe -- which carries our components -- cannot stand in for.
    #             Its four-cell window is +3.35, which does NOT collapse: the registered
    #             cold-start-collapse prediction FAILS at 8B and the paper prints it as failed.
    # Two arms ALREADY in this family had their cells completed this pass and therefore move their
    # own q without any family growth: a8Tlp (the TSCL baseline) gained a full step-25 cell
    # (316 -> 590 rows) and a new step-30 cell, taking its window from the three-step -2.0 to the
    # four-step -2.58; and a8Tpe gained steps 20 and 25. a8Tpe still has NO step-30 cell (its
    # step-30 save is truncated) and is read on three steps, which the caption states.
    # NOT ADMITTED this pass: a8Tr (the 8B revision) has no scorable cell yet.
    # 2026-08-28 micro-fold: 39 -> 40 by a8Tr, the scale-corrected-rho revision at 8B (nu from the
    # 8B refit). It replicates the 4B window effect (+4.25 against the method's +3.11) and makes
    # scale-adaptive calibration a two-scale result rather than a 4B curiosity. One seed.
    # 2026-08-29 FOLD: 40 -> 41 by ONE arm.
    #   + a8Tvipf FAITHFUL published VIP at 8B -- a8Tvip minus our warm bank, the 8B counterpart of
    #             q4bTvip and q2bVf. Window -3.39 (-3.2/-2.7/-3.9/-3.7, n=590 at every step): it is
    #             the only arm in this family that is negative at every one of the four matched
    #             steps. Its transfer cells are BFCL 21.62 and NESTFUL 15.37 at step 30, both far
    #             below the untrained base policy, so the registered "collapses without the bank"
    #             prediction holds at 8B on all three axes.
    # 2026-08-31 FOLD v: 41 -> 42 by ONE arm (a second follows the moment a8Tiid banks its
    # window; see ABL_ARMS).
    #   + a8Tnw   the WARM-BANK ablation at 8B: a8T3g2 minus --warm-bank and nothing else. Window
    #             +0.89 (-0.2/+1.0/+0.8/+1.9, n=590 at every step; solve 12.2) against the full
    #             method's +4.58 at solve 15.9. Removing the bank costs 3.7 pp at 8B, the largest
    #             of the three scales, which is what makes the warm-bank claim a scale claim.
    # 2026-09-02 FOLD vi: 42 -> 43 by ONE arm, the second the previous entry said would follow.
    #   + a8Tiid  the SHRINKAGE-CALIBRATION ablation at 8B: a8T3g2 with --warm-shrink group ->
    #             none and no other recipe flag moved. Window +3.01 (+1.4/+1.9/+4.1/+4.7, n=590
    #             at every step) against the full method's +4.58 and the warm-bank ablation's
    #             +0.89, so the ordering full > no-shrink > no-bank that 2B and 4B print holds at
    #             8B too, and the "w/o shrinkage calibration" row of tab:ablation loses its last
    #             unmeasured cell. Its own step-15 raw p is 0.268 and its window q is 1.000, so
    #             the arm it admits carries no claim of its own; what it costs is m.
    #             m 42 -> 43 moves every 8B q upward and NOTHING crosses 0.05 in either
    #             direction: the closest pair, a8T3g/a8Tvip at step 15, goes 0.0514 -> 0.0531,
    #             still outside, and the arms inside stay inside (a8Cu/a8T2n 0.0435 -> 0.0448).
    # 2026-09-04 _fold10: 43 -> 44 by ONE arm, THE 8B RUNG OF THE BANK LADDER, admitted under the
    # same membership rule as every other member and not because of what it measures, which runs
    # against the component the paper prices highest:
    #   + t8Tbk  FAITHFUL PUBLISHED TRACE AT 8B PLUS OUR WARM BANK: t8Tf with --warm-bank at its
    #            default and no other flag changed, one seed at the same offset 500 its parent
    #            ran. It is the 8B counterpart of t2bTbk and t4bTbk and it completes the bank
    #            ladder at all three scales. Window -0.23 (+0.00/+0.68/-0.90/-0.68, n=590 at
    #            every step) against the bank-less parent's +3.35 and the full method's +4.58;
    #            transfer BFCL 37.50 (parent 36.88, base 35.88) and NESTFUL 37.51 (parent 39.07,
    #            base 38.26, i.e. BELOW the untrained policy). Infrastructure, because this arm
    #            needed a different one to fit: TP=2, parameter and optimizer offload, eager
    #            attention, sequence-parallel 2, util 0.29 -- the same list a8Tiid's disclosure
    #            carries. It has a printed row in tab:ladder2b, so it is pinned rather than read
    #            outside the correction.
    # 2026-09-06 _fold12: 44 -> 45 by ONE arm, THE 8B RUNG OF THE SAMPLING-TEMPERATURE DIAL,
    # admitted under the same membership rule as every other member and printed because of what it
    # measures, which runs FOR the arm and against the sharpening the 8B history selected:
    #   + a8Trt  a8Tr with --temp 0.5 -> 1.12 and no other token changed, one seed at the parent's
    #            own offset (500), TP=2 with parameter and optimizer offload, eager attention and
    #            sequence-parallel 2 -- the same infrastructure list a8Tiid and t8Tbk carry.
    #            Window +4.36 (+3.56/+3.05/+5.08/+5.76, n=590 at every step) against the parent's
    #            +4.25; BFCL 40.75 (parent 35.00, base 35.88) and NESTFUL 41.75 (parent 40.73,
    #            base 38.26). It is above its parent on all three axes, which is the OPPOSITE of
    #            what the same one-flag change does at 2B and 4B, and its BFCL cell is the highest
    #            any arm in this project has recorded on that benchmark. It carries a printed row
    #            in tab:negatives, so it is pinned rather than read outside the correction.
    # WHAT THE GROWTH MOVES: m 44 -> 45 raises every pre-existing 8B q and none falls; the
    # movements are a8T3g2 (<0.0005, unchanged to four places), a8Tr 0.0012 -> 0.0012,
    # a8F 0.0017 -> 0.0018, a8T5k 0.0024 -> 0.0024, a8Tpe 0.0320 -> 0.0329, a8T2n 0.0799 -> 0.0820
    # and a8A 0.1231 -> 0.1264, every other 8B arm being already at a q this growth cannot move.
    # NO VERDICT CHANGES: the same six arms clear 0.05 and the same thirty-nine do not; the closest
    # pair to the line, a8Tpe inside at 0.0329 and a8T2n outside at 0.0820, both stay on their own
    # side. a8Trt itself enters at raw p = 0.0079, q = 0.2929, so what it costs is m and nothing
    # else -- its case is made on the transfer benchmarks, where this family is not the test.
    "8B": ["a8A", "a8A3", "a8Ae0", "a8Auni", "a8Az0", "a8C", "a8Cu", "a8F", "a8Fr", "a8T", "t8Tf",
           "t8Tbk", "a8Tiid", "a8Tnw", "a8Tr", "a8Trt", "a8Tvipf",
           "a8T2", "a8T2b", "a8T2g1", "a8T2n", "a8T2nr", "a8T2r", "a8T2w", "a8T3", "a8T3g",
           "a8T3g2", "a8T3gr", "a8T4", "a8T5", "a8T5k", "a8T5kr", "a8T6d", "a8Te0", "a8Tlp",
           "a8Tpe", "a8Tvip", "a8Tvipr", "atscfix", "atscfix_lr1e5", "b8plr", "b8plrr", "b8ret",
           "b8van", "dapo"],
}


def full_family(model="8B"):
    """Every arm with at least one scorable cell in the window: effects, worst-case p, Holm q.

    The Holm family is deliberately ALL arms OF ONE MODEL -- baselines, the method's seeds, and
    every one of our own ablations -- so the multiplicity cost of the configuration search is paid,
    not hidden.

    ONE MODEL, and that is a membership rule fixed in advance (paper_audit.md B12, written before
    any 2B cell existed), not a filter chosen after seeing outcomes. The glob below admits every
    work/coadapt_eval_* directory and the fleet now trains 2B and 4B arms; folding them in would
    both score them against the wrong anchor (this base cell is the 8B policy) and raise m for
    every 8B arm. main_table owns the registry so there is exactly one definition of it.

    The discovered set is then intersected with FAMILY_PIN[model] -- see that dict for why the
    published family is pinned rather than left to the glob, and for how to grow it.
    """
    base = load_cell(BASE_CELLS[model])
    registry = arm_models()
    pin = FAMILY_PIN.get(model)
    rows = {}
    for d in sorted(glob.glob(f"{R}/work/coadapt_eval_*")):
        arm = os.path.basename(d).replace("coadapt_eval_", "")
        if model_of(arm, registry) != model or arm in ABSORBED:
            continue
        if pin is not None and arm not in pin:
            continue
        per, ps, ns = {}, {}, {}
        # 2026-09-09 (PI): fig:stepcurve now plots the ABSOLUTE solve rate, so the rate and its
        # anchor are carried alongside the effect. Both are computed on the SAME intersection the
        # effect is, which is what makes the two readings reconcile exactly: absr - basr == per.
        # Nothing here changes `per`, so every table, Holm family and statistic is untouched.
        absr, basr = {}, {}
        for st in WINDOW:
            cur = load_cell(cell_path(arm, st))
            keys = sorted(set(cur) & set(base))
            if len(keys) < 300:
                continue
            b = sum(1 for k in keys if base[k] and not cur[k])
            c = sum(1 for k in keys if cur[k] and not base[k])
            per[st] = (sum(cur[k] for k in keys) - sum(base[k] for k in keys)) / len(keys)
            absr[st] = sum(cur[k] for k in keys) / len(keys)
            basr[st] = sum(base[k] for k in keys) / len(keys)
            ps[st] = mcnemar(b, c)
            ns[st] = len(keys)
        if per:
            rows[arm] = {"per": per, "mean": sum(per.values()) / len(per), "p": max(ps.values()),
                         "ns": ns, "absr": absr, "basr": basr}
    if pin is not None:
        # Loud, not silent: a pinned arm that stops being scorable would shrink m and move every
        # published q downward, which is the same failure as the growth this pin exists to stop.
        missing = [a for a in pin if a not in rows]
        if missing:
            raise SystemExit("FAMILY_PIN[%r] names %d arm(s) with no scorable cell: %s -- the "
                             "published family cannot be reproduced; fix the pin deliberately "
                             "and narrate it, do not let m drift." % (model, len(missing), missing))
    qs = holm([r["p"] for r in rows.values()])
    for r, q in zip(rows.values(), qs):
        r["q"] = q
    return rows, len(base), len(rows)


def cell_turns(path):
    """{(scenario, task, seed): n_turns} from one eval cell.

    Exactly load_cell's keying and first-occurrence rule -- same file, same records, same
    de-duplication -- reading the `n_turns` field instead of `reward`. It is a SECOND READ OF THE
    SAME CELL, not a second definition of anything: every eval record this project has ever written
    carries n_turns, and the coverage is checked (not assumed) by main_row_metrics below.
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
        t = d.get("n_turns")
        if t is None:
            continue
        out[k] = float(t)
    return out


def cell_records(path):
    """{(scenario, task, seed): record} from one eval cell.

    Exactly load_cell's keying and first-occurrence rule -- same file, same records, same
    de-duplication -- returning the WHOLE record instead of one field. cell_turns() is the same
    read for n_turns alone and is kept; this exists because tab:main's cost columns need three
    fields of the same record and reading the file three times would be three chances for the
    three columns to disagree about which records they are averaging.
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
        out[k] = d
    return out


# --- tab:main's two COST columns (2026-08-29, PI 21:12) ---------------------------------------
# THE INVALID TOOL-CALL RATE IS A NAMED METRIC AND NOT ONE WE COINED. It is the share of the
# policy's tool-call ATTEMPTS that the environment could not execute, which is what the
# function-calling and tool-use literature reports under that name. Numerator and denominator are
# both counted per episode by the harness, in every record it has ever written:
#     invalid = n_parse_fail + n_unknown_tool          a call that did not parse, or that named a
#                                                      tool outside the advertised surface
#     attempts = n_backend_calls + invalid             every turn at which the policy tried to call
# so the rate is invalid / attempts. It is NOT the backend error rate (n_errors), which counts a
# well-formed call that the backend then refused: that is the environment's answer to a valid
# request and it is a different quantity, kept out of this column deliberately.
#
# WHY THIS COLUMN AND NOT ANOTHER. It was chosen for what it measures, not for what it says about
# us, and it does not flatter us: at 2B the method's rate is above the untrained policy's. The
# alternatives were audited on the same records and rejected on grounds stated here rather than
# after seeing them -- tool calls per task is ~ turns minus one and would have re-printed the
# turns column, which is the redundancy this pass exists to remove; generated tokens per task is
# a cost metric the turns column already tracks closely. Every candidate was 100% populated on all
# 24 row x scale cells, so this is a choice among filled options and not a choice forced by
# coverage.
#
# ONE CONVENTION FOR THE WHOLE TABLE. Both cost columns are WINDOW MEANS over the pre-registered
# {15,20,25,30}, on the same shared paired index the solve rate is read on, on the same best seed.
# Until this pass the turns column was read at step 15 alone; it is now the window mean like
# everything else in the float, which is a changed convention, is stated in the note, and is the
# reason its digits moved. tab:mainabs keeps the step-15 reading.
def window_costs(arm, base, steps=None):
    """(invalid tool-call rate in %, mean turns per task) for one run over the window.

    Read on the tasks this arm and the anchor both attempted, cell by cell, exactly as the solve
    rate is. Returns (None, None) if no cell of the window is scorable, which is a refusal
    upstream rather than a dash here: tab:main prints no coverage markers.
    """
    inv = att = 0.0
    tsum = 0.0
    tn = 0
    for st in (WINDOW if steps is None else steps):
        cur = cell_records(cell_path(arm, st) if st is not None else arm)
        keys = sorted(set(cur) & set(base))
        if len(keys) < 300:
            continue
        for k in keys:
            d = cur[k]
            pf = float(d.get("n_parse_fail") or 0)
            un = float(d.get("n_unknown_tool") or 0)
            bc = float(d.get("n_backend_calls") or 0)
            inv += pf + un
            att += pf + un + bc
            t = d.get("n_turns")
            if t is not None:
                tsum += float(t)
                tn += 1
    if not att or not tn:
        return None, None
    return 100.0 * inv / att, tsum / tn


def main_row_metrics(arms, base, step=None):
    """tab:main's per-checkpoint columns for ONE row, from the ONE step every arm has banked.

    Returns None if no listed arm has a cell, else
        dict(rate, ci, turns, n, k)   -- percent, percent, turns, paired pairs, records with n_turns

    ONE DEFINITION, APPLIED IDENTICALLY TO EVERY ROW, which is the whole point of this function:

      * The key set is the SAME set the paired effect uses -- the tasks this arm and the base
        anchor both attempted -- so the absolute rate and the effect are read off the same
        evaluation and a row cannot be scored on tasks its own effect ignores.
      * A multi-seed row POOLS its seeds' cells. With the fleet's equal-size cells that is the seed
        mean, and it is a single rule rather than one rule for replicated rows and another for
        single-seed ones. That is deliberate: seed-sd is NOT available for a single-seed row, so no
        seed-derived spread may appear in this table at all.
      * `ci` is the half-width of the 95% binomial (Wald) interval on that pooled rate,
        1.96*sqrt(p(1-p)/n). It is SAMPLING ERROR ON THE EVALUATION and is defined for every row
        from n alone, which is exactly why it is the spread this table can carry. It is NOT the
        seed spread; that lives in tab:mainsteps and tab:perseed and the caption says so.

    Training-log quantities (episodes, live groups, val-solve rate) are deliberately absent: they
    are undefined for a8Tvip (no episode log at all), and for a8F/b8plr (no val episodes), so a
    column built on them would blank exactly the baseline rows a reader most wants to check.
    """
    st = MAIN_STEP if step is None else step
    solves = n = 0
    tsum = 0.0
    tn = 0
    got = False
    for a in arms or []:
        cur = load_cell(cell_path(a, st))
        keys = sorted(set(cur) & set(base))
        if len(keys) < 300:
            continue
        got = True
        turns = cell_turns(cell_path(a, st))
        solves += sum(cur[k] for k in keys)
        n += len(keys)
        for k in keys:
            if k in turns:
                tsum += turns[k]
                tn += 1
    if not got or not n:
        return None
    p = solves / n
    return dict(rate=100.0 * p, ci=100.0 * 1.96 * math.sqrt(p * (1 - p) / n),
                turns=(tsum / tn) if tn else None, n=n, k=tn)


def pp(x, dec=1):
    """Proportions -> PERCENTAGE POINTS, one decimal. The paper reports every effect in pp: a
    reader compares 4.8 against 3.4 far faster than 0.0479 against 0.0339, and the raw-proportion
    precision is preserved in the reproduction tables of Appendix~\\ref{app:perseed}."""
    return "%+.*f" % (dec, 100 * x)


DAGGER = "^{\\dagger}"   # cell whose paired count has not reached the ~590 a finished cell reaches
STAR = "^{*}"            # clears Holm at 0.05 WITHIN ITS OWN family (families are per model)
PARTIAL_N = 560          # below this a cell is still filling and is marked rather than printed flat


def _cells(r, full_n=None):
    """The four matched-step cells of one arm, daggered where the cell is still filling."""
    out = []
    for st in WINDOW:
        v = r["per"].get(st) if r else None
        if v is None:
            out.append("--")
            continue
        partial = full_n is not None and r.get("ns", {}).get(st, full_n) < PARTIAL_N
        out.append("$" + pp(v) + (DAGGER if partial else "") + "$")
    return out


# =============================================================================================
# SEED AGGREGATION. Every main-text float reports METHODS; only the appendix reproduction tables
# report seeds.
#
# THE RULE IN FORCE SINCE 2026-08-28 (PI decision, recorded in PLAN_TRIAGE.md 23:00 and in the
# entry this pass writes): EVERY MULTI-SEED ARM REPORTS ITS BEST SEED BY WINDOW MEAN -- ours, the
# control, and every multi-seed published baseline alike -- in tab:main, fig:stepcurve and
# tab:transfermain (and in tab:mainsteps / tab:mainabs, which are those columns' appendix homes).
# The rule is SYMMETRIC and it is disclosed in every caption that carries a selected cell: a row is
# "best of n", n is in tab:perseed, and tab:perseed prints every seed AND each block's seed mean.
# The selected seed carries ITS OWN Holm q -- no q is recomputed, because each seed is already its
# own member of the pinned per-scale family (see full_family: the family is arms, not methods), so
# the multiplicity of the search is paid exactly as before. Single-seed arms are untouched.
#
# WHY THIS IS PRINTED AS A SELECTION AND NEVER AS "THE METHOD'S NUMBER". This paper measures a
# within-configuration seed sd of ~1.5 pp against a between-arm sd of the same 1.5 pp, and it
# derives a 6.0 pp resolution floor from it. A best-of-n cell is therefore an upper reading of a
# quantity the benchmark cannot resolve, and the paper says so where it prints one. What makes the
# convention defensible rather than flattering is that it is applied to EVERY multi-seed row,
# including the control (whose 2B best seed is +1.2 against a seed mean of -0.2, and whose 8B best
# seed is +4.0 against +3.2) and including a published baseline -- so no comparison in any of these
# floats is between one arm's maximum and another arm's average.
#
# THE RULE THIS REPLACED, KEPT AS HISTORY (in force to 2026-08-28): "one row per method, the marker
# is the seed mean, the spread is the seed sd where replicates exist, and the seed count is a
# column of its own rather than a footnote. Where a method has one seed the sd is omitted rather
# than printed as zero, because zero would claim a precision that one run cannot support." That
# rule STILL governs every float this pass did not name -- tab:framecfg, tab:ladder,
# tab:negatives, tab:scalemain, tab:mech, tab:transfer and the per-scale transfer grids -- so
# agg()'s default is unchanged and only the callers listed above pass best=True.
# =============================================================================================

_OFF = re.compile(r"off=(\d+)")


def arm_offsets():
    """{tag: task-offset} from supervisor.sh's case block -- the arm's SEED, in the trainer's own
    terms. Seed numbers in the reproduction tables are derived from this rather than from a row's
    position in a list, because position renumbers whenever an arm is added, dropped or reordered:
    the 2B TRIAGE replicate scored on the transfer benchmark and the one scored in-distribution are
    different runs, and calling them both 'seed 2' by position would silently merge two arms."""
    out = {}
    try:
        text = open(SUPERVISOR, errors="ignore").read()
    except Exception:
        return out
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_|]+)\)\s", line)
        mm = _OFF.search(line)
        if not m or not mm:
            continue
        for tag in m.group(1).split("|"):
            out.setdefault(tag, int(mm.group(1)))
    return out


# The offsets the fleet actually uses, in launch order. An arm at an unlisted offset keeps its
# numeric offset as its label rather than being forced into a slot it does not occupy.
SEED_OF_OFFSET = {500: 1, 1500: 2, 2500: 3}


def seed_label(tag, offsets):
    o = offsets.get(tag)
    return str(SEED_OF_OFFSET.get(o, o)) if o is not None else "--"


def best_seed(fam, arms):
    """The seed of one configuration with the HIGHEST WINDOW MEAN, or None.

    THE SELECTION RULE, IN ONE PLACE, SO EVERY FLOAT THAT USES IT USES THE SAME ONE. The ordering
    key is the four-step window mean and nothing else: not the q, not a transfer score, not a
    single step. A seed with no scorable window cannot be ranked and is not a candidate -- that is
    a coverage fact, and where it bites (the 2B transfer replicate q2bT2 has BFCL records and no
    coadapt window at all) the emitting caller names the arm in its own comment rather than
    letting the reader believe the maximum ranged over more runs than it did. Ties break on the
    lower q and then on the tag, so the choice is deterministic across re-emissions.
    """
    got = [a for a in arms if a in fam]
    if not got:
        return None
    return max(got, key=lambda a: (fam[a]["mean"], -fam[a]["q"], a))


def agg(fam, arms, best=False):
    """One METHOD's row. None when no listed seed has a scorable cell.

    best=False (the default, and every float this pass did not name): seeds AGGREGATED. Returns
    per-step seed means, the mean of the seed window means, the seed sd (None if n=1), the seed
    count, and the BEST per-arm Holm q among the seeds -- q values are never averaged, which is
    meaningless, so the best is reported and the caption says that it is one seed's.

    best=True (tab:main, fig:stepcurve, tab:mainsteps, tab:mainabs, tab:transfermain, and the
    Figure-1 bar macros that are emitted beside tab:main): the row is ONE SEED -- the one with the
    highest window mean, by best_seed() above -- and every quantity in the returned dict is that
    seed's: its four per-step effects, its window mean, its own Holm q. `sd` is None, so nothing
    downstream prints a +- on a single run; `n` still carries how many seeds were CANDIDATES, and
    `lo`/`hi` still span every candidate's window mean, because a caption that says "best of n"
    needs n and a reader who wants the spread is owed the range. `sel` names the selected arm and
    `seedmeans` carries every candidate's window mean, both for the audit block and for the
    per-seed table's selection marker.
    """
    got = [a for a in arms if a in fam]
    if not got:
        return None
    ms = [fam[a]["mean"] for a in got]
    mean0 = sum(ms) / len(ms)
    sel = best_seed(fam, got)
    if best:
        r = fam[sel]
        per = dict(r["per"])
        absr, basr = dict(r.get("absr", {})), dict(r.get("basr", {}))
        ns = dict(r["ns"])
        mean, sd, cells = r["mean"], None, len(r["per"])
        q = r["q"]
    else:
        per, ns = {}, {}
        absr, basr = {}, {}
        for st in WINDOW:
            v = [fam[a]["per"][st] for a in got if st in fam[a]["per"]]
            if v:
                per[st] = sum(v) / len(v)
                ns[st] = min(fam[a]["ns"][st] for a in got if st in fam[a]["per"])
                av = [fam[a]["absr"][st] for a in got if st in fam[a].get("absr", {})]
                bv = [fam[a]["basr"][st] for a in got if st in fam[a].get("basr", {})]
                if av:
                    absr[st] = sum(av) / len(av)
                    basr[st] = sum(bv) / len(bv)
        mean = mean0
        sd = (sum((m - mean) ** 2 for m in ms) / (len(ms) - 1)) ** 0.5 if len(ms) > 1 else None
        cells = min(len(fam[a]["per"]) for a in got)
        q = fam[min(got, key=lambda a: fam[a]["q"])]["q"]
    return dict(arms=got, per=per, absr=absr, basr=basr,
                ns=ns, mean=mean, sd=sd, n=len(got), lo=min(ms), hi=max(ms),
                q=q, best=min(got, key=lambda a: fam[a]["q"]), cells=cells,
                sel=sel, seedmeans=dict(zip(got, ms)), seedmean=mean0, selected=best)


def eff(a, bold=False, star=True):
    """A method's window effect as one cell: mean, seed sd where it exists, Holm mark."""
    if a is None:
        return "--"
    s = pp(a["mean"])
    if a["sd"] is not None:
        s += "\\pm%.1f" % (100 * a["sd"])
    if star and a["q"] < 0.05:
        s += STAR          # at least one seed of this row clears Holm in its own family
    return "$\\mathbf{" + s + "}$" if bold else "$" + s + "$"


def qcell(a):
    """The row's Holm q: for a multi-seed row, its best seed's.

    NO seed count and NO seed identity is printed. Seed information is out of every main-text
    float entirely (2026-08-17 PI directive) -- a count parenthesised in a cell is still an
    invitation to read the row as a set of runs rather than as a method, which is the reading this
    paper spends a subsection arguing the benchmark cannot support. Each caption carries one clause
    saying that q is the best seed's, and the appendix reproduction table carries every seed.
    """
    if a is None:
        return "--"
    if a["q"] < 0.0005:
        return "$<0.001%s$" % STAR
    return "$%.3f%s$" % (a["q"], STAR if a["q"] < 0.05 else "")


# ---------------------------------------------------------------------------------------------
# FROZEN MECHANISM ROWS. work/verl/run_dapo/ was recycled for a different, still-running arm on
# 2026-08-14 (.arm_since=73678 written 08:56 while episodes.jsonl kept growing), so train_rows()
# now reads 8 optimizer steps of somebody else's run and labels them DAPO. The segment boundary is
# unrecoverable -- .arm_since was destructively overwritten at the relaunch -- and none of the three
# candidate segments reproduces the published 44 / 0.883 / 23.4:
#     rows [0, 73678)   -> 445 steps, 0.897, 21.0
#     rows [73678, EOF) ->   8 steps, 0.871, 25.8   <- what the emitter would take
#     whole file        -> 454 steps, 0.897, 21.1
# The published row is the one a human verified, so it is frozen HERE rather than hand-patched into
# coadapt2x2.tex after every regeneration. Delete this entry once the arm is re-measured.
# {tag: (steps, degenerate_fraction, live_groups_per_1k)}
FROZEN_MECH = {"dapo": (44, 0.883, 23.4)}


# =============================================================================================
# TABLE SPECIFICATIONS. Every arm the paper prints appears in exactly one of these lists, so the
# set of arms a table claims is reviewable in one place instead of being a glob.
# =============================================================================================

# --- Table: the main held-out comparison. Method / Control / Published baselines. -------------
# The measured-negatives block that used to close this table now lives in the appendix
# (negatives.tex): those arms are falsifications of our own earlier ideas, they are discussed in
# the ablation subsection, and carrying four extra rows through the main table cost more space than
# the reminder was worth.
# 2026-08-18 (PI directive on Table 3, second directive of the day). Three changes here:
#
#  (1) ONE METHOD ROW. The sharpened variant ($\tau{=}0.3$) left this table. It is a variant of the
#      method, not the method of record, and it is already a row of the ablation table (ABLATION's
#      "sharpened sampling" row) and of the per-seed appendix table (PERSEED). Nothing is lost and
#      nothing is hidden; the main comparison now carries exactly one row a reader is asked to read
#      as "the method": TRIAGE v3g, the three-seed configuration of record.
#
#  (2) CITATIONS ON THE BASELINES. Every non-ours row now names its source. All six keys were
#      already in refs.bib; only gan2025ragmcp was previously uncited.
#
#      A NOTE THE CAPTION REPEATS, because it is a provenance claim and not a formatting one:
#      every baseline row is OUR reimplementation of the cited rule inside this paper's own
#      allocator and protocol, not the cited authors' code. Two labels already carried that hedge
#      ("-style", "-shaped") and it is now stated once for all of them. In particular the
#      retrieval-shaped surface is a BM25 stand-in for the retrieval-front-end practice
#      gan2025ragmcp describes (see surface_retrieval.py's header: "a fair, cheap, reproducible
#      stand-in for a sensible deployed retriever"), NOT a reimplementation of that system.
# 2026-08-22, FINAL REGENERATION (PI directive A, "baseline grids by scale"). The table is now
# SCALE-BLOCKED: the same seven method rows under a spanning divider per model size, each block
# scored against ITS OWN model's anchor and Holm-corrected inside ITS OWN model's pinned family.
# Nothing about the columns, the selection rule or the seed convention changed -- only that the
# grid the paper previously had at 8B alone is now printed at all three sizes, which is what makes
# the 4B non-separation visible in the same float as the 8B separation instead of two floats away.
#
# 2026-08-24, PRE-FINAL FOLD: THE 4B GRID IS NOW COMPLETE AND NO ROW OF THIS TABLE IS A COVERAGE
# GAP ANY MORE. The two rows that printed `not run at this scale` -- VIP and TSCL at 4B -- are
# measured rows now (q4bV, q4bLp), and the 4B TRIAGE row gains its second seed (q4bT2), so all
# three model blocks carry the same seven configurations under the same seed convention.
#
# `NOT_RUN` IS KEPT AND IS DELIBERATELY NOT DELETED, THOUGH NOTHING USES IT TODAY. It is a THIRD
# state, distinct from both "a cell exists" and "the cell is below the scoring bar": the arm was
# registered and never produced a window. It renders as a spanning \emph{not run at this scale}
# across the data columns; a "--" would have read as a missing measurement on a measured arm,
# which is the one thing a coverage gap must not be confused with. The machinery stays so that the
# next gap is disclosed in the row rather than argued about in a caption. It is currently unused,
# and the assertion below fails the build if any prose still claims a gap while no row declares
# one -- the two must not be able to drift apart.
NOT_RUN = "@NOTRUN"
# A cell that is neither "measured" nor "not run": the arm exists, is registered and is training or
# scoring as this build is cut, but its window is incomplete, so it is NOT in the pinned family and
# has no row value. Printing "not run at this scale" for it would be false, and printing a partial
# window would be worse -- a two-cell mean is not the four-cell statistic every other cell in this
# column reports. So it gets its own honest marker, and it promises no outcome.
IN_FLIGHT = "@INFLIGHT"
# A FOURTH state, and the one the 2026-08-29 02:25 coordinator decision needs. "sharpened
# (tau=0.3)" is a8T3g2's recipe with --temp 0.3 and TRIAGE_EPS=0.05 -- two scalars off the 8B
# fixed-rho arm, which runs --temp 0.5 and the default eps. But the 2B and 4B fixed-rho arms
# (q2bT/q2bT3, q4bT/q4bT2) ALREADY train at --temp 0.3 with TRIAGE_EPS=0.05, so a "sharpened 2B"
# or "sharpened 4B" arm would be byte-identical to that scale's fixed-rho launch but for the seed
# offset: a second fixed-rho seed printed under a second configuration name, at two GPU-days each.
# It was NOT launched. Printing NOT_RUN ("not run at this scale --- coverage, not a null") would
# be false in the other direction: the configuration is not absent here, it COINCIDES with one
# already in the table. So it gets its own cell text and its own table note.
SAME_FIXED = "@SAMEFIXED"

# 2026-08-28 PI DIRECTIVE: ONE method row per scale, carrying the scale-adaptive-calibration
# configuration. THE RULE IS ONE ALGORITHM, NOT A PER-SCALE PICK: rho is set to the scale's own
# measured refit (2B 0.6858, 4B 0.7407, 8B 0.7292 -- Appendix J.7's numbers, fitted before these
# arms ran), so the configuration is specified by a stated rule rather than by choosing whichever
# arm won at each scale. Two things are stated in the caption rather than left implicit, because
# without them this row WOULD be a post-hoc selection: (a) this configuration was measured AFTER
# the fixed-rho one and is a revision of it, and (b) the fixed-rho configuration is not removed --
# it heads tab:framecfg, together with every other configuration of the framework.
#
# 2026-08-28, BEST-SEED PASS. The per-scale choice is UNCHANGED IN KIND -- it is still made after
# the measurements existed and the caption still says so -- but the quantity it is made on moved
# with the reporting convention, from the seed mean to the BEST SEED'S window mean, with the same
# transfer-collapse exclusion as before. Recomputed rather than assumed, and printed here so the
# selection can be checked without rerunning anything (every value is a window mean in pp):
#     2B   fixed-rho  best seed +2.59 (q2bT)      scale-refit +2.67 (q2bTr, ONE seed)
#          -> FIXED-RHO. The scale-refit arm is still numerically ahead and is still EXCLUDED for
#             the same measured reason as before: 9.00 on BFCL against an untrained 15.12.
#     4B   fixed-rho  best seed +3.64 (q4bT2)     scale-refit best seed +5.21 (q4bTr)
#          -> SCALE-REFIT (the revision), as before.
#     8B   fixed-rho  best seed +4.58 (a8T3g2)    scale-refit +4.25 (a8Tr, ONE seed)
#          -> FIXED-RHO. THIS ROW CHANGES CONFIGURATION under the new convention: on seed means the
#             revision led (+4.25 against the three-seed +3.86) and the row was the revision; on
#             best seeds the fixed-rho configuration's third seed leads it. The change is not
#             silent -- it is printed here, in the caption, in CHANGES.md and in PLAN_TRIAGE.md --
#             and it has one incidental benefit worth naming: the 8B method row now carries the
#             SAME configuration every transfer table in this paper reports (fixed-rho), which the
#             single-seed revision did not.
# --- THE FRAMEWORK'S CONFIGURATIONS, DEFINED ONCE ---------------------------------------------
# One dict, read by BOTH tab:main's configuration block and tab:framecfg, so the two floats cannot
# come to carry different arm sets for the same configuration name. Every arm listed here is in
# FAMILY_PIN for its scale (verified 2026-08-29, arm by arm); this dict may not admit an arm that
# is not, because an unpinned arm in a main-text row would either raise m silently or print a cell
# with no correction behind it.
# NOT_RUN is the third state and is printed as such. It applied to the plug-in configuration at
# 2B until 2026-08-31, when q2bTpe banked its window and its transfer pair; the sharpened variant
# still exists only at 8B, and that is the one NOT_RUN this dict now carries.
CFG_ARMS = {
    "fixed":   {"2B": ["q2bT", "q2bT3"], "4B": ["q4bT", "q4bT2"],
                "8B": ["a8T3g", "a8T3gr", "a8T3g2"]},
    "refit":   {"2B": ["q2bTr"], "4B": ["q4bTr", "q4bTr2", "q4bTr3"], "8B": ["a8Tr"]},
    "vipbank": {"2B": ["q2bV"], "4B": ["q4bV", "q4bV2"], "8B": ["a8Tvip", "a8Tvipr"]},
    # 2026-08-31: the 2B cell is q2bTpe, the registered recipe with --estimator point and no
    # other change. It cost three launches to get (two ran the wrong recipe off a regenerated
    # command, one lost its actors to a missing KEEPCKPT); the arm that banked is the original
    # gpu023 command with --jobid and GPU_UTIL changed and nothing else.
    "plugin":  {"2B": ["q2bTpe"], "4B": ["q4bTpe", "q4bTpe2"], "8B": ["a8Tpe"]},
    # The sharpened variant is a configuration of the method (tau=0.3 on the sampling weights) and
    # has been a row of tab:ablation since 2026-08-17; the PI's 2026-08-29 directive names it as a
    # configuration, so it is a row of tab:main's configuration block and of tab:framecfg as well.
    # It was run at 8B only. 2026-08-29 (coordinator decision, PLAN 02:25, resolution A): its 2B
    # and 4B cells are NOT a coverage gap -- at those scales the sharpened configuration IS the
    # fixed-rho configuration, so they print "= fixed-rho" and the table note says why.
    "sharp":   {"2B": SAME_FIXED, "4B": SAME_FIXED, "8B": ["a8T5k", "a8T5kr"]},
}
# The label each configuration carries in BOTH tables. Short enough for tab:main's four-column
# scale groups; tab:framecfg uses the identical strings so a reader moving between the two floats
# never has to match two names for one configuration.
# NESTFUL cells that are NOT MEASURABLE, and not for want of trying: the arm's step-30 LoRA was
# pruned off scratch after its window was scored, so there is no checkpoint left to score on this
# benchmark and retraining is the only way to fill the cell. That is a different fact from "never
# run at this scale" and from "not measured on this benchmark", and it prints as its own marker
# with its own note (PLAN 2026-08-29 02:10). Verified arm by arm: every global_step_* of each holds
# fsdp_config.json + huggingface/ + lora_train_meta.json and no tensor.
NESTFUL_PRUNED = {"q2bV", "q2bLp", "q4bLp", "b8plrr"}

# 2026-08-31: THE THREE PRUNED CELLS ARE MEASURED, AND NOT BY RECOVERING ANYTHING. Each arm above
# that a main-text row actually displays was RE-TRAINED, off the same launch command at the SAME
# --seed-offset as its parent (500 for all three), and the rerun was pre-registered on 2026-08-29
# before any of its cells existed. It is never labelled with the parent's tag, and it is never
# called a second seed either: same offset, different execution, so what separates the two runs is
# rollout sampling and kernel nondeterminism. Its own window cells are printed beside its parent's
# in tab:reruns, and the parent keeps every window row it had.
#
# WHY THE RERUN SUPPLIES BOTH TRANSFER CELLS AND NOT ONLY THE NESTFUL ONE. A row whose BFCL cell
# came from the parent run and whose NESTFUL cell came from the rerun would be two runs printed as
# one row, which is exactly what this table's note forbids everywhere else ("every other one is
# the seed its BFCL cell is read on"). So the override replaces the DISPLAYED ARM at that scale,
# both cells move together, and both carry \ddag.
#
# WHY NOT SIMPLY ADD THE RERUN TO THE ROW'S ARM LIST: checked before choosing. _pick takes the
# highest window mean and that is the PARENT at all three (q2bV +1.95 over q2bV2 +1.27, q2bLp
# +0.68 over q2bLp2 +0.55, q4bLp +2.63 over q4bLp2 +2.54), so the pruned arm would be selected
# again and the dash would come straight back.
TRANSFER_RERUN = {"q2bV": "q2bV2", "q2bLp": "q2bLp2", "q4bLp": "q4bLp2"}

# --- review P0-5, RESOLVED 2026-08-29 (fold iii) -----------------------------------------------
# THE 8B METHOD ROW'S NESTFUL CELL IS NO LONGER THE PRE-REGISTERED a8T CHECKPOINT. The reviewer's
# P0-5 was that tab:transfermain's row labelled "\methodname{} (fixed-rho)" printed a cell scored
# on a8T, the COLD-START v1 arm (tab:nestrep names it that, and its in-distribution window is
# -1.5 pp at q = 1.000), while the row's own selected seed at 8B is a8T3g2. One arm was labelled,
# another was measured. The cell below is the row's own seed, at step 30.
#
# THE ARM TAG IS a8T3g2_s30 AND THAT SUFFIX IS NOT COSMETIC. The 2026-08-29 20:10 job scored
# a8T3g2 at step 30 and then at step 15 into ONE --out-dir, and bench_nestful.py both RESUMES
# generation from and overwrites artifacts in that directory. So the step-15 pass merged and
# served global_step_15, logged "resume: 1861 of 1861 already generated" and "generate: nothing
# to do" (run_p05_a8T3g2.log 3985-3986), NEVER SAMPLED, re-scored the step-30 completions and
# stamped step 15 on all 1861 records -- identical win=750 full=467 parse_err=167 in both passes
# (log 3963 and 7922) because they are the same completions scored twice. run_a8T3g2/records.jsonl
# is therefore internally consistent and wrong. run_a8T3g2_s30 is a copy of it with that one field
# corrected, the original left exactly as written, and PROVENANCE.txt in it citing those log
# lines. Nothing was recomputed. THERE IS NO a8T3g2 STEP-15 CELL and none may be read out of
# run_a8T3g2; the step filter this file passes to NR.load_run cannot detect the mislabel, because
# the file carries one step, the wrong one.
#
# WHAT THIS COSTS, STATED IN THE TABLE RATHER THAN ABSORBED. The control's 8B NESTFUL checkpoint
# is a8F at step 15, the only step it was scored at, and it is NOT being re-run (one measurement
# per role; PI 20:35). So the 8B NESTFUL difference row is a step-30 cell minus a step-15 cell.
# It is printed, marked with \P, and the mismatch is stated in the note. The PRE-REGISTERED
# paired contrast is untouched: nestful_records.SCALES still names a8T against a8F at step 15,
# so tab:transfer and tab:nestrep carry exactly the m=3 family that was fixed before any cell was
# scored, and a8T's 39.01 keeps its place there under its own tag.
METHOD_NEST = {"8B": ("a8T3g2_s30", 30)}

# 2026-08-29 (PI 21:12/21:50): THE INTERNAL CONFIGURATION JARGON IS GONE FROM EVERY PRINTED
# LABEL. "fixed-rho posterior", "scale-refit rho", "plug-in estimator", "VIP rule + bank" and
# "sharpened (tau=0.3)" named our own switches, not the thing a reader can check against Section 6.
# Each label now names the ONE component the arm changes, in the ablation idiom every recent
# RL-for-LLM paper uses (full method first, "w/o X" or "X -> Y", one change per row).
# VERIFIED AGAINST slurm/supervisor.sh's actual launch flags, arm by arm, not against memory:
#   fixed   the full method: --task-alloc gradmass --warm-bank default --temp T --exclude-tasks
#           --warm-shrink group, at the default correlation constant.
#   refit   the full method + TRIAGE_NU=<that scale's own refit> (0.3501 at 4B, 0.3713 at 8B).
#           ONE change: the correlation constant stops being one number for every scale.
#   plugin  the full method + --estimator point. ONE change: w is g(p_hat,k) at the posterior
#           mean instead of E[g(p,k)] under the posterior. Same evidence, same rule, same budget.
#   vipbank --task-alloc vip --warm-bank default. NOT a one-line change and the note says so: it
#           swaps the allocation rule AND drops --temp, --exclude-tasks and --warm-shrink.
#   sharp   the full method at --temp 0.3 where the 8B default is 0.5 (plus TRIAGE_EPS=0.05).
#           At 2B and 4B the full method ALREADY runs at 0.3, so those cells are the full row's.
# SYMBOLS. rho is defined in Section 6 (nu = (1-rho)/rho, and "the single constant rho = 0.78"),
# so it may appear in a label. TAU IS NOT DEFINED IN THE MAIN TEXT, so the sharpening row says
# what it does in words; tab:coadapt in the appendix, which does define it, keeps tau.
# TWO LABEL SETS, AND THE SPLIT IS A WIDTH FACT RATHER THAN A STYLE ONE. CFG_LABEL is read by
# tab:transfermain and by four appendix floats (framecfg, mainfull, mainsteps, mainabs) whose
# label columns were sized for the old jargon; the ablation phrasing Table 2 needs ("estimator:
# posterior -> plug-in") is 20pt wider than any of them can carry and overflowed all four when it
# was tried. So CFG_LABEL keeps SHORT plain-language names -- no "fixed-rho posterior", no
# "scale-refit rho", nothing a reader has to decode -- and ABL_LABEL below carries the full
# ablation phrasing for the one float that is an ablation table. Same arms, same rows, two
# lengths of the same description, and the appendix carries the internal tag beside them.
CFG_LABEL = {"fixed": "\\methodname{} (full)",
             "refit": "per-scale $\\rho$",
             "vipbank": "\\textsc{Vip} rule, bank kept",
             "plugin": "plug-in estimate",
             "sharp": "sharper weights"}
# Table 2's own labels: one component per row, in the ablation idiom.
ABL_LABEL = {"nobank":  "\\quad w/o warm bank",
             "noshrink": "\\quad w/o shrinkage calibration",
             "plugin":  "\\quad estimator: posterior $\\to$ plug-in",
             "refit":   "\\quad correlation $\\rho$: one $\\to$ per-scale",
             "vipbank": "\\quad allocation rule $\\to$ \\textsc{Vip}'s (bank kept)",
             "sharp":   "\\quad sharper sampling weights",
             "control": "\\quad w/o adaptive allocation (uniform \\textsc{Grpo})"}
CFG_ORDER = ["fixed", "refit", "plugin", "vipbank", "sharp"]

# The control and the published baselines, defined once each and read by BOTH main tables. Before
# 2026-08-29 tab:transfermain named its own arms in TRANSFER_FAITHFUL and took everything else from
# BR/NR's scale blocks, so a row could exist in one main table and not the other with nothing in
# the source saying so. Now both tables walk these lists.
CONTROL_ARMS = {"2B": ["q2bF5", "q2bF6", "q2bF7"], "4B": ["q4bF", "q4bF2"], "8B": ["a8F", "a8Fr"]}
# 2026-08-29 (PI directive, 19:05, DRAFTING pass): THE PUBLISHED ROWS CARRY THE NAME EACH PAPER
# GIVES ITS OWN METHOD AND NOTHING ELSE. Until this pass the labels were DESCRIPTORS with the
# method name embedded in them ("PLR-style prompt replay", "RAG-style retrieval surface",
# "VIP (no bank)", "TRACE (no bank)", "DAPO dynamic sampling", "TSCL learning progress"), which
# the PI rejected: a row label is the method's name, not a one-line summary of what we did to it.
# THE NAMES ARE THE PAPERS' OWN, verified against each paper rather than transcribed from memory:
#   PLR     Prioritized Level Replay                                   (jiang2021plr, title)
#   RAG-MCP Retrieval-Augmented Generation for the Model Context       (gan2025ragmcp, title)
#           Protocol
#   DAPO    Decoupled Clip and Dynamic sAmpling Policy Optimization    (yu2025dapo, abstract)
#   TSCL    Teacher-Student Curriculum Learning                        (matiisen2020tscl, title)
#   TRACE   Tree Rollout Allocation for Contrastive Exploration        (zou2026trace, abstract)
#   VIP     Variance-Informed Predictive allocation strategy           (nguyen2026vip, abstract)
# Set in small caps the way this paper sets every other acronym (\textsc{Plr}, not \textsc{PLR}),
# so a row label and the prose that discusses it are the same string.
# WHAT THE DESCRIPTORS CARRIED IS NOT LOST, IT MOVED INTO THE TABLE NOTE, one sentence each and
# verbatim where the sentence already existed: "our reimplementations inside this paper's own
# allocator and protocol", "the retrieval surface is a \textsc{Bm25} stand-in", and "\textsc{Vip}
# and \textsc{Trace} are run as published, without our warm bank" (MAIN_CITE below). The label
# says WHICH METHOD; the note says WHAT WE RAN.
PUBLISHED_REIMPL = [
    # The 8B PLR row is TWO seeds from 2026-08-24: b8plrr is the second-seed rung at offset 1500,
    # the same construction as a8Fr / a8T3gr / a8Tvipr.
    ("\\quad \\textsc{Plr}",
     {"2B": ["q2bP"], "4B": ["q4bP"], "8B": ["b8plr", "b8plrr"]}),
    ("\\quad \\textsc{Rag}-\\textsc{Mcp}",
     {"2B": ["q2bR"], "4B": ["q4bR"], "8B": ["b8ret"]}),
    ("\\quad \\textsc{Dapo}",
     {"2B": ["q2bD"], "4B": ["q4bD"], "8B": ["dapo"]}),
    ("\\quad \\textsc{Tscl}",
     {"2B": ["q2bLp"], "4B": ["q4bLp"], "8B": ["a8Tlp"]})]
PUBLISHED_ASPUB = [
    # The published configurations, with NONE of our components. 2026-08-29 FOLD: both rows are now
    # COMPLETE AT ALL THREE SCALES. Until this morning VIP was measured at 4B only and the 2B TRACE
    # cell printed IN_FLIGHT ("registered, window incomplete", a third state and not a missing
    # measurement); q2bVf, a8Tvipf and t2bTf closed all three gaps overnight. No cell in either row
    # is a projection and none is a substitution: each is that scale's own faithful arm.
    ("\\quad \\textsc{Vip}",
     {"2B": ["q2bVf"], "4B": ["q4bTvip"], "8B": ["a8Tvipf"]}),
    ("\\quad \\textsc{Trace}",
     {"2B": ["t2bTf"], "4B": ["q4bTf"], "8B": ["t8Tf"]})]

# THE CITATION EACH PUBLISHED ROW CARRIES IN tab:main, KEYED ON THE ROW LABEL ABOVE so that a
# renamed row loses its citation loudly (emit_tables refuses on a key that matches no row) rather
# than silently. It is a SEPARATE dict and not part of the label because only tab:main prints the
# citation: tab:mainsteps, tab:mainabs and tab:transfermain print the same formal NAMES, and a
# per-row \citep in each of them would print the same six references four times.
# WHERE THE TWO "run as published" ROWS SAY SO NOW THAT "(no bank)" HAS LEFT THE LABEL. It was a
# disclosure and not decoration, so it did not go away: MAIN_CITE's last clause ("\textsc{Vip} and
# \textsc{Trace} are run as published, without our warm bank") carries it, word for word, under
# every table that prints those two rows. The label names the method; the note states what we ran.
# ONE citation per row and no row cites twice: matiisen2020tscl is the TSCL paper (graves2017acl
# and schaul2016per stay cited in related work, where the ancestry is discussed).
# 2026-09-02 (PI 03:40): THE UNIFORM CONTROL IS A ROW OF tab:main AND CARRIES ITS CITATION ON THE
# ROW LIKE EVERY OTHER PUBLISHED RULE. It was named only in the provenance note while it was a
# Table 2 row; it is the published rule this paper's whole training half is read against, it
# competes for bold here exactly as the six others do, and a reader of Table 1 alone could not see
# it. The citation is the one the appendix already used for it.
# The control's label in tab:main. It is MAIN's own "uniform \\textsc{Grpo}" plus the one word
# that says what the row is FOR, because tab:main carries no group headers and a role marker
# costs nothing where a header row would cost a line of page 7. The name is the method's own.
T1_CONTROL_LABEL = "\\quad uniform \\textsc{Grpo} (control)"
ROW_CITE = {
    "\\quad \\textsc{Plr}":                "jiang2021plr",
    "\\quad \\textsc{Rag}-\\textsc{Mcp}": "gan2025ragmcp",
    "\\quad \\textsc{Dapo}":               "yu2025dapo",
    "\\quad \\textsc{Tscl}":               "matiisen2020tscl",
    "\\quad \\textsc{Vip}":                "nguyen2026vip",
    "\\quad \\textsc{Trace}":              "zou2026trace",
    T1_CONTROL_LABEL:                      "shao2024deepseekmath",
}

# 2026-08-29, TABLE-ENRICHMENT PASS (PI: "the two main tables are thin"). Two changes to this
# registry, both of them layout payments made deliberately and disclosed in the table's own notes:
#
#  (1) THE CONFIGURATION BLOCK MOVES BACK INTO THIS TABLE, as its own row group between the method
#      row and the control. Those rows are not new arms and not a new statistic: they are exactly
#      CFG_ARMS below, which tab:framecfg has printed since 2026-08-28, read here on tab:main's own
#      best-seed rule instead of as seed means. EVERY ONE OF THEM IS ALREADY A MEMBER OF ITS
#      SCALE'S PINNED FAMILY (checked arm by arm against FAMILY_PIN before this pass: q2bT/q2bT3,
#      q2bTr, q2bV, q4bT/q4bT2, q4bTr/q4bTr2/q4bTr3, q4bV/q4bV2, q4bTpe/q4bTpe2, a8T3g/a8T3gr/
#      a8T3g2, a8Tr, a8Tvip/a8Tvipr, a8Tpe, a8T5k/a8T5kr), so NO FAMILY GROWS, no m moves and no
#      published q changes because of this pass.
#  (2) THE ROW LABELS ARE SHORTENED AND THE CITATIONS LEAVE THE GRID. With four columns per scale
#      the label column can be at most ~60pt wide inside \textwidth at \scriptsize; the citations
#      alone were 64pt of it (measured: 486pt against a 397.5pt \textwidth with them, 422pt
#      without). They are NOT deleted -- they are carried, with each row's full descriptive name,
#      in the table's own note under the rule, so tab:main still cites every baseline it prints.
#      The short names are used in EVERY table that reads this registry (tab:main, tab:mainsteps,
#      tab:mainabs), so no two floats disagree about a method's name -- which is the rule the
#      2026-08-28 pass refused to break by abbreviating in one table only.
# =============================================================================================
# THE METHOD ROW IS NOW CHOSEN BY A RULE, NOT NAMED PER SCALE (PI decision 2026-08-29, review
# P0-8). Until this pass the three arm lists below were written out by hand, which made the row a
# hand pick over a 5x3 grid however carefully the caption described it. The rule is stated here
# once and evaluated from the same families every other row of tab:main is read from:
#
#     the method row at a scale is the configuration of CFG_ARMS with the HIGHEST BEST-SEED
#     WINDOW MEAN at that scale, over the configurations that have a rankable cell there, minus
#     any configuration excluded for a STATED, MEASURED transfer collapse (METHOD_EXCLUDE).
#
# It is still a selection made after the measurements existed -- the rule ranks on the same
# best-seed statistic tab:main prints -- and the caption still says so in those words. What
# changes is that the selection is reproducible from this file rather than transcribed, so it
# cannot silently disagree with the configuration block printed directly beneath it.
#
# WHAT THE RULE RETURNS TODAY, printed by --emit-tables' audit block at every run:
#     2B  fixed-rho    +2.59 (q2bT)     [scale-refit +2.67 leads but is EXCLUDED, below]
#     4B  scale-refit  +5.21 (q4bTr)
#     8B  sharpened    +4.70 (a8T5k)    [fixed-rho +4.58, scale-refit +4.25, plug-in +3.84]
# The 8B row therefore CHANGES CONFIGURATION this pass, from fixed-rho to sharpened (tau=0.3).
# It is not a new arm and not a new number: a8T5k has been a row of tab:ablation since
# 2026-08-17 and of tab:main's configuration block since 2026-08-29, at these same values.
# THE TRANSFER TABLES ARE NOT AFFECTED and do not follow this rule: tab:transfermain's method
# row is fixed-rho at every scale by its own long-standing convention (every paired transfer
# contrast in this paper is built on fixed-rho), which that table's caption states.
#
# THE ONE HAND-ENTERED PART OF THE RULE, and it is hand-entered so that it has to be justified in
# writing: a configuration is excluded from selection only for a transfer collapse that is
# measured and printed. The value below is in tab:transfermain and tab:bfcl2b.
METHOD_EXCLUDE = {
    "2B": {"refit": "9.00 on \\textsc{Bfcl} against an untrained 15.12, a collapse 6.1 pp below"
                    " the base policy"},
}

# =============================================================================================
# 2026-09-02 (PI 02:40): ONE SELECTION RULE, AND THE RULE IS THAT THERE IS NO SELECTION.
# tab:main's method row is THE REGISTERED FULL RECIPE at every scale. Flip this to "selected" to
# restore the per-scale best-configuration row exactly as it stood in _fold6c; nothing else in
# this file has to move, because every consumer of the method row reads METHOD_SPEC.
#
# WHY, AND IT IS THE PAPER'S OWN CLAUSE APPLIED UNIFORMLY RATHER THAN A NEW PREFERENCE. The
# selection rule excluded a configuration for a MEASURED TRANSFER COLLAPSE (METHOD_EXCLUDE
# above), and it was stated once and honoured once. Applied at every scale it disqualifies the
# same configuration the 2B clause disqualified and the 8B row the ladder headed:
#   2B  per-scale rho  BFCL 9.00 against an untrained 15.12          -6.1 pp   EXCLUDED (as before)
#   4B  per-scale rho  NESTFUL 11.34 against an untrained 30.79     -19.45 pp  EXCLUDED
#   8B  sharper weights  BFCL 35.2 v 35.9 and NESTFUL 37.29 v 38.26  below base on BOTH  EXCLUDED
# What the clause returns at all three scales once it is applied at all three is the full recipe,
# which is why the row can now be STATED rather than selected: the method row is the registered
# configuration, the other configurations are reported as measured in the appendix ladder
# (tab:ladder and tab:framecfg), and no cell of this paper is a maximum over configurations.
# THE COST IS PRINTED AND IS NOT SMALL: at 4B the row goes +5.2 (q = 0.001) to +3.6 (q = 0.130)
# and LOSES ITS STAR. At 8B it goes +4.7 (q = 0.002) to +4.6 (q < 0.001) and is stronger, not
# weaker. At 2B nothing moves, because the rule already returned the full recipe there.
TABLE1_ROW = "full"          # "full" = the registered recipe; "selected" = the retired rule

# =============================================================================================
# 2026-09-02 23:05 (PI RULING, taken after both readings were computed and shown): TABLE 1's
# METHOD ROW IS READ AT ITS OWN BEST CHECKPOINT AND EVERY OTHER ROW STAYS ON THE WINDOW MEAN.
#
# WHAT THE FLAG DOES. When it is True the \methodname{} row of tab:main -- and only that row --
# is read at the checkpoint of {15,20,25,30} where its OWN SOLVE RATE is highest, and all three
# of its cells (solve, invalid, turns) are read at that same checkpoint, so the row is internally
# one checkpoint rather than one column of one convention beside two of another. Every other row,
# and the untrained base row (which has no checkpoint at all), is unchanged.
#
# THE STEP IS CHOSEN HERE, BY CODE, AND PRINTED INTO THE FLOAT'S NOTE AND INTO THE AUDIT BLOCK.
# No step number is typed anywhere in the paper: if a later checkpoint lands, or a cell is
# rescored, the selected step and every number the note quotes move at the next emission.
#
# WHAT MAKES IT SAFE, AND IT IS STATED IN THE NOTE RATHER THAN LEFT HERE:
#  (1) The ordering does not depend on the asymmetry. Read EVERY row at its own best checkpoint
#      and the method still leads at all three scales; the emitter computes that symmetric
#      reading and writes its three comparisons into the note.
#  (2) The window mean stays the pre-registered readout, tab:mainsteps still prints all four
#      steps and the window mean for every row of this float, and the note points there.
#  (3) tab:main CARRIES NO STARS AND NO p, so no significance statement is read at a selected
#      step. tab:configs, tab:mainfull, tab:mainsteps, both transfer tables and every Holm
#      family stay on the window statistic and are untouched by this flag.
#  (4) The selection is worth something and the emitter measures it rather than asserting it is
#      small: the mean uplift of best-of-four-correlated-steps over the window mean, taken over
#      every competing row of this float, is printed in the audit block and quoted in
#      app:tableconv beside the best-of-n seed figure.
T1_METHOD_BEST_STEP = True
T1_METHOD_LABEL = "\\quad \\methodname{}"


def select_method_cfg(model, fam=None):
    """The configuration key tab:main's method row carries at `model`, by the rule above."""
    f = fam if fam is not None else full_family(model)[0]
    best, bestv = None, None
    for k in CFG_ORDER:
        if k in METHOD_EXCLUDE.get(model, {}):
            continue
        arms = CFG_ARMS[k][model]
        if not isinstance(arms, list):
            continue                      # NOT_RUN / SAME_FIXED carry no rankable cell
        a = agg(f, arms, best=True)
        if a is None:
            continue
        # Ties break on the lower q and then on the configuration key, so the choice is
        # deterministic across re-emissions -- the same convention best_seed() uses for seeds.
        key = (a["mean"], -a["q"], k)
        if bestv is None or key > bestv:
            best, bestv = k, key
    if best is None:
        raise SystemExit("select_method_cfg(%r): no configuration has a rankable window cell "
                         "-- tab:main cannot have a method row" % model)
    return best


class _MethodSpec(dict):
    """tab:main's method row, resolved per scale on first access by select_method_cfg().

    A dict rather than a function because every consumer of MAIN calls `spec.get(model)`; a lazy
    dict keeps that interface and keeps the family read out of module import, which several
    scripts do without ever emitting a table.
    """

    def get(self, model, default=None):
        if model not in self:
            key = "fixed" if TABLE1_ROW == "full" else select_method_cfg(model)
            dict.__setitem__(self, model, CFG_ARMS[key][model])
        return dict.get(self, model, default)

    def __getitem__(self, model):
        return self.get(model)


METHOD_SPEC = _MethodSpec()

MAIN = [("@HDR", "\\textbf{Method}", None),
        # ONE ROW, CHOSEN BY THE RULE ABOVE, and the caption states which configuration it is at
        # each scale. Every configuration's full record, at every scale, is in the configuration
        # block directly beneath and in tab:framecfg; no cell here mixes configurations.
        ("", "\\quad \\methodname{}", METHOD_SPEC),
        # The configuration block, 2026-08-29. Same statistics, same anchors, same pinned
        # families, same best-seed rule as every other row here; only the arm sets differ, and
        # they are CFG_ARMS, the dict tab:framecfg reads. The method row above remains THE single
        # selected row and the caption says which configuration it is at each scale.
        ("@HDR", "\\textbf{This framework, other configurations}", None),
        ] + [("", "\\quad " + CFG_LABEL[k], CFG_ARMS[k]) for k in CFG_ORDER] + [
        ("@HDR", "\\textbf{Control}", None),
        ("", "\\quad uniform \\textsc{Grpo}", CONTROL_ARMS),
        ("@HDR", "\\textbf{Published baselines}", None),
        # 2026-08-29: the arm sets moved up to PUBLISHED_REIMPL / PUBLISHED_ASPUB so that
        # tab:transfermain reads the SAME lists. The ORDER here is unchanged -- our
        # reimplementations of the two prompt-level rules, then the two allocators run as
        # published, then the two remaining reimplementations -- because that is the order the
        # results text walks the block in. tab:transfermain groups the as-published pair into
        # their own block instead, which is what that table has always done.
        # 8B VIP is two seeds (a8Tvipr is the clean rerun of the arm carrying a mid-run optimizer
        # restart) and 8B PLR is two (b8plrr at offset 1500); both are one row here under the
        # best-seed rule, with every seed in tab:perseed. ONE citation on the TSCL row, not two:
        # matiisen2020tscl is the TSCL paper and graves2017acl stays cited in related work.
        ] + [("", lab, spec) for lab, spec in
             (PUBLISHED_REIMPL[:2] + PUBLISHED_ASPUB + PUBLISHED_REIMPL[2:])]

# ---------------------------------------------------------------------------------------------
# THE TWO MAIN-TEXT TABLES' OWN ROW LISTS (2026-08-29, PI directives 19:25 / 19:55 / 20:05).
# MAIN above is UNCHANGED and is still the full row set: it is what tab:mainfull, tab:mainsteps,
# tab:mainabs, the Figure-1 outcome macros and the stdout audit walk, in the order they have
# always walked it. NOTHING LEFT THE PAPER IN THIS PASS -- the wide four-statistics grid that WAS
# tab:main is emitted unchanged, cell for cell, as tab:mainfull in the appendix. What changed is
# which rows the two MAIN-TEXT floats print and which columns they print for them:
#
#   tab:main    (Table 1) the untrained base policy, \methodname{} and the six published rules,
#                         at three formal in-distribution metrics per scale and nothing else.
#   tab:configs (Table 2) this framework's other configurations and the uniform control, at the
#                         window solve rate and the two statistics of record.
#
# THE BASE POLICY IS A ROW OF TABLE 1 AND OF NOTHING ELSE. It is not an allocation rule and has
# no window, no effect and no q, so it cannot be a row of tab:mainfull / tab:mainsteps /
# tab:mainabs without printing three empty cells in every one of them.
BASE_ROW = "@BASE"
# ORDER, STATED SO IT IS NOT MISTAKEN FOR A RANKING: the four rules we reimplemented in the order
# the results text walks them (PLR, RAG-MCP, DAPO, TSCL), then the two allocators run as
# published (TRACE, VIP). It is NOT by year and NOT by any measured value; MAIN's own published
# block keeps the order it has always had, which is why tab:mainfull's rows are in that order and
# these are in this one.
# 2026-09-02 (PI 03:40): THE UNIFORM CONTROL IS THE LAST ROW. It is a published rule
# \citep{shao2024deepseekmath} and it is read here on the same window, the same anchors, the same
# pinned families and the same best-seed rule as every other row -- CONTROL_ARMS, the dict
# tab:configs' control row reads, so the two floats cannot name one arm set and print another. It
# COMPETES FOR BOLD: it is not a reference row, and the base row remains the only row excluded.
# It goes AFTER the published block rather than under the base row because the base row and the
# control are not two references: the base row is what every effect in the paper is measured
# against, and the control is a rule that is being compared.
T1ROWS = ([("", "\\quad untrained base policy", BASE_ROW),
           ("", "\\quad \\methodname{}", METHOD_SPEC)]
          + [("", lab, spec) for lab, spec in PUBLISHED_REIMPL + PUBLISHED_ASPUB[::-1]]
          + [("", T1_CONTROL_LABEL, CONTROL_ARMS)])
# Table 2's rows: the framework's other configurations in CFG_ORDER, then the control. Same
# dicts, same best-seed rule and same pinned families as every other float that reads them.
# --- TABLE 2 IS THE ABLATION STUDY (PI 21:55: "where is ablation study? is it table 2? then
# convert it to ablation study style") -------------------------------------------------------
# It was a list of "configurations", which is the same rows read as a catalogue rather than as an
# experiment. Read as an ablation it needs three things it did not have: the FULL METHOD FIRST as
# the reference, a Delta AGAINST THAT ROW so the cost of each removal is the number the reader
# sees, and the two genuine component REMOVALS the fleet already ran.
#
# THE TWO ROWS THAT ARE NEW HERE AND WHY THEY COST NOTHING. q2bTnw/q4bTnw (the full method minus
# --warm-bank) and q2bTiid (the full method with --warm-shrink none instead of group) are ALREADY
# MEMBERS of their scales' pinned Holm families -- checked arm by arm against FAMILY_PIN before
# this pass -- so printing them here grows no family, moves no m and changes no published q. They
# are the paper's two most load-bearing ablations (the bank is priced at -2.0pp at 2B in
# tab:coadapt2b) and an ablation table without them would be an odd thing to publish.
# Their coverage was real and disclosed while it was partial: the bank ablation had no 8B cell and
# the shrinkage ablation only a 2B one, and those cells printed the coverage dash rather than a
# substitute. Both are complete at all three scales as of 2026-09-02 and neither prints a dash.
# 2026-08-31: both removal rows were pre-registered to all three scales on 2026-08-29 22:25 as
# single-seed arms differing from their scale's full-method recipe by ONE launch flag, and the
# diffs were verified against slurm/supervisor.sh rather than against the registry's names.
#   nobank    a8Tnw = a8T3g2 minus --warm-bank. COMPLETE at all three scales 2026-08-31 07:00.
#   noshrink  q4bTiid = q4bT2 and a8Tiid = a8T3g2, each with --warm-shrink group -> none.
#
# a8Tiid'S INFRASTRUCTURE DEVIATIONS, AND THEY ARE SEGMENTED RATHER THAN UNIFORM. THIS MUST BE
# PRINTED WHEN THE ROW ENTERS (2026-09-01, final; every claim checked against
# slurm/verl_awm_train.sh and the run's own relaunch record rather than taken on trust). The 8B
# shrinkage ablation would not fit: two OOMs at update_actor (util 0.35, then 0.29), three
# engine-init failures at util 0.25, then a crash-loop in cycle 3. Final launch config is TP=2 on
# the A100 pair, MAXTOK 28672 (the recipe), resuming from checkpoint 20. EXACTLY TWO THINGS ARE
# DISCLOSABLE DEVIATIONS, and the list is deliberately short because two candidates were struck:
#   1. TP=2 THROUGHOUT, where its siblings a8Tnw and a8T3g2 ran TP=1 on the H200.
#   2. OPTIMIZER STEPS 11-20 TRAINED UNDER 20480 PACKING. MAXTOK is the recipe's 28672 for steps
#      1-10 and 21-30; the cut was made 2026-08-31 15:43 to fit the update peak and reverted
#      2026-09-01 06:17, forced rather than chosen, because 20480 breaks verl's own
#      max_token_len >= max_seq_len assert as soon as a batch reaches 22.4k tokens, which is what
#      crash-looped cycle 3.
# STRUCK FROM THIS NOTE, and each for a reason worth keeping so nobody re-adds it:
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NEVER TRAINED A STEP. It was tried on
#     2026-09-01 06:17 and removed after failed boots, because vLLM's cumem engine refuses to
#     initialize under it. Disclosing a setting that produced no gradient would be a false
#     statement in the paper, not a conservative one.
#   * GPU_UTIL (0.28 here) is a SERVING parameter, set per card from measured free memory for
#     every arm in this paper. It is not a recipe deviation and naming it as one would imply this
#     arm was configured differently when it was configured the same way as everything else.
# WHY THE PACKING WINDOW IS A DISCLOSURE AND NOT A FOOTNOTE. MAXTOK is
# actor_rollout_ref.actor.ppo_max_token_len_per_gpu and the trainer sets use_dynamic_bsz=True
# beside it (verl_awm_train.sh:171-173), so it governs how ONE MINI-BATCH IS PACKED INTO
# MICRO-BATCHES for the actor update. BSZ=32, MINIBSZ=8, RESP_LEN=12288 and k=5 never moved, so
# the optimizer batch, the per-step generation budget and every recipe flag are the parent's, and
# the ONE recipe difference from a8T3g2 remains --warm-shrink group -> none. What packing and TP
# do change is gradient-accumulation and reduction ORDER, so this arm is not the bitwise run the
# recipe on one card would have produced, and for ten of its thirty steps it was not even the same
# packing as its own other twenty. Say that plainly; do not round it to "MAXTOK matches the
# recipe".
# EVERY OTHER ARM THE PAPER PRINTS RUNS MAXTOK=28672 AT TP=1 (audited across every recoverable
# launch record, 2026-08-31), so app:trainconf's "shares that configuration to the digit ... with
# one exception" sentence gains a SECOND named exception beside DAPO's 320-task pool the moment
# this row is printed.
ABL_ARMS = {
    "nobank":  {"2B": ["q2bTnw"], "4B": ["q4bTnw"], "8B": ["a8Tnw"]},
    "noshrink": {"2B": ["q2bTiid"], "4B": ["q4bTiid"], "8B": ["a8Tiid"]},
}
# ORDER, and it is the ablation idiom's order, not a ranking: the full method, then the removals,
# then the replacements, then the allocation itself removed.
# NO UNMEASURED CELL IN THIS FLOAT (PI 2026-08-29 22:15: "never leave not measurable in the
# draft"). Table 2 prints ONLY rows measured at all three scales, so it carries no coverage dash
# and no "= full" placeholder. THE ROWS HELD OUT ARE NOT DELETED -- they are in the appendix
# ablation ladder (tab:ladder, tab:ladder2b) with the cells they do have, and each comes back
# here by restoring one line the moment its missing cell exists:
#   sharp     8B only, and at 2B/4B it IS the full method, so its cells there were never an
#             ablation at all. It stays in the appendix as the 8B variant it is.
# 2026-08-31 (FOLD v): TWO of the four rows held out on 2026-08-29 come back, each because its
# missing cell now exists and for no other reason. w/o warm bank was waiting on a8Tnw (banked
# 07:00 today) and the plug-in row on q2bTpe (banked 08-30). w/o shrinkage calibration had
# q4bTiid by then but not a8Tiid, so its line stayed commented out rather than printing a 4B/2B
# row with an 8B dash.
# 2026-09-02 (FOLD vi): the shrinkage row comes back, and it is the last of the four. a8Tiid
# banked its fourth window cell 00:18 today (590 rows at each of 15/20/25/30, verified by row
# count before this line was restored), so all three of that row's cells are measured and the
# rule the float was built under -- a row prints here only when every one of its three cells is
# measured -- admits it without being bent. The one row still held out is `sharp`, which is not
# an ablation at 2B/4B at all.
ABLROWS = ([("@FULL", "\\methodname{} (full)", CFG_ARMS["fixed"]),
            ("", ABL_LABEL["nobank"], ABL_ARMS["nobank"]),
            ("", ABL_LABEL["noshrink"], ABL_ARMS["noshrink"]),
            ("", ABL_LABEL["plugin"], CFG_ARMS["plugin"]),
            ("", ABL_LABEL["refit"], CFG_ARMS["refit"]),
            ("", ABL_LABEL["vipbank"], CFG_ARMS["vipbank"]),
            # ("", ABL_LABEL["sharp"], CFG_ARMS["sharp"]),         # 8B-only variant
            ("", ABL_LABEL["control"], CONTROL_ARMS)])
CFGROWS = ABLROWS

# The PROVENANCE note under tab:main. It is a provenance claim and not a formatting one: it says
# that every reimplemented row is OUR code inside this paper's own allocator and protocol.
# 2026-08-29 (PI directive, DRAFTING pass): THE CITATION ENUMERATION IS GONE FROM THIS SENTENCE
# because it is now redundant -- each of those six references is printed on the row it belongs to
# (ROW_CITE above), and a note that repeats them makes the reader match two lists. NOTHING ELSE
# LEFT IT. The clause naming the control and its reference stays (that row carries no in-grid
# citation), the BM25 stand-in stays (it is a disclosure about what our retrieval row IS, not a
# reference), and the "run as published, without our warm bank" clause stays. The sentence as it
# stood until this pass is archived VERBATIM in Appendix~\ref{app:capfull} under tab:main, beside
# the caption clauses the earlier passes cut, so the version this replaces is still in the paper.
MAIN_CITE = ("Rows are our reimplementations inside this paper's own allocator and protocol, not"
             " the cited authors' code: uniform \\textsc{Grpo} \\citep{shao2024deepseekmath} is the"
             " control, and the \\textsc{Rag}-\\textsc{Mcp} row is a \\textsc{Bm25} stand-in for the"
             " retrieval surface; \\textsc{Vip} and"
             " \\textsc{Trace} are run as published, without our warm bank.")

# --- Table: configurations of the framework, moved out of tab:main 2026-08-28 -------------------
# NOTHING IS DELETED BY THE MAIN-TABLE RESTRUCTURE. tab:main now carries one method row (the
# scale-adaptive configuration), one control and the published baselines; every OTHER configuration
# of this framework -- including the fixed-rho posterior that headed tab:main until this pass, and
# the VIP rule running on our warm bank, which Proposition 2 makes a configuration of ours rather
# than a rival objective -- lives here, at the same statistics and read against the same anchors.
# 2026-08-29: this list is now GENERATED from CFG_ARMS/CFG_LABEL, the same dict tab:main's
# configuration block reads, in the same order, under the same names. Before this pass the two
# floats carried the arm sets twice; a configuration admitted to one and not the other would have
# been invisible. tab:framecfg is NOT retired by the enrichment pass and is not a duplicate: it
# carries these configurations as SEED MEANS with the mean-turns column tab:main has no width for,
# which is the fuller record the main table's per-scale selection is checked against.
FRAMECFG = [("", "\\quad " + CFG_LABEL[k], CFG_ARMS[k]) for k in CFG_ORDER]

# Blocks top to bottom. Smallest policy first, so a reader walks up the scale axis and meets the
# 4B non-separation before the 8B result rather than after it.
MAIN_SCALES = ["2B", "4B", "8B"]

#  (3) ONE CHECKPOINT PER ROW instead of the four-column step list.
#
# THE SELECTION RULE IS A FIXED STEP, IDENTICAL FOR EVERY ROW, AND IT IS NOT A SELECTION AT ALL.
# The two rules the directive offered were both checked against the data first and both fail on
# coverage, which is a fact about the fleet and not a preference:
#
#   * "each arm's window step selected by validation solve-rate" -- NOT APPLICABLE. Three arms in
#     this table log ZERO held-out validation episodes: a8F (the control's own first seed;
#     VAL_BEFORE was never set on it, a gap already disclosed at tab:termination), b8plr, and
#     a8Tvip. A rule that cannot be evaluated on the control row is not a uniform rule.
#   * "step 30 fixed for all" -- MISREPRESENTS. dapo and a8Tlp have no step-30 cell (their windows
#     are 15/20/25), so two published baselines would print "--" in the only effect column and read
#     as uncovered when they are in fact measured at three steps each.
#
#   * STEP 15 fixed for all -- ADOPTED. It is the ONLY step at which every row of this table has a
#     cell, and that is a coverage fact this paper recorded before this directive existed (the
#     Protocol section's "at step 15 every arm in the family now has a cell, so m=35 there too").
#     No arm is read at a step chosen for it, no row can be flattered by the rule, and the rule is
#     stated in the caption.
#
# WHAT DID NOT CHANGE, and this is the integrity point: the STATISTIC OF RECORD is still the
# pre-registered four-step window mean and its Holm q, both of which are computed over the same
# four steps for every arm and neither of which is a single-checkpoint reading. The step-15 column
# is descriptive and the caption says so, exactly as the Results prose already says that reading a
# single step "is exactly the maximum-over-checkpoints procedure the matched window exists to
# prevent". The four per-step cells this table used to print are not deleted from the paper: they
# move, digit for digit, to mainsteps.tex in the appendix.
MAIN_STEP = 15

# --- Table: our own measured negatives, moved to the appendix ---------------------------------
# 2026-09-03 _fold7h: THIS FLOAT GAINS THE TWO TEMPERATURE ARMS, and gains a model column with
# them, because they are 2B and 4B arms and every row above is 8B. A row is read against ITS OWN
# scale's base anchor and corrected inside ITS OWN scale's pinned family -- the same discipline
# tab:ladder2b's 4B rung is read under -- so the third entry of each tuple is the model and the
# emitter looks the family up rather than sharing one. Both arms are measured negatives in the
# strict sense this table means: each is one flag away from a configuration this paper prints,
# each is below the configuration it came from, and neither is a shipped recipe at any scale.
#
# 2026-09-06 _fold12: THE FLOAT GAINS FIVE MORE ROWS AND A SECOND BLOCK HEADER, and with them it
# stops being a table of negatives only. Three of the five ARE negatives in the strict sense above
# (the two interior dial points at 4B and the 2B variable-k rung); two ARE NOT, and saying so is
# the point of the block headers and of the caption:
#   * a8Trt, the dial's 8B rung, is ABOVE its parent on all three axes. It is here because it is
#     the same one-flag construction as the rows around it, not because it failed; a dial printed
#     at the scale where it falls and hidden at the scale where it rises is not a dial.
#   * q4bK, the 4B variable-k rung, is above its parent AND above the shipped recipe's best seed
#     in-window, with BFCL above base and NESTFUL 1.6 under it.
# The block headers name what each block IS rather than what it cost, and the caption carries the
# reading. Every row is still one flag from a configuration this paper prints, still read against
# its own scale's anchor, and still corrected inside its own scale's pinned family.
NEGATIVES = [("", "adaptive surface (v1)", ["a8A"], "8B"),
             ("", "\\textsc{Elsa}", ["a8A3"], "8B"),
             ("", "\\textsc{Accord} (certified dense shaping)", ["a8C"], "8B"),
             ("", "group size $k{=}2$ (falsified axis)", ["a8T4"], "8B"),
             ("@HDR", "\\emph{The sampling temperature moved on the per-scale $\\rho$"
                      " configuration}", None, None),
             ("", "\\quad at $2$B, $\\tau{=}1.12$", ["q2bTrt"], "2B"),
             ("", "\\quad at $4$B, $\\tau{=}0.60$", ["q4bTm"], "4B"),
             ("", "\\quad at $4$B, $\\tau{=}0.85$", ["q4bTm8"], "4B"),
             ("", "\\quad at $4$B, $\\tau{=}1.12$", ["q4bTrt"], "4B"),
             ("", "\\quad at $8$B, $\\tau{=}1.12$", ["a8Trt"], "8B"),
             ("@HDR", "\\emph{The group size freed across tasks under one budget}", None, None),
             ("", "\\quad at $2$B", ["q2bK2"], "2B"),
             ("", "\\quad at $4$B", ["q4bK"], "4B")]

# THE SAMPLING-TEMPERATURE DIAL, one entry per measured point per scale, lowest tau first. THE
# LOWEST POINT OF EACH SCALE IS ALREADY PRINTED ELSEWHERE -- it is the "per-scale rho" row of
# tab:configs, tab:mainfull and tab:transfermain -- so this registry names every point and the
# emitter computes the curve rather than letting a number be typed twice. tau is the only token
# that differs between consecutive points of one scale (checked at launch, launch command by
# launch command), and every member ran at seed offset 500, so each contrast is one flag at one
# offset.
#
# THE REGISTERED VALUE IS THE FIRST ENTRY OF EACH LIST AND IT IS NOT THE SAME NUMBER AT EVERY
# SCALE: tau = 0.3 at 2B and 4B, tau = 0.5 at 8B, which is what Algorithm 1 prints and what the
# configuration grid says row by row. A "dial" quoted as one curve across scales would be three
# different contrasts averaged, so the emitter keeps the scales apart and the note names each
# scale's own starting point.
TEMP_DIAL = {"2B": [(0.30, "q2bTr"), (1.12, "q2bTrt")],
             "4B": [(0.30, "q4bTr"), (0.60, "q4bTm"), (0.85, "q4bTm8"), (1.12, "q4bTrt")],
             "8B": [(0.50, "a8Tr"), (1.12, "a8Trt")]}

# THE UNTRAINED ANCHOR, PER BENCHMARK, because the two benchmarks scored the 8B base policy into
# differently named run directories (BFCL's is run_base and NESTFUL's is run_base8b -- see
# bfcl_records.SCALES against nestful_records.SCALES) and a single map would silently read one
# scale's anchor off a directory that does not exist. The same fact is already recorded at
# DUR_BFCL_BASE and at the transfer emitter's own base registry; this is the third and last use.
BFCL_BASE = {"2B": "base2b", "4B": "base4b", "8B": "base"}
NEST_BASE = {"2B": "base2b", "4B": "base4b", "8B": "base8b"}

# THE PRE-REGISTERED SHIP TEST for a 4B dial point, written down before any of the four ran: a
# point ships only if it beats the shipped 4B recipe's window AND is at or above the untrained 4B
# policy on BOTH transfer benchmarks. The emitter evaluates it rather than the prose asserting it.
SHIP_TEST_4B = dict(win_ref="q4bT2", bases=("base4b", "base4b"))

# THE VARIABLE-k PILOTS. One flag (VARK=1) from their fixed-k parents, same seed offset, and the
# per-step rollout total pinned to the uniform k=5 spend so the pilot is budget-matched rather
# than budget-freed. The per-cycle histogram of k is READ OFF THE RUN LOG rather than transcribed:
# a histogram typed into a caption is a number with no record behind it, and this one carries the
# paper's answer to whether a calibrated level, freed to act, acts.
VARK_PILOTS = [("2B", "q2bT", "q2bK2"), ("4B", "q4bT", "q4bK")]
VARK_LOG = R + "/logs/rl_%s.log"
_VARK_RE = re.compile(r"\[triage\] vark: (\d+) rows in \d+ blocks of \d+; "
                      r"k histogram \{([^}]*)\}; sum k = (\d+)")


def vark_cycles(arm):
    """[(n_rows, {k: count}, sum_k)] per training cycle, from that arm's own run log.

    The log line is written by the allocator at the moment it builds the cycle's parquet, so it is
    the allocation itself and not a reconstruction of it. Cycles are returned in the order they
    were logged.
    """
    out = []
    try:
        text = open(VARK_LOG % arm, errors="ignore").read()
    except Exception:
        return out
    for m in _VARK_RE.finditer(text):
        hist = {}
        for part in m.group(2).split(","):
            k, v = part.split(":")
            hist[int(k.strip())] = int(v.strip())
        out.append((int(m.group(1)), hist, int(m.group(3))))
    return out

# --- Table: the ablation study, as component attribution --------------------------------------
# Restructured 2026-08-17 from a chronological "configuration ladder". The ladder recorded the
# order we searched in, which is our history and not the reader's question; this asks the reader's
# question instead -- what does each component contribute -- by naming the component that changed
# and giving the delta against the full method. The blocks are honest about isolation: the first
# block changes exactly one line of the arm's configuration, the second removes several at once,
# and the header says which is which rather than letting every row read as a clean ablation.
FULL_METHOD = ["a8T3g", "a8T3gr", "a8T3g2"]
ABLATION = [
    ("@FULL", "\\textbf{Full method} (\\methodname{})", FULL_METHOD),
    ("@HDR", "\\textbf{Exactly one line of the configuration changed}", None),
    ("", "\\quad estimator: posterior $\\to$ \\textsc{Trace}-style plug-in", ["a8Tpe"]),
    ("", "\\quad discount: $\\gamma{=}1 \\to \\gamma{=}0.9$", ["a8T3"]),
    ("", "\\quad warm bank: task identities shuffled", ["a8T6d"]),
    ("", "\\quad training surface: fixed $\\to$ adaptive control", ["a8T5"]),
    ("@HDR", "\\textbf{Components removed}", None),
    ("", "\\quad $-$ certified exclusion, $-$ group shrinkage", ["a8T2g1"]),
    ("", "\\quad \\ldots\\ and $\\gamma{=}0.9$", ["a8T2", "a8T2r"]),
    ("", "\\quad $-$ sharpening (warm start only, $\\tau{=}1$)", ["a8T2w"]),
    ("", "\\quad $-$ warm-started prior (cold start, $\\epsilon{=}0.1$)", ["a8T"]),
    ("", "\\quad \\ldots\\ cold start, $\\epsilon{=}0$", ["a8Te0"]),
    ("@HDR", "\\textbf{Weight rule, sampling and composition}", None),
    ("", "\\quad posterior weight $\\to$ difficulty-band threshold", ["a8T2n", "a8T2nr"]),
    ("", "\\quad sharpened sampling, $\\tau{=}0.3$", ["a8T5k", "a8T5kr"]),
    ("", "\\quad $\\times$ retrieval-shaped surface", ["a8T2b"]),
    ("", "\\quad group size $k{=}5 \\to k{=}2$ (falsified)", ["a8T4"])]

# --- Table: the SAME two structural ablations, replicated at 2B -------------------------------
# Added 2026-08-18 when both 2B cells completed their four-step window. FULL_METHOD_2B is q2bT
# ALONE, not q2bT+q2bT3: both ablations are pinned to seed-offset 500, exactly q2bT's offset, per
# the pairing rule PLAN_TRIAGE.md states for this family ("an ablation on a fresh offset would
# confound the removed component with the surface draw") -- q2bT3 is a different offset (2500) and
# reading either ablation against it would reintroduce that confound. This is therefore a smaller,
# single-seed table by construction, not by omission; every row is still in the 2B family's own
# Holm correction (full_family("2B")), so the multiplicity cost of running it is charged same as
# every other 2B arm.
FULL_METHOD_2B = ["q2bT"]
# 2026-09-01 _fold6: THE LADDER GAINS ITS MIRROR IMAGE. Every row above removes a component from
# our method; the two rows below ADD one of them -- our warm bank, and only that -- to a published
# rule, which is the only construction that prices the bank separately from the allocation rule it
# is bundled with. t2bTbk is t2bTf with --warm-bank at its default and no other change, one seed at
# the parent's own offset (500), so the pair differs by that one flag and by nothing else. Both are
# members of the 2B Holm family; neither is a row of Table 1, which keeps the published rules as
# their papers run them.
ABLATION2B = [
    ("@FULL", "\\textbf{Full method} (\\methodname{})", FULL_METHOD_2B),
    ("", "\\quad $-$ warm bank", ["q2bTnw"]),
    ("", "\\quad $-$ calibration (independent shrinkage)", ["q2bTiid"]),
    ("@HDR", "\\emph{The same component added to a published rule}", None),
    ("", "\\quad \\textsc{Trace}, as published (no bank)", ["t2bTf"]),
    ("", "\\quad \\textsc{Trace} $+$ our warm bank", ["t2bTbk"])]

# 2026-09-02 _fold7c: THE BANK LADDER GETS ITS SECOND SCALE, AND THAT IS THE WHOLE POINT OF IT.
# The 2B rung above is the reviewer's construction and at 2B it goes against us, so a single-scale
# answer to a confound is not an answer. t4bTbk is q4bTf with --warm-bank at its default and no
# other flag changed, one seed at the same offset 500 its parent ran -- the exact construction
# t2bTbk is at 2B, one scale up.
#
# THE REFERENCE THE DELTA IS READ AGAINST IS OFFSET-MATCHED, which is this table's own pairing
# rule and not a choice made after the numbers existed. At 2B the full method's offset-500 seed
# (q2bT) also happens to be its best seed, so the rule and tab:main's printed row agree there and
# the question never arose. At 4B they do NOT agree: the offset-500 seed is +2.5 and the other
# seed, the one tab:main prints as the 4B \methodname{} row, is +3.6. The delta below is against
# the offset-matched seed, because a delta against the other one would confound the warm bank with
# a different surface draw -- the same reason every removal row in this table is pinned to an
# offset. BOTH readings are printed: the note carries the best seed's window, so a reader sees
# that this rung sits BETWEEN the full method's two 4B seeds and can check either comparison.
LADDER_FULL_4B = ["q4bT"]
LADDER_BANK_4B = ("\\quad \\textsc{Trace} $+$ our warm bank, at $4$B", ["t4bTbk"])

# 2026-09-04 _fold10: THE LADDER REACHES THE HEADLINE SCALE, AND THERE IT REVERSES.
# t8Tbk is t8Tf with --warm-bank at its default and no other flag changed, one seed at the same
# offset 500 its parent ran -- the identical construction at 8B. With the two rungs above it the
# reviewer's bank-confound question is now answered by measurement at every scale this paper
# trains, which is the only form of answer a confound accepts.
#
# THE OFFSET-MATCHED REFERENCE AT 8B IS a8T3g AND NOT THE ROW Table~1 PRINTS. a8T3g carries seed
# offset 500 (supervisor.sh), which is t8Tf's and t8Tbk's; a8T3g2, the seed tab:main prints, is
# offset 2500. The delta below is against the offset-matched seed for the same reason the 4B rung
# is -- a delta against a different surface draw would confound the warm bank with the draw -- and
# the note carries the reading against tab:main's seed as well, so both are printed.
LADDER_FULL_8B = ["a8T3g"]
LADDER_BANK_8B = ("\\quad \\textsc{Trace} $+$ our warm bank, at $8$B", ["t8Tbk"])

# --- Table: pool-matched robustness, on DAPO's own training pool -------------------------------
# Added 2026-08-18 (PLAN_TRIAGE.md 2026-08-18 01:55 amendment, executed 16:50). Table~\ref{tab:mech}
# discloses that \textsc{Dapo} trains on a 320-task subset of the pool, denser than every other row
# there sees, as a confound stated rather than fixed. q2bTs/q2bFs are TRIAGE and uniform run on that
# SAME pool_big320, converting the disclosed confound into a designed, same-pool comparison. Single
# seed each, by design (P1/P2 readout, not a replication claim); no scarcity claim is made (PLAN
# 2026-08-18 01:55: budget=256 against 307 eligible tasks funds the entire informative band, so any
# scarcity-interaction reading is structurally compressed and demoted to exploratory-only).
POOL_ROBUST = [("\\methodname{}", ["q2bTs"]), ("uniform \\textsc{Grpo}", ["q2bFs"])]

# --- Table: the learning-rate ladder (internal review item 3 / W5) -----------------------------
# Added 2026-08-22. W5's charge is that the 2B/4B transfer result is inseparable from "LoRA rank 32
# at LR 1e-4 for 30 steps destabilises a small policy": the one uniform arm at 1e-5 scored 14.9 on
# the transfer benchmark, i.e. no damage at all, because it barely trained. The missing middle rung
# now exists.
#
# EVERY RUNG IS ONE RUN AT SEED OFFSET 500, WHICH IS THE POINT AND NOT A SHORTCUT. This is a
# one-line ladder in exactly the sense tab:ladder2b is one: the arms differ from each other by the
# LR token alone, and a rung on a fresh offset would confound the learning rate with the surface
# draw. The multi-seed values for the two CONFIGURATIONS at their shipped LR are in tab:main; the
# caption says so and prints the other seeds' transfer scores, so no reader can mistake a rung for
# a seed-aggregated row.
#
# TRIAGE AT 3e-5 WAS NOT RUN, and the table's shape says so rather than leaving a reader to notice
# a missing cell: the dominance statement the paper makes -- TRIAGE at the aggressive LR beats
# uniform at EITHER LR on the (in-distribution gain, transfer) pair -- does not need it, and an
# unrun cell is disclosed, never filled.
# 2026-09-01 (_fold6): THE LADDER BECOMES A COMPLETE 2x2 and stops being three corners of one.
# q2bT3e is q2bT with the learning-rate token alone changed to 3e-5 (diff verified at launch,
# one seed at the parent's own offset 500, exactly the construction every other rung uses), and
# it is the cell this table's own caption said had NOT been run and would not be interpolated.
# It is placed with the method rungs and BELOW the uniform block so each allocation rule reads
# down its own two rates.
# WHAT IT SETTLES, and it is the reviewer's confound rather than a new claim: if the method's
# transfer protection at 1e-4 were a hidden learning-rate reduction, the same method at a genuinely
# lower rate should transfer at least as well. It does not -- 14.00 against 16.62 -- so the
# protection is not a rate effect wearing the allocator's name.
# WHAT IT COSTS US, printed on the same row and never separated from the sentence above: at 3e-5
# the method is BELOW the uniform rung on transfer (14.00 against 15.62, and 1.1 pp below the
# untrained base), so at the safe rate it is uniform allocation that carries the transfer cell.
LR_LADDER = [("uniform \\textsc{Grpo}", "$10^{-5}$", "q2bF2"),
             ("uniform \\textsc{Grpo}", "$3\\times10^{-5}$", "q2bF3e5"),
             ("uniform \\textsc{Grpo}", "$10^{-4}$", "q2bF5"),
             ("\\methodname{}", "$3\\times10^{-5}$", "q2bT3e"),
             ("\\methodname{}", "$10^{-4}$", "q2bT")]
LR_BASE = "base2b"
# The other corrected uniform seeds at 1e-4 and the other method seed, printed to stdout and
# quoted in the caption so the single-seed rungs are bounded by the seeds that exist.
LR_CONTEXT = {"uniform 1e-4, other seeds": ["q2bF6", "q2bF7"], "TRIAGE 1e-4, other seed": ["q2bT2"]}

# 2026-09-04 _fold10: THE LEARNING-RATE LADDER GETS ITS SECOND SCALE, AND THE READOUT WAS FIXED
# BEFORE EITHER ARM RAN (logs/.lr_rung_after_candidates.sh header, quoted in PLAN_TRIAGE.md
# 2026-09-04 18:15). Same construction as the 2B block: one run per rung at seed offset 500, the
# rungs differing from each other by the learning-rate token alone, read against the 4B anchor and
# corrected inside the 4B family.
#
# THE PRE-REGISTERED READING FIRED ON ITS FIRST CLAUSE AND IT RUNS PARTLY AGAINST US. At 3e-5 the
# uniform control's held-out tool use RECOVERS -- BFCL 19.38 -> 28.50 against an untrained 29.00,
# i.e. back to base within a single run -- so the 4B transfer headline is partly an
# optimisation-stability result and the paper says so at 4B exactly as it does at 2B. What is
# measured at BOTH scales after this rung: at the aggressive rate, where the in-distribution gain
# lives, uniform allocation destroys held-out tool use (-10.6 pp at 2B, -9.6 at 4B) while the
# method gains (+1.5 and +5.1); at the safe rate the control returns to base with no gain while
# the method still sits above base on BFCL (+2.0 at 4B) with a softened window (+1.99 v +2.46).
# The 2B block prints the same structure one scale down, which is why the two are one float.
LR_LADDER_4B = [("uniform \\textsc{Grpo}", "$3\\times10^{-5}$", "q4bF3e5"),
                ("uniform \\textsc{Grpo}", "$10^{-4}$", "q4bF"),
                ("\\methodname{}", "$3\\times10^{-5}$", "q4bT3e5"),
                ("\\methodname{}", "$10^{-4}$", "q4bT")]
LR_BASE_4B = "base4b"
LR_HDR_4B = "\\emph{The same two rates at $4$B, one run per rung at the same offset}"

# --- Table: group-size sensitivity at 2B ------------------------------------------------------
# Added 2026-08-22. Same construction as the LR ladder and the same caveat structure: one run per
# rung at offset 500, differing from the method by the group-size token alone.
#
# THE CAVEAT THAT MUST TRAVEL WITH THIS TABLE, registered with the arm and not invented after it:
# k=8 is BATCH-MATCHED, not compute-matched. It draws 8 rollouts per group at the same batch size,
# so it spends more generation per optimizer step than the k=5 method does. The comparison
# therefore answers "does the transfer protection survive a wider group", not "is k=5 the
# compute-optimal group size", and the 8B k=2 arm -- the falsified axis, which IS compute-matched
# -- is quoted in the caption rather than given a row here, because a table mixing a batch-matched
# rung with a compute-matched one under one column heading would be reporting two designs as one.
K_LADDER = [("\\quad $k=5$ (the method)", "q2bT"),
            ("\\quad $k=8$", "q2bTk8")]
K_BASE = "base2b"

# --- Table: the mechanism decomposition on the cap-free benchmark ------------------------------
# Added 2026-08-22 (pre-registered PLAN_TRIAGE.md 2026-08-19 19:40, read out 20:35/21:40).
#
# WHAT THIS TABLE IS FOR. The mechanism section's claim is that removing the warm bank costs a
# TERMINATE ACTION rather than general tool-use capability. Until this pass that was an
# interpretation: every measurement of the ablation was on BFCL, an interactive benchmark with a
# 20-step turn cap, where "loops forever" and "cannot use tools" are not separable. NESTFUL has no
# turn cap and no loop to enter -- one completion emits a whole call graph and is scored by
# execution -- so an arm whose only defect is a lost stop should score AT BASE there while
# collapsing on BFCL, and an arm with a general deficit should be down on both. That is a
# dissociation, and the three columns below are what make it readable in one place.
#
# ONE DEFINITION PER COLUMN. The force-terminated share is BFCL's OWN recorded error type
# (`multi_turn:force_terminated`) over the same 800 instances as the pass rate beside it. It is
# NOT the 83.2% "runaway" of the outcome taxonomy (which folds in context-overflow crashes) and
# NOT the 88.4% per-instance turn-cap flag; both of those are stated in the text with their own
# names, and the caption says all three exist rather than letting a reader merge them.
NESTMECH = [("@HDR", "\\textbf{The allocation contrast}", None),
            ("", "\\quad untrained base policy", "base2b"),
            ("", "\\quad \\methodname{}", "q2bT"),
            ("", "\\quad uniform \\textsc{Grpo}", "q2bF5"),
            ("@HDR", "\\textbf{Components removed (one line each, same seed)}", None),
            ("", "\\quad $-$ warm bank", "q2bTnw"),
            ("", "\\quad $-$ calibration (independent shrinkage)", "q2bTiid"),
            ("@HDR", "\\textbf{The same two rules on \\textsc{Dapo}'s own $320$-task pool}", None),
            ("", "\\quad \\methodname{}", "q2bTs"),
            ("", "\\quad uniform \\textsc{Grpo}", "q2bFs")]

# --- Appendix table: the 4B BFCL transfer grid ------------------------------------------------
# NEW 2026-08-24. Until this pass the only checkpoint-level transfer float in the paper was
# tab:bfcl, which is 8B-only: at 4B the paper could print TRIAGE and its uniform control and
# nothing else, so the coverage caveat "PLR / retrieval / DAPO transfer is unmeasured at this
# scale" was true and was printed. Those three arms have now been scored on the identical 800-task
# index, joining VIP, TSCL and the no-warm-bank ablation, and the caveat is deleted because it is
# false -- every configuration of tab:main's 4B block now has a transfer cell.
#
# WHAT THIS TABLE SHOWS RUNS AGAINST US AND IS PRINTED AT FULL STRENGTH. Two published baselines
# (VIP 36.75, PLR-style replay 33.62) sit above the method's better seed, and retrieval (32.62)
# ties its two-seed mean exactly. The paper's transfer claim is and has always been
# TRIAGE-against-UNIFORM (32.62 against 19.38 here, and uniform is 9.6 pp BELOW the untrained
# policy), and that claim is untouched; what this grid does is turn an admission the draft already
# carried -- that TRIAGE does not separate from published PRE-GENERATION allocators on transfer --
# from a concession into a measurement. Proposition 2 predicts exactly this for VIP, which ranks
# tasks the same way TRIAGE does at matched temperature.
#
# THE REFERENCE IS BOTH METHOD SEEDS, NOT ONE. tab:bfcl's last column is a paired difference
# against a single pinned 8B checkpoint because at 8B the matched window pinned one. At 4B the
# method has two seeds and no rule picks between them, so the column is the MEAN over both
# pairings with the WEAKEST of the two p values -- the same convention tab:transfer's BFCL block
# uses for every between-method quantity in this paper, and the reason it exists is that a gap
# quoted from the more favourable of two pairings is the error this paper spends its length
# arguing against.
#
# EVERY CONTRAST HERE IS A MEMBER OF NO HOLM FAMILY IN THIS PAPER, and every row is therefore
# marked \ddagger under tab:bfcl's own legend. The merged 19-contrast family was fixed and
# published before any of these cells existed; enrolling them now would let a result decide the
# size of the correction that judges it. The p values printed are RAW and UNCORRECTED, they are
# printed rather than hidden.
# CORRECTED 2026-08-24 (internal re-review N4). This comment used to end "and no claim in the paper
# rests on one of them", and the caption said the same thing. IT IS FALSE. The uniform-GRPO row
# here (+13.25 pp, p <= 1.1e-13) IS the 4B half of the transfer contribution, of fig:transfer's
# annotation and of tab:transfer's 4B row. At that p no correction changes the verdict, so the
# claim stands -- but it stands UNCORRECTED, and the caption now says so instead of denying that
# any claim rests here. The corrected/raw surface of the whole paper is summarised once, in the
# multiplicity paragraph of Section 9.1, rather than left to be assembled from row marks.
# LABELS SHORTENED to the same forms tab:bfcl adopted on 2026-08-22 for the same reason: this
# table sets its own width, the full names cost it 20pt it does not have, and the full names are
# one float away in tab:main's own 4B block. No row's identity is ambiguous under the short form.
# THE STEP COLUMN IS DELIBERATELY ABSENT, unlike tab:bfcl's: every 4B checkpoint here is at step
# 30 and a column that carries one value in every row is width spent on nothing. The step is one
# fact about the whole table and the caption states it, which is the same rule tab:main uses for
# its own fixed step.
# 2026-08-27 COMPREHENSIVE GRID. Six rows added/relabelled so this table carries every 4B
# configuration that has a scored BFCL cell, not just the ones that existed in August's first pass:
# q4bV is relabelled "+ warm bank" (it always ran on our bank), its second seed q4bV2 joins it, and
# the faithful bank-less VIP, the two revision seeds and the plug-in configuration are new rows.
BFCL4B = [("\\textsc{Vip} rule $+$ warm bank (seed 1)", "q4bV"),
          ("\\textsc{Vip} rule $+$ warm bank (seed 2)", "q4bV2"),
          ("plug-in estimator config (seed 1)", "q4bTpe"),
          ("plug-in estimator config (seed 2)", "q4bTpe2"),
          ("scale-corrected $\\rho$ (seed 3)", "q4bTr3"),
          ("scale-corrected $\\rho$ (seed 2)", "q4bTr2"),
          ("scale-corrected $\\rho$ (seed 1)", "q4bTr"),
          ("\\textsc{Trace} (published, no bank)", "q4bTf"),
          ("\\textsc{Vip} (published, no warm bank)", "q4bTvip"),
          ("\\methodname{} (seed 1)", "q4bT"),
          ("\\textsc{Plr}-style replay", "q4bP"),
          ("retrieval-shaped surface", "q4bR"),
          ("\\methodname{} (seed 2)", "q4bT2"),
          ("\\textsc{Dapo} dynamic sampling", "q4bD"),
          ("\\methodname{}, no warm bank", "q4bTnw"),
          ("learning progress (\\textsc{Tscl})", "q4bLp"),
          ("uniform \\textsc{Grpo}", "q4bF")]
BFCL4B_STEP = 30
BFCL4B_BASE = "base4b"
BFCL4B_REF = ["q4bT", "q4bT2"]

# --- Appendix table: the 2B BFCL transfer grid --------------------------------------------------
# NEW 2026-08-24 (final fold), and it is the readout of a PRE-REGISTERED PREDICTION. The four
# published-baseline arms at 2B had never been scored on this benchmark, which is why Section 9.6
# said the 2B block "still does not" have a transfer cell for every configuration. Before any of
# the four cells ran, PLAN_TRIAGE.md 2026-08-24 02:10 registered what the protection mechanism
# implies for them: all four train WITHOUT the warm bank, and at 2B the bank is what prevents
# termination collapse (q2bTnw 2.25 against the method's 16.62/15.62; q2bFs 0.00), so each should
# land in the COLLAPSED BAND -- below 10 pass, with a HIGH force-terminated share -- while TRIAGE
# holds near 16. Both branches were declared publishable in advance: confirmed is a decisive
# all-baselines separation at 2B, refuted narrows the protection story to allocation-rule-specific
# and the cells print as measured. The outcome is in this table and is narrated in Section 9.6.
#
# THE FORCE-TERMINATED COLUMN IS HERE AND NOT IN tab:bfcl4b, and that asymmetry is deliberate: the
# registered prediction is a CONJUNCTION (low pass AND high force-terminated), so a table that
# printed only the pass rate could not be checked against the prediction it exists to read out.
# It is bfcl_records.forced_share -- BFCL's own recorded `multi_turn:force_terminated` error type
# over the same 800 instances -- and it is the NARROWEST of this project's three termination
# statistics, exactly as tab:nestmech's caption already says.
#
# DAPO IS NOT IN THIS TABLE AND ITS ABSENCE IS A COVERAGE GAP, NOT A NULL. q2bD has no BFCL cell,
# so six of tab:main's seven 2B configurations have one and the seventh does not. It prints a
# spanning `not run at this scale` row through the NOT_RUN machinery rather than a dash, which is
# the third state that machinery exists for and the first time anything has used it.
BFCL2B = [("\\methodname{} (seed 1)", "q2bT"),
          ("\\methodname{} (seed 2)", "q2bT2"),
          ("\\textsc{Vip} allocation", "q2bV"),
          ("\\textsc{Plr}-style replay", "q2bP"),
          ("retrieval-shaped surface", "q2bR"),
          ("learning progress (\\textsc{Tscl})", "q2bLp"),
          ("\\textsc{Dapo} dynamic sampling", NOT_RUN),
          ("uniform \\textsc{Grpo} (seed 1)", "q2bF5"),
          ("uniform \\textsc{Grpo} (seed 2)", "q2bF6"),
          ("uniform \\textsc{Grpo} (seed 3)", "q2bF7")]
BFCL2B_STEP = 30
BFCL2B_BASE = "base2b"
BFCL2B_REF = ["q2bT", "q2bT2"]
# The four arms the registration is about. All four or none: a partial readout of a registered
# prediction is the one thing this table must not print, so the emitter refuses to write a
# half-filled grid rather than writing one and captioning the gap.
BFCL2B_REGISTERED = ["q2bV", "q2bP", "q2bR", "q2bLp"]


# --- MAIN-TEXT TRANSFER TABLE (2026-08-28, presentation pass II) --------------------------------
# The paper's transfer evidence at all three scales, on both held-out benchmarks, in ONE main-text
# float. It is NOT a new measurement and NOT a new statistic: the base / control / method cells and
# both difference rows are read straight off BR.scale_block() and NR.scale_block() -- the same two
# calls tab:transfer is built from -- so a digit here cannot disagree with a digit there. What is
# new is two ROWS: the two published allocators run AS PUBLISHED, with none of our components,
# which until this pass appeared only as scattered cells in the per-scale appendix grids.
#
# THE STEP CONVENTION, WHICH IS THE ONE THING THIS LAYOUT COULD GET WRONG. NESTFUL scores exactly
# one checkpoint per (scale, arm), and the step is NOT the same for every row of the 8B column: the
# registered method and control cells are step 15 (the only step ckpt_a8T holds -- the asymmetry
# tab:transfer has always carried) while the faithful-TRACE cell is step 30. A column that mixed
# those silently would be a step-mixing claim, so EVERY NESTFUL cell prints its own step as a
# subscript and the note says what the subscript is. BFCL cells are seed means over each arm's
# matched-window checkpoint, tab:bfcl's convention, and their per-arm steps are in that table.
# TRANSFER_FAITHFUL: the two published-as-published rows, (label, {scale: BFCL arm}, {scale:
# NESTFUL arm}). NOT_RUN is the third state -- registered, never trained at that scale -- and is
# printed as a marked em dash, never as a missing measurement on a measured arm.
TRANSFER_FAITHFUL = [
    ("\\quad \\textsc{Trace}",
     {"2B": "t2bTf", "4B": "q4bTf", "8B": "t8Tf"},
     {"2B": "t2bTf", "4B": "q4bTf", "8B": "t8Tf"}),
    # 2026-08-29 FOLD: the 2B and 8B faithful-VIP cells are measured now (q2bVf BFCL 800 /
    # NESTFUL 1861 at step 30; a8Tvipf the same), so this row carries no NOT_RUN state any more.
    ("\\quad \\textsc{Vip}",
     {"2B": "q2bVf", "4B": "q4bTvip", "8B": "a8Tvipf"},
     {"2B": "q2bVf", "4B": "q4bTvip", "8B": "a8Tvipf"}),
]
# BFCL's four multi_turn splits, in the order tab:bfcl prints them.
BFCL_CATS = ["multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func",
             "multi_turn_miss_param"]

# --- Appendix table: every scored NESTFUL checkpoint, and the 8B replication -------------------
# Added 2026-08-22 (internal review W12). W12's charge is that NESTFUL is single-seed at every
# scale, so its 4B q=0.010 is a paired test between two individual runs.
#
# WHAT THE REPLICATION IS, STATED EXACTLY, BECAUSE THE FLEET'S OWN SHORTHAND FOR IT IS WRONG.
# The three 8B arms now scored on the method side -- a8T, a8T2, a8T3 -- are NOT three seeds. They
# all carry seed offset 500 and they are three CONFIGURATIONS of this paper's own TRIAGE ladder
# (v1 cold-start; v2, warm bank + tau; v3 at gamma=0.9), which is exactly how tab:family blocks
# them. The uniform side IS a seed pair (a8F off=500, a8Fr off=1500). So the 8B NESTFUL null is
# replicated across three configurations of the method at one seed AND across two seeds of the
# control -- a real and reportable strengthening, and not the "3-seed mean" the lab log called it.
# Getting this backwards would have put a configuration ladder in the paper under the word "seed",
# in a paper whose own standard is that seeds and configurations are never merged.
#
# THE PRE-REGISTERED m=3 FAMILY IS UNTOUCHED BY ALL OF THIS. transfer.tex still prints exactly the
# contrast that was registered before any cell was scored (a8T vs a8F at step 15, Holm over the
# three scales). These extra cells are a POST-REGISTRATION robustness readout and are emitted into
# their own appendix float with that label on them, because moving the primary contrast to a
# pooled one after seeing the pooled numbers is precisely what the registration exists to prevent.
NESTREP_LADDER = [("\\methodname{} v1 (cold start)", "a8T"),
                  ("\\methodname{} v2 (warm bank $+$ $\\tau$)", "a8T2"),
                  ("\\methodname{} v3 ($\\gamma{=}0.9$)", "a8T3")]
NESTREP_UNIFORM = [("uniform \\textsc{Grpo}, seed 1", "a8F"),
                   ("uniform \\textsc{Grpo}, seed 2", "a8Fr")]

# 2026-08-24, PRE-FINAL FOLD -- W12 IS NOW CLOSED AT ALL THREE SCALES, AND CLOSING IT COST US THE
# LARGEST NESTFUL RESULT IN THE PAPER.
#
# W12's charge was that NESTFUL was single-seed at 2B and 4B, so the +6.07 pp at 2B and the +1.99
# at 4B were paired tests between two individual runs. The seed checkpoints existed and were not
# scored; they are scored now (n=1861 each, same commit fc2c4123, same steps): q2bT3 and q2bF6 at
# 2B, q4bT2 and q4bF2 at 4B. Both scales become 2-against-2.
#
# THE 2B GAP DOES NOT SURVIVE, AND IT DOES NOT MERELY SHRINK -- IT CHANGES SIGN. On the method side
# q2bT wins 25.47 and q2bT3 wins 17.62, a 7.85 pp spread across two seeds of ONE configuration; on
# the control side q2bF5 wins 19.40 and q2bF6 wins 27.08, a 7.68 pp spread. The four pairings run
# +6.07, -1.61, -1.77 and -9.46 pp: mean -1.69, and the registered cell (+6.07) is the single most
# favourable of the four. The 4B gap survives in sign but is halved: +1.99, -0.16, +2.53, +0.38,
# mean +1.18, and one of the four pairings is negative.
#
# THIS TABLE PRINTS THAT AT FULL STRENGTH AND THE PRIMARY CONTRAST IS NOT MOVED ONTO IT. The
# pre-registered m=3 family of transfer.tex still prints exactly the nine cells that were fixed
# before any completion was generated -- retiring a registered readout because its replication is
# unfavourable is the same act as adopting a pooled one because it is favourable, and this paper
# does neither. What changes is that the replication is reported beside it, in the prose that
# reads it, and in Limitations.
#
# THE SEED SPREADS ABOVE ARE ALSO THE PAPER'S OWN SCOPE CONDITION ARRIVING ON THE TRANSFER AXIS.
# Section 9.6 has argued from the start that a single-seed contrast at these effect sizes is not
# interpretable; it argued it about the in-distribution benchmark. At 2B on NESTFUL the
# within-configuration spread (7.9 and 7.7 pp) is larger than every between-method gap in the
# block. That is the same finding, one benchmark over, and it is stated as such rather than as a
# surprise.
#
# {scale: (base tag, step, (method header, rows), (control header, rows))}. Blocks print smallest
# policy first, the same order as tab:main and tab:transfer.
NESTREP = [
    ("2B", "base2b", 30,
     ("Two seeds of the method", [("\\methodname{}, seed 1", "q2bT"),
                                  ("\\methodname{}, seed 2", "q2bT3")]),
     ("Two seeds of the uniform control", [("uniform \\textsc{Grpo}, seed 1", "q2bF5"),
                                           ("uniform \\textsc{Grpo}, seed 2", "q2bF6")])),
    ("4B", "base4b", 30,
     ("Two seeds of the method", [("\\methodname{}, seed 1", "q4bT"),
                                  ("\\methodname{}, seed 2", "q4bT2")]),
     ("Two seeds of the uniform control", [("uniform \\textsc{Grpo}, seed 1", "q4bF"),
                                           ("uniform \\textsc{Grpo}, seed 2", "q4bF2")])),
    # The 8B block is the 2026-08-22 readout, unchanged. Its method side is a CONFIGURATION ladder
    # at one offset and its control side is a seed pair; the block header says so, and it is the
    # one block of this table whose two sides are different kinds of replicate.
    ("8B", "base8b", 15,
     ("Three configurations of the method's ladder, one seed", NESTREP_LADDER),
     ("Two seeds of the uniform control", NESTREP_UNIFORM)),
]

# --- Table: mechanism -------------------------------------------------------------------------
MECH = [("uniform \\textsc{Grpo}", ["a8F", "a8Fr"]),
        ("\\textsc{Dapo} dynamic sampling", ["dapo"]),
        # 2026-08-24: two seeds, the same admission as tab:main's PLR row. This row was the last
        # single-seed published-baseline row in this table that HAD a replicate on disk.
        ("\\textsc{Plr}-style prompt replay", ["b8plr", "b8plrr"]),
        ("learning progress (\\textsc{Tscl})", ["a8Tlp"]),
        ("\\textsc{Vip} variance allocation", ["a8Tvip"]),
        ("difficulty band filter", ["a8T2n", "a8T2nr"]),
        ("\\textsc{Trace}-style plug-in estimator", ["a8Tpe"]),
        ("\\methodname{} (v2, warm $+$ $\\tau$)", ["a8T2"]),
        ("\\methodname{}", FULL_METHOD)]

# --- Table: the same configurations at three model scales -------------------------------------
# Restructured 2026-08-17. Was two stacked per-model blocks covering 2B and 4B only, with the 2B
# control split into three seed rows. Now it is ONE structure: configurations down the side,
# scales across the top, seeds aggregated, so "does this reproduce as the policy shrinks" is a row
# to read across instead of a comparison the reader has to assemble from two blocks and a third
# table. Each cell is still scored against ITS OWN model's anchor and Holm-corrected within ITS
# OWN model's family -- merging the families is the one thing this table must never do, since a
# smaller policy scored against a larger anchor reports model capacity as an allocation effect.
SCALES3 = ["2B", "4B", "8B"]
# 2026-08-22, FINAL REGENERATION. This table's baseline block MOVED, it was not deleted: the four
# baseline rows it used to carry ("the same baselines at each scale") are now full rows of the
# scale-blocked tab:main, with five columns each instead of one, so keeping them here would have
# printed the same measurement twice in two floats a page apart. What is left is the one question
# tab:main does not answer -- whether the allocation contrast is larger than the disagreement
# between two seeds of one configuration -- which is the @DIFF and @SD pair and the two rows they
# are computed from. No cell that survives here changed.
#
# ONE THING THAT MOVED IS A CORRECTION, NOT A RELOCATION, AND IT IS FLAGGED RATHER THAN QUIETLY
# FIXED. The retired row labelled "VIP (variance allocation)" mapped 2B->q2bP and 4B->q4bP, which
# are the PLR variance-REWEIGHT arms, not the VIP variance-ALLOCATION arms; only its 8B cell
# (a8Tvip) was the method the label named. The row was therefore one method at 8B and a different
# method at 2B/4B. It could not be fixed before this pass because no VIP arm existed below 8B;
# q2bV now does, and in tab:main VIP and PLR are separate rows with their own arms at every scale
# where an arm exists. See CHANGES.md 2026-08-22.
SCALEROWS = [
    ("@HDR", "\\textbf{The allocation contrast}", None),
    # 4B became a seed pair 2026-08-24 (q4bT2's window completed), so the @SD row below is now
    # defined on BOTH sides at every scale -- until this pass the 4B method column had one run and
    # the within-configuration spread this table exists to print was half-missing there.
    # 2026-08-28: NAMED, not relabelled away. tab:main's method row is now the scale-adaptive
    # configuration; this table's paired method-vs-control contrast is built on the FIXED-RHO
    # configuration's seeds (the scale-adaptive arms have no matched control seeds and, at 2B, no
    # transfer cells yet). Two tables must not mean different things by one word, so this row says
    # which configuration it carries.
    ("", "\\quad \\methodname{} (fixed-$\\rho$ configuration)",
     {"2B": ["q2bT", "q2bT3"], "4B": ["q4bT", "q4bT2"], "8B": FULL_METHOD}),
    ("", "\\quad uniform \\textsc{Grpo} (matched control)",
     {"2B": ["q2bF5", "q2bF6", "q2bF7"], "4B": ["q4bF", "q4bF2"], "8B": ["a8F", "a8Fr"]}),
    ("@DIFF", "\\quad \\emph{difference} (fixed-$\\rho$ \\methodname{} $-$ uniform)", None),
    # The comparator the difference row has to be read against. The scope condition of this paper
    # -- "the lead is smaller than the disagreement between seeds of one configuration" -- was
    # asserted in this table's caption and verifiable only by eye against the +- values two rows
    # up. This row prints it, so the claim is read off the table. It is NOT a new measurement: it
    # is the larger of the two contrast rows' OWN seed sds, the same quantity already printed as
    # +- on those cells, and no seed row enters any main-text float.
    ("@SD", "\\quad \\emph{within-configuration seed sd}", None),
    # The one non-published allocation rule that is measured at more than one scale and has no
    # row in tab:main, because it is our own weight-rule ablation rather than a published method.
    # It stays here so the ablation is not the only float that carries it.
    ("@HDR", "\\textbf{Our own weight-rule ablation, at the scales it was run}", None),
    ("", "\\quad difficulty band filter",
     {"2B": ["q2bN"], "4B": [], "8B": ["a8T2n", "a8T2nr"]})]

# --- Appendix: per-seed reproduction ----------------------------------------------------------
# THE ONLY PLACE PER-SEED ROWS ARE ALLOWED. Every multi-seed configuration in every main-text
# float appears here broken out, so nothing is hidden by the aggregation -- it is relocated.
#
# 2026-08-28, BEST-SEED PASS. This table carries TWO NEW OBLIGATIONS, because the main-text floats
# now print one seed per multi-seed row and this is the table their captions send the reader to:
#   (1) it prints each block's SEED MEAN as its own row, since that statistic no longer appears in
#       any float (tab:main and tab:mainsteps used to carry it), and
#   (2) it MARKS the selected seed in every multi-seed block, so a reader can see which run each
#       main-text cell is without recomputing a maximum.
# The mark is on the SEED LABEL, never on a number: this paper's no-bolded-winner rule is about
# cells, and what is marked here is a selection, which is a fact about the reporting convention
# rather than a verdict about the run.
# The 4B scale-refit block is ADDED this pass. It is the 4B method row of tab:main and had never
# been broken out here -- a real gap under this table's own rule ("every multi-seed row of every
# main-text float"), inherited when that row was promoted on 2026-08-28 and closed now.
PERSEED = [("8B", [("\\methodname{}", FULL_METHOD),
                   ("\\methodname{}, sharpened ($\\tau{=}0.3$)", ["a8T5k", "a8T5kr"]),
                   ("uniform \\textsc{Grpo}", ["a8F", "a8Fr"]),
                   ("difficulty band filter", ["a8T2n", "a8T2nr"]),
                   ("\\methodname{} v2 ($\\gamma{=}0.9$)", ["a8T2", "a8T2r"]),
                   # 2026-08-24: the two PUBLISHED baselines that are multi-seed rows of tab:main
                   # are broken out here for the first time. This table's own rule is "every
                   # multi-seed row of every main-text float", and until this pass it silently
                   # meant "every multi-seed row of ours": VIP has been a 2-seed row of tab:main
                   # since 2026-08-22 and never had a per-seed row here. PLR became one today and
                   # would have inherited the same gap. Both are added; the rule now holds as
                   # written, and a reader can check a baseline's seed spread against ours in the
                   # one table that is allowed to show seeds.
                   ("\\textsc{Vip} variance allocation", ["a8Tvip", "a8Tvipr"]),
                   ("\\textsc{Plr}-style prompt replay", ["b8plr", "b8plrr"])]),
           # 2026-08-31: THE THREE RERUNS ARE DELIBERATELY NOT HERE, and the reason is a fact
           # about them rather than a presentation choice. Each was launched with the PARENT'S
           # OWN --seed-offset (500 for all three), so it is a repetition of one configuration
           # at one nominal seed and not a second seed of it. Printing it as a seed row would
           # make its spread read as seed variance, which is the quantity this table exists to
           # show and the quantity a same-offset repetition cannot estimate. Their windows are
           # printed beside their parents' in tab:reruns instead, under a caption that says
           # what they are.
           ("2B", [("\\methodname{}", ["q2bT", "q2bT3"]),
                   ("uniform \\textsc{Grpo}", ["q2bF5", "q2bF6", "q2bF7"])]),
           ("4B", [("\\methodname{}, scale-refit $\\rho$ (the $4$B method row of "
                    "Table~\\ref{tab:main})", ["q4bTr", "q4bTr2", "q4bTr3"]),
                   ("\\methodname{}, fixed-$\\rho$", ["q4bT", "q4bT2"]),
                   ("uniform \\textsc{Grpo}", ["q4bF", "q4bF2"])])]

# --- Table: SECONDARY OUTCOME, policy termination quality --------------------------------------
# The non-terminating (turn-capped) failure rate on an arm's own held-out VAL episodes, first third
# of the run against last third. `stop_reason == "max_turns"` is an episode that ended because the
# turn cap was hit rather than because the policy answered -- the looping failure mode.
#
# THIS IS A SECONDARY OUTCOME AND THE EMITTER TREATS IT AS ONE. It is reported alongside the solve
# rate, never instead of it, and the val solve rate is carried in the same table so a reader can
# see both move together or not. Coverage is uneven and the table says so rather than hiding it:
# VAL_BEFORE=True was not set on the 8B uniform seed 1, so that arm logged no val episodes at all
# and the 8B control column rests on one seed.
TERM = [("2B", [("\\methodname{}", ["q2bT", "q2bT3"]),
                ("uniform \\textsc{Grpo}", ["q2bF5", "q2bF6", "q2bF7"])]),
        ("4B", [("\\methodname{}", ["q4bT", "q4bT2"]),
                ("uniform \\textsc{Grpo}", ["q4bF"])]),
        ("8B", [("\\methodname{}", ["a8T3g", "a8T3gr", "a8T3g2"]),
                ("uniform \\textsc{Grpo}", ["a8Fr"])])]


def val_rows(tag):
    """Held-out val episodes of one arm, in order, `.arm_since` honoured."""
    p = "%s/work/verl/run_%s/episodes.jsonl" % (R, tag)
    try:
        since = int(open("%s/work/verl/run_%s/.arm_since" % (R, tag)).read().strip() or 0)
    except Exception:
        since = 0
    out = []
    if not os.path.exists(p):
        return out
    for i, line in enumerate(open(p, errors="ignore")):
        if i < since:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("split") != "val":
            continue
        out.append((d.get("stop_reason"), 1 if float(d.get("reward", 0)) > 0 else 0))
    return out


def term_thirds(tag):
    """(max_turns% first third, last third, solve first, solve last) or None."""
    v = val_rows(tag)
    if len(v) < 60:
        return None
    k = len(v) // 3
    def st(seg):
        return (100.0 * sum(1 for s, _ in seg if s == "max_turns") / len(seg),
                100.0 * sum(r for _, r in seg) / len(seg))
    (a, b), (c, d) = st(v[:k]), st(v[-k:])
    return a, c, b, d


# --- Table: post-window durability ------------------------------------------------------------
# REBUILT 2026-08-31 (fold v). The float this replaces was 8B-ONLY and read six columns
# {30,35,40,45,50,60} whose cells came from whichever step each arm happened to have been scored
# at, two of them below the finished-cell row count and carrying a dagger. It answered "does the
# 8B method decay after the window"; the question the paper now has the measurements for is
# whether the ORDERING the window reports survives past it AT EVERY SCALE, and that needs one
# convention rather than six columns of opportunistic coverage.
#
# THE CONVENTION, and every row obeys all of it: three absolute training steps 40, 50 and 60;
# in-distribution cells of n=590 against that scale's own base anchor, the same anchor and the
# same paired statistic as the window; ONE SEED per row; and the two transfer benchmarks scored
# once each at step 60. Nothing here is ever folded into the window statistic -- the window was
# fixed in advance, and reading an arm at whichever later step flatters it is the
# maximum-over-checkpoints procedure the window exists to prevent.
#
# THREE PROTOCOLS, and the difference is a real caveat rather than bookkeeping, so it is a COLUMN
# and not a sentence somebody has to find:
#   resume    the child's checkpoint dir is seeded with the parent's global_step_30 plus its Adam
#             moments and dataloader state, and verl's resume_mode=auto restores all three. Step
#             numbering stays ABSOLUTE (child step s == parent step s) and no seam is introduced.
#   merge     forced where the parent's checkpoint is model-only: every verl resume path opens
#             extra_state_world_size_*_rank_*.pt unconditionally. The parent's adapter is merged
#             into the base and the child trains from it, so the optimizer moments and the LR
#             schedule RESTART at step 30 and the child's own counter restarts with them --
#             cell_STEP{10,20,30} ARE absolute 40/50/60, which is what CELLSTEP encodes.
#   straight  one optimizer-continuous run to step 60, no seam at all. Exactly one row needs it
#             and it is a disclosure, not a convenience: the entire registered 2B control family
#             (q2bF5, q2bF6, q2bF7, and q2bF2/q2bF3/q2bFs besides) is pruned at every step, so
#             there is no 2B uniform-GRPO checkpoint anywhere to continue from. q2bF6L is the
#             byte-identical q2bF6 command at the SAME seed offset trained straight through. IT
#             IS NOT A CONTINUATION AND IS NEVER CALLED ONE.
#
# THE 4B METHOD ROW IS SEED 3, NOT THE WINDOW'S SEED 1, and that is stated wherever it is drawn:
# q4bTr's every checkpoint was pruned, and inside the same configuration q4bTr2 and q4bTr3 have
# window means equal to six places, so best_seed's own documented tie-break (lower Holm q) picked
# q4bTr3. The 4B post-window points and the 4B window points are not the same run.
#
# (scale, label, arm, protocol). The BFCL/NESTFUL cells are read from the arm's own archive run,
# which is scored at step 60 for every row; a MERGE child's records carry its OWN step 30 in the
# step field, so the subscript is not printed here at all and the column header carries the step.
DURROWS = [("2B", METHOD_TEX, "q2bTc", "merge"),
           ("2B", "uniform \\textsc{Grpo}", "q2bF6L", "straight"),
           ("2B", "\\textsc{Plr}", "q2bPc", "resume"),
           ("4B", METHOD_TEX, "q4bTr3c", "resume"),
           ("4B", "uniform \\textsc{Grpo}", "q4bFc", "resume"),
           ("4B", "\\textsc{Plr}", "q4bPc", "resume"),
           ("8B", METHOD_TEX, "a8T3g2c", "merge"),
           ("8B", "uniform \\textsc{Grpo}", "a8Fc", "merge")]
DSTEP = [40, 50, 60]
CELLSTEP = {"merge": {40: 10, 50: 20, 60: 30},
            "resume": {40: 40, 50: 50, 60: 60},
            "straight": {40: 40, 50: 50, 60: 60}}
PROTO_TEX = {"merge": "merge", "resume": "resume", "straight": "straight"}
# The transfer anchors, one per scale per benchmark -- the SAME untrained base runs
# tab:transfermain prints on its top row, named here so the two floats cannot drift apart.
DUR_BFCL_BASE = {"2B": "base2b", "4B": "base4b", "8B": "base"}

# The 8B-only post-window record this float used to be. It is NOT deleted, because its cells are
# real measurements of arms the rebuilt table does not carry: the method's FIRST seed (which ran
# straight past the window to 40), the control's own straight-through post-window steps, and the
# two measured negatives. They are simply not on the new convention -- their steps are 35 and 45
# as often as 40 and 50, and two of their cells never reached a finished row count -- so they are
# printed beside it under their own caption rather than folded in and rounded to look uniform.
DURLEGACY = [("\\methodname{} (seed $1$)", "a8T3g"),
             ("uniform \\textsc{Grpo}", "a8F"),
             ("adaptive surface (v1)", "a8A"),
             ("\\textsc{Elsa}", "a8A3")]
DLSTEP = [30, 35, 40, 45, 50, 60]


def emit_tables(outdir):
    import bfcl_records as BR
    import nestful_records as NR
    fam, nbase, nfam = full_family()
    fams = {m: full_family(m) for m in SCALES3}

    def bfcl_audit(arms):
        """One MAIN row's BFCL checkpoints, for the stdout audit block only.

        tab:main NO LONGER CARRIES A TRANSFER COLUMN -- see the printed block at the end of this
        function for why, and CHANGES.md for the full decision. These numbers still matter (they
        are what the decision was made on, and they are the paper's transfer evidence), so they are
        printed here rather than dropped: the paper's home for them is tab:transfer and tab:bfcl.
        """
        got = []
        for a in arms or []:
            s, _, st, _ = BR.load_run(a)
            if s:
                got.append((a, BR.rate(s), st))
        return got

    def W(name, lines):
        open("%s/%s" % (outdir, name), "w").write("\n".join(lines) + "\n")

    def notes_block(items, gap="3pt", env="tabular"):
        """Booktabs table notes emitted OUTSIDE the tabular, wrapping at \\textwidth.

        THEY USED TO BE `\\multicolumn{7}{@{}l}{...}` ROWS AND THAT WAS A LAYOUT BUG, not a style
        preference (found in the 2026-08-28 review of tab:transfermain). A spanning note row is a
        BOX, not a paragraph: it never wraps, so a note wider than the tabular's natural width
        makes the tabular that wide, and LaTeX pushes the surplus into the last spanned column.
        In tab:transfermain that shoved the NESTFUL 8B column to the right margin and opened a
        gap after the 4B column that a reader reads as a missing column group.
        Emitted here instead: each note is its own PARAGRAPH at the float's own width, so a long
        note wraps and no note can ever set a column position. `\\raggedright` is needed because
        the float sets `\\centering`, and parindent/parskip are zeroed so the notes sit tight
        under the rule the way booktabs notes do. Both main-text tables use this one mechanism.
        """
        return (["\\end{%s}" % env,
                 "\\par\\addvspace{%s}" % gap,
                 "{\\scriptsize\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{0pt}"
                 "\\raggedright"]
                + [it + "\\par" for it in items]
                + ["}"])

    # ------------------------------------------------------------------------------------------
    # 2026-08-29 PRESENTATION PASS III (PI directive: the two main-text tables keep exactly the
    # content they have and get a clearer structure). THE TWO HELPERS BELOW ARE USED BY tab:main
    # AND tab:transfermain AND BY NOTHING ELSE, which is why they live here and not beside the
    # registries: every other emitted float must stay byte-identical through this pass, and a
    # helper applied in a registry would have changed tab:mainsteps, tab:mainabs, tab:framecfg
    # and tab:nestmech with it.
    def grey(cell):
        """The Delta-vs-base columns, in a lighter ink than the rates beside them.

        A rate is the measurement; the Delta is that rate read against a base policy. Setting the
        Delta columns in a lighter ink makes the grid read rate-first without hiding anything --
        every Delta is still a printed signed number, still in the same cell, still explained in
        the note. NOTHING in this table depends on the colour: it survives greyscale printing as
        grey, and a copy-paste that loses it loses no digit and no marker.
        black!60 (#666666, about 5.7:1 on white) and deliberately NOT the figure palette's
        trGrey: head.tex FIXES trGrey's meaning as \"the uniform control\" in every figure of
        this paper, so tinting a column with it would say something about the control that is not
        meant, and at #8C8C8C it clears only ~3.4:1, which is thin for 7pt digits in print.
        """
        return cell if cell == "" else "\\textcolor{black!60}{%s}" % cell

    def blocklab(lab):
        """A row-group label, bold in the registries, set in italic in the two main-text tables.

        The registries' \\textbf is what tab:mainsteps, tab:mainabs and tab:framecfg print and is
        not touched; only the two main-text floats restyle it, and they restyle it identically.
        Italic rather than bold because bold inside a table of numbers competes with the numbers
        for attention, and this paper marks no winner in bold.
        """
        return lab.replace("\\textbf{", "\\textit{", 1)

    # The three family sizes are READ OFF FAMILY_PIN rather than typed here. They were typed here
    # until 2026-09-02, and by then the 2B count in this header was one arm stale (t2bTbk grew that
    # family on 2026-09-01 and only the per-table scale headers, which are computed, followed it),
    # which is the drift this whole header exists to warn about.
    GEN = ("% generated by surface/verl_rl/paper_numbers.py --emit-tables; do not edit by hand\n"
           "% Holm family membership is PINNED in paper_numbers.FAMILY_PIN, not taken from the\n"
           "% live glob: the fleet keeps training, and an arm that banks its first cell after a\n"
           "% table ships would raise m and move published q values with nothing about the paper\n"
           f"% having changed. The pinned families are {len(FAMILY_PIN['2B'])} /"
           f" {len(FAMILY_PIN['4B'])} / {len(FAMILY_PIN['8B'])} arms at 2B / 4B / 8B, grown\n"
           "% deliberately on 2026-08-22 (six arms), on 2026-08-24 (three 4B arms), in the final\n"
           "% fold of 2026-08-24 (one 8B arm, b8plrr), on 2026-08-27, the comprehensive-grid\n"
           "% pass, by five 4B arms (q4bV2, q4bTvip, q4bTr, q4bTr2, q4bTpe) and one 8B arm (t8Tf)\n"
           "% -- which is why every 4B and 8B q moved in that pass and no 2B q did -- on\n"
           "% 2026-08-29 by the three FAITHFUL published arms whose windows completed overnight:\n"
           "% q2bVf and t2bTf at 2B (20 -> 22) and a8Tvipf at 8B (40 -> 41), on 2026-08-31 by the\n"
           "% two removal ablations (a8Tnw at 8B, 41 -> 42) and on 2026-09-01/02 by t2bTbk at 2B\n"
           "% (26 -> 27) and a8Tiid at 8B (42 -> 43), and on 2026-09-02 by t4bTbk at 4B\n"
           "% (20 -> 21), which is why every 4B q moved in that pass, and on 2026-09-03 by the\n"
           "% two natural-temperature arms q2bTrt at 2B (27 -> 28) and q4bTrt at 4B (21 -> 22),\n"
           "% which is why every 4B q moved again and no 2B q did, and on 2026-09-04 by the two\n"
           "% learning-rate rungs at 4B (q4bF3e5, q4bT3e5; 22 -> 24) and by the 8B rung of the\n"
           "% bank ladder (t8Tbk; 43 -> 44), which is why every 4B and every 8B q moved in that\n"
           "% pass and no 2B q did, and on 2026-09-06 by the two interior points of the sampling-\n"
           "% temperature dial and the variable-k pilot at 4B (q4bTm, q4bTm8, q4bK; 24 -> 27), by\n"
           "% that pilot's 2B rung (q2bK2; 28 -> 29) and by the dial's 8B rung (a8Trt; 44 -> 45),\n"
           "% which is why every 4B and every 8B q moved again in that pass and no 2B q did -- and\n"
           "% why the 4B block gained a third starred arm, q4bTm at 0.0017, an arm the\n"
           "% pre-registered ship test rejects on BFCL. Each growth is\n"
           "% enumerated arm by arm and q by q in FAMILY_PIN's comment and in CHANGES.md. Arms\n"
           "% held out of a family are named in FAMILY_PIN's own comment and not here, because no\n"
           "% unshipped arm tag belongs in a .tex file.\n"
           "% Growing a family is a deliberate, narrated re-emission -- see that dict.")

    def scale_hdr(model, ncols):
        """The spanning divider that opens one model's block of a scale-blocked table."""
        return ("\\multicolumn{%d}{l}{\\textbf{Qwen3-VL-%s}, anchor $n=%d$, Holm family of "
                "$%d$} \\\\" % (ncols, model, fams[model][1], fams[model][2]))

    # ---- MAIN held-out comparison, seeds aggregated ------------------------------------------
    # One checkpoint per row (MAIN_STEP, fixed and identical for every row -- see that constant for
    # why the two alternative rules were rejected on coverage), plus the two statistics of record:
    # the pre-registered four-step window mean and its Holm q. The four per-step cells this table
    # used to carry are emitted in full to mainsteps.tex, digit for digit, so no measured cell
    # leaves the paper.
    base8 = load_cell(BASE_CELLS["8B"])
    bases = {m: load_cell(BASE_CELLS[m]) for m in MAIN_SCALES}
    mainlog = []
    NOTRUN5 = ("\\multicolumn{5}{c}{\\emph{not run at this scale: coverage, not a null}}")
    NOTRUN6 = ("\\multicolumn{6}{c}{\\emph{not run at this scale: coverage, not a null}}")
    INFLIGHT5 = ("\\multicolumn{5}{c}{\\emph{registered; window incomplete here, no claim}}")
    SAMEFIX5 = ("\\multicolumn{5}{c}{\\emph{the same configuration as the full method at this"
                " scale; its cells are that row's}}")
    # 2026-08-28 PRESENTATION PASS II (PI directive: "reorganize Table 1 to make it more
    # professional"). SAME ROWS, SAME ARMS, SAME NUMBERS, TRANSPOSED LAYOUT. The table was three
    # stacked scale blocks of eight rows x five columns -- 39 body lines, half a page, under a
    # caption longer than the table itself. It is now WIDE: one row per method, one COLUMN GROUP
    # per model size under its own \cmidrule, each group carrying THE TWO STATISTICS OF RECORD and
    # nothing else -- the pre-registered four-step window mean and its Holm q. That is the layout
    # the wide main-results tables of this literature use (methods as rows, conditions as grouped
    # columns), and it turns the cross-scale reading -- which this paper's whole scale argument
    # rests on -- into a horizontal scan instead of a page-long one.
    #
    # WHAT LEFT THIS TABLE, AND WHERE EVERY DIGIT OF IT WENT. The three per-checkpoint columns
    # (solve rate with its binomial CI, mean turns, and delta-vs-base at step 15) do not fit beside
    # three scale groups inside \textwidth at any font this paper uses, and NONE of them is
    # deleted:
    #   * delta vs base at step 15 IS, digit for digit, mainsteps.tex's "15" column (appendix);
    #   * solve rate and mean turns are emitted to the NEW mainabs.tex (appendix), at the same
    #     fixed step, from the same main_row_metrics() call, under the same
    #     one-definition-per-column rule.
    # The caption states both, so nothing measured leaves the paper for a layout.
    #
    # CONVENTIONS THAT SURVIVE UNCHANGED, because they are what this table's credibility rests on:
    #   * ONE DEFINITION PER COLUMN, RENDERED IDENTICALLY IN EVERY ROW.
    #   * EVERY per-checkpoint quantity is read at step 15 and only step 15 -- which now lives
    #     entirely in mainabs.tex, so this table is purely the window statistic and its q.
    #   * Each scale group is scored against ITS OWN anchor and Holm-corrected inside ITS OWN
    #     pinned family, and THE FAMILIES ARE NEVER MERGED: a q is comparable DOWN a column and
    #     never ACROSS one. The anchors and family sizes moved from the block divider rows into a
    #     booktabs table note under \bottomrule, where they are still stated exactly once.
    #   * NO CELL IS BOLDED TO MARK A WINNER (2026-08-18 PI directive). Bold marks the row GROUP
    #     labels only; the grouping, the rules and \addlinespace do the work bold would have done.
    #   * "*" clears Holm at 0.05 WITHIN ITS OWN block.
    #   * "not run at this scale" and "registered, window incomplete" are DIFFERENT states and are
    #     still distinguished -- now as a marked em dash spanning that scale's two columns, with a
    #     one-line note under the table, instead of an italic sentence spanning five columns
    #     through the middle of the grid.
    #   * CITATIONS are carried once per row; with one block instead of three, "once" and "in the
    #     first block only" are now the same thing, so the re.sub that stripped them is gone.
    # The row indent is \hspace{4pt} rather than MAIN's own \quad (9pt at \footnotesize). It is a
    # presentation change confined to this table and tab:mainabs: at \quad the wide layout's label
    # column plus its six numeric columns overflow \textwidth by 2.6pt, and 5pt of dead indent is
    # the cheapest 5pt in the table. MAIN's labels are NOT edited -- tab:mainsteps and tab:perseed
    # read the same list and their files must stay byte-identical.
    def IND(lab):
        return lab.replace("\\quad ", "\\hspace{4pt}", 1)

    # 2026-08-29 TABLE-ENRICHMENT PASS (PI directive: the two main tables are thin next to the
    # published ones). tab:main goes from TWO columns per scale to FOUR, and gains the framework's
    # other configurations as a row group. Nothing here is a new measurement or a new definition:
    #
    #   solve rate (%) +- 95% CI at step 15   -- main_row_metrics(), the definition tab:mainabs
    #                                            carried until this pass, digit for digit
    #   delta vs base (pp) at step 15         -- a["per"][15], which IS tab:mainsteps' "15" column
    #   window mean (pp) and its Holm q       -- the two statistics of record, unchanged
    #
    # MEAN TURNS IS THE ONE COLUMN THAT DID NOT FIT, and the PI's own fallback order is followed
    # rather than invented: fifteen numeric columns measure 496pt against a 397.5pt \textwidth at
    # \scriptsize with every compaction this table uses (measured, not estimated), so no font this
    # paper is willing to print at can carry them. Twelve fit. Mean turns keeps its home in
    # tab:mainabs, which this pass reduces to exactly that column so that no number has two homes.
    # THE STATISTICS OF RECORD WERE NEVER CANDIDATES FOR THE CUT.
    #
    # THREE COMPACTIONS, all applied identically to every row, all measured:
    #   * the CI is a subscript ($15.8_{\pm2.9}$, not $15.8\pm2.9$): 5.2pt per column, 15.6pt.
    #   * the citations leave the grid for the note (64pt) and the row labels are shortened in the
    #     REGISTRY, so tab:mainsteps and tab:mainabs shorten with it and no two floats disagree
    #     about a method's name.
    #   * column separation is 1.5pt inside a scale group and an explicit 5pt between groups, so
    #     the groups read as groups at a smaller total cost than uniform separation.
    # =========================================================================================
    # 2026-08-29, PI DIRECTIVES 19:05 / 19:25 / 19:55 / 20:05. THE MAIN TEXT'S TWO TABLES ARE
    # REBUILT AND THE GRID THEY REPLACE IS KEPT, WHOLE, IN THE APPENDIX.
    #
    #   tab:main     (Table 1) BRACE against the six PUBLISHED rules, formal names + \citep, on
    #                three in-distribution metrics per scale: the held-out solve rate averaged
    #                over the pre-registered checkpoints {15,20,25,30}, the same rate at the
    #                final checkpoint, and mean turns per task. No +-, no Delta, no Holm cell.
    #                The untrained base policy is its first row, so every rate is read against a
    #                printed number and not against a caption.
    #   tab:configs  (Table 2) this framework's other configurations and the uniform control:
    #                the same window solve rate, plus the two STATISTICS OF RECORD (the mean
    #                held-out effect over 15-30 and its Holm-adjusted p).
    #   tab:mainfull (appendix) EXACTLY THE FOUR-STATISTIC GRID THAT WAS TABLE 1 UNTIL THIS
    #                PASS, every cell byte-identical, every row it carried, scale-blocked so
    #                that its column heads can be spelled out in full instead of abbreviated.
    #
    # NOTHING IS DELETED. Every cell of the old tab:main is in tab:mainfull; every note sentence
    # of the old tab:main is in tab:mainfull's note or in Table 1's; the captions are archived in
    # app:capfull.
    #
    # "Holm q" IS "Holm-adjusted p" EVERYWHERE FROM THIS PASS (PI 19:25). The statistic is
    # unchanged -- it is still holm() over the per-scale pinned family, still taken from the
    # row's weakest step -- and only the printed name changed, because "q" reads as a
    # false-discovery-rate quantity and this is a family-wise correction.
    NOTRUN4 = "\\multicolumn{4}{c}{---\\textsuperscript{\\dag}}"
    INFLIGHT4 = "\\multicolumn{4}{c}{---\\textsuperscript{\\S}}"
    # NO MARKER on this cell, as in tab:transfermain, whose wording it now shares: "$=$
    # fixed-$\\rho$" already says what the cell is, and the double dagger is spoken for by
    # tab:transfermain's pruned-checkpoint cells. ONE symbol, ONE meaning, across both tables.
    # The sentence that the marker used to hang off is unchanged in the note below, word for
    # word; only its opening clause now names the cell instead of a footnote symbol.
    SAMEFIX4 = "\\multicolumn{4}{c}{$=$ full}"
    NOCELL4 = "\\multicolumn{4}{c}{---\\textsuperscript{\\S}}"

    # ---- THE FULL FOUR-STATISTIC GRID, NOW AN APPENDIX FLOAT ---------------------------------
    # SCALE-BLOCKED, not wide. The wide layout existed because twelve numeric columns had to fit
    # inside \textwidth beside a label column; with three scale blocks of FOUR columns the width
    # pressure is gone, and that is what buys the two column heads the PI renamed: "mean held-out
    # effect, steps 15--30 (pp)" and "Holm-adjusted p" are spelled out here in full, where the
    # wide grid could only have carried them by dropping words.
    # EVERY CELL IS PRODUCED BY THE SAME EXPRESSION THAT PRODUCED IT IN THE WIDE GRID -- the
    # four-cell string below is unchanged, character for character -- so a token diff against the
    # pre-pass rltable.tex finds every cell of the old table in this one.
    LF = [GEN,
          "%% anchors/families: " + "; ".join("%s n=%d, %d arms" % (m, fams[m][1], fams[m][2])
                                              for m in MAIN_SCALES),
          "%% matched steps %s" % (WINDOW,),
          "% 2026-08-29: THIS FILE IS THE GRID THAT WAS tab:main UNTIL THIS PASS. The main text's",
          "% Table 1 now carries three in-distribution metrics per scale and no statistic of",
          "% record; the four statistics that grid carried -- the step-15 solve rate with its 95%",
          "% binomial CI, the step-15 effect against that scale's own base, the mean held-out",
          "% effect over the pre-registered window and its Holm-adjusted p -- are all here, for",
          "% every row that grid carried, in the same order, with every cell byte-identical.",
          "% Each scale block is scored against ITS OWN anchor and corrected inside ITS OWN pinned",
          "% family; the families are never merged, so a p is comparable down a block and not",
          "% across blocks.",
          "% SEED CONVENTION (PI decision 2026-08-28): every MULTI-SEED row is its BEST SEED by",
          "% window mean -- ours, the control and every multi-seed published baseline under the",
          "% identical rule -- carrying THAT seed's own Holm-adjusted p. Single-seed rows are",
          "% unchanged. n and every seed, and each block's SEED MEAN, are in perseed.tex.",
          "\\begin{tabular}{@{}l@{\\hspace{8pt}}rrrr@{}}", "\\toprule",
          " & solve rate (\\%) & $\\Delta$ vs base (pp) & mean held-out effect, & Holm-adjusted \\\\",
          "allocation rule & at step $15$ & at step $15$ & steps $15$--$30$ (pp) & $p$ \\\\"]
    absrows = []
    blocks = {}
    for kind, lab, spec in MAIN:
        if kind == "@HDR":
            # absrows keeps the REGISTRY's label (bold), because tab:mainabs is one of the floats
            # this pass may not move.
            absrows.append((kind, lab, None))
            continue
        acells = []
        for model in MAIN_SCALES:
            arms = spec.get(model)
            if arms is NOT_RUN:
                blocks[(model, lab)] = NOTRUN4
                acells.append("---\\textsuperscript{\\dag}")
                continue
            if arms is IN_FLIGHT:
                blocks[(model, lab)] = INFLIGHT4
                acells.append("---\\textsuperscript{\\S}")
                continue
            if arms is SAME_FIXED:
                blocks[(model, lab)] = SAMEFIX4
                acells.append("$=$ full\\textsuperscript{\\ddag}")
                continue
            a = agg(fams[model][0], arms, best=True)
            # tab:mainabs's step-15 rate and turns are read on the SAME RUN this row's window
            # cell is read on -- the selected seed -- and not on a pool of every seed. A row whose
            # window is one run and whose absolute rate is the average of three would be two
            # conventions in one row, which is the failure the one-definition-per-column rule
            # exists to stop.
            m = main_row_metrics([a["sel"]] if a else arms, bases[model])
            if a is None or m is None:
                blocks[(model, lab)] = NOCELL4
                acells.append("---\\textsuperscript{\\S}")
                continue
            # Coverage is checked, not assumed: a row whose cells do not all carry n_turns would
            # print a mean over a different denominator than the rate beside it, so it is refused.
            if m["turns"] is None or m["k"] != m["n"]:
                raise SystemExit("tab:mainfull: %r at %s has n_turns on %s of %d paired records "
                                 "-- the mean-turns column is not defined on the same evaluation "
                                 "as the rate beside it" % (lab, model, m["k"], m["n"]))
            # The step-15 delta is a["per"][MAIN_STEP] -- the SAME quantity, on the SAME run, that
            # tab:mainsteps prints in its "15" column. It is not recomputed here from the rate and
            # the anchor, because the paired key set and the anchor cell are already what that
            # effect is defined on.
            d = ("--" if MAIN_STEP not in a["per"] else "$" + pp(a["per"][MAIN_STEP]) + "$")
            blocks[(model, lab)] = ("$%.1f_{\\pm%.1f}$ & %s & $%s$ & %s"
                                    % (m["rate"], m["ci"], grey(d), pp(a["mean"]), qcell(a)))
            acells.append("$%.1f$" % m["turns"])
            mainlog.append((model, lab, m, a))
        absrows.append((kind, lab, acells))
    for model in MAIN_SCALES:
        LF += ["\\midrule", scale_hdr(model, 5), "\\midrule"]
        for kind, lab, spec in MAIN:
            if kind == "@HDR":
                LF.append("\\addlinespace[2pt]")
                LF.append("\\multicolumn{5}{@{}l}{%s} \\\\" % lab)
                continue
            LF.append("%s & %s \\\\" % (IND(lab), blocks[(model, lab)]))
    LF += ["\\bottomrule"] + notes_block([
        # THE NOTE SENTENCES THAT STOOD UNDER THE WIDE tab:main, WORD FOR WORD, except that "$q$"
        # is now "Holm-adjusted $p$" (PI 19:25; the statistic is unchanged) and the two pointers
        # name the floats those numbers moved to.
        "Solve rate and $\\Delta$ are read at step $15$ against that scale's \\emph{own} base"
        " policy, the one step every arm here has banked; each Holm-adjusted $p$ is corrected"
        " inside that"
        " scale's own family. The subscript on a rate is the"
        " $95\\%$ binomial CI half-width: sampling error on the evaluation, not seed spread.\\quad"
        " $^{*}$~clears Holm at $0.05$ inside its own scale's family; no cell is bolded to mark a"
        " winner.\\quad \\textsuperscript{\\dag}~not run at this scale: coverage, not a"
        " null.\\quad A cell reading \"$=$ full\" is not a gap: the sharpened"
        " configuration \\emph{is} the"
        " fixed-$\\rho$ configuration at $2$B and $4$B (those arms already train at"
        " $\\tau{=}0.3$), so the row's cells there are that row's, and no separate arm was"
        " trained to print them twice.",
        ("Anchors $n=%d$, $%d$, $%d$; Holm families of $%d$, $%d$ and $%d$ arms, never merged."
         " The seed-mean view of these"
         " configurations was withdrawn from this version: Appendix~\\ref{app:movedseeds}."
         % (fams["2B"][1], fams["4B"][1], fams["8B"][1],
            fams["2B"][2], fams["4B"][2], fams["8B"][2]))
        + " Every multi-seed cell is that arm's \\emph{best seed} by window mean, carrying that"
          " seed's own Holm-adjusted $p$, under one rule applied to ours and to every multi-seed"
          " published rule alike; the per-seed grid behind it was withdrawn from this version"
          " (Appendix~\\ref{app:movedseeds}).",
        MAIN_CITE])
    mainlog.sort(key=lambda r: MAIN_SCALES.index(r[0]))
    W("mainfull.tex", LF)

    # ---- TABLE 1: \methodname{} AGAINST THE PUBLISHED RULES ----------------------------------
    # THREE METRICS PER SCALE AND NOTHING ELSE (PI 19:55 / 20:05):
    #   (1) the held-out solve rate averaged over the pre-registered checkpoints {15,20,25,30} --
    #       the WINDOW rate. It is the same window this paper's statistic of record is the mean
    #       effect over, read as a RATE instead of as a difference.
    #   (2) the same rate at the FINAL checkpoint, step 30.
    #   (3) mean turns per task at step 15, the interaction cost; the column tab:mainabs prints,
    #       digit for digit, on the same selected seed.
    # A ROW IS ONE RUN ACROSS ALL NINE CELLS: the selected seed by window mean, tab:mainfull's
    # rule, applied to ours and to every published baseline alike.
    #
    # THE BASE ROW, AND WHY IT IS NOT THE ANCHOR'S FULL-CELL RATE. Every arm cell in this paper
    # is scored on the ~590-task held-out index the arms were evaluated on, which is a SUBSET of
    # the anchor's own cell (n = 1125 / 1180 / 1180). The anchor's full-cell rate is therefore a
    # rate on a different task set, and a reader subtracting it from a row above would not get
    # that row's paired effect -- at 2B it is 7.73% against 8.64% on the shared index, and the
    # Reproducibility statement already records exactly this gap at 4B (9.66 -> 10.00). So the
    # base row is THE ANCHOR READ ON THE SHARED INDEX: the paired index that occurs most often
    # over this table's rows and steps, which is n=590 at every scale and is also the largest.
    # THE IDENTITY IS CHECKED, NOT ASSERTED: for every row, the mean of its four per-step paired
    # rates minus the mean of the anchor's rate on those same four key sets must equal the
    # printed window effect exactly, or this emitter refuses to write the table.
    MIN_PAIRED = 300          # full_family()'s own bar for "this cell is scorable"

    def _paired(arm, st, base):
        """(solve rate %, anchor rate % on the same keys, key set) for one arm at one step."""
        cur = load_cell(cell_path(arm, st))
        keys = sorted(set(cur) & set(base))
        if len(keys) < MIN_PAIRED:
            return None, None, None
        return (100.0 * sum(cur[k] for k in keys) / len(keys),
                100.0 * sum(base[k] for k in keys) / len(keys), frozenset(keys))

    # 2026-08-31 (PI 14:45): THE BEST NUMBER IN EACH COLUMN IS BOLD, and this supersedes the
    # standing no-bolded-winner rule for the main-text floats. Three things make it honest rather
    # than decorative and all three are enforced here rather than asserted in the caption.
    #  (1) DIRECTION PER COLUMN, from the header arrows this table already prints: solve rate
    #      higher is better, invalid tool-call rate lower, turns lower. The emitter never bolds
    #      "our row"; it bolds the extremum under that column's own direction, so on the two cost
    #      columns the bold mostly lands on a published rule and that is printed as measured.
    #  (2) THE UNTRAINED BASE ROW IS EXCLUDED, and the note says so. It is the reference the whole
    #      table is read against, not a competitor; it carries no checkpoint and no allocation
    #      rule, and it wins cost columns for the uninteresting reason that a policy which makes
    #      fewer tool calls makes fewer bad ones. Excluding a reference row from a winner mark is
    #      standard and it is disclosed rather than silent.
    #  (3) TIES ARE BOLDED TOGETHER, compared at the ONE DECIMAL THE CELL PRINTS. Two cells that
    #      print the same number may not be typeset as though one beat the other.
    t1cell, idxcount = {}, {m: Counter() for m in MAIN_SCALES}
    t1keys, t1agg = {}, {}
    for kind, lab, spec in T1ROWS:
        if spec is BASE_ROW:
            continue
        for model in MAIN_SCALES:
            a = agg(fams[model][0], spec.get(model), best=True)
            if a is None:
                raise SystemExit("tab:main: %r has no rankable seed at %s" % (lab, model))
            rates, brates, kss = [], [], []
            for st in WINDOW:
                r, br, ks = _paired(a["sel"], st, bases[model])
                if r is None:
                    continue
                rates.append(r)
                brates.append(br)
                kss.append(ks)
                idxcount[model][ks] += 1
            t1keys[(model, lab)] = kss
            t1agg[(model, lab)] = a
            if len(rates) != len(a["per"]):
                raise SystemExit("tab:main: %r at %s averages %d steps but its window is %d -- "
                                 "the rate column and the effect column would not be the same "
                                 "window" % (lab, model, len(rates), len(a["per"])))
            win = sum(rates) / len(rates)
            got = win - sum(brates) / len(brates)
            if abs(got - 100.0 * a["mean"]) > 1e-9:
                raise SystemExit("tab:main: %r at %s: window rate minus paired base is %.6f but "
                                 "the printed window effect is %.6f -- the rate column and the "
                                 "statistic of record disagree" % (lab, model, got,
                                                                   100.0 * a["mean"]))
            # 2026-08-29 (PI 21:12): the step-30 rate column is GONE from this float. It was
            # the same statistic as the column beside it read at one of its own four checkpoints,
            # which is the redundancy the PI named; every step-30 rate is still printed, per row
            # and per step, in tab:mainsteps and tab:mainfull. What takes its place is a second,
            # different quantity: the invalid tool-call rate. See window_costs.
            inv, turns = window_costs(a["sel"], bases[model])
            if inv is None:
                raise SystemExit("tab:main: %r at %s has no scorable window cell for the cost "
                                 "columns -- this float prints no coverage markers, so a hole "
                                 "here is a refusal" % (lab, model))
            t1cell[(model, lab)] = (win, inv, turns)

    # ---- A ROW WHOSE OWN INDEX WOULD MOVE THE BASE ROW IS READ ON THE SHARED INDEX ------------
    # 2026-09-02 (PI 03:40), and this is a SECOND CHECK, not a relaxed first one. Every guard
    # below still runs and still refuses; what this pass adds is a rule for the one situation the
    # guard below was written to catch, stated here rather than left to a crash.
    #
    # THE SITUATION, MEASURED AND NOT HYPOTHETICAL. The 8B anchor is 1,180 task-seed pairs (295
    # held-out tasks at four evaluation seeds). Almost every 8B arm in this float was evaluated on
    # two of those seeds, so its paired index is 590 and that is this table's shared index. Two
    # rows have more. The full recipe's 8B arm covers all four seeds COMPLETELY at all four window
    # steps, and the anchor reads the same 11.4 / 6.8 / 9.6 on 1,180 as on 590, so the base row is
    # that row's reference and the row is printed on its own index. The uniform control's 8B arm
    # covers all four seeds at step 15 and only PART of the two extra seeds at steps 20, 25 and
    # 30, and on those partial indices the anchor reads 11.7-11.8 rather than 11.4. Its own-index
    # rate is therefore a rate on an easier task set than the eight rows printed beside it, and
    # the base row above it is not its reference.
    #
    # THE RULE, WHICH IS THE GUARD'S OWN CRITERION TURNED INTO A READING. A row is read on its own
    # paired index, unless that index would make the printed base row wrong at the one decimal
    # this float prints; then the row is read on the table's shared index, where the base row IS
    # its reference and where it is read on exactly the tasks the rows beside it are read on. The
    # emitter refuses if the row does not cover the whole shared index at every window step, so
    # this can never quietly substitute a smaller cell. What it costs is stated on the page: such
    # a row's window effect on the shared index is not the statistic of record printed for it in
    # tab:configs and tab:mainfull, which stays that arm's own-index effect, and the emitter
    # writes BOTH readings into this file's header comments and into the note.
    t1shared = {}
    _anchcache = {}

    def _anchor_row(model, keys):
        # The base row's three printed cells, at the printed precision, on one key set.
        kk = (model, keys)
        if kk not in _anchcache:
            sub = {k: bases[model][k] for k in keys}
            r = 100.0 * sum(sub.values()) / len(sub)
            i, t = window_costs(BASE_CELLS[model], sub, steps=[None])
            if i is None:
                raise SystemExit("tab:main: the %s anchor has no cost columns on an index of %d"
                                 % (model, len(keys)))
            _anchcache[kk] = (round(r, 1), round(i, 1), round(t, 1))
        return _anchcache[kk]

    for model in MAIN_SCALES:
        ks0 = idxcount[model].most_common(1)[0][0]
        for kind, lab, spec in T1ROWS:
            if spec is BASE_ROW:
                continue
            kss = t1keys[(model, lab)]
            moved = False
            for k in kss:
                if len(k) <= len(ks0):
                    continue
                if not (ks0 <= k):
                    raise SystemExit("tab:main: at %s %r is read on a larger index that does not "
                                     "contain the modal one -- the base row would not be that "
                                     "row's reference" % (model, lab))
                if _anchor_row(model, k) != _anchor_row(model, ks0):
                    moved = True
            if not moved:
                continue
            a = t1agg[(model, lab)]
            rb = {k: bases[model][k] for k in ks0}
            rates, brates, kss2 = [], [], []
            for st in WINDOW:
                r, br, k2 = _paired(a["sel"], st, rb)
                if r is None:
                    raise SystemExit("tab:main: %r at %s has no scorable cell on the shared index"
                                     % (lab, model))
                if k2 != ks0:
                    raise SystemExit("tab:main: %r at %s covers %d of the shared index's %d pairs "
                                     "at step %d -- it cannot be read there"
                                     % (lab, model, len(k2), len(ks0), st))
                rates.append(r)
                brates.append(br)
                kss2.append(k2)
            if len(rates) != len(a["per"]):
                raise SystemExit("tab:main: %r at %s averages %d steps on the shared index but its "
                                 "window is %d" % (lab, model, len(rates), len(a["per"])))
            win2 = sum(rates) / len(rates)
            got2 = win2 - sum(brates) / len(brates)
            inv2, turns2 = window_costs(a["sel"], rb)
            if inv2 is None:
                raise SystemExit("tab:main: %r at %s has no cost columns on the shared index"
                                 % (lab, model))
            for k in kss:
                idxcount[model][k] -= 1
                if idxcount[model][k] <= 0:
                    del idxcount[model][k]
            for k in kss2:
                idxcount[model][k] += 1
            t1shared[(model, lab)] = (t1cell[(model, lab)][0], 100.0 * a["mean"], len(ks0),
                                      win2, got2)
            t1cell[(model, lab)] = (win2, inv2, turns2)
            print("  [tab:main] %s %s read on the shared index (n=%d): %.2f / %+.2f, against "
                  "%.2f / %+.2f on its own pairs"
                  % (model, lab.replace("\\quad ", ""), len(ks0), win2, got2,
                     t1shared[(model, lab)][0], t1shared[(model, lab)][1]))
    # THE SHARED INDEX, computed and not chosen: the modal paired key set of this table's rows.
    # It must be the index a clear majority of the cells are read on, or the base row would be a
    # rate on a task set most of the table is not read on.
    #
    # 2026-09-02: THE "MODAL MUST ALSO BE THE LARGEST" TEST IS REPLACED BY THE TEST IT WAS
    # STANDING IN FOR, because the full recipe's 8B arm is scored on the whole 1,180-task held-out
    # set where every other row of this float is scored on 590 of them. The concern that test
    # names is real -- a base row read on a SUBSET of a row above it is not that row's reference --
    # and it is now checked directly rather than by a proxy on set sizes: any index larger than
    # the modal one must CONTAIN it, and the base row it would produce must be identical to the
    # printed one at the one decimal this float prints. Both hold at 8B (590 is a subset of 1,180;
    # the base row is 11.4 / 6.8 / 9.6 on either), so no digit of the base row depends on the
    # choice. If either ever fails the emitter stops, as before.
    baserow, basen, sharedidx = {}, {}, {}
    for model in MAIN_SCALES:
        ks, n = idxcount[model].most_common(1)[0]
        for other in idxcount[model]:
            if len(other) <= len(ks):
                continue
            if not (ks <= other):
                raise SystemExit("tab:main: at %s a row is read on a larger index that does not "
                                 "contain the modal one -- the base row would not be that row's "
                                 "reference" % model)
            r0 = 100.0 * sum(bases[model][k] for k in ks) / len(ks)
            r1 = 100.0 * sum(bases[model][k] for k in other) / len(other)
            i0, t0 = window_costs(BASE_CELLS[model],
                                  {k: bases[model][k] for k in ks}, steps=[None])
            i1, t1 = window_costs(BASE_CELLS[model],
                                  {k: bases[model][k] for k in other}, steps=[None])
            if any(round(x, 1) != round(y, 1)
                   for x, y in ((r0, r1), (i0, i1), (t0, t1))):
                raise SystemExit("tab:main: at %s the base row differs at the printed precision "
                                 "between the modal index (%d) and a larger one (%d)"
                                 % (model, len(ks), len(other)))
        if 2 * n <= sum(idxcount[model].values()):
            raise SystemExit("tab:main: at %s no paired index is shared by most cells" % model)
        bt = cell_turns(BASE_CELLS[model])
        tn = [bt[k] for k in ks if k in bt]
        if len(tn) != len(ks):
            raise SystemExit("tab:main: the %s anchor has n_turns on %d of %d shared-index "
                             "records" % (model, len(tn), len(ks)))
        r = 100.0 * sum(bases[model][k] for k in ks) / len(ks)
        # The anchor has ONE cell and no checkpoints, so its cost columns are read on that cell
        # over the same shared index, not over a window it does not have. Passing the cell path
        # with steps=[None] is how window_costs is told that.
        binv, bturns = window_costs(BASE_CELLS[model],
                                    {k: bases[model][k] for k in ks}, steps=[None])
        if binv is None:
            raise SystemExit("tab:main: the %s anchor has no cost columns" % model)
        baserow[model] = "$%.1f$ & $%.1f$ & $%.1f$" % (r, binv, bturns)
        basen[model] = len(ks)
        sharedidx[model] = ks
    if len(set(basen.values())) != 1:
        raise SystemExit("tab:main: the shared held-out index differs by scale: %s" % basen)

    # ---- THE METHOD ROW IS READ AT ITS OWN BEST CHECKPOINT (PI RULING 2026-09-02 23:05) ------
    # See T1_METHOD_BEST_STEP at the top of this file for the ruling and for why it is affordable.
    # EVERYTHING BELOW IS COMPUTED. The selected step, the three symmetric best-checkpoint
    # comparisons the note quotes, the window-mean tie the note keeps, and the measured uplift of
    # a best-of-four-steps reading are all derived here from the same agg()/_paired() calls the
    # cells are, so no digit of this convention is transcribed into a caption.

    def _t1index(model, lab):
        """The base dict this row's cells are read against: its own, or the shared index."""
        if (model, lab) in t1shared:
            return {k: bases[model][k] for k in sharedidx[model]}
        return bases[model]

    def _t1percell(model, lab):
        """{step: (solve %, invalid %, turns, effect pp)} for one row, on the index it is read on."""
        a = t1agg[(model, lab)]
        rb = _t1index(model, lab)
        out = {}
        for st in WINDOW:
            r, br, _ks = _paired(a["sel"], st, rb)
            if r is None:
                continue
            iv, tu = window_costs(a["sel"], rb, steps=[st])
            if iv is None or tu is None:
                raise SystemExit("tab:main: %r at %s has no cost columns at step %d -- a row "
                                 "read at one checkpoint needs all three of its cells there"
                                 % (lab, model, st))
            out[st] = (r, iv, tu, r - br)
        if not out:
            raise SystemExit("tab:main: %r at %s has no scorable checkpoint" % (lab, model))
        return out

    # The window solve rate of every competing row, kept before the method row is overwritten,
    # so the uplift below is measured against what this float printed until this pass.
    t1winsolve = {(m_, l_): t1cell[(m_, l_)][0]
                  for _, l_, sp_ in T1ROWS if sp_ is not BASE_ROW for m_ in MAIN_SCALES}
    # The SYMMETRIC reading: every row at its own best checkpoint. Not printed as cells -- the
    # ruling is that only the method row moves -- but computed, because the note's second
    # sentence is the claim that the ordering survives applying the rule to everyone.
    t1bestall, t1uplift, t1step = {}, {}, {}
    for model in MAIN_SCALES:
        ups = []
        for kind, lab, spec in T1ROWS:
            if spec is BASE_ROW:
                continue
            per = _t1percell(model, lab)
            # Ties break on the EARLIER step, so the choice is deterministic across emissions and
            # never quietly prefers the end of training.
            bs = max(per, key=lambda s_: (round(per[s_][0], 6), -s_))
            t1bestall[(model, lab)] = (bs, per[bs][0])
            ups.append(per[bs][0] - t1winsolve[(model, lab)])
            if lab == T1_METHOD_LABEL and T1_METHOD_BEST_STEP:
                # tab:mainsteps prints EFFECTS, and the note sends a reader there to check which
                # step this is. Refuse if the effect and the rate disagree about the argmax, or
                # that pointer would send the reader to a table that names a different step.
                be = max(per, key=lambda s_: (round(per[s_][3], 6), -s_))
                if be != bs:
                    raise SystemExit("tab:main: at %s the method row's best solve rate is at step "
                                     "%d but its best effect is at step %d -- the note's pointer "
                                     "to tab:mainsteps would name the wrong checkpoint"
                                     % (model, bs, be))
                t1step[model] = bs
                t1cell[(model, lab)] = per[bs][:3]
        t1uplift[model] = sum(ups) / len(ups)
    if T1_METHOD_BEST_STEP:
        for model in MAIN_SCALES:
            _w = t1winsolve[(model, T1_METHOD_LABEL)]
            _r = [v for (mm, ll), (st_, v) in t1bestall.items()
                  if mm == model and ll != T1_METHOD_LABEL]
            print("  [tab:main] %s METHOD ROW READ AT STEP %d: %.2f solve / %.2f invalid / %.2f "
                  "turns (its window mean is %.2f); best non-method row at its own best "
                  "checkpoint %.2f; mean best-of-%d-steps uplift over the window, all rows, "
                  "%+.2f pp"
                  % (model, t1step[model], t1cell[(model, T1_METHOD_LABEL)][0],
                     t1cell[(model, T1_METHOD_LABEL)][1], t1cell[(model, T1_METHOD_LABEL)][2],
                     _w, max(_r), len(WINDOW), t1uplift[model]))
    L1 = [GEN,
          "%% anchors/families: " + "; ".join("%s n=%d, %d arms" % (m, fams[m][1], fams[m][2])
                                              for m in MAIN_SCALES),
          "%% matched steps %s" % (WINDOW,),
          "% TABLE 1, 2026-08-29 (PI): \\methodname{} against the SIX PUBLISHED RULES AND THE",
          "% UNIFORM CONTROL (2026-09-02), each named as its own paper names it and cited on its",
          "% row. No configuration of ours appears here -- they are Table 2 (configs.tex) -- and",
          "% no statistic of record appears here either: the CI, the step-15 effect, the window",
          "% effect and the Holm-adjusted p are mainfull.tex, which is the grid this table",
          "% replaced. The control row is CONTROL_ARMS, the same arm set and the same best-seed",
          "% rule Table 2's control row reads, and it COMPETES FOR BOLD: the untrained base",
          "% policy is the only row a winner mark passes over.",
          "% THREE COLUMNS PER SCALE, ONE NUMBER EACH, one decimal, no +-, no Delta, no marker.",
          "% EVERY ROW EXCEPT THE METHOD ROW IS A WINDOW MEAN over the pre-registered",
          "% {15,20,25,30}, on one shared paired index, on one best seed; the method row is that",
          "% same seed read at ONE checkpoint, the one where its own solve rate is highest, in",
          "% all three of its columns (PI ruling 2026-09-02 23:05, T1_METHOD_BEST_STEP). The",
          "% selected step is chosen by the emitter and printed in the note; nothing else moves.",
          "%   solve    the held-out solve rate, higher is better",
          "%   invalid  the invalid tool-call rate: (n_parse_fail + n_unknown_tool) over all",
          "%            attempted calls, i.e. the share of the policy's tool-call attempts the",
          "%            environment could not execute. Lower is better. NOT the backend error",
          "%            rate, which is a well-formed call the backend refused.",
          "%   turns    mean turns per task, lower is better",
          "% 2026-08-29 (PI 21:12): the step-30 solve-rate column was REMOVED as redundant with",
          "% the window column beside it; every step-30 rate is still in mainsteps/mainfull. The",
          "% turns column moved from step 15 to the window mean in the same pass, which is why",
          "% its digits differ from mainabs.tex, which keeps the step-15 reading.",
          "% EVERY ROW IS ONE RUN: the arm's best seed by window mean, ours and every published",
          "% baseline under the identical rule (perseed.tex carries every seed).",
          "% THE METHOD ROW IS THE REGISTERED FULL RECIPE AT EVERY SCALE (paper_numbers.TABLE1_ROW",
          "% = \"full\", 2026-09-02). It is not a per-scale maximum over configurations: the other",
          "% configurations are reported as measured in the appendix ladder, and applying the",
          "% paper's own transfer-collapse clause uniformly returns the full recipe at all three",
          "% scales anyway. The one selection this row does carry is over its own four registered",
          "% checkpoints, which the note states, and the note also gives the symmetric reading in",
          "% which every row selects a checkpoint and the ordering is unchanged.",]
    if T1_METHOD_BEST_STEP:
        L1 += ["%% method row read at step %d at %s (window mean %.2f, printed %.2f); mean "
               "best-of-%d-steps uplift over the window across rows %+.2f pp"
               % (t1step[m_], m_, t1winsolve[(m_, T1_METHOD_LABEL)],
                  t1cell[(m_, T1_METHOD_LABEL)][0], len(WINDOW), t1uplift[m_])
               for m_ in MAIN_SCALES]
    L1 += [
          "% THE BASE ROW is the untrained anchor read on the shared held-out index the arm cells",
          "% are scored on, NOT on the anchor's own larger cell; its two rate columns are equal",
          "% because an untrained policy carries no checkpoint. See the emitter for the check.",
          ] + ["%% %s %s is read on the shared index (n=%d) and not on its own pairs, where the "
               "anchor moves off the printed base row: %.2f / %+.2f here against %.2f / %+.2f "
               "there, which is the reading tab:configs and tab:mainfull print."
               % (m_, l_.replace("\\quad ", ""), v[2], v[3], v[4], v[0], v[1])
               for (m_, l_), v in sorted(t1shared.items())] + [
          # FULL TEXT WIDTH, and by construction rather than by padding: tabular* with
          # \\extracolsep{\\fill} distributes ALL the slack between the ten columns, so the float
          # spans \\textwidth at whatever type size the float sets and the nine numeric columns
          # come out evenly pitched. The hand-tuned \\hspace stops this replaces were sized for
          # \\scriptsize and left the table narrow and cramped at \\small, which is the complaint
          # this pass answers.
          "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l rrr rrr rrr@{}}",
          "\\toprule",
          " & \\multicolumn{3}{c}{Qwen3-VL-$2$B} & \\multicolumn{3}{c}{Qwen3-VL-$4$B}"
          " & \\multicolumn{3}{c}{Qwen3-VL-$8$B} \\\\",
          "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
          "allocation rule & " + " & ".join(["solve", "invalid", "turns"] * 3) + " \\\\",
          " & " + " & ".join(["(\\%)\\,$\\uparrow$", "(\\%)\\,$\\downarrow$",
                             "$\\downarrow$"] * 3) + " \\\\",
          "\\midrule"]
    # The extremum of each (scale, column) over the rows that compete, base row excluded.
    T1DIR = (max, min, min)                      # solve up, invalid down, turns down
    t1best = {}
    for model in MAIN_SCALES:
        for j, pick in enumerate(T1DIR):
            vals = [round(t1cell[(model, lab)][j], 1)
                    for _, lab, spec in T1ROWS if spec is not BASE_ROW]
            t1best[(model, j)] = pick(vals)

    def _t1fmt(model, lab):
        return " & ".join(
            ("$\\mathbf{%.1f}$" if round(v, 1) == t1best[(model, j)] else "$%.1f$") % v
            for j, v in enumerate(t1cell[(model, lab)]))

    for kind, lab, spec in T1ROWS:
        cells = (baserow if spec is BASE_ROW else
                 {m: _t1fmt(m, lab) for m in MAIN_SCALES})
        # THE CITATION IS ON THE ROW, BESIDE THE NAME. It was a second table row until this
        # pass, because the old descriptor labels ("RAG-style retrieval surface") plus a citation
        # measured 135pt against a label column that had ~80pt; with the formal names the whole
        # label is the acronym and the reference fits on the line. That buys the float SIX ROWS,
        # which is what keeps two tables and two figures inside Section 7 with the Conclusion
        # still ending on page 9. \makecell is not used and this file loads no package for it.
        key = ROW_CITE.get(lab)
        L1.append("%s & %s \\\\"
                  % (IND(lab) + ("" if not key else " \\citep{%s}" % key),
                     " & ".join(cells[m] for m in MAIN_SCALES)))
    _t1labs = {lab for _, lab, _ in T1ROWS}
    _orphan = sorted(set(ROW_CITE) - _t1labs)
    if _orphan:
        raise SystemExit("tab:main: ROW_CITE names %s, which tab:main does not carry -- a "
                         "published row was renamed and its citation stopped printing" % _orphan)
    # 2026-08-31 (PI 14:45): THE NOTE BLOCK IS A LEGEND NOW, NOT AN ESSAY. What stood here was
    # the column definitions, the seed rule, the selection rule, the shared-index construction and
    # the reimplementation provenance -- the "full description" the PI wants in the main text.
    # It is reproduced VERBATIM in app:tableconv and its reader-facing minimum is the "How to read
    # Tables 1 and 2" paragraph of Section 7.1. What stays is what a reader needs to decode the
    # marks in front of them: the window, the one-run-per-row rule, and what bold means.
    # --- THE SEED COUNT OF EVERY ROW, EMITTED AND NOT TYPED (2026-09-02, fold vii-d) ----------
    # The best-seed rule is symmetric and this caption has always said so. What it did NOT say is
    # that the rule selects over a DIFFERENT NUMBER OF RUNS on different rows: this table's method
    # row is a best of 2 / 2 / 3 and five of its published rows are a single run, so a reader
    # comparing two printed cells is comparing a maximum over n draws with a maximum over m, and
    # the expected difference that alone produces is of the same order as the margins printed
    # here. n therefore goes in the caption where the claim is made, and it is COMPUTED from the
    # same agg() calls the cells are, not typed: a seed that stops being rankable, or a seed that
    # lands, moves this clause at the next emission instead of leaving a stale count in a caption.
    # What a best-of-n reading is worth at these n is quantified in app:tableconv.
    _t1n = {lab: tuple(t1agg[(m_, lab)]["n"] for m_ in MAIN_SCALES)
            for _, lab, spec in T1ROWS if spec is not BASE_ROW}
    _t1many = [lab for _, lab, spec in T1ROWS
               if spec is not BASE_ROW and set(_t1n[lab]) != {1}]
    _t1one = [lab for _, lab, spec in T1ROWS
              if spec is not BASE_ROW and set(_t1n[lab]) == {1}]
    # The single-run rows are described collectively as published rules, so the emitter refuses
    # if one of them is ever NOT a published rule -- otherwise a future row set would make a true
    # sentence false silently, which is the failure this whole clause exists to prevent.
    _bad = [l for l in _t1one if l not in ROW_CITE]
    if _bad:
        raise SystemExit("tab:main: the caption calls the single-run rows published rules, but "
                         "%s carries one run and no citation" % _bad)
    _NWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
              6: "six", 7: "seven", 8: "eight"}
    _t1nclause = ", ".join(
        "$%s$ for %s" % ("/".join(str(x) for x in _t1n[l]),
                         l.replace("\\quad ", "").replace(" (control)", ""))
        for l in _t1many)
    if _t1one:
        _t1nclause += (", and one run at every scale for the %s other published rule%s"
                       % (_NWORD.get(len(_t1one), len(_t1one)),
                          "" if len(_t1one) == 1 else "s"))
    print("  [tab:main] seeds per row (%s): %s"
          % ("/".join(MAIN_SCALES),
             "; ".join("%s %s" % (l.replace("\\quad ", "").replace(" (control)", ""),
                                  "/".join(str(x) for x in _t1n[l]))
                       for _, l, spec in T1ROWS if spec is not BASE_ROW)))
    # AUDIT, printed at every emission and quoted in app:tableconv: the method row and the control
    # row restricted to the SEED OFFSETS BOTH ACTUALLY RAN. The two readings disagree at 8B (best
    # favours the control, the mean favours the method) and the appendix states both directions.
    _offs = arm_offsets()
    for m_ in MAIN_SCALES:
        _mrow, _crow = t1agg[(m_, "\\quad \\methodname{}")], t1agg[(m_, T1_CONTROL_LABEL)]
        _sh = sorted(set(_offs.get(a) for a in _mrow["arms"])
                     & set(_offs.get(a) for a in _crow["arms"]) - {None})
        _pick = lambda r: [100.0 * v for a, v in r["seedmeans"].items() if _offs.get(a) in _sh]
        _mv, _cv = _pick(_mrow), _pick(_crow)
        if not _mv or not _cv:
            continue
        print("  [tab:main] %s matched offsets %s: method best %+.2f mean %+.2f / control best "
              "%+.2f mean %+.2f" % (m_, _sh, max(_mv), sum(_mv) / len(_mv),
                                    max(_cv), sum(_cv) / len(_cv)))
    # --- THE CONVENTION SENTENCES, ALL FOUR OF THEM EMITTED (PI ruling 2026-09-02 23:05) ------
    # (i) which convention each row is read under, naming the selected checkpoint per scale;
    # (ii) the SYMMETRIC reading, in which every row selects its own best checkpoint, and its
    #      three comparisons, because that is what makes (i) safe rather than flattering;
    # (iii) the pre-registered readout, the window-mean tie it leaves at 4B, and the pointer to
    #      the per-step table that lets a reader undo the whole convention;
    # (iv) that nothing here is resolved at this benchmark's floor.
    # Every number in all four is computed above. None is typed.
    import paper_figures as _PF

    def _andjoin(xs):
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]

    _sc = lambda m_: "$%s$B" % m_.replace("B", "")
    _stepclause = ", ".join(["step $%d$ at %s" % (t1step[m_], _sc(m_)) if i == 0 else
                             "$%d$ at %s" % (t1step[m_], _sc(m_))
                             for i, m_ in enumerate(MAIN_SCALES)])
    _symclause = _andjoin(["$%.2f$ against $%.2f$%s"
                           % (t1bestall[(m_, T1_METHOD_LABEL)][1],
                              max(v for (mm, ll), (st_, v) in t1bestall.items()
                                  if mm == m_ and ll != T1_METHOD_LABEL),
                              "")
                           for i, m_ in enumerate(MAIN_SCALES)])
    # The symmetric reading is a CLAIM ("still leads at all three scales"), so it is checked here
    # rather than asserted: if the method ever stops leading under it, this emitter stops.
    for m_ in MAIN_SCALES:
        _riv = max(v for (mm, ll), (st_, v) in t1bestall.items()
                   if mm == m_ and ll != T1_METHOD_LABEL)
        if t1bestall[(m_, T1_METHOD_LABEL)][1] <= _riv:
            raise SystemExit("tab:main: at %s the method row does not lead under the symmetric "
                             "best-checkpoint reading (%.2f against %.2f) -- the note's second "
                             "sentence would be false"
                             % (m_, t1bestall[(m_, T1_METHOD_LABEL)][1], _riv))
    # The window-mean reading, and where it leaves a tie at the printed decimal.
    _ties, _wmargins = [], []
    for m_ in MAIN_SCALES:
        _mw = t1winsolve[(m_, T1_METHOD_LABEL)]
        _rv, _rl = max((t1winsolve[(m_, l_)], l_) for _, l_, sp_ in T1ROWS
                       if sp_ is not BASE_ROW and l_ != T1_METHOD_LABEL)
        _wmargins.append(_mw - _rv)
        if round(_mw, 1) == round(_rv, 1):
            _ties.append("the %s cell ties %s ($%.2f$ against $%.2f$)"
                         % (_sc(m_), _rl.replace("\\quad ", "").replace(" (control)", ""),
                            _mw, _rv))
    _winclause = (_andjoin(_ties) if _ties else "\\methodname{} leads at every scale")
    # (iv) is a claim about ALL THREE readings and is computed over all three.
    _allmargins = _wmargins + [t1cell[(m_, T1_METHOD_LABEL)][0]
                               - max(t1winsolve[(m_, l_)] for _, l_, sp_ in T1ROWS
                                     if sp_ is not BASE_ROW and l_ != T1_METHOD_LABEL)
                               for m_ in MAIN_SCALES] \
                 + [t1bestall[(m_, T1_METHOD_LABEL)][1]
                    - max(v for (mm, ll), (st_, v) in t1bestall.items()
                          if mm == m_ and ll != T1_METHOD_LABEL)
                    for m_ in MAIN_SCALES]
    if max(_allmargins) >= _PF.FLOOR_PP:
        raise SystemExit("tab:main: a margin of %.2f pp reaches the %.1f pp resolution floor -- "
                         "the note's last sentence would be false" % (max(_allmargins),
                                                                      _PF.FLOOR_PP))
    print("  [tab:main] largest margin over any non-method row, over all three readings: "
          "%+.2f pp against a %.1f pp floor" % (max(_allmargins), _PF.FLOOR_PP))
    _t1note = (
        "The \\methodname{} row is the registered full recipe, read in all three of its columns at"
        " the checkpoint where its own solve rate is highest (" + _stepclause + "); every other"
        " row is the mean over the pre-registered $\\{15,20,25,30\\}$. The ordering does not turn"
        " on that asymmetry: at every row's own best checkpoint \\methodname{} still leads at all"
        " three scales, reading $2$B$/4$B$/8$B, " + _symclause + ". The window mean is the pre-registered readout, under"
        " which " + _winclause + "; all four steps and the window mean, for every row, are in"
        " Table~\\ref{tab:mainsteps}. No margin under any of the three readings approaches the"
        " $%.1f$\\,pp this benchmark resolves, so none is a measured separation. One run per row,"
        " that arm's best of $n$ runs, at an $n$ that differs by row: reading $2$B$/4$B$/8$B, "
        % _PF.FLOOR_PP
        # 2026-09-04 _fold9 (6e, PI 00:30): THE NOTE IS TRIMMED TO THE CONVENTION SENTENCES THE
        # PI REQUIRES AND NOTHING ELSE. Kept verbatim, in order: the best-checkpoint statement
        # with its per-scale steps, the symmetric-reading sentence, the window/tie sentence with
        # the tab:mainsteps pointer, the floor sentence, the seed-count sentence and the
        # warm-bank sentence. MOVED OUT, verbatim, to Appendix~\ref{app:tableconv}: the bold
        # legend (which Section~\ref{sec:triage-results} already states for both main tables) and
        # the trailing provenance pointer (which this float's own caption already carries). No
        # cell, bold, header or caption changes; this is a note-text edit only, and rltable.tex is
        # the one emitted file this pass is allowed to alter.
        + _t1nclause + ". Our warm bank runs inside this paper's allocator: the \\methodname{} and"
        " \\textsc{Tscl} rows carry it and no other does, worth $+1.5$\\,pp to \\textsc{Trace} at"
        " $2$B (Table~\\ref{tab:ladder2b}).")
    for (m_, l_), v in sorted(t1shared.items()):
        _t1note += (" At $%s$B the %s row is read on this shared index; on its own pairs it reads"
                    " $%.1f$ with a $%+.1f$ effect, which is what Table~\\ref{tab:configs} prints"
                    " (Appendix~\\ref{app:tableconv})."
                    % (m_.replace("B", ""),
                       l_.replace("\\quad ", "").replace(" (control)", ""), v[0], v[1]))
    # 2026-09-04 _fold9 (6e): the trailing provenance pointer moved to Appendix~\ref{app:tableconv}
    # with the bold legend. tab:main's caption already ends "Conventions and per-arm effects:
    # Section~\ref{sec:triage-results} and Table~\ref{tab:mainfull}", so the note was repeating a
    # pointer the reader meets two lines above it.
    L1 += ["\\bottomrule"] + notes_block(env="tabular*", items=[_t1note])
    W("rltable.tex", L1)

    # ---- TABLE 2: THIS FRAMEWORK'S CONFIGURATIONS AND THE UNIFORM CONTROL --------------------
    # The rows that left Table 1 in this pass, with the window solve rate Table 1 prints AND the
    # two statistics of record (PI 19:40): the mean held-out effect over the pre-registered
    # window and its Holm-adjusted p, with the star that marks a cell clearing Holm at 0.05
    # inside its own scale's family. Same arms, same pinned families, same best-seed rule, same
    # anchors as Table 1 and as tab:mainfull; the effect and p cells are byte-identical to that
    # grid's, and the rate cells are built by the same expression Table 1's are.
    SAMEFIX3 = "\\multicolumn{3}{c}{$=$ full}"
    NOTRUN3 = "\\multicolumn{3}{c}{---\\textsuperscript{\\dag}}"
    # ---- TABLE 2: THE ABLATION STUDY (PI 21:55) ---------------------------------------------
    # THREE COLUMNS PER SCALE. The solve rate Table 1 prints; the DELTA AGAINST THE FULL METHOD,
    # which is what makes this an ablation rather than a catalogue; and the effect against that
    # scale's own base policy, which is the paper's statistic of record, carrying its Holm star.
    # THE HOLM p ITSELF MOVED TO THE APPENDIX and the note points there: a fourth numeric column
    # per scale does not fit inside \textwidth at \footnotesize once the labels say what each row
    # changes, and the star already carries "clears Holm at 0.05". No number was dropped from the
    # paper: tab:mainfull prints every p for every one of these rows.
    fullwin = {}
    for model in MAIN_SCALES:
        a_ = agg(fams[model][0], CFG_ARMS["fixed"].get(model), best=True)
        rr = [_paired(a_["sel"], st, bases[model])[0] for st in WINDOW]
        rr = [x for x in rr if x is not None]
        fullwin[model] = sum(rr) / len(rr)
    L2 = [GEN,
          "%% anchors/families: " + "; ".join("%s n=%d, %d arms" % (m, fams[m][1], fams[m][2])
                                              for m in MAIN_SCALES),
          "%% matched steps %s" % (WINDOW,),
          "% TABLE 2 = THE ABLATION STUDY. Each row changes ONE component of the full method,",
          "% except the VIP row, which the note discloses as changing more than one. Labels are",
          "% verified against slurm/supervisor.sh's launch flags arm by arm; see CFG_LABEL.",
          "% Delta is against the full method's own window rate at that scale, not against base.",
          "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l rrr rrr rrr@{}}",
          "\\toprule",
          " & \\multicolumn{3}{c}{Qwen3-VL-$2$B} & \\multicolumn{3}{c}{Qwen3-VL-$4$B}"
          " & \\multicolumn{3}{c}{Qwen3-VL-$8$B} \\\\",
          "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
          "ablation & " + " & ".join(["solve", "$\\Delta$ vs", "effect"] * 3) + " \\\\",
          " & " + " & ".join(["(\\%)", "full", "vs base"] * 3) + " \\\\",
          "\\midrule"]
    SPAN3 = "\\multicolumn{3}{c}{%s}"
    # 2026-08-31 (PI 14:45): best-in-column in bold here too, on the two PERFORMANCE columns of
    # each scale -- the solve rate and the effect against base, both higher-is-better. The
    # $\Delta$-vs-full column is NOT bolded: it is a difference against this table's own first
    # row, zero there by construction, and a "winner" among differences is not a quantity this
    # table claims. Every row here is a configuration of ours, so every bold is ours, and that
    # is a fact about which rows the float carries rather than about how it marks them.
    # Ties bold together at the printed precision, as in tab:main.
    ablv = {}
    for kind, lab, spec in ABLROWS:
        for model in MAIN_SCALES:
            arms = spec.get(model)
            if arms is NOT_RUN or arms is SAME_FIXED:
                continue
            a = agg(fams[model][0], arms, best=True)
            rr = [_paired(a["sel"], st, bases[model])[0] for st in WINDOW]
            rr = [x for x in rr if x is not None]
            ablv[(model, lab)] = (sum(rr) / len(rr), 100.0 * a["mean"])
    ablbest = {(model, j): max(round(v[j], 1) for (m_, l_), v in ablv.items() if m_ == model)
               for model in MAIN_SCALES for j in (0, 1)}
    for kind, lab, spec in ABLROWS:
        cells = []
        for model in MAIN_SCALES:
            arms = spec.get(model)
            if arms is NOT_RUN:
                cells.append(SPAN3 % "---\\textsuperscript{\\dag}")
                continue
            if arms is SAME_FIXED:
                cells.append(SPAN3 % "$=$ full")
                continue
            a = agg(fams[model][0], arms, best=True)
            rates, brates = [], []
            for st in WINDOW:
                r, br, _ = _paired(a["sel"], st, bases[model])
                if r is None:
                    continue
                rates.append(r)
                brates.append(br)
            win = sum(rates) / len(rates)
            if abs((win - sum(brates) / len(brates)) - 100.0 * a["mean"]) > 1e-9:
                raise SystemExit("tab:configs: %r at %s: the rate column and the statistic of "
                                 "record disagree" % (lab, model))
            d = "---" if kind == "@FULL" else "$%+.1f$" % (win - fullwin[model])
            star = "^{*}" if a.get("q", 1) < 0.05 else ""
            bw = round(win, 1) == ablbest[(model, 0)]
            be = round(100.0 * a["mean"], 1) == ablbest[(model, 1)]
            cells.append(("$\\mathbf{%.1f}$" if bw else "$%.1f$") % win + " & %s & " % d
                         + (("$\\mathbf{%s}%s$" % (pp(a["mean"]), star)) if be
                            else ("$%s%s$" % (pp(a["mean"]), star))))
        L2.append("%s & %s \\\\" % (lab, " & ".join(cells)))
    # THE SHARED CONVENTIONS ARE STATED ONCE, IN TABLE 1'S NOTE, AND POINTED AT HERE. Both notes
    # used to carry the window definition, the shared index, the best-seed rule, the anchors and
    # the family sizes in full, which is the same six lines of \scriptsize printed twice on one
    # page. Nothing is dropped: the reader meets every convention on this page, once, under the
    # table that introduces it. What stays here is what is TRUE OF THIS TABLE ONLY.
    # Same treatment as tab:main's note, and for the same instruction: the sentences that
    # defined this float's columns and named what each row changes are reproduced verbatim in
    # app:tableconv, and what stays is the legend a reader needs in front of the numbers.
    L2 += ["\\bottomrule"] + notes_block(env="tabular*", items=[
        "Each row changes one component of the full method; $\\Delta$ vs full is that row's cost"
        " in solve rate at the same scale, and \\emph{effect vs base} is that row's own paired"
        " difference from the anchor on the tasks it and the anchor both attempted, not its solve"
        " rate minus Table~\\ref{tab:main}'s base row."
        " $^{*}$~clears Holm at $0.05$ in its own scale's family;"
        " \\textbf{bold} is the best solve rate and the best effect at each scale. The full"
        " method here is Table~\\ref{tab:main}'s \\methodname{} row at every scale. The $8$B"
        " shrinkage cell's arm differs from its siblings in infrastructure as well as in that"
        " component, at tensor parallelism $2$ and under a $20{,}480$-token packing limit on ten"
        " of its thirty optimizer steps (Appendix~\\ref{app:trainconf})."
        " What each row changes, and the rows measured at fewer than three scales:"
        " Section~\\ref{sec:triage-results} and Appendix~\\ref{app:tableconv}."])
    W("configs.tex", L2)

    # ---- THE PER-CHECKPOINT COLUMNS TAB:MAIN NO LONGER HAS ROOM FOR, APPENDIX ----------------
    # NEW 2026-08-28 with the wide tab:main. These are not new measurements and not a new
    # definition: they are the SAME main_row_metrics() values, at the SAME fixed step 15, that
    # tab:main printed until this pass, moved into their own float because three scale groups and
    # five columns per group do not fit inside \textwidth. Delta vs base at step 15 is NOT
    # repeated here -- it is mainsteps.tex's "15" column, digit for digit, and duplicating it
    # would create two homes for one number.
    # 2026-08-29: this table is now MEAN TURNS ALONE. The solve rate and its CI moved INTO
    # tab:main with the enrichment pass, and printing them here as well would give one number two
    # homes -- the failure this file refuses everywhere else. Mean turns is the column that did
    # not fit in tab:main's four-per-scale grid, so this float is exactly what tab:main dropped
    # and nothing else. Same step, same rows, same selected seed, same definition.
    LA = [GEN,
          "%% anchors/families: " + "; ".join("%s n=%d, %d arms" % (m, fams[m][1], fams[m][2])
                                              for m in MAIN_SCALES),
          "%% Read at step %d and ONLY step %d -- the one step every arm in tab:main has banked at"
          % (MAIN_STEP, MAIN_STEP),
          "% every scale, so the rule is the same rule for every row and no arm is read at a step",
          "% chosen for it.",
          "% 2026-08-29: the solve rate and its binomial CI LEFT this table, upward -- they are",
          "% columns of tab:main again. What is left is the one per-checkpoint column tab:main has",
          "% no width for. Nothing measured left the paper and nothing is printed twice.",
          "% SEED CONVENTION 2026-08-28: each multi-seed row is read on the SAME single run",
          "% tab:main's cell is read on -- that arm's best seed by window mean -- and not on a pool",
          "% of its seeds.",
          "% Rows key to tab:main's by label.",
          "\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
          " & \\multicolumn{3}{c}{mean turns at step $%d$} \\\\" % MAIN_STEP,
          "\\cmidrule(lr){2-4}",
          "allocation rule & Qwen3-VL-$2$B & Qwen3-VL-$4$B & Qwen3-VL-$8$B \\\\",
          "\\midrule"]
    a_hdr = False
    for kind, lab, acells in absrows:
        if kind == "@HDR":
            if a_hdr:
                LA.append("\\addlinespace[3pt]")
            LA.append("\\multicolumn{4}{@{}l}{%s} \\\\" % lab)
            a_hdr = True
            continue
        LA.append("%s & %s \\\\" % (IND(lab), " & ".join(acells)))
    LA += ["\\bottomrule"] + notes_block([
        # 2026-08-29: the \\S clause went with the last IN_FLIGHT row (2B faithful TRACE, whose
        # window completed overnight) and is replaced by the marker this table now uses.
        "\\textsuperscript{\\dag}not run at this scale: coverage, not a null.\\quad"
        " \\textsuperscript{\\ddag}the sharpened configuration \\emph{is} the fixed-$\\rho$"
        " configuration at $2$B and $4$B, so its cells there are that row's.",
        "Solve rate and $\\Delta$ vs base at this step, with the $95\\%$ binomial CI, are"
        " columns of Table~\\ref{tab:main}."])
    W("mainabs.tex", LA)

    # ---- NINE MACROS THE FIGURE-1 OUTCOME BAND READS ----------------------------------------
    # 2026-08-28. Figure 1's band (c) encodes tab:main's window means as BAR LENGTHS. Until this
    # pass those lengths were literals in the figure source, and they had already drifted once
    # (the 2B method bar was still the scale-refit configuration's +2.7 after tab:main's 2B cell
    # reverted to the fixed-rho +1.9). Emitting them makes that drift impossible: the figure reads
    # these macros, so a re-emission moves the bars with the table or moves neither.
    # Method = tab:main's TRIAGE row; Control = its uniform GRPO row; Best = the LARGEST window
    # mean among the Published-baseline rows that carry a number at that scale (rows in the
    # not-run and window-incomplete states carry none and are skipped, which is why this is a max
    # over what exists and not over the block). Formatted exactly as tab:main prints them: one
    # decimal, sign shown only when negative.
    OVNAME = {"2B": "ovTwoB", "4B": "ovFourB", "8B": "ovEightB"}
    # The floor gauge Figure 1 draws beside the bars, and the one Figure 3's panels draw, are the
    # same length and are now the same DEFINITION: paper_figures.FLOOR_PP. Importing it here costs
    # nothing (that module imports only os at the top level; matplotlib is imported inside its one
    # function) and removes the last place where 6.0 was a literal in a drawing.
    import paper_figures as PF
    LB = [GEN,
          "% Bar lengths for Figure 1's band (c), in percentage points, from tab:main's own rows.",
          "% Method = the TRIAGE row, at the ONE checkpoint tab:main reads that row at since the",
          "% 2026-09-02 ruling (\\ov*MethodStep below names it); Control = the uniform GRPO row",
          "% and Best = the largest window mean among the Published-baseline rows that carry a",
          "% number at that scale, both of them window means as before.",
          "% One decimal, sign only when negative -- the way tab:main prints them.",
          "% 2026-08-28: these move with tab:main's seed convention, which is now the BEST SEED of",
          "% each multi-seed row. That is the point of emitting them beside the table: a bar and a",
          "% cell can never again be two different statistics of the same row. (Figure 1 no longer",
          "% draws band (c) -- PI decision 2026-08-28 22:40 -- so these macros are currently INERT;",
          "% they are still emitted so that the definition, not a literal, is what survives.)",
          "% ovFloor is the benchmark's resolution floor in the same units, from",
          "% paper_figures.FLOOR_PP -- the same length Figure 3's gauge is drawn at.",
          "\\def\\ovFloor{%.1f}" % PF.FLOOR_PP]
    ovlog = []
    for model in MAIN_SCALES:
        vals, block = {}, None
        for kind, lab, spec in MAIN:
            if kind == "@HDR":
                block = lab
                continue
            arms = spec.get(model)
            if arms is NOT_RUN or arms is IN_FLIGHT or arms is SAME_FIXED:
                continue
            a = agg(fams[model][0], arms, best=True)
            if a is None:
                continue
            v = 100.0 * a["mean"]
            if "configurations" in block:
                # 2026-08-29: tab:main gained a configuration row group. Those rows are OURS and
                # are neither the method row nor a published baseline, so they enter none of the
                # three bar lengths -- folding them into "Best" would have drawn a bar labelled
                # "best published baseline" from one of our own arms.
                continue
            if "Method" in block:
                # 2026-09-02: tab:main's method row is read at ONE checkpoint (T1_METHOD_BEST_STEP),
                # so this bar is that checkpoint's effect and not the window mean. The point of
                # emitting these lengths beside the table is that a bar and a cell can never be
                # two different statistics of the same row, and that rule survives the convention
                # change only if the bar follows the cell.
                if T1_METHOD_BEST_STEP and model in t1step:
                    v = 100.0 * a["per"][t1step[model]]
                vals["Method"] = v
            elif "Control" in block:
                vals["Control"] = v
            else:
                vals["Best"] = max(vals.get("Best", v), v)
        for role in ("Method", "Control", "Best"):
            if role not in vals:
                raise SystemExit("overview_bars: no %s value at %s -- Figure 1's band (c) would "
                                 "be drawn from a value that does not exist" % (role, model))
            LB.append("\\def\\%s%s{%.1f}" % (OVNAME[model], role, vals[role]))
            ovlog.append((model, role, vals[role]))
        if T1_METHOD_BEST_STEP and model in t1step:
            LB.append("\\def\\%sMethodStep{%d}" % (OVNAME[model], t1step[model]))
    W("overview_bars.tex", LB)

    # ---- FRAMEWORK CONFIGURATIONS, moved out of tab:main 2026-08-28 --------------------------
    # Same statistics, same anchors, same pinned families as tab:main -- only the row set differs,
    # so a q here is comparable down a block exactly as it is there.
    tagmap = {}
    LF = [GEN,
          "%% anchors/families: " + "; ".join("%s n=%d, %d arms" % (m, fams[m][1], fams[m][2])
                                              for m in MAIN_SCALES),
          "%% matched steps %s" % (WINDOW,),
          "% Every configuration of the framework, at every scale, as the FULL RECORD behind",
          "% tab:main's own configuration block. 2026-08-29: those rows are in the main table now",
          "% (PI directive), and this float is not a duplicate of them -- see the seed convention",
          "% below and the mean-turns column, neither of which tab:main carries. Both tables read",
          "% ONE dict (paper_numbers.CFG_ARMS) and use ONE label per configuration, so the row",
          "% sets and the names cannot drift apart.",
          "%% Per-checkpoint columns read at step %d, the same fixed step tab:main uses."
          % (MAIN_STEP,),
          "% SEED CONVENTION -- DELIBERATELY NOT tab:main's, and the note under the table says so.",
          "% Cells here are SEED MEANS (the convention this paper used everywhere until",
          "% 2026-08-28). tab:main, fig:stepcurve and tab:transfermain now print each multi-seed",
          "% row's BEST SEED, so a configuration listed in both tables shows two different numbers",
          "% of the same runs -- e.g. the 8B fixed-rho row is +3.9 here (three-seed mean) and +4.6",
          "% in tab:main (its third seed). That is a difference of statistic, not of data, and it",
          "% is stated rather than reconciled: this table is the FULL RECORD the main table's",
          "% per-scale selection is meant to be checked against, and a full record is a mean.",
          "\\begin{tabular}{lrrrrr}", "\\toprule",
          " & solve rate & $\\Delta$ vs base & mean & window & Holm-adj. \\\\",
          "configuration & (\\%) & (pp) & turns & mean (pp) & $p$ \\\\"]
    for si, model in enumerate(MAIN_SCALES):
        LF += ["\\midrule", scale_hdr(model, 6), "\\midrule"]
        for kind, lab, spec in FRAMECFG:
            if kind == "@HDR":
                LF.append("\\multicolumn{6}{l}{%s} \\\\" % lab)
                continue
            arms = spec.get(model)
            if arms is NOT_RUN:
                LF.append("%s & %s \\\\" % (lab, NOTRUN5))
                continue
            if arms is IN_FLIGHT:
                LF.append("%s & %s \\\\" % (lab, INFLIGHT5))
                continue
            if arms is SAME_FIXED:
                LF.append("%s & %s \\\\" % (lab, SAMEFIX5))
                continue
            a = agg(fams[model][0], arms)
            m = main_row_metrics(arms, bases[model])
            if a is None or m is None:
                LF.append("%s & \\multicolumn{5}{c}{\\emph{no scorable cell}} \\\\" % lab)
                continue
            d = ("--" if MAIN_STEP not in a["per"] else "$" + pp(a["per"][MAIN_STEP]) + "$")
            # The CI is a subscript here for the same reason it is one in tab:main: one
            # rendering of one quantity everywhere in the paper.
            # THE INTERNAL ARM TAGS, printed in the appendix and nowhere else (2026-08-29,
            # PI 21:55). The main-text labels say what a row CHANGES; the archive has to say
            # which run produced it, or a reader cannot tie a published cell to the fleet.
            # Every seed of the configuration is listed, not just the one a main-text float
            # selects, because these cells are seed means over all of them.
            tagmap.setdefault(model, []).append((lab, sorted(arms)))
            LF.append("%s & $%.1f_{\\pm%.1f}$ & %s & $%.1f$ & $%s$ & %s \\\\"
                      % (lab, m["rate"], m["ci"], d, m["turns"], pp(a["mean"]), qcell(a)))
    LF += ["\\bottomrule"] + notes_block([
        # THE INTERNAL ARM TAGS, in the note rather than in a column (2026-08-29, PI 21:55).
        # A dedicated column was built first and measured: the multi-seed tag lists run this
        # tabular 85pt past the text width, and it had about 4pt of slack, so the column cannot
        # exist here at any sensible width. The note carries the same mapping at no width cost,
        # which keeps the archive traceable, which is what the column was for.
        ("The main-text labels in Tables~\\ref{tab:main} and~\\ref{tab:configs} name what each"
         " configuration \\emph{changes}; the internal arm tags behind the rows, so a published"
         " cell can be traced to the fleet, are "
         + "; ".join("%s~%s" % (lab.replace("\\quad ", ""),
                                "{\\ttfamily\\scriptsize " + ", ".join(t) + "}")
                     for lab, t in tagmap.get("8B", []))
         + " at $8$B, and the same configurations at $2$B and $4$B carry that scale's own tags"
           " (\\texttt{q2b\\ldots} and \\texttt{q4b\\ldots} respectively).")
        + (" Cells are \\emph{seed means} over every seed of the configuration, and the mean-turns"
        " column is this table's own. Table~\\ref{tab:main} prints the same configurations on each row's"
        " \\emph{best seed} (as do Figure~\\ref{fig:stepcurve} and"
        " Table~\\ref{tab:transfermain}), so a configuration appears in both floats with two"
        " different statistics of the same runs; the per-seed grid was withdrawn from this"
        " version (Appendix~\\ref{app:movedseeds})."),
        "The subscript on a rate is the $95\\%%$ binomial CI half-width at step %d, as in"
        " Table~\\ref{tab:main}." % MAIN_STEP])
    W("framecfg.tex", LF)
    # Each anchor's own values, which the caption carries as the reference for the two absolute
    # columns. Computed on the FULL anchor cell, and the caption says that is what it is.
    anchors = {}
    for m in MAIN_SCALES:
        bt = cell_turns(BASE_CELLS[m])
        anchors[m] = (len(bases[m]), 100.0 * sum(bases[m].values()) / len(bases[m]),
                      (sum(bt.values()) / len(bt)) if bt else float("nan"))
    base_rate, base_turns = anchors["8B"][1], anchors["8B"][2]

    # ---- The four per-step cells of every main-table row, APPENDIX ---------------------------
    # Emitted 2026-08-18 when tab:main collapsed to one checkpoint. This is the same computation
    # that produced tab:main's four step columns, over the same rows, so every digit here is a
    # digit that used to be in the main table -- the point of the float is that nothing measured
    # was dropped when the main table was simplified. It also keeps BOTH terms of the
    # compute-parity comparison (the method at step 15 against its control at step 30) visible in
    # one float, which is where that reading now lives. Scale-blocked 2026-08-22 with tab:main.
    L = [GEN,
         "%% anchors/families: " + "; ".join("%s n=%d, %d arms" % (m, fams[m][1], fams[m][2])
                                             for m in MAIN_SCALES),
         "% The per-step window behind tab:main, which reports the window mean and its q only.",
         "% Same rows, same blocks, same seed convention, same digits: since 2026-08-28 a",
         "% multi-seed row here is the SAME single run tab:main and fig:stepcurve print -- that",
         "% arm's best seed by window mean -- so the four cells of a row are one trajectory rather",
         "% than a seed average, and no +- is printed on them. Every seed of every one of these",
         "% rows, and each block's seed mean, are in perseed.tex. The baseline CITATIONS are",
         "% stripped here and only here: they are carried once, in tab:main, and repeating them",
         "% costs this table the 66pt of width that made it overfull. Rows key to tab:main's by",
         "% label.",
         "\\begin{tabular}{lrrrrrr}", "\\toprule",
         " & \\multicolumn{4}{c}{effect (pp), by step} & window & Holm-adj. \\\\",
         "\\cmidrule(lr){2-5}",
         "method & $15$ & $20$ & $25$ & $30$ & mean & $p$ \\\\"]
    for model in MAIN_SCALES:
        L += ["\\midrule", scale_hdr(model, 7), "\\midrule"]
        for kind, lab, spec in MAIN:
            lab = re.sub(r"\s*\\citep\{[^}]*\}", "", lab)
            if kind == "@HDR":
                L.append("\\multicolumn{7}{l}{%s} \\\\" % lab)
                continue
            arms = spec.get(model)
            if arms is NOT_RUN:
                L.append("%s & %s \\\\" % (lab, NOTRUN6))
                continue
            if arms is SAME_FIXED:
                L.append("%s & \\multicolumn{6}{c}{$=$ full at this scale} \\\\" % lab)
                continue
            a = agg(fams[model][0], arms, best=True)
            if a is None:
                L.append("%s & \\multicolumn{4}{c}{\\emph{no scorable cell}} & -- & -- \\\\" % lab)
                continue
            per = ["--" if st not in a["per"] else "$" + pp(a["per"][st]) + "$" for st in WINDOW]
            L.append("%s & %s & %s & %s \\\\"
                     % (lab, " & ".join(per), eff(a, star=False), qcell(a)))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("mainsteps.tex", L)

    # ---- Our own measured negatives, appendix ------------------------------------------------
    # 2026-09-03 _fold7h: two rows and one block header, and each new row is read at ITS OWN
    # scale -- its own base anchor, its own pinned family, its own Holm correction -- which is
    # why the loop takes the model off the registry instead of closing over the 8B family.
    L = [GEN,
         "% the measured-negative arms, moved out of the main table 2026-08-17.",
         "% 2026-09-03: the last two rows are 2B and 4B arms and carry their own scales'",
         "% anchors and Holm families; the note says so and the caption repeats it.",
         "% 2026-09-06: the temperature block is now a FOUR-POINT curve at 4B with one rung",
         "% at 2B and one at 8B, and a second block carries the two variable-k pilots. The",
         "% 8B temperature row and the 4B variable-k row are ABOVE the configurations they",
         "% came from; they sit here because they are the same one-flag construction as the",
         "% rows beside them, not because they failed. The caption says which is which.",
         "\\begin{tabular}{lrrrrrr}", "\\toprule",
         " & \\multicolumn{4}{c}{effect (pp), by step} & window & Holm-adj. \\\\",
         "\\cmidrule(lr){2-5}",
         "configuration & $15$ & $20$ & $25$ & $30$ & mean & $p$ \\\\", "\\midrule"]
    for kind, lab, arms, model in NEGATIVES:
        if kind == "@HDR":
            L += ["\\midrule", "\\multicolumn{7}{l}{%s} \\\\" % lab]
            continue
        f_ = fams[model][0]
        a = agg(f_, arms)
        if a is None:
            continue
        L.append("%s & %s & %s & %s \\\\"
                 % (lab, " & ".join(_cells(f_[a["arms"][0]], FULL_N)),
                    eff(a, star=False), qcell(a)))

    # THE WHOLE DIAL, EMITTED RATHER THAN TYPED. The lowest point of each scale is a printed cell
    # of this paper already (the per-scale rho row of tab:configs, tab:mainfull and
    # tab:transfermain); what did not exist anywhere is the curve, and a curve typed into prose is
    # the defect the 2026-09-02 sweep spent a whole pass repairing. Every value below is computed
    # here from the same families and the same benchmark records the floats are built from.
    _dial = {}
    for _sc, _pts in sorted(TEMP_DIAL.items()):
        _f = fams[_sc][0]
        _row = []
        for _tau, _t in _pts:
            if _t not in _f:
                raise SystemExit("tab:negatives: the temperature dial at %s names %s, which is "
                                 "not in the pinned family" % (_sc, _t))
            _b, _n = BR.load_run(_t)[0], NR.load_run(_t)[0]
            if _b is None or _n is None:
                raise SystemExit("tab:negatives: %s has no transfer records -- the temperature "
                                 "dial cannot be written from the records" % _t)
            _row.append(dict(tau=_tau, arm=_t, win=_f[_t]["mean"],
                             bfcl=BR.rate(_b), nest=NR.rate(_n)))
        _b = BR.load_run(BFCL_BASE[_sc])[0]
        _n = NR.load_run(NEST_BASE[_sc])[0]
        if _b is None or _n is None:
            raise SystemExit("tab:negatives: the %s anchor has no transfer records" % _sc)
        _dial[_sc] = dict(pts=_row, bfcl_base=BR.rate(_b), nest_base=NR.rate(_n))

    # SIX BUILD-REFUSALS, one per claim the prose and the caption make about this dial. If any of
    # them stops being true of the records, the paper does not build.
    #   (1) the window FALLS with tau at 2B and 4B;
    #   (2) and RISES at 8B -- the inversion, which is the whole 8B reading;
    #   (3) NESTFUL rises with tau at every scale;
    #   (4) the 4B window is monotone DOWN across all four measured points;
    #   (5) BFCL never reaches the untrained 4B policy anywhere on the 4B curve, which is what
    #       falsifies the interpolation hypothesis;
    #   (6) at 8B the natural temperature is above its parent on ALL THREE axes.
    for _sc, _d in sorted(_dial.items()):
        _lo, _hi = _d["pts"][0], _d["pts"][-1]
        _falls = _sc in ("2B", "4B")
        if _falls and not _hi["win"] < _lo["win"]:
            raise SystemExit("tab:negatives: the dial claim says the window FALLS at %s and the "
                             "cells say otherwise: %.4f -> %.4f"
                             % (_sc, 100 * _lo["win"], 100 * _hi["win"]))
        if not _falls and not _hi["win"] > _lo["win"]:
            raise SystemExit("tab:negatives: the dial claim says the window RISES at %s (the "
                             "inversion) and the cells say otherwise: %.4f -> %.4f"
                             % (_sc, 100 * _lo["win"], 100 * _hi["win"]))
        if not _hi["nest"] > _lo["nest"]:
            raise SystemExit("tab:negatives: the dial claim says NESTFUL RISES at %s and the "
                             "records say otherwise: %.2f -> %.2f"
                             % (_sc, _lo["nest"], _hi["nest"]))
    _w4 = [p["win"] for p in _dial["4B"]["pts"]]
    if not all(_w4[i] > _w4[i + 1] for i in range(len(_w4) - 1)):
        raise SystemExit("tab:negatives: the 4B window is claimed monotone decreasing in tau and "
                         "is not: %s" % [round(100 * w, 2) for w in _w4])
    _b4max = max(p["bfcl"] for p in _dial["4B"]["pts"])
    if _b4max >= _dial["4B"]["bfcl_base"]:
        raise SystemExit("tab:negatives: the dial's falsification says BFCL never reaches the "
                         "untrained 4B policy on this curve, and one point now does: %.2f vs %.2f"
                         % (_b4max, _dial["4B"]["bfcl_base"]))
    _l8, _h8 = _dial["8B"]["pts"][0], _dial["8B"]["pts"][-1]
    if not (_h8["win"] > _l8["win"] and _h8["bfcl"] > _l8["bfcl"] and _h8["nest"] > _l8["nest"]):
        raise SystemExit("tab:negatives: the 8B reading says the natural temperature is above its "
                         "parent on all three axes and the records say otherwise: window %.4f v "
                         "%.4f, BFCL %.2f v %.2f, NESTFUL %.2f v %.2f"
                         % (100 * _h8["win"], 100 * _l8["win"], _h8["bfcl"], _l8["bfcl"],
                            _h8["nest"], _l8["nest"]))
    # THE SUPERLATIVE IS CHECKED, NOT TYPED. The note calls the 8B row's BFCL cell the highest this
    # project has recorded on that benchmark; that is a claim over EVERY scored checkpoint in the
    # archive, not over the rows of this table, so it is verified against the archive. Runs scored
    # against a different task index are skipped rather than compared, which is the same rule
    # bfcl_records enforces when it refuses to average them; a partial cell is skipped too.
    # NOTHING OF THE KIND IS CLAIMED FOR NESTFUL AND THE REASON IS THAT IT WOULD BE FALSE: that
    # arm's 41.75 is sixth on the same archive, behind cells of the uniform control, the VIP
    # baseline's second seed, the plug-in configuration, its own parent read at step 15, and the
    # replay baseline. The note says BFCL and stops there.
    _best_bfcl = []
    for _d_ in sorted(glob.glob(os.path.join(BR.BFCL_ROOT, "run_*"))):
        _a_ = os.path.basename(_d_)[4:]
        try:
            _s_ = BR.load_run(_a_, require_index=True)[0]
        except SystemExit:
            continue
        if _s_ and len(_s_) >= 700:
            _best_bfcl.append((BR.rate(_s_), _a_))
    if _best_bfcl and max(_best_bfcl)[0] > _h8["bfcl"] + 1e-9:
        raise SystemExit("tab:negatives: the note calls %.2f the highest BFCL cell in the archive "
                         "and %s now scores %.2f" % (_h8["bfcl"], max(_best_bfcl)[1],
                                                     max(_best_bfcl)[0]))

    # THE PRE-REGISTERED 4B SHIP TEST, evaluated here rather than asserted in prose. A point ships
    # only if all three clauses hold; the emitter records which clause each point fails on, and
    # the note prints it.
    _ref = fams["4B"][0][SHIP_TEST_4B["win_ref"]]["mean"]
    _fails = []
    for _p in _dial["4B"]["pts"]:
        _bad = []
        if not _p["win"] > _ref:
            _bad.append("window")
        if not _p["bfcl"] >= _dial["4B"]["bfcl_base"]:
            _bad.append("BFCL")
        if not _p["nest"] >= _dial["4B"]["nest_base"]:
            _bad.append("NESTFUL")
        _fails.append((_p["tau"], _bad))
    if any(not b for _, b in _fails):
        raise SystemExit("tab:negatives: a 4B dial point now PASSES the pre-registered ship test "
                         "and the paper says none does: %s" % _fails)
    if not all("BFCL" in b for _, b in _fails):
        raise SystemExit("tab:negatives: the paper says every 4B dial point fails on BFCL and one "
                         "does not: %s" % _fails)

    # THE VARIABLE-k PILOTS, read off the run logs. The budget identity (sum k per cycle equal to
    # the uniform k=5 spend) is checked rather than asserted: a pilot that spent more rollouts
    # than its parent would not be a one-flag contrast at all.
    _vk = []
    for _sc, _par, _arm in VARK_PILOTS:
        _f = fams[_sc][0]
        if _arm not in _f or _par not in _f:
            raise SystemExit("tab:negatives: the variable-k pilot at %s needs both %s and %s in "
                             "the pinned family" % (_sc, _par, _arm))
        _cy = vark_cycles(_arm)
        if not _cy:
            raise SystemExit("tab:negatives: %s has no allocator log line, so its k histogram "
                             "cannot be written from the record" % _arm)
        if len({s for _, _, s in _cy}) != 1:
            raise SystemExit("tab:negatives: %s did not hold one budget across cycles: %s"
                             % (_arm, [s for _, _, s in _cy]))
        _off = max(sum(v for k, v in h.items() if k != 5) / n for n, h, _ in _cy)
        _r = {}
        for _t in (_par, _arm):
            _b, _n = BR.load_run(_t)[0], NR.load_run(_t)[0]
            if _b is None or _n is None:
                raise SystemExit("tab:negatives: %s has no transfer records" % _t)
            _r[_t] = (BR.rate(_b), NR.rate(_n))
        # THE TERMINATION SHARE, because "the model loops" is an observation about transcripts and
        # this is the benchmark's own record of it, on the same instances the pass rate beside it
        # is read from. It is bfcl_records.forced_share, the NARROWEST of the three termination
        # statistics this project quotes; see that function for why the other two are larger and
        # are not interchangeable with it.
        _ft = {t: BR.forced_share(t)[0] for t in (_par, _arm, BFCL_BASE[_sc])}
        _vk.append(dict(scale=_sc, cycles=_cy, off=_off, sumk=_cy[0][2],
                        win=_f[_arm]["mean"], win_par=_f[_par]["mean"],
                        bfcl=_r[_arm][0], bfcl_par=_r[_par][0],
                        bfcl_base=BR.rate(BR.load_run(BFCL_BASE[_sc])[0]),
                        nest=_r[_arm][1], nest_par=_r[_par][1],
                        nest_base=NR.rate(NR.load_run(NEST_BASE[_sc])[0]),
                        ft=_ft[_arm], ft_par=_ft[_par], ft_base=_ft[BFCL_BASE[_sc]]))
    if max(d["off"] for d in _vk) >= 0.10:
        raise SystemExit("tab:negatives: the paper says the freed allocator moves fewer than 10%% "
                         "of rows off k=5 and one cycle now moves more: %s"
                         % [round(100 * d["off"], 2) for d in _vk])

    def _hist(h):
        return "$\\{%s\\}$" % ",\\,".join("%d{:}%d" % (k, h[k]) for k in sorted(h))

    def _curve(sc):
        return "; ".join("$\\tau{=}%.2f$ ($%s$, $%.2f$, $%.2f$)"
                         % (p["tau"], pp(p["win"], 2), p["bfcl"], p["nest"])
                         for p in _dial[sc]["pts"])

    _d4, _d2, _d8 = _dial["4B"], _dial["2B"], _dial["8B"]
    L += ["\\bottomrule"] + notes_block(items=[
        "\\emph{The temperature block.} Every row is the per-scale $\\rho$ configuration of"
        " Table~\\ref{tab:configs} with the sampling temperature moved and no other token changed,"
        " one seed at the same offset its parent ran, so each is one flag from a printed"
        " configuration; each is read at its own scale throughout, against that scale's base anchor"
        " and inside that scale's pinned Holm family. The registered value it moves from is"
        " $\\tau{=}0.3$ at $2$B and $4$B and $\\tau{=}0.5$ at $8$B, and those starting points are"
        " the printed per-scale $\\rho$ rows, not rows of this table. Reading (window, \\textsc{Bfcl},"
        " \\textsc{Nestful}) against $4$B bases of $%.2f$ and $%.2f$, the $4$B curve runs %s."
        " The window falls monotonically; \\textsc{Nestful} recovers by $\\tau{=}0.60$ and stays up;"
        " \\textsc{Bfcl} is \\emph{not} monotone, dips to $%.2f$ at the midpoint, and reaches the"
        " untrained policy at no point on the curve, its best being $%.2f$ at $\\tau{=}%.2f$."
        % (_d4["bfcl_base"], _d4["nest_base"], _curve("4B"),
           min(p["bfcl"] for p in _d4["pts"]),
           max(p["bfcl"] for p in _d4["pts"]),
           max(_d4["pts"], key=lambda p: p["bfcl"])["tau"]),
        "The pre-registered test for shipping a point of this dial was set before any of the four"
        " ran: beat the shipped $4$B recipe's window of $%s$ \\emph{and} sit at or above the"
        " untrained $4$B policy on both transfer benchmarks. All four points fail it, and all four"
        " fail on \\textsc{Bfcl}; two of them, $\\tau{=}0.60$ and $\\tau{=}0.85$, pass the other two"
        " clauses. The closest miss is $\\tau{=}0.85$ at $%.2f$ against $%.2f$, which is $%.1f$\\,pp"
        " on $n=800$ and inside what one run of this benchmark separates; it is recorded as a"
        " distance, not as a shipping claim."
        % (pp(_ref), _d4["pts"][2]["bfcl"], _d4["bfcl_base"],
           _d4["bfcl_base"] - _d4["pts"][2]["bfcl"]),
        "\\emph{The dial inverts with scale, and the $8$B row is the reason it is printed here.}"
        " At $2$B the same move costs $%s$\\,pp of window ($%s$ to $%s$) and at $4$B $%s$"
        " ($%s$ to $%s$), while at $8$B it \\emph{gains} $%s$ ($%s$ to $%s$) and takes"
        " \\textsc{Bfcl} from $%.2f$ to $%.2f$ and \\textsc{Nestful} from $%.2f$ to $%.2f$, against"
        " $8$B bases of $%.2f$ and $%.2f$: above its parent on all three axes, and the highest"
        " \\textsc{Bfcl} cell this project has recorded. It is not a recipe change, because the"
        " same one flag fails at the two smaller scales; what it is evidence for is that the"
        " registered recipe is right to set this constant per scale."
        % (pp(_d2["pts"][-1]["win"] - _d2["pts"][0]["win"], 2),
           pp(_d2["pts"][0]["win"], 2), pp(_d2["pts"][-1]["win"], 2),
           pp(_d4["pts"][-1]["win"] - _d4["pts"][0]["win"], 2),
           pp(_d4["pts"][0]["win"], 2), pp(_d4["pts"][-1]["win"], 2),
           pp(_d8["pts"][-1]["win"] - _d8["pts"][0]["win"], 2),
           pp(_d8["pts"][0]["win"], 2), pp(_d8["pts"][-1]["win"], 2),
           _d8["pts"][0]["bfcl"], _d8["pts"][-1]["bfcl"],
           _d8["pts"][0]["nest"], _d8["pts"][-1]["nest"],
           _d8["bfcl_base"], _d8["nest_base"]),
        "The $4$B \\textsc{Nestful} movement over the dial, $%+.2f$\\,pp from $\\tau{=}0.30$ to"
        " $\\tau{=}1.12$, takes that configuration from $%.2f$\\,pp \\emph{below} the untrained $4$B"
        " policy's $%.2f$ to $%.2f$ above it, and is far outside the $7.85$\\,pp seed spread this"
        " paper measures on the transfer axis (Appendix~\\ref{app:genfull}); the $2$B movements,"
        " $%+.2f$ on \\textsc{Bfcl} and $%+.2f$ on \\textsc{Nestful}, are inside it."
        % (_d4["pts"][-1]["nest"] - _d4["pts"][0]["nest"],
           _d4["nest_base"] - _d4["pts"][0]["nest"], _d4["nest_base"],
           _d4["pts"][-1]["nest"] - _d4["nest_base"],
           _d2["pts"][-1]["bfcl"] - _d2["pts"][0]["bfcl"],
           _d2["pts"][-1]["nest"] - _d2["pts"][0]["nest"]),
        "\\emph{The variable-$k$ block.} Each row is its scale's method arm with the group size"
        " freed across tasks and nothing else changed, one seed at the parent's own offset, and the"
        " per-step rollout total pinned to the uniform $k{=}5$ spend ($\\sum_i k_i = %d$ at every"
        " cycle of both runs), so the pilot is budget-matched rather than budget-freed."
        " Proposition~\\ref{prop:rankcal} leaves the calibrated level inert at fixed $k$; freed, it"
        " is what decides. What it does is little: the per-cycle histograms over $256$ rows run %s"
        " at $2$B and %s at $4$B, so at most $%.1f\\%%$ of rows leave $k{=}5$ at either scale."
        " The two runs then part. At $2$B the window falls from $%s$ to $%s$ and both transfer"
        " benchmarks collapse, \\textsc{Bfcl} to $%.2f$ against a base of $%.2f$ and \\textsc{Nestful}"
        " to $%.2f$ against $%.2f$; that collapse is a termination pathology on the benchmark's own"
        " error field, $%.1f\\%%$ of instances \\texttt{force\\_terminated} against $%.1f\\%%$ for the"
        " fixed-$k$ parent and $%.1f\\%%$ for the untrained policy."
        " At $4$B the window \\emph{rises}, from $%s$ to $%s$, with"
        " \\textsc{Bfcl} at $%.2f$ above the base $%.2f$ and \\textsc{Nestful} at $%.2f$, $%.2f$ under"
        " it and inside the $7.85$\\,pp transfer seed spread. One run each way, every in-window"
        " difference inside the $6.0$\\,pp this benchmark resolves: measured, and not settled."
        % (_vk[0]["sumk"],
           ", ".join(_hist(h) for _, h, _ in _vk[0]["cycles"]),
           ", ".join(_hist(h) for _, h, _ in _vk[1]["cycles"]),
           100 * max(d["off"] for d in _vk),
           pp(_vk[0]["win_par"]), pp(_vk[0]["win"]),
           _vk[0]["bfcl"], _vk[0]["bfcl_base"], _vk[0]["nest"], _vk[0]["nest_base"],
           _vk[0]["ft"], _vk[0]["ft_par"], _vk[0]["ft_base"],
           pp(_vk[1]["win_par"]), pp(_vk[1]["win"]),
           _vk[1]["bfcl"], _vk[1]["bfcl_base"], _vk[1]["nest"],
           _vk[1]["nest_base"] - _vk[1]["nest"])])
    W("negatives.tex", L)

    # ---- Per-seed reproduction, appendix -----------------------------------------------------
    L = [GEN,
         "% THE per-seed table. Every multi-seed row of every main-text float, broken out.",
         "% 2026-08-28: this table now carries the two things the best-seed convention takes out",
         "% of the main-text floats and owes the reader somewhere.",
         "%  * The SELECTED seed of each block -- the one tab:main, fig:stepcurve and (for BFCL)",
         "%    tab:transfermain print -- is marked with a triangle on its SEED LABEL. The mark is",
         "%    on the label and never on a number: the no-bolded-winner rule is about cells, and",
         "%    this marks a reporting choice, not a verdict about a run.",
         "%  * The block's SEED MEAN is printed as its own row, because no float carries it any",
         "%    more (tab:main and tab:mainsteps did until this pass). It has no q: a Holm q is a",
         "%    property of an arm's own test and averaging q values is meaningless, which is the",
         "%    same reason qcell() has always reported one seed's rather than a mean.",
         "\\begin{tabular}{llrrrrrr}", "\\toprule",
         " & & \\multicolumn{4}{c}{effect (pp), by step}"
         " & window & Holm-adj. \\\\",
         "\\cmidrule(lr){3-6}",
         "configuration & seed & $15$ & $20$ & $25$ & $30$ & mean & $p$ \\\\"]
    offsets = arm_offsets()
    pslog = []
    for model, groups in PERSEED:
        f, nb, nf = fams[model]
        L += ["\\midrule",
              "\\multicolumn{8}{l}{\\textbf{Qwen3-VL-%s}, anchor $n=%d$, family of %d} \\\\"
              % (model, nb, nf)]
        for lab, arms in groups:
            got = [x for x in arms if x in f]
            if len(got) < 2:
                continue
            sel = best_seed(f, got)
            L.append("\\multicolumn{8}{l}{\\quad %s} \\\\" % lab)
            for x in got:
                r = f[x]
                q = "<0.001" if r["q"] < 0.0005 else "%.3f" % r["q"]
                mark = "$\\blacktriangleright$\\," if x == sel else ""
                L.append(" & %s%s & %s & $%s%s$ & $%s$ \\\\"
                         % (mark, seed_label(x, offsets), " & ".join(_cells(r, FULL_N)),
                            pp(r["mean"]), STAR if r["q"] < 0.05 else "", q))
            # The seed mean, on the same four steps, over exactly the seeds listed above it. A
            # step at which not every listed seed has a cell carries the contributing count as a
            # subscript, because "mean of 3" over a column where one seed is missing is a mean of
            # two and must not print as though it were not.
            a = agg(f, got)
            per = []
            for st in WINDOW:
                if st not in a["per"]:
                    per.append("--")
                    continue
                k = sum(1 for x in got if st in f[x]["per"])
                per.append("$" + pp(a["per"][st])
                           + ("" if k == len(got) else "_{(%d)}" % k) + "$")
            L.append(" & \\emph{mean of %d} & %s & $%s$ & --- \\\\"
                     % (len(got), " & ".join(per), pp(a["mean"])))
            pslog.append((model, lab, sel, seed_label(sel, offsets), 100 * f[sel]["mean"],
                          f[sel]["q"], 100 * a["mean"], list(got)))
    L += ["\\bottomrule",
          "\\addlinespace[2pt]",
          "\\multicolumn{8}{l}{\\scriptsize $\\blacktriangleright$~marks the seed each main-text"
          " float reports for that configuration, the one with the highest window mean. It"
          " marks a} \\\\",
          "\\multicolumn{8}{l}{\\scriptsize \\emph{selection}, not a winner: the selected seed"
          " and its neighbours are separated by far less than this benchmark can resolve. A"
          " subscript on a} \\\\",
          "\\multicolumn{8}{l}{\\scriptsize mean cell is how many of the block's seeds have a cell"
          " at that step, where that is fewer than all of them. The three same-offset} \\\\",
          "\\multicolumn{8}{l}{\\scriptsize repetitions are in Table~\\ref{tab:reruns} and not"
          " here: they are one seed run twice, so their spread is not a seed spread.} \\\\",
          "\\end{tabular}"]
    W("perseed.tex", L)

    # ---- THE THREE SAME-OFFSET REPETITIONS, AND WHAT THEY MOVED ------------------------------
    # The disclosure float for TRANSFER_RERUN. Three rows of this paper have a transfer cell that
    # could not exist on the run their window cells come from: the run's step-30 adapter was
    # pruned off scratch once its window was scored, and a pruned checkpoint directory holds a
    # config and no tensor. Each configuration was therefore trained again, off the SAME launch
    # command at the SAME --seed-offset (500 for all three), and tab:transfermain reads BOTH
    # transfer cells of that row on the repetition, marked with a double dagger.
    #
    # WHY THIS IS ITS OWN FLOAT AND NOT A BLOCK OF tab:perseed. A repetition at one seed offset
    # is one seed run twice. Its spread measures run-to-run nondeterminism -- rollout sampling,
    # kernel nondeterminism, a different pod -- and NOT seed variance, which is the quantity
    # tab:perseed exists to display and the quantity the paper's 6.0 pp resolution floor is
    # derived from. Printing these two runs as "seed 1 and seed 2" would corrupt both readings.
    L = [GEN,
         "% TRANSFER_RERUN's disclosure float: the original run and the same-offset repetition,",
         "% window cells side by side, plus the transfer pair each carries. The repetition is",
         "% NOT a second seed: same --seed-offset, same command, different execution.",
         "\\begin{tabular}{llrrrr rrr}", "\\toprule",
         " & & \\multicolumn{4}{c}{window effect (pp), by step} & window"
         " & \\textsc{Bfcl} & \\textsc{Nestful} \\\\",
         "\\cmidrule(lr){3-6}",
         "row & run & $15$ & $20$ & $25$ & $30$ & mean & (\\%) & (\\%) \\\\", "\\midrule"]
    rrlog = []
    for par, rr in sorted(TRANSFER_RERUN.items()):
        m_ = model_of(par)
        lab = None
        for l_, sp_ in PUBLISHED_REIMPL + [(CFG_LABEL["vipbank"], CFG_ARMS["vipbank"])]:
            if par in (sp_.get(m_) or []):
                lab = l_.replace("\\quad ", "")
        for j, arm in enumerate((par, rr)):
            r = fams[m_][0].get(arm)
            if r is None:
                continue
            bs_, _, _, _ = BR.load_run(arm)
            ns_, _, _ = NR.load_run(arm)
            cells = [("$" + pp(r["per"][st]) + "$") if st in r["per"] else "--" for st in WINDOW]
            L.append("%s & %s & %s & $%s$ & %s & %s \\\\"
                     % (("%s, %s" % (lab, m_)) if j == 0 else "",
                        "original" if j == 0 else "\\emph{repetition}",
                        " & ".join(cells), pp(r["mean"]),
                        "---\\textsuperscript{\\ddag}" if bs_ is None else "$%.2f$" % BR.rate(bs_),
                        "\\emph{pruned}" if ns_ is None else "$%.2f$" % NR.rate(ns_)))
            rrlog.append((m_, arm, 100 * r["mean"]))
        L.append("\\addlinespace[2pt]")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
    W("reruns.tex", L)

    # ---- ABLATION STUDY: component attribution -----------------------------------------------
    full = agg(fam, FULL_METHOD)
    L = [GEN,
         "%% base anchor n=%d; every row is in the same %d-arm Holm family as the main table"
         % (nbase, nfam),
         "% Component attribution: each row names what changed from the full method and the",
         "% delta it costs. Seeds aggregated; per-seed rows in perseed.tex.",
         "\\begin{tabular}{lrrrr}", "\\toprule",
         "component changed & window & cells & $\\Delta$ vs & Holm-adj. \\\\",
         " & mean (pp) & of $4$ & full (pp) & $p$ \\\\", "\\midrule"]
    for kind, lab, arms in ABLATION:
        if kind == "@HDR":
            L += ["\\midrule", "\\multicolumn{5}{l}{%s} \\\\" % lab]
            continue
        a = agg(fam, arms)
        if a is None:
            L.append("%s & \\multicolumn{4}{c}{\\emph{no scorable cell}} \\\\" % lab)
            continue
        d = "---" if kind == "@FULL" else "$%s$" % pp(a["mean"] - full["mean"])
        # No cell in this paper's main-text tables is bolded to mark a "winner" (2026-08-18 PI
        # directive, presentation-standards pass): with ties, non-monotone per-step orderings and
        # a seed-spread column that routinely exceeds the gaps between rows, a single bolded cell
        # overclaims. The full-method row is still named in its own label; only the numeric mark
        # is dropped. See tab:main's caption for the one place this convention is stated.
        L.append("%s & %s & %d & %s & %s \\\\"
                 % (lab, eff(a, bold=False, star=False),
                    a["cells"], d, qcell(a)))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("coadapt.tex", L)

    # ---- ABLATION STUDY, replicated at 2B (single seed, same offset as the full method) ------
    fam2b, nbase2b, nfam2b = fams["2B"]
    full2b = agg(fam2b, FULL_METHOD_2B)
    L = [GEN,
         "%% base anchor n=%d; every row is in the same %d-arm 2B Holm family as Table~scalemain"
         % (nbase2b, nfam2b),
         "% The 2B replicate of the two structural ablations of coadapt.tex. One seed each, pinned",
         "% to the full method's own seed-offset (500) by the ablation pairing rule.",
         "% The last two rows are the same bank rung at 4B (2026-09-02) and at 8B (2026-09-04):",
         "% same construction, same offset, each read against its own scale's anchor and corrected",
         "% inside its own scale's family. See the note.",
         "\\begin{tabular}{lrrrr}", "\\toprule",
         "component changed & window & cells & $\\Delta$ vs & Holm-adj. \\\\",
         " & mean (pp) & of $4$ & full (pp) & $p$ \\\\", "\\midrule"]
    for kind, lab, arms in ABLATION2B:
        if kind == "@HDR":
            L += ["\\midrule", "\\multicolumn{5}{l}{%s} \\\\" % lab]
            continue
        a = agg(fam2b, arms)
        if a is None:
            L.append("%s & \\multicolumn{4}{c}{\\emph{no scorable cell}} \\\\" % lab)
            continue
        d = "---" if kind == "@FULL" else "$%s$" % pp(a["mean"] - full2b["mean"])
        # Same no-bold convention as coadapt.tex above; see that comment.
        L.append("%s & %s & %d & %s & %s \\\\"
                 % (lab, eff(a, bold=False, star=False), a["cells"], d, qcell(a)))
    # ---- THE SAME RUNG, ONE SCALE UP (2026-09-02 _fold7c) ------------------------------------
    # A confound answered at one scale is not answered. See LADDER_BANK_4B for the construction
    # and for why this row's delta is read against the OFFSET-MATCHED full-method seed; both
    # readings are printed, this one in the cell and the other in the note.
    fam4b, nbase4b, nfam4b = fams["4B"]
    _full4b = agg(fam4b, LADDER_FULL_4B)
    _best4b = agg(fam4b, CFG_ARMS["fixed"]["4B"], best=True)
    _bank4b = agg(fam4b, LADDER_BANK_4B[1])
    _pub4b = agg(fam4b, ["q4bTf"])
    for _nm, _a in (("the offset-matched 4B full method", _full4b),
                    ("the 4B full method's best seed", _best4b),
                    ("TRACE with our warm bank at 4B", _bank4b),
                    ("TRACE as published at 4B", _pub4b)):
        if _a is None:
            raise SystemExit("tab:coadapt2b: %s has no scorable window -- the 4B rung of the bank "
                             "ladder cannot be printed" % _nm)
    L.append("%s & %s & %d & $%s$ & %s \\\\"
             % (LADDER_BANK_4B[0], eff(_bank4b, bold=False, star=False), _bank4b["cells"],
                pp(_bank4b["mean"] - _full4b["mean"]), qcell(_bank4b)))
    # ---- AND THE SAME RUNG AT THE HEADLINE SCALE (2026-09-04 _fold10) -------------------------
    # See LADDER_BANK_8B for the construction and for why the reference is a8T3g (offset 500) and
    # not the seed tab:main prints. This rung is the one that reverses the 2B and 4B readings, so
    # every claim the note makes about it is a build refusal below rather than a sentence.
    fam8b, nbase8b, nfam8b = fams["8B"]
    _full8b = agg(fam8b, LADDER_FULL_8B)
    _best8b = agg(fam8b, CFG_ARMS["fixed"]["8B"], best=True)
    _bank8b = agg(fam8b, LADDER_BANK_8B[1])
    _pub8b = agg(fam8b, ["t8Tf"])
    for _nm, _a in (("the offset-matched 8B full method", _full8b),
                    ("the 8B full method's best seed", _best8b),
                    ("TRACE with our warm bank at 8B", _bank8b),
                    ("TRACE as published at 8B", _pub8b)):
        if _a is None:
            raise SystemExit("tab:coadapt2b: %s has no scorable window -- the 8B rung of the bank "
                             "ladder cannot be printed" % _nm)
    L.append("%s & %s & %d & $%s$ & %s \\\\"
             % (LADDER_BANK_8B[0], eff(_bank8b, bold=False, star=False), _bank8b["cells"],
                pp(_bank8b["mean"] - _full8b["mean"]), qcell(_bank8b)))
    # THE TRANSFER CELLS OF THIS TABLE'S ROWS, IN THE NOTE, because they exist and were printed
    # nowhere (2026-09-02). The window column is the in-distribution reading; the two held-out
    # benchmarks are where this paper says the result lives, and the bank-carrying \textsc{Trace}
    # row leads the method's own 2B cells on BOTH of them. Read from the same records
    # tab:bfcl2b and tab:nestrep are read from (bfcl_records / nestful_records at step 30), never
    # transcribed, so a note and a table cannot disagree.
    # The untrained anchor is the one tag the two record sets spell differently at 8B: BFCL's
    # directory is run_base and NESTFUL's is run_base8b (bfcl_records.SCALES / nestful_records.
    # SCALES). The pair is written out here rather than guessed, so a missing anchor is a refusal
    # and never a silently dropped comparison.
    _t7 = {}
    for _tag, _btag, _ntag in (("base2b", "base2b", "base2b"), ("q2bT", "q2bT", "q2bT"),
                               ("t2bTf", "t2bTf", "t2bTf"), ("t2bTbk", "t2bTbk", "t2bTbk"),
                               ("base4b", "base4b", "base4b"), ("q4bTf", "q4bTf", "q4bTf"),
                               ("t4bTbk", "t4bTbk", "t4bTbk"),
                               ("base8b", "base", "base8b"), ("t8Tf", "t8Tf", "t8Tf"),
                               ("t8Tbk", "t8Tbk", "t8Tbk")):
        _b = BR.load_run(_btag)[0]
        _n = NR.load_run(_ntag)[0]
        if _b is None or _n is None:
            raise SystemExit("tab:coadapt2b: %s has no transfer records -- the note cannot be "
                             "written from the records" % _tag)
        _t7[_tag] = (BR.rate(_b), NR.rate(_n))
    if not (_t7["t2bTbk"][0] > _t7["q2bT"][0] and _t7["t2bTbk"][1] > _t7["q2bT"][1]):
        raise SystemExit("tab:coadapt2b: the note asserts that TRACE with our bank leads the "
                         "method's 2B transfer cells on both benchmarks and the records say "
                         "otherwise: %s" % _t7)
    # THE 4B SENTENCE ASSERTS FOUR THINGS AND EACH ONE IS CHECKED AGAINST THE RECORDS HERE, so a
    # cell that moves takes the sentence out rather than leaving it standing and wrong.
    if not (_t7["t4bTbk"][0] < _t7["base4b"][0] and _t7["t4bTbk"][0] < _t7["q4bTf"][0]
            and _t7["t4bTbk"][1] > _t7["base4b"][1] and _t7["t4bTbk"][1] > _t7["q4bTf"][1]):
        raise SystemExit("tab:coadapt2b: the note asserts that at 4B the bank takes TRACE below "
                         "the untrained base policy on BFCL and above it on NESTFUL, and the "
                         "records say otherwise: %s" % _t7)
    if not (_full4b["mean"] < _bank4b["mean"] < _best4b["mean"]):
        raise SystemExit("tab:coadapt2b: the note asserts that the 4B rung sits between the full "
                         "method's two 4B seeds and the cells say otherwise: %.4f %.4f %.4f"
                         % (_full4b["mean"], _bank4b["mean"], _best4b["mean"]))
    # THE 8B SENTENCE IS THE ONE THAT REVERSES THE OTHER TWO SCALES, so all four of its claims are
    # refusals here (2026-09-04). If any cell moves the sentence comes out; it is never left
    # standing on a reading the records no longer support.
    if not (_bank8b["mean"] < _pub8b["mean"] and _bank8b["mean"] < _full8b["mean"]
            and _bank8b["mean"] < _best8b["mean"]):
        raise SystemExit("tab:coadapt2b: the note asserts that at 8B the bank takes the published "
                         "rule below both its bank-less parent and both of the full method's "
                         "seeds in-window, and the cells say otherwise: bank %.4f pub %.4f "
                         "off500 %.4f best %.4f"
                         % (_bank8b["mean"], _pub8b["mean"], _full8b["mean"], _best8b["mean"]))
    if not (_t7["t8Tbk"][0] > _t7["base8b"][0] and _t7["t8Tbk"][0] > _t7["t8Tf"][0]):
        raise SystemExit("tab:coadapt2b: the note asserts that at 8B the bank moves the published "
                         "rule UP on BFCL and above the untrained policy, and the records say "
                         "otherwise: %s" % _t7)
    if not (_t7["t8Tbk"][1] < _t7["base8b"][1] and _t7["t8Tbk"][1] < _t7["t8Tf"][1]):
        raise SystemExit("tab:coadapt2b: the note asserts that at 8B the bank moves the published "
                         "rule DOWN on NESTFUL and below the untrained policy, and the records "
                         "say otherwise: %s" % _t7)
    L += ["\\bottomrule"] + notes_block(items=[
        "The two transfer benchmarks at step $30$, which this table's window column does not"
        " carry. \\textsc{Trace} with our warm bank scores $%.2f$ on \\textsc{Bfcl} and $%.2f$ on"
        " \\textsc{Nestful}, above \\methodname{}'s own $2$B seed on both ($%.2f$ and $%.2f$) and"
        " above the untrained base policy ($%.2f$ and $%.2f$); \\textsc{Trace} as published is at"
        " $%.2f$ and $%.2f$, so the bank moves that rule up on \\textsc{Bfcl} and down on"
        " \\textsc{Nestful}. Same records as the transfer floats."
        % (_t7["t2bTbk"][0], _t7["t2bTbk"][1], _t7["q2bT"][0], _t7["q2bT"][1],
           _t7["base2b"][0], _t7["base2b"][1], _t7["t2bTf"][0], _t7["t2bTf"][1]),
        "The next-to-last row is that rung at $4$B and it is read at $4$B throughout: against the"
        " $4$B anchor, in the $%d$-arm $4$B Holm family, and with a $\\Delta$ against the $4$B full"
        " method's own offset-$500$ seed ($%s$), the pairing every removal row above uses. Read"
        " instead against the $4$B seed Table~\\ref{tab:main} prints ($%s$) it is $%s$, so this"
        " rung sits between the full method's two $4$B seeds. On transfer it does not repeat the"
        " $2$B result: \\textsc{Bfcl} $%.2f$, below both the untrained $4$B base policy ($%.2f$)"
        " and the same rule run as published ($%.2f$); \\textsc{Nestful} $%.2f$, above both"
        " ($%.2f$ and $%.2f$). \\methodname{}'s own $4$B cells: Table~\\ref{tab:transfermain}."
        % (nfam4b, pp(_full4b["mean"]), pp(_best4b["mean"]),
           pp(_bank4b["mean"] - _best4b["mean"]),
           _t7["t4bTbk"][0], _t7["base4b"][0], _t7["q4bTf"][0],
           _t7["t4bTbk"][1], _t7["base4b"][1], _t7["q4bTf"][1]),
        "The last row is the same rung at $8$B, read at $8$B throughout: against the $8$B anchor,"
        " in the $%d$-arm $8$B Holm family, and with a $\\Delta$ against the $8$B full method's own"
        " offset-$500$ seed ($%s$); against the seed Table~\\ref{tab:main} prints ($%s$) it is"
        " $%s$, so it falls below both. \\textbf{At this scale the bank does not lift the published"
        " rule, it sinks it}: the window goes $%s \\to %s$, and on transfer \\textsc{Bfcl} rises to"
        " $%.2f$ from $%.2f$, above the untrained $8$B base policy ($%.2f$), while"
        " \\textsc{Nestful} falls to $%.2f$ from $%.2f$, \\emph{below} that policy ($%.2f$)."
        " \\methodname{} carries the identical bank at this scale and is above the base policy on"
        " both benchmarks: Table~\\ref{tab:transfermain}."
        % (nfam8b, pp(_full8b["mean"]), pp(_best8b["mean"]),
           pp(_bank8b["mean"] - _best8b["mean"]),
           pp(_pub8b["mean"]), pp(_bank8b["mean"]),
           _t7["t8Tbk"][0], _t7["t8Tf"][0], _t7["base8b"][0],
           _t7["t8Tbk"][1], _t7["t8Tf"][1], _t7["base8b"][1])])
    W("coadapt2b.tex", L)

    # ---- POOL ROBUSTNESS: TRIAGE vs uniform on pool_big320 (DAPO's own training pool) --------
    L = [GEN,
         "%% base anchor n=%d; pool_big320 arms, same %d-arm 2B Holm family as Table~scalemain"
         % (nbase2b, nfam2b),
         "% Pool-matched comparison: q2bTs/q2bFs trained on pool_big320, the pool Dapo trains on,",
         "% converting the disclosed pool-mismatch confound of Table~mech into a designed",
         "% same-pool comparison. One seed each.",
         "\\begin{tabular}{lrrrrr}", "\\toprule",
         " & \\multicolumn{4}{c}{effect (pp), by step} & window \\\\",
         "\\cmidrule(lr){2-5}",
         "allocation (pool\\_big320) & $15$ & $20$ & $25$ & $30$ & mean \\\\", "\\midrule"]
    pool_agg = {}
    for lab, arms in POOL_ROBUST:
        a = agg(fam2b, arms)
        pool_agg[lab] = a
        if a is None:
            L.append("%s & \\multicolumn{4}{c}{\\emph{no scorable cell}} & -- \\\\" % lab)
            continue
        per = ["--" if st not in a["per"] else "$" + pp(a["per"][st]) + "$" for st in WINDOW]
        L.append("%s & %s & %s \\\\" % (lab, " & ".join(per), eff(a, star=False)))
    t, u = pool_agg.get("\\methodname{}"), pool_agg.get("uniform \\textsc{Grpo}")
    if t is not None and u is not None:
        per = ["$" + pp(t["per"][st] - u["per"][st]) + "$" for st in WINDOW]
        L.append("\\emph{difference} (\\methodname{} $-$ uniform) & %s & $%s$ \\\\"
                 % (" & ".join(per), pp(t["mean"] - u["mean"])))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("poolbig.tex", L)

    nrows = NR.scale_block()

    # ---- MECHANISM, seeds aggregated ---------------------------------------------------------
    L = [GEN,
         "% degenerate fraction and live groups are triage_report.py's statistic, recomputed here",
         "% EXCEPT the DAPO row, which is FROZEN in paper_numbers.FROZEN_MECH: run_dapo/ was recycled",
         "% for another arm and its segment boundary is unrecoverable. See that dict for the evidence.",
         "% One row per METHOD; multi-seed rows carry the seed mean and the seed sd.",
         "\\begin{tabular}{lrccr}", "\\toprule",
         "allocation rule & steps & degenerate & live groups & held-out \\\\",
         " & & fraction & per $1{,}000$ & window (pp) \\\\", "\\midrule"]
    for lab, arms in MECH:
        deg, live, nst = [], [], []
        for tag in arms:
            if tag in FROZEN_MECH:
                ns, d, lv = FROZEN_MECH[tag]
            else:
                steps = groups_of(train_rows(tag))
                g = sum(len(s) for s in steps)
                if not g:
                    continue
                dd = sum(1 for s in steps for _, v in s if len(set(v)) == 1)
                ns, d = len(steps), dd / g
                lv = 1000.0 * (g - dd) / (len(steps) * BSZ * NROLL)
            deg.append(d)
            live.append(lv)
            nst.append(ns)
        if not deg:
            continue
        a = agg(fam, arms)
        h = eff(a, star=False) if a else "\\emph{in flight}"
        stc = str(nst[0]) if len(set(nst)) == 1 else "%d--%d" % (min(nst), max(nst))
        if len(deg) > 1:
            dm = sum(deg) / len(deg)
            lm = sum(live) / len(live)
            dsd = (sum((x - dm) ** 2 for x in deg) / (len(deg) - 1)) ** 0.5
            lsd = (sum((x - lm) ** 2 for x in live) / (len(live) - 1)) ** 0.5
            dc, lc = "$%.3f\\pm%.3f$" % (dm, dsd), "$%.1f\\pm%.1f$" % (lm, lsd)
        else:
            dc, lc = "$%.3f$" % deg[0], "$%.1f$" % live[0]
        L.append("%s & %s & %s & %s & %s \\\\" % (lab, stc, dc, lc, h))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("coadapt2x2.tex", L)

    # ---- THREE SCALES IN ONE STRUCTURE -------------------------------------------------------
    # No seed columns. A seed count in a cell is still seed information inside a float, and the
    # 2026-08-17 directive puts all of it in the caption or the appendix reproduction table. The
    # counts are printed to stdout below so the caption's one clause can be checked against them.
    L = [GEN,
         "% One structure, three scales. Each cell is scored against ITS OWN model's anchor and",
         "% Holm-corrected within ITS OWN model's family; the families are never merged.",
         "%% anchors/families: " + "; ".join("%s n=%d, %d arms" % (m, fams[m][1], fams[m][2])
                                             for m in SCALES3),
         "% Seed counts are NOT printed here on purpose; they are a caption clause.",
         "\\begin{tabular}{lrrr}", "\\toprule",
         "configuration & " + " & ".join("Qwen3-VL-" + m for m in SCALES3) + " \\\\",
         "\\midrule"]
    tri_u, seedn = {}, {}
    for kind, lab, spec in SCALEROWS:
        if kind == "@HDR":
            if not L[-1].endswith("\\midrule"):
                L.append("\\midrule")
            L.append("\\multicolumn{4}{l}{%s} \\\\" % lab)
            continue
        if kind == "@DIFF":
            cs = []
            for m in SCALES3:
                t, u = tri_u.get(("T", m)), tri_u.get(("U", m))
                cs.append("--" if (t is None or u is None)
                          else "$%s$" % pp(t["mean"] - u["mean"]))
            L.append("%s & %s \\\\" % (lab, " & ".join(cs)))
            continue
        if kind == "@SD":
            # The larger of the two contrast rows' own seed sds -- the comparator the @DIFF row
            # above is read against. A single-seed arm contributes NO sd (one run cannot estimate
            # a spread); it is skipped rather than counted as zero, because a zero would claim a
            # precision that would make the difference row look resolvable when it is not. Only
            # if NEITHER arm has replicates is the cell "--".
            cs = []
            for m in SCALES3:
                sds = [a["sd"] for a in (tri_u.get(("T", m)), tri_u.get(("U", m)))
                       if a is not None and a["sd"] is not None]
                cs.append("$%.1f$" % (100 * max(sds)) if sds else "--")
            L.append("%s & %s \\\\" % (lab, " & ".join(cs)))
            continue
        cs = []
        for m in SCALES3:
            a = agg(fams[m][0], spec.get(m, []))
            cs.append(eff(a))
            if a is not None:
                seedn[(lab, m)] = a["n"]
            # 2026-08-29: METHOD_TEX, not the name. This test read `"Triage" in lab` and the
            # TRIAGE->BRACE rename silently emptied it: tab:scalemain's difference row
            # printed "--" at all three scales and its seed-sd row lost the method side.
            if METHOD_TEX in lab and "uniform" not in lab:
                tri_u[("T", m)] = a
            if "uniform" in lab:
                tri_u[("U", m)] = a
        L.append("%s & %s \\\\" % (lab, " & ".join(cs)))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("scaletable.tex", L)
    SEEDCOUNTS = seedn

    # ---- POST-WINDOW DURABILITY, THREE SCALES, ONE CONVENTION --------------------------------
    # Steps AFTER the pre-registered matched window, reported separately and never folded into the
    # window statistic: the window was fixed in advance, and reading an arm at whichever later step
    # flatters it is the maximum-over-checkpoints procedure the window exists to prevent.
    # See DURROWS for the convention, the three run protocols and why the 4B method row is seed 3.
    def _durcell(arm, cst, bcells):
        cur = load_cell(cell_path(arm, cst))
        keys = sorted(set(cur) & set(bcells))
        if len(keys) < 300:
            return None, 0
        b = sum(1 for k in keys if bcells[k] and not cur[k])
        c = sum(1 for k in keys if cur[k] and not bcells[k])
        return (c - b) / len(keys), len(keys)

    # The two transfer anchors, read HERE from the same loaders and the same base runs
    # tab:transfermain's top row is built from, so the two floats cannot drift apart. They are
    # read again rather than shared because that table is emitted after this one.
    durbase = {m: load_cell(BASE_CELLS[m]) for m in SCALES3}
    durbb, durnb = {}, {}
    for m in SCALES3:
        bs_, _, _, _ = BR.load_run(DUR_BFCL_BASE[m])
        durbb[m] = None if bs_ is None else BR.rate(bs_)
        nb_, _, _ = NR.load_run("base%s" % m.lower())
        durnb[m] = None if nb_ is None else NR.rate(nb_)
    DHDR = "$\\Delta$ (pp)"

    def _durdelta(x):
        return grey("$%+.2f$" % x)
    L = [GEN,
         "% POST-WINDOW ONLY. Three absolute steps, one convention, one seed per row; never",
         "% folded into the window statistic. The run column is the protocol that produced the",
         "% row: resume = optimizer-continuous from the parent's step-30 state, absolute step",
         "% numbering; merge = the parent's adapter merged into the base because its checkpoint",
         "% is model-only, so the optimizer moments and the LR schedule restart at the seam and",
         "% the child's cell_STEP{10,20,30} ARE absolute 40/50/60; straight = one continuous run",
         "% to 60 with no seam, used for the 2B control because every registered 2B control",
         "% checkpoint was pruned and none survives to continue from.",
         "% The 4B method row is seed 3 (q4bTr's checkpoints are pruned; seeds 2 and 3 tie on the",
         "% window mean to six places and the documented tie-break takes the lower Holm q), so it",
         "% is NOT the run the 4B window points come from.",
         "\\begin{tabular}{ll%s r@{\\hspace{2pt}}r r@{\\hspace{2pt}}r}" % ("r" * len(DSTEP)),
         "\\toprule",
         # The spanning header is kept NARROWER than the three columns it spans. At
         # "held-out effect (pp), $n=590$" the span was wider than its own columns and LaTeX
         # pushed the whole excess into the last one, printing a gap between steps 50 and 60
         # that read as a group boundary. n=590 is in the caption, where it was anyway.
         " & & \\multicolumn{%d}{c}{held-out effect (pp)}"
         " & \\multicolumn{4}{c}{transfer at step $60$ (\\%%)} \\\\" % len(DSTEP),
         "\\cmidrule(lr){3-%d}\\cmidrule(lr){%d-%d}"
         % (2 + len(DSTEP), 3 + len(DSTEP), 6 + len(DSTEP)),
         "allocation rule & run & %s & \\textsc{Bfcl} & %s & \\textsc{Nestful} & %s \\\\"
         % (" & ".join("$%d$" % st for st in DSTEP), grey(DHDR), grey(DHDR)),
         "\\midrule"]
    # 2026-08-31 (PI 14:45): best-in-column in bold, WITHIN EACH SCALE BLOCK and never across
    # them. A 2B cell and an 8B cell are scored against different base policies on different
    # anchors, so a bold that ranged over the whole column would be comparing two different
    # questions. A block with only one row gets no bold at all: there is nothing to be best of,
    # and bolding the only row would read as a result. Higher is better in all five bolded
    # columns; the grey deltas are context and are not bolded.
    durvals = {}
    for m, lab, arm, proto in DURROWS:
        row = []
        for st in DSTEP:
            v, n_ = _durcell(arm, CELLSTEP[proto][st], durbase[m])
            row.append(None if v is None else 100.0 * v)
        bs_, _, _, _ = BR.load_run(arm)
        ns_, _, _ = NR.load_run(arm)
        row.append(None if bs_ is None else BR.rate(bs_))
        row.append(None if ns_ is None else NR.rate(ns_))
        if any(x is not None for x in row):
            durvals[(m, lab)] = row
    durbest = {}
    for m in SCALES3:
        rows_ = [v for (m_, _), v in durvals.items() if m_ == m]
        if len(rows_) < 2:
            continue
        for j in range(len(DSTEP) + 2):
            vals = [round(r[j], 2 if j >= len(DSTEP) else 1)
                    for r in rows_ if r[j] is not None]
            if vals:
                durbest[(m, j)] = max(vals)

    def _durbold(m, j, txt, v, dec):
        return txt if durbest.get((m, j)) is None or round(v, dec) != durbest[(m, j)] \
            else txt.replace("$", "$\\mathbf{", 1)[:-1] + "}$"

    # A ROW IS COMPLETE OR ABSENT, and that is a correctness guard rather than tidiness.
    # 2026-08-31: caught in the act. a8Fc was mid-scoring while this table was emitted; its
    # cell_STEP20 held 583 paired tasks and was still being appended, which is ABOVE the
    # PARTIAL_N=560 bar the dagger fires below, so the row printed +4.5 at one emission and +4.3
    # at the next with no mark to say the number was still moving. The dagger cannot catch this
    # because it tests a count, and a file three quarters written passes any count it will
    # eventually pass. So a continuation is printed only when EVERY one of its post-window cells
    # is scorable; until then the arm is not reported at all, which is what the caption already
    # promises ("rows for arms still training are absent rather than dashed").
    durcomplete = {}
    for m, lab, arm, proto in DURROWS:
        vals = [_durcell(arm, CELLSTEP[proto][st], durbase[m])[0] for st in DSTEP]
        durcomplete[arm] = all(v is not None for v in vals)
    durlog, seenscale = [], None
    for m, lab, arm, proto in DURROWS:
        if not durcomplete[arm]:
            durlog.append((m, arm, "incomplete: held out until every post-window cell is scored"))
            continue
        cells, got = [], False
        for st in DSTEP:
            v, n = _durcell(arm, CELLSTEP[proto][st], durbase[m])
            if v is None:
                cells.append("--")
                continue
            got = True
            cells.append(_durbold(m, DSTEP.index(st),
                                  "$" + pp(v) + (DAGGER if n < PARTIAL_N else "") + "$",
                                  100.0 * v, 1))
        bs, _, _, _ = BR.load_run(arm)
        ns, _, _ = NR.load_run(arm)
        # A row with no in-distribution cell AND no transfer cell is an arm that has not reported
        # yet: it is left out entirely rather than printed as a line of dashes, which would read
        # as a measured null. Its absence is a coverage clause in the caption.
        if not got and bs is None and ns is None:
            durlog.append((m, arm, "not reported"))
            continue
        if m != seenscale:
            if seenscale is not None:
                L.append("\\addlinespace[2pt]")
            L.append("\\multicolumn{%d}{@{}l}{\\textit{Qwen3-VL-$%s$B}} \\\\"
                     % (6 + len(DSTEP), m[:-1]))
            seenscale = m
        for jj, (r, bcell) in enumerate(((bs, True), (ns, False))):
            if r is None:
                cells += ["--", ""]
                continue
            v = BR.rate(r) if bcell else NR.rate(r)
            b0 = durbb[m] if bcell else durnb[m]
            cells.append(_durbold(m, len(DSTEP) + jj, "$%.2f$" % v, v, 2))
            cells.append("" if b0 is None else _durdelta(v - b0))
        L.append("\\quad %s & %s & %s \\\\" % (lab, PROTO_TEX[proto], " & ".join(cells)))
        durlog.append((m, arm, proto))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("durability.tex", L)

    # ---- THE 8B-ONLY POST-WINDOW RECORD THE REBUILT TABLE REPLACES ---------------------------
    # Kept, not deleted: these are real cells of arms the new table does not carry, on steps it
    # does not read. See DURLEGACY.
    b8 = durbase["8B"]
    L = [GEN,
         "% The 8B-only post-window record, on the heterogeneous steps each arm was scored at.",
         "% Superseded as the paper's durability exhibit by tab:durability, which reads three",
         "% scales on one convention; retained because these cells exist and are nobody else's.",
         "\\begin{tabular}{l%s}" % ("r" * len(DLSTEP)), "\\toprule",
         " & \\multicolumn{1}{c}{window} & \\multicolumn{%d}{c}{after the window} \\\\"
         % (len(DLSTEP) - 1),
         "\\cmidrule(lr){2-2}\\cmidrule(lr){3-%d}" % (len(DLSTEP) + 1),
         "configuration (pp) & %s \\\\"
         % " & ".join("$%d$" % st for st in DLSTEP), "\\midrule"]
    for lab, arm in DURLEGACY:
        cells = []
        for st in DLSTEP:
            v, n = _durcell(arm, st, b8)
            cells.append("--" if v is None
                         else "$" + pp(v) + (DAGGER if n < PARTIAL_N else "") + "$")
        L.append("%s & %s \\\\" % (lab, " & ".join(cells)))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("durability8b.tex", L)

    # ---- SECONDARY OUTCOME: policy termination quality ---------------------------------------
    L = [GEN,
         "% Secondary outcome, reported ALONGSIDE the solve rate and never instead of it.",
         "% max_turns rate on each arm's own held-out val episodes, first third -> last third.",
         "\\begin{tabular}{llcccc}", "\\toprule",
         " & & \\multicolumn{2}{c}{non-terminating (\\%)}"
         " & \\multicolumn{2}{c}{val solve rate (\\%)} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
         "scale & allocation & start & end & start & end \\\\", "\\midrule"]
    termlog = []
    for scale, rows in TERM:
        for i, (lab, arms) in enumerate(rows):
            got = [term_thirds(a) for a in arms]
            got = [g for g in got if g]
            if not got:
                continue
            m = [sum(x[j] for x in got) / len(got) for j in range(4)]
            L.append("%s & %s & $%.1f$ & $%.1f$ & $%.1f$ & $%.1f$ \\\\"
                     % (scale if i == 0 else "", lab, m[0], m[1], m[2], m[3]))
            termlog.append((scale, lab, len(got), m))
        L.append("\\addlinespace[2pt]")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
    W("termination.tex", L)

    # ---- PER-SEED VALIDATION SOLVE CURVES, the band Figure 4 no longer draws ------------------
    # Figure 4 plots seed MEANS and draws no seed band: the spread is 2-7pp against gaps of 1-5pp,
    # so a tinted band covered the comparison it was drawn beside. The spread is not allowed to
    # vanish with it, so every seed's own curve is reduced to the five numbers that characterise
    # it here -- the same window the figure plots, so the figure's means are recoverable from this
    # table and the reader can check that no gap in the figure exceeds the spread in this one.
    import val_curve as VC
    L = [GEN,
         "% Per-seed validation solve rate over the window Figure 4 plots (the fully-seeded span:",
         "% the steps at which BOTH arms of a scale still have all of their seeds).",
         "\\begin{tabular}{llrrrrr}", "\\toprule",
         "scale & allocation & steps & start & end & min & max \\\\", "\\midrule"]
    vlog = []
    for scale, rows in TERM:
        curves = {}
        for lab, arms in rows:
            per = {t: {r["step"]: 100 * r["solve"] for r in VC.solve_curve(t)} for t in arms}
            curves[lab] = {k: v for k, v in per.items() if v}
        # The window is defined exactly as the figure defines it, from the same curves.
        spans = [sorted(set.intersection(*(set(v) for v in c.values())))
                 for c in curves.values() if c]
        lo, hi = max(s[0] for s in spans), min(s[-1] for s in spans)
        for i, (lab, arms) in enumerate(rows):
            for j, tag in enumerate(a for a in arms if a in curves[lab]):
                v = curves[lab][tag]
                xs = [x for x in sorted(v) if lo <= x <= hi]
                ys = [v[x] for x in xs]
                L.append("%s & %s & $%d$ & $%.1f$ & $%.1f$ & $%.1f$ & $%.1f$ \\\\"
                         % (scale if i == 0 and j == 0 else "",
                            lab if j == 0 else "", len(xs), ys[0], ys[-1], min(ys), max(ys)))
            vlog.append((scale, lab, len(curves[lab]), lo, hi,
                         max((max(c[x] for c in curves[lab].values())
                              - min(c[x] for c in curves[lab].values())
                              for x in range(lo, hi + 1)
                              if all(x in c for c in curves[lab].values())), default=0.0)))
        L.append("\\addlinespace[2pt]")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
    W("valseed.tex", L)

    # ---- SCALE TRANSFER, BOTH BENCHMARKS, ONE TABLE ------------------------------------------
    # Merged 2026-08-18 from bfclscale.tex + nestful.tex. The two carried the SAME shape -- three
    # scales x {base, TRIAGE, uniform} x a TRIAGE-minus-uniform column -- as two floats a page
    # apart, so the paper's two-benchmark claim had to be assembled by the reader. Stacked under
    # spanning divider rows, the 8B NESTFUL null now sits three rows below the 8B BFCL positive.
    #
    # THE TWO THINGS THIS MERGE MUST NOT LOSE, and how each is kept:
    #  (1) The BENCHMARKS' STATISTICS ARE DIFFERENT and are never merged into one column. BFCL's
    #      difference is the MEAN OVER ALL SEED PAIRINGS with the range and the WEAKEST pairing's
    #      p; NESTFUL's is a single paired contrast with a Holm q over the m=3 scale family. Each
    #      block's divider row names its own statistic, so no reader can read one convention
    #      across the rule.
    #  (2) NESTFUL's STEP MISMATCH stays visible: 2B/4B at step 30 and 8B at step 15 (the only
    #      step ckpt_a8T holds) is a disclosed asymmetry, carried here as a row-label suffix. It
    #      is NOT promoted to a shared column, because BFCL's step is not one number per scale --
    #      its per-arm steps differ within a row and are in tab:bfcl.
    # Generated from the records rather than transcribed from the lab notebook, and aggregated:
    # the seed rows this table used to carry repeated the base column three times and invited a
    # reading ("the method beats control seed 1 by 12.1") that no seed pairing can support.
    L = [GEN,
         "% TWO BENCHMARKS, ONE TABLE, under spanning divider rows; the blocks' statistics are",
         "% DIFFERENT and each divider row names its own -- a difference cell is never comparable",
         "% across the rule.",
         "% BFCL: run_<arm>/records.jsonl via surface/verl_rl/bfcl_records.py; n=800 tasks per arm,",
         "% task index byte-identical across arms (checked, not assumed). Between-method quantities",
         "% are the MEAN over seed pairings with the range and the WEAKEST pairing's p -- never the",
         "% best pairing.",
         "% NESTFUL: run_<arm>/records.jsonl via surface/verl_rl/nestful_records.py; n=1861 tasks",
         "% per arm, one checkpoint per role (no seed aggregation -- see that module's header).",
         "% Holm over the m=3 scale family, pre-registered PLAN_TRIAGE.md 2026-08-18 02:05/08:25.",
         "\\begin{tabular}{lrccl}", "\\toprule",
         " & \\multicolumn{3}{c}{rate (\\%)}"
         " & \\methodname{} $-$ uniform \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}",
         "scale & base & \\methodname{} & uniform & (pp), with each block's own statistic \\\\",
         "\\midrule",
         "\\multicolumn{5}{l}{\\textbf{\\textsc{Bfcl} v4 multi-turn} ($n=800$/arm, pass rate)"
         ": mean over \\emph{all} seed pairings, range, weakest pairing's $p$} \\\\"]
    figrows = []
    for r in BR.scale_block():
        def cell(m, sd, n):
            # no seed count in the cell: the caption carries it in one clause
            s = "$%.1f" % m
            if sd is not None:
                s += "\\pm%.1f" % sd
            return s + "$"
        L.append("%s & $%.1f$ & %s & %s & $%+.1f$ \\ {\\small[$%+.1f,%+.1f$], $p\\le%s$} \\\\"
                 % (r["scale"], r["base"],
                    cell(r["tri_mean"], r["tri_sd"], len(r["tri"])),
                    cell(r["uni_mean"], r["uni_sd"], len(r["uni"])),
                    r["tu_mean"], r["tu_lo"], r["tu_hi"], _psci(r["tu_pmax"])))
        figrows.append(r)
    L += ["\\midrule",
          "\\multicolumn{5}{l}{\\textbf{\\textsc{Nestful}} ($n=1{,}861$/arm, win rate): one "
          "checkpoint per role, Holm-adjusted $p$ over the $m{=}3$ scale family} \\\\"]
    for r in nrows:
        L.append("%s \\ {\\small(step $%d$)} & $%.2f$ & $%.2f$ & $%.2f$ & "
                 "$%+.2f$, \\ $q=%s$ \\\\"
                 % (r["scale"], r["step"], r["base"], r["tri"], r["uni"], r["tu_diff"],
                    _psci(r["tu_q"])))
    L += ["\\midrule",
          "\\multicolumn{5}{l}{\\emph{uniform \\textsc{Grpo} against the untrained policy of its "
          "own scale} (\\textsc{Bfcl})} \\\\"]
    for r in figrows:
        L.append("%s & \\multicolumn{3}{l}{\\quad $%+.1f$\\,pp \\ {\\small[$%+.1f,%+.1f$]}} "
                 "& {\\small $p\\le%s$} \\\\"
                 % (r["scale"], r["ub_mean"], r["ub_lo"], r["ub_hi"], _psci(r["ub_pmax"])))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("transfer.tex", L)
    print("  -- NESTFUL (n=1861/arm), second block of transfer.tex --")
    for r in nrows:
        print("     %-4s step %-3d base %.2f  TRIAGE(%s) %.2f  uniform(%s) %.2f  "
              "gap %+.2f pp  q=%.3g" % (r["scale"], r["step"], r["base"], r["tri_arm"], r["tri"],
                                        r["uni_arm"], r["uni"], r["tu_diff"], r["tu_q"]))

    # ---- THE SAME TRANSFER EVIDENCE, WIDE, FOR THE MAIN TEXT --------------------------------
    # See TRANSFER_FAITHFUL for the row set and for the step convention. Every base / control /
    # method cell and both difference rows come from the SAME BR.scale_block() and NR.scale_block()
    # results tab:transfer is built from, a few lines above, so the two floats cannot disagree.
    brow = {r["scale"]: r for r in figrows}
    nrow = {r["scale"]: r for r in nrows}
    missing = [s for s in MAIN_SCALES if s not in brow or s not in nrow]
    if missing:
        raise SystemExit("tab:transfermain: no transfer block at %s" % missing)
    DAG = "---$^{\\dagger}$"

    def _nest(arm, want=None):
        """One NESTFUL cell as rate-with-its-own-step, or the coverage dash."""
        if arm is NOT_RUN:
            return DAG
        s, st, _ = NR.load_run(arm)
        if s is None:
            return DAG
        if len(s) != NR.EXPECTED_N:
            raise SystemExit("tab:transfermain: %s has %d NESTFUL records, not %d"
                             % (arm, len(s), NR.EXPECTED_N))
        return "$%.2f_{\\,%s}$" % (NR.rate(s), "" if st is None else str(int(st)))

    def _bfcl(arm):
        if arm is NOT_RUN:
            return DAG
        s, _, _, _ = BR.load_run(arm)
        if s is None:
            return DAG
        return "$%.1f$" % BR.rate(s)

    # 2026-08-28, BEST-SEED PASS. The TRIAGE and uniform rows now print each arm's BEST SEED,
    # chosen by the SAME window-mean rule tab:main uses and NEVER by a transfer score -- selecting
    # a transfer cell on its own value would be exactly the maximum-over-outcomes this paper
    # refuses. Two facts about coverage make this table's application of the rule narrower than
    # tab:main's, and both are printed under the table rather than absorbed:
    #  * BFCL is multi-seed and takes the rule. The selected arms are computed here from the SAME
    #    pinned families the window tables use, and the candidate set is BR.SCALES' own arm lists.
    #    At 2B the TRIAGE candidates are q2bT and q2bT2, and q2bT2 HAS NO COADAPT WINDOW AT ALL
    #    (it is the transfer-only replicate), so it cannot be ranked and is not a candidate: the
    #    2B cell is q2bT, a best-of-ONE-rankable-seed, and the note says the maximum ranged over
    #    one run. Its transfer value is not lost -- it is in tab:transfer and tab:bfcl2b.
    #  * NESTFUL IS NOT MULTI-SEED AND TAKES NO SELECTION. Exactly one checkpoint per role per
    #    scale was registered and scored (PLAN_TRIAGE 2026-08-18 08:25), so there is no seed to
    #    choose between and these cells are UNCHANGED by this pass. Saying "best seed" of a
    #    single registered run would be false, so the note says which convention each half of the
    #    table is under. At 8B the registered TRIAGE arm (a8T) is not even one of the fixed-rho
    #    seeds, which is the same disclosed asymmetry tab:transfer has always carried.
    # The difference row therefore loses its significance markers: it is now the difference of the
    # two DISPLAYED rows, and the displayed BFCL rows are a selection, so a p computed over seed
    # pairings no longer describes it. The registered paired contrasts and their p keep their home
    # in tab:transfer and tab:nestrep, and the note points there.
    trisel, unisel = {}, {}
    for s in MAIN_SCALES:
        trisel[s] = best_seed(fams[s][0], brow[s]["tri_arms"])
        unisel[s] = best_seed(fams[s][0], brow[s]["uni_arms"])
        if trisel[s] is None or unisel[s] is None:
            raise SystemExit("tab:transfermain: no rankable seed at %s (tri %s, uni %s) -- the "
                             "best-seed rule cannot be applied to a row with no window"
                             % (s, brow[s]["tri_arms"], brow[s]["uni_arms"]))
    tri_b = {s: BR.rate(BR.load_run(trisel[s])[0]) for s in MAIN_SCALES}
    uni_b = {s: BR.rate(BR.load_run(unisel[s])[0]) for s in MAIN_SCALES}
    # 2026-08-29 TABLE-ENRICHMENT PASS (PI directive). This table was six rows; it is now every
    # arm in the paper that has a measured cell on either held-out benchmark, grouped exactly as
    # tab:main groups its rows, with a DELTA AGAINST THE UNTRAINED BASE POLICY beside every rate.
    #
    # THE ROW SET IS NOT A NEW MEASUREMENT AND NOT A NEW SELECTION RULE:
    #  * the arm lists are CFG_ARMS / CONTROL_ARMS / PUBLISHED_REIMPL / TRANSFER_FAITHFUL, the same
    #    lists tab:main walks, so a configuration cannot appear in one main table and vanish from
    #    the other with nothing in the source saying so;
    #  * each multi-seed row prints the seed with the highest four-step WINDOW mean -- tab:main's
    #    rule, never a transfer score. Verified in this emitter rather than asserted: the selection
    #    computed over every seed in the pinned family is compared against the selection computed
    #    over only the seeds that were scored on BFCL, and the emitter REFUSES to write the table
    #    if they ever differ, because that is the case where the rule would print a coverage dash
    #    for a configuration that has a measured cell on another seed. They agree for every row of
    #    this emission;
    #  * the METHOD and CONTROL rows keep their PRE-REGISTERED NESTFUL checkpoints (NR.SCALES), so
    #    the m=3 registered contrast at the bottom of the table is the contrast that was fixed
    #    before any cell was scored. Every other row's NESTFUL cell is read on the SAME seed its
    #    BFCL cell is read on -- which is a choice made by the window and never by a NESTFUL score;
    #  * a cell that was never measured prints ---(dagger). NOTHING is substituted for it. The
    #    published reimplementations were never scored on NESTFUL at any scale, and that whole
    #    block of dashes is a coverage fact this table now states instead of leaving to an
    #    appendix caption.
    #
    # DELTAS are the difference of the DISPLAYED cell and the DISPLAYED base cell of that
    # benchmark and scale, computed unrounded, and they carry NO significance marker: the BFCL
    # rates are selected seeds, so no p over seed pairings describes them. The registered paired
    # contrasts keep their home in tab:transfer / tab:nestrep and the note points there.
    #
    # THE METHOD ROW OF THIS TABLE IS FIXED-RHO AT EVERY SCALE, which at 4B is NOT the
    # configuration tab:main's method row carries (that row is the scale-refit revision). This is
    # the transfer tables' long-standing convention -- every paired contrast in the paper is built
    # on fixed-rho -- and the 4B scale-refit arm is printed one block below, so the reader sees
    # both. The caption says it in those words.
    # ONE SYMBOL, ONE MEANING, ACROSS BOTH MAIN-TEXT TABLES, and set the same way in both:
    # \\textsuperscript, not a math superscript, so the two floats' markers are the same size and
    # sit at the same height. The dagger means "not measured here: coverage, not a null" in
    # tab:main and in this table; the double dagger is this table's alone (nothing is pruned in
    # tab:main); the section sign is this table's alone (tab:main has no summary row); and the
    # star is tab:main's alone (this table carries no significance marker at all, which its own
    # note says). The notes below explain them in that order, as tab:main's note does.
    DAG = "---\\textsuperscript{\\dag}"
    # 2026-08-31: THERE IS NO "NOT MEASURABLE" CELL IN THIS TABLE ANY MORE. Every arm whose
    # step-30 adapter was pruned and that this table displays has been re-trained (TRANSFER_RERUN)
    # and the rerun is substituted upstream, at the arm pick, so both of its cells move together.
    # If a pruned arm ever reaches this helper it means a row is displaying one with no rerun
    # registered, and that is a refusal rather than a dash: printing "not measurable" over a cell
    # somebody could measure is the state this fold exists to remove.
    RERUN = "\\textsuperscript{\\ddag}"

    def _nest_cell(arm, step=None):
        """One NESTFUL cell as rate-with-its-own-step, or the coverage dash, plus its value."""
        if arm is NOT_RUN or arm is IN_FLIGHT or arm is None:
            return DAG, None
        if arm in NESTFUL_PRUNED:
            raise SystemExit("tab:transfermain: %s has no scorable NESTFUL checkpoint and no "
                             "rerun in TRANSFER_RERUN -- register the rerun, do not print a dash"
                             % arm)
        s, st, _ = NR.load_run(arm)
        if s is None:
            return DAG, None
        if len(s) != NR.EXPECTED_N:
            raise SystemExit("tab:transfermain: %s has %d NESTFUL records, not %d"
                             % (arm, len(s), NR.EXPECTED_N))
        st = step if step is not None else st
        return "$%.2f_{\\,%s}$" % (NR.rate(s), "" if st is None else str(int(st))), NR.rate(s)

    def _method_nest(s_):
        """The METHOD row's NESTFUL cell at one scale: (text, value, step).

        ONE DEFINITION, read by the row, by the per-column best and by the difference row, so the
        three cannot disagree about what the method's NESTFUL cell is. Without the override this
        is the pre-registered checkpoint at its registered step, exactly as before; with it, the
        row's own selected seed. See METHOD_NEST for why 8B has one.
        """
        ov = METHOD_NEST.get(s_)
        if ov is None:
            return ("$%.2f_{\\,%d}$" % (nrow[s_]["tri"], nrow[s_]["step"]),
                    nrow[s_]["tri"], nrow[s_]["step"])
        arm, st = ov
        cur, gst, _ = NR.load_run(arm, step=st)
        if cur is None:
            raise SystemExit("tab:transfermain: METHOD_NEST[%r] names %s@%s and it has no "
                             "records" % (s_, arm, st))
        if len(cur) != NR.EXPECTED_N:
            raise SystemExit("tab:transfermain: %s@%s has %d NESTFUL records, not %d"
                             % (arm, st, len(cur), NR.EXPECTED_N))
        return "$%.2f_{\\,%d}$" % (NR.rate(cur), gst), NR.rate(cur), gst

    def _bfcl_cell(arm):
        if arm is NOT_RUN or arm is IN_FLIGHT or arm is None:
            return DAG, None
        s, _, _, _ = BR.load_run(arm)
        if s is None:
            return DAG, None
        return "$%.1f$" % BR.rate(s), BR.rate(s)

    def _delta(v, base, dec):
        # Set in the Delta columns' lighter ink -- see grey() for why the tint is black!60 and
        # not the figure palette's trGrey. The cell is the same signed number either way.
        return "" if v is None else grey("$%+.*f$" % (dec, v - base))

    def _pick(arms, model):
        """tab:main's rule: the seed with the highest window mean, among the pinned family.

        The second selection below is the same rule restricted to the seeds that were scored on
        BFCL. It exists only to be compared: if the two ever disagree, a row's displayed seed has
        no transfer cell while a sibling seed does, and the table would print a dash where a
        measurement exists. That is a refusal, not a fallback.
        """
        if arms is NOT_RUN or arms is IN_FLIGHT or arms is SAME_FIXED:
            return arms
        f = fams[model][0]
        wide = best_seed(f, [a for a in arms if a in f])
        scored = best_seed(f, [a for a in arms if a in f and BR.load_run(a)[0] is not None])
        if scored is not None and wide != scored:
            raise SystemExit("tab:transfermain: at %s the best-window seed (%s) has no BFCL cell "
                             "while %s does -- resolve deliberately, do not print a dash over a "
                             "measured sibling" % (model, wide, scored))
        return wide

    # The row list, in the PI's order: method, our other configurations, control, the published
    # rules we reimplemented, the published allocators run as published. "@REG" marks the two rows
    # whose NESTFUL cell is the pre-registered checkpoint rather than the selected seed's.
    # ---- THE PER-COLUMN BEST CONFIGURATION (PI decision 2026-08-29) --------------------------
    # ONE SUMMARY ROW at the top of the Method block. For each benchmark x scale COLUMN it prints
    # the maximum over THE FRAMEWORK'S OWN CONFIGURATIONS -- exactly the rows printed directly
    # beneath it -- and never a cell from a baseline, a control or a faithful row. It is a
    # selection made after the measurements existed, on the same displayed statistic, and both the
    # caption and the table note say so in those words; every configuration it ranges over is
    # printed in full in the same table, so the row summarises rather than replaces.
    #
    # THE CELLS ARE THE DISPLAYED CELLS, produced by the same two helpers the rows below use --
    # never a recomputation -- so the summary can never carry a number that appears nowhere else.
    # Consequences of that, each deliberate: the fixed-rho candidate's NESTFUL value is the
    # PRE-REGISTERED checkpoint (that is what its row shows); a "not measurable" cell is not a
    # candidate (there is no number to rank); and a configuration that COINCIDES with fixed-rho at
    # a scale is not counted twice.
    #
    # MARKER. \S and not the star: in every other float of this paper a star means "clears Holm in
    # its own family", and this table prints no significance marker at all. \dag and \ddag are
    # already spoken for here by the two coverage states.
    def _cfg_cells(k, s_):
        """One configuration's DISPLAYED transfer cells at one scale: (bfcl, nestful) pairs."""
        arms = CFG_ARMS[k][s_]
        if arms is NOT_RUN or arms is IN_FLIGHT or arms is SAME_FIXED:
            return (None, None), (None, None)
        arm = _pick(arms, s_)
        # the same rerun substitution the rows below apply, so the summary ranks the cells the
        # table actually prints and never a value that appears nowhere in it
        arm = TRANSFER_RERUN.get(arm, arm) if isinstance(arm, str) else arm
        b = _bfcl_cell(arm)
        if k == "fixed":
            # the DISPLAYED method cell, so the per-column best ranks what the row shows
            mt, mv, _ = _method_nest(s_)
            n = (mt, mv)
        else:
            n = _nest_cell(arm)
        # A summary cell that came off a rerun carries the rerun's marker too. Without this the
        # per-column best could print a number the reader cannot tell is a rerun's while the row
        # it was taken from, two lines below, is marked -- the same cell, disclosed in one place
        # and not the other.
        if isinstance(arm, str) and arm in TRANSFER_RERUN.values():
            b = (b[0] + RERUN, b[1]) if b[1] is not None else b
            n = (n[0] + RERUN, n[1]) if n[1] is not None else n
        return b, n

    BESTCFG, bestlog = {}, []
    for j, bm in enumerate(("bfcl", "nest")):
        for s_ in MAIN_SCALES:
            got = []
            for k in CFG_ORDER:
                txt, val = _cfg_cells(k, s_)[j]
                if val is not None:
                    got.append((val, k, txt))
            if not got:
                raise SystemExit("tab:transfermain: no configuration has a %s cell at %s -- the "
                                 "per-column summary row would print a dash over a table that "
                                 "shows measurements" % (bm, s_))
            val, k, txt = max(got)
            BESTCFG[(bm, s_)] = (k, txt, val)
            bestlog.append((bm, s_, k, val, [g[1] for g in got]))

    TMROWS = ([("@HDR", "\\textbf{Method}", None),
               # LABEL LENGTH IS A HARD CONSTRAINT HERE, not a style choice: this table measures
               # 386pt against a 397.5pt \textwidth at \scriptsize, so the label column has about
               # 11pt of slack and "(best configuration per column)" overran it by 45pt. The row
               # says "best per column" and the caption and the note below say what a column's
               # best is selected over. Nothing was dropped, only moved off the widest row.
               ("@BEST", "\\quad \\methodname{} (column best)\\textsuperscript{\\S}", None),
               ("@REG", "\\quad \\methodname{} (full)", CFG_ARMS["fixed"]),
               ("@HDR", "\\textbf{This framework, other configurations}", None)]
              + [("", "\\quad " + CFG_LABEL[k], CFG_ARMS[k]) for k in CFG_ORDER if k != "fixed"]
              + [("@HDR", "\\textbf{Control}", None),
                 ("@REG", "\\quad uniform \\textsc{Grpo}", CONTROL_ARMS),
                 ("@HDR", "\\textbf{Published baselines}", None)]
              + [("", lab, spec) for lab, spec in PUBLISHED_REIMPL]
              + [("@HDR", "\\textbf{Published allocators, run as published}", None)]
              + [("@FAITH", lab, (bm, nm)) for lab, bm, nm in TRANSFER_FAITHFUL])

    # The unit of every Delta column of this table, carried in the header and never in a cell,
    # exactly as tab:main carries "(pp)" under its own Delta. Named once here because it is
    # printed six times.
    DELTA_HDR = "$\\Delta$ (pp)"
    LT = [GEN,
          "% MAIN-TEXT transfer table. Same records as tab:transfer, read from the same",
          "% BR.load_run()/NR.load_run() calls, laid out wide: rules as rows, the two held-out",
          "% benchmarks as column groups, the three model sizes inside each group, and each rate",
          "% followed by its delta against the untrained base policy of that scale.",
          "% BFCL: n=800 tasks per arm, pass rate, task index byte-identical across arms (checked).",
          "%   SEED CONVENTION: every multi-seed row is that arm's BEST SEED by four-step WINDOW",
          "%   mean (tab:main's rule) and never by a transfer score.",
          "%   The 2B TRIAGE maximum ranged over ONE rankable seed: the other 2B replicate scored",
          "%   on BFCL (q2bT2) has no coadapt window, so it has no window mean to rank. Its cell",
          "%   is in tab:transfer and tab:bfcl2b, not dropped.",
          "%   No +- is printed anywhere in this table: these are single runs, and the seed sd is",
          "%   undefined for the base and the two faithful rows as well.",
          "%   Seed means, ranges and per-seed values: tab:transfer, tab:bfcl, tab:bfcl4b,",
          "%   tab:bfcl2b.",
          "% NESTFUL: n=1861 tasks per checkpoint, win rate, ONE checkpoint per cell -- so each",
          "%   cell carries ITS OWN STEP as a subscript. The column is genuinely not step-uniform",
          "%   (the registered 8B cells are step 15, the faithful rows are step 30, and the 8B",
          "%   plug-in configuration is step 25) and the subscripts are how that is disclosed",
          "%   rather than hidden. The METHOD and CONTROL rows carry the PRE-REGISTERED",
          "%   checkpoints; every other row is read on the seed its BFCL cell is read on.",
          "% The difference row is the difference of the two displayed rows, computed unrounded,",
          "% and carries NO significance marker, and neither does any delta: the BFCL half is a",
          "% difference of selected seeds, which no p over seed pairings describes. The registered",
          "% paired contrasts -- BFCL's mean over all seed pairings with its weakest pairing's",
          "% exact p, and NESTFUL's single registered contrast with its Holm q over the",
          "% pre-registered m=3 scale family -- are unchanged in tab:transfer and tab:nestrep.",
          # Explicit separations, as in tab:main: a rate and its own delta sit 3pt apart, scales
          # 6pt apart, the two benchmarks 10pt apart, so the three groupings are visible without
          # a single vertical rule. THE SCALE GAP IS 6pt AND NOT 7pt SINCE 2026-08-29, AND THAT
          # 4pt IS WHAT PAYS FOR THE "(pp)" IN THE DELTA HEADERS: with the unit in the header the
          # grid measures 400.4pt against a 397.5pt \textwidth, and 396.4pt with the four scale
          # gaps at 6pt. The ordering that makes the three groupings readable is untouched
          # (3pt inside a scale < 6pt between scales < 10pt between benchmarks), and no unit
          # moved into a cell to buy the space.
          # 2026-08-29 (PI directive, DRAFTING pass): EVERY SEPARATION STEPS DOWN ONE NOTCH --
          # 5->4 after the label, 3->2 inside a scale, 6->5 between scales, 10->8 between the two
          # benchmarks -- and that is what pays for the formal published-row labels this table
          # reads from PUBLISHED_REIMPL. MEASURED: with the formal labels and the old separations
          # the grid is 406.6pt against a 397.5pt \textwidth (the widest label goes from
          # "\methodname{} (column best)" at 68.6pt to "RAG-style retrieval surface" at 79.7pt,
          # +11.2pt); with them it is 394.0pt, which is 1.5pt NARROWER than the 395.4pt this
          # table measured before the pass. THE THREE-TIER ORDERING THAT MAKES THE GROUPINGS
          # READABLE IS PRESERVED, which is the property the 08-29 note below defends: 2pt inside
          # a scale < 5pt between scales < 8pt between benchmarks. 2pt inside a group is not a new
          # tightness for this paper -- it is exactly what tab:main sets. NO UNIT MOVED INTO A
          # CELL and no number, marker or subscript moved.
          "\\begin{tabular}{@{}l@{\\hspace{4pt}} r@{\\hspace{2pt}}r @{\\hspace{5pt}}"
          " r@{\\hspace{2pt}}r @{\\hspace{5pt}} r@{\\hspace{2pt}}r @{\\hspace{8pt}}"
          " r@{\\hspace{2pt}}r @{\\hspace{5pt}} r@{\\hspace{2pt}}r @{\\hspace{5pt}}"
          " r@{\\hspace{2pt}}r@{}}", "\\toprule",
          # TWO-LEVEL HEADER, the same two levels tab:main sets: the benchmark GROUPS on their
          # own row under \\cmidrule(lr), each carrying its metric, its unit and its own n, then
          # the per-column scales. The benchmark name and its "pass rate (\%), n=800" used to be
          # two separate spanning rows; folding them costs NO width (a group span was never this
          # grid's binding column) and gives the float back a line, which matters because the
          # Conclusion has to end on page 9. Units live here, never in a cell.
          " & \\multicolumn{6}{c}{\\textsc{Bfcl} v4 multi-turn, pass rate (\\%), $n=800$}"
          " & \\multicolumn{6}{c}{\\textsc{Nestful}, win rate (\\%), $n=1{,}861$} \\\\",
          "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}",
          "allocation rule & "
          + " & ".join(["$2$B", grey(DELTA_HDR), "$4$B", grey(DELTA_HDR),
                        "$8$B", grey(DELTA_HDR)] * 2) + " \\\\",
          "\\midrule",
          "untrained base policy & %s \\\\"
          % " & ".join(["$%.1f$ & " % brow[s]["base"] for s in MAIN_SCALES]
                       + ["$%.2f$ & " % nrow[s]["base"] for s in MAIN_SCALES]).rstrip(" &")]
    tmlog = []
    seen = False
    # ---- THE SAME DISPLAYED CELLS, CAPTURED AS DATA FOR fig:transfer -------------------------
    # 2026-08-29 (PI: the main-text transfer float becomes a FIGURE). fig:transfer plots exactly
    # the cells this table prints, and the only way to guarantee that is to hand the figure THIS
    # loop's own values rather than let paper_figures.py re-derive them from the loaders. So every
    # value the moment it is formatted into a cell is also appended here, unrounded, and written
    # beside the table as transferfig.json. paper_figures.py refuses to draw if that file is
    # missing or if its recorded hash does not match the transfermain.tex it sits next to, so a
    # figure built on a stale emission cannot be produced silently. NOTHING in LT changes here:
    # the capture is append-only and the emitted table is byte-identical with and without it.
    figrows_out, figblock = [], None
    for kind, lab, spec in TMROWS:
        if kind == "@HDR":
            LT.append("\\addlinespace[2pt]")
            LT.append("\\multicolumn{13}{@{}l}{%s} \\\\" % blocklab(lab))
            seen = True
            figblock = lab
            continue
        # A scale whose configuration COINCIDES with fixed-rho spans both of its columns with one
        # statement instead of printing two dashes that would read as a coverage gap.
        # No footnote marker here: "$=$ fixed-rho" says what it is, and in THIS table the double
        # dagger is already spoken for by the pruned-checkpoint cells.
        SAMEFIX2 = "\\multicolumn{2}{c}{$=$ full}"
        if kind == "@BEST":
            bcells, blog = [], []
            for bm in ("bfcl", "nest"):
                for s_ in MAIN_SCALES:
                    k, txt, val = BESTCFG[(bm, s_)]
                    bcells.append(txt)
                    bcells.append(_delta(val, (brow if bm == "bfcl" else nrow)[s_]["base"],
                                         1 if bm == "bfcl" else 2))
                    blog.append("%s %s %s" % (bm, s_, CFG_LABEL[k]))
            LT.append("%s & %s \\\\" % (IND(lab), " & ".join(bcells)))
            tmlog.append((lab, blog))
            figrows_out.append({"block": figblock, "label": lab, "kind": kind,
                                "bfcl": {s_: BESTCFG[("bfcl", s_)][2] for s_ in MAIN_SCALES},
                                "nest": {s_: BESTCFG[("nest", s_)][2] for s_ in MAIN_SCALES},
                                "best_cfg": {bm: {s_: BESTCFG[(bm, s_)][0] for s_ in MAIN_SCALES}
                                             for bm in ("bfcl", "nest")}})
            continue
        cells, log = [], []
        figb, fign = {}, {}
        # THE DISPLAYED ARM AT EACH SCALE, decided once and reused by both benchmarks, so a row
        # can never print a BFCL cell from one run beside a NESTFUL cell from another.
        xarm = {}
        for s_ in MAIN_SCALES:
            a0 = spec[0][s_] if kind == "@FAITH" else _pick(spec.get(s_), s_)
            xarm[s_] = TRANSFER_RERUN.get(a0, a0) if isinstance(a0, str) else a0
        for s_ in MAIN_SCALES:
            arm = xarm[s_]
            if arm is SAME_FIXED:
                cells.append(SAMEFIX2)
                log.append("%s = fixed-rho" % s_)
                figb[s_] = "@SAMEFIXED"
                continue
            txt, val = _bfcl_cell(arm)
            if isinstance(arm, str) and arm in TRANSFER_RERUN.values():
                txt += RERUN
            cells.append(txt)
            cells.append(_delta(val, brow[s_]["base"], 1))
            log.append("%s %s %s" % (s_, arm if isinstance(arm, str) else "-", txt))
            figb[s_] = val
        for s_ in MAIN_SCALES:
            if kind == "@REG":
                # review P0-5, RESOLVED 2026-08-29. The METHOD row now prints ITS OWN SEED:
                # _method_nest applies METHOD_NEST, which at 8B is a8T3g2 at step 30, the fixed-
                # rho row's selected seed. Until this fold the cell was the pre-registered a8T
                # checkpoint at step 15, which tab:nestrep reads as the cold-start v1 arm (window
                # -1.5 pp at q = 1.000) while this row is labelled "\methodname{} (fixed-rho)":
                # one arm labelled, another measured. The CONTROL row is unchanged and still
                # prints its pre-registered checkpoint at its registered step. The m=3 registered
                # contrast is NOT this table's difference row and never was; it lives in
                # tab:transfer and tab:nestrep, built by nestful_records.SCALES, which still names
                # a8T against a8F at step 15 and is untouched by this change.
                # 2026-08-29: METHOD_TEX, not the name. As `"Triage" in lab` this test went
                # False at the rename and the METHOD row printed the CONTROL's NESTFUL
                # cells -- the two rows came out identical while the difference row below
                # still printed a nonzero difference of them.
                if METHOD_TEX in lab:
                    txt, v, _ = _method_nest(s_)
                else:
                    v = nrow[s_]["uni"]
                    txt = "$%.2f_{\\,%d}$" % (v, nrow[s_]["step"])
            elif kind == "@FAITH":
                txt, v = _nest_cell(spec[1][s_])
            else:
                # xarm, not a second _pick: the same displayed arm the BFCL half used, rerun
                # substitution included, so the two halves of a row are one run.
                a_ = xarm[s_]
                if a_ is SAME_FIXED:
                    cells.append(SAMEFIX2)
                    fign[s_] = "@SAMEFIXED"
                    continue
                txt, v = _nest_cell(a_)
                if isinstance(a_, str) and a_ in TRANSFER_RERUN.values():
                    txt += RERUN
            cells.append(txt)
            cells.append(_delta(v, nrow[s_]["base"], 2))
            fign[s_] = v
        LT.append("%s & %s \\\\" % (IND(lab), " & ".join(cells)))
        tmlog.append((lab, log))
        figrows_out.append({"block": figblock, "label": lab, "kind": kind,
                            "bfcl": dict(figb), "nest": dict(fign)})
    LT += ["\\midrule",
           # THE DIFFERENCE ROW, set off by the one \\midrule this table has below its header and
           # by an italic label: it is not an allocation rule, it is two of the rows above
           # subtracted. \\textup around \\methodname is not decoration -- the printed name is
           # small caps, OT1/ptm has no italic small caps, and without it every build logs a
           # font-shape substitution warning for a shape that would render upright anyway.
           # THE NESTFUL HALF IS BUILT FROM _method_nest, NOT FROM nrow["tri"], so it is the
           # subtraction of the two cells actually printed above it. Where the two cells are read
           # at different steps -- 8B, since the method cell moved to step 30 and the control was
           # only ever scored at 15 -- the difference carries \P and the note says so. It is
           # printed rather than withheld: the cells are both real and the reader is told what
           # the subtraction is.
           "\\textit{\\textup{\\methodname{}} $-$ uniform (pp)} & %s \\\\"
           % " & ".join(["$%+.1f$ & " % (tri_b[s] - uni_b[s]) for s in MAIN_SCALES]
                        + ["$%+.2f$%s & " % (_method_nest(s)[1] - nrow[s]["uni"],
                                             "\\textsuperscript{\\P}"
                                             if _method_nest(s)[2] != nrow[s]["step"] else "")
                           for s in MAIN_SCALES]).rstrip(" &"),
           "\\bottomrule"] + notes_block([
        # MARKER ORDER IS THE SAME IN BOTH MAIN-TEXT TABLES (2026-08-29 presentation pass III):
        # star, dagger, double dagger, section sign, and then the prose sentence about the
        # "$=$ full" cells. tab:main has the star and no double dagger or section sign;
        # this table has the last three and no star. Every sentence below is the one that stood
        # before the pass, word for word; only the section sign's paragraph changed, so that a
        # reader meets the markers in one order in both floats.
        "Each $\\Delta$ is the cell to its left minus the untrained base policy of the same"
        " benchmark and scale, computed unrounded; subscripts are the checkpoint step a"
        " \\textsc{Nestful} cell is read at.\\quad \\textsuperscript{\\dag}~not measured on this"
        " benchmark at this scale: coverage, not a null.\\quad"
        " \\textsuperscript{\\ddag}~both transfer cells of this row at this scale are read on a"
        " \\emph{rerun}: the arm's step-$30$ adapter was pruned once its window was scored, so the"
        " configuration was trained again at the same seed offset, and a rerun is a fresh draw"
        " rather than a second reading of one run. Its window cells are its own"
        " (Appendix~\\ref{app:reportrules} names the three arms) and the row's window cells in"
        " Table~\\ref{tab:main} are still"
        " the original run's.\\quad \\textsuperscript{\\S}~the per-column maximum over the"
        " configurations below it, printed as measured. No claim in this paper is read off it:"
        " the method row of Table~\\ref{tab:main}, Figure~\\ref{fig:transfermain}'s"
        " \\methodname{} bar and the \\methodname{} $-$ uniform row at the foot are all the"
        " full recipe."
        "\\quad \\textsuperscript{\\P}~a \\emph{step-mismatched} difference: the \\methodname{}"
        " cell above it is read at step $30$ and the control's at step $15$, the only step that"
        " control checkpoint was scored at.",
        # ONE note paragraph where there were three (2026-08-29 P0 pass). This float has to share
        # its page with Figure 3 or the Conclusion loses page 9, and each paragraph break here
        # cost a partial line at \scriptsize. No sentence was dropped and no marker lost its
        # explanation; the caption as written is archived verbatim in app:capfull.
        "A cell reading \"$=$ full\" is not a gap: at $2$B and $4$B the sharpened and"
        " the fixed-$\\rho$ configuration are the same configuration."
        "\\quad \\textsc{Bfcl} cells are each arm's \\emph{best seed} by window mean"
        " (Table~\\ref{tab:main}'s rule, not a transfer maximum); the control row's"
        " \\textsc{Nestful} cells and the $2$B and $4$B method cells are the"
        " \\emph{pre-registered} checkpoints, the $8$B method cell is that row's own selected seed"
        " at step $30$, and every other one is the"
        " seed its \\textsc{Bfcl} cell is read on. \\emph{No significance marker appears here}:"
        " the registered paired-seed contrasts and their $p$ are in"
        " Appendix~\\ref{app:genfull} and, for the seed-replicated \\textsc{Nestful} contrast,"
        " among the floats withdrawn in Appendix~\\ref{app:movedseeds}."])
    # The method and control BFCL cells above are produced by _pick()/_bfcl_cell(); tri_b/uni_b
    # come from BR.scale_block()'s own arm lists and feed the difference row. They must be the
    # same numbers or the table would disagree with its own last line.
    for s_ in MAIN_SCALES:
        got_t, _, _, _ = BR.load_run(_pick(CFG_ARMS["fixed"][s_], s_))
        got_u, _, _, _ = BR.load_run(_pick(CONTROL_ARMS[s_], s_))
        if abs(BR.rate(got_t) - tri_b[s_]) > 1e-9 or abs(BR.rate(got_u) - uni_b[s_]) > 1e-9:
            raise SystemExit("tab:transfermain: the row path and the difference path disagree at "
                             "%s -- one table, two answers" % s_)
    W("transfermain.tex", LT)
    # transferfig.json: the SAME displayed values, for fig:transfer. Written beside the table and
    # carrying the sha256 of the exact transfermain.tex bytes just written, so the figure emitter
    # can prove it is drawing the shipped table's numbers and not an earlier emission's.
    import hashlib as _hashlib
    _tmtex = "\n".join(LT) + "\n"
    open("%s/transferfig.json" % outdir, "w").write(json.dumps({
        "_generated_by": "paper_numbers.py --emit-tables; consumed by paper_figures.py",
        "transfermain_sha256": _hashlib.sha256(_tmtex.encode("utf-8")).hexdigest(),
        "scales": MAIN_SCALES,
        # cfg key -> the ROW LABEL that configuration is printed under, so the figure can ring the
        # per-column best configuration on its own row without re-deriving the row set.
        "cfg_row_label": dict([("fixed", "\\quad " + METHOD_TEX + " (fixed-$\\rho$)")]
                              + [(k, "\\quad " + CFG_LABEL[k])
                                 for k in CFG_ORDER if k != "fixed"]),
        "base": {"bfcl": {s_: brow[s_]["base"] for s_ in MAIN_SCALES},
                 "nest": {s_: nrow[s_]["base"] for s_ in MAIN_SCALES}},
        "diff": {"bfcl": {s_: tri_b[s_] - uni_b[s_] for s_ in MAIN_SCALES},
                 "nest": {s_: nrow[s_]["tri"] - nrow[s_]["uni"] for s_ in MAIN_SCALES}},
        "rows": figrows_out}, indent=1, sort_keys=True) + "\n")
    print("  -- tab:transfermain (main-text transfer, both benchmarks, three scales) --")
    for lab, log in tmlog:
        print("     %-40s %s"
              % (lab.replace("\\quad ", "").replace("\\textsc{", "").replace("}", ""),
                 "  ".join(log)))

    # ---- THE LEARNING-RATE LADDER (internal review item 3 / W5) ------------------------------
    def _bfcl(arm):
        s, _, st, _ = BR.load_run(arm)
        return (BR.rate(s), st) if s else (None, None)

    lrbase, _ = _bfcl(LR_BASE)
    L = [GEN,
         "%% 2B anchor n=%d; the 2B rungs are in the same %d-arm 2B Holm family as tab:main"
         % (fams["2B"][1], fams["2B"][2]),
         "%% 4B anchor n=%d; the 4B rungs are in the same %d-arm 4B Holm family as tab:main"
         % (fams["4B"][1], fams["4B"][2]),
         "% One run per rung, all at seed offset 500: the rungs differ from each other by the LR",
         "% token alone, and a rung on a fresh offset would confound the learning rate with the",
         "% surface draw. The seed-aggregated values for these two CONFIGURATIONS are tab:main's.",
         "% The transfer column is the SAME BFCL evaluation as tab:transfer's, n=800, step 30.",
         "% The 4B block (2026-09-04) is the same construction one scale up; each block's delta",
         "% is against ITS OWN scale's untrained base policy, never across scales.",
         "\\begin{tabular}{llrrr}", "\\toprule",
         " & learning & window & \\textsc{Bfcl} & $\\Delta$ vs \\\\",
         "allocation & rate & mean (pp) & pass (\\%) & base (pp) \\\\", "\\midrule",
         "untrained base policy & --- & --- & $%.1f$ & --- \\\\" % lrbase]
    lrlog = []
    for lab, lr, arm in LR_LADDER:
        a = agg(fams["2B"][0], [arm])
        b, _ = _bfcl(arm)
        L.append("%s & %s & $%s$ & $%.1f$ & $%+.1f$ \\\\"
                 % (lab, lr, pp(a["mean"]), b, b - lrbase))
        lrlog.append((lab, lr, arm, 100 * a["mean"], b))
    # ---- THE SAME TWO RATES AT 4B (2026-09-04 _fold10) ---------------------------------------
    # Rows only: no column and no float is added, and the 2B block above is byte-identical to what
    # it printed before this pass. See LR_LADDER_4B for the pre-registered readout.
    lrbase4, _ = _bfcl(LR_BASE_4B)
    L += ["\\midrule", "\\multicolumn{5}{l}{%s} \\\\" % LR_HDR_4B,
          "untrained base policy & --- & --- & $%.1f$ & --- \\\\" % lrbase4]
    lr4 = {}
    for lab, lr, arm in LR_LADDER_4B:
        a = agg(fams["4B"][0], [arm])
        b, _ = _bfcl(arm)
        if a is None or b is None:
            raise SystemExit("tab:lrsweep: 4B rung %s has no %s cell; a partial ladder is not "
                             "printed" % (arm, "window" if a is None else "BFCL"))
        L.append("%s & %s & $%s$ & $%.1f$ & $%+.1f$ \\\\"
                 % (lab, lr, pp(a["mean"]), b, b - lrbase4))
        lr4[arm] = (100 * a["mean"], b)
        lrlog.append((lab + " (4B)", lr, arm, 100 * a["mean"], b))
    # THE PRE-REGISTERED READOUT, AS A BUILD REFUSAL. The caption and Section 4's transfer
    # paragraph both say that at 4B the uniform control's transfer damage is rate-specific and
    # that the method is above the untrained policy on BFCL at the safe rate; if either stops
    # being true the build fails rather than reprinting the claim.
    if not (lr4["q4bF3e5"][1] > lr4["q4bF"][1]):
        raise SystemExit("tab:lrsweep: the caption asserts that the 4B uniform control RECOVERS "
                         "on BFCL at 3e-5 and the records say otherwise: %s" % lr4)
    if not (lr4["q4bT3e5"][1] > lrbase4 and lr4["q4bT3e5"][1] > lr4["q4bF3e5"][1]):
        raise SystemExit("tab:lrsweep: the caption asserts that at 4B and 3e-5 the method is above "
                         "both the untrained policy (%.2f) and the matched uniform rung, and the "
                         "records say otherwise: %s" % (lrbase4, lr4))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("lrsweep.tex", L)

    # ---- GROUP-SIZE SENSITIVITY AT 2B --------------------------------------------------------
    kb, _ = _bfcl(K_BASE)
    KN, _, _ = NR.load_run(K_BASE)
    L = [GEN,
         "%% 2B anchor n=%d; both rungs are in the same %d-arm 2B Holm family as tab:main"
         % (fams["2B"][1], fams["2B"][2]),
         "% One run per rung at seed offset 500, differing from the method by the group-size token",
         "% alone. k=8 is BATCH-matched, NOT compute-matched -- it draws 8 rollouts per group at",
         "% the same batch size and therefore spends more generation per optimizer step. The",
         "% caveat was registered with the arm, not added after it, and it is why the falsified",
         "% compute-matched 8B k=2 arm is quoted in the caption instead of given a row here.",
         "\\begin{tabular}{lrrr}", "\\toprule",
         " & window & \\textsc{Bfcl} & \\textsc{Nestful} \\\\",
         "group size ($2$B) & mean (pp) & pass (\\%) & win (\\%) \\\\", "\\midrule",
         "\\quad untrained base policy & --- & $%.1f$ & $%.2f$ \\\\" % (kb, NR.rate(KN))]
    klog = []
    for lab, arm in K_LADDER:
        a = agg(fams["2B"][0], [arm])
        b, _ = _bfcl(arm)
        nn, _, _ = NR.load_run(arm)
        L.append("%s & $%s$ & $%.1f$ & $%.2f$ \\\\" % (lab, pp(a["mean"]), b, NR.rate(nn)))
        klog.append((lab, arm, 100 * a["mean"], b, NR.rate(nn)))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("ksens.tex", L)

    # ---- THE MECHANISM DECOMPOSITION ON THE CAP-FREE BENCHMARK -------------------------------
    L = [GEN,
         "% 2B, one run per row at seed offset 500, every checkpoint at step 30.",
         "% NESTFUL has no turn cap and no loop to enter (one completion emits a whole call graph,",
         "% scored by execution), so an arm whose only defect is a lost STOP scores at base here",
         "% while collapsing on BFCL, and an arm with a general capability deficit is down on both.",
         "% The third column is BFCL's OWN recorded error type over the same 800 instances as the",
         "% second -- see bfcl_records.forced_share for why it is deliberately not the outcome",
         "% taxonomy's runaway share nor the per-instance turn-cap flag.",
         "\\begin{tabular}{lrrr}", "\\toprule",
         " & \\textsc{Nestful} & \\textsc{Bfcl} & \\textsc{Bfcl} force- \\\\",
         "configuration ($2$B, step $30$) & win (\\%) & pass (\\%) & terminated (\\%) \\\\",
         "\\midrule"]
    nmlog = []
    for kind, lab, arm in NESTMECH:
        if kind == "@HDR":
            if not L[-1].endswith("\\midrule"):
                L.append("\\midrule")
            L.append("\\multicolumn{4}{l}{%s} \\\\" % lab)
            continue
        nn, _, _ = NR.load_run(arm)
        b, _ = _bfcl(arm)
        f, fn = BR.forced_share(arm)
        if nn is None or b is None or f is None:
            raise SystemExit("tab:nestmech: %s is missing a cell on one of the two benchmarks; a "
                             "row of this table is only readable if all three columns exist" % arm)
        L.append("%s & $%.2f$ & $%.1f$ & $%.1f$ \\\\" % (lab, NR.rate(nn), b, f))
        nmlog.append((lab, arm, NR.rate(nn), b, f))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("nestmech.tex", L)

    # ---- THE 4B BFCL TRANSFER GRID, APPENDIX -------------------------------------------------
    # Emitted, not transcribed: tab:bfcl's 8B rows predate this module and are a hand-verified
    # transcription, which is why they stay where they are and this block is generated beside them
    # rather than merged into them. See BFCL4B for what it shows and why every row is \ddagger.
    RB, _, _, _ = BR.load_run(BFCL4B_BASE)
    if RB is None:
        raise SystemExit("tab:bfcl4b: no %s records" % BFCL4B_BASE)
    refs = []
    for a in BFCL4B_REF:
        s, _, _, _ = BR.load_run(a)
        if s is None:
            raise SystemExit("tab:bfcl4b: reference arm %s has no records" % a)
        refs.append((a, s))
    L = [GEN,
         "% The 4B half of the checkpoint-level transfer evidence. tab:bfcl is the 8B half and is",
         "% a hand-verified transcription; this block is generated from the records by",
         "% surface/verl_rl/bfcl_records.py, n=800 tasks per arm, one sample each, task index",
         "% sha256 cd489490 CHECKED identical across arms rather than assumed.",
         "% Rows sorted by overall pass rate; the untrained base policy is banded by rules, so an",
         "% arm above the band improves held-out tool use and an arm below it does not.",
         "% LAST COLUMN: the paired difference of the METHOD minus this row, meaned over BOTH",
         "% method seeds, with the WEAKEST of the two exact p values -- never the more favourable",
         "% pairing. On the method's OWN two rows the self-pairing is dropped, so those two cells",
         "% are the single seed-against-seed contrast and are the size of the method's own seed",
         "% spread on this benchmark (3.0 pp), which is why they are worth reading first.",
         "% Every contrast here belongs to NO Holm family in this paper (the merged",
         "% 19-contrast family is 8B and was fixed before these cells existed), so every row",
         "% carries \\ddagger and every p is RAW and UNCORRECTED. No claim rests on one of them.",
         "\\begin{tabular}{lrrrrrl}", "\\toprule",
         " & & \\multicolumn{4}{c}{by category, $200$ tasks each} & \\methodname{} $-$ row \\\\",
         "\\cmidrule(lr){3-6}",
         "checkpoint ($4$B, step $%d$) & overall & \\texttt{base} & \\texttt{long\\_ctx} & "
         "\\texttt{miss\\_func} & \\texttt{miss\\_param} & (pp), weakest $p$ \\\\"
         % BFCL4B_STEP, "\\midrule"]

    def _cat(c):
        out = []
        for k in BFCL_CATS:
            v = c.get(k)
            out.append("--" if not v or not v[1] else "$%.1f$" % (100.0 * v[0] / v[1]))
        return " & ".join(out)

    rows4b = []
    for lab, arm in BFCL4B:
        s, c, st, _ = BR.load_run(arm)
        if s is None:
            raise SystemExit("tab:bfcl4b: %s has no records" % arm)
        # The step is in the header, once, so it must be the same in every row -- checked here
        # rather than trusted, because a header that names a step no row was scored at is worse
        # than a step column.
        if st is not None and int(st) != BFCL4B_STEP:
            raise SystemExit("tab:bfcl4b: %s was scored at step %s but this table's header names "
                             "step %d for every row" % (arm, st, BFCL4B_STEP))
        ds = [BR.paired(rs, s) for ra, rs in refs if ra != arm]
        rows4b.append((lab, arm, BR.rate(s), c, ds))
    rows4b.sort(key=lambda r: (-r[2], r[1]))
    banded = False
    bl = []
    for lab, arm, r, c, ds in rows4b:
        if not banded and r < BR.rate(RB):
            bl += ["\\midrule",
                   "base policy, untrained & $%.1f$ & %s & --- \\\\"
                   % (BR.rate(RB), _cat(BR.load_run(BFCL4B_BASE)[1])),
                   "\\midrule"]
            banded = True
        d = sum(x[0] for x in ds) / len(ds)
        pw = max(x[2] for x in ds)
        bl.append("%s & $%.1f$ & %s & $%+.2f^{\\ddagger}$ \\ $p\\le%s$ \\\\"
                  % (lab, r, _cat(c), d, _psci(pw)))
    if not banded:
        raise SystemExit("tab:bfcl4b: no arm scored below the untrained 4B policy -- the banding "
                         "rule this table is read by would print no band at all")
    L += bl + ["\\bottomrule", "\\end{tabular}"]
    W("bfcl4b.tex", L)

    # ---- THE 2B BFCL TRANSFER GRID, APPENDIX -- READOUT OF A PRE-REGISTERED PREDICTION -------
    # See BFCL2B for the registration (PLAN_TRIAGE.md 2026-08-24 02:10) and for why this table
    # carries a force-terminated column that tab:bfcl4b does not.
    # ALL FOUR OR NONE, AND THE REFUSAL IS LOUD RATHER THAN SILENT. A half-filled readout of a
    # registered prediction is the selective reporting the registration exists to prevent, so the
    # emitter declines to write the file and says so on stderr and in the stdout audit -- it does
    # NOT abort, because the other eighteen tables have nothing to do with this one and a
    # straggler must not be able to block a rebuild.
    have = [a for a in BFCL2B_REGISTERED if BR.load_run(a)[0] is not None]
    if have and len(have) < len(BFCL2B_REGISTERED):
        msg = ("tab:bfcl2b NOT WRITTEN: %d of %d registered arms have records (%s). A registered "
               "prediction is read out on all its cells or on none; a half-filled grid is the "
               "selective readout the registration exists to prevent."
               % (len(have), len(BFCL2B_REGISTERED), ", ".join(sorted(have))))
        print("[paper_numbers] " + msg, file=sys.stderr)
        print("  -- " + msg)
        have = []
    if have:
        RB2, _, _, _ = BR.load_run(BFCL2B_BASE)
        if RB2 is None:
            raise SystemExit("tab:bfcl2b: no %s records" % BFCL2B_BASE)
        refs2 = []
        for a in BFCL2B_REF:
            s, _, _, _ = BR.load_run(a)
            if s is None:
                raise SystemExit("tab:bfcl2b: reference arm %s has no records" % a)
            refs2.append((a, s))
        L = [GEN,
             "% The 2B half of the checkpoint-level transfer evidence, and the READOUT OF A",
             "% PRE-REGISTERED PREDICTION: PLAN_TRIAGE.md 2026-08-24 02:10 fixed, before any of",
             "% the four published-baseline cells ran, that each should land BELOW 10 pass WITH a",
             "% HIGH force-terminated share, because none of them trains with the warm bank and at",
             "% 2B the bank is what prevents termination collapse. Both branches were declared",
             "% publishable in advance. The outcome is in these rows and in Section 9.6.",
             "% Generated from the records by surface/verl_rl/bfcl_records.py, n=800 tasks per arm,",
             "% one sample each, task index sha256 cd489490 CHECKED identical across arms.",
             "% THE FORCE-TERMINATED COLUMN is BFCL's own recorded `multi_turn:force_terminated`",
             "% error type over the same 800 instances. It is here because the registered",
             "% prediction is a CONJUNCTION and a pass-rate-only table could not be checked against",
             "% it. It is the NARROWEST of this project's three termination statistics.",
             "% Rows sorted by overall pass rate; the untrained base policy is banded by rules.",
             "% LAST COLUMN: the paired difference of the METHOD minus this row, meaned over BOTH",
             "% method seeds, with the WEAKEST of the two exact p values -- never the more",
             "% favourable pairing. On the method's own two rows the self-pairing is dropped.",
             "% Every contrast here belongs to NO Holm family in this paper (the merged",
             "% 19-contrast family is 8B and was fixed before these cells existed), so every row",
             "% carries \\ddagger and every p is RAW and UNCORRECTED. Section 9.1's coverage note",
             "% states which contrasts in this paper are corrected and which are not.",
             "\\begin{tabular}{lrrrl}", "\\toprule",
             " & overall & force- & & \\methodname{} $-$ row \\\\",
             "checkpoint ($2$B, step $%d$) & pass (\\%%) & term.\\ (\\%%) & & (pp), weakest $p$ \\\\"
             % BFCL2B_STEP, "\\midrule"]
        rows2b = []
        for lab, arm in BFCL2B:
            if arm is NOT_RUN:
                rows2b.append((lab, arm, None, None, None))
                continue
            s, _, st, _ = BR.load_run(arm)
            if s is None:
                raise SystemExit("tab:bfcl2b: %s has no records" % arm)
            if st is not None and int(st) != BFCL2B_STEP:
                raise SystemExit("tab:bfcl2b: %s was scored at step %s but this table's header "
                                 "names step %d for every row" % (arm, st, BFCL2B_STEP))
            ds = [BR.paired(rs, s) for ra, rs in refs2 if ra != arm]
            rows2b.append((lab, arm, BR.rate(s), BR.forced_share(arm)[0], ds))
        # NOT_RUN rows sort to the bottom of the table, under the band, because an unmeasured arm
        # has no pass rate to place it by and must never be placed by a guess.
        rows2b.sort(key=lambda r: (r[2] is None, -(r[2] or 0.0), r[1]))
        banded2 = False
        bl2 = []
        bf, _ = BR.forced_share(BFCL2B_BASE)
        for lab, arm, r, f, ds in rows2b:
            if arm is NOT_RUN:
                bl2.append("%s & \\multicolumn{4}{c}{\\emph{not run at this scale: coverage, "
                           "not a null}} \\\\" % lab)
                continue
            if not banded2 and r < BR.rate(RB2):
                bl2 += ["\\midrule",
                        "base policy, untrained & $%.1f$ & $%.1f$ & & --- \\\\"
                        % (BR.rate(RB2), bf), "\\midrule"]
                banded2 = True
            d = sum(x[0] for x in ds) / len(ds)
            pw = max(x[2] for x in ds)
            bl2.append("%s & $%.1f$ & $%.1f$ & & $%+.2f^{\\ddagger}$ \\ $p\\le%s$ \\\\"
                       % (lab, r, f, d, _psci(pw)))
        if not banded2:
            raise SystemExit("tab:bfcl2b: no arm scored below the untrained 2B policy -- the "
                             "banding rule this table is read by would print no band at all")
        L += bl2 + ["\\bottomrule", "\\end{tabular}"]
        W("bfcl2b.tex", L)
        print("  -- tab:bfcl2b, REGISTERED READOUT (predicted: all four baselines <10 pass with "
              "high force-terminated) --")
        print("     base2b %.2f (forced %.1f%%);  TRIAGE %s"
              % (BR.rate(RB2), bf, ["%.2f" % BR.rate(s) for _, s in refs2]))
        for lab, arm, r, f, ds in rows2b:
            if arm is NOT_RUN:
                continue
            mark = ""
            if arm in BFCL2B_REGISTERED:
                mark = "  <== REGISTERED: %s" % ("IN the collapsed band" if r < 10 else
                                                 "NOT in the collapsed band (>=10)")
            print("     %-6s %6.2f pass  %5.1f%% force-terminated%s" % (arm, r, f, mark))

    # ---- NESTFUL SEED REPLICATION AT ALL THREE SCALES, APPENDIX (internal review W12) --------
    # POST-REGISTRATION. transfer.tex still prints exactly the contrast registered before any cell
    # was scored; these cells are extra and are labelled as extra. Scale-blocked 2026-08-24 when
    # the 2B and 4B replicates landed and closed W12 -- see NESTREP for what they cost.
    L = [GEN,
         "% POST-REGISTRATION robustness readout, all three scales. The pre-registered m=3 scale",
         "% family of transfer.tex is untouched by this table and is NOT recomputed from it: the",
         "% registered cell stays the registered cell whichever way its replicate falls.",
         "% n=1861 tasks per checkpoint, one sample each; step 30 at 2B/4B and step 15 at 8B (the",
         "% only step ckpt_a8T holds), the same disclosed asymmetry tab:transfer carries.",
         "% THE TWO SIDES ARE THE SAME KIND OF REPLICATE AT 2B AND 4B (seed pairs, offsets 500 and",
         "% 1500 on both sides) AND ARE NOT AT 8B, where the method side is three CONFIGURATIONS of",
         "% this paper's own ladder at ONE offset. Each block header says which it is; the two axes",
         "% are never merged.",
         "% The difference column is 'vs base' in the per-checkpoint blocks and 'Triage - uniform'",
         "% in the contrast block; each block's own header names it, so the column header is the",
         "% neutral word.",
         "\\begin{tabular}{lrrl}", "\\toprule",
         " & win & difference & exact \\\\",
         "checkpoint (\\textsc{Nestful}, $n=1{,}861$) & (\\%) & (pp) & $p$ \\\\"]
    nrlog, prlog = [], []
    for scale, basetag, step, (mhdr, mrows), (chdr, crows) in NESTREP:
        Bx, _, _ = NR.load_run(basetag)
        if Bx is None:
            raise SystemExit("tab:nestrep: %s has no base cell (%s)" % (scale, basetag))
        L += ["\\midrule",
              "\\multicolumn{4}{l}{\\textbf{Qwen3-VL-%s}, step $%d$, $n=%d$} \\\\"
              % (scale, step, len(Bx)),
              "\\midrule",
              "untrained base policy & $%.2f$ & --- & --- \\\\" % NR.rate(Bx)]
        for hdr, group in ((mhdr, mrows), (chdr, crows)):
            L.append("\\multicolumn{4}{l}{%s: \\emph{difference vs base}} \\\\" % hdr)
            for lab, arm in group:
                s, st, _ = NR.load_run(arm)
                if s is None:
                    raise SystemExit("tab:nestrep: %s has no NESTFUL cell" % arm)
                if st is not None and int(st) != step:
                    raise SystemExit("tab:nestrep: %s is scored at step %s, not the %s block's "
                                     "step %d -- a replication read at a different checkpoint "
                                     "than the cell it replicates is not a replication"
                                     % (arm, st, scale, step))
                d, p, n, _, _ = NR.paired(s, Bx)
                L.append("\\quad %s & $%.2f$ & $%+.2f$ & $%s$ \\\\"
                         % (lab, NR.rate(s), d, _psci(p)))
                nrlog.append((scale, lab, arm, NR.rate(s), d, p))
        # Every method x control pairing of this block, never a chosen one.
        pairs = []
        for tl, ta in mrows:
            for ul, ua in crows:
                A, _, _ = NR.load_run(ta)
                U, _, _ = NR.load_run(ua)
                d, p, n, _, _ = NR.paired(A, U)
                pairs.append((tl, ul, ta, ua, d, p))
        L.append("\\multicolumn{4}{l}{\\emph{All %d method $\\times$ control pairings} "
                 "(\\methodname{} $-$ uniform)} \\\\" % len(pairs))
        for tl, ul, ta, ua, d, p in pairs:
            L.append("\\quad %s $-$ %s & --- & $%+.2f$ & $%s$ \\\\"
                     % (tl, ul.replace("uniform \\textsc{Grpo}, ", ""), d, _psci(p)))
            prlog.append((scale, ta, ua, d, p))
        # The range goes in the DIFFERENCE cell, not in the p cell: this table's last column is
        # one exact p and a range is not one, and a mean over pairings has no p of its own.
        pm = sum(x[4] for x in pairs) / len(pairs)
        L.append("\\quad \\emph{mean over pairings} & --- & $%+.2f$ \\ "
                 "{\\small[$%+.2f,%+.2f$]} & --- \\\\"
                 % (pm, min(x[4] for x in pairs), max(x[4] for x in pairs)))
    L += ["\\bottomrule", "\\end{tabular}"]
    W("nestrep.tex", L)

    print("wrote rltable, mainsteps, negatives, perseed, coadapt, coadapt2b, poolbig, "
          "coadapt2x2, scaletable, durability, transfer, lrsweep, ksens, nestmech, bfcl4b, "
          "nestrep to %s" % outdir)
    print("  (bfcl4b.tex is the 4B half of the checkpoint-level transfer evidence, 2026-08-24)")
    print("  (transfer.tex REPLACES bfclscale.tex + nestful.tex, merged 2026-08-18)")
    print("  (mainsteps.tex carries tab:main's former four per-step columns, 2026-08-18)")
    print("  8B anchor n=%d, Holm family %d arms" % (nbase, nfam))
    # WHY tab:main HAS NO TRANSFER COLUMN. The column was added earlier on 2026-08-18 and removed
    # the same day under the PI's Table 3 directive, branch (b). The block below is the evidence
    # the decision was made on and is printed so the decision stays auditable: with the sharpened
    # variant row removed, TRIAGE's three-seed BFCL mean is still exceeded by two GENUINE published
    # baselines, and no honest fix was available -- the alternative summary metrics the directive
    # named (val-solve gain, the live-groups efficiency statistic) are NOT well-defined for every
    # row of this table, so substituting one would have blanked the very baselines that lead. BFCL
    # therefore goes back to the dedicated transfer table in full; nothing is deleted.
    # tab:main's caption obligations, printed so a caption clause that disagrees is provably wrong.
    print("  -- tab:main per-checkpoint block (all at step %d, the one step every row has) --"
          % MAIN_STEP)
    def _plain(lab):
        nm = re.sub(r"\s*\\citep\{[^}]*\}", "", lab)
        return (nm.replace("\\quad ", "").replace("\\textsc{", "").replace("}", "")
                  .replace("\\emph{", "").replace("\\", "")).strip()

    print("     %-4s %-34s %7s %8s %7s %7s %8s %6s %-9s"
          % ("mdl", "row", "n", "solve%", "+-CI", "turns", "window", "of n", "seed shown"))
    for model, lab, m, a in mainlog:
        print("     %-4s %-34s %7d %8.2f %8.2f %7.2f %+8.2f %6d %-9s"
              % (model, _plain(lab), m["n"], m["rate"], m["ci"], m["turns"],
                 100 * a["mean"], a["n"], a["sel"]))
    # THE SELECTION, PRINTED. Every cell above that is a best-of-n is listed here beside the seed
    # mean it replaced and beside every candidate, so the convention can be audited from stdout
    # without opening a .tex file. A row with n=1 is not a selection and is not listed.
    print("  -- BEST-SEED SELECTION (PI decision 2026-08-28; symmetric, every multi-seed arm) --")
    print("     %-4s %-42s %-9s %8s %10s %9s %s"
          % ("mdl", "configuration", "selected", "window", "Holm q", "seed mean", "candidates"))
    for model, lab, sel, sd_lab, wmean, q, smean, cands in pslog:
        f = fams[model][0]
        print("     %-4s %-42s %-9s %+8.2f %10s %+9.2f %s"
              % (model, _plain(lab), "%s (s%s)" % (sel, sd_lab), wmean,
                 ("<0.001" if q < 0.0005 else "%.3f" % q), smean,
                 " ".join("%s %+.2f" % (a, 100 * f[a]["mean"]) for a in cands)))
    print("     tab:transfermain BFCL selection (same window rule, never a transfer maximum):")
    for s in MAIN_SCALES:
        print("       %-4s TRIAGE %-8s %5.2f   uniform %-8s %5.2f   difference %+6.2f pp"
              % (s, trisel[s], tri_b[s], unisel[s], uni_b[s], tri_b[s] - uni_b[s]))
    print("     Each selected seed carries ITS OWN Holm q -- every seed is already its own member")
    print("     of the pinned per-scale family, so no q was recomputed and no m moved.")
    for model in MAIN_SCALES:
        n, r, t = anchors[model]
        print("     %-4s %-34s %7d %8.2f %8s %7.2f" % (model, "ANCHOR (untrained, full cell)",
                                                       n, r, "--", t))
    print("     CI = 1.96*sqrt(p(1-p)/n) on the pooled paired cells; NOT seed spread.")
    # THE INVALID TOOL-CALL COLUMN, DECOMPOSED. That column is a RATIO whose denominator is the
    # attempted calls, so a row can regress on it while emitting fewer bad calls per task than the
    # reference. Section 7.2 states that at 2B, and the counts it states are printed here rather
    # than derived in prose.
    print("  -- tab:main invalid tool-call column: numerator and denominator per task --")
    print("     %-4s %-22s %9s %9s %9s %9s" % ("mdl", "row", "att/task", "bad/task",
                                               "rate %", "turns"))
    for model in MAIN_SCALES:
        sub = {k: bases[model][k] for k in sharedidx[model]}
        for lab, arm, steps in (("untrained base", BASE_CELLS[model], [None]),
                                ("method row", None, None)):
            if arm is None:
                a = agg(fams[model][0], METHOD_SPEC.get(model), best=True)
                arm = a["sel"]
                # The method row's printed rate is read at ONE checkpoint since the 2026-09-02
                # ruling, so its decomposition is read there too: Section 7.2 quotes these two
                # counts as the explanation of the printed ratio, and an explanation read over a
                # window would not decompose the number beside it.
                if T1_METHOD_BEST_STEP and model in t1step:
                    lab = "method row @%d" % t1step[model]
                    steps = [t1step[model]]
            inv = att = tsum = 0.0
            tn = 0
            for st in (WINDOW if steps is None else steps):
                cur = cell_records(cell_path(arm, st) if st is not None else arm)
                # THE INDEX IS THE ONE EACH ROW IS PRINTED ON, and it is keyed off the row and
                # not off `steps`: the anchor is read on the table's shared index (it has no
                # checkpoint), the method row on its own pairs, exactly as tab:main reads them.
                keys = sorted(set(cur) & set(sub if arm is BASE_CELLS[model]
                                             else bases[model]))
                if len(keys) < 300:
                    continue
                for k in keys:
                    d = cur[k]
                    pf = float(d.get("n_parse_fail") or 0)
                    un = float(d.get("n_unknown_tool") or 0)
                    bc = float(d.get("n_backend_calls") or 0)
                    inv += pf + un
                    att += pf + un + bc
                    t = d.get("n_turns")
                    if t is not None:
                        tsum += float(t)
                        tn += 1
            print("     %-4s %-22s %9.2f %9.2f %9.2f %9.2f"
                  % (model, lab, att / tn, inv / tn, 100.0 * inv / att, tsum / tn))
    print("     4B coverage gap: VIP and TSCL have no 4B arm with a window (allocation expired).")
    print("  -- BFCL per tab:main row (NOT a column any more; see tab:transfer/tab:bfcl) --")
    for kind, lab, spec in MAIN:
        if kind == "@HDR":
            continue
        for model in MAIN_SCALES:
            arms = spec.get(model)
            if arms is SAME_FIXED:
                print("     %-4s %-34s = FIXED-RHO AT THIS SCALE" % (model, _plain(lab)))
                continue
            if arms is NOT_RUN:
                print("     %-4s %-34s NOT RUN AT THIS SCALE" % (model, _plain(lab)))
                continue
            got = bfcl_audit(arms)
            mean = (" -> mean %.2f" % (sum(r for _, r, _ in got) / len(got))) if got else ""
            print("     %-4s %-34s %s%s" % (model, _plain(lab),
                                            ", ".join("%s@%s %.2f" % (a, st, r)
                                                      for a, r, st in got) or "none", mean))
    for tag, lbl in (("base", "8B"), ("base4b", "4B"), ("base2b", "2B")):
        bg = bfcl_audit([tag])
        print("     %-4s %-34s %s" % (lbl, "untrained base policy",
                                      ", ".join("%s@%s %.2f" % (a, st, r) for a, r, st in bg)))
    print("  -- tab:lrsweep (reviewer item 3 / W5), 2B, one run per rung at off=500 --")
    for lab, lr, arm, w, b in lrlog:
        print("     %-18s LR %-12s %-9s window %+6.2f  BFCL %5.2f" % (_plain(lab), lr, arm, w, b))
    for k, arms in LR_CONTEXT.items():
        print("     context: %-28s %s" % (k, ", ".join(
            "%s %s" % (x, ("%.2f" % BR.rate(BR.load_run(x)[0])) if BR.load_run(x)[0] else "no BFCL")
            for x in arms)))
    print("     NOT RUN, disclosed rather than filled: TRIAGE at 3e-5.")
    print("  -- tab:ksens (2B, batch-matched not compute-matched) --")
    for lab, arm, w, b, nn in klog:
        print("     %-22s %-9s window %+6.2f  BFCL %5.2f  NESTFUL %5.2f"
              % (_plain(lab), arm, w, b, nn))
    print("  -- tab:nestmech (the termination-specificity decomposition, 2B step 30) --")
    for lab, arm, nn, b, f in nmlog:
        print("     %-38s %-9s NESTFUL %5.2f  BFCL %5.2f  forced %5.1f%%"
              % (_plain(lab), arm, nn, b, f))
    print("  -- tab:nestrep (POST-REGISTRATION NESTFUL replication, all scales, W12) --")
    for scale, lab, arm, r, d, p in nrlog:
        print("     %-4s %-38s %-7s %5.2f  vs base %+5.2f (p=%.3g)"
              % (scale, _plain(lab), arm, r, d, p))
    for scale, basetag, step, (_, mrows), (_, crows) in NESTREP:
        mset, cset = [a for _, a in mrows], [a for _, a in crows]
        tri = [r for sc, _, arm, r, _, _ in nrlog if sc == scale and arm in mset]
        uni = [r for sc, _, arm, r, _, _ in nrlog if sc == scale and arm in cset]
        ds = [d for sc, _, _, d, _ in prlog if sc == scale]
        print("     %-4s method mean %.2f (spread %.2f) vs control mean %.2f (spread %.2f); "
              "base %.2f" % (scale, sum(tri) / len(tri), max(tri) - min(tri),
                             sum(uni) / len(uni), max(uni) - min(uni),
                             NR.rate(NR.load_run(basetag)[0])))
        print("     %-4s %d pairings: %s  -> mean %+0.2f"
              % (scale, len(ds), ", ".join("%+0.2f" % d for d in ds), sum(ds) / len(ds)))
    for m in SCALES3:
        print("  %s anchor n=%d, family %d" % (m, fams[m][1], fams[m][2]))
    for r in figrows:
        print("  transfer %s: base %.2f  TRIAGE %.2f (%d seeds)  uniform %.2f (%d seeds)  "
              "gap %+.2f pp" % (r["scale"], r["base"], r["tri_mean"], len(r["tri"]),
                                r["uni_mean"], len(r["uni"]), r["tu_mean"]))
    # SEED COUNTS FOR THE CAPTIONS. No float prints these any more, so they are emitted here:
    # a caption clause that disagrees with this block is a caption clause that is wrong.
    print("  -- Figure 4's window and the seed spread its caption must carry --")
    for scale, lab, k, lo, hi, spread in vlog:
        nm = lab.replace("\\methodname{}", "TRIAGE").replace("\\textsc{Grpo}", "GRPO")
        print("     %-4s %-22s %d seed(s)  steps %d-%d  widest seed range %s"
              % (scale, nm, k, lo, hi,
                 "n/a (single seed)" if k < 2 else "%.2f pp" % spread))
    print("  -- termination secondary outcome (seed means) --")
    for scale, lab, k, m in termlog:
        nm = lab.replace("\\methodname{}", "TRIAGE").replace("\\textsc{Grpo}", "GRPO")
        print("     %-4s %-22s %d seed(s)  non-term %.1f -> %.1f (%+.1f)   solve %.1f -> %.1f"
              % (scale, nm, k, m[0], m[1], m[1] - m[0], m[2], m[3]))
    print("  -- seed counts the captions must carry --")
    for lab in sorted({k[0] for k in SEEDCOUNTS}):
        print("     %-46s %s" % (lab.replace("\\quad ", "").replace("\\methodname{}", "TRIAGE"),
                                 "/".join(str(SEEDCOUNTS.get((lab, m), 0)) for m in SCALES3)))
    for name, arms in (("tab:main TRIAGE", FULL_METHOD),
                       ("tab:ladder sharpened (left tab:main 2026-08-18)", ["a8T5k", "a8T5kr"]),
                       ("tab:main uniform", ["a8F", "a8Fr"]),
                       ("tab:ladder band filter", ["a8T2n", "a8T2nr"]),
                       ("tab:ladder v2 gamma=0.9", ["a8T2", "a8T2r"])):
        a = agg(fam, arms)
        print("     %-46s %d" % (name, 0 if a is None else a["n"]))


def _psci(p):
    """A p-value as the paper sets them: a plain decimal where that reads, else m x 10^{-k}.
    Rounded UP in the last digit so the printed bound is never weaker than the measurement."""
    if p >= 0.001:
        return "%.3f" % p
    e = 0
    while p < 1 and e < 320:
        p *= 10
        e += 1
    m = math.ceil(p * 10) / 10.0
    if m >= 10:
        m, e = m / 10.0, e - 1
    return "%.1f\\times10^{-%d}" % (m, e)


SECTIONS = {"DIAG": sec_diag_cert_rho, "CERT": sec_diag_cert_rho, "RHO": sec_diag_cert_rho,
            "CALIB": sec_calib, "SEED": sec_seed, "H2H": sec_h2h, "EST": sec_est}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="DIAG|CALIB|SEED|H2H|EST")
    ap.add_argument("--emit-tables", default="", metavar="DIR",
                    help="write the paper's three TRIAGE tables into DIR instead of printing")
    # 2026-08-28: the main-text step-curve figure is drawn from the SAME agg() calls the tables
    # are emitted from (see paper_figures.py's header), so it lives behind a flag on this
    # module rather than in make_figures.py, which reads the receipts and not these cells.
    # matplotlib is imported only inside emit_figures, so --emit-tables never needs it.
    ap.add_argument("--emit-figures", default="", metavar="DIR",
                    help="write the main-text figures drawn from these cells into DIR")
    a = ap.parse_args()
    if a.emit_tables:
        emit_tables(a.emit_tables)
        return 0
    if a.emit_figures:
        import paper_figures
        paper_figures.emit_figures(a.emit_figures)
        return 0
    order = ["DIAG", "CALIB", "SEED", "H2H", "EST"]
    for name in ([a.only.upper()] if a.only else order):
        SECTIONS[name]()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
