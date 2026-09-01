"""Second out-of-distribution transfer benchmark: NESTFUL (IBM Research, arXiv:2409.03797),
read the same way bfcl_records.py reads BFCL. One definition, shared by the table emitter.

WHY A SEPARATE MODULE RATHER THAN A FLAG ON bfcl_records.py. The two benchmarks share nothing but
the paired-statistic convention: different root, different scoring field (`win_rate`, executable
end-to-end check, not BFCL's state-diff `reward` -- though the two agree bit for bit on every
record checked, `reward == int(win_rate > 0)` for all 9 x 1861 rows), and NESTFUL has exactly one
scored checkpoint per (scale, arm) rather than BFCL's multi-seed design. Folding that into
bfcl_records.py would either grow a benchmark-specific branch into a shared function or silently
assume seed structure NESTFUL does not have.

WHAT A RUN IS. One directory per scored checkpoint, one JSON object per task attempt:

    <NESTFUL>/run_<arm>/records.jsonl   {"task_id", "sample", "reward", "win_rate", ...}

All 1861 tasks, one sample per task, byte-identical task set across arms (checked here via
`nestful_commit`, not assumed). PRE-REGISTERED READOUT (PLAN_TRIAGE.md 2026-08-18 02:05 / 08:25):
win rate is the primary metric, paired exact McNemar per scale, Holm across the m=3-scale family.

TWO DOCUMENTED, ARM-INDEPENDENT PROMPT DEVIATIONS from the benchmark's released Llama branch,
applied identically to all nine cells and stated again wherever this table is presented: (1) the
function spec is embedded as JSON rather than as the double-encoded escaped string the released
Llama branch produces; (2) before the benchmark's own parser sees a completion, the first BALANCED
top-level `[...]` span is extracted, because instruct models fence their JSON. Consequence: these
numbers are NOT digit-comparable with the benchmark paper's own Llama rows, and are not meant to
be -- this cell is a WITHIN-CELL paired contrast between our own arms, nothing else.
"""

from __future__ import annotations

import json
import math
import os

NESTFUL_ROOT = os.path.join(os.environ.get("BRACE_WORK", "work"), "nestful")
NESTFUL_COMMIT = "fc2c4123e73500a56185a5fb354f05d1c8b4890c"
EXPECTED_N = 1861


def load_run(arm, root=NESTFUL_ROOT, require_commit=True, step=None):
    """(solve dict, step, commit) for one scored checkpoint, or (None, None, None) if absent.

    solve: {(task_id, sample): 0|1} on the win-rate (executable) score, first occurrence wins --
    the same torn-write convention every other cell loader in this paper uses.

    `step` SELECTS ONE CHECKPOINT OUT OF A RUN DIRECTORY THAT HOLDS MORE THAN ONE, and it exists
    because of a hazard this loader used to have rather than because of a feature anyone wanted.
    Until 2026-08-29 every run_*/records.jsonl in the archive carried exactly one step (verified
    dir by dir: a second step of the same arm was always written to its own directory, which is
    what run_a8Tpe_s15, run_a8Tr_s15 and run_t8Tf_s15 are). The de-duplication key is
    (task_id, sample) and carries NO step, so a file holding two steps would have been silently
    reduced to whichever step's row for a task happened to be written first -- a cell that is
    neither step, reported under one step's subscript. The review P0-5 re-measurement writes
    a8T3g2 at step 30 and at step 15 into ONE directory, so:
      * with `step` given, only that step's rows are read, and the returned step is that step;
      * with `step` omitted and the file carrying more than one step, this REFUSES rather than
        picking one, because picking one silently is the failure above.
    Single-step directories are unaffected in every particular, which is checked by re-emitting
    every table and diffing (zero drift, 2026-08-29 17:40).
    """
    p = os.path.join(root, "run_%s" % arm, "records.jsonl")
    if not os.path.exists(p):
        return None, None, None
    solve, steps, commits = {}, set(), set()
    for line in open(p, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        k = (d.get("task_id"), d.get("sample"))
        if None in k or k in solve:
            continue
        if step is not None and _as_step(d.get("step")) != _as_step(step):
            continue
        try:
            s = int(float(d["win_rate"]) > 0)
        except Exception:
            continue
        solve[k] = s
        if d.get("step") is not None:
            steps.add(d["step"])
        if d.get("nestful_commit"):
            commits.add(d["nestful_commit"])
    if not solve:
        # A step that was ASKED FOR and is not in the file is a mistake in the caller, not an
        # absent arm: _nest_cell turns (None, None, None) into a coverage dash, so without this
        # a mistyped step would print "not measured" over a directory full of measurements.
        if step is not None and os.path.getsize(p) > 0:
            raise SystemExit("%s: no record carries step %s (the file is not empty) -- a dash "
                             "here would claim a measurement is missing that is not" % (arm, step))
        return None, None, None
    commit = sorted(commits)[0] if commits else None
    if require_commit and commit and commit != NESTFUL_COMMIT:
        raise SystemExit("%s was scored against nestful commit %s, not %s -- not comparable"
                         % (arm, commit[:8], NESTFUL_COMMIT[:8]))
    if step is None and len(steps) > 1:
        raise SystemExit("%s holds %d steps in one records.jsonl (%s) and no step was asked for "
                         "-- the (task_id, sample) key carries no step, so reading it unfiltered "
                         "would mix them; pass step=" % (arm, len(steps), sorted(steps)))
    if step is not None:
        got = sorted(steps)
        if got and got != [_as_step(step)]:
            raise SystemExit("%s: step=%s selected rows carrying steps %s"
                             % (arm, step, got))
        return solve, _as_step(step), commit
    step = sorted(steps)[0] if len(steps) == 1 else (sorted(steps) or [None])[0]
    return solve, step, commit


def _as_step(v):
    """Steps are written as ints by every scorer, but compare on a number, not on a repr."""
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return v


def rate(solve):
    """Win rate in percent."""
    return 100.0 * sum(solve.values()) / len(solve)


def mcnemar_exact(b, c):
    """Exact two-sided conditional test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return min(1.0, 2 * tail)


def paired(A, B):
    """A minus B on the tasks both answered: (diff_pp, p, n, b, c)."""
    keys = sorted(set(A) & set(B))
    n = len(keys)
    if not n:
        return None
    b = sum(1 for k in keys if B[k] and not A[k])
    c = sum(1 for k in keys if A[k] and not B[k])
    d = 100.0 * (c - b) / n
    return d, mcnemar_exact(b, c), n, b, c


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, run = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


# One checkpoint per (scale, role): TRIAGE and uniform steps are fixed exactly as pre-registered --
# 2B/4B at step 30 (the primary matched-window step), 8B at step 15 (the only step ckpt_a8T holds),
# disclosed rather than hidden. No selection on this benchmark.
SCALES = [("2B", "base2b", "q2bT", "q2bF5", 30),
          ("4B", "base4b", "q4bT", "q4bF", 30),
          ("8B", "base8b", "a8T", "a8F", 15)]


def scale_block(root=NESTFUL_ROOT):
    """Everything the NESTFUL transfer table needs, per scale, plus the pre-registered Holm test.

    Unlike BFCL, each scale here is ONE checkpoint per role (no seed aggregation): the cells were
    registered and run once each (PLAN_TRIAGE.md 2026-08-18 08:25). The m=3 Holm family is exactly
    the three TRIAGE-vs-uniform contrasts, pre-registered before any cell was scored.
    """
    out = []
    pvals = []
    for name, basetag, tri, uni, step in SCALES:
        B, _, _ = load_run(basetag, root)
        T, tstep, _ = load_run(tri, root)
        U, ustep, _ = load_run(uni, root)
        if B is None or T is None or U is None:
            continue
        tu = paired(T, U)
        tb = paired(T, B)
        ub = paired(U, B)
        row = dict(scale=name, base=rate(B), n=len(B), step=step,
                   tri_arm=tri, uni_arm=uni, tri=rate(T), uni=rate(U),
                   tu_diff=tu[0], tu_p=tu[1], tu_n=tu[2],
                   tb_diff=tb[0], tb_p=tb[1], ub_diff=ub[0], ub_p=ub[1])
        out.append(row)
        pvals.append(tu[1])
    qs = holm(pvals) if pvals else []
    for r, q in zip(out, qs):
        r["tu_q"] = q
    return out


if __name__ == "__main__":
    for r in scale_block():
        print("%s (step %d)  base %.2f  TRIAGE(%s) %.2f  uniform(%s) %.2f  n=%d"
              % (r["scale"], r["step"], r["base"], r["tri_arm"], r["tri"], r["uni_arm"], r["uni"],
                 r["n"]))
        print("   T-U %+.2f pp  p=%.3g  q(Holm,m=3)=%.3g   (T-base %+.2f p=%.3g, U-base %+.2f p=%.3g)"
              % (r["tu_diff"], r["tu_p"], r["tu_q"], r["tb_diff"], r["tb_p"], r["ub_diff"], r["ub_p"]))
