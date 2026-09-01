"""One definition of how a BFCL transfer run is read, shared by the table emitter and the figure.

WHY THIS IS ITS OWN MODULE. Two consumers need the same numbers: paper_numbers.py writes the
scale-transfer table and plot_curve.py --mode transfer draws the same data as grouped bars. Before
this file the transfer numbers were transcribed by hand into bfclscale.tex from a markdown lab
notebook, which is exactly the path a typo takes into a submission -- and it had already produced
one: the 2B TRIAGE-minus-uniform gap was being quoted as a single seed's paired difference while the
surrounding text called it a method mean.

WHAT A RUN IS. One directory per scored checkpoint, one JSON object per task attempt:

    <BFCL>/run_<arm>/records.jsonl   {"task_id", "sample", "reward", "category", "step", ...}

Every arm answers the SAME 800 tasks (task_index_sha256 is byte-identical across arms and is
checked here, not assumed), one sample per task, so a paired difference is just the difference of
two solve vectors over the shared keys. Keying on (task_id, sample) and taking the first occurrence
makes a re-run that appended twice harmless, the same convention main_table.load_cell uses for the
in-distribution cells.

NOTE ON `step`. Each run scores ONE checkpoint and the step it was taken at is recorded per row.
The selection rule is not uniform across arms -- the method's headline seed is pinned to the step
the matched window had already chosen, every other arm is at its best matched-window step -- so the
step is returned alongside the rate and the caption states the convention rather than hiding it.
"""

from __future__ import annotations

import json
import math
import os

BFCL_ROOT = os.path.join(os.environ.get("BRACE_WORK", "work"), "bfcl")

# The task index every arm must have answered. An arm scored against a different index is not
# comparable and is dropped loudly rather than averaged in.
TASK_INDEX_SHA256 = "cd489490938d51a60edc8a0a7cffeedf00c7cfa9193233221ba6bd4ac9f1020c"


def load_run(arm, root=BFCL_ROOT, require_index=True):
    """(solve dict, category counts, step, sha) for one scored checkpoint, or (None, ...) if absent.

    solve   : {(task_id, sample): 0|1}, first occurrence wins
    category: {category: [solved, attempted]}
    """
    p = os.path.join(root, "run_%s" % arm, "records.jsonl")
    if not os.path.exists(p):
        return None, None, None, None
    solve, cat, steps, shas = {}, {}, set(), set()
    for line in open(p, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        k = (d.get("task_id"), d.get("sample"))
        if None in k or k in solve:
            continue
        try:
            s = int(float(d["reward"]) > 0)
        except Exception:
            continue
        solve[k] = s
        c = d.get("category")
        cat.setdefault(c, [0, 0])
        cat[c][0] += s
        cat[c][1] += 1
        if d.get("step") is not None:
            steps.add(d["step"])
        if d.get("task_index_sha256"):
            shas.add(d["task_index_sha256"])
    if not solve:
        return None, None, None, None
    sha = sorted(shas)[0] if shas else None
    if require_index and sha and sha != TASK_INDEX_SHA256:
        raise SystemExit("%s was scored against task index %s, not %s -- not comparable"
                         % (arm, sha[:8], TASK_INDEX_SHA256[:8]))
    step = sorted(steps)[0] if len(steps) == 1 else (sorted(steps) or [None])[0]
    return solve, cat, step, sha


def rate(solve):
    """Overall pass rate in percent."""
    return 100.0 * sum(solve.values()) / len(solve)


def forced_share(arm, root=BFCL_ROOT):
    """(percent, n) of instances BFCL itself recorded as `multi_turn:force_terminated`.

    ONE OF THREE TERMINATION STATISTICS THIS PROJECT QUOTES, AND IT IS THE NARROWEST OF THE THREE.
    This one is BFCL's own recorded `error_type` on exactly the instances the pass rate beside it
    is read from -- same file, same keying, same first-occurrence rule as load_run -- so a table can
    carry both columns over one denominator. It is deliberately NOT either of the other two, and
    they are not interchangeable:

      * the OUTCOME TAXONOMY's "runaway" share (83.2% for the no-warm-bank ablation) additionally
        counts the context-overflow crashes an endless call loop produces, which BFCL files under
        a different error type; and
      * the PER-INSTANCE TURN-CAP FLAG (88.4% for the same arm) counts every instance that touched
        the 20-step cap at all, whether or not the cap is what the instance finally failed on.

    Both of those are computed in work/analysis/protection_mechanism.md, are quoted in the paper
    under their own names, and are larger than this statistic by construction. A table that
    silently swapped one for another would move a headline number by five to ten points, so this
    function exists to make the narrow one reproducible from the records rather than transcribed.
    Returns (None, 0) when the arm has no records.
    """
    p = os.path.join(root, "run_%s" % arm, "records.jsonl")
    if not os.path.exists(p):
        return None, 0
    seen = {}
    for line in open(p, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        k = (d.get("task_id"), d.get("sample"))
        if None in k or k in seen:
            continue
        seen[k] = d.get("error_type")
    if not seen:
        return None, 0
    n = len(seen)
    f = sum(1 for v in seen.values() if v == "multi_turn:force_terminated")
    return 100.0 * f / n, n


def paired(A, B):
    """A minus B on the tasks both answered: (diff_pp, z, p, n, b, c).

    b = B solved and A did not, c = A solved and B did not. Concordant pairs carry no information
    about the difference, so both the estimate and its uncertainty come from the discordants -- the
    same McNemar convention as the in-distribution tables.
    """
    keys = sorted(set(A) & set(B))
    n = len(keys)
    if not n:
        return None
    b = sum(1 for k in keys if B[k] and not A[k])
    c = sum(1 for k in keys if A[k] and not B[k])
    d = 100.0 * (c - b) / n
    z = (c - b) / math.sqrt(b + c) if (b + c) else 0.0
    return d, z, mcnemar_exact(b, c), n, b, c


def mcnemar_exact(b, c):
    """Exact two-sided conditional test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return min(1.0, 2 * tail)


def mean_sd(v):
    """(mean, sd or None). sd is the sample sd; a single value has no spread to report."""
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, None
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


# ---------------------------------------------------------------------------------------------
# THE THREE SCALES, and which runs constitute each cell. Seeds are listed explicitly rather than
# globbed: an arm appearing in this file is an arm the paper claims, and a glob would silently
# enrol the next probe run somebody scores.
SCALES = [("2B", "base2b", ["q2bT", "q2bT2"], ["q2bF5", "q2bF6", "q2bF7"]),
          ("4B", "base4b", ["q4bT", "q4bT2"], ["q4bF"]),
          ("8B", "base",   ["a8T3g", "a8T3gr", "a8T3g2"], ["a8F", "a8Fr"])]


def scale_block(root=BFCL_ROOT):
    """Everything the transfer table and the transfer figure both need, per scale.

    Multi-seed cells are summarised by the seed MEAN, and every between-method quantity is the
    mean over the seed PAIRINGS with its range and its weakest pairing's p -- never the best
    pairing. A gap quoted from one lucky pair is the error this paper spends its length arguing
    against, so the emitter cannot produce one.
    """
    out = []
    for name, basetag, tri, uni in SCALES:
        B, _, _, _ = load_run(basetag, root)
        if B is None:
            continue
        T = [(a, load_run(a, root)[0]) for a in tri]
        U = [(a, load_run(a, root)[0]) for a in uni]
        T = [(a, s) for a, s in T if s]
        U = [(a, s) for a, s in U if s]
        if not T or not U:
            continue
        tr, ur = [rate(s) for _, s in T], [rate(s) for _, s in U]
        tm, tsd = mean_sd(tr)
        um, usd = mean_sd(ur)
        tu = [paired(ts, us) for _, ts in T for _, us in U]
        tb = [paired(ts, B) for _, ts in T]
        ub = [paired(us, B) for _, us in U]
        out.append(dict(
            scale=name, base=rate(B), n=len(B),
            tri=tr, uni=ur, tri_arms=[a for a, _ in T], uni_arms=[a for a, _ in U],
            tri_mean=tm, tri_sd=tsd, uni_mean=um, uni_sd=usd,
            # between-method: mean over pairings, range, weakest p
            tu_mean=sum(x[0] for x in tu) / len(tu),
            tu_lo=min(x[0] for x in tu), tu_hi=max(x[0] for x in tu),
            tu_pmax=max(x[2] for x in tu), tu_npair=len(tu),
            tb_mean=sum(x[0] for x in tb) / len(tb),
            tb_lo=min(x[0] for x in tb), tb_hi=max(x[0] for x in tb),
            tb_pmax=max(x[2] for x in tb),
            ub_mean=sum(x[0] for x in ub) / len(ub),
            ub_lo=min(x[0] for x in ub), ub_hi=max(x[0] for x in ub),
            ub_pmax=max(x[2] for x in ub)))
    return out


if __name__ == "__main__":
    for r in scale_block():
        print("%s  base %.2f (n=%d)" % (r["scale"], r["base"], r["n"]))
        print("   TRIAGE  %s -> %.2f%s  (%d seeds)"
              % (["%.2f" % x for x in r["tri"]], r["tri_mean"],
                 "" if r["tri_sd"] is None else " +/- %.2f" % r["tri_sd"], len(r["tri"])))
        print("   uniform %s -> %.2f%s  (%d seeds)"
              % (["%.2f" % x for x in r["uni"]], r["uni_mean"],
                 "" if r["uni_sd"] is None else " +/- %.2f" % r["uni_sd"], len(r["uni"])))
        print("   T-U %+.2f pp  [%+.2f,%+.2f] over %d pairings, weakest p=%.2g"
              % (r["tu_mean"], r["tu_lo"], r["tu_hi"], r["tu_npair"], r["tu_pmax"]))
        print("   T-base %+.2f [%+.2f,%+.2f] weakest p=%.2g;  U-base %+.2f [%+.2f,%+.2f] "
              "weakest p=%.2g" % (r["tb_mean"], r["tb_lo"], r["tb_hi"], r["tb_pmax"],
                                  r["ub_mean"], r["ub_lo"], r["ub_hi"], r["ub_pmax"]))
