"""The paper's main comparison: every arm against the base policy on the SAME held-out tasks.

WHY THIS EXISTS SEPARATELY FROM rl_learning_curve.py. That script prints one row per (arm, step),
which is the right diagnostic while a run is in flight and the wrong table for a paper: with ~15
checkpoints per arm, reporting each arm at its own best step is a maximum over 15 correlated tests,
and the winner is then whoever got the luckiest checkpoint. a8F's +0.049 at step 20 and atscfix's
+0.044 at step 56 are not comparable numbers -- they are two different selection procedures.

THREE THINGS THIS DOES INSTEAD

1. MATCHED STEPS. Every arm is compared at the same training step, so the column means what it
   says. The peak window (15-30) is reported because every arm measured so far peaks there and
   decays afterwards -- a8F runs +0.049, +0.041, +0.022, -0.018 across steps 20, 30, 35, 40 -- so
   a late-step comparison would rank arms by how fast they degrade.

2. PAIRED TESTING ON THE SHARED TASK SET. The effect is McNemar's exact conditional test on the
   tasks the arm and the base policy both attempted, which is the same statistic \textsc{Discord}
   uses for harness edits. Unpaired accuracy differences would be dominated by which tasks each
   cell happened to cover.

3. HOLM CORRECTION ACROSS ARMS. Ten arms tested against one base is ten chances at p<0.05; the
   probability at least one clears by luck is ~40%. Holm-Bonferroni controls that at the family
   level while being uniformly more powerful than plain Bonferroni. The uncorrected p is kept
   alongside so the reader can see both.

4. ONE FAMILY PER MODEL, AND ONE ANCHOR PER MODEL. Added 2026-08-14. The glob below admits every
   `work/coadapt_eval_*` directory, and the fleet now trains 2B and 4B arms alongside the 8B ones.
   Two things go wrong if they share a table. The anchor is wrong: --base defaults to the 8B base
   policy, so a 2B arm's "effect" is (2B policy - 8B policy), a model-capability difference wearing
   an allocation label. And the family is wrong: a 2B arm landing a cell raises m for every 8B arm
   and moves every q in the paper, for a comparison that was never part of the 8B claim. Holm
   families are therefore PER MODEL -- the audit's recommendation, made before any of the 2B cells
   existed -- and --model selects which one is built. Each model's table needs its own base cell:

       8B   work/coadapt_eval_coadapt/cell_A.jsonl       (n=1180)  <- --base default
       2B   work/coadapt_eval_probe2b/cell_PROBE2B.jsonl (n=1125)
       4B   work/coadapt_eval_probe4b/cell_PROBE4B.jsonl  (n=590)

   All three anchors are now above --min-pairs and all three families are built (2026-08-16).

   Membership comes from the arm's REGISTERED MODEL, parsed out of slurm/supervisor.sh's case
   block, not from the tag spelling: `MODEL=` appears there per tag and verl_awm_train.sh defaults
   it to Qwen/Qwen3-VL-8B-Instruct, so a tag with no MODEL= is 8B by the same rule the trainer
   uses. If that file cannot be read the code falls back to the q2b*/q4b* tag prefix and says so.

WHAT IT REFUSES TO DO. A cell with fewer than --min-pairs paired tasks is reported as insufficient
rather than scored. Small cells are exactly where a spurious +0.05 comes from: at n=40 a single
discordant pair moves the estimate by 0.025.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_cell(path):
    """{(scenario, task, seed): solved} from one eval cell, first occurrence wins.

    Duplicates and torn records both occur: two cards briefly shared a cell on 2026-08-09 and
    appended concurrently, leaving 410 lines holding 162 distinct tasks plus 40 unparsable rows.
    Keying the dict is what makes that harmless.
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


SUPERVISOR = f"{R}/slurm/supervisor.sh"
# The trainer's own default (slurm/verl_awm_train.sh: MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}),
# so a tag whose case line sets no MODEL= is 8B by exactly the rule that launched it.
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_CASE = re.compile(r"^\s*([A-Za-z0-9_|]+)\)\s")
_MODEL = re.compile(r"MODEL=([^\s\"']+)")
# ---------------------------------------------------------------------------------------------
# PREEMPTION CONTINUATIONS. An arm that was preempted and relaunched from its own merged
# checkpoint is ONE arm with one trajectory, not two, and must occupy one row and one slot in the
# Holm family -- counting it twice both invents a comparison and raises m for everyone else.
#
# a8Tvip was preempted at step 15 and cannot be resumed exactly (its optimizer state was not
# checkpointed); a8Tvipc restarts from merged_a8Tvip_step15 with a fresh optimizer. supervisor.sh
# (see its a8Tvipc block) fixes the mapping: a8Tvipc step s == a8Tvip step 15+s. The optimizer
# reinit at the seam is a real caveat and the paper states it; it is not a reason to score the two
# halves as unrelated arms.
#
# {arm: {window_step: (source_arm, source_step)}}
CONTINUATION = {"a8Tvip": {20: ("a8Tvipc", 5), 25: ("a8Tvipc", 10), 30: ("a8Tvipc", 15)}}
# Arms that exist only as the tail of another arm: never a family member in their own right.
ABSORBED = {"a8Tvipc"}


def cell_path(arm: str, step: int) -> str:
    """Where arm's window-step cell actually lives, honouring CONTINUATION."""
    src, st = CONTINUATION.get(arm, {}).get(step, (arm, step))
    return f"{R}/work/coadapt_eval_{src}/cell_STEP{st}.jsonl"


# Anchors, one per model size. All three tables are now built from these.
BASE_CELLS = {"8B": f"{R}/work/coadapt_eval_coadapt/cell_A.jsonl",
              "2B": f"{R}/work/coadapt_eval_probe2b/cell_PROBE2B.jsonl",
              "4B": f"{R}/work/coadapt_eval_probe4b/cell_PROBE4B.jsonl"}


def model_of(tag: str, registry: dict | None = None) -> str:
    """'2B' | '4B' | '8B' for one arm tag, from its registered MODEL where that is readable."""
    reg = registry if registry is not None else arm_models()
    name = reg.get(tag)
    if name is None:                      # unregistered eval-only dir (probes, old pool names)
        # 2026-08-31 BUG FIX, found while pinning the fold-v arms. This branch used to yield the
        # BARE SIZE ("2B"), which the regex below -- written for a HuggingFace name like
        # Qwen/Qwen3-VL-2B-Instruct -- cannot match, so it fell through to the 8B default and
        # every unregistered q2b*/q4b* tag was classified as an 8B arm. It was latent only
        # because every arm a table had ever named was in supervisor.sh's case block; the arms
        # launched from logs/.ablate_finish*.sh and logs/.rerun_finish*.sh are not, and all seven
        # of them came back 8B. The fallback now builds a name of the SAME SHAPE the registry
        # returns, so there is one parse and not two.
        # 2026-09-01: this test was a literal prefix match on "q2b"/"q4b", so an unregistered arm
        # launched under any other leading letter fell through to the 8B default. t2bTbk, the
        # TRACE-plus-our-bank arm at 2B, came back "8B" and could not join its own family. The size
        # is the SECOND character whenever the third is "b", so the test reads that instead of
        # enumerating prefixes. Checked over all 135 eval directories before the change: exactly
        # one arm is reclassified (t2bTbk, 8B -> 2B) and nothing else moves.
        m0 = re.match(r"^[a-z](\d)b", tag)
        size = (m0.group(1) + "B") if m0 else None
        name = DEFAULT_MODEL if size is None else "Qwen/Qwen3-VL-%s-Instruct" % size
    m = re.search(r"-(\d+)B-", name)
    return f"{m.group(1)}B" if m else "8B"


def arm_models() -> dict:
    """{tag: MODEL string} from slurm/supervisor.sh's case block. Read-only; never written.

    Parsed rather than hard-coded because the fleet adds arms faster than any table would be
    updated by hand, and a stale hard-coded list is how a 2B arm ends up inside the 8B family in
    the first place. A tag that appears with no MODEL= is left out and picks up DEFAULT_MODEL.
    """
    out = {}
    try:
        text = open(SUPERVISOR, errors="ignore").read()
    except Exception:
        return out                        # caller falls back to the tag prefix and reports it
    for line in text.splitlines():
        m = _CASE.match(line)
        if not m:
            continue
        mm = _MODEL.search(line)
        for tag in m.group(1).split("|"):
            if mm:
                out[tag] = mm.group(1)
            else:
                out.setdefault(tag, DEFAULT_MODEL)
    return out


def mcnemar(b, c):
    """Exact two-sided conditional test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, run = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="",
                    help="base-policy anchor; defaults to BASE_CELLS[--model]. advertised_init.txt "
                         "is byte-identical across arms, so one base measurement serves every "
                         "comparison AT THAT MODEL SIZE -- and only at that size")
    ap.add_argument("--model", default="8B", choices=sorted(BASE_CELLS),
                    help="which per-model Holm family to build. Arms of other sizes are excluded "
                         "from the family entirely, not merely reported separately: they would "
                         "otherwise raise m for every arm here (see the header, item 4)")
    ap.add_argument("--steps", default="15,20,25,30",
                    help="matched steps to report. Default is the window where every arm measured "
                         "so far peaks; later steps rank arms by rate of decay instead.")
    ap.add_argument("--min-pairs", type=int, default=300,
                    help="paired tasks a cell needs before it is scored at all")
    ap.add_argument("--unpinned", action="store_true",
                    help="FORENSIC ONLY: build the family from the live glob instead of "
                         "paper_numbers.FAMILY_PIN. The paper's q values come from the pin; this "
                         "switch exists to show what the glob would say, not to report it")
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split(",") if s.strip()]
    base_path = a.base or BASE_CELLS[a.model]
    # ONE DEFINITION OF THE FAMILY, IMPORTED RATHER THAN RESTATED (2026-08-22). This reporter used
    # to glob while paper_numbers.py pinned, so the step-15 family quoted in the Results section
    # could disagree with the window family printed two tables above it -- and by 2026-08-22 it
    # did, 40 against 37, which moved three quoted q values. The pin is the paper's family; this
    # file now reads it. Imported inside main() because paper_numbers imports THIS module at
    # module level and the pair would otherwise be a circular import.
    pin = None
    if not a.unpinned:
        from paper_numbers import FAMILY_PIN                  # noqa: E402  one definition
        pin = FAMILY_PIN.get(a.model)

    registry = arm_models()
    if not registry:
        print(f"[warn] could not read {SUPERVISOR}; falling back to the q2b*/q4b* tag prefix "
              f"for model membership")

    base = load_cell(base_path)
    if not base:
        print(f"no base cell at {base_path}; cannot anchor the comparison")
        return 1
    print(f"MAIN TABLE  model {a.model}  base policy n={len(base)} ({os.path.basename(base_path)})"
          f"  matched steps {steps}  min pairs {a.min_pairs}\n")

    # arm -> step -> (diff, p, n, base_rate, arm_rate)
    res, other = defaultdict(dict), defaultdict(set)
    held_out = set()
    for d in sorted(glob.glob(f"{R}/work/coadapt_eval_*")):
        arm = os.path.basename(d).replace("coadapt_eval_", "")
        # THE FAMILY FILTER. An arm of another size is dropped before it can be scored -- it has
        # the wrong anchor here anyway, so scoring it would produce a number that means nothing and
        # then charge every arm in this family for it.
        am = model_of(arm, registry)
        if am != a.model or arm in ABSORBED:      # ABSORBED: the tail of another arm, not an arm
            if am != a.model and glob.glob(f"{d}/cell_STEP*.jsonl"):
                other[am].add(arm)
            continue
        if pin is not None and arm not in pin:
            # Only REGISTERED arms with a scorable cell are named as held out. work/coadapt_eval_*
            # also holds probe and pool directories that were never arms, and listing those as
            # "excluded from the family" would overstate what the pin is doing by a factor of ten.
            if arm in registry and any(
                    os.path.exists(cell_path(arm, st))
                    and len(set(load_cell(cell_path(arm, st))) & set(base)) >= a.min_pairs
                    for st in steps):
                held_out.add(arm)
            continue
        for st in steps:
            f = cell_path(arm, st)                # honours CONTINUATION
            if not os.path.exists(f):
                continue
            cur = load_cell(f)
            keys = sorted(set(cur) & set(base))
            if len(keys) < a.min_pairs:
                res[arm][st] = None
                continue
            b = sum(1 for k in keys if base[k] and not cur[k])
            c = sum(1 for k in keys if cur[k] and not base[k])
            br = sum(base[k] for k in keys) / len(keys)
            ar = sum(cur[k] for k in keys) / len(keys)
            res[arm][st] = (ar - br, mcnemar(b, c), len(keys), br, ar)

    if not res:
        print("no cells at the matched steps yet")
        return 0

    # one row per arm: the mean effect over the matched steps it actually has
    rows = []
    for arm, by_step in res.items():
        got = [(s, v) for s, v in sorted(by_step.items()) if v]
        if not got:
            continue
        diffs = [v[0] for _, v in got]
        # combine the per-step tests conservatively: the arm's weakest evidence in the window
        pmax = max(v[1] for _, v in got)
        rows.append({"arm": arm, "mean": sum(diffs) / len(diffs), "n_steps": len(got),
                     "steps": [s for s, _ in got], "p": pmax,
                     "per_step": {s: v for s, v in got}})
    if not rows:
        print("cells exist but none reached --min-pairs")
        return 0
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = q
    rows.sort(key=lambda r: -r["mean"])

    hdr = f"  {'arm':<10} {'mean diff':>10} {'steps':>6} {'p':>9} {'p(holm)':>9}   per-step"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        per = " ".join(f"{s}:{r['per_step'][s][0]:+.3f}" for s in r["steps"])
        star = " *" if r["p_holm"] < 0.05 else ""
        print(f"  {r['arm']:<10} {r['mean']:>+10.4f} {r['n_steps']:>6} "
              f"{r['p']:>9.4f} {r['p_holm']:>9.4f}{star}   {per}")
    print(f"\n  * significant after Holm correction across the {len(rows)} arms of the {a.model} "
          f"family")
    for m in sorted(other):
        print(f"  [excluded, {m} family] {' '.join(sorted(other[m]))} -- own family, own anchor "
              f"({os.path.basename(BASE_CELLS.get(m, 'none registered'))})")
    if held_out:
        print(f"  [held out of FAMILY_PIN[{a.model!r}]] {' '.join(sorted(held_out))} -- see that "
              f"dict for the reason on each; --unpinned shows what the glob would say")

    # what is missing, so the queue can be aimed at it
    missing = []
    for arm, by_step in sorted(res.items()):
        for s in steps:
            if by_step.get(s) is None:
                missing.append(f"STEP{s}@{arm}")
    if missing:
        print(f"\n  cells still needed for a complete matched table ({len(missing)}):")
        print("   ", " ".join(missing[:24]) + (" ..." if len(missing) > 24 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
