#!/usr/bin/env python
"""In-training validation curve: one held-out reward per checkpoint, comparable ACROSS arms.

WHY THIS EXISTS
---------------
Training reward cannot rank arms against each other. Arms train on different tool surfaces by
construction -- b8van sits at level 0.0, the ELSA arms drift to ~0.47, the reward-axis arms
(a8C/a8Cu/a8F) are pinned at 0.5 -- and a richer surface makes tasks easier for reasons that have
nothing to do with what the policy learned. A curve built from training reward therefore ranks
arms by how many tools they advertise. Three further reasons the raw numbers do not compare:

  * the training pool is what the policy was fit on, so its reward is partly memorisation;
  * arms with AWM_DENSE_REWARD (a8C, a8Cu, b8dense) put a SHAPED score in verl's reward tensor,
    so their `val-core/...` metric is not on the same scale as an unshaped arm's;
  * an arm that reweights its pool (ELSA/AWS) changes the task mix between cycles, so even its
    own training reward is not comparable with itself across cycles.

The validation pass fixes all four: a held-out pool (pool_coadapt_heldout, 0% overlap with
pool_max), one FIXED surface for every arm (work/coadapt_coadapt/advertised_init.txt, the same
surface the held-out eval cells score on), and greedy decoding so the number is reproducible.

WHAT THIS READS, AND WHY NOT verl's OWN METRIC
----------------------------------------------
verl prints `step:N - val-core/awm/reward/mean@1:V` on the console, and V is the mean of
AgentLoopOutput.reward_score. awm_agent_loop sets reward_score to the SHAPED reward when dense
shaping is on, so val-core is not comparable across the reward-axis arms -- exactly the arms this
curve exists to separate. episodes.jsonl carries the true binary outcome in `reward` no matter
what shaping is configured, so the curve is computed from the episodes and verl's metric is
reported alongside as a cross-check (`verl_metric`), never as the ranking quantity.

THE AUDIT
---------
`split` only records which branch the agent loop believed it was on. `n_tools` records how many
tools were actually advertised to the policy for that episode, which is what decides whether two
arms were scored in the same environment. Every val episode is checked against the tool count the
FIXED surface implies for its scenario; a point whose episodes disagree is reported as
surface_ok=False and is refused as a ranking key, because a point measured on the wrong surface
is worse than a missing point -- it looks like a result.

STEP NUMBERS
------------
Episodes carry no step index (episodes.jsonl is a flat append log), so validation passes are
recovered as maximal runs of consecutive `split="val"` records -- validation is its own
generate_sequences call, so no train episode interleaves. Steps come from the arm's console log,
which is authoritative. The log is truncated on every relaunch while episodes.jsonl is appended
forever, so the two are aligned at the TAIL and any earlier burst is labelled step=None rather
than being given a guessed number.

USAGE
    val_curve.py --tag b8k16v                    # one arm's curve
    val_curve.py --all                           # every arm that has validation episodes
    val_curve.py --rank STEP20@b8k16v STEP15@a8F # order candidate curve points, best first
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import re
import sys

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SURF = os.path.join(R, "surface", "gate_surface", "work", "surfaces")
FIXED_SURFACE = os.path.join(R, "work", "coadapt_coadapt", "advertised_init.txt")

# `step:N - key:value - key:value`, LocalLogger.concat_dict_to_str in
# verl/utils/logger/aggregate_logger.py. The core metric name carries the data_source and the
# @n of the validation rollout count, neither of which we want to hardcode.
_STEP_RE = re.compile(r"step:(\d+)")
# The value is pprint.pformat of whatever numpy handed back, so it is `0.42` on one verl build and
# `np.float64(0.42)` on another -- the smoke run printed the latter. Accept both rather than
# silently matching nothing, which would look exactly like "this arm never validated".
_VALCORE_RE = re.compile(
    r"val-core/[^ :]*?/(?:acc|reward)/mean@\d+:(?:np\.float\d+\()?([-+0-9.eE]+)\)?")


# ---------------------------------------------------------------- episode reading


def _episode_files(rundir: str) -> list[str]:
    """Every episode log in the run dir, oldest first.

    sft_traj.jsonl lives in the same directory and holds token tensors, not episodes; it is
    excluded by name rather than by schema sniffing so a malformed episode cannot be mistaken
    for one of its records.
    """
    ep = os.path.join(rundir, "episodes.jsonl")
    if os.path.exists(ep):
        return [ep]
    return sorted(f for f in glob.glob(os.path.join(rundir, "*.jsonl"))
                  if os.path.basename(f) != "sft_traj.jsonl")


# Memoised per (tag, tail_bytes, since_lines). The ranking path asks for the same arm's episodes
# twice -- once for its curve and once for the training-tail fallback -- and queue_curve.sh runs it
# on every supervisor cycle across every live arm, so the second read is pure duplicated I/O on
# files that only grow. Safe because every invocation of this module is a short-lived process.
_EP_CACHE: dict[tuple[str, int | None, int], tuple[list[dict], bool]] = {}


def arm_since(tag: str) -> int:
    """Rows of episodes.jsonl written by a PREVIOUS occupant of this run directory.

    Run directories are recycled between arms, and the supervisor stamps the row count at handover
    into `.arm_since`. Reading past it mixes two arms' episodes into one curve. Missing file means
    the directory has only ever held this arm.
    """
    try:
        p = os.path.join(R, "work", "verl", f"run_{tag}", ".arm_since")
        return int(open(p).read().strip() or 0)
    except Exception:
        return 0


def read_episodes(tag: str, tail_bytes: int | None = None,
                  since_lines: int = 0) -> tuple[list[dict], bool]:
    """Episodes for one arm, oldest first, and whether the read actually skipped any history.

    tail_bytes bounds the read to the end of the log. queue_curve.sh calls the ranking path on
    every supervisor cycle across ~16 arms whose episode logs run to megabytes and only grow, so
    an unbounded read would put a linear-in-history cost on the queue refill. The first line of a
    bounded read is discarded because a byte offset lands mid-record, and the first BURST is then
    dropped by the caller because a burst clipped by the window would report the mean of a
    fraction of a validation pass as if it were the whole thing.

    The truncation flag is measured, not inferred from `tail_bytes is not None`. Asking for a
    bound that the file is smaller than reads the whole file, and treating that as truncated
    threw away the FIRST validation pass of every arm -- which, with val_before_train, is the
    step-0 anchor the rest of the curve is read against.

    since_lines drops that many RAW leading lines -- raw, because `.arm_since` counts rows of the
    file and not rows that parsed, so filtering first would shift the offset. It is refused
    together with tail_bytes: a byte-bounded read has already thrown away the head the offset is
    measured from, so honouring both would silently drop a second, unrelated prefix.
    """
    if since_lines and tail_bytes is not None:
        raise ValueError("since_lines and tail_bytes cannot both be given: a tail-bounded read has "
                         "already discarded the head that .arm_since is an offset into")
    key = (tag, tail_bytes, since_lines)
    if key in _EP_CACHE:
        return _EP_CACHE[key]
    out: list[dict] = []
    truncated = False
    seen = 0
    for f in _episode_files(os.path.join(R, "work", "verl", f"run_{tag}")):
        with open(f, "rb") as fh:
            if tail_bytes is not None:
                size = os.fstat(fh.fileno()).st_size
                if size > tail_bytes:
                    fh.seek(size - tail_bytes)
                    fh.readline()                # partial record at the seek point
                    truncated = True
            for raw in fh:
                seen += 1
                if seen <= since_lines:
                    continue                     # a previous occupant of this run directory
                try:
                    d = json.loads(raw)
                except Exception:
                    continue                     # torn record from a concurrent append
                if "reward" in d and "scenario" in d:
                    out.append(d)
    _EP_CACHE[key] = (out, truncated)
    return out, truncated


def val_bursts(eps: list[dict], truncated: bool = False) -> list[list[dict]]:
    """Maximal runs of consecutive split="val" episodes; one run == one validation pass."""
    bursts, cur = [], []
    for e in eps:
        if e.get("split") == "val":
            cur.append(e)
        elif cur:
            bursts.append(cur)
            cur = []
    if cur:
        bursts.append(cur)
    # A window that opens inside a validation pass yields a first burst missing its head.
    if truncated and bursts and eps and eps[0].get("split") == "val":
        bursts = bursts[1:]
    return bursts


# ---------------------------------------------------------------- the fixed-surface audit


_EXPECTED: dict[str, int] | None = None


def expected_tool_counts(surface_file: str = FIXED_SURFACE) -> dict[str, int]:
    """Tools the FIXED surface advertises per scenario -- what a val episode must have seen.

    This mirrors awm_agent_loop._scenario_assets' val branch exactly (filter the scenario's raw
    tool list by name against the fixed set) without importing it, which would pull in verl and
    torch for what is a set intersection.
    """
    global _EXPECTED
    if _EXPECTED is not None:
        return _EXPECTED
    names = {ln.strip() for ln in open(surface_file) if ln.strip()}
    counts: dict[str, int] = {}
    for path in glob.glob(os.path.join(SURF, "*.json")):
        sc = os.path.basename(path)[:-5]
        try:
            tools = json.load(open(path))["tools"]
        except Exception:
            continue
        counts[sc] = sum(1 for t in tools if t["name"] in names)
    _EXPECTED = counts
    return counts


def audit_burst(burst: list[dict], surface_file: str = FIXED_SURFACE) -> tuple[bool, str]:
    """Did these val episodes actually run on the fixed surface?"""
    exp = expected_tool_counts(surface_file)
    missing = bad = 0
    for e in burst:
        n = e.get("n_tools")
        want = exp.get(e.get("scenario"))
        if n is None:
            missing += 1                        # episode predates the n_tools field
        elif want is None or int(n) != int(want):
            bad += 1
    if bad:
        return False, f"{bad}/{len(burst)} episodes advertised a tool count the fixed surface does not imply"
    if missing == len(burst):
        return False, "no n_tools recorded; surface cannot be verified from this log"
    if missing:
        return True, f"{len(burst) - missing}/{len(burst)} episodes verified against the fixed surface"
    return True, "all episodes verified against the fixed surface"


# ---------------------------------------------------------------- step recovery


def _log_files(tag: str, log: str | None = None) -> list[str]:
    """The live console log plus its rotations, oldest run first.

    The supervisor used to truncate logs/rl_<tag>.log on every launch, and the step index of a
    validation pass exists ONLY in that log -- episodes.jsonl carries no step. So each relaunch
    silently cut points out of the curve, worst for the arms that crash most, which is exactly
    where a gap reads as a flat curve. It now rotates to rl_<tag>.log.N(.gz) instead; reading
    them back is what turns that rotation into a curve rather than an archive nobody opens.
    """
    if log:
        return [log] if os.path.exists(log) else []
    live = os.path.join(R, "logs", f"rl_{tag}.log")
    rots = []
    for p in glob.glob(live + ".*"):
        m = re.search(r"\.(\d+)(?:\.gz)?$", p)
        if m:
            rots.append((int(m.group(1)), p))
    # rotation numbers increase with age of the *rotation event*, so ascending N is oldest run
    # first, and the live log is newer than all of them.
    return [p for _, p in sorted(rots)] + ([live] if os.path.exists(live) else [])


def logged_val_steps(tag: str, log: str | None = None) -> list[tuple[int, float]]:
    """(step, verl val-core value) for each validation pass the console logs still hold."""
    out: list[tuple[int, float]] = []
    for path in _log_files(tag, log):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt", errors="ignore") as fh:
                for line in fh:
                    if "val-core/" not in line:
                        continue
                    m_step, m_val = _STEP_RE.search(line), _VALCORE_RE.search(line)
                    if m_step and m_val:
                        try:
                            out.append((int(m_step.group(1)), float(m_val.group(1))))
                        except ValueError:
                            pass
        except OSError:
            continue                      # a rotation being gzipped underneath us is not an error
    return out


def _mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if not n:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def curve(tag: str, log: str | None = None, surface_file: str = FIXED_SURFACE,
          tail_bytes: int | None = None) -> list[dict]:
    """One record per validation pass, oldest first."""
    eps, truncated = read_episodes(tag, tail_bytes)
    eps_tail_is_val = bool(eps) and eps[-1].get("split") == "val"
    bursts = val_bursts(eps, truncated=truncated)
    if not bursts:
        return []
    logged = logged_val_steps(tag, log)
    # A burst is CLOSED once a training episode follows it; the trailing one is still open. An
    # open burst with no metric to match it is a validation pass in flight -- its mean is over a
    # fraction of the pool and would read as a collapse in reward, so it is carried with
    # step=None instead of being paired. (verl also validates once at the end of a run, and that
    # burst stays open forever; it is real, and the len() test below is what tells the two apart.)
    open_tail = bool(bursts) and eps_tail_is_val
    if open_tail and len(bursts) > len(logged):
        pairable, in_flight = bursts[:-1], 1
    else:
        pairable, in_flight = bursts, 0
    # Align at the TAIL, pairing only the last k = min(len) of each. episodes.jsonl survives
    # relaunches and the log does not, so the log usually has FEWER passes; but a bounded
    # (tail_bytes) episode read can invert that, so neither list may be assumed longer. Anything
    # outside the paired window gets step=None rather than a guessed number -- a curve plotted
    # against invented x values is not a curve.
    k = min(len(pairable), len(logged))
    b0, l0 = len(pairable) - k, len(logged) - k
    rows = []
    for i, b in enumerate(bursts):
        in_win = i >= b0 and i < len(pairable)
        step, verl_metric = (logged[i - b0 + l0] if in_win else (None, None))
        ok, why = audit_burst(b, surface_file)
        if i >= len(pairable) and in_flight:
            why += "; validation pass still in flight, not a curve point yet"
        rewards = [float(e["reward"]) > 0 for e in b]
        m, se = _mean_se([float(x) for x in rewards])
        rows.append(dict(
            tag=tag, pass_index=i, step=step, n=len(b),
            val_reward=round(m, 4), se=(None if se != se else round(se, 4)),
            n_scenarios=len({e.get("scenario") for e in b}),
            surface_ok=ok, audit=why, complete=True,
            verl_metric=(None if verl_metric is None else round(verl_metric, 4)),
        ))
    # A pass killed part-way through -- a preempted arm, an engine death mid-validation -- leaves
    # a short burst that is still closed by the training episodes of the relaunch, so nothing
    # above would catch it. Its mean is over whichever slice of the pool finished first, which is
    # not a random sample of it: the pool is grouped by scenario, so a truncated pass measures
    # some scenarios and none of the others. The pool size is not knowable from the episode log,
    # so the arm's own largest pass stands in for it.
    nmax = max(r["n"] for r in rows)
    for r in rows:
        if r["n"] < 0.9 * nmax:
            r["complete"] = False
            r["audit"] += f"; only {r['n']}/{nmax} of the pool -- truncated pass, not a curve point"
    return rows


# ---------------------------------------------------------------- solve rate against training step
#
# WHY A SECOND CURVE FUNCTION AND NOT `curve()`. `curve()` reports one row per BURST, and a burst is
# not a validation pass: a relaunch that validates before its first training step appends a second
# pass with no training episode between the two, so the two merge into one burst and `curve()`
# reports their pooled mean against ONE of the two steps. That is harmless for ranking checkpoints
# (the two steps are adjacent and the ranking only needs an ordering) and wrong for a figure with a
# step axis. It also aligns bursts to log lines one-for-one, which is off by however many bursts
# merged.
#
# So passes are recovered here as fixed-size chunks of the pool, and -- this is the part that makes
# it trustworthy rather than plausible -- every chunk's mean is CHECKED against verl's own
# `val-core` value for the log line it was aligned to. The two are computed from different sources
# (our episodes.jsonl versus verl's reward tensor) and agree to 0.06pp on a correct alignment,
# because verl averages over 295 of the 296 episodes. A chunk that disagrees is a chunk whose step
# is a guess, and it is dropped rather than plotted: q2bT3 crashed and re-validated ~14 times at
# step 5, leaving one 4298-episode burst with a partial pass somewhere inside it that no fixed-size
# chunking can locate, and this check is what removes exactly those five points and keeps the ten
# that are sound.
#
# The value plotted is the SOLVE RATE on the held-out val pool under greedy decoding -- the same
# quantity, from the same episodes and the same `.arm_since` window, that the paper's termination
# table reports as its first-third/last-third contrast.

# Tolerance for calling an episode-derived pass mean and verl's logged val-core the same
# measurement, as a fraction. 0.001 is ~15x the 0.0000-0.0006 spread a correct alignment shows and
# ~10x below the smallest disagreement a wrong one has produced.
VERIFY_TOL = 0.001


def solve_curve(tag: str, log: str | None = None, surface_file: str = FIXED_SURFACE,
                tol: float = VERIFY_TOL) -> list[dict]:
    """Validation solve rate per training step for one arm, oldest first.

    One row per STEP, not per pass: an arm that validated twice at the same step (a relaunch
    revalidating the checkpoint it resumed from) has both passes pooled, because they are two
    samples of one checkpoint and averaging them is the whole of what the second pass is for.
    """
    eps, _ = read_episodes(tag, since_lines=arm_since(tag))
    bursts = val_bursts(eps)
    if not bursts:
        return []
    # The pool size is the modal burst length. Every merged burst is an exact multiple of it and
    # every truncated one is short of it, so the mode is the pass size in any run whose passes are
    # not mostly broken -- and the val-core check below fails loudly if it ever is not.
    sizes = [len(b) for b in bursts]
    pool = max(set(sizes), key=lambda s: (sizes.count(s), -s))
    passes = [b[j * pool:(j + 1) * pool] for b in bursts for j in range(len(b) // pool)]
    logged = logged_val_steps(tag, log)
    # Align at the TAIL, exactly as curve() does and for the same reason: the console log is
    # truncated on relaunch while episodes.jsonl is appended forever, so the log holds the LAST k
    # passes and never the first k.
    k = min(len(passes), len(logged))
    p0, l0 = len(passes) - k, len(logged) - k
    by_step: dict[int, list[list[dict]]] = {}
    rejected = 0
    for i in range(k):
        p = passes[p0 + i]
        step, verl_metric = logged[l0 + i]
        m = sum(1 for e in p if float(e["reward"]) > 0) / len(p)
        if verl_metric is None or abs(m - verl_metric) > tol or not audit_burst(p, surface_file)[0]:
            rejected += 1
            continue
        by_step.setdefault(step, []).append(p)
    rows = []
    for step in sorted(by_step):
        flat = [e for p in by_step[step] for e in p]
        m, se = _mean_se([1.0 if float(e["reward"]) > 0 else 0.0 for e in flat])
        rows.append(dict(tag=tag, step=step, solve=round(m, 6), se=round(se, 6),
                         n=len(flat), passes=len(by_step[step]),
                         dropped=rejected, pool=pool,
                         unpaired=len(passes) - k))
    return rows


# ---------------------------------------------------------------- ranking curve points


# Bytes of episode log the ranking path reads per arm. 8 MB is ~40k episodes at the observed
# ~200 B/record, which covers every arm's whole history today and still bounds the cost as the
# fleet runs for weeks.
RANK_TAIL_BYTES = 8 << 20


def _train_reward_tail(eps: list[dict], n: int = 640) -> float | None:
    """Last-n TRAINING reward: the fallback ranking key, and NOT comparable across arms.

    Used only to order points that have no validation measurement at all, so the eval queue keeps
    working while arms are being migrated onto VAL_FREQ. Any point with a real validation number
    outranks every point ordered by this, which is why it is returned in its own tier rather than
    mixed into the same scale.
    """
    tr = [e for e in eps if e.get("split") != "val"]
    if not tr:
        return None
    tail = tr[-n:]
    return sum(float(e["reward"]) > 0 for e in tail) / len(tail)


def score_point(tag: str, step: int, cache: dict) -> tuple[int, float, int, str]:
    """Ranking key for one STEP<step>@<tag> curve point. Higher sorts first.

    Returns (tier, score, -distance, why):
      tier 2  a verified validation measurement for this arm, nearest to `step`
      tier 1  the arm's recent training reward -- an ordering of last resort, not a comparison
      tier 0  nothing known
    """
    if tag not in cache:
        rows = [r for r in curve(tag, tail_bytes=RANK_TAIL_BYTES)
                if r["step"] is not None and r["surface_ok"] and r["complete"]]
        cache[tag] = dict(val=rows,
                          train=_train_reward_tail(read_episodes(tag, RANK_TAIL_BYTES)[0]))
    c = cache[tag]
    if c["val"]:
        best = min(c["val"], key=lambda r: abs(r["step"] - step))
        d = abs(best["step"] - step)
        return (2, best["val_reward"], -d,
                f"val={best['val_reward']:.3f} at step {best['step']}"
                + ("" if d == 0 else f" (nearest to {step}, {d} steps away)"))
    if c["train"] is not None:
        return (1, c["train"], 0, f"no validation data; train tail={c['train']:.3f} (NOT comparable across arms)")
    return (0, 0.0, 0, "no episodes")


def rank_points(points: list[str]) -> list[tuple[str, str]]:
    """Order STEP<n>@<arm> ids best-validation-reward first."""
    cache: dict = {}
    scored = []
    for p in points:
        m = re.match(r"^STEP(\d+)@(.+)$", p.strip())
        if not m:
            # Same arity as every other key so the sort never falls back to comparing tuple
            # lengths, and below tier 0 so an id we cannot read is never promoted over one we can.
            scored.append(((-1, 0.0, 0, 0), p, "unparsable point id; left at the end"))
            continue
        step, tag = int(m.group(1)), m.group(2)
        tier, score, dist, why = score_point(tag, step, cache)
        # Step ascending inside a tie so an arm's own curve is still filled in order.
        scored.append(((tier, score, dist, -step), p, why))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(p, why) for _, p, why in scored]


# ---------------------------------------------------------------- cli


def _arms_with_val() -> list[str]:
    out = []
    for d in sorted(glob.glob(os.path.join(R, "work", "verl", "run_*"))):
        tag = os.path.basename(d)[4:]
        if any(val_bursts(read_episodes(tag)[0])):
            out.append(tag)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", action="append", default=[], help="arm tag; repeatable")
    ap.add_argument("--all", action="store_true", help="every arm that has validation episodes")
    ap.add_argument("--log", help="console log to take step numbers from (default logs/rl_<tag>.log)")
    ap.add_argument("--surface", default=FIXED_SURFACE,
                    help="the fixed validation surface the audit checks against")
    ap.add_argument("--format", choices=["table", "tsv", "json"], default="table")
    ap.add_argument("--rank", nargs="*", default=None,
                    help="order STEP<n>@<arm> ids best-first; reads stdin when given no arguments")
    a = ap.parse_args()

    if a.rank is not None:
        pts = a.rank or [ln.strip() for ln in sys.stdin if ln.strip()]
        for p, why in rank_points(pts):
            print(f"{p}\t{why}" if a.format != "tsv" else f"{p}\t{why}")
        return 0

    tags = a.tag or (_arms_with_val() if a.all else [])
    if not tags:
        print("no arm selected and none has validation episodes; pass --tag or --all",
              file=sys.stderr)
        return 1

    rows = [r for t in tags for r in curve(t, a.log, a.surface)]
    if a.format == "json":
        print(json.dumps(rows, indent=2))
        return 0
    cols = ["tag", "step", "n", "val_reward", "se", "n_scenarios", "verl_metric",
            "surface_ok", "complete"]
    if a.format == "tsv":
        print("\t".join(cols))
        for r in rows:
            print("\t".join("" if r[c] is None else str(r[c]) for c in cols))
        return 0
    if not rows:
        print("no validation passes found (arm has no split=\"val\" episodes)")
        return 0
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(("" if r[c] is None else str(r[c])).ljust(w[c]) for c in cols))
    bad = [r for r in rows if not (r["surface_ok"] and r["complete"])]
    for r in bad:
        print(f"  ! {r['tag']} pass {r['pass_index']}: {r['audit']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
