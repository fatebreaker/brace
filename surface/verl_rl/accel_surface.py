"""ACCEL: evolve the tool surface by REGRET, the closest published relative of ATSC.

WHY THIS IS THE BASELINE THAT MATTERS. ATSC's contribution is not "adapt the environment" -- UED
has done that for years -- it is the specific claim that the right control target is gradient
availability g(p,k)=1-p^k-(1-p)^k, and that an MCP tool surface is the dial that reaches it. The
strongest existing method making the general claim is ACCEL (Parker-Holder et al., 2022): keep an
archive of environment configurations, prioritise the high-REGRET ones, EDIT them with small
mutations, and keep the children that stay high-regret. PLR (Jiang et al., 2021) only *selects*
among a fixed set of levels -- b8plr already covers that -- whereas ACCEL *edits* them, which is
structurally the same move ATSC makes. If ACCEL over the same dial, on the same pool, at the same
budget, reaches the same place, then the availability target is decoration and the paper says so.

The two arms are deliberately identical everywhere except the search rule:
  * same dial          per-scenario restoration level in [0,1] over the same withheld names
  * same step size     0.25 per cycle, so one edit is one edit
  * same realisation   identical level -> advertised-set mapping, byte-for-byte
  * same budget/pool   pool_max, same handicapped init surface
  * DIFFERENT rule     ATSC: proportional control on (p - 0.5), a directed closed loop.
                       ACCEL: random mutation + regret-prioritised archive, an evolutionary
                       search that never encodes which DIRECTION helps.

================================ THE REGRET ESTIMATOR ================================
ACCEL scores a level by regret, approximated in the paper by POSITIVE VALUE LOSS (PVL), the
average positive part of the GAE residual:

        PVL  =  (1/T) * sum_t  max( sum_{k>=t} (gamma*lambda)^{k-t} delta_k , 0 )

PVL is not computable here, for a reason that is structural rather than an implementation gap:
GRPO is CRITIC-FREE. There is no value head, so there is no delta_k and no GAE. Substituting a
learned critic purely to score levels would add an estimator the method under test does not have
and make the baseline's compute and failure modes incomparable.

WHAT IS USED INSTEAD, and why it is the same quantity. GRPO replaces the critic with the
group-relative advantage: for the k rollouts of one task with returns r_1..r_k,

        A_i = (r_i - mean(r)) / std(r)

This IS the advantage estimate the arm trains on -- verl's `grpo` adv_estimator computes exactly
this. Taking the positive part of it and averaging is therefore the literal GRPO analogue of PVL:
"how much better than its own expectation did this environment let the policy do, when it did
better at all". So

        regret(task)      =  (1/k) * sum_i  max(A_i, 0)
        regret(scenario)  =  mean over that scenario's tasks

is PVL with the only available value baseline, not a loose proxy. It is computed from the rollouts
coadapt.py already banks in episodes.jsonl; the baseline costs no extra generation, which is also
what makes it a fair comparison rather than one that wins or loses on sample budget.

CONSEQUENCE, STATED UP FRONT BECAUSE IT IS THE OBVIOUS ATTACK. For binary rewards this estimator
has a closed form: with solve rate p, exactly kp rollouts have A = (1-p)/s and s = sqrt(p(1-p)),
so regret = p(1-p)/s = SQRT(P(1-P)). That is maximised at p = 0.5 -- the same argmax as gradient
availability. Two honest readings:

  1. It does NOT make the arms equivalent. Both objectives peak at p=0.5, but ATSC *knows the
     sign*: it observes p and moves the surface toward 0.5 in one proportional step. ACCEL sees
     only a scalar score, mutates at random, and needs a full cycle of rollouts per scenario to
     discover whether the mutation helped. The experiment is whether a directed controller beats
     an evolutionary search on the identical landscape, which is a real and unanswered question
     at 120 GRPO steps -- ACCEL's published results use tens of thousands of updates.

  2. It is the STRONGEST form of the baseline, and that is the point. Picking a regret proxy that
     peaked somewhere other than p=0.5 would hand ATSC the win by construction. Handing ACCEL an
     estimator whose optimum coincides with ATSC's target means any gap that survives is a gap in
     the SEARCH, which is the only thing being claimed.

`--regret {advantage,success-variance}` also exposes the closed form directly. `advantage` (the
default) is computed empirically per group from the banked returns, so unequal group sizes,
truncated cycles and the k=5 sample std are all handled by the data rather than by an assumption.

================================ THE ACCEL LOOP ================================
Per training cycle, over the scenarios (one lineage each, seeded at level 0.5):

  1. EVALUATE   regret of each scenario's CURRENT level from this cycle's banked rollouts.
  2. ADMIT      a configuration enters the archive when its regret beats the running median of
                measured regret. The threshold is a statistic of the run, not a constant: absolute
                regret scales with solve rate, which moves as the policy trains, so a fixed
                threshold admits everything early and nothing later.
  3. SURVIVE    a mutated child that scored WORSE than its parent's archived regret is discarded
                and the lineage reverts to the archived level. This is what makes it evolution
                rather than a random walk.
  4. SELECT     which lineages to edit, by PLR's prioritisation:
                    score = (1 - rho) * P_regret + rho * P_staleness
                with rank-based P_regret ∝ (1/rank)^(1/beta) and P_staleness ∝ cycles since the
                lineage was last measured. Staleness is not optional: a scenario whose tasks stop
                appearing in the sampled batches would otherwise keep a stale regret forever and
                either monopolise or never receive edits.
  5. EDIT       level' = clip(level + 0.25 * u), u uniform in {-1, +1}. Deliberately UNDIRECTED.

State lives in accel_state.json next to levels.json so a supervisor restart resumes the archive
instead of restarting evolution from scratch -- an arm that silently reseeds its archive on every
relaunch is a random-surface arm wearing ACCEL's name.

DETERMINISM. The level -> advertised-set realisation is copied from surface_control.py with ONE
correction: the per-scenario shuffle is seeded from a CRC of the scenario name, not from the
builtin hash(). Python salts str hashing per process, so surface_control's
`random.Random(hash(sc) & 0xFFFF)` draws a different restoration order in every invocation, and
its "FIXED order so a level change is monotone and reproducible" comment does not hold. Mutation
here would be unreadable through that noise -- a level that moved down could come back with a
different set of tools at the same size -- so this module does not inherit the bug.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics
import zlib

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SURF = f"{R}/surface/gate_surface/work/surfaces"


def scenario_tools(scen: str) -> list[str]:
    p = os.path.join(SURF, f"{scen}.json")
    if not os.path.exists(p):
        return []
    return [t["name"] for t in json.load(open(p)).get("tools", [])]


def group_regret(rewards: list[float], mode: str) -> float:
    """Positive value loss with GRPO's group-relative baseline. See the module docstring."""
    n = len(rewards)
    if n < 2:
        return 0.0
    mu = sum(rewards) / n
    if mode == "success-variance":
        return math.sqrt(max(mu * (1.0 - mu), 0.0))
    var = sum((r - mu) ** 2 for r in rewards) / n
    sd = math.sqrt(var)
    if sd <= 1e-9:
        return 0.0                      # degenerate group: no advantage, hence no regret signal
    return sum(max(r - mu, 0.0) for r in rewards) / (n * sd)


def rank_probs(order: list[str], beta: float) -> dict[str, float]:
    """PLR rank prioritisation: P(i) ∝ (1/rank_i)^(1/beta), rank 1 = highest score."""
    w = {s: (1.0 / (i + 1)) ** (1.0 / max(beta, 1e-6)) for i, s in enumerate(order)}
    z = sum(w.values()) or 1.0
    return {s: v / z for s, v in w.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--pool", default=f"{R}/surface/gate_caller/pools/pool_max.json")
    ap.add_argument("--init-surface", required=True, help="the handicapped starting surface")
    ap.add_argument("--levels", required=True, help="scenario -> restoration level in [0,1]")
    ap.add_argument("--out-map", required=True, help="scenario -> advertised tool names")
    ap.add_argument("--state", required=True, help="archive + staleness, so a restart resumes")
    ap.add_argument("--regret", choices=["advantage", "success-variance"], default="advantage")
    ap.add_argument("--step", type=float, default=0.25, help="mutation size; == ATSC's step")
    ap.add_argument("--edit-frac", type=float, default=0.5,
                    help="fraction of lineages edited per cycle. NOT 1.0 -- editing everything "
                         "would delete the prioritisation that defines the method -- but not the "
                         "0.05-0.25 of the paper either: ATSC re-tunes all 158 scenarios every "
                         "cycle and this arm gets only ~15 cycles, so a small edit budget would "
                         "lose on mutation count rather than on search rule. At 0.5 the archive "
                         "still freezes half the lineages each cycle while the per-cycle edit "
                         "budget stays within 2x of the method it is measured against.")
    ap.add_argument("--rho", type=float, default=0.3, help="staleness mixing coefficient (PLR)")
    ap.add_argument("--beta", type=float, default=0.3, help="rank-prioritisation temperature")
    ap.add_argument("--min-eps", type=int, default=2,
                    help="rollouts of a task needed before its regret is trusted")
    ap.add_argument("--window", type=int, default=4000, help="most recent episodes to trust")
    ap.add_argument("--seed", type=int, default=0, help="shifts the mutation stream per replicate")
    ap.add_argument("--k", type=int, default=5, help="group size, for the availability readout")
    a = ap.parse_args()

    pool = json.load(open(a.pool))
    scens = sorted({p["scenario"] for p in pool})
    init = {ln.strip() for ln in open(a.init_surface) if ln.strip()}

    rows = []
    if os.path.exists(a.episodes):
        for line in open(a.episodes, errors="ignore"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    rows = rows[-a.window:]

    st = {}
    if os.path.exists(a.state):
        try:
            st = json.load(open(a.state))
        except Exception:
            st = {}
    cycle = int(st.get("cycle", 0)) + 1
    archive: dict = st.get("archive", {})        # scenario -> {level, regret, cycle}
    seen: dict = st.get("seen", {})              # scenario -> cycle it was last measured
    levels = json.load(open(a.levels)) if os.path.exists(a.levels) else {}
    for sc in scens:
        levels.setdefault(sc, 0.5)               # every lineage starts mid-dial, as ATSC does
    rng = random.Random(1000 * a.seed + cycle)

    # ---- 1. EVALUATE: regret of each scenario's current level -----------------------------
    by_task: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in rows:
        try:
            by_task[(r["scenario"], int(r["task_idx"]))].append(float(r["reward"]))
        except Exception:
            pass
    per_scen: dict[str, list[float]] = collections.defaultdict(list)
    solve: dict[str, list[float]] = collections.defaultdict(list)
    for (sc, _ti), rs in by_task.items():
        if len(rs) < a.min_eps:
            continue
        per_scen[sc].append(group_regret(rs, a.regret))
        solve[sc].append(sum(rs) / len(rs))
    regret = {sc: sum(v) / len(v) for sc, v in per_scen.items() if v}
    for sc in regret:
        seen[sc] = cycle

    # ---- 2/3. ADMIT and SURVIVE ------------------------------------------------------------
    # Median over the POSITIVE regrets, not over all of them. 68% of tasks sit at p=0, their
    # groups are degenerate, and their regret is exactly 0 -- so the plain median is 0.0 and the
    # threshold admits every lineage including the ones that produced no gradient at all,
    # which makes the archive a list of everything and the admission step a no-op. Measured on
    # a8F's banked episodes at this module's own --window 4000: 158 scenarios measured, only 78
    # with regret > 0, plain median 0.0000, positive-only median 0.1225, mean 0.0733. At the
    # per-task level only 17.2% of 653 groups carry any regret at all, which is the same
    # bimodality ATSC exists to attack, seen through this estimator.
    pos = [g for g in regret.values() if g > 0.0]
    thresh = statistics.median(pos) if pos else 0.0
    admitted = reverted = 0
    for sc, g in regret.items():
        prev = archive.get(sc)
        if prev is None:
            # ADMIT: a lineage with no archived parent seeds one only if it clears the threshold.
            # A level nobody can learn from is not worth preserving as something to mutate FROM.
            if g >= thresh:
                archive[sc] = {"level": levels[sc], "regret": g, "cycle": cycle}
                admitted += 1
        elif g >= float(prev["regret"]):
            # the child is at least as good as its parent: it becomes the new parent
            archive[sc] = {"level": levels[sc], "regret": g, "cycle": cycle}
            admitted += 1
        else:
            # SURVIVE: the mutation made this lineage worse -- discard the child, restore the
            # parent. Without this step the archive is decoration and the search is a random walk.
            levels[sc] = float(prev["level"])
            reverted += 1

    # ---- 4. SELECT which lineages to edit ---------------------------------------------------
    stale = {sc: float(cycle - int(seen.get(sc, 0))) for sc in scens}
    max_stale = max(stale.values()) or 1.0
    order = sorted(scens, key=lambda s: (-regret.get(s, 0.0), s))
    p_reg = rank_probs(order, a.beta)
    p_stale = {sc: stale[sc] / max_stale for sc in scens}
    zs = sum(p_stale.values()) or 1.0
    prio = {sc: (1.0 - a.rho) * p_reg[sc] + a.rho * (p_stale[sc] / zs) for sc in scens}

    n_edit = max(1, int(round(a.edit_frac * len(scens))))
    cand, weights = list(scens), [max(prio[s], 1e-12) for s in scens]
    chosen: list[str] = []
    for _ in range(min(n_edit, len(cand))):
        tot = sum(weights)
        x = rng.random() * tot
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if acc >= x:
                chosen.append(cand[i])
                weights[i] = 0.0
                break

    # ---- 5. EDIT: an UNDIRECTED mutation of the selected lineages ----------------------------
    up = down = 0
    for sc in chosen:
        u = 1.0 if rng.random() < 0.5 else -1.0
        new = min(1.0, max(0.0, levels[sc] + a.step * u))
        if new > levels[sc]:
            up += 1
        elif new < levels[sc]:
            down += 1
        levels[sc] = new
    json.dump(levels, open(a.levels, "w"), indent=1)
    json.dump({"cycle": cycle, "archive": archive, "seen": seen,
               "regret": regret, "threshold": thresh}, open(a.state, "w"), indent=1)

    # ---- realise the levels as an advertised set per scenario --------------------------------
    amap, tot_adv, tot_all = {}, 0, 0
    for sc in scens:
        full = scenario_tools(sc)
        held_back = sorted(set(full) - init)
        random.Random(zlib.crc32(sc.encode()) & 0xFFFFFFFF).shuffle(held_back)
        n = int(round(levels[sc] * len(held_back)))
        adv = sorted((set(full) & init) | set(held_back[:n]))
        amap[sc] = adv
        tot_adv += len(adv)
        tot_all += len(full)
    json.dump(amap, open(a.out_map, "w"))

    mean_reg = sum(regret.values()) / len(regret) if regret else float("nan")
    avail = (sum(1 - p ** a.k - (1 - p) ** a.k
                 for v in solve.values() for p in v)
             / max(sum(len(v) for v in solve.values()), 1)) if solve else float("nan")
    print(f"[accel] cycle {cycle} | measured {len(regret)}/{len(scens)} scenarios "
          f"({len(pos)} with regret>0) | "
          f"mean regret {mean_reg:.4f} (thresh {thresh:.4f}) | archive {len(archive)} "
          f"({admitted} admitted, {reverted} reverted) | edited {len(chosen)} "
          f"({up} up, {down} down) | advertising {tot_adv}/{tot_all} names "
          f"| observed availability {100 * avail:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
