"""ACCORD -- certify a shaped reward by its CONCORDANCE with solving.

Shape the reward from the task's own verifier, but only where that shaping is CERTIFIED to point
toward solving. Refuse it everywhere else, and keep the binary reward there.

THE NAME IS THE STATISTIC, and it is the other half of \textsc{Discord}'s. \textsc{Discord}
decides whether a harness edit helped by counting the pairs on which incumbent and candidate
DISAGREE -- McNemar's exact conditional test is defined on precisely those discordant pairs.
\textsc{Accord} decides whether a reward signal tracks the objective by counting the pairs on
which progress and solvability AGREE: the AUC below is the concordance index, accumulated over
solvable x dead task pairs. Same paired evaluation data, same refusal to spend on an effect that
does not clear its own noise, opposite side of the same coin:

    Discord   refuses an edit whose outcomes are not DISCORDANT enough to show it helped.
    Accord    refuses a reward whose progress is not CONCORDANT enough with solving.

-------------------------------------------------------------------------------------------------
WHY THE REWARD, AND NOT THE TOOL SURFACE

This project's thesis is that an MCP environment is EDITABLE: the harness is a design choice, not
a property of the task. \textsc{Discord} applied that to the advertised tool set and certified the
edits. ELSA applied it to the tool surface as a training-time dial. Measured over 45 GRPO steps on
a8A3/a8Az0, the tool surface turned out to have almost no authority over the gradient:

    slope of availability vs surface level: distribution centred at ZERO, symmetric,
    only 20 of 99 eligible scenarios crediting any positive slope even with the noise floor off.

The reason is visible in the substrate rather than the controller. Across six live arms:

    never solved (p=0)   63.5% - 78.0%      <- k identical zero rewards, advantage identically 0
    always solved (p=1)  12.3% - 17.5%      <- k identical ones, same
    gradient availability          5.0% - 17.5%
    actor/grad_norm                0.006 - 0.031

~88% of tasks cannot produce a gradient at all, and the optimizer confirms it. No tool has ever
been the reason those tasks fail; they fail on multi-step reasoning and verifier strictness. A
lever that reallocates rollouts among tasks -- or advertises more tools to them -- is redistributing
a gradient that mostly does not exist.

What DOES exist is unused information: the verifier already knows how far the episode got. In AWM
the verifiers are sequential guard clauses -- a chain of `if <precondition fails>: return others`
ending in `return complete` -- so the index of the guard that fired is progress, produced by the
same oracle that defines success. That oracle is part of the harness, which makes editing it the
same move the paper already makes, applied to the axis that responds.

-------------------------------------------------------------------------------------------------
WHY IT MUST BE CERTIFIED, WITH THE COUNTEREXAMPLE ALREADY MEASURED

Dense reward is not free, and the failure mode is not hypothetical. Shaping from process telemetry
(tool calls, errors, response length) raises availability 10.0% -> 70.6% at k=5 -- a 7x gain that
is entirely fake. The within-task AUC of those counters for predicting actual success, over
555,923 solved/unsolved pairs:

    emitted a final answer 0.590 | error rate 0.540 | response tokens 0.525
    fewer parse failures   0.485 | more tool calls 0.479        <- BELOW chance

0.50 is noise. Any near-continuous nuisance variable manufactures availability the same way:
response length alone takes 6,004 distinct values among failures, enough to drive availability to
~100% while teaching the policy only to change its output length. Availability is therefore not a
safe objective on its own -- it is trivially gameable by shaping on anything continuous.

So the shaping signal has to earn its use, and the test has to be the right one. The obvious test
is tautological: gate_progress equals exactly 1.000 on every solved episode, so "solved outranks
unsolved" has AUC 1.000 by construction and certifies nothing. The question that matters is
whether progress discriminates AMONG FAILURES in the direction of solvability:

    CERTIFICATION STATISTIC
        over tasks with at least one failed rollout, compare mean failure progress on tasks that
        are sometimes solvable (p > 0) against tasks never solved (p = 0):

            A = AUC( mean failure progress  ->  task is solvable )

        certify iff  A - z*se(A) > 0.5 + margin

Measured on b8dense's banked episodes: A = 0.679, se 0.017, n = 714 pairs -- about ten standard
errors above chance, and far above the best telemetry counter (0.590). Failures on solvable tasks
reach 0.564 progress; failures on dead tasks reach 0.405. The signal is real, and the same test
would have rejected every telemetry counter.

-------------------------------------------------------------------------------------------------
PER-SCENARIO, BECAUSE THE ASSUMPTION IS PER-SCENARIO

dense_reward.py states its own weak point: guard index in SOURCE ORDER is taken as progress order,
which holds for straight-line guard clauses but scores a verifier backwards if it checks the
expensive condition first. That is not a property of the pool, it is a property of each scenario's
verifier -- so it is certified per scenario, from that scenario's own rollouts, and re-certified
every cycle as evidence accumulates. A scenario whose verifier is written backwards fails its own
test and keeps the binary reward; nothing global has to be assumed or hand-audited.

Scenarios with too little evidence fall back to the POOL-level certificate rather than to a guess,
and the pool certificate is computed the same way. Hierarchy, not default-on: shaping is off until
something measurable turns it on.

-------------------------------------------------------------------------------------------------
WHAT THIS DOES NOT DO

It never touches the binary reward that evaluation reads. dense_reward.py already shapes only
`AgentLoopOutput.reward_score` while `episodes.jsonl` keeps the true 0/1 under `reward`; ACCORD only
decides WHERE that shaping is allowed to apply. Every held-out measurement, every eval cell, and
every analysis in the paper continues to read the unshaped binary outcome, so a certified-but-wrong
shaping can cost training efficiency and can never manufacture a result.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def auc_and_se(pos, neg):
    """AUC of `pos` over `neg` by exhaustive pair counting, with its binomial standard error.

    Exhaustive rather than rank-sum because the samples here are small (tens of tasks), ties are
    common when progress lands on the same guard, and the 0.5-credit-for-ties convention has to be
    explicit -- a tie is evidence of nothing and must not count as agreement.
    """
    conc = ties = tot = 0
    for a in pos:
        for b in neg:
            tot += 1
            if a > b:
                conc += 1
            elif a == b:
                ties += 1
    if tot == 0:
        return float("nan"), float("inf"), 0
    a = (conc + 0.5 * ties) / tot
    return a, math.sqrt(max(a * (1 - a), 1e-12) / tot), tot


def failure_progress(rows):
    """(mean progress over FAILED rollouts, solve rate) per task.

    Solved episodes are excluded from the progress average on purpose: progress is exactly 1.0
    whenever the task is solved, so including them would make every statistic here a restatement
    of the label instead of a test of it.
    """
    by = collections.defaultdict(list)
    for sc, ti, g, y in rows:
        by[(sc, ti)].append((g, y))
    out = {}
    for k, v in by.items():
        f = [g for g, y in v if y == 0]
        if f:
            out[k] = (sum(f) / len(f), sum(y for _, y in v) / len(v))
    return out


def certify(rows, z, margin, min_pairs):
    """Certify a set of episodes: does failure progress rank solvable tasks above dead ones?"""
    fp = failure_progress(rows)
    live = [m for m, p in fp.values() if p > 0]
    dead = [m for m, p in fp.values() if p == 0]
    a, se, n = auc_and_se(live, dead)
    ok = n >= min_pairs and not math.isnan(a) and (a - z * se) > (0.5 + margin)
    return {"auc": None if math.isnan(a) else round(a, 4), "se": None if se == float("inf") else round(se, 4),
            "pairs": n, "n_live": len(live), "n_dead": len(dead), "certified": bool(ok)}


def contradicts(rows, z, min_units):
    """Does this scenario's OWN evidence say its verifier runs BACKWARDS?

    WHY THIS SHAPE, AND NOT "every scenario proves itself". The pool statistic compares TASKS --
    solvable versus never-solved -- and the pool has 7.1 tasks per scenario, so a single scenario
    can offer at most a handful of solvable x dead task pairs. Measured on a8C's own evidence:
    max 6 pairs, median 0, and NO threshold makes per-scenario judgement reachable (0 of 132 at
    min_pairs=8; grouping into families gives 3 of 85). Requiring each scenario to earn its own
    certificate therefore certifies nothing on its own evidence and every scenario silently
    inherits the pool -- which makes the arm identical to shaping everywhere, and the ablation
    null for a structural reason rather than a scientific one.
    #
    So invert the burden. The pool certificate is well powered (6,156 pairs, AUC 0.612, se 0.006,
    ~18 sigma), and it is the sensible default. A scenario is EXCLUDED only when its own evidence
    positively contradicts it -- which is precisely the documented failure mode: dense_reward.py
    assumes guard index in SOURCE ORDER is progress order, and a verifier that tests the expensive
    condition first is scored backwards. Backwards means AUC significantly BELOW chance, so this
    is a one-sided test against 0.5 and needs far less evidence than proving a positive.
    #
    POWER COMES FROM ROLLOUTS, HONESTY FROM TASKS. The point estimate counts rollout-level pairs
    (every failed rollout on a solvable task against every failed rollout on a dead one), which is
    an order of magnitude more data. The standard error is computed from the number of independent
    TASK pairs instead, because rollouts of one task are not independent draws -- using the rollout
    count there would overstate power by ~sqrt(k) and start excluding scenarios on noise.
    """
    by = collections.defaultdict(list)
    for sc, ti, g, y in rows:
        by[(sc, ti)].append((g, y))
    live_r, dead_r, live_t, dead_t = [], [], 0, 0
    for v in by.values():
        f = [g for g, y in v if y == 0]
        if not f:
            continue
        if sum(y for _, y in v) > 0:
            live_r += f; live_t += 1
        else:
            dead_r += f; dead_t += 1
    n_units = live_t * dead_t
    if n_units < min_units:
        return None                      # not enough independent tasks to contradict anything
    a, _, n = auc_and_se(live_r, dead_r)
    if math.isnan(a):
        return None
    se = math.sqrt(max(a * (1 - a), 1e-12) / n_units)     # task-level, deliberately conservative
    return {"auc": round(a, 4), "se": round(se, 4), "rollout_pairs": n, "task_pairs": n_units,
            "backwards": bool((a + z * se) < 0.5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True, help="certificate consumed by dense_reward.py")
    ap.add_argument("--z", type=float, default=1.64,
                    help="one-sided normal quantile the AUC must clear after subtracting its own "
                         "standard error. 1.64 is 95%%; this is the same refusal rule Discord "
                         "applies to harness edits and ELSA applies to elasticity, so a shaping "
                         "that cannot be resolved from the evidence is simply not used.")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="how far above chance the AUC must sit before shaping is allowed. Guards "
                         "against a signal that is statistically detectable but practically "
                         "useless; the telemetry counters (0.479-0.590) sit in exactly that band.")
    ap.add_argument("--min-pairs", type=int, default=40,
                    help="minimum live x dead task pairs before a scenario may be certified on its "
                         "own evidence; below this it inherits the pool certificate")
    ap.add_argument("--min-units", type=int, default=2,
                    help="independent solvable x dead TASK pairs a scenario needs before its own "
                         "evidence may exclude it. 2 is low on purpose: the test is one-sided "
                         "against chance and the standard error is computed from these units, so "
                         "thin evidence produces a wide interval and simply fails to exclude.")
    ap.add_argument("--window", type=int, default=40000)
    a = ap.parse_args()

    def read(path):
        out = []
        if not path or not os.path.exists(path):
            return out
        with open(path, errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if "gate_progress" in d and "reward" in d:
                    try:
                        out.append((d.get("scenario"), int(d["task_idx"]),
                                    float(d["gate_progress"]), int(d["reward"] > 0)))
                    except Exception:
                        continue
        return out

    # SEED CORPUS. The certificate asks a question about the VERIFIERS -- does guard order track
    # progress? -- not about the policy, so rollouts from any run of the same harness are valid
    # evidence for it. That matters because the certifying statistic needs tasks the policy
    # sometimes solves, and at ~8% availability those accrue slowly: a8C's second cycle had ONE
    # solvable task against 27 dead, far under the pair floor. Without a seed the arm would train
    # on the binary reward for many cycles and the method would never engage -- a null result for a
    # bookkeeping reason rather than a scientific one.
    #
    # The seed is an explicit file the operator places next to the certificate, never a silent
    # default, and it is reported separately in the diagnostic so a certificate is always
    # attributable to the evidence that produced it.
    rows = read(a.episodes)[-a.window:]
    seed_path = os.path.join(os.path.dirname(a.out) or ".", "accord_seed.jsonl")
    seed = read(seed_path)
    if seed:
        rows = seed + rows

    pool = certify(rows, a.z, a.margin, a.min_pairs)

    by_sc = collections.defaultdict(list)
    for r in rows:
        by_sc[r[0]].append(r)
    scen = {}
    own = inherited = refused = excluded = 0
    for sc, rs in by_sc.items():
        c = certify(rs, a.z, a.margin, a.min_pairs)
        if c["pairs"] >= a.min_pairs:
            scen[sc] = c
            own += 1
            if not c["certified"]:
                refused += 1
            continue
        # Not enough of its own task-level evidence to earn a certificate. Inherit the pool --
        # UNLESS this scenario's rollouts positively say its verifier is backwards, in which case
        # exclude it however strong the pool is. This is the only per-scenario test the data can
        # actually support; see contradicts().
        ctr = contradicts(rs, a.z, a.min_units)
        allow = bool(pool["certified"])
        if ctr and ctr["backwards"]:
            allow = False
            excluded += 1
        scen[sc] = dict(c, certified=allow, inherited=True,
                        own_check=ctr if ctr else None)
        inherited += 1

    cert = {"pool": pool, "scenarios": scen,
            "allow": sorted([s for s, c in scen.items() if c["certified"]]),
            "params": {"z": a.z, "margin": a.margin, "min_pairs": a.min_pairs}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(cert, open(a.out, "w"), indent=1)

    if seed:
        print(f"[accord] evidence: {len(rows) - len(seed)} own episodes + {len(seed)} seeded from "
              f"{os.path.basename(seed_path)} (same harness, prior run)", flush=True)
    pa = pool["auc"]
    print(f"[accord] pool AUC {pa if pa is not None else float('nan')} "
          f"(se {pool['se']}, {pool['pairs']} pairs, {pool['n_live']} solvable vs {pool['n_dead']} dead) "
          f"-> pool {'CERTIFIED' if pool['certified'] else 'REFUSED'}", flush=True)
    print(f"[accord] {len(scen)} scenarios: {own} judged on own evidence ({refused} refused), "
          f"{inherited} inherited the pool verdict of which {excluded} EXCLUDED as backwards "
          f"| shaping allowed on {len(cert['allow'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
