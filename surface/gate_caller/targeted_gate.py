"""DISCORD-T: discordance-targeted evaluation for harness-edit acceptance.

THE INVENTION. Section 7 establishes that only DISCORDANT pairs -- tasks one arm solves and the
other does not -- carry information about an edit. Concordant pairs contribute exactly nothing to
the conditional test statistic. Yet every acceptance procedure in the literature, and every one in
this paper so far, allocates evaluation UNIFORMLY over the task pool.

That is provably wasteful, and measurably so: on our broad pool 64% of tasks are solved almost
never and 21% almost always, so roughly 85% of episodes buy no information. The expected
information per evaluated task is proportional to its discordance probability pi_d(t), and pi_d
varies by an order of magnitude across tasks.

ALGORITHM. Maintain a per-task estimate pi_hat_d(t) from evaluation history, and for each
acceptance decision spend the episode budget on tasks in decreasing pi_hat_d, stopping when the
curtailed exact test decides.

    1. rank tasks by pi_hat_d(t)                        (from prior rounds; uniform prior at t=0)
    2. evaluate the top block under both arms
    3. run the exact conditional test on all pairs seen so far
    4. retain / reject / continue                       (curtailment as in Section 7)
    5. update pi_hat_d(t) with the observed concordance

VALIDITY. Targeting does not invalidate the test. The exact conditional test conditions on the
realised discordant pairs, and its null -- b ~ Binomial(b+c, 1/2) -- depends only on the arms
being exchangeable WITHIN a pair, which task selection does not affect. What targeting changes is
the ESTIMAND: from "does this edit help on the pool" to "does it help among tasks where it can
act". For an accept/reject decision that is the right question, because concordant tasks are by
construction unaffected by the edit.

WHAT IS MEASURED HERE
    calibration   realised false-positive rate under a genuine null, targeted vs uniform
    power         detection of injected effects at matched EPISODE budget
    cost          episodes to a decision, targeted vs uniform
    cold start    how much history pi_hat_d needs before targeting pays

Runs on banked episodes; no GPU.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCP = os.path.dirname(ROOT)


def load(paths):
    by = collections.defaultdict(list)
    got = False
    for w in paths:
        p = os.path.join(w, "null_full.jsonl")
        if not os.path.exists(p):
            continue
        got = True
        for r in (json.loads(l) for l in open(p)):
            by[(r["scenario"], r["task_idx"])].append((str(r["seed"]), int(r["reward"])))
    return by if got else None


def nondeg_map(by, grp):
    out = {}
    for t, v in by.items():
        x = [r for s, r in v if s in grp]
        if len(x) == len(grp):
            out[t] = 0.0 if (sum(x) == 0 or sum(x) == len(x)) else 1.0
    return out


def p_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def inject_one(v, flip):
    return 1.0 if flip else v


def run_decision(order, pa, pb, delta, rng, block, budget, alpha, targeted):
    """Walk tasks in `order`, evaluating `block` at a time, until decided or budget spent.

    Returns (decision, tasks_used, discordant_seen).
    """
    b = c = used = 0
    idx = 0
    n = len(order)
    while idx < n and used < budget:
        hi = min(idx + block, n, budget)
        for t in order[idx:hi]:
            xa, xb = pa[t], pb[t]
            # injected effect: flip a delta-fraction of the candidate's failures to successes
            if delta > 0 and xb < 0.5 and rng.random() < delta / max(1e-9, 1 - np.mean(list(pb.values()))):
                xb = 1.0
            d = xb - xa
            if d > 0.5:
                b += 1
            elif d < -0.5:
                c += 1
            used += 1
        idx = hi
        if b + c >= 4:
            if p_exact(b, c) <= alpha:
                return 1, used, b + c
            # curtail: even if EVERY remaining task were discordant in our favour, so that b
            # gains rem and c is unchanged, could the test still reach significance?
            rem = min(n - idx, budget - used)
            if p_exact(b + rem, c) > alpha:
                return 0, used, b + c
    return 0, used, b + c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--history", type=int, default=6, help="prior decisions used to estimate pi_d")
    ap.add_argument("--out", default=os.path.join(ROOT, "receipts", "targeted_gate.json"))
    a = ap.parse_args()

    POOLS = {"qwen": [f"{MCP}/work/bignull_s{i}" for i in range(3)],
             "granite": [f"{MCP}/work/bignull_granite_s{i}" for i in range(3)],
             "mistral": [f"{MCP}/work/bignull_mistral_s{i}" for i in range(3)]}
    deltas = [0.0, 0.05, 0.10, 0.20]
    out = {}
    rng = np.random.default_rng(23)

    for caller, paths in POOLS.items():
        by = load(paths)
        if by is None:
            continue
        have = sorted({s for v in by.values() for s, _ in v})
        if len(have) < 2 * a.group:
            continue
        tasks = sorted(by)
        if len(tasks) < 80:
            continue

        # ---- estimate pi_d(t) from `history` prior no-edit decisions (cold start = uniform)
        disc = collections.Counter(); seen = collections.Counter()
        for _ in range(a.history):
            pm = list(have); rng.shuffle(pm)
            pa, pb = nondeg_map(by, set(pm[:a.group])), nondeg_map(by, set(pm[a.group:2 * a.group]))
            for t in set(pa) & set(pb):
                seen[t] += 1
                if abs(pb[t] - pa[t]) > 0.5:
                    disc[t] += 1
        pihat = {t: (disc[t] + 0.5) / (seen[t] + 1.0) for t in tasks}
        ranked = sorted(tasks, key=lambda t: -pihat[t])

        print(f"\n  === {caller} ===  {len(tasks)} tasks; estimated discordance "
              f"top-decile {np.mean([pihat[t] for t in ranked[:len(ranked)//10]]):.3f} "
              f"vs bottom-decile {np.mean([pihat[t] for t in ranked[-len(ranked)//10:]]):.3f}")
        print(f"  {'delta':>6s} {'targeted rate':>14s} {'uniform rate':>13s} "
              f"{'targ tasks':>11s} {'unif tasks':>11s} {'saving':>8s}")
        print("  " + "-" * 68)
        out[caller] = {}
        for d in deltas:
            ht = hu = 0; ct = []; cu = []; ok = 0
            for _ in range(a.trials):
                pm = list(have); rng.shuffle(pm)
                pa, pb = nondeg_map(by, set(pm[:a.group])), nondeg_map(by, set(pm[a.group:2*a.group]))
                common = set(pa) & set(pb)
                if len(common) < 40:
                    continue
                ok += 1
                ordT = [t for t in ranked if t in common]
                ordU = list(common); rng.shuffle(ordU)
                dt, ut, _ = run_decision(ordT, pa, pb, d, rng, a.block, a.budget, a.alpha, True)
                du, uu, _ = run_decision(ordU, pa, pb, d, rng, a.block, a.budget, a.alpha, False)
                ht += dt; ct.append(ut); hu += du; cu.append(uu)
            if not ok:
                continue
            mt, mu = float(np.mean(ct)), float(np.mean(cu))
            tag = "  <- null" if d == 0 else ""
            print(f"  {d:6.2f} {ht/ok:14.3f} {hu/ok:13.3f} {mt:11.1f} {mu:11.1f} "
                  f"{1-mt/max(mu,1e-9):7.0%}{tag}")
            out[caller][f"{d:.2f}"] = {"targeted_rate": ht/ok, "uniform_rate": hu/ok,
                                       "targeted_tasks": mt, "uniform_tasks": mu, "n": ok}
    print(f"\n  Targeting is usable only if the null row stays at or below {a.alpha} while the")
    print(f"  detection rate at delta>0 is preserved and fewer tasks are consumed.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
