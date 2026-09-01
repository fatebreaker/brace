"""A sequential crossover gate: decide edit retention with as few episodes as the data allows.

MOTIVATION. Our own budget result says a fixed-n gate with adequate power needs 160-2560 tasks
per decision where these loops spend ~40. That makes the recommendation "add a valid test"
correct but unaffordable, which is a poor place for a paper to stop.

Most edits, however, are not borderline. A tool removal that breaks half the tasks does not need
2560 tasks to detect; a null edit does not need 2560 tasks to reject. Fixed-n spends the
worst-case budget on every decision. A sequential design spends it only on the hard ones.

THE RULE. Evaluate in blocks of `block` tasks. After each block, compute the sign-flip
permutation p-value on all tasks seen so far, and

    retain   if p <= alpha_k        (efficacy boundary)
    reject   if p >= beta_k         (futility boundary)
    continue otherwise, up to `max_tasks`

with alpha_k an error-spending boundary over looks. Because we test repeatedly, a naive
"p <= 0.025 at any look" inflates the false-positive rate; the boundary must be tightened per
look. Rather than assume a Lan-DeMets/O'Brien-Fleming form is valid for our zero-inflated
discrete differences -- exactly the assumption failure documented for the t-test -- we CALIBRATE
the boundary empirically on no-edit replicates, then verify on held-out replicates.

WHAT IS MEASURED
    calibration   realised false-positive rate under a genuine null (arms differ only by seed)
    power         detection rate against injected effects of known size
    cost          mean tasks consumed per decision, versus fixed-n at matched power

All of it runs on banked episodes. No GPU, no new rollouts.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCP = os.path.dirname(ROOT)
CALLERS = ["qwen", "granite", "mistral", "falcon3"]


# ------------------------------------------------------------------ data
def load(paths):
    """Merge one or more episode files. A sequential design needs a task pool deeper than one
    block, so the 40-task band-enriched null is unusable here: it is exhausted at the first
    look. We use the 320-task pool, sharded across three files."""
    by = collections.defaultdict(list)
    got = False
    for work in paths:
        p = os.path.join(work, "null_full.jsonl")
        if not os.path.exists(p):
            continue
        got = True
        for r in (json.loads(l) for l in open(p)):
            by[(r["scenario"], r["task_idx"])].append((str(r["seed"]), int(r["reward"])))
    return by if got else None


def nondeg(by, grp):
    out = {}
    for t, v in by.items():
        x = [r for s, r in v if s in grp]
        if len(x) == len(grp):
            out[t] = 0.0 if (sum(x) == 0 or sum(x) == len(x)) else 1.0
    return out


def inject(v, d, rng):
    if d <= 0:
        return v
    v = v.copy()
    z = np.flatnonzero(v < 0.5)
    k = int(round(d * len(v)))
    if k > 0 and len(z):
        v[rng.choice(z, size=min(k, len(z)), replace=False)] = 1.0
    return v


def p_signflip(d, rng, B):
    """One-sided sign-flip permutation p-value. Exact in distribution for any n."""
    if len(d) < 8 or np.all(np.abs(d) < 1e-12):
        return 1.0
    obs = d.mean()
    null = (rng.choice([-1.0, 1.0], size=(B, len(d))) * d).mean(axis=1)
    return float(((null >= obs).sum() + 1) / (B + 1))


# ------------------------------------------------------------------ the gate
def sequential_decision(diff, rng, block, max_tasks, alpha_look, beta_look, perms):
    """Walk the difference vector in blocks. -> (decision, tasks_used)

    decision: +1 retain, 0 reject (futility or exhaustion).
    """
    n = min(len(diff), max_tasks)
    looks = list(range(block, n + 1, block))
    if not looks or looks[-1] != n:
        looks.append(n)
    for i, k in enumerate(looks):
        p = p_signflip(diff[:k], rng, perms)
        a = alpha_look[min(i, len(alpha_look) - 1)]
        b = beta_look[min(i, len(beta_look) - 1)]
        if p <= a:
            return 1, k
        if p >= b:
            return 0, k
    return 0, n


def fixed_decision(diff, rng, n, alpha, perms):
    k = min(len(diff), n)
    return (1 if p_signflip(diff[:k], rng, perms) <= alpha else 0), k


# ------------------------------------------------------------------ trial machinery
def draw(by, have, group, delta, rng, pool_cap):
    """One (incumbent, candidate) comparison. Returns the per-task difference vector."""
    pm = list(have); rng.shuffle(pm)
    A, B = set(pm[:group]), set(pm[group:2 * group])
    pa, pb = nondeg(by, A), nondeg(by, B)
    common = sorted(set(pa) & set(pb))
    if len(common) < 16:
        return None
    idx = rng.permutation(len(common))[:pool_cap]
    xa = np.array([pa[common[i]] for i in idx], float)
    xb = inject(np.array([pb[common[i]] for i in idx], float), delta, rng)
    return xb - xa


def run(by, have, group, deltas, trials, rng, block, max_tasks, alpha_look, beta_look,
        perms, fixed_n, alpha):
    hit = collections.defaultdict(int)
    cost = collections.defaultdict(list)
    n_ok = collections.defaultdict(int)
    for d in deltas:
        for _ in range(trials):
            diff = draw(by, have, group, d, rng, max_tasks)
            if diff is None:
                continue
            n_ok[d] += 1
            dec, used = sequential_decision(diff, rng, block, max_tasks, alpha_look,
                                            beta_look, perms)
            hit[("seq", d)] += dec
            cost[("seq", d)].append(used)
            dec2, used2 = fixed_decision(diff, rng, fixed_n, alpha, perms)
            hit[("fix", d)] += dec2
            cost[("fix", d)].append(used2)
    return hit, cost, n_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--block", type=int, default=40)
    ap.add_argument("--max-tasks", type=int, default=280)
    ap.add_argument("--fixed-n", type=int, default=280)
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--perms", type=int, default=399)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--calib-trials", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(ROOT, "receipts", "sequential_gate.json"))
    a = ap.parse_args()

    deltas = [0.0, 0.05, 0.10, 0.20]
    # Calibrate on qwen+granite, verify on mistral+falcon3: the boundary must not be tuned
    # on the same data that reports its false-positive rate.
    CAL, VER = ["qwen"], ["granite", "mistral"]
    data = {}
    BIG = {"qwen": [f"{MCP}/work/bignull_s{i}" for i in range(3)],
           "granite": [f"{MCP}/work/bignull_granite_s{i}" for i in range(3)],
           "mistral": [f"{MCP}/work/bignull_mistral_s{i}" for i in range(3)]}
    for c in CALLERS:
        if c not in BIG:
            continue
        by = load(BIG[c])
        if by is None:
            continue
        have = sorted({s for v in by.values() for s, _ in v})
        if len(have) >= 2 * a.group:
            data[c] = (by, have)
    CAL=[c for c in CAL if c in data]; VER=[c for c in VER if c in data]
    if not CAL:
        print("  calibration callers unavailable"); return 1

    n_looks = max(1, a.max_tasks // a.block)
    rng = np.random.default_rng(31)

    # ---- calibrate the efficacy boundary: scale a constant boundary until the realised
    #      false-positive rate on CAL data hits alpha. Futility is fixed and generous.
    print(f"  Calibrating a {n_looks}-look sequential boundary on {CAL} "
          f"(block={a.block}, max={a.max_tasks})")
    beta_look = [0.90] * n_looks
    lo, hi = 0.0005, a.alpha
    for _ in range(9):
        mid = (lo + hi) / 2
        al = [mid] * n_looks
        fp = []
        for c in CAL:
            by, have = data[c]
            hit, _, nok = run(by, have, a.group, [0.0], a.calib_trials // 2, rng, a.block,
                              a.max_tasks, al, beta_look, a.perms, a.fixed_n, a.alpha)
            if nok[0.0]:
                fp.append(hit[("seq", 0.0)] / nok[0.0])
        f = float(np.mean(fp)) if fp else 1.0
        if f > a.alpha:
            hi = mid
        else:
            lo = mid
    alpha_look = [lo] * n_looks
    print(f"    per-look threshold {lo:.5f}  (naive would use {a.alpha}); "
          f"futility at p>={beta_look[0]}")

    # ---- verify on held-out callers
    res = {"block": a.block, "max_tasks": a.max_tasks, "per_look_alpha": lo,
           "futility": beta_look[0], "fixed_n": a.fixed_n, "calibrated_on": CAL}
    print(f"\n  Held-out verification on {[c for c in VER if c in data]}")
    print(f"  {'delta':>6s} {'seq power':>10s} {'fix power':>10s} {'seq tasks':>10s} "
          f"{'fix tasks':>10s} {'saving':>8s}")
    print("  " + "-" * 60)
    for c in VER:
        if c not in data:
            continue
        by, have = data[c]
        hit, cost, nok = run(by, have, a.group, deltas, a.trials, rng, a.block, a.max_tasks,
                             alpha_look, beta_look, a.perms, a.fixed_n, a.alpha)
        res[c] = {}
        print(f"    --- {c} ---")
        for d in deltas:
            if not nok[d]:
                continue
            sp, fp = hit[("seq", d)] / nok[d], hit[("fix", d)] / nok[d]
            sc, fc = float(np.mean(cost[("seq", d)])), float(np.mean(cost[("fix", d)]))
            tag = "  <- null" if d == 0 else ""
            print(f"  {d:6.2f} {sp:10.3f} {fp:10.3f} {sc:10.1f} {fc:10.1f} "
                  f"{1 - sc / max(fc, 1e-9):7.0%}{tag}")
            res[c][f"{d:.2f}"] = {"seq_rate": sp, "fixed_rate": fp,
                                  "seq_tasks": sc, "fixed_tasks": fc}
    print(f"\n  A sequential gate is only usable if the null row holds at or below "
          f"{a.alpha} on data\n  it was not calibrated on, while retaining power and "
          f"spending fewer tasks.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
