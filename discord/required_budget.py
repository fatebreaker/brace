"""How many tasks does a USABLE harness-evolution acceptance gate actually need?

THE DILEMMA THIS ANSWERS. Our two live loops give the two halves of a practical problem:

  no test (current practice)  -> credits provably-inert edits 43% of the time
                                 (receipts/harness_attribution.json)
  a test at the budget these loops actually run
                              -> at 40 tasks x 4 seeds, power against a real +0.10 effect is
                                 0.10-0.16 (receipts/null_power.json). In the handicapped
                                 positive-control loop, a candidate that improved the objective
                                 by 10 points was rejected at t=0.31.

Neither is usable. So the actionable quantity is not "which gate is better" but "what budget
makes ANY gate work", and how much pairing reduces it.

METHOD. Purely empirical -- no normal approximation, which §4 showed is exactly what fails on
this objective. For each task-count n we resample n tasks with replacement from the banked null
episodes, inject a known effect, and measure the permutation test's power directly. The smallest
n reaching the target power is the answer.

CAVEAT, stated plainly: resampling with replacement from 40 real tasks cannot manufacture task
diversity a real pool of n tasks would have. Required n is therefore a LOWER BOUND -- a real
task pool that heterogeneous would need at least this many, likely more.
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
# The checkout root: where the study mains look for paired evaluation records, under
# $BRACE_ROOT/work/. Set BRACE_ROOT when the records live elsewhere.
MCP = os.environ.get("BRACE_ROOT", ROOT)
CALLERS = ["qwen", "granite", "mistral", "falcon3"]


def load(work):
    p = os.path.join(work, "null_full.jsonl")
    if not os.path.exists(p):
        return None
    by = collections.defaultdict(list)
    for r in (json.loads(l) for l in open(p)):
        by[(r["scenario"], r["task_idx"])].append((str(r["seed"]), int(r["reward"])))
    return by


def nondeg(by, tasks, grp):
    out = {}
    for t in tasks:
        v = [r for s, r in by.get(t, []) if s in grp]
        if len(v) == len(grp):
            out[t] = 0.0 if (sum(v) == 0 or sum(v) == len(v)) else 1.0
    return out


def inject(v, delta, rng):
    if delta <= 0:
        return v
    v = v.copy()
    z = np.flatnonzero(v < 0.5)
    k = int(round(delta * len(v)))
    if k > 0 and len(z):
        v[rng.choice(z, size=min(k, len(z)), replace=False)] = 1.0
    return v


def power_at(by, tasks, have, grp, n, delta, trials, perms, rng, paired):
    """Directly measured power of the permutation gate at n tasks."""
    hits = ok = 0
    for _ in range(trials):
        pm = list(have); rng.shuffle(pm)
        A, B = set(pm[:grp]), set(pm[grp:2 * grp])
        pa, pb = nondeg(by, tasks, A), nondeg(by, tasks, B)
        common = sorted(set(pa) & set(pb))
        if len(common) < 8:
            continue
        idx = rng.integers(0, len(common), size=n)           # resample n tasks
        xa = np.array([pa[common[i]] for i in idx], float)
        xb = inject(np.array([pb[common[i]] for i in idx], float), delta, rng)
        ok += 1
        if paired:
            d = xb - xa
            obs = d.mean()
            null = (rng.choice([-1.0, 1.0], size=(perms, n)) * d).mean(axis=1)
        else:
            h = n // 2
            a_, b_ = xa[:h], xb[h:]
            obs = b_.mean() - a_.mean()
            pool = np.concatenate([a_, b_])
            sh = pool[np.argsort(rng.random((perms, len(pool))), axis=1)]
            null = sh[:, len(a_):].mean(axis=1) - sh[:, :len(a_)].mean(axis=1)
        hits += (((null >= obs).sum() + 1) / (perms + 1)) <= 0.025
    return hits / max(ok, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--trials", type=int, default=240)
    ap.add_argument("--perms", type=int, default=299)
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--pool", default=os.path.join(HERE, "pools", "pool_bandenriched.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "receipts", "required_budget.json"))
    a = ap.parse_args()

    pool = json.load(open(a.pool))
    tasks = sorted({(p["scenario"], p["task_idx"]) for p in pool})
    grid = [40, 80, 160, 320, 640, 1280, 2560]
    deltas = [0.05, 0.10, 0.20]
    out = {"grid": grid, "target_power": a.target, "seeds_per_eval": a.group}

    for c in CALLERS:
        w = os.path.join(MCP, "work", "evolve_null" if c == "qwen" else f"evolve_null_{c}")
        by = load(w)
        if by is None:
            continue
        have = sorted({s for v in by.values() for s, _ in v})
        if len(have) < 2 * a.group:
            continue
        probe = nondeg(by, tasks, set(have[:a.group]))
        if len(probe) < 8:
            print(f"  {c}: episodes still banking -- skipped", flush=True)
            continue
        rng = np.random.default_rng(17)
        print(f"\n  === {c} ===  tasks needed for {int(100*a.target)}% power "
              f"(permutation gate, one-sided 0.025, {a.group} seeds/eval)")
        print(f"  {'delta':>6s} {'paired':>22s} {'unpaired':>22s} {'ratio':>7s}")
        print("  " + "-" * 60)
        out[c] = {}
        for d in deltas:
            req = {}
            for lab, pr in (("paired", True), ("unpaired", False)):
                need = None
                for n in grid:
                    if power_at(by, tasks, have, a.group, n, d, a.trials, a.perms, rng, pr) >= a.target:
                        need = n
                        break
                req[lab] = need
            fmt = lambda v: (f"{v} tasks ({v*a.group*2} eps)" if v else f">{grid[-1]} tasks")
            ratio = (f"{req['unpaired']/req['paired']:.1f}x"
                     if req["paired"] and req["unpaired"] else "n/a")
            print(f"  {d:6.2f} {fmt(req['paired']):>22s} {fmt(req['unpaired']):>22s} {ratio:>7s}")
            out[c][f"{d:.2f}"] = req

    print(f"\n  Episode cost = tasks x {a.group} seeds x 2 arms, per acceptance decision.")
    print(f"  Our live loops used 40 tasks. Compare that with the column above: a gate at that")
    print(f"  budget is not underpowered by accident, it is underpowered by construction.")
    print(f"  Resampling caveat: required n is a LOWER bound (see module docstring).")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, sort_keys=True, default=str)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
