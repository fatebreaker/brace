"""Positive control: does paired evaluation still detect REAL edits, and at what budget?

Our measured false-positive story is only half a result. A gate that rejects everything
trivially achieves a low false-positive rate and is useless. This measures the other half --
detection power against edits of KNOWN effect size -- for the paired (crossover / common
random numbers) design versus the unpaired one the field uses, at MATCHED episode budget.

DESIGN. Semi-synthetic, built on the real G-LANDSCAPE cube so the nuisance-variance structure
is real rather than assumed:

  arm A   ONE interface, seeds {0,1,2,3}, aggregated per task
  arm B   the SAME interface, seeds {4,5,6,7}, per task, plus a synthetic known effect delta

CORRECTION (2026-08-01). The first version compared two DIFFERENT interfaces at delta=0 and
called that the null. It is not: MACRO is measurably worse than SPECIALIZED (0.367 vs 0.457),
so "delta=0" carried real effects up to ~9 points. The observed false-positive rates were
0.213 paired / 0.157 unpaired against a nominal alpha of 0.05 -- the calibration check caught
it. Comparing one interface against ITSELF across disjoint seed halves is a genuine null: the
two arms differ only by sampling noise, so any rejection at delta=0 is a false positive by
construction. The pairing unit is then the TASK, which is the nuisance factor shared between
arms and the analogue of "same task, same initial state" in the real setting.

  matched budget: the paired design spends 2n episodes (n units x 2 arms); the unpaired design
                  spends the same 2n episodes (n per arm, disjoint units). Equal cost.
  paired test    one-sample t on per-unit differences
  unpaired test  Welch two-sample t on independent unit sets
  reported       false-positive rate at delta = 0 (calibration) and detection power at each
                 delta > 0, plus the minimum detectable effect at 80% power.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IFACES = ["RAW", "SELECTED", "SPECIALIZED", "MACRO"]
CALLERS = ["qwen", "granite"]   # headroom_all: 800 tasks, RAW only, 8 seeds
ALPHA = 0.05


def load_units(work, caller):
    """-> dict iface -> array [n_tasks, 2]: per-task mean over seed half 0 and half 1."""
    rec = collections.defaultdict(lambda: collections.defaultdict(dict))
    p = os.path.join(work, f"{caller}_sampled.jsonl")
    seeds = set()
    for r in (json.loads(l) for l in open(p)):
        rec[r["interface"]][(r["scenario"], r["task_idx"])][r["seed"]] = int(r["reward"])
        seeds.add(r["seed"])
    sl = sorted(seeds); h = len(sl) // 2
    lo, hi = set(sl[:h]), set(sl[h:])
    out = {}
    for iface, tasks in rec.items():
        rows = []
        for t, sv in tasks.items():
            if len(sv) != len(sl):
                continue
            rows.append([np.mean([sv[s] for s in lo]), np.mean([sv[s] for s in hi])])
        if rows:
            out[iface] = np.array(rows, dtype=float)
    return out


def inject(x, delta, rng):
    """Raise this arm's per-task mean by `delta`, capped at 1.0 (rates, not 0/1 now)."""
    if delta <= 0:
        return x.copy()
    return np.minimum(x + delta, 1.0)


def t_paired(d):
    n = len(d)
    s = d.std(ddof=1)
    return 0.0 if s < 1e-12 else d.mean() / (s / np.sqrt(n))


def t_welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    den = np.sqrt(va + vb)
    return 0.0 if den < 1e-12 else (a.mean() - b.mean()) / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "..", "work", "headroom_all"))
    ap.add_argument("--n", type=int, default=120, help="paired units; budget = 2n episodes")
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(ROOT, "receipts", "pairing_power.json"))
    a = ap.parse_args()

    work = os.path.abspath(a.work)
    pool = []
    for c in CALLERS:
        if os.path.exists(os.path.join(work, f"{c}_sampled.jsonl")):
            byif = load_units(work, c)
            if byif and min(len(v) for v in byif.values()) >= 2 * a.n + 5:
                pool.append((c, byif))
    if not pool:
        print("  no caller has enough complete units")
        return 1

    deltas = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    # 1.96 is the large-sample two-sided 5% critical value; n>=120 here so the t/z gap is <1%.
    crit = 1.96
    rng = np.random.default_rng(0)

    print(f"  budget = 2n = {2*a.n} episodes per comparison; alpha = {ALPHA}; "
          f"{a.trials} trials\n")
    print(f"  {'delta':>6s} {'paired':>9s} {'unpaired':>10s} {'gain':>8s}")
    print("  " + "-" * 37)
    res = {}
    for d in deltas:
        hp = hu = 0
        for _ in range(a.trials):
            c, byif = pool[rng.integers(len(pool))]
            ks = list(byif)
            u = byif[ks[rng.integers(len(ks))]]             # ONE interface: a true null
            # --- paired: same tasks, arm A = seed half 0, arm B = seed half 1 (+delta)
            idx = rng.choice(len(u), size=a.n, replace=False)
            xa, xb = u[idx, 0], inject(u[idx, 1], d, rng)
            hp += abs(t_paired(xb - xa)) > crit
            # --- unpaired: same episode budget, DISJOINT tasks per arm
            jdx = rng.choice(len(u), size=2 * a.n, replace=False)
            ya = u[jdx[:a.n], 0]
            yb = inject(u[jdx[a.n:], 1], d, rng)
            hu += abs(t_welch(yb, ya)) > crit
        pp, pu = hp / a.trials, hu / a.trials
        lab = "  (FP rate)" if d == 0 else ""
        print(f"  {d:6.2f} {pp:9.3f} {pu:10.3f} {pp-pu:+8.3f}{lab}")
        res[f"{d:.2f}"] = {"paired": pp, "unpaired": pu}

    def mde(key):
        prev_d, prev_p = 0.0, res["0.00"][key]
        for d in deltas[1:]:
            p = res[f"{d:.2f}"][key]
            if p >= 0.80:
                if p == prev_p:
                    return d
                return prev_d + (0.80 - prev_p) * (d - prev_d) / (p - prev_p)
            prev_d, prev_p = d, p
        return float("nan")

    mp, mu = mde("paired"), mde("unpaired")
    print("\n  minimum detectable effect at 80% power:")
    print(f"    paired    {mp:.3f}")
    print(f"    unpaired  {mu:.3f}")
    if mp == mp and mu == mu and mp > 0:
        print(f"    paired detects effects {mu/mp:.2f}x smaller at the same budget")
    print(f"\n  Calibration check: both procedures should sit near alpha={ALPHA} at delta=0.")
    print(f"  Power is what decides whether the variance reduction (rho=0.346, 1.5x fewer")
    print(f"  episodes) is worth the extra machinery.")

    res["n_units"] = a.n
    res["mde_paired"], res["mde_unpaired"] = mp, mu
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, sort_keys=True, default=float)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
