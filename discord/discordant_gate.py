"""Exploit the structure the objective actually has: harness-edit acceptance is a paired-binary problem.

THE OBSERVATION. Section 5 of the paper shows the paired t-test fails because per-task differences
live in {-1,0,+1} and are predominantly zero. We treated that as a nuisance to be routed around
with a permutation test. It is better read as a statement about what kind of problem this is:
per-task outcomes are binary, arms are paired, so the data are a 2x2 table of concordant and
discordant pairs, and only the DISCORDANT pairs carry information about the edit.

    b = #tasks the candidate solves and the incumbent does not
    c = #tasks the incumbent solves and the candidate does not
    concordant pairs (both or neither) are uninformative by construction

Conditional on the discordant count n_d = b+c, the null distribution of b is exactly
Binomial(n_d, 1/2). This is McNemar's test in exact conditional form. Three consequences follow,
and they are what make this a better gate than a generic permutation test:

  (1) EXACTNESS FOR FREE. The reference distribution is a binomial, not a resampling estimate;
      no permutations, no Monte Carlo error, exact for any n.

  (2) AN ANALYTIC BUDGET. Required sample size depends on the DISCORDANCE RATE
      pi_d = P(pair is discordant), not on the raw task count:
          n_tasks ~ n_discordant / pi_d
      so a pool where 90% of tasks behave identically under both arms needs ten times the tasks
      of one where 10% do. The budget simulation of Section 6 becomes a formula that a
      practitioner can evaluate on one no-edit replicate.

  (3) A SHARPER STOPPING RULE. Sequential decisions can be CURTAILED: once the undecided pairs
      remaining cannot change the outcome no matter how they fall, stop immediately. This is
      exact rather than calibrated, unlike the empirical boundary of sequential_gate.py.

UNIFICATION. The tie rate P(tie) that fixes the naive rule's error in the paper's identity and
the concordance rate 1-pi_d that fixes this test's power are the same structural quantity. The
property that breaks the field's rule is the property that tells you how to test properly and how
much data you need.

Everything here runs on banked episodes.
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
# The checkout root: where the study mains look for paired evaluation records, under
# $BRACE_ROOT/work/. Set BRACE_ROOT when the records live elsewhere.
MCP = os.environ.get("BRACE_ROOT", ROOT)
CALLERS = ["qwen", "granite", "mistral", "falcon3"]


# ------------------------------------------------------------------ data
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


# ------------------------------------------------------------------ tests
def p_exact_discordant(diff):
    """Exact conditional one-sided p-value. diff in {-1,0,+1} per task."""
    b = int((diff > 0.5).sum())
    c = int((diff < -0.5).sum())
    n = b + c
    if n == 0:
        return 1.0
    # P(X >= b), X ~ Binomial(n, 1/2)
    tail = sum(math.comb(n, k) for k in range(b, n + 1))
    return tail / (2 ** n)


def p_signflip(diff, rng, B=999):
    if len(diff) < 8 or np.all(np.abs(diff) < 1e-12):
        return 1.0
    obs = diff.mean()
    null = (rng.choice([-1.0, 1.0], size=(B, len(diff))) * diff).mean(axis=1)
    return float(((null >= obs).sum() + 1) / (B + 1))


def t_paired(diff):
    s = diff.std(ddof=1)
    if s < 1e-12:
        return 1.0
    t = diff.mean() / (s / np.sqrt(len(diff)))
    return 1.0 if t <= 0 else float(0.5 * math.erfc(t / math.sqrt(2)))


def curtailed_decision(diff, alpha, block):
    """Exact sequential rule. Walk in blocks; stop when the remaining pairs cannot change it.

    At each look with b successes of n_d discordant so far and R pairs still unexamined:
      best case  -> b + R_max_discordant successes ; if even that cannot reach significance, stop.
      worst case -> b successes and the rest against ; if significance already holds regardless,
                    stop and retain.
    """
    n = len(diff)
    looks = list(range(block, n + 1, block)) or [n]
    if looks[-1] != n:
        looks.append(n)
    for k in looks:
        d = diff[:k]
        b = int((d > 0.5).sum()); c = int((d < -0.5).sum())
        if p_exact_discordant(d) <= alpha:
            return 1, k
        rem = n - k
        # optimistic completion: every remaining task discordant in the candidate's favour
        bo, no = b + rem, b + c + rem
        if no > 0:
            best = sum(math.comb(no, j) for j in range(bo, no + 1)) / (2 ** no)
        else:
            best = 1.0
        if best > alpha:
            return 0, k          # cannot reach significance even in the best case -> stop early
    return 0, n


# ------------------------------------------------------------------ experiment
def draw(by, have, group, delta, rng, cap):
    pm = list(have); rng.shuffle(pm)
    A, B = set(pm[:group]), set(pm[group:2 * group])
    pa, pb = nondeg(by, A), nondeg(by, B)
    common = sorted(set(pa) & set(pb))
    if len(common) < 16:
        return None
    idx = rng.permutation(len(common))[:cap]
    xa = np.array([pa[common[i]] for i in idx], float)
    xb = inject(np.array([pb[common[i]] for i in idx], float), delta, rng)
    return xb - xa


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--trials", type=int, default=800)
    ap.add_argument("--cap", type=int, default=280)
    ap.add_argument("--block", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(ROOT, "receipts", "discordant_gate.json"))
    a = ap.parse_args()

    BIG = {"qwen": [f"{MCP}/work/bignull_s{i}" for i in range(3)],
           "granite": [f"{MCP}/work/bignull_granite_s{i}" for i in range(3)],
           "mistral": [f"{MCP}/work/bignull_mistral_s{i}" for i in range(3)],
           "falcon3": [f"{MCP}/work/bignull_falcon3_s{i}" for i in range(3)]}
    SMALL = {c: [f"{MCP}/work/" + ("evolve_null" if c == "qwen" else f"evolve_null_{c}")]
             for c in CALLERS}

    deltas = [0.0, 0.05, 0.10, 0.20]
    rng = np.random.default_rng(17)
    out = {}

    print("  === calibration and power: exact discordant vs permutation vs paired t ===")
    print(f"  {'caller':9s} {'delta':>6s} {'exact':>8s} {'perm':>8s} {'t':>8s} "
          f"{'pi_d':>7s} {'n_disc':>7s}")
    print("  " + "-" * 60)
    for c in CALLERS:
        by = load(BIG.get(c, [])) or load(SMALL.get(c, []))
        if by is None:
            continue
        have = sorted({s for v in by.values() for s, _ in v})
        if len(have) < 2 * a.group:
            continue
        out[c] = {}
        for d in deltas:
            he = hp = ht = 0; nd = []; nn = 0
            for _ in range(a.trials):
                diff = draw(by, have, a.group, d, rng, a.cap)
                if diff is None:
                    continue
                nn += 1
                he += p_exact_discordant(diff) <= a.alpha
                hp += p_signflip(diff, rng, 399) <= a.alpha
                ht += t_paired(diff) <= a.alpha
                nd.append(float(np.mean(np.abs(diff) > 0.5)))
            if not nn:
                continue
            pid = float(np.mean(nd))
            print(f"  {c:9s} {d:6.2f} {he/nn:8.3f} {hp/nn:8.3f} {ht/nn:8.3f} "
                  f"{pid:7.3f} {pid*len(diff):7.1f}")
            out[c][f"{d:.2f}"] = {"exact": he / nn, "perm": hp / nn, "t": ht / nn,
                                  "discordance": pid, "n": nn}

    # ---- analytic budget from the discordance rate, checked against simulation
    print("\n  === analytic budget from the discordance rate ===")
    print("  For a shift that makes a fraction delta of tasks flip in the candidate's favour,")
    print("  the discordant count needed for 80% power at one-sided alpha is approximately")
    print("      n_d >= ( z_alpha*sqrt(1/4) + z_beta*sqrt(p(1-p)) )^2 / (p-1/2)^2 ,  p = share of")
    print("  discordant pairs favouring the candidate; tasks required is then n_d / pi_d.\n")
    za, zb = 1.959963985, 0.841621234
    print(f"  {'caller':9s} {'pi_d':>7s} {'delta':>6s} {'p':>6s} {'n_d':>7s} {'tasks (pred)':>13s}")
    print("  " + "-" * 54)
    for c, v in out.items():
        pid = v.get("0.00", {}).get("discordance", float("nan"))
        for d in (0.05, 0.10, 0.20):
            k = f"{d:.2f}"
            if k not in v or pid != pid or pid <= 0:
                continue
            # injected flips are all in the candidate's favour, so they add to b
            p = min(0.999, 0.5 + d / (2 * max(pid + d, 1e-9)))
            if p <= 0.5:
                continue
            nd_req = ((za * 0.5 + zb * math.sqrt(p * (1 - p))) ** 2) / ((p - 0.5) ** 2)
            print(f"  {c:9s} {pid:7.3f} {d:6.2f} {p:6.3f} {nd_req:7.0f} "
                  f"{nd_req/(pid+d):13.0f}")
            out[c].setdefault("budget", {})[k] = {"pi_d": pid, "p": p, "n_discordant": nd_req,
                                                  "tasks_pred": nd_req / (pid + d)}

    # ---- curtailed sequential: exact, no calibration needed
    print("\n  === curtailed sequential (exact) vs fixed-n exact ===")
    print(f"  {'caller':9s} {'delta':>6s} {'seq rate':>9s} {'fix rate':>9s} "
          f"{'seq tasks':>10s} {'saving':>8s}")
    print("  " + "-" * 56)
    for c in list(out):
        by = load(BIG.get(c, [])) or load(SMALL.get(c, []))
        if by is None:
            continue
        have = sorted({s for v in by.values() for s, _ in v})
        for d in deltas:
            hs = hf = 0; cost = []; nn = 0
            for _ in range(a.trials // 2):
                diff = draw(by, have, a.group, d, rng, a.cap)
                if diff is None:
                    continue
                nn += 1
                dec, used = curtailed_decision(diff, a.alpha, a.block)
                hs += dec; cost.append(used)
                hf += p_exact_discordant(diff) <= a.alpha
            if not nn:
                continue
            mc = float(np.mean(cost))
            print(f"  {c:9s} {d:6.2f} {hs/nn:9.3f} {hf/nn:9.3f} {mc:10.1f} "
                  f"{1 - mc/len(diff):7.0%}")
            out[c].setdefault("curtailed", {})[f"{d:.2f}"] = {
                "seq_rate": hs / nn, "fixed_rate": hf / nn, "seq_tasks": mc}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, sort_keys=True, default=float)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
