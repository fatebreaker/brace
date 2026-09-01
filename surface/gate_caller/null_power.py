"""Calibration AND power for all four acceptance procedures, on the loop's real objective.

WHY THIS EXISTS. evolve_null.py showed the sign-flip permutation test fixes the t-test's
anti-conservatism (granite paired t: 0.052 against a 0.025 nominal rate). But it also showed
paired_perm sitting at 0.004 -- far BELOW nominal. A conservative test buys calibration with
power, and a gate that never fires is useless no matter how well calibrated it is. Recommending
the permutation test without measuring its power would be exactly the error this paper is about:
changing an acceptance rule on the strength of one number.

So this measures both halves for the same four procedures, on the same banked episodes:

    paired t / unpaired t / paired permutation / unpaired permutation
    x  delta in {0, 0.05, 0.10, 0.20}

INJECTION. The objective is a per-task 0/1 indicator (can this task yield any GRPO gradient?),
so a synthetic effect of size delta is injected by flipping a random delta-fraction of arm B's
currently-0 tasks to 1. This matches the discrete structure of the objective rather than adding
Gaussian noise to it, which is what makes the null at delta=0 a genuine null: arms differ only
by which seeds they drew.

Reads the episodes evolve_null.py already banked. No GPU, no new rollouts.
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


def load(work):
    by = collections.defaultdict(list)
    p = os.path.join(work, "null_full.jsonl")
    if not os.path.exists(p):
        return None
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


def inject(vec, delta, rng):
    """Flip a delta-fraction of the zeros to ones: a real, positive, discrete effect."""
    if delta <= 0:
        return vec
    v = vec.copy()
    zeros = np.flatnonzero(v < 0.5)
    k = int(round(delta * len(v)))
    if k > 0 and len(zeros):
        v[rng.choice(zeros, size=min(k, len(zeros)), replace=False)] = 1.0
    return v


def t_paired(d):
    s = d.std(ddof=1)
    return 0.0 if s < 1e-12 else float(d.mean() / (s / np.sqrt(len(d))))


def t_welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    den = np.sqrt(va + vb)
    return 0.0 if den < 1e-12 else float((b.mean() - a.mean()) / den)


def p_signflip(d, rng, B):
    obs = d.mean()
    null = (rng.choice([-1.0, 1.0], size=(B, len(d))) * d).mean(axis=1)
    return float(((null >= obs).sum() + 1) / (B + 1))


def p_labelperm(a, b, rng, B):
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b])
    sh = pool[np.argsort(rng.random((B, len(pool))), axis=1)]
    null = sh[:, len(a):].mean(axis=1) - sh[:, :len(a)].mean(axis=1)
    return float(((null >= obs).sum() + 1) / (B + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--trials", type=int, default=800)
    ap.add_argument("--perms", type=int, default=499)
    ap.add_argument("--pool", default=os.path.join(HERE, "pools", "pool_bandenriched.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "receipts", "null_power.json"))
    a = ap.parse_args()

    pool = json.load(open(a.pool))
    tasks = sorted({(p["scenario"], p["task_idx"]) for p in pool})
    deltas = [0.0, 0.05, 0.10, 0.20]
    ARMS = ["paired_t", "unpaired_t", "paired_perm", "unpaired_perm"]
    ALPHA1 = 0.025          # one-sided, matching the loop's t > 1.96 rule
    CRIT = 1.96
    out = {}

    for c in CALLERS:
        w = os.path.join(MCP, "work", "evolve_null" if c == "qwen" else f"evolve_null_{c}")
        by = load(w)
        if by is None:
            continue
        have = sorted({s for v in by.values() for s, _ in v})
        if len(have) < 2 * a.group:
            print(f"  {c}: only {len(have)} seeds, need {2*a.group} -- skipped", flush=True)
            continue
        rng = np.random.default_rng(5)
        hit = {(m, d): 0 for m in ARMS for d in deltas}
        n = 0
        for _ in range(a.trials):
            perm = list(have); rng.shuffle(perm)
            A, Bs = set(perm[:a.group]), set(perm[a.group:2 * a.group])
            pa, pb = nondeg(by, tasks, A), nondeg(by, tasks, Bs)
            common = sorted(set(pa) & set(pb))
            if len(common) < 16:
                continue
            n += 1
            xa = np.array([pa[t] for t in common])
            xb0 = np.array([pb[t] for t in common])
            # unpaired split fixed per trial so all four procedures see the same episodes
            idx = rng.permutation(len(common)); h = len(common) // 2
            for d in deltas:
                xb = inject(xb0, d, rng)
                hit[("paired_t", d)] += t_paired(xb - xa) > CRIT
                ua, ub = xa[idx[:h]], xb[idx[h:]]
                hit[("unpaired_t", d)] += t_welch(ua, ub) > CRIT
                hit[("paired_perm", d)] += p_signflip(xb - xa, rng, a.perms) <= ALPHA1
                hit[("unpaired_perm", d)] += p_labelperm(ua, ub, rng, a.perms) <= ALPHA1

        if n == 0:
            print(f"\n  === {c} ===   no usable trials (episodes still banking) -- skipped",
                  flush=True)
            continue
        print(f"\n  === {c} ===   {n} trials, {a.perms} permutations, "
              f"one-sided alpha={ALPHA1}")
        print(f"  {'procedure':16s}" + "".join(f"{('d='+format(d,'.2f')):>10s}" for d in deltas))
        print("  " + "-" * (16 + 10 * len(deltas)))
        out[c] = {}
        for m in ARMS:
            row = [hit[(m, d)] / n for d in deltas]
            flag = "  <- FP too high" if row[0] > ALPHA1 * 1.5 else ""
            print(f"  {m:16s}" + "".join(f"{v:10.3f}" for v in row) + flag)
            out[c][m] = {f"{d:.2f}": hit[(m, d)] / n for d in deltas}
        print(f"  {'':16s}{'(null)':>10s}" + "".join(f"{'(power)':>10s}" for _ in deltas[1:]))

    # cross-caller summary: which procedure is calibrated everywhere AND most powerful?
    print("\n\n  ==== SUMMARY over callers ====")
    print(f"  {'procedure':16s} {'max FP':>8s} {'calibrated?':>12s} {'mean power d=.05':>18s} "
          f"{'d=.10':>8s}")
    print("  " + "-" * 66)
    for m in ARMS:
        cs = [c for c in CALLERS if c in out and m in out[c]]
        if not cs:
            continue
        fps = [out[c][m]["0.00"] for c in cs]
        p05 = [out[c][m]["0.05"] for c in cs]
        p10 = [out[c][m]["0.10"] for c in cs]
        ok = "yes" if max(fps) <= ALPHA1 * 1.5 else "NO"
        print(f"  {m:16s} {max(fps):8.3f} {ok:>12s} {float(np.mean(p05)):18.3f} "
              f"{float(np.mean(p10)):8.3f}")
        out.setdefault("summary", {})[m] = {
            "max_fp": max(fps), "calibrated": ok == "yes",
            "mean_power_05": float(np.mean(p05)), "mean_power_10": float(np.mean(p10))}
    print("\n  A procedure is only recommendable if it is calibrated on EVERY caller and retains")
    print("  power. Calibration alone is trivially achievable by never rejecting.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
