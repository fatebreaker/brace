"""Run the paper's core measurements on ANY substrate, from one episode file.

Takes any jsonl of {scenario, task_idx, seed, reward} and reports the quantities the paper
claims are general, so AWM and BFCL go through byte-identical analysis code:

  reward / non-degenerate group rate   the RL-relevant quantity
  tie rate and chance credit rate      G1: accept-if-larger FP = (1 - P(tie)) / 2
  paired vs unpaired t realised alpha  is the t-test valid on THIS substrate?
  permutation realised alpha           is the recommended test valid here?
  sign-flip rate of both estimators    does the single-round estimator point the right way?
"""
from __future__ import annotations
import argparse, collections, json, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--group", type=int, default=4)
ap.add_argument("--trials", type=int, default=1500)
ap.add_argument("--perms", type=int, default=299)
a = ap.parse_args()

by = collections.defaultdict(list)
for r in (json.loads(l) for l in open(a.episodes) if l.strip()):
    by[(r["scenario"], r["task_idx"])].append((str(r["seed"]), int(r["reward"])))
have = sorted({s for v in by.values() for s, _ in v})
neps = sum(len(v) for v in by.values())
print(f"\n  ===== {a.label or a.episodes} =====")
print(f"  {len(by)} tasks, {len(have)} seeds, {neps} episodes")
if len(have) < 2 * a.group:
    sys.exit(f"  need >= {2*a.group} seeds")

allr = [r for v in by.values() for _, r in v]
print(f"  reward mean {np.mean(allr):.3f}")

def nondeg(grp):
    o = {}
    for t, v in by.items():
        x = [r for s, r in v if s in grp]
        if len(x) == len(grp):
            o[t] = 0.0 if (sum(x) == 0 or sum(x) == len(x)) else 1.0
    return o

rng = np.random.default_rng(13)
CRIT = 1.959963985
nd_rates, ties, creds = [], 0, 0
tot = 0
ht = hu = hpp = hpu = 0
fp_flip = fu_flip = nz = 0
ok = 0
for _ in range(a.trials):
    pm = list(have); rng.shuffle(pm)
    A, B = set(pm[:a.group]), set(pm[a.group:2*a.group])
    pa, pb = nondeg(A), nondeg(B)
    common = sorted(set(pa) & set(pb))
    if len(common) < 16:
        continue
    ok += 1
    nd_rates.append(np.mean([pa[t] for t in common]))
    xa = np.array([pa[t] for t in common]); xb = np.array([pb[t] for t in common])
    # G1: chance credit rate of accept-if-larger, on the mean over tasks
    tot += 1
    if abs(xb.mean() - xa.mean()) < 1e-12: ties += 1
    elif xb.mean() > xa.mean(): creds += 1
    # tests, under a genuine null (arms differ only by seeds)
    d = xb - xa; s = d.std(ddof=1)
    t = 0.0 if s < 1e-12 else d.mean()/(s/np.sqrt(len(d)))
    ht += t > CRIT
    idx = rng.permutation(len(common)); h = len(common)//2
    ua, ub = xa[idx[:h]], xb[idx[h:]]
    va, vb = ua.var(ddof=1)/len(ua), ub.var(ddof=1)/len(ub)
    den = np.sqrt(va+vb); tu = 0.0 if den < 1e-12 else (ub.mean()-ua.mean())/den
    hu += tu > CRIT
    null = (rng.choice([-1.0,1.0], size=(a.perms, len(d)))*d).mean(axis=1)
    hpp += ((null >= d.mean()).sum()+1)/(a.perms+1) <= 0.025
    pool = np.concatenate([ua,ub]); sh = pool[np.argsort(rng.random((a.perms,len(pool))),axis=1)]
    nu = sh[:,len(ua):].mean(axis=1)-sh[:,:len(ua)].mean(axis=1)
    hpu += ((nu >= (ub.mean()-ua.mean())).sum()+1)/(a.perms+1) <= 0.025
    # sign flip vs the realised effect
    true = xb.mean()-xa.mean()
    if abs(true) > 1e-12:
        nz += 1
        fp_flip += np.sign(d.mean()) != np.sign(true)
        fu_flip += np.sign(ub.mean()-ua.mean()) != np.sign(true)

tr = ties/tot
print(f"  non-degenerate group rate {np.mean(nd_rates):.3f}")
print(f"\n  G1  tie rate {tr:.3f} -> predicted chance credit {(1-tr)/2:.3f} | "
      f"measured {creds/tot:.3f}  (gap {creds/tot-(1-tr)/2:+.3f})")
print(f"\n  realised alpha under a genuine null (nominal 0.025):")
print(f"    paired t          {ht/ok:.3f}{'   INVALID' if ht/ok > 0.0375 else ''}")
print(f"    unpaired t        {hu/ok:.3f}{'   INVALID' if hu/ok > 0.0375 else ''}")
print(f"    paired perm       {hpp/ok:.3f}{'   INVALID' if hpp/ok > 0.0375 else ''}")
print(f"    unpaired perm     {hpu/ok:.3f}{'   INVALID' if hpu/ok > 0.0375 else ''}")
print(f"\n  sign-flip vs realised effect ({nz} non-tied):")
print(f"    crossover     {fp_flip/max(nz,1):.3f}")
print(f"    single-round  {fu_flip/max(nz,1):.3f}")
