"""Does the closed form for gradient availability predict realized GRPO behaviour?

A group of k rollouts on a task solved with probability p is non-degenerate, meaning it carries
a within-group advantage signal, with probability 1 - p^k - (1-p)^k. That formula is the second
lever on gradient availability, and the paper uses it to argue the default of five rollouts is
not optimal. Here we test it against real training rollouts rather than asserting it.

Groups are recovered from the episode stream: verl writes train_batch_size x n episodes per
step, so consecutive blocks of that size partition into exactly n rollouts per prompt, which we
verify before using a run. Realized rates at k < n come from averaging over all subsets of size
k within each recovered group, which is exact rather than sampled.

Two tests:
  IN SAMPLE      per-task p estimated from the whole run, compared against realized rate.
  OUT OF SAMPLE  per-task p estimated from the FIRST half of training, used to predict the
                 realized rate in the SECOND half. This is the honest test, since it asks the
                 formula to predict data it has not seen, under a policy that has since moved.
"""
from __future__ import annotations
import collections, itertools, json, os, sys
import numpy as np

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BATCH, NROLL = 4, 5

def groups(path):
    rs = [json.loads(l) for l in open(path)]
    out = []
    for i in range(0, len(rs) - (BATCH*NROLL - 1), BATCH*NROLL):
        blk = rs[i:i+BATCH*NROLL]
        c = collections.defaultdict(list)
        for r in blk:
            c[(r["scenario"], r["task_idx"])].append(int(r["reward"]))
        if len(c) == BATCH and all(len(v) == NROLL for v in c.values()):
            out.append([(t, v) for t, v in c.items()])
    return out

def realized(gs, k):
    """Exact mean over all C(n,k) subsets of each group."""
    hits = tot = 0
    for blk in gs:
        for _, v in blk:
            for sub in itertools.combinations(v, k):
                hits += 0 if (sum(sub) == 0 or sum(sub) == k) else 1
                tot += 1
    return hits / tot

def predicted(gs, rates, k):
    vals = []
    for blk in gs:
        for t, _ in blk:
            p = rates.get(t)
            if p is None: continue
            vals.append(1 - p**k - (1-p)**k)
    return float(np.mean(vals))

def rates_from(gs):
    acc = collections.defaultdict(list)
    for blk in gs:
        for t, v in blk:
            acc[t].extend(v)
    return {t: float(np.mean(v)) for t, v in acc.items()}

out = {}
for run in ["run_bandlong", "run_control"]:
    p = f"{R}/work/verl/{run}/episodes.jsonl"
    if not os.path.exists(p): continue
    gs = groups(p)
    if len(gs) < 20: continue
    half = len(gs) // 2
    r_all, r_first = rates_from(gs), rates_from(gs[:half])
    row = {"groups": len(gs), "in_sample": {}, "out_of_sample": {}}
    print(f"\n  === {run} === {len(gs)} recovered groups ({len(gs)*BATCH} prompt-groups)")
    print(f"  {'k':>3s} {'predicted':>10s} {'realized':>9s} {'error':>8s}   "
          f"{'pred(1st half)':>14s} {'realized(2nd)':>13s} {'error':>8s}")
    for k in (2, 3, 4, 5):
        pr, re_ = predicted(gs, r_all, k), realized(gs, k)
        pro, reo = predicted(gs[half:], r_first, k), realized(gs[half:], k)
        row["in_sample"][k] = {"predicted": pr, "realized": re_, "error": pr - re_}
        row["out_of_sample"][k] = {"predicted": pro, "realized": reo, "error": pro - reo}
        print(f"  {k:3d} {pr:10.4f} {re_:9.4f} {pr-re_:+8.4f}   {pro:14.4f} {reo:13.4f} {pro-reo:+8.4f}")
    # the extrapolation the k=8 run will test
    row["extrapolation_k8"] = predicted(gs, r_all, 8)
    print(f"  extrapolated k=8 (to be tested by the k=8 arm): {row['extrapolation_k8']:.4f}")
    out[run] = row
json.dump(out, open(f"{R}/surface/receipts/groupsize_validation.json", "w"), indent=2)
print("\n  wrote receipts/groupsize_validation.json")

# ---------------------------------------------------------------- bias diagnosis
# The closed form is evaluated at a per-task solve rate pooled over the whole run. But the
# policy moves during training, so a task's p is not constant: pooling mixes several p values
# into one. By Jensen, E[p^k] >= (E p)^k, so a pooled estimate UNDER-states the chance of an
# all-same group and therefore OVER-states gradient availability. That is the sign of the error
# we observe, and it is the same policy drift Section 8 measures as decay. Re-estimating p
# within local windows of training should remove it.
def rates_windowed(gs, nwin):
    per = []
    w = max(1, len(gs) // nwin)
    for s in range(0, len(gs), w):
        chunk = gs[s:s+w]
        acc = collections.defaultdict(list)
        for blk in chunk:
            for t, v in blk:
                acc[t].extend(v)
        per.append(({t: float(np.mean(v)) for t, v in acc.items()}, chunk))
    return per

print("\n  === bias diagnosis: local vs pooled estimation of p ===")
diag = {}
for run in ["run_bandlong"]:
    gs = groups(f"{R}/work/verl/{run}/episodes.jsonl")
    diag[run] = {}
    print(f"  {'k':>3s} {'realized':>9s} {'pooled':>9s} {'err':>8s} {'windowed':>10s} {'err':>8s}")
    for k in (2, 3, 4, 5):
        re_ = realized(gs, k)
        pooled = predicted(gs, rates_from(gs), k)
        vals = []
        for rates, chunk in rates_windowed(gs, 10):
            for blk in chunk:
                for t, _ in blk:
                    p = rates.get(t)
                    if p is not None:
                        vals.append(1 - p**k - (1-p)**k)
        win = float(np.mean(vals))
        diag[run][k] = {"realized": re_, "pooled": pooled, "windowed": win,
                        "pooled_err": pooled-re_, "windowed_err": win-re_}
        print(f"  {k:3d} {re_:9.4f} {pooled:9.4f} {pooled-re_:+8.4f} {win:10.4f} {win-re_:+8.4f}")
    # corrected k=8 extrapolation
    vals = []
    for rates, chunk in rates_windowed(gs, 10):
        for blk in chunk:
            for t, _ in blk:
                p = rates.get(t)
                if p is not None: vals.append(1 - p**8 - (1-p)**8)
    diag[run]["k8_windowed"] = float(np.mean(vals))
    print(f"  corrected k=8 prediction: {diag[run]['k8_windowed']:.4f}")
json.dump(diag, open(f"{R}/surface/receipts/groupsize_bias.json", "w"), indent=2)
print("  wrote receipts/groupsize_bias.json")
