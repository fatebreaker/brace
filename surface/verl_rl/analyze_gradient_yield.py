"""Fraction of GRPO groups that can produce ANY gradient, per training arm.

A GRPO group is the n rollouts of one prompt. If every rollout in a group scores the same, the
within-group advantage is exactly 0 and the group contributes nothing to the update, no matter
how good the optimiser is. verl's own AWM run recorded 92.5% degenerate groups -- "the pipeline
is proven; the task pool is not."

This measures whether selecting the training pool by our non-degenerate-group-rate objective
fixes that. Both arms are identical except for which 40 tasks were selected: same 12 scenarios,
same RAW tool surfaces, same model, same caps, same steps, same rollouts per prompt.

Groups are reconstructed from episodes.jsonl by (scenario, task_idx) within consecutive blocks
of train_batch_size*n episodes, which is how the trainer emits them.
"""
from __future__ import annotations
import collections, json, os, sys

NROLL = int(os.environ.get("NROLL", "5"))
BSZ = int(os.environ.get("BSZ", "4"))
R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print(f"  {'arm':10s} {'episodes':>9s} {'groups':>7s} {'non-degen':>10s} {'rate':>7s} "
      f"{'all-0':>7s} {'all-1':>7s} {'reward':>7s}")
print("  " + "-" * 70)
out = {}
for arm in ("control", "band"):
    ep = os.path.join(R, "work", "verl", f"run_{arm}", "episodes.jsonl")
    if not os.path.exists(ep):
        print(f"  {arm:10s} (no episodes yet)")
        continue
    recs = [json.loads(l) for l in open(ep) if l.strip()]
    if not recs:
        print(f"  {arm:10s} (empty)")
        continue
    # group by task within each block of BSZ*NROLL episodes
    blk = BSZ * NROLL
    nd = z = o = tot = 0
    for s in range(0, len(recs), blk):
        chunk = recs[s:s + blk]
        by = collections.defaultdict(list)
        for r in chunk:
            by[(r["scenario"], r["task_idx"])].append(int(r["reward"]))
        for k, v in by.items():
            if len(v) < 2:
                continue
            tot += 1
            if sum(v) == 0:
                z += 1
            elif sum(v) == len(v):
                o += 1
            else:
                nd += 1
    rate = nd / tot if tot else float("nan")
    rew = sum(r["reward"] for r in recs) / len(recs)
    print(f"  {arm:10s} {len(recs):9d} {tot:7d} {nd:10d} {rate:7.3f} {z:7d} {o:7d} {rew:7.3f}")
    out[arm] = {"episodes": len(recs), "groups": tot, "nondegenerate": nd,
                "rate": rate, "all_zero": z, "all_one": o, "reward_mean": rew}

if "control" in out and "band" in out and out["control"]["rate"] > 0:
    r = out["band"]["rate"] / out["control"]["rate"]
    print(f"\n  gradient-yield ratio band/control: {r:.2f}x")
    print(f"  Every degenerate group is compute spent for exactly zero parameter update.")
json.dump(out, open(os.path.join(R, "surface", "receipts", "rl_gradient_yield.json"), "w"),
          indent=2, sort_keys=True)
print(f"\n  wrote surface/receipts/rl_gradient_yield.json")
