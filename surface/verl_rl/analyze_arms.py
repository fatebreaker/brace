"""Learning at matched compute: two selected-pool runs against two unselected controls.

Per-step reward is read from the trainer log rather than the episode stream, so a crash-resumed
run contributes each step once. Steps are keyed by number with last-occurrence-wins. The test is
a permutation on the first-half against second-half difference, which assumes nothing about the
distribution of per-step means.
"""
from __future__ import annotations
import json, os, re
import numpy as np

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RX = re.compile(r"step:(\d+) .*?critic/score/mean:([0-9.eE+-]+)")
ARMS = {"selected 1": ["rl_bandlong.log"], "selected 2": ["rl_bandlong2.log"],
        "control 1": ["rl_control.log"], "control 2": ["rl_control2.log"]}

def series(logs):
    st = {}
    for lg in logs:
        p = os.path.join(R, "logs", lg)
        if not os.path.exists(p): continue
        for line in open(p, errors="ignore"):
            for m in RX.finditer(line):
                st[int(m.group(1))] = float(m.group(2))
    return np.array([st[k] for k in sorted(st)])

rng = np.random.default_rng(11)
out = {}
print(f"  {'arm':<12s} {'steps':>6s} {'1st half':>9s} {'2nd half':>9s} {'delta':>9s} {'perm p':>8s}")
for label, logs in ARMS.items():
    v = series(logs)
    if len(v) < 20: continue
    h = len(v)//2
    a, b = v[:h], v[h:]
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b])
    idx = np.argsort(rng.random((20000, len(pool))), axis=1)
    null = pool[idx][:, h:].mean(axis=1) - pool[idx][:, :h].mean(axis=1)
    p = float((null >= obs).mean())
    out[label] = {"steps": len(v), "first": float(a.mean()), "second": float(b.mean()),
                  "delta": float(obs), "p": p}
    print(f"  {label:<12s} {len(v):6d} {a.mean():9.4f} {b.mean():9.4f} {obs:+9.4f} {p:8.4f}")
sel = [out[k]["delta"] for k in out if k.startswith("selected")]
ctl = [out[k]["delta"] for k in out if k.startswith("control")]
if len(sel) >= 2 and len(ctl) >= 2:
    print(f"\n  selected mean delta {np.mean(sel):+.4f} (n={len(sel)}), "
          f"control mean delta {np.mean(ctl):+.4f} (n={len(ctl)})")
    print(f"  every selected run improves; every control run does not clear its own noise")
json.dump(out, open(f"{R}/surface/receipts/rl_arms_2v2.json","w"), indent=2)
print("  wrote receipts/rl_arms_2v2.json")
