"""Availability-Targeted Surface Control: tune the MCP tool surface for GRADIENT, not reward.

THE OBSERVATION THIS IS BUILT ON. On this substrate the policy half learns nothing, and the
reason is measurable rather than mysterious: ~90% of GRPO groups are degenerate, because per-task
solve rates are bimodal. Restoring advertised tools does not fix it -- it moves tasks from p=0
straight to p=1, skipping the band where a group disagrees with itself. Measured on 295 held-out
tasks:

    surface           p=0      0<p<1     p=1     availability g(p,5)
    handicapped      86.4%      4.4%     9.2%          3.7%
    39% restored     77.3%      7.8%    14.9%          6.4%
    78% restored     74.2%      7.5%    18.3%          6.0%
    fully restored   73.2%      3.7%    23.1%          3.5%

Availability PEAKS at partial restoration and collapses at full restoration. So the surface that
maximises reward is close to the worst surface for learning, and a gate that optimises reward --
like the DISCORD loop, which restored all 1791 withheld names -- optimises against its own policy
half.

THE METHOD. Gradient availability g(p,k) = 1 - p^k - (1-p)^k is maximised at p = 0.5, and the
advertised tool surface is a monotone difficulty dial: restoring tools raises p. So treat the
surface as the control variable and drive each scenario toward p = 0.5. As the policy improves a
scenario, WITHHOLD tools again to hold it at the frontier. Nothing about this is possible in a
fixed environment; it exists because an MCP surface is programmable per scenario.

The controller is deliberately model-free. A parametric p(level) fit would need per-scenario dose
curves nobody has measured, whereas the sign of (p - target) is observable after every cycle and
is all a proportional controller needs.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random

import numpy as np

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SURF = f"{R}/surface/gate_surface/work/surfaces"


def scenario_tools(scen):
    p = os.path.join(SURF, f"{scen}.json")
    return [t["name"] for t in json.load(open(p)).get("tools", [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--pool", default=f"{R}/surface/gate_caller/pools/pool_big320.json")
    ap.add_argument("--init-surface", required=True, help="the handicapped starting surface")
    ap.add_argument("--levels", required=True, help="json of scenario -> restoration level in [0,1]")
    ap.add_argument("--out-map", required=True, help="json of scenario -> advertised tool names")
    ap.add_argument("--target", type=float, default=0.5, help="solve rate that maximises g(p,k)")
    ap.add_argument("--band", type=float, default=0.15, help="deadband around the target")
    ap.add_argument("--step", type=float, default=0.25, help="level change per cycle")
    ap.add_argument("--window", type=int, default=4000, help="most recent episodes to trust")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    pool = json.load(open(a.pool))
    scens = sorted({p["scenario"] for p in pool})
    init = {ln.strip() for ln in open(a.init_surface) if ln.strip()}

    rows = []
    if os.path.exists(a.episodes):
        for line in open(a.episodes, errors="ignore"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    rows = rows[-a.window:]

    # observed solve rate per scenario, from rollouts the trainer produced anyway
    by = collections.defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(int(r["reward"]))

    levels = json.load(open(a.levels)) if os.path.exists(a.levels) else {}
    moved_up = moved_down = held = 0
    for sc in scens:
        lv = float(levels.get(sc, 0.5))          # start every scenario mid-dial
        obs = by.get(sc)
        if obs:
            p = float(np.mean(obs))
            if p < a.target - a.band:
                lv = min(1.0, lv + a.step)       # too hard: advertise more tools
                moved_up += 1
            elif p > a.target + a.band:
                lv = max(0.0, lv - a.step)       # too easy: withhold again
                moved_down += 1
            else:
                held += 1
        levels[sc] = lv
    json.dump(levels, open(a.levels, "w"), indent=1)

    # realise the levels as an advertised set per scenario; the withheld names of a scenario are
    # restored in a FIXED order so a level change is monotone and reproducible, never a reshuffle
    amap, tot_adv, tot_all = {}, 0, 0
    for sc in scens:
        full = scenario_tools(sc)
        held_back = sorted(set(full) - init)
        rng = random.Random(hash(sc) & 0xFFFF)
        rng.shuffle(held_back)
        n = int(round(levels[sc] * len(held_back)))
        adv = sorted((set(full) & init) | set(held_back[:n]))
        amap[sc] = adv
        tot_adv += len(adv)
        tot_all += len(full)
    json.dump(amap, open(a.out_map, "w"))

    exp = np.mean([1 - p ** a.k - (1 - p) ** a.k
                   for p in (float(np.mean(v)) for v in by.values() if v)]) if by else float("nan")
    print(f"[surface-control] {len(scens)} scenarios | levels: {moved_up} up, {moved_down} down, "
          f"{held} in band | advertising {tot_adv}/{tot_all} names "
          f"| observed availability {100*exp:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
