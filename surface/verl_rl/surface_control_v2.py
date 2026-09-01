"""ATSC-v2: shape the tool surface by RESPONSIVENESS, not by raw solve rate.

WHY v1 FAILS. The original controller is proportional control on (p - 0.5): if a scenario's solve
rate is below the band, advertise more tools. That assumes the surface can move p. Measured on the
live a8A arm after 65 steps, it cannot for most scenarios:

    level 0.00:   8 scenarios
    level 0.25:   8
    level 0.50:   6
    level 0.75:  10
    level 1.00: 126        <-- 80% pinned at the ceiling, mean level 0.877

Those 126 scenarios sit at p=0 even with every tool advertised -- they are hard for reasons the
tool surface does not touch (multi-step reasoning, verifier strictness). v1 keeps opening tools,
observes no change, opens again, and saturates. From there it is indistinguishable from "advertise
everything", which is exactly why a static 0.5 surface performs comparably and why ATSC advertises
95.6% of the universe against static's 79.9%. A controller pinned at its ceiling is not
controlling anything.

WHAT v2 DOES. Spend the dial where it actually moves the objective. GRPO's usable signal per
scenario is gradient availability g(p,k) = 1 - p^k - (1-p)^k, so the quantity worth maximising is
not p but the RESPONSIVENESS of g to the level:

    responsiveness(sc) = |g(p_now, k) - g(p_before, k)| / |level_now - level_before|

- A scenario whose g moves when the level moves is worth controlling: keep steering it to p=0.5.
- A scenario whose g does not move after two level increases is TOOL-INSENSITIVE: freeze it and
  stop spending range on it. Freezing also stops it inflating the advertised-name count, which is
  what made the a8F comparison confounded.

This is the same insight that makes PLR and ACCEL work -- prioritise by learning potential -- but
applied to the ENVIRONMENT rather than to task selection. PLR picks which levels to replay; v2
decides which levels are worth shaping at all, and shapes those.

Two-sided control is also restored in practice. Under v1 nothing pushed back: opening was free and
almost no scenario ever reached p > 0.65 to trigger a close. v2 frees range by freezing the
insensitive scenarios, so the responsive ones can be driven down as well as up.

State carried in levels.json alongside each level: the previous level, the previous g, and a
strike count of consecutive unresponsive moves. Written back every cycle, so a restart resumes the
controller rather than restarting it at 0.5.
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


def g_of(p, k):
    return 1.0 - p ** k - (1.0 - p) ** k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--pool", default=f"{R}/surface/gate_caller/pools/pool_max.json")
    ap.add_argument("--init-surface", required=True)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--out-map", required=True)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--band", type=float, default=0.15)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=4000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--freeze-after", type=int, default=2,
                    help="consecutive level moves with no change in g before a scenario is judged "
                         "tool-insensitive and frozen. 2 is deliberately impatient: under v1, 126 "
                         "of 158 scenarios reached the ceiling, so the cost of steering a "
                         "hopeless scenario is far higher than the cost of freezing a live one, "
                         "which any later responsiveness un-freezes anyway.")
    ap.add_argument("--dead-eps", type=float, default=0.02,
                    help="|delta g| below this counts as no response")
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

    by = collections.defaultdict(list)
    for r in rows:
        try:
            by[r["scenario"]].append(int(r["reward"] > 0))
        except Exception:
            pass

    state = json.load(open(a.levels)) if os.path.exists(a.levels) else {}
    up = down = held = frozen = thawed = 0

    for sc in scens:
        st = state.get(sc)
        if not isinstance(st, dict):                     # migrate v1's bare float
            st = {"lv": float(st) if st is not None else 0.5,
                  "prev_lv": None, "prev_g": None, "strikes": 0, "frozen": False}
        obs = by.get(sc)
        if not obs:
            state[sc] = st
            continue

        p = float(np.mean(obs))
        g = g_of(p, a.k)

        # responsiveness: did g move when the level last moved?
        if st["prev_lv"] is not None and abs(st["lv"] - st["prev_lv"]) > 1e-9:
            dg = abs(g - (st["prev_g"] if st["prev_g"] is not None else g))
            if dg < a.dead_eps:
                st["strikes"] += 1
            else:
                st["strikes"] = 0
                if st["frozen"]:
                    st["frozen"] = False                 # it moved after all
                    thawed += 1

        if st["strikes"] >= a.freeze_after and not st["frozen"]:
            st["frozen"] = True
            frozen += 1

        st["prev_lv"], st["prev_g"] = st["lv"], g

        if st["frozen"]:
            held += 1                                    # spend no range here
        elif p < a.target - a.band:
            st["lv"] = min(1.0, st["lv"] + a.step); up += 1
        elif p > a.target + a.band:
            st["lv"] = max(0.0, st["lv"] - a.step); down += 1
        else:
            held += 1
        state[sc] = st

    json.dump(state, open(a.levels, "w"), indent=1)

    amap, tot_adv, tot_all = {}, 0, 0
    for sc in scens:
        st = state.get(sc)
        lv = st["lv"] if isinstance(st, dict) else float(st or 0.5)
        full = scenario_tools(sc)
        held_back = sorted(set(full) - init)
        rng = random.Random(hash(sc) & 0xFFFF)
        rng.shuffle(held_back)
        n = int(round(lv * len(held_back)))
        adv = sorted((set(full) & init) | set(held_back[:n]))
        amap[sc] = adv
        tot_adv += len(adv)
        tot_all += len(full)
    json.dump(amap, open(a.out_map, "w"))

    live = [v for v in state.values() if isinstance(v, dict) and not v.get("frozen")]
    exp = np.mean([g_of(float(np.mean(v)), a.k) for v in by.values() if v]) if by else float("nan")
    mean_lv = np.mean([v["lv"] for v in state.values() if isinstance(v, dict)]) if state else 0.0
    print(f"[atsc-v2] {len(scens)} scenarios | {up} up, {down} down, {held} held | "
          f"froze {frozen}, thawed {thawed}, {len(live)} still steerable | "
          f"mean level {mean_lv:.3f} | advertising {tot_adv}/{tot_all} names | "
          f"observed availability {100*exp:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
