"""Non-learned tool-surface selection: pick tools by keyword relevance to the task text.

WHY THIS IS THE MOST DANGEROUS BASELINE. Our method learns which tools to advertise per scenario,
tuning each toward the solve rate that maximises gradient availability. This is the same
intervention with the learning removed: score each tool by word overlap with the task, keep the
top fraction, advertise those. No controller, no rollouts, no gradient. If it recovers most of
the ceiling then the learned control is not the contribution and the paper has to say so.

WHAT THE OFFLINE ANALYSIS ALREADY SAYS, and why this needs measuring rather than estimating.
Over 184 held-out tasks that were solved at least once (so the tools they actually require are
known from the successful episodes' tool_calls), at the handicap's own 60% budget:

    TASK COVERAGE = the task's entire required tool set survives the surface
        budget   keyword heuristic   random (= the handicap)   advantage
          40%          0.408                 0.098              +0.310
          60%          0.527                 0.223              +0.304
          80%          0.630                 0.391              +0.239

So the heuristic more than doubles the fraction of tasks that remain solvable at the same surface
size. Coverage is NOT reward -- the policy still has to solve the task -- but interpolating the
measured endpoints (handicap 22.3% coverage -> 0.1136 reward, full surface -> 0.2497) puts a
keyword surface near 0.167, i.e. roughly 40% of our +0.1322. That is an extrapolation from two
points and must not be reported as a result; it is the reason to run the arm.

TWO GRANULARITIES. --mode map writes the per-scenario surface_map.json the trainer consumes via
AWM_ADVERTISED_MAP, scoring each tool against the concatenation of that scenario's task texts.
--mode flat writes the newline-separated name list coadapt_eval.sh passes as --subset, which is
the format every existing eval cell uses and therefore the one that makes the comparison
apples-to-apples. Per-task selection would be stronger still, but nothing downstream can consume
it: the advertised set is resolved once per scenario, not per episode.

BUDGET MATCHING is the whole point. Compare against the handicap at the SAME number of advertised
names, or the comparison measures surface size rather than surface choice.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

R = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ENVS = os.environ.get("BRACE_ENVS", os.path.join(R, "envs"))
AWM_PY = os.environ.get("AWM_PY", os.path.join(ENVS, "mcp_awm", "bin", "python"))
SURF = os.path.join(R, "surface", "gate_surface", "work", "surfaces")
STOP = set(
    "the a an of for to and or in on at my me i with by from get list all current return "
    "details please using that this it is are be as into new set your you have has was were "
    "which who what when where any each their there then than".split()
)


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2 and w not in STOP}


def score_tools(tools, query_tokens):
    """Overlap between the task wording and each tool's name and description.

    The name is split on underscores because tool names carry most of the signal
    (`list_user_addresses` -> {list,user,addresses}); the description is truncated because the
    AWM tool docs embed long response examples that would swamp the overlap with boilerplate.
    """
    out = {}
    for t in tools:
        tt = toks(t["name"].replace("_", " ")) | toks(str(t.get("description", ""))[:400])
        out[t["name"]] = len(query_tokens & tt)
    return out


def scenario_tools(sc):
    p = os.path.join(SURF, f"{sc}.json")
    if not os.path.exists(p):
        return []
    return json.load(open(p)).get("tools", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--mode", choices=["flat", "map"], default="flat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=None,
                    help="fraction of each scenario's tools to advertise")
    ap.add_argument("--match-surface", default=None,
                    help="instead of --budget, match the NAME COUNT of this surface file so the "
                         "comparison isolates which tools are chosen, not how many")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "gate_surface"))
    os.environ.setdefault("AWM_PY", AWM_PY)
    from awm_env import load_tasks

    pool = json.load(open(a.pool))
    scens = sorted({p["scenario"] for p in pool})
    universe = sorted({t["name"] for sc in scens for t in scenario_tools(sc)})

    budget = a.budget
    if a.match_surface:
        target = len({ln.strip() for ln in open(a.match_surface) if ln.strip()} & set(universe))
        budget = target / max(len(universe), 1)
        print(f"[heur] matching {a.match_surface}: {target}/{len(universe)} names "
              f"-> budget {budget:.3f}", flush=True)
    if budget is None:
        budget = 0.6

    amap, flat = {}, set()
    gscore: dict[str, int] = {}
    for sc in scens:
        tools = scenario_tools(sc)
        if not tools:
            continue
        names = [t["name"] for t in tools]
        idxs = sorted({p["task_idx"] for p in pool if p["scenario"] == sc})
        try:
            tasks = load_tasks(sc)
        except Exception:
            tasks = []
        q = set()
        for i in idxs:
            if i < len(tasks):
                q |= toks(str(tasks[i]))
        sc_score = score_tools(tools, q)
        k = max(1, int(round(budget * len(names))))
        chosen = sorted(names, key=lambda n: (-sc_score[n], n))[:k]
        amap[sc] = sorted(chosen)
        flat |= set(chosen)
        for n in chosen:
            gscore[n] = max(gscore.get(n, 0), sc_score[n])

    # Per-scenario rounding inflates the union, and a surface that is merely BIGGER would beat
    # the handicap for reasons that have nothing to do with choosing tools well. Trim the
    # globally weakest names back to the matched count so only the selection differs.
    if a.match_surface and len(flat) > target:
        keep = sorted(flat, key=lambda n: (-gscore.get(n, 0), n))[:target]
        dropped = len(flat) - len(keep)
        flat = set(keep)
        amap = {sc: sorted(set(v) & flat) for sc, v in amap.items()}
        print(f"[heur] trimmed {dropped} lowest-scoring names to match the target exactly",
              flush=True)

    if a.mode == "map":
        json.dump(amap, open(a.out, "w"))
        tot = sum(len(v) for v in amap.values())
        print(f"[heur] wrote per-scenario map for {len(amap)} scenarios, {tot} name-slots "
              f"-> {a.out}", flush=True)
    else:
        with open(a.out, "w") as fh:
            fh.write("\n".join(sorted(flat)) + "\n")
        print(f"[heur] wrote flat surface {len(flat)}/{len(universe)} names "
              f"({len(flat)/max(len(universe),1):.1%}) -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
