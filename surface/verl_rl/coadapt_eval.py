"""The 2x2 that decides whether co-adaptation earned anything: policy x harness, held out.

WHY THIS DESIGN. Four completed RL arms improved their training reward and transferred nothing:
bandlong2 gained +0.175 in sample at p < 1e-4 and lost 0.019 on held-out tasks. So a single
held-out number for a co-adaptation run proves nothing on its own -- if it is up, the obvious
objection is that the policy half did the work, and the policy half is exactly the part already
known not to transfer.

Crossing the two factors answers that objection with measurement instead of argument:

                        handicapped surface        certified surface
    base policy              cell A                    cell B
    trained policy           cell C                    cell D

  harness effect   (B - A) and (D - C): the same policy, two harnesses
  policy effect    (C - A) and (D - B): the same harness, two policies
  interaction      (D - C) - (B - A):   does a co-adapted policy exploit its own harness

The prediction the method makes, and that this can refute: the harness effect is positive on
tasks never trained on, because restoring a tool changes the environment for every task that
needs it, while the policy effect stays near zero as it did in every earlier arm.

HELD OUT MEANS UNSEEN TASKS, NOT NOVEL SCENARIOS. A certified surface is a list of tool names,
so it has no meaning for a scenario whose tools were never in the gate's universe -- filtering a
novel scenario by it would strip every tool and score zero for a reason that has nothing to do
with the method. The held-out pool is therefore 295 tasks drawn from the same 158 scenarios and
disjoint from the 320 the arm trained on.
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
MCP = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "gate_surface"))
sys.path.insert(0, os.path.join(ROOT, "gate_caller"))

CELLS = {"A": ("base", "init"), "B": ("base", "final"),
         "C": ("trained", "init"), "D": ("trained", "final")}


def p_exact(b, c):
    """Exact conditional test on discordant pairs, as in the gate itself."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def run_cell(a) -> int:
    import parse_multi, policy_vllm as policy_mod, run_gate
    fam = policy_mod.family()
    if run_gate.MAX_TURNS < 20 or run_gate.MAX_NEW_TOKENS < 1024:
        raise SystemExit("caps not raised: need GATE_MAX_TURNS=20 GATE_MAX_NEW_TOKENS=1024")
    run_gate.SYSTEM = parse_multi.system_for(fam)
    _f = run_gate.parse_call
    run_gate.parse_call = lambda t, _g=_f: (_g(t) if _g(t) is not None else parse_multi._fallback(t))
    _r = policy_mod.render
    policy_mod.render = lambda m, t, _q=_r, _fm=fam: _q(parse_multi.normalise_history(m, _fm), t)
    sys.modules["policy"] = policy_mod
    run_gate.TEMPERATURE = 0.7

    subset = {ln.strip() for ln in open(a.subset) if ln.strip()}
    os.environ["GATE_TOOL_SUBSET"] = ",".join(sorted(subset))
    # PER-SCENARIO surface. Training advertises a different tool list per scenario (ATSC rewrites
    # it every cycle; even --surface-fixed 0.5 restores a per-scenario half), while this eval
    # applied ONE flat list to every scenario. atscfix trains advertising 4380 of 5483 names and
    # was scored on 2687 -- a 40% shift in the action space, which turns "the policy did not
    # improve" into a statement about out-of-distribution transfer rather than about learning.
    # With a map, a checkpoint is scored on the surface it actually trained on.
    smap = json.load(open(a.subset_map)) if getattr(a, "subset_map", None) else None
    pool = json.load(open(a.pool))
    by_scen = collections.defaultdict(list)
    for p in pool:
        by_scen[p["scenario"]].append(p["task_idx"])
    seeds = [str(900 + i) for i in range(a.seeds)]

    banked = collections.Counter()
    if os.path.exists(a.out):
        for r in (json.loads(l) for l in open(a.out)):
            banked[(r["scenario"], str(r["task_idx"]), str(r["seed"]))] += 1
    fails = 0
    for sc in sorted(by_scen):
        pend = [i for i in sorted(set(by_scen[sc]))
                if any(banked[(sc, str(i), s)] == 0 for s in seeds)]
        if not pend:
            continue
        if smap is not None:
            names = smap.get(sc)
            if not names:          # scenario missing from the map: skip it rather than silently
                continue           # scoring on a different surface than the one requested
            os.environ["GATE_TOOL_SUBSET"] = ",".join(sorted(names))
        argv = ["run_gate", "--scenarios", sc, "--tasks", ",".join(str(i) for i in pend),
                "--seeds", *seeds, "--interfaces", "RAW", "--gpu", "0",
                "--slots", os.environ.get("AWM_SLOTS", "8"),
                "--micro", os.environ.get("AWM_SLOTS", "8"),
                "--mem-fraction", str(a.mem_fraction), "--out", a.out, "--resume-from", a.out]
        old = sys.argv
        sys.argv = argv
        try:
            run_gate.main()
            fails = 0
        except SystemExit:
            fails = 0
        except Exception as e:
            fails += 1
            print(f"[coadapt-eval] {sc} FAILED {type(e).__name__}: {e}", flush=True)
            if fails >= 3:
                print("[coadapt-eval] 3 consecutive engine failures, exiting for clean restart",
                      flush=True)
                return 7
        finally:
            sys.argv = old
    return 0


def per_task(path):
    by = collections.defaultdict(list)
    if not os.path.exists(path):
        return {}
    for r in (json.loads(l) for l in open(path)):
        by[(r["scenario"], r["task_idx"])].append(int(r["reward"]))
    return {t: float(np.mean(v)) for t, v in by.items()}


def contrast(x, y, label):
    """Paired contrast on the tasks both cells actually evaluated."""
    both = sorted(set(x) & set(y))
    if not both:
        return f"  {label}: no shared tasks"
    d = np.array([y[t] - x[t] for t in both])
    b = int((d > 0.5).sum())
    c = int((d < -0.5).sum())
    return (f"  {label}: {np.mean(d):+.4f} over {len(both)} tasks "
            f"(better {b}, worse {c}, exact p={p_exact(b, c):.4g})")


def report(a) -> int:
    cells = {k: per_task(os.path.join(a.dir, f"cell_{k}.jsonl")) for k in CELLS}
    print("[coadapt-eval] cell means on the held-out pool")
    for k, (pol, surf) in CELLS.items():
        v = cells[k]
        m = f"{np.mean(list(v.values())):.4f}" if v else "n/a"
        print(f"  {k} ({pol:>7} policy, {surf:>5} surface): {m} on {len(v)} tasks")
    print("[coadapt-eval] effects")
    print(contrast(cells["A"], cells["B"], "harness effect, base policy    (B-A)"))
    print(contrast(cells["C"], cells["D"], "harness effect, trained policy (D-C)"))
    print(contrast(cells["A"], cells["C"], "policy  effect, init surface   (C-A)"))
    print(contrast(cells["B"], cells["D"], "policy  effect, final surface  (D-B)"))
    both = sorted(set(cells["A"]) & set(cells["B"]) & set(cells["C"]) & set(cells["D"]))
    if both:
        inter = np.mean([(cells["D"][t] - cells["C"][t]) - (cells["B"][t] - cells["A"][t])
                         for t in both])
        print(f"  interaction (D-C)-(B-A): {inter:+.4f} over {len(both)} tasks")
    json.dump({k: {"n": len(v), "mean": (float(np.mean(list(v.values()))) if v else None)}
               for k, v in cells.items()},
              open(os.path.join(a.dir, "cells.json"), "w"), indent=2)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cell", "report"], required=True)
    ap.add_argument("--pool", default=os.path.join(ROOT, "gate_caller", "pools",
                                                   "pool_coadapt_heldout.json"))
    ap.add_argument("--subset", help="advertised-name file for this cell")
    ap.add_argument("--subset-map", dest="subset_map",
                    help="json scenario -> advertised names; overrides --subset per scenario so a "
                         "checkpoint can be scored on the surface it actually trained on")
    ap.add_argument("--out", help="episodes jsonl for this cell")
    ap.add_argument("--dir", help="directory of cell_*.jsonl, for --mode report")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--mem-fraction", type=float, default=0.62)
    a = ap.parse_args()
    return run_cell(a) if a.mode == "cell" else report(a)


if __name__ == "__main__":
    sys.exit(main())
