"""Harness evolution over an MCP tool surface, with PAIRED vs UNPAIRED edit acceptance.

THE METHOD. Harness-evolution systems (AHE 2604.25850, EvoTrainer 2606.03108, HarnessX
2606.14249, ADAS, AFlow, GEPA, DGM, ShinkaEvolve, ...) accept or reject each edit by comparing
its score on one set of episodes against the incumbent's score on a DIFFERENT set. Task, seed
and initial-state variance therefore ride on top of the edit's true effect. We measured the
consequence on 5,616 banked episodes: a single-round comparison credits 43.3% of edits as
improvements and 43.1% under a null where the edit provably cannot matter -- the same number.

This runs the alternative: evaluate incumbent and candidate as a CROSSOVER TRIAL from
byte-identically restored state (same task, same seed, same initial DB), so the shared
nuisance cancels in the per-task difference. Restore costs 0.09 ms here; Shepherd's
(2605.10913) trace-level fork -- the closest prior art -- reports 134-143 ms, and by its own
account degenerates to full re-execution exactly for structural edits like these.

EDIT SPACE = the advertised tool set. Protocol-level and MCP-native: the server chooses what
appears in tools/list. SafeMCP (2606.01991, ACL 2026 Main) established that server-side tool
filtering is a learnable object; it optimises for safety, we optimise for training signal.
Capability-CHANGING by construction -- our own G-LANDSCAPE found capability-EQUIVALENT
presentation edits are inert (below their own redeal null on 3 of 4 callers), and AHE's
ablation independently localises gains to tools/middleware/memory rather than the prompt.

OBJECTIVE = non-degenerate group rate. A GRPO group whose rollouts all pass or all fail
contributes exactly zero gradient. We measured 87.2% degenerate on this substrate, and a live
verl run wasted 14 of 20 steps. So the thing worth maximising is the fraction of tasks that
can produce gradient at all, not raw success.

Both arms get an identical episode budget. The only difference is whether the comparison is
paired.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCP = os.path.dirname(ROOT)
SURF = os.path.join(ROOT, "gate_surface", "work", "surfaces")


def scenario_tools(scen):
    p = os.path.join(SURF, f"{scen}.json")
    return [t["name"] for t in json.load(open(p)).get("tools", [])]


def run_cell(pod, subset, tasks_by_scen, seeds, out, gpu_mem=90000):
    """Run episodes for one advertised tool subset. Returns path to the episode file."""
    env = dict(os.environ)
    env["GATE_TOOL_SUBSET"] = ",".join(sorted(subset))
    env["GATE_MAX_TURNS"] = "20"
    env["GATE_MAX_NEW_TOKENS"] = "1024"
    env["GATE_MEM_BUDGET_MB"] = str(gpu_mem)
    env["CALLER_FAMILY"] = env.get("EVOLVE_CALLER", "qwen")
    env["CALLER_PREFIX_CACHE"] = "1"
    cmd = ["srun", f"--jobid={pod}", "--ntasks=1", "--overlap", "--exact",
           "--gres=gpu:1", "--cpus-per-task=7", "--mem=110G",
           "bash", os.path.join(MCP, "slurm", "run_evolve_cell.sh"),
           json.dumps(tasks_by_scen), ",".join(str(s) for s in seeds), out]
    subprocess.run(cmd, env=env, check=False,
                   stdout=open(out + ".log", "a"), stderr=subprocess.STDOUT)
    return out


def objective(path, tasks):
    """Non-degenerate rate: fraction of tasks whose seeds are neither all-0 nor all-1."""
    by = collections.defaultdict(list)
    if not os.path.exists(path):
        return None, {}
    for r in (json.loads(l) for l in open(path)):
        by[(r["scenario"], r["task_idx"])].append(int(r["reward"]))
    per = {}
    for t in tasks:
        v = by.get(t)
        if v:
            per[t] = 0.0 if (sum(v) == 0 or sum(v) == len(v)) else 1.0
    return (float(np.mean(list(per.values()))) if per else None), per


def paired_test(base_per, cand_per):
    """One-sample t on per-task differences over the SHARED task set."""
    common = sorted(set(base_per) & set(cand_per))
    if len(common) < 8:
        return 0.0, 0.0, len(common)
    d = np.array([cand_per[t] - base_per[t] for t in common], float)
    s = d.std(ddof=1)
    t = 0.0 if s < 1e-12 else d.mean() / (s / np.sqrt(len(d)))
    return float(d.mean()), float(t), len(common)


def unpaired_test(base_per, cand_per, rng):
    """Welch two-sample t on DISJOINT task halves -- the field's protocol."""
    bt, ct = sorted(base_per), sorted(cand_per)
    common = sorted(set(bt) & set(ct))
    if len(common) < 16:
        return 0.0, 0.0, len(common)
    sh = list(common); rng.shuffle(sh)
    h = len(sh) // 2
    a = np.array([base_per[t] for t in sh[:h]], float)
    b = np.array([cand_per[t] for t in sh[h:]], float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    den = np.sqrt(va + vb)
    t = 0.0 if den < 1e-12 else (b.mean() - a.mean()) / den
    return float(b.mean() - a.mean()), float(t), len(common)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True)
    ap.add_argument("--gate", choices=["paired", "unpaired"], required=True)
    ap.add_argument("--pool", default=os.path.join(HERE, "pools", "pool_smoke20.json"))
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--crit", type=float, default=1.96, help="two-sided 5% critical value")
    ap.add_argument("--work", default=os.path.join(MCP, "work", "evolve"))
    a = ap.parse_args()

    rng = random.Random(20260801)
    nprng = np.random.default_rng(7)
    os.makedirs(a.work, exist_ok=True)
    pool = json.load(open(a.pool))
    tasks = sorted({(p["scenario"], p["task_idx"]) for p in pool})
    by_scen = collections.defaultdict(list)
    for s, i in tasks:
        by_scen[s].append(i)
    scens = sorted(by_scen)
    full = sorted({t for s in scens for t in scenario_tools(s)})
    print(f"[evolve:{a.gate}] {len(tasks)} tasks, {len(scens)} scenarios, "
          f"{len(full)} distinct tools", flush=True)

    incumbent = set(full)
    seeds = list(range(700, 700 + a.seeds))
    base_out = os.path.join(a.work, f"{a.gate}_r0_base.jsonl")
    run_cell(a.pod, incumbent, {k: v for k, v in by_scen.items()}, seeds, base_out)
    base_obj, base_per = objective(base_out, tasks)
    if base_obj is None:
        print(f"[evolve:{a.gate}] baseline produced no episodes; aborting", flush=True)
        return 1
    print(f"[evolve:{a.gate}] round 0 baseline non-degenerate rate {base_obj:.3f} "
          f"({len(incumbent)} tools)", flush=True)

    hist = [{"round": 0, "obj": base_obj, "n_tools": len(incumbent), "accepted": None}]
    for rd in range(1, a.rounds + 1):
        best = None
        for c in range(a.candidates):
            cand = set(incumbent)
            # capability-CHANGING edit: drop an advertised tool (or restore a dropped one)
            if len(cand) > 4 and (rng.random() < 0.75 or len(cand) == len(full)):
                cand.discard(rng.choice(sorted(cand)))
            else:
                missing = sorted(set(full) - cand)
                if missing:
                    cand.add(rng.choice(missing))
            out = os.path.join(a.work, f"{a.gate}_r{rd}_c{c}.jsonl")
            run_cell(a.pod, cand, {k: v for k, v in by_scen.items()}, seeds, out)
            obj, per = objective(out, tasks)
            if obj is None:
                continue
            if a.gate == "paired":
                delta, t, n = paired_test(base_per, per)
            else:
                delta, t, n = unpaired_test(base_per, per, nprng)
            keep = t > a.crit
            print(f"[evolve:{a.gate}] r{rd} c{c}: {len(cand)} tools obj={obj:.3f} "
                  f"delta={delta:+.3f} t={t:+.2f} n={n} -> "
                  f"{'ACCEPT' if keep else 'reject'}", flush=True)
            if keep and (best is None or t > best[1]):
                best = (cand, t, obj, per)
        if best:
            incumbent, _, base_obj, base_per = best
            hist.append({"round": rd, "obj": base_obj, "n_tools": len(incumbent),
                         "accepted": True})
        else:
            hist.append({"round": rd, "obj": base_obj, "n_tools": len(incumbent),
                         "accepted": False})
        print(f"[evolve:{a.gate}] round {rd} -> obj={base_obj:.3f} "
              f"tools={len(incumbent)}", flush=True)

    res = {"gate": a.gate, "history": hist, "final_tools": sorted(incumbent)}
    dest = os.path.join(a.work, f"history_{a.gate}.json")
    json.dump(res, open(dest, "w"), indent=2)
    print(f"[evolve:{a.gate}] DONE -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
