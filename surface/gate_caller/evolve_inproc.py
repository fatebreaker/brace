"""Harness evolution over an MCP tool surface — PAIRED vs UNPAIRED gate, single process.

Supersedes evolve_toolset.py, which shelled out per candidate and therefore reloaded vLLM
every time: ~3 min of model load x 36 evaluations is ~2 h of pure loading. run_landscape.py
already showed the fix — call run_gate.main() repeatedly inside ONE process and the engine
persists — so the model loads once and every candidate reuses it.

METHOD. Harness-evolution systems accept an edit by comparing its score on one set of episodes
against the incumbent's on a DIFFERENT set, so task/seed/state variance rides on the edit's
true effect. Measured here on 5,616 banked episodes: that protocol credits 43.3% of edits and
43.1% under a null where the edit provably cannot matter. This runs the alternative — a
crossover trial from byte-identically restored state (0.09 ms), so shared nuisance cancels in
the per-task difference.

EDIT SPACE = the advertised tool set (the server chooses what appears in tools/list; cf.
SafeMCP, ACL 2026 Main). Capability-CHANGING by construction: our G-LANDSCAPE result and AHE's
own ablation agree that presentation-only edits are inert.

OBJECTIVE = non-degenerate group rate, the fraction of tasks that can yield ANY GRPO gradient.
We measure 12.8% on this substrate and a live verl run wasted 14 of 20 steps, so this is the
binding constraint, not raw success.

Both arms get an identical episode budget; only the gate differs.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCP = os.path.dirname(ROOT)
SURF = os.path.join(ROOT, "gate_surface", "work", "surfaces")
sys.path.insert(0, os.path.join(ROOT, "gate_surface"))
sys.path.insert(0, HERE)


def scenario_tools(scen):
    p = os.path.join(SURF, f"{scen}.json")
    return [t["name"] for t in json.load(open(p)).get("tools", [])]


def nondegenerate(path, tasks, nseeds):
    by = collections.defaultdict(list)
    if not os.path.exists(path):
        return None, {}
    for r in (json.loads(l) for l in open(path)):
        by[(r["scenario"], r["task_idx"])].append(int(r["reward"]))
    per = {t: (0.0 if (sum(v) == 0 or sum(v) == len(v)) else 1.0)
           for t in tasks if (v := by.get(t))}
    return (float(np.mean(list(per.values()))) if per else None), per


def paired_t(base, cand):
    common = sorted(set(base) & set(cand))
    if len(common) < 8:
        return 0.0, 0.0, len(common)
    d = np.array([cand[t] - base[t] for t in common], float)
    s = d.std(ddof=1)
    return float(d.mean()), (0.0 if s < 1e-12 else float(d.mean() / (s / np.sqrt(len(d))))), len(common)


def unpaired_t(base, cand, rng):
    common = sorted(set(base) & set(cand))
    if len(common) < 16:
        return 0.0, 0.0, len(common)
    sh = list(common); rng.shuffle(sh); h = len(sh) // 2
    a = np.array([base[t] for t in sh[:h]], float)
    b = np.array([cand[t] for t in sh[h:]], float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    den = np.sqrt(va + vb)
    return float(b.mean() - a.mean()), (0.0 if den < 1e-12 else float((b.mean() - a.mean()) / den)), len(common)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["paired", "unpaired"], required=True)
    ap.add_argument("--pool", default=os.path.join(HERE, "pools", "pool_bandenriched.json"))
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--handicap", type=int, default=0,
                    help="POSITIVE CONTROL. Start from an incumbent with this many advertised "
                         "tools removed at random, so restoring one is a genuinely BENEFICIAL "
                         "edit. Without it the edit space contains only neutral-to-harmful "
                         "single-tool removals, both gates correctly reject everything, and "
                         "8/8 gate agreement tests only the reject direction -- which is what "
                         "happened in our first run (receipts/evolve_result.json).")
    ap.add_argument("--p-remove", type=float, default=0.75,
                    help="probability a candidate is a removal; the rest are restorations. "
                         "Set well below 0.5 with --handicap so the search can climb back.")
    ap.add_argument("--crit", type=float, default=1.96)
    ap.add_argument("--mem-fraction", type=float, default=0.63)
    ap.add_argument("--work", default=os.path.join(MCP, "work", "evolve3"))
    a = ap.parse_args()

    import parse_multi, policy_vllm as policy_mod, run_gate
    fam = policy_mod.family()
    if run_gate.MAX_TURNS < 20 or run_gate.MAX_NEW_TOKENS < 1024:
        raise SystemExit("caps not raised: need GATE_MAX_TURNS=20 GATE_MAX_NEW_TOKENS=1024")
    import difflib
    fl, sl = run_gate.SYSTEM.splitlines(True), parse_multi.system_for(fam).splitlines(True)
    if len([o for o in difflib.SequenceMatcher(None, fl, sl).get_opcodes() if o[0] != "equal"]) > 1:
        raise RuntimeError("system_for differs in more than one place")
    run_gate.SYSTEM = parse_multi.system_for(fam)
    _f = run_gate.parse_call
    run_gate.parse_call = lambda t, _g=_f: (_g(t) if _g(t) is not None else parse_multi._fallback(t))
    _r = policy_mod.render
    policy_mod.render = lambda m, t, _q=_r, _fm=fam: _q(parse_multi.normalise_history(m, _fm), t)
    sys.modules["policy"] = policy_mod
    run_gate.TEMPERATURE = 0.7

    os.makedirs(a.work, exist_ok=True)
    pool = json.load(open(a.pool))
    tasks = sorted({(p["scenario"], p["task_idx"]) for p in pool})
    by_scen = collections.defaultdict(list)
    for s, i in tasks:
        by_scen[s].append(i)
    full = sorted({t for s in by_scen for t in scenario_tools(s)})
    seeds = [str(700 + i) for i in range(a.seeds)]
    rng = random.Random(20260801)
    nprng = np.random.default_rng(7)
    print(f"[{a.gate}] {len(tasks)} tasks, {len(by_scen)} scenarios, {len(full)} tools, "
          f"seeds={seeds}", flush=True)

    def evaluate(subset, tag):
        out = os.path.join(a.work, f"{a.gate}_{tag}.jsonl")
        os.environ["GATE_TOOL_SUBSET"] = ",".join(sorted(subset))
        for sc, idxs in by_scen.items():
            argv = ["run_gate", "--scenarios", sc,
                    "--tasks", ",".join(str(i) for i in sorted(idxs)),
                    "--seeds", *seeds, "--interfaces", "RAW", "--gpu", "0",
                    "--slots", "16", "--micro", "16",
                    "--mem-fraction", str(a.mem_fraction),
                    "--out", out, "--resume-from", out]
            old = sys.argv; sys.argv = argv
            try:
                run_gate.main()
            except SystemExit:
                pass
            except Exception as e:
                print(f"[{a.gate}] {sc} FAILED {type(e).__name__}: {e}", flush=True)
            finally:
                sys.argv = old
        return nondegenerate(out, tasks, a.seeds)

    incumbent = set(full)
    if a.handicap:
        # Deterministic given the seed, and IDENTICAL across gates: both arms must start from
        # the same handicapped surface or the comparison is confounded by the starting point.
        hrng = random.Random(31337)
        removed = hrng.sample(sorted(full), min(a.handicap, len(full) - 8))
        incumbent -= set(removed)
        print(f"[{a.gate}] HANDICAP: removed {len(removed)} of {len(full)} tools; "
              f"restoring any of them is a genuinely beneficial edit. "
              f"md5(removed)={__import__('hashlib').md5(','.join(sorted(removed)).encode()).hexdigest()[:12]}",
              flush=True)
    base_obj, base_per = evaluate(incumbent, "r0_base")
    if base_obj is None:
        print(f"[{a.gate}] baseline empty; abort", flush=True)
        return 1
    print(f"[{a.gate}] r0 baseline nondegenerate={base_obj:.3f} tools={len(incumbent)}", flush=True)
    hist = [{"round": 0, "obj": base_obj, "n_tools": len(incumbent), "accepted": None}]

    for rd in range(1, a.rounds + 1):
        best = None
        for c in range(a.candidates):
            cand = set(incumbent)
            miss = sorted(set(full) - cand)
            if len(cand) > 4 and (rng.random() < a.p_remove or not miss):
                cand.discard(rng.choice(sorted(cand)))
            else:
                cand.add(rng.choice(miss))
            obj, per = evaluate(cand, f"r{rd}_c{c}")
            if obj is None:
                continue
            delta, t, n = (paired_t(base_per, per) if a.gate == "paired"
                           else unpaired_t(base_per, per, nprng))
            keep = t > a.crit
            print(f"[{a.gate}] r{rd}c{c} tools={len(cand)} obj={obj:.3f} d={delta:+.3f} "
                  f"t={t:+.2f} n={n} {'ACCEPT' if keep else 'reject'}", flush=True)
            if keep and (best is None or t > best[1]):
                best = (cand, t, obj, per)
        if best:
            incumbent, _, base_obj, base_per = best
        hist.append({"round": rd, "obj": base_obj, "n_tools": len(incumbent),
                     "accepted": bool(best)})
        print(f"[{a.gate}] round {rd} -> obj={base_obj:.3f} tools={len(incumbent)}", flush=True)

    dest = os.path.join(a.work, f"history_{a.gate}.json")
    json.dump({"gate": a.gate, "history": hist, "final_tools": sorted(incumbent)},
              open(dest, "w"), indent=2)
    print(f"[{a.gate}] DONE -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
