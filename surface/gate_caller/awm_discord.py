"""A DISCORD-gated harness-evolution loop on the AWM/MCP substrate.

WHY THIS RUN EXISTS. The paper's budget analysis says a correctly powered acceptance decision
needs 160+ tasks, and our own 40-task AWM loops retained nothing across 46 decisions, consistent
with being underpowered by construction. The gate divergence results therefore live on BFCL, the
cheap substrate, and the paper's most attackable sentence says so. DISCORD's pitch is that exact
curtailment makes a correctly powered loop affordable. This run is that claim, executed: a
320-task, 158-scenario pool on the stateful MCP substrate, gated by the exact conditional test
with curtailment.

DESIGN
  pool        pool_big320.json: 320 tasks over 158 scenarios, per-task solve rates already
              banked from the no-edit replicates (bignull_s*).
  handicap    withhold a fraction of ALL advertised tool names (union over scenarios), chosen
              deterministically, so restoring names is a genuinely beneficial edit and the loop
              has a known direction of improvement.
  edit        restore (p=0.8) or withhold a block of tool names.
  objective   per-task success (mean over seeds). A pair counts discordant when the per-task
              mean moves by more than 0.5, i.e. a clear flip.
  gate        exact conditional test on discordant pairs, evaluated scenario-block by
              scenario-block with EXACT curtailment: accept as soon as p <= alpha; stop-reject
              as soon as even an all-favourable completion could not reach alpha; and a
              futility rule after 200 tasks (p > 0.30 with >= 8 discordant pairs) that can only
              reduce power, never inflate alpha, since the accept boundary is untouched.

The evaluation machinery (run_gate + policy swap) is copied from evolve_inproc.py so that
episodes here are byte-comparable with every earlier AWM run.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
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


def p_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(HERE, "pools", "pool_big320.json"))
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--block-scenarios", type=int, default=12,
                    help="scenarios evaluated between curtailment checks")
    ap.add_argument("--handicap-frac", type=float, default=0.10)
    ap.add_argument("--handicap-seed", type=int, default=555)
    ap.add_argument("--edit-block", type=int, default=20, help="tool names per edit")
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--futility-after", type=int, default=200)
    ap.add_argument("--futility-p", type=float, default=0.30)
    ap.add_argument("--mem-fraction", type=float, default=0.62)
    ap.add_argument("--proposal-seed", type=int, default=20260803,
                    help="edit proposal stream; vary to replicate a loop on independent edits")
    ap.add_argument("--init-advertised", default=None,
                    help="start from this advertised set (one tool name per line) instead of "
                         "applying the handicap; co-adaptation carries the certified surface "
                         "from one cycle into the next rather than re-handicapping each time")
    ap.add_argument("--work", default=os.path.join(MCP, "work", "awm_discord"))
    a = ap.parse_args()

    # ---- evaluation machinery, identical to evolve_inproc.py ----
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

    os.makedirs(a.work, exist_ok=True)
    pool = json.load(open(a.pool))
    tasks = sorted({(p["scenario"], p["task_idx"]) for p in pool})
    by_scen = collections.defaultdict(list)
    for s, i in tasks:
        by_scen[s].append(i)
    scens = sorted(by_scen)
    full = sorted({t for s in scens for t in scenario_tools(s)})
    seeds = [str(900 + i) for i in range(a.seeds)]
    print(f"[discord-awm] {len(tasks)} tasks, {len(scens)} scenarios, {len(full)} distinct "
          f"tool names, {a.seeds} seeds", flush=True)

    if a.init_advertised and os.path.exists(a.init_advertised):
        carried = {ln.strip() for ln in open(a.init_advertised) if ln.strip()} & set(full)
        withheld = set(full) - carried
        print(f"[discord-awm] CARRIED SURFACE: {len(carried)}/{len(full)} names advertised, "
              f"{len(withheld)} still withheld", flush=True)
    else:
        hr = random.Random(a.handicap_seed)
        withheld = set(hr.sample(full, int(round(a.handicap_frac * len(full)))))
        print(f"[discord-awm] HANDICAP: withholding {len(withheld)}/{len(full)} names "
              f"md5={hashlib.md5(','.join(sorted(withheld)).encode()).hexdigest()[:12]}", flush=True)

    def run_scenarios(subset, scen_list, out):
        """Evaluate scen_list under advertised set `subset`, appending to out."""
        os.environ["GATE_TOOL_SUBSET"] = ",".join(sorted(subset))
        # skip fully-banked scenarios BEFORE booting an engine: back-to-back boot/teardown
        # cycles for no-work scenarios race the previous engine's memory release and fail
        banked = collections.Counter()
        if os.path.exists(out):
            for r in (json.loads(l) for l in open(out)):
                banked[(r["scenario"], str(r["task_idx"]), str(r["seed"]))] += 1
        fails = 0
        for sc in scen_list:
            pend = [i for i in sorted(by_scen[sc])
                    if any(banked[(sc, str(i), s)] == 0 for s in seeds)]
            if not pend:
                continue
            argv = ["run_gate", "--scenarios", sc,
                    "--tasks", ",".join(str(i) for i in pend),
                    "--seeds", *seeds, "--interfaces", "RAW", "--gpu", "0",
                    "--slots", os.environ.get("AWM_SLOTS", "8"), "--micro", os.environ.get("AWM_SLOTS", "8"),
                    "--mem-fraction", str(a.mem_fraction),
                    "--out", out, "--resume-from", out]
            old = sys.argv; sys.argv = argv
            try:
                run_gate.main()
                fails = 0
            except SystemExit:
                fails = 0
            except Exception as e:
                fails += 1
                print(f"[discord-awm] {sc} FAILED {type(e).__name__}: {e}", flush=True)
                # a failing engine boot never self-heals inside this process (leaked host
                # memory survives); exit so the outer retry loop restarts us clean
                if fails >= 3:
                    print("[discord-awm] 3 consecutive engine failures, exiting for clean restart",
                          flush=True)
                    sys.exit(7)
            finally:
                sys.argv = old

    def per_task(out):
        by = collections.defaultdict(list)
        if not os.path.exists(out):
            return {}
        for r in (json.loads(l) for l in open(out)):
            by[(r["scenario"], r["task_idx"])].append(int(r["reward"]))
        return {t: float(np.mean(v)) for t, v in by.items()}

    # ---- baseline: incumbent evaluated on the FULL pool ----
    incumbent = set(full) - withheld
    base_out = os.path.join(a.work, "incumbent_r0.jsonl")
    print(f"[discord-awm] evaluating incumbent on the full pool "
          f"({len(tasks)} tasks x {a.seeds} seeds)", flush=True)
    run_scenarios(incumbent, scens, base_out)
    base = per_task(base_out)
    print(f"[discord-awm] r0 incumbent success={np.mean(list(base.values())):.4f} "
          f"on {len(base)} tasks", flush=True)

    rng = random.Random(a.proposal_seed)
    hist = [{"round": 0, "obj": float(np.mean(list(base.values()))),
             "advertised": len(incumbent), "accepted": None, "episodes": len(tasks) * a.seeds}]
    total_eps = len(tasks) * a.seeds

    for rd in range(1, a.rounds + 1):
        accepted = None
        for c in range(a.candidates):
            cand = set(incumbent)
            missing = sorted(set(full) - cand)
            if missing and (rng.random() < 0.8 or len(cand) == len(full)):
                for nm in rng.sample(missing, min(a.edit_block, len(missing))):
                    cand.add(nm)               # restore withheld names (beneficial direction)
            else:
                for nm in rng.sample(sorted(cand), min(a.edit_block, max(len(cand) - 8, 0))):
                    cand.discard(nm)           # withhold advertised names
            order = scens[:]; rng.shuffle(order)
            out = os.path.join(a.work, f"cand_r{rd}c{c}.jsonl")
            b = ccount = used_tasks = 0
            decision = 0; pval = 1.0
            for lo in range(0, len(order), a.block_scenarios):
                chunk = order[lo:lo + a.block_scenarios]
                run_scenarios(cand, chunk, out)
                cur = per_task(out)
                b = sum(1 for t, v in cur.items() if t in base and v - base[t] > 0.5)
                ccount = sum(1 for t, v in cur.items() if t in base and base[t] - v > 0.5)
                used_tasks = len(cur)
                pval = p_exact(b, ccount)
                if pval <= a.alpha:
                    decision = 1
                    break
                rem = len(tasks) - used_tasks
                if p_exact(b + rem, ccount) > a.alpha:
                    break                       # exact stop-reject
                if used_tasks >= a.futility_after and (b + ccount) >= 8 and pval > a.futility_p:
                    break                       # futility (cannot inflate alpha)
            total_eps += used_tasks * a.seeds
            print(f"[discord-awm] r{rd}c{c} advertised={len(cand)} b={b} c={ccount} "
                  f"p={pval:.4f} tasks={used_tasks}/{len(tasks)} "
                  f"{'ACCEPT' if decision else 'reject'}", flush=True)
            if decision:
                accepted = (cand, out)
                break
        if accepted:
            incumbent = accepted[0]
            # Persist the certified surface THE MOMENT it is certified. Everything below is
            # expensive and can die; when it did, the accepted edit was lost with no error and the
            # next training cycle silently ran on the unedited surface. A decision this costly to
            # earn must not depend on later work succeeding.
            json.dump({"advertised_names": sorted(incumbent), "round": rd,
                       "b": b, "c": ccount, "p": pval, "tasks_used": used_tasks},
                      open(os.path.join(a.work, "accepted_surface.json"), "w"), indent=2)
            if a.rounds > 1:
                base_out = os.path.join(a.work, f"incumbent_r{rd}.jsonl")
                print(f"[discord-awm] re-evaluating new incumbent on the full pool", flush=True)
                run_scenarios(incumbent, scens, base_out)
                base = per_task(base_out)
                total_eps += len(tasks) * a.seeds
            else:
                # A single-round gate has no next round to pair against, so re-evaluating the new
                # incumbent over the whole pool buys nothing and costs 320x2 episodes.
                print("[discord-awm] single-round gate: skipping post-accept re-evaluation",
                      flush=True)
        obj = float(np.mean(list(base.values()))) if base else float("nan")
        hist.append({"round": rd, "obj": obj, "advertised": len(incumbent),
                     "accepted": bool(accepted), "episodes": total_eps})
        print(f"[discord-awm] round {rd} -> success={obj:.4f} advertised={len(incumbent)} "
              f"episodes-so-far={total_eps}", flush=True)

    dest = os.path.join(a.work, "history_discord_awm.json")
    json.dump({"history": hist, "withheld_initial": len(withheld),
               "advertised_final": len(incumbent), "total_episodes": total_eps,
               # the names themselves, so the next training cycle can roll out against exactly
               # the surface this gate certified
               "advertised_names": sorted(incumbent)},
              open(dest, "w"), indent=2)
    print(f"[discord-awm] DONE -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
