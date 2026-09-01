"""The NAIVE-gated contrast arm for the DISCORD-AWM loop.

WHY THIS RUN EXISTS. The DISCORD loop on the H200 shows what a correctly powered acceptance
procedure does on the stateful MCP substrate. This run is the other half of that figure: the
SAME pool, the SAME handicap (seed 555), the SAME edit proposal distribution, gated the way
deployed evolution systems actually gate. ADAS scores each candidate on a validation subset and
keeps the best-so-far score; AFlow and PromptBreeder do the same with different subset sizes.
Accept when the candidate's subsample score beats the remembered best. No test, no pairing, and
the remembered score is the accepted candidate's own optimistically biased estimate, which is
the winner's-curse ratchet the paper names.

GROUND TRUTH IS FREE HERE. Every edit either restores names that the handicap withheld (a
genuinely beneficial direction) or withholds currently advertised real names (harmful or inert).
So each ACCEPT is classifiable without any extra episodes: net_restored > 0 is a good accept,
net_restored < 0 is a bad one. One full-pool evaluation of the final incumbent at the end gives
the true objective for the trajectory comparison against the DISCORD arm.

Machinery (run_gate + policy swap) is identical to awm_discord.py so episodes are
byte-comparable across the two arms.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(HERE, "pools", "pool_big320.json"))
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--subsample", type=int, default=40,
                    help="tasks per naive decision, matching deployed validation-subset sizes")
    ap.add_argument("--handicap-frac", type=float, default=0.10)
    ap.add_argument("--handicap-seed", type=int, default=555)
    ap.add_argument("--edit-block", type=int, default=20)
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--mem-fraction", type=float, default=0.62)
    ap.add_argument("--proposal-seed", type=int, default=20260803,
                    help="edit proposal stream; match the DISCORD arm to pair them")
    ap.add_argument("--init-advertised", default=None,
                    help="start from this advertised set (one tool name per line) instead of "
                         "applying the handicap, so a co-adaptation arm carries its surface "
                         "from one cycle into the next")
    ap.add_argument("--skip-final", action="store_true",
                    help="skip the closing full-pool evaluation (co-adaptation cycles)")
    ap.add_argument("--work", default=os.path.join(MCP, "work", "awm_naive"))
    a = ap.parse_args()

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
    print(f"[naive-awm] {len(tasks)} tasks, {len(scens)} scenarios, {len(full)} names, "
          f"{a.seeds} seeds, subsample={a.subsample}", flush=True)

    if a.init_advertised and os.path.exists(a.init_advertised):
        carried = {ln.strip() for ln in open(a.init_advertised) if ln.strip()} & set(full)
        withheld = set(full) - carried
        print(f"[naive-awm] CARRIED SURFACE: {len(carried)}/{len(full)} names advertised, "
              f"{len(withheld)} still withheld", flush=True)
    else:
        hr = random.Random(a.handicap_seed)
        withheld = set(hr.sample(full, int(round(a.handicap_frac * len(full)))))
        print(f"[naive-awm] HANDICAP: withholding {len(withheld)}/{len(full)} names "
              f"md5={hashlib.md5(','.join(sorted(withheld)).encode()).hexdigest()[:12]}", flush=True)

    def run_tasks(subset, task_list, out):
        os.environ["GATE_TOOL_SUBSET"] = ",".join(sorted(subset))
        grouped = collections.defaultdict(list)
        for s, i in task_list:
            grouped[s].append(i)
        banked = collections.Counter()
        if os.path.exists(out):
            for r in (json.loads(l) for l in open(out)):
                banked[(r["scenario"], str(r["task_idx"]), str(r["seed"]))] += 1
        fails = 0
        for sc in sorted(grouped):
            pend = [i for i in sorted(grouped[sc])
                    if any(banked[(sc, str(i), s)] == 0 for s in seeds)]
            if not pend:
                continue
            argv = ["run_gate", "--scenarios", sc,
                    "--tasks", ",".join(str(i) for i in pend),
                    "--seeds", *seeds, "--interfaces", "RAW", "--gpu", "0",
                    "--slots", str(a.slots), "--micro", str(a.slots),
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
                print(f"[naive-awm] {sc} FAILED {type(e).__name__}: {e}", flush=True)
                if fails >= 3:
                    print("[naive-awm] 3 consecutive engine failures, exiting for clean restart",
                          flush=True)
                    sys.exit(7)
            finally:
                sys.argv = old

    def mean_reward(out, task_list):
        by = collections.defaultdict(list)
        if not os.path.exists(out):
            return float("nan"), 0
        for r in (json.loads(l) for l in open(out)):
            by[(r["scenario"], r["task_idx"])].append(int(r["reward"]))
        vals = [float(np.mean(v)) for t, v in by.items() if t in set(task_list)]
        return (float(np.mean(vals)) if vals else float("nan")), len(vals)

    rng = random.Random(a.proposal_seed)   # same proposal stream as the DISCORD arm
    srng = random.Random(777)              # subsample draws

    # ---- round 0: incumbent scored the deployed way, one 40-task subsample ----
    incumbent = set(full) - withheld
    sub0 = srng.sample(tasks, a.subsample)
    out0 = os.path.join(a.work, "incumbent_r0.jsonl")
    print(f"[naive-awm] scoring incumbent on {a.subsample} tasks", flush=True)
    run_tasks(incumbent, sub0, out0)
    best_score, n0 = mean_reward(out0, sub0)
    total_eps = n0 * a.seeds
    print(f"[naive-awm] r0 remembered score={best_score:.4f} on {n0} tasks", flush=True)

    hist = [{"round": 0, "remembered": best_score, "advertised": len(incumbent),
             "accepted": None, "episodes": total_eps}]
    accepts = []

    for rd in range(1, a.rounds + 1):
        accepted = False
        for c in range(a.candidates):
            cand = set(incumbent)
            missing = sorted(set(full) - cand)
            if missing and (rng.random() < 0.8 or len(cand) == len(full)):
                for nm in rng.sample(missing, min(a.edit_block, len(missing))):
                    cand.add(nm)
            else:
                for nm in rng.sample(sorted(cand), min(a.edit_block, max(len(cand) - 8, 0))):
                    cand.discard(nm)
            net_restored = len(cand - incumbent) - len(incumbent - cand)
            sub = srng.sample(tasks, a.subsample)
            out = os.path.join(a.work, f"cand_r{rd}c{c}.jsonl")
            run_tasks(cand, sub, out)
            score, n = mean_reward(out, sub)
            total_eps += n * a.seeds
            take = bool(score > best_score)
            print(f"[naive-awm] r{rd}c{c} advertised={len(cand)} net_restored={net_restored:+d} "
                  f"score={score:.4f} vs remembered={best_score:.4f} "
                  f"{'ACCEPT' if take else 'reject'}", flush=True)
            if take:
                accepts.append({"round": rd, "cand": c, "net_restored": net_restored,
                                "claimed": score, "over": best_score})
                incumbent = cand
                best_score = score          # the winner's-curse ratchet, verbatim
                accepted = True
                break
        hist.append({"round": rd, "remembered": best_score, "advertised": len(incumbent),
                     "accepted": accepted, "episodes": total_eps})

    # ---- the bill: one full-pool evaluation of the final incumbent ----
    # A co-adaptation cycle skips it: the surface is handed straight to the next training cycle
    # and the honest number comes from the held-out 2x2 at the end, not from this pool.
    if a.skip_final:
        true_obj, nf = float("nan"), 0
        print("[naive-awm] skipping the full-pool evaluation (co-adaptation cycle)", flush=True)
    else:
        fin = os.path.join(a.work, "final_full.jsonl")
        print(f"[naive-awm] final full-pool evaluation ({len(tasks)} tasks)", flush=True)
        run_tasks(incumbent, tasks, fin)
        true_obj, nf = mean_reward(fin, tasks)
        total_eps += nf * a.seeds
    still_withheld = len(withheld & (set(full) - incumbent))
    print(f"[naive-awm] TRUE final objective={true_obj:.4f} on {nf} tasks | "
          f"remembered={best_score:.4f} | advertised={len(incumbent)}/{len(full)} | "
          f"withheld names still missing: {still_withheld}/{len(withheld)}", flush=True)

    good = sum(1 for x in accepts if x["net_restored"] > 0)
    bad = sum(1 for x in accepts if x["net_restored"] < 0)
    print(f"[naive-awm] accepts={len(accepts)} (good-direction {good}, bad-direction {bad})",
          flush=True)
    dest = os.path.join(a.work, "history_naive_awm.json")
    json.dump({"history": hist, "accepts": accepts, "true_final_objective": true_obj,
               "remembered_final": best_score, "withheld_initial": len(withheld),
               "withheld_still_missing": still_withheld,
               "advertised_final": len(incumbent), "total_episodes": total_eps,
               "advertised_names": sorted(incumbent)},
              open(dest, "w"), indent=2)
    print(f"[naive-awm] DONE -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
