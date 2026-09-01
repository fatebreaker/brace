"""A harness-evolution loop with ENOUGH BUDGET to actually make decisions.

Our AWM loops made 46 acceptance decisions and accepted nothing. That was not a quirk of the
gate: the paired t-test there was anti-conservative (realised alpha 0.052 vs 0.025 nominal), so
it should have accepted MORE readily, not less. The effects were simply below detection --
single-tool edits on a 443-tool surface move the objective 2-8 points, and required_budget.json
says resolving that needs 160-2560 tasks where we could afford 40. On AWM one candidate at 320
tasks costs ~67 minutes, so a properly-powered loop is unaffordable there.

BFCL is single-turn and stateless: 3,200 episodes in ~4 minutes. That buys the budget our own
analysis says is required, so this is where we can show what a correctly-sized loop does.

DESIGN
  substrate   BFCL v3 live_multiple, 200 tasks, 8 seeds per evaluation (1,600 episodes/candidate)
  handicap    a fraction of function NAMES is withheld globally, so restoring them is a
              genuinely beneficial edit -- the positive control the AWM loops lacked
  edit        restore or withhold a BLOCK of k function names, not a single tool, so the
              effect size is above the detection floor
  objective   task success (what AHE/HarnessX actually optimise). Non-degenerate group rate is
              our RL-specific objective and is reported alongside, but it is a poor search
              target here: BFCL groups are mostly all-pass for a strong caller.
  gates       crossover + sign-flip permutation test  vs  single-round + label permutation.
              Both use the PERMUTATION test this paper recommends, so the only difference
              between arms is the pairing, not the test.
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
sys.path.insert(0, HERE)

import bfcl_env as B  # noqa: E402
import math


def perm_exact(base, cand, rng, B_=999):
    """Exact conditional test on discordant pairs (Section 7 of the paper)."""
    common = sorted(set(base) & set(cand))
    if len(common) < 8:
        return 0.0, 1.0, len(common)
    d = np.array([cand[t] - base[t] for t in common], float)
    b = int((d > 0.5).sum()); c = int((d < -0.5).sum()); n = b + c
    if n == 0:
        return 0.0, 1.0, len(common)
    p = sum(math.comb(n, k) for k in range(b, n + 1)) / (2 ** n)
    return float(d.mean()), float(p), len(common)


def per_task(rewards_by_task, tasks):
    return {t: float(np.mean(rewards_by_task[t])) for t in tasks if rewards_by_task.get(t)}


def nondegen(rewards_by_task, tasks):
    out = {}
    for t in tasks:
        v = rewards_by_task.get(t)
        if v:
            out[t] = 0.0 if (sum(v) == 0 or sum(v) == len(v)) else 1.0
    return out


def perm_paired(base, cand, rng, B_=999):
    common = sorted(set(base) & set(cand))
    if len(common) < 16:
        return 0.0, 1.0, len(common)
    d = np.array([cand[t] - base[t] for t in common], float)
    obs = d.mean()
    null = (rng.choice([-1.0, 1.0], size=(B_, len(d))) * d).mean(axis=1)
    return float(obs), float(((null >= obs).sum() + 1) / (B_ + 1)), len(common)


def perm_unpaired(base, cand, rng, B_=999):
    common = sorted(set(base) & set(cand))
    if len(common) < 16:
        return 0.0, 1.0, len(common)
    sh = list(common); rng.shuffle(sh); h = len(sh) // 2
    a = np.array([base[t] for t in sh[:h]], float)
    b = np.array([cand[t] for t in sh[h:]], float)
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b])
    p = pool[np.argsort(rng.random((B_, len(pool))), axis=1)]
    null = p[:, len(a):].mean(axis=1) - p[:, :len(a)].mean(axis=1)
    return float(obs), float(((null >= obs).sum() + 1) / (B_ + 1)), len(common)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["paired", "unpaired", "naive"], required=True)
    # "naive" = the field's modal rule per the verified survey in the paper: accept
    # whenever the candidate's mean beats the incumbent's. No test. Same budget.
    ap.add_argument("--split", default="BFCL_v3_live_multiple")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--block", type=int, default=3, help="function names added/removed per edit")
    ap.add_argument("--handicap-frac", type=float, default=0.30)
    ap.add_argument("--handicap-seed", type=int, default=555,
                    help="identical across arms; vary to replicate the whole loop")
    ap.add_argument("--proposal-seed", type=int, default=20260802)
    ap.add_argument("--objective", choices=["success","nondegen"], default="success",
                    help="what the loop maximises. nondegen = fraction of tasks whose\n"
                         "seed group is neither all-pass nor all-fail, i.e. can yield a\n"
                         "GRPO gradient at all. Tests whether the advertised tool set is\n"
                         "an effective control knob for gradient availability.")
    ap.add_argument("--exact-gate", action="store_true",
                    help="use the exact conditional discordant-pair test")
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--mem-fraction", type=float, default=0.60)
    ap.add_argument("--work", default=os.path.join(MCP, "work", "bfcl_evolve"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    rows, ans = B.load_split(a.split)
    rows = rows[:a.limit]
    tasks = [r["id"] for r in rows]
    allfn = sorted({f["name"] for r in rows for f in r["function"]})
    print(f"[{a.gate}] {len(rows)} tasks, {len(allfn)} distinct functions, "
          f"{a.seeds} seeds -> {len(rows)*a.seeds} episodes/candidate", flush=True)

    # identical handicap in both arms
    hr = random.Random(a.handicap_seed)
    withheld = set(hr.sample(allfn, int(round(a.handicap_frac * len(allfn)))))
    import hashlib
    print(f"[{a.gate}] HANDICAP: withholding {len(withheld)}/{len(allfn)} function names "
          f"md5={hashlib.md5(','.join(sorted(withheld)).encode()).hexdigest()[:12]}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, gpu_memory_utilization=a.mem_fraction, max_model_len=8192,
              enable_prefix_caching=True, trust_remote_code=True)

    def evaluate(withheld_set, tag):
        prompts, meta = [], []
        for r in rows:
            funcs = [f for f in r["function"] if f["name"] not in withheld_set]
            if not funcs:
                funcs = r["function"]           # never advertise nothing
            tools = B.to_openai_tools(funcs)
            turns = r["question"][0] if r["question"] and isinstance(r["question"][0], list) else r["question"]
            msgs = [m for m in turns if isinstance(m, dict)]
            try:
                txt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                              tokenize=False)
            except Exception:
                continue
            prompts.append(txt); meta.append(r["id"])
        byt = collections.defaultdict(list)
        for s in range(a.seeds):
            sp = SamplingParams(temperature=a.temperature, top_p=0.8, max_tokens=256,
                                seed=2000 + s)
            for tid, o in zip(meta, llm.generate(prompts, sp, use_tqdm=False)):
                byt[tid].append(B.score_call(B.parse_call(o.outputs[0].text), ans[tid]))
        pt = per_task(byt, tasks)
        nd = nondegen(byt, tasks)
        primary = nd if a.objective == 'nondegen' else pt
        return primary, (float(np.mean(list(pt.values()))) if pt else 0.0), \
            (float(np.mean(list(nd.values()))) if nd else 0.0)

    rng = random.Random(a.proposal_seed)
    nprng = np.random.default_rng(3)
    incumbent = set(withheld)
    base_per, base_obj, base_nd = evaluate(incumbent, "r0")
    print(f"[{a.gate}] r0 baseline success={base_obj:.4f} nondegen={base_nd:.4f} "
          f"withheld={len(incumbent)}", flush=True)
    hist = [{"round": 0, "obj": base_obj, "nondegen": base_nd,
             "withheld": len(incumbent), "accepted": None}]

    for rd in range(1, a.rounds + 1):
        best = None
        for c in range(a.candidates):
            cand = set(incumbent)
            restorable = sorted(cand)
            if restorable and (rng.random() < 0.8 or not (set(allfn) - cand)):
                for f in rng.sample(restorable, min(a.block, len(restorable))):
                    cand.discard(f)                    # restore a block
            else:
                addable = sorted(set(allfn) - cand)
                for f in rng.sample(addable, min(a.block, len(addable))):
                    cand.add(f)                        # withhold a block
            per, obj, nd = evaluate(cand, f"r{rd}c{c}")
            if a.gate == "naive":
                # single-round comparison on disjoint task halves, accept if larger
                common = sorted(set(base_per) & set(per))
                sh = list(common); nprng.shuffle(sh); h = len(sh) // 2
                delta = float(np.mean([per[t] for t in sh[h:]])
                              - np.mean([base_per[t] for t in sh[:h]]))
                p, n, keep = float("nan"), len(common), delta > 0
            else:
                if a.exact_gate and a.gate == "paired":
                    delta, p, n = perm_exact(base_per, per, nprng)
                else:
                    delta, p, n = (perm_paired(base_per, per, nprng) if a.gate == "paired"
                                   else perm_unpaired(base_per, per, nprng))
                keep = p <= a.alpha
            print(f"[{a.gate}] r{rd}c{c} withheld={len(cand)} success={obj:.4f} "
                  f"nondegen={nd:.4f} d={delta:+.4f} p={p:.4f} n={n} "
                  f"{'ACCEPT' if keep else 'reject'}", flush=True)
            if keep and (best is None or delta > best[1]):
                best = (cand, delta, obj, per, nd)
        if best:
            incumbent, _, base_obj, base_per, base_nd = best
        hist.append({"round": rd, "obj": base_obj, "nondegen": base_nd,
                     "withheld": len(incumbent), "accepted": bool(best)})
        print(f"[{a.gate}] round {rd} -> success={base_obj:.4f} nondegen={base_nd:.4f} "
              f"withheld={len(incumbent)}", flush=True)

    dest = os.path.join(a.work, f"history_{a.gate}{a.tag}.json")
    json.dump({"gate": a.gate, "history": hist, "n_withheld_final": len(incumbent)},
              open(dest, "w"), indent=2)
    print(f"[{a.gate}] DONE -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
