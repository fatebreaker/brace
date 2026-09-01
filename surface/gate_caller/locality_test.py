"""Is the effect of a tool-surface edit LOCAL to the tasks that need those tools?

WHY THIS DECIDES A METHOD. Each BFCL task's reference solution invokes exactly one function, so
withholding a block of 3 function names can only change the correct answer for ~1.8% of a
200-task pool. If the edit's effect really is confined to those tasks, an acceptance gate need
only evaluate them, and the cost of a decision falls by roughly the inverse of that share.

But withholding a function also changes the PROMPT for every task, since the advertised tool list
is shared. A model may behave differently on a task it could always solve simply because the menu
changed. That is a distraction effect, it is not local, and if it is large the targeting is
invalid. This measures it.

DESIGN. Evaluate the same 200 tasks under two surfaces -- full, and with a block of function
names withheld -- with identical seeds. Partition tasks into

    AFFECTED    ground-truth function is among those withheld
    UNAFFECTED  ground-truth function is still advertised

and compare the per-task change in each group. Under locality the AFFECTED group moves and the
UNAFFECTED group does not, beyond seed noise. A no-edit replicate (full surface against itself,
different seeds) calibrates what "beyond seed noise" means.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCP = os.path.dirname(ROOT)
sys.path.insert(0, HERE)

import bfcl_env as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="BFCL_v3_live_multiple")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--reps", type=int, default=6, help="independent withheld blocks")
    ap.add_argument("--mem-fraction", type=float, default=0.35)
    ap.add_argument("--out", default=os.path.join(ROOT, "receipts", "locality.json"))
    a = ap.parse_args()

    rows, ans = B.load_split(a.split)
    rows = rows[:a.limit]
    need = {}
    for r in rows:
        fns = set()
        for g in ans[r["id"]]:
            fns.update(g.keys())
        need[r["id"]] = fns
    allfn = sorted({f["name"] for r in rows for f in r["function"]})
    print(f"[locality] {len(rows)} tasks, {len(allfn)} functions, block={a.block}, "
          f"{a.reps} independent blocks", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, gpu_memory_utilization=a.mem_fraction, max_model_len=8192,
              enable_prefix_caching=True, trust_remote_code=True)

    def evaluate(withheld, seed_offset=0):
        prompts, meta = [], []
        for r in rows:
            funcs = [f for f in r["function"] if f["name"] not in withheld] or r["function"]
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
            sp = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256,
                                seed=5000 + seed_offset + s)
            for tid, o in zip(meta, llm.generate(prompts, sp, use_tqdm=False)):
                byt[tid].append(B.score_call(B.parse_call(o.outputs[0].text), ans[tid]))
        return {t: float(np.mean(v)) for t, v in byt.items()}

    rng = np.random.default_rng(99)
    base = evaluate(set(), 0)
    null = evaluate(set(), 1000)          # full surface again, different seeds: the noise floor

    res = {"affected": [], "unaffected": [], "null_unaffected": [], "shares": []}
    for rep in range(a.reps):
        W = set(rng.choice(allfn, size=a.block, replace=False))
        cand = evaluate(W, 100 * (rep + 1))
        aff = [t for t in base if need.get(t, set()) & W]
        una = [t for t in base if t in cand and t not in aff]
        if not aff:
            continue
        da = float(np.mean([cand[t] - base[t] for t in aff if t in cand]))
        du = float(np.mean([cand[t] - base[t] for t in una]))
        dn = float(np.mean([null[t] - base[t] for t in una if t in null]))
        res["affected"].append(da); res["unaffected"].append(du); res["null_unaffected"].append(dn)
        res["shares"].append(len(aff) / len(base))
        print(f"[locality] block {rep+1}: {len(aff)} affected tasks "
              f"({len(aff)/len(base):.1%})  mean change affected {da:+.3f}  "
              f"unaffected {du:+.3f}  (no-edit floor {dn:+.3f})", flush=True)

    if res["affected"]:
        A = np.array(res["affected"]); U = np.array(res["unaffected"]); N = np.array(res["null_unaffected"])
        print(f"\n  mean change on AFFECTED tasks   {A.mean():+.4f}")
        print(f"  mean change on UNAFFECTED tasks {U.mean():+.4f}")
        print(f"  no-edit floor on the same tasks {N.mean():+.4f}")
        print(f"  affected tasks are {np.mean(res['shares']):.1%} of the pool")
        if abs(U.mean() - N.mean()) > 1e-9:
            print(f"  leakage beyond noise: {U.mean()-N.mean():+.4f}")
        print(f"\n  Locality holds if the unaffected group moves no more than the no-edit floor.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
