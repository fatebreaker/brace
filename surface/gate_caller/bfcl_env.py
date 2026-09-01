"""BFCL as a SECOND substrate for the acceptance-gate measurements.

Every result in this project so far comes from one substrate (AWM). That is the limitation we
most need to remove, because our central claims -- the chance-rate result, the caller-dependent
t-test failure, the sign-flip rate, the non-degenerate group rate -- are claims about
harness-evolution evaluation in general, not about SQLite MCP servers.

BFCL v3 `multiple` / `live_multiple` is the right second substrate:

  * the model must pick the correct function from SEVERAL advertised ones, which is exactly the
    MCP tool-selection problem and makes the advertised set an editable object here too;
  * scoring is a binary AST match against a published ground truth, so reward needs no LLM judge;
  * it is single-turn, so no environment servers, no restore primitive, no state -- which is a
    useful contrast: if our findings hold HERE, they are not artifacts of stateful rollouts.

Scoring follows BFCL's AST rules in the subset this needs: the emitted call must name a function
present in the ground truth, every ground-truth parameter must match one of its acceptable
values, and a parameter whose acceptable list contains "" may be omitted. Values are compared
after light normalisation (numeric strings, bools, list order preserved).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCP = os.path.dirname(ROOT)
DATA = os.path.join(MCP, "work", "bfcl")


# ------------------------------------------------------------------ data
def load_split(split):
    q = os.path.join(DATA, f"{split}.json")
    a = os.path.join(DATA, f"{split}_answer.json")
    qs = [json.loads(l) for l in open(q) if l.strip()]
    ans = {r["id"]: r["ground_truth"] for r in (json.loads(l) for l in open(a) if l.strip())}
    return [r for r in qs if r["id"] in ans], ans


def to_openai_tools(funcs):
    """BFCL function specs -> chat-template tool specs. 'dict' is BFCL's spelling of 'object'."""
    out = []
    for f in funcs:
        p = json.loads(json.dumps(f.get("parameters", {})))

        def fix(node):
            if isinstance(node, dict):
                if node.get("type") == "dict":
                    node["type"] = "object"
                if node.get("type") == "float":
                    node["type"] = "number"
                if node.get("type") == "tuple":
                    node["type"] = "array"
                for v in node.values():
                    fix(v)
            elif isinstance(node, list):
                for v in node:
                    fix(v)
        fix(p)
        out.append({"type": "function",
                    "function": {"name": f["name"],
                                 "description": f.get("description", ""),
                                 "parameters": p}})
    return out


# ------------------------------------------------------------------ scoring
def _norm(v):
    if isinstance(v, str):
        s = v.strip()
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            if re.fullmatch(r"-?\d+", s):
                return int(s)
            if re.fullmatch(r"-?\d*\.\d+", s):
                return float(s)
        except Exception:
            pass
        return s
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def _match_value(got, acceptable):
    g = _norm(got)
    for a in acceptable:
        an = _norm(a)
        if g == an:
            return True
        if isinstance(g, (int, float)) and isinstance(an, (int, float)) and float(g) == float(an):
            return True
        if isinstance(g, str) and isinstance(an, str) and g.lower() == an.lower():
            return True
    return False


def score_call(call, ground_truth):
    """call = {'name':..., 'arguments':{...}}. ground_truth = [{fname: {param: [values]}}]"""
    if not call or not isinstance(call, dict):
        return 0
    name = call.get("name")
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return 0
    for gt in ground_truth:
        for gname, gparams in gt.items():
            if gname != name:
                continue
            ok = True
            for p, acceptable in gparams.items():
                if p in args:
                    if not _match_value(args[p], acceptable):
                        ok = False
                        break
                else:
                    # omitted: only allowed when "" is an acceptable value
                    if not any(a == "" for a in acceptable):
                        ok = False
                        break
            # reject hallucinated params not in the ground truth spec
            if ok and any(k not in gparams for k in args):
                ok = False
            if ok:
                return 1
    return 0


# ------------------------------------------------------------------ parsing
TOOLCALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
JSONOBJ = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.S)


def parse_call(text):
    m = TOOLCALL.search(text or "")
    cands = [m.group(1)] if m else []
    if not cands:
        cands = JSONOBJ.findall(text or "")
    for c in reversed(cands):
        try:
            o = json.loads(c)
        except Exception:
            continue
        if isinstance(o, dict) and "name" in o:
            a = o.get("arguments", o.get("parameters", {}))
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            return {"name": o["name"], "arguments": a if isinstance(a, dict) else {}}
    return None


# ------------------------------------------------------------------ run
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="BFCL_v3_multiple")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--mem-fraction", type=float, default=0.60)
    ap.add_argument("--tool-subset", default="", help="comma-separated fn names to keep")
    ap.add_argument("--out", default=os.path.join(DATA, "runs", "qwen_multiple.jsonl"))
    a = ap.parse_args()

    rows, ans = load_split(a.split)
    rows = rows[:a.limit]
    print(f"[bfcl] {len(rows)} tasks from {a.split}, {a.seeds} seeds, T={a.temperature}",
          flush=True)
    nfun = [len(r["function"]) for r in rows]
    print(f"[bfcl] functions per task: min {min(nfun)} p50 {sorted(nfun)[len(nfun)//2]} "
          f"max {max(nfun)}", flush=True)

    keep = {s for s in a.tool_subset.split(",") if s} if a.tool_subset else None

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, gpu_memory_utilization=a.mem_fraction, max_model_len=8192,
              enable_prefix_caching=True, trust_remote_code=True)

    prompts, meta = [], []
    for r in rows:
        funcs = r["function"]
        if keep is not None:
            f2 = [f for f in funcs if f["name"] in keep]
            funcs = f2 if f2 else funcs      # never advertise an empty tool list
        tools = to_openai_tools(funcs)
        turns = r["question"][0] if r["question"] and isinstance(r["question"][0], list) else r["question"]
        msgs = [m for m in turns if isinstance(m, dict)]
        try:
            text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                           tokenize=False)
        except Exception as e:
            print(f"[bfcl] template failed for {r['id']}: {e}", flush=True)
            continue
        prompts.append(text)
        meta.append((r["id"], len(funcs)))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_written = 0
    with open(a.out, "w") as fh:
        for s in range(a.seeds):
            sp = SamplingParams(temperature=a.temperature, top_p=0.8,
                                max_tokens=a.max_new, seed=1000 + s)
            outs = llm.generate(prompts, sp)
            for (tid, nf), o in zip(meta, outs):
                txt = o.outputs[0].text
                rew = score_call(parse_call(txt), ans[tid])
                fh.write(json.dumps({"scenario": a.split, "task_idx": tid,
                                     "seed": str(1000 + s), "reward": int(rew),
                                     "n_functions": nf}) + "\n")
                n_written += 1
            fh.flush()
            print(f"[bfcl] seed {s+1}/{a.seeds} done, {n_written} episodes", flush=True)
    print(f"[bfcl] wrote {n_written} -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
