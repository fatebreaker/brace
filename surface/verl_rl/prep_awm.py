"""AWM task pool -> verl parquet, plus the token budget the config has to be sized for.

The rows are deliberately thin: the agent loop rebuilds the real prompt (system +
task + the scenario's RAW tool surface) itself, because the tool schemas are
per-scenario and cannot live in a global verl tool registry. `prompt` here exists
only because RLHFDataset requires it; `extra_info.{scenario,task_idx}` is what the
agent loop actually reads.
"""
from __future__ import annotations
import argparse, json, os, sys

MCP_ROOT = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SURFACE = os.path.join(MCP_ROOT, "surface")
for p in (os.path.join(SURFACE, "gate_surface"), os.path.join(SURFACE, "gate_caller")):
    if p not in sys.path:
        sys.path.insert(0, p)

import interfaces as IF          # noqa: E402
import parse_multi               # noqa: E402
from awm_env import load_tasks    # noqa: E402

SURF = os.path.join(SURFACE, "gate_surface", "work", "surfaces")

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default=os.path.join(SURFACE, "gate_caller/pools/pool_smoke20.json"))
ap.add_argument("--out", default=os.path.join(MCP_ROOT, "work/verl/awm"))
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--family", default=os.environ.get("CALLER_FAMILY", "qwen"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

pool = json.load(open(a.pool))
system = parse_multi.system_for(a.family)

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

rows, lens = [], []
for i, rec in enumerate(pool):
    sc, ti = rec["scenario"], rec["task_idx"]
    tools = json.load(open(os.path.join(SURF, f"{sc}.json")))["tools"]
    task = load_tasks(sc)[ti]
    specs = IF.build("RAW", tools, task).specs()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    n = len(tok.apply_chat_template(msgs, tools=specs, add_generation_prompt=True, tokenize=True))
    lens.append((n, sc, len(tools)))
    # TRIAGE v4: carry the per-task group size through to the parquet when the pool has one.
    # Absent on every other arm, so the column simply does not exist and the trainer's
    # per-row-repeat path is never entered.
    _vark = {"vark_k": int(rec["vark_k"])} if "vark_k" in rec else {}
    rows.append({
        **_vark,
        "data_source": "awm",
        "prompt": [{"role": "user", "content": task}],
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"scenario": sc, "task_idx": ti, "index": i,
                       "n_tools": len(tools), "turn0_prompt_tokens": n},
    })

lens.sort(reverse=True)
print(f"pool={a.pool}  rows={len(rows)}")
print("turn-0 prompt tokens (system + task + RAW tool surface):")
print(f"  max {lens[0][0]:6d}  ({lens[0][1]}, {lens[0][2]} tools)")
print(f"  p50 {lens[len(lens)//2][0]:6d}")
print(f"  min {lens[-1][0]:6d}  ({lens[-1][1]}, {lens[-1][2]} tools)")
print(f"  -> data.max_prompt_length must be >= {lens[0][0]}")

import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)
tr = os.path.join(a.out, "train.parquet")
df.to_parquet(tr)
df.to_parquet(os.path.join(a.out, "test.parquet"))
print(f"wrote {tr} rows={len(df)}")
