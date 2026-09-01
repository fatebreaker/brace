#!/usr/bin/env python3
"""
bench_bfcl.py -- transfer evaluation of already-trained checkpoints on a
PUBLIC benchmark (BFCL v4, multi-turn split), for PLAN_TRIAGE.md section 5b
item 4, replacing the dropped MCP-Atlas cell.

EVAL ONLY. This script never trains. It takes one checkpoint, serves it, runs
the benchmark's OWN harness and its OWN evaluator against it, and emits one
JSON record per task so the result drops into the same paired-analysis
machinery we use for the in-domain cells.

--------------------------------------------------------------------------
WHY THIS BENCHMARK (the PI's constraint: NO LLM-as-judge)
--------------------------------------------------------------------------
BFCL's scoring is fully programmatic. Two independent checks, both verified
by reading the source at the pinned commit (see BFCL_COMMIT below):

  1. There is no judge anywhere in the evaluator. Over the whole tree
     bfcl_eval/eval_checker/**.py:
         grep -rniE "judge|llm_as|as_a_judge"                  -> 0 hits
         grep -rniE "import openai|from openai|litellm|anthropic|
                     requests\\.post|\\.chat\\.completions"      -> 0 hits
     The only network call under eval_checker/ is SerpAPI inside
     multi_turn_eval/func_source_code/web_search.py, which belongs to the
     `web_search` CATEGORY we do not run, not to the scorer.

  2. The multi-turn metric is a Python-object state diff. Verbatim from
     bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py:

         def state_checker(model_instances: dict, ground_truth_instances: dict):
             \"\"\"
             Checks if, after executing the function calls, the model_instance has
             the same state (defined by the attributes) as the ground_truth_instance.
             It checks if every instance in the model_instances has the same
             attributes as their corresponding instance (of the same class) from
             ground_truth_instances.
             \"\"\"

     and the comparison itself, `_compare_instances`, is a plain attribute
     walk with `if model_attr != ground_truth_attr`. The complementary
     `response_checker` is an unordered subsequence test over returned
     values. Single-turn categories go through ast_checker (AST parse +
     type/value match). Nothing calls a model.

--------------------------------------------------------------------------
WHAT WE RUN
--------------------------------------------------------------------------
  code     https://github.com/ShishirPatil/gorilla   Apache License 2.0
           berkeley-function-call-leaderboard, pinned at
           BFCL_COMMIT below, VERSION_PREFIX "BFCL_v4".
  data     ships inside the package, bfcl_eval/data/*.json (one JSON object
           per line). No download, no credentials.

  default categories: the four multi-turn subsets, 200 entries each, 800 total
      multi_turn_base            200
      multi_turn_miss_func       200
      multi_turn_miss_param      200
      multi_turn_long_context    200
  optional `--categories non_live`: the seven single-turn AST subsets, 1390
  total (simple_python 400, irrelevance 240, multiple/parallel/
  parallel_multiple 200 each, simple_java 100, simple_javascript 50). Same
  judge-free guarantee, one completion per entry, no execution -- cheap to add.
  `--categories both` = 2190 tasks.

  THE COMPARABLE PROTOCOL. TRACE (arXiv:2606.11119, Tencent/Tsinghua) trains
  and evaluates on exactly this split. Appendix E.1.3, verbatim:

      "We use the multi-turn split of BFCL v4 (Patil et al., 2025), rather than
       the full BFCL v4 suite, which also contains many single-turn, live,
       memory, and web-search categories. We use the four agentic multi-turn
       subsets ...: base, long-context, missing-function, and missing-parameter.
       We split these multi-turn examples into an 80% training portion and a
       20% held-out test portion."
      "We follow the official BFCL evaluation protocol and report success rates
       over the base, missing-function, missing-parameter, and long-context
       subsets. Validation uses four sampled trajectories per example, so subset
       scores are reported as avg@4."

  Their Table 5, Qwen3-8B backbone, Function Calling columns
  (Base / Miss-Func / Miss-Param / Long / Avg):

      ReAct   36.7  21.5  31.1  22.3  28.0     <- untrained Qwen3-8B, our anchor
      GRPO    58.5  31.0  45.8  38.7  43.5
      PCL     59.0  32.1  46.8  39.3  44.3
      TreePO  58.9  32.0  46.5  39.4  44.2
      TRACE   61.2  34.4  48.8  40.4  46.2

  CAVEAT, and it is not small: TRACE evaluates on THEIR OWN 20% held-out
  portion of the multi-turn split, not on all 800 entries, and their split is
  not published. Our numbers are on the full 800 and are therefore NOT
  digit-comparable with that table. It is a scale anchor for what an 8B policy
  scores here (~28 untrained, ~44-46 after agentic RL), not a leaderboard row.
  Report it as such.

--------------------------------------------------------------------------
PIPELINE
--------------------------------------------------------------------------
  merge     LoRA adapter + base -> a merged HF dir     (surface/verl_rl/merge_lora.py)
  serve     merged dir -> OpenAI-compatible endpoint   (vLLM, mcp_vllm env)
  generate  `bfcl generate --skip-server-setup`        (benchmark's own harness)
  evaluate  `bfcl evaluate`                            (benchmark's own checkers)
  records   result+score json -> records.jsonl         (one row per task)

Stages are selectable with --stages so a crashed run resumes without redoing
generation, which is the expensive part.

Two facts about BFCL's OSS path drive the wiring, both read out of
bfcl_eval/model_handler/local_inference/base_oss_handler.py:

  * it posts to /v1/COMPLETIONS, not /v1/chat/completions, with a prompt
    string it formats itself, and it parses `<tool_call>` XML back out with
    its own regex (qwen_fc.py `_extract_tool_calls`). So vLLM does NOT need
    --enable-auto-tool-choice / --tool-call-parser here; unlike the MCP-Atlas
    harness, BFCL never asks vLLM to emit `tool_calls` on the wire.
  * `self.client.completions.create(model=self.model_path_or_id, ...)` sends
    the value of --local-model-path as the model id. That is why stage_serve
    passes the merged directory path itself to --served-model-name.

--------------------------------------------------------------------------
WHAT records.jsonl IS FOR
--------------------------------------------------------------------------
One row per (task, sample), carrying the five fields main_table.py /
rl_learning_curve.py / plot_curve.py already key on -- scenario, task_idx,
seed, interface, reward -- so a BFCL cell drops into the existing paired
machinery with no code change:

    main_table.load_cell(records.jsonl) -> {("bfcl", task_idx, seed): 0/1}

`scenario` is "bfcl" for the multi-turn set and "bfcl_non_live"/"bfcl_both"
otherwise, because task_idx is a position WITHIN the category set: two sets
sharing a scenario name would collide on (scenario, task_idx, seed) while
meaning different tasks.

BFCL's own metric is already binary per entry, so unlike the MCP-Atlas cell
there is no continuous score to fall back on and no threshold to choose. The
mapping is lossless.

--------------------------------------------------------------------------
PREREQUISITES (one-time, none of them need a GPU)
--------------------------------------------------------------------------
  1. git clone https://github.com/ShishirPatil/gorilla   ($W/gorilla)
     git -C $W/gorilla checkout <BFCL_COMMIT>
  2. an interpreter with bfcl_eval on top of mcp_vllm:
        $MCP_ENVS/mcp_vllm/bin/python -m venv --system-site-packages $W/bfcl_venv
        $W/bfcl_venv/bin/pip install -e $W/gorilla/berkeley-function-call-leaderboard
        $W/bfcl_venv/bin/pip install soundfile
     soundfile is not a BFCL dependency. It is needed because
     constants/model_config.py imports every handler eagerly, including
     api_inference/qwen.py -> qwen_agent -> soundfile, and mcp_vllm already
     has qwen_agent without it. Without soundfile `bfcl generate` dies at
     import, before it ever reaches our model.
  No sandbox image, no node, no judge endpoint, no API keys. That is the
  whole setup; it is the cheapest of the benchmarks we looked at.

  NOTE on the venv: bfcl_eval pins numpy==1.26.4, which lands in bfcl_venv and
  shadows mcp_vllm's numpy 2.x. That is harmless HERE only because vLLM runs
  as a separate process out of mcp_vllm/bin/vllm, never out of bfcl_venv. Do
  not try to launch vLLM with bfcl_venv's interpreter.

--------------------------------------------------------------------------
LAUNCH, one per checkpoint. NOT on the login node.
--------------------------------------------------------------------------
    R=$BRACE_ROOT
    W=$BRACE_WORK/bfcl

  One card is enough. The reference runs used a single 141G H200 with 8 CPUs
  and 120G of host memory, so --cpus-per-task must not exceed the allocation's
  own CPU count.

    srun --jobid=$POD --ntasks=1 --overlap --gres=gpu:1 --cpus-per-task=8 \
         --mem=120G bash -c "
      . $R/env.sh
      export HF_HOME=$R/hf_cache
      $W/bfcl_venv/bin/python $R/surface/verl_rl/bench_bfcl.py \
        --arm a8T5k --step 25 \
        --ckpt $R/work/verl/ckpt_a8T5k/global_step_25/actor \
        --merged-out $W/merged_a8T5k_step25 \
        --out-dir $W/run_a8T5k_step25 \
        --bfcl-repo $W/gorilla/berkeley-function-call-leaderboard \
        --bfcl-python $W/bfcl_venv/bin/python"

  and the same with:
    --arm a8T3g2 --step 30 --ckpt $R/work/verl/ckpt_a8T3g2/global_step_30/actor
    --arm a8F    --step 20 --ckpt $R/work/verl/ckpt_a8F/global_step_20/actor
    --arm base   --step 0  --ckpt $R/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

  The base-policy arm skips merging automatically (it is already a full HF
  dir with no lora_train_meta.json). Run the arms SEQUENTIALLY: one card, and
  each arm wants the whole of it at --gpu-memory-utilization 0.85.

  Add `--categories non_live --out-dir $W/run_<arm>_nonlive` for a second,
  much cheaper cell (1390 single-turn AST tasks, one completion each). It
  lands under scenario "bfcl_non_live" so it cannot collide with the
  multi-turn cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse the process/probe/merge machinery from the MCP-Atlas runner rather than
# reimplementing it. These are all module-level definitions with no import-time
# side effects beyond csv.field_size_limit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_transfer import (  # noqa: E402
    MCP_ROOT,
    VERL_PY,
    VLLM_BIN,
    Service,
    free_port,
    log,
    resolve_base,
    stage_merge,
    tail,
)

# --------------------------------------------------------------------------
# Benchmark pin
# --------------------------------------------------------------------------
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
VERSION_PREFIX = "BFCL_v4"
BFCL_MODEL = "Qwen/Qwen3-8B-FC"  # registry name -> QwenFCHandler (native <tool_call>)

MULTI_TURN_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]
# Single-turn AST categories. Judge-free too, and nearly free to run: no
# execution, no state, one completion per entry.
NON_LIVE_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]
CATEGORY_SETS = {
    "multi_turn": MULTI_TURN_CATEGORIES,
    "non_live": NON_LIVE_CATEGORIES,
    "both": MULTI_TURN_CATEGORIES + NON_LIVE_CATEGORIES,
}
# sha256 of the sorted "<category>\t<id>" listing, checked at run time so a
# dataset revision bump fails loudly instead of silently changing the cell.
TASK_INDEX_SHA256 = {
    "multi_turn": "cd489490938d51a60edc8a0a7cffeedf00c7cfa9193233221ba6bd4ac9f1020c",  # 800
    "non_live": "f31266c092b51ca9aa93fc404291680a9ed13c99367c91e692016dd72e676161",   # 1390
    "both": "26a992c648b39f25a1b734c796457518cb6d15daa3f8cfbbaa42619e4bd33abc",       # 2190
}

_NUM = re.compile(r"(\d+)")


def natural_key(s: str):
    return tuple(int(p) if p.isdigit() else p for p in _NUM.split(s))


# ==========================================================================
# Task set
# ==========================================================================

def group_of(category: str) -> str:
    """Mirror of bfcl_eval.utils.get_directory_structure_by_category for the
    categories we run: multi_turn_* -> 'multi_turn', the non-live AST ones ->
    'non_live'. Memory/web_search (which nest one level deeper) are excluded."""
    if category.startswith("multi_turn"):
        return "multi_turn"
    return "non_live"


def data_file(bfcl_repo: Path, category: str) -> Path:
    return bfcl_repo / "bfcl_eval" / "data" / f"{VERSION_PREFIX}_{category}.json"


def load_task_index(bfcl_repo: Path, categories: list[str]) -> tuple[dict[str, int], str]:
    """{test_id: task_idx} plus the sha256 that pins it.

    task_idx is the position in the natural-sorted (category, id) listing.
    Natural sort so multi_turn_base_2 precedes multi_turn_base_10; the order
    only has to be deterministic, but a readable one makes the records easier
    to eyeball against the score files.
    """
    ids: list[str] = []
    for cat in categories:
        p = data_file(bfcl_repo, cat)
        if not p.exists():
            raise SystemExit(f"missing benchmark data file {p}")
        for line in p.read_text().splitlines():
            if line.strip():
                ids.append(json.loads(line)["id"])
    ids.sort(key=natural_key)
    blob = "\n".join(ids) + "\n"
    return {tid: i for i, tid in enumerate(ids)}, hashlib.sha256(blob.encode()).hexdigest()


# ==========================================================================
# Stages
# ==========================================================================

def stage_serve_bfcl(a, model_dir: Path, port: int) -> Service:
    """
    vLLM, serving the merged dir under its own PATH as the model id.

    BFCL's OSS handler sends `model=self.model_path_or_id`, which is whatever
    we pass to `bfcl generate --local-model-path`. vLLM 404s on an unknown
    model id, so --served-model-name has to be that same string. We register
    the short name too, which is what shows up in logs.

    No --enable-auto-tool-choice / --tool-call-parser: BFCL hits
    /v1/completions with a pre-formatted prompt and parses `<tool_call>` XML
    itself (qwen_fc.py `_extract_tool_calls`), so vLLM must NOT rewrite the
    output into wire-level tool_calls. This is the one substantive difference
    from bench_transfer.stage_serve.

    --max-model-len matters here. BFCL reads max_position_embeddings from the
    merged config (262144 for Qwen3-VL-8B) and then asks for
    min(4096, max_context - input - 2) completion tokens. It does not know
    what vLLM was started with, so if a long_context prompt plus 4096 exceeds
    --max-model-len, vLLM 400s that entry. Keep --max-model-len well above
    the longest prompt; the default below is sized for an H200.
    """
    # Compute nodes carry no CUDA toolkit, and FlashInfer JIT-compiles its
    # sampling kernel at warmup -- without nvcc the EngineCore dies before
    # the server ever answers /health. The native sampler needs no toolkit.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    cmd = [str(VLLM_BIN), "serve", str(model_dir),
           "--served-model-name", str(model_dir), a.served_model_name,
           "--host", "127.0.0.1", "--port", str(port),
           "--max-model-len", str(a.max_model_len),
           "--gpu-memory-utilization", str(a.gpu_memory_utilization),
           "--tensor-parallel-size", str(a.tensor_parallel_size),
           "--enforce-eager"]
    if a.chat_template:
        cmd += ["--chat-template", a.chat_template]
    cmd += shlex.split(a.vllm_extra_args)
    # Probe /health, not /v1/models: no --api-key is set here, but /health is
    # the cheap unguarded endpoint either way.
    return Service("vllm", cmd, Path(a.out_dir) / "logs" / "vllm.log",
                   f"http://127.0.0.1:{port}/health", timeout=a.serve_timeout)


def bfcl_env(a, port: int) -> dict:
    """
    BFCL_PROJECT_ROOT relocates result/, score/ and .env out of the package.
    One project root per arm, so two arms never share a lock dir or a
    result file.

    The port has to go in the .env, not just the environment: __main__.py
    calls load_dotenv(..., override=True), so anything the .env DOES define
    beats the inherited environment. We define it there and keep the process
    env in sync.
    """
    root = Path(a.out_dir) / "bfcl_root"
    root.mkdir(parents=True, exist_ok=True)
    (root / ".env").write_text(
        "LOCAL_SERVER_ENDPOINT=127.0.0.1\n"
        f"LOCAL_SERVER_PORT={port}\n"
    )
    return {
        "BFCL_PROJECT_ROOT": str(root),
        "LOCAL_SERVER_ENDPOINT": "127.0.0.1",
        "LOCAL_SERVER_PORT": str(port),
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def stage_generate(a, model_dir: Path, port: int, categories: list[str], sample: int) -> None:
    """`bfcl generate`, the benchmark's own harness, against our served model.

    --skip-server-setup stops BFCL from launching its own vLLM/sglang; it
    still loads the tokenizer and config from --local-model-path to size the
    request, which is why that flag points at the merged dir and not at the
    HF id.
    """
    env = bfcl_env(a, port)  # also creates the project root and writes its .env
    root = Path(a.out_dir) / "bfcl_root"
    cmd = [str(a.bfcl_python), "-m", "bfcl_eval", "generate",
           "--model", BFCL_MODEL,
           "--test-category", ",".join(categories),
           "--skip-server-setup",
           "--local-model-path", str(model_dir),
           "--backend", "vllm",
           "--result-dir", str(root / f"result_s{sample}"),
           "--temperature", str(a.temperature if sample == 0 else max(a.temperature, 0.7)),
           "--num-threads", str(a.num_threads),
           "--allow-overwrite"]
    if a.run_ids_file:
        shutil.copy(a.run_ids_file, root / "test_case_ids_to_generate.json")
        cmd += ["--run-ids"]
    log("generate: " + " ".join(shlex.quote(c) for c in cmd))
    logfile = Path(a.out_dir) / "logs" / f"generate_s{sample}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, **env}).returncode
    if rc != 0:
        raise SystemExit(f"bfcl generate exited {rc}; tail of {logfile}:\n" + tail(logfile))
    log(f"generate done -> {root / f'result_s{sample}'}")


def stage_evaluate(a, port: int, categories: list[str], sample: int) -> None:
    """`bfcl evaluate`, the benchmark's own checkers. CPU only, no network."""
    root = Path(a.out_dir) / "bfcl_root"
    cmd = [str(a.bfcl_python), "-m", "bfcl_eval", "evaluate",
           "--model", BFCL_MODEL,
           "--test-category", ",".join(categories),
           "--result-dir", str(root / f"result_s{sample}"),
           "--score-dir", str(root / f"score_s{sample}")]
    if a.run_ids_file:
        cmd += ["--partial-eval"]
    log("evaluate: " + " ".join(shlex.quote(c) for c in cmd))
    logfile = Path(a.out_dir) / "logs" / f"evaluate_s{sample}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, **bfcl_env(a, port)}).returncode
    if rc != 0:
        raise SystemExit(f"bfcl evaluate exited {rc}; tail of {logfile}:\n" + tail(logfile))
    log(f"evaluate done -> {root / f'score_s{sample}'}")


def flatten_sum(v):
    """Total of an arbitrarily nested list of numbers; None stays None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, list):
        parts = [flatten_sum(x) for x in v]
        parts = [p for p in parts if p is not None]
        return sum(parts) if parts else None
    return None


def read_cell(a, categories: list[str], sample: int) -> dict[str, dict]:
    """{test_id: {...}} for one sample, read out of BFCL's own files.

    The score file holds a summary header followed by ONLY the FAILED entries
    -- eval_runner.multi_turn_runner appends to `result` only when
    `entry_result["valid"]` is false, and save_eval_results does
    `result.insert(0, header)`. So a task passed iff it was attempted (present
    in the result file) and absent from the score file's failure rows. That is
    the whole reward derivation; there is no thresholding anywhere.
    """
    root = Path(a.out_dir) / "bfcl_root"
    model_dir = BFCL_MODEL.replace("/", "_")
    out: dict[str, dict] = {}
    for cat in categories:
        grp = group_of(cat)
        rp = root / f"result_s{sample}" / model_dir / grp / f"{VERSION_PREFIX}_{cat}_result.json"
        sp = root / f"score_s{sample}" / model_dir / grp / f"{VERSION_PREFIX}_{cat}_score.json"
        if not rp.exists():
            log(f"WARNING: no result file for {cat} at {rp}; skipping category")
            continue
        attempted = []
        for line in rp.read_text().splitlines():
            if line.strip():
                attempted.append(json.loads(line))
        failures: dict[str, dict] = {}
        header = {}
        if sp.exists():
            rows = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()]
            if rows:
                header = rows[0]
            for r in rows[1:]:
                failures[str(r.get("id"))] = r
        else:
            log(f"WARNING: no score file for {cat} at {sp}; every entry will read as unscored")
        for e in attempted:
            tid = str(e["id"])
            fail = failures.get(tid)
            out[tid] = {
                "category": cat,
                "reward": 0 if fail else (1 if sp.exists() else 0),
                "scored": sp.exists(),
                "error_type": (fail or {}).get("error", {}).get("error_type")
                if isinstance((fail or {}).get("error"), dict) else None,
                "n_turns": len(e["result"]) if isinstance(e.get("result"), list) else None,
                # BFCL reports these per turn AND per step within the turn, so
                # they arrive as [[float]]. The in-domain cells carry scalars.
                "latency": flatten_sum(e.get("latency")),
                "input_token": flatten_sum(e.get("input_token_count")),
                "output_token": flatten_sum(e.get("output_token_count")),
            }
        if header:
            log(f"  {cat}: accuracy={header.get('accuracy')} "
                f"{header.get('correct_count')}/{header.get('total_count')}")
    return out


def stage_records(a, categories: list[str], task_index: dict[str, int],
                  index_sha: str, records: Path, manifest: dict) -> None:
    """Emit one JSON object per (task, sample) for the paired machinery.

    Verified against main_table.load_cell:
        k = (d.get("scenario"), d.get("task_idx"), d.get("seed"))
        out[k] = int(d["reward"] > 0)
    so:  scenario = "bfcl"
         task_idx = position in the sha256-pinned sorted id list
         seed     = a.seed + sample
         interface= "RAW", matching the in-domain cells
         reward   = BFCL's own per-entry pass/fail, unmodified
    """
    # task_idx is a position within THIS category set, so two different sets
    # must not share a scenario name or their cells would collide on
    # (scenario, task_idx, seed) while meaning different tasks.
    scenario = "bfcl" if a.categories == "multi_turn" else f"bfcl_{a.categories}"
    n = 0
    with open(records, "w") as f:
        for sample in range(a.n_samples):
            cell = read_cell(a, categories, sample)
            if not cell:
                log(f"WARNING: sample {sample} produced no rows")
            for tid, d in sorted(cell.items(), key=lambda kv: natural_key(kv[0])):
                if tid not in task_index:
                    log(f"WARNING: {tid} is not in the pinned task index; skipping")
                    continue
                rec = {
                    # ---- fields our paired tooling keys on, unchanged ----
                    "scenario": scenario,
                    "task_idx": task_index[tid],
                    "seed": a.seed + sample,
                    "interface": "RAW",
                    "reward": d["reward"],
                    "n_turns": d["n_turns"],
                    "wall_s": d["latency"],
                    "verifier_status": "success" if d["scored"] else "unscored",
                    # ---- benchmark-native ----
                    "bench": "bfcl_v4",
                    "subset": a.categories,
                    "category": d["category"],
                    "arm": a.arm,
                    "step": a.step,
                    "task_id": tid,
                    "sample": sample,
                    "error_type": d["error_type"],
                    "input_token": d["input_token"],
                    "output_token": d["output_token"],
                    "bfcl_commit": BFCL_COMMIT,
                    "bfcl_model": BFCL_MODEL,
                    "task_index_sha256": index_sha,
                    "run_id": manifest["run_id"],
                }
                f.write(json.dumps(rec) + "\n")
                n += 1

    rows = [json.loads(l) for l in records.read_text().splitlines()]
    expected = len(task_index) * a.n_samples
    log(f"wrote {n} records -> {records}")
    if n != expected and not a.run_ids_file:
        log(f"WARNING: {n} records but {expected} expected "
            f"({len(task_index)} tasks x {a.n_samples} samples). Entries missing "
            f"from the result file were never attempted -- check the generate log.")
    scored = [r for r in rows if r["verifier_status"] == "success"]
    if scored:
        k = sum(r["reward"] for r in scored)
        log(f"  pass rate = {k}/{len(scored)} = {100.0*k/len(scored):.1f}")
        by_cat: dict[str, list[int]] = {}
        for r in scored:
            by_cat.setdefault(r["category"], []).append(r["reward"])
        for cat in sorted(by_cat):
            v = by_cat[cat]
            log(f"    {cat:26s} {sum(v):4d}/{len(v):4d} = {100.0*sum(v)/len(v):5.1f}")


# ==========================================================================
# mock: dry run with no GPU
# ==========================================================================

MOCK_SERVER_SRC = r'''
"""Stub OpenAI /v1/completions endpoint. Two behaviours, chosen per request:

  empty   -> returns a plain sentence with no <tool_call>, so every entry fails
             the state check. Exercises decode -> checker -> records end to end.
  oracle  -> not served here; see stage_oracle, which bypasses generation
             entirely and feeds ground truth straight into the real evaluator.
"""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

MODE = sys.argv[2] if len(sys.argv) > 2 else "empty"
TEXT = {
    "empty": "I need more information before I can help with that.",
}[MODE]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
        if self.path.endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]}); return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        self._send(200, {
            "id": "cmpl-mock", "object": "text_completion", "created": 0,
            "model": req.get("model", "mock"),
            "choices": [{"index": 0, "text": TEXT, "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


def start_mock(a, port: int) -> Service:
    stub = Path(a.out_dir) / "logs" / "mock_server.py"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(MOCK_SERVER_SRC)
    return Service("mock-vllm", [sys.executable, str(stub), str(port), "empty"],
                   Path(a.out_dir) / "logs" / "mock_vllm.log",
                   f"http://127.0.0.1:{port}/health", timeout=60)


def load_param_order(bfcl_repo: Path) -> dict[str, list[str]]:
    """{function name: [parameter names, in declaration order]}.

    Needed because BFCL's ground truth mixes positional and keyword calls --
    `sort('final_report.pdf')` sits next to `cd(folder='document')` -- while a
    `<tool_call>` payload is a NAMED arguments object. Without this map the
    positional ones silently become `{"arguments": {}}`, which is how the
    first version of the oracle check scored 2/3 instead of 1.0.
    JSON object order is insertion order, so `properties` gives the signature.
    """
    order: dict[str, list[str]] = {}
    doc_dir = bfcl_repo / "bfcl_eval" / "data" / "multi_turn_func_doc"
    # memory_*.json and web_search.json belong to the agentic categories we
    # never run, and they redefine names like core_memory_add with a different
    # signature. Loading them would only manufacture spurious conflicts.
    for f in sorted(doc_dir.glob("*.json")):
        if f.stem.startswith("memory_") or f.stem == "web_search":
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            params = list((d.get("parameters") or {}).get("properties", {}).keys())
            prev = order.get(d["name"])
            if prev is not None and prev != params:
                log(f"WARNING: conflicting signature for {d['name']}: {prev} vs {params}")
            order[d["name"]] = params
    return order


def func_call_to_tool_call(call: str, param_order: dict[str, list[str]]) -> dict | None:
    """`cd(folder='document')` -> {"name": "cd", "arguments": {"folder": "document"}}

    BFCL stores ground truth as Python call strings. ast.parse gives us the
    name and the arguments without eval'ing anything; positional arguments are
    named via param_order.
    """
    import ast
    try:
        node = ast.parse(call.strip(), mode="eval").body
        if not isinstance(node, ast.Call):
            return None
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        args = {}
        names = param_order.get(name, [])
        for i, pos in enumerate(node.args):
            if i >= len(names):
                log(f"WARNING: {name} got positional arg {i} but the doc lists {names}")
                return None
            args[names[i]] = ast.literal_eval(pos)
        for kw in node.keywords:
            args[kw.arg] = ast.literal_eval(kw.value)
        return {"name": name, "arguments": args}
    except Exception as e:
        log(f"WARNING: could not parse ground-truth call {call!r}: {e}")
        return None


# --------------------------------------------------------------------------
# Single-turn (non-live) oracle: turning BFCL's possible_answer back into a
# model-shaped `<tool_call>` string.
# --------------------------------------------------------------------------
# BFCL stores a single-turn answer as {func_name: {param: [accepted values]}}.
# "" inside that list means "the parameter may be omitted", so it is never the
# value we emit. Dict-valued parameters nest the same alternative-list shape
# one level deeper (dict_checker reads {key: [alternatives]}), while
# list-valued parameters hold literal lists (list_checker compares them whole).
_SKIP = object()

ORACLE_PROSE = "I'm sorry, I don't have enough information to answer that."
# A syntactically perfect call to a function that exists in no BFCL entry: it
# decodes cleanly, so it is a VALUE failure for the AST categories and a
# "should not have called anything" failure for irrelevance.
NEGATIVE_CALL = [{"name": "definitely_not_a_real_function",
                  "arguments": {"unexpected_param": 1}}]


def _pick(alts):
    """One concrete value out of BFCL's list of accepted alternatives."""
    if not isinstance(alts, list):
        return _resolve(alts)
    for a in alts:
        if isinstance(a, str) and a == "":
            continue          # optional-parameter marker, never a real value
        return _resolve(a)
    return _SKIP              # every alternative was "" -> omit the parameter


def _resolve(v):
    if isinstance(v, dict):
        out = {}
        for k, alts in v.items():
            x = _pick(alts)
            if x is not _SKIP:
                out[k] = x
        return out
    if isinstance(v, list):
        return [_resolve(x) for x in v]
    return v


def _source_literal(value, ptype: str, nested: str | None, language: str) -> str:
    """Java/JS arguments reach the checker as SOURCE TEXT, not as JSON values.

    simple_function_checker rejects any non-str value for those two languages
    and then runs it through java_type_converter / js_type_converter. So the
    oracle has to emit the literal a Java/JS programmer would write. We pick
    the candidate whose round trip through the benchmark's OWN converter
    reproduces the value the checker will compare against; if none does, the
    first candidate is emitted anyway so the failure is visible rather than
    silently papered over.
    """
    from bfcl_eval.eval_checker.ast_eval.type_convertor.java_type_converter import (
        java_type_converter)
    from bfcl_eval.eval_checker.ast_eval.type_convertor.js_type_converter import (
        js_type_converter)
    conv = java_type_converter if language == "java" else js_type_converter

    if isinstance(value, str):
        cands = [value]
    elif isinstance(value, bool):
        cands = ["true" if value else "false"]
    elif isinstance(value, int):
        cands = [str(value), f"{value}L", f"{value}n"]
    elif isinstance(value, float):
        cands = [str(value), f"{value}f"]
    else:
        cands = [json.dumps(value), str(value)]
    for c in cands:
        try:
            got = conv(c, ptype, nested)
        except Exception:
            continue
        if got == value and type(got) == type(value):
            return c
    return cands[0]


_source_literal.repo = ""


def non_live_oracle_text(entry: dict, gt_calls: list, cat: str, mode: str) -> str:
    """The model-shaped string the oracle feeds to the real evaluator."""
    if mode == "prose" or (mode == "gold" and "irrelevance" in cat):
        return ORACLE_PROSE
    if mode == "wrong":
        return _tool_call_text(NEGATIVE_CALL)
    specs = {f["name"]: f for f in entry.get("function", [])}
    language = ("java" if cat.endswith("_java")
                else "javascript" if cat.endswith("_javascript") else "python")
    calls = []
    for call in gt_calls:
        for fname, params in call.items():
            args = {}
            for param, alts in params.items():
                v = _pick(alts)
                if v is _SKIP:
                    continue
                if language != "python":
                    props = (specs.get(fname, {}).get("parameters", {})
                             .get("properties", {}).get(param, {}))
                    ptype = props.get("type", "any")
                    nested = (props.get("items") or {}).get("type")
                    v = _source_literal(v, ptype, nested, language)
                args[param] = v
            calls.append({"name": fname, "arguments": args})
    return _tool_call_text(calls)


def _tool_call_text(calls: list[dict]) -> str:
    return "\n".join("<tool_call>\n" + json.dumps(c) + "\n</tool_call>"
                     for c in calls)


def stage_oracle(a, categories: list[str], limit: int, mode: str = "gold") -> None:
    """Feed hand-built output through the REAL evaluator and assert the score.

    This is the half of the dry run that a stub model cannot cover. The mock
    server proves the failure path (reward 0) end to end; this proves the
    success path, on the benchmark's own checkers, with no model and no GPU.
    If this does not come back at the expected accuracy, the harness wiring is
    wrong, not the policy.

    modes, and what each asserts per category:
      gold   ground truth  -> 1.0 everywhere (irrelevance's "ground truth" is
                              to call nothing, so it gets the prose answer)
      prose  no call at all-> 0.0 on the AST categories, 1.0 on irrelevance
      wrong  a well-formed call to a function that does not exist
                           -> 0.0 everywhere, including irrelevance
    `limit` <= 0 means the whole category.
    """
    root = Path(a.out_dir) / "bfcl_root"
    model_dir = BFCL_MODEL.replace("/", "_")
    repo = Path(a.bfcl_repo)
    _source_literal.repo = str(repo)
    param_order = load_param_order(repo)
    lim = limit if limit and limit > 0 else None
    expected: dict[str, float] = {}
    ids: dict[str, list[str]] = {}
    for cat in categories:
        grp = group_of(cat)
        out_dir = root / "result_oracle" / model_dir / grp
        out_dir.mkdir(parents=True, exist_ok=True)
        ans = repo / "bfcl_eval" / "data" / "possible_answer" / f"{VERSION_PREFIX}_{cat}.json"

        if grp == "multi_turn":
            if not ans.exists():
                log(f"no possible_answer file for {cat}; skipping oracle")
                continue
            rows = [json.loads(l) for l in ans.read_text().splitlines() if l.strip()][:lim]
            with open(out_dir / f"{VERSION_PREFIX}_{cat}_result.json", "w") as f:
                for row in rows:
                    turns = []
                    for turn in row["ground_truth"]:
                        calls = [func_call_to_tool_call(c, param_order) for c in turn]
                        text = _tool_call_text([c for c in calls if c])
                        # one "step" holding the whole turn, then an empty step
                        # so the handler sees the turn terminate
                        turns.append([text, "Done."])
                    f.write(json.dumps({"id": row["id"], "result": turns}) + "\n")
            ids[cat] = [r["id"] for r in rows]
            expected[cat] = 1.0
            log(f"oracle result file written for {cat} ({len(rows)} entries)")
            continue

        # ---- single turn: `result` is a plain string, not a list of turns ----
        entries = [json.loads(l) for l in
                   data_file(repo, cat).read_text().splitlines() if l.strip()][:lim]
        gt = {}
        if ans.exists():
            gt = {json.loads(l)["id"]: json.loads(l)["ground_truth"]
                  for l in ans.read_text().splitlines() if l.strip()}
        elif "irrelevance" not in cat:
            log(f"no possible_answer file for {cat}; skipping oracle")
            continue
        with open(out_dir / f"{VERSION_PREFIX}_{cat}_result.json", "w") as f:
            for e in entries:
                text = non_live_oracle_text(e, gt.get(e["id"], []), cat, mode)
                f.write(json.dumps({"id": e["id"], "result": text}) + "\n")
        ids[cat] = [e["id"] for e in entries]
        if mode == "gold":
            expected[cat] = 1.0
        elif mode == "prose":
            expected[cat] = 1.0 if "irrelevance" in cat else 0.0
        else:
            expected[cat] = 0.0
        log(f"oracle[{mode}] result file written for {cat} ({len(entries)} entries)")

    (root / "test_case_ids_to_generate.json").write_text(json.dumps(ids))

    cmd = [str(a.bfcl_python), "-m", "bfcl_eval", "evaluate",
           "--model", BFCL_MODEL,
           "--test-category", ",".join(categories),
           "--result-dir", str(root / "result_oracle"),
           "--score-dir", str(root / "score_oracle"),
           "--partial-eval"]
    log("oracle evaluate: " + " ".join(shlex.quote(c) for c in cmd))
    logfile = Path(a.out_dir) / "logs" / "evaluate_oracle.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, **bfcl_env(a, 1)}).returncode
    if rc != 0:
        raise SystemExit(f"oracle evaluate exited {rc}; tail:\n" + tail(logfile))
    ok = True
    for cat in categories:
        if cat not in expected:
            continue
        sp = (root / "score_oracle" / model_dir / group_of(cat) /
              f"{VERSION_PREFIX}_{cat}_score.json")
        if not sp.exists():
            log(f"ORACLE[{mode}] {cat}: no score file"); ok = False; continue
        h = json.loads(sp.read_text().splitlines()[0])
        got = h.get("accuracy")
        hit = (got == expected[cat])
        log(f"ORACLE[{mode}] {cat}: accuracy={got} "
            f"{h.get('correct_count')}/{h.get('total_count')} "
            f"expected={expected[cat]} {'OK' if hit else 'MISMATCH'}")
        if not hit:
            ok = False
    log(f"ORACLE CHECK [{mode}] " + ("PASSED" if ok else "FAILED"))
    if not ok and not a.oracle_soft:
        raise SystemExit(f"oracle[{mode}] did not hit the expected accuracy; "
                         "the harness wiring is wrong")


# ==========================================================================
# main
# ==========================================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="BFCL v4 transfer eval for a trained checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("what to evaluate")
    g.add_argument("--arm", required=True, help="label, e.g. a8T5k")
    g.add_argument("--ckpt", help="verl global_step_N/actor dir, or a full HF dir")
    g.add_argument("--merged-dir", help="skip merging; serve this dir")
    g.add_argument("--step", type=int, default=None, help="training step, recorded only")
    g.add_argument("--out-dir", required=True, help="run directory (put it on /scratch)")

    g = p.add_argument_group("benchmark")
    g.add_argument("--bfcl-repo", required=True,
                   help="path to gorilla/berkeley-function-call-leaderboard")
    g.add_argument("--bfcl-python", default=None,
                   help="interpreter with bfcl_eval installed (default: --bfcl-repo's venv guess)")
    g.add_argument("--categories", default="multi_turn", choices=sorted(CATEGORY_SETS),
                   help="multi_turn (800 tasks, the TRACE-comparable set), "
                        "non_live (1390 single-turn AST tasks), or both")
    g.add_argument("--print-subset", action="store_true",
                   help="print the pinned task-id list and its sha256, then exit")
    g.add_argument("--run-ids-file", default=None,
                   help="JSON {category: [ids]} to restrict the run (BFCL --run-ids)")

    g = p.add_argument_group("generation")
    g.add_argument("--n-samples", type=int, default=1, help="samples per task; 1 = pass@1")
    g.add_argument("--temperature", type=float, default=0.001, help="BFCL's default")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--num-threads", type=int, default=8)

    g = p.add_argument_group("serving")
    g.add_argument("--served-model-name", default="policy")
    g.add_argument("--max-model-len", type=int, default=131072)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    g.add_argument("--tensor-parallel-size", type=int, default=1)
    g.add_argument("--chat-template", default=None)
    g.add_argument("--vllm-extra-args", default="")
    g.add_argument("--serve-timeout", type=int, default=1800)

    g = p.add_argument_group("merging")
    g.add_argument("--base", default=None, help="base HF snapshot dir")
    g.add_argument("--base-repo", default="Qwen/Qwen3-VL-8B-Instruct")
    g.add_argument("--merged-out", default=None)
    g.add_argument("--merge-python", default=str(VERL_PY))
    g.add_argument("--merge-script",
                   default=str(MCP_ROOT / "surface" / "verl_rl" / "merge_lora.py"))
    g.add_argument("--merge-args", default="")
    g.add_argument("--force-merge", action="store_true")

    g = p.add_argument_group("control")
    g.add_argument("--stages", default="merge,serve,generate,evaluate,records")
    g.add_argument("--mock", action="store_true",
                   help="no GPU: stub completions endpoint + real harness, real "
                        "evaluator, real records; plus the oracle check")
    g.add_argument("--oracle-limit", type=int, default=3,
                   help="entries per category for the oracle check; <=0 = all")
    g.add_argument("--oracle-mode", default="gold",
                   choices=["gold", "prose", "wrong"],
                   help="gold: ground truth, expect 1.0. prose: no call at all, "
                        "expect 0.0 except irrelevance. wrong: a well-formed call "
                        "to a nonexistent function, expect 0.0 everywhere.")
    g.add_argument("--oracle-soft", action="store_true",
                   help="report the oracle mismatch instead of exiting nonzero")

    a = p.parse_args()
    a.bfcl_repo = str(Path(a.bfcl_repo).resolve())
    if a.bfcl_python is None:
        a.bfcl_python = sys.executable
    categories = CATEGORY_SETS[a.categories]

    task_index, index_sha = load_task_index(Path(a.bfcl_repo), categories)
    if a.print_subset:
        for tid, i in sorted(task_index.items(), key=lambda kv: kv[1]):
            print(f"{i}\t{tid}")
        log(f"{len(task_index)} tasks, sha256 {index_sha}")
        return 0

    pinned = TASK_INDEX_SHA256.get(a.categories)
    if pinned and pinned != index_sha:
        raise SystemExit(
            f"task index no longer reproduces for '{a.categories}':\n"
            f"  expected {pinned}\n  got      {index_sha}\n"
            "The benchmark data changed. Re-pin deliberately -- do not silently "
            "compare across revisions.")
    log(f"{len(task_index)} tasks over {len(categories)} categories, "
        f"index sha256 {index_sha}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    if a.mock:
        stages = [s for s in stages if s != "merge"]

    manifest = {
        "run_id": f"{a.arm}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "arm": a.arm, "step": a.step, "categories": a.categories,
        "n_tasks": len(task_index), "n_samples": a.n_samples,
        "task_index_sha256": index_sha,
        "bfcl_commit": BFCL_COMMIT, "bfcl_model": BFCL_MODEL,
        "argv": sys.argv, "mock": a.mock,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"run {manifest['run_id']}  stages={stages}")

    services: list[Service] = []
    try:
        model_dir = Path(a.merged_dir) if a.merged_dir else None
        if "merge" in stages:
            model_dir = stage_merge(a)
        if model_dir is None:
            model_dir = Path(a.ckpt) if a.ckpt else Path("/nonexistent")

        port = free_port()
        if "serve" in stages:
            svc = start_mock(a, port) if a.mock else stage_serve_bfcl(a, model_dir, port)
            services.append(svc.start())
            svc.wait_ready()

        if a.mock:
            # The stub answers as any model id, so point --local-model-path at
            # whatever real HF dir we have for the tokenizer.
            model_dir = Path(a.base) if a.base else resolve_base(a)
            log(f"mock: tokenizer/config from {model_dir}")

        if "generate" in stages:
            for s in range(a.n_samples):
                stage_generate(a, model_dir, port, categories, s)
        if "evaluate" in stages:
            for s in range(a.n_samples):
                stage_evaluate(a, port, categories, s)
        if "records" in stages:
            stage_records(a, categories, task_index, index_sha,
                          out / "records.jsonl", manifest)
        if a.mock or "oracle" in stages:
            stage_oracle(a, categories, a.oracle_limit, a.oracle_mode)
    finally:
        for s in reversed(services):
            s.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
