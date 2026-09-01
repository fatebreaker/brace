#!/usr/bin/env python3
"""
bench_nestful.py -- SECOND out-of-distribution tool-use benchmark for the
transfer claim (PLAN_TRIAGE.md). Companion to bench_bfcl.py.

EVAL ONLY. This script never trains. It takes one already-trained checkpoint,
serves it, renders the benchmark's OWN prompt, runs the benchmark's OWN scorer
over the raw completions, and emits one JSON record per task so the result
drops into the same paired-analysis machinery as every other cell.

==========================================================================
WHY THIS BENCHMARK
==========================================================================
BFCL v4 multi-turn already carries the transfer claim (work/analysis/
bfcl_results.md, 17 arms). One benchmark is one benchmark: a reviewer can ask
whether the TRIAGE-over-uniform gap is a property of allocation or a property
of BFCL. This cell answers that with a benchmark from a DIFFERENT FAMILY.

  NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls
  IBM Research, arXiv:2409.03797 (v3).
    code  https://github.com/IBM/NESTFUL      Apache-2.0
    data  data_v2/nestful_data.jsonl, ships INSIDE the repo (1861 rows) and is
          mirrored at https://huggingface.co/datasets/ibm-research/nestful.
          No download at run time, no credentials, no network.

Different from BFCL along every axis that matters:
  * capability      nested composition -- the OUTPUT of one call is bound to a
                    label ($var_2) and passed as the INPUT of a later call.
                    BFCL multi-turn tests stateful multi-turn execution; it has
                    no variable binding at all.
  * episode shape   single shot. One completion emits the WHOLE call graph.
                    BFCL is an interactive loop.
  * scoring         see below -- an executable end-to-end check. BFCL is a
                    Python-object state diff.
  * provenance      IBM, from MathQA + StarCoder2-Instruct. BFCL is Berkeley,
                    hand-authored multi-turn scenarios.
  * n               1861 tasks (BFCL multi-turn: 800).

CANDIDATES REJECTED, and why (all checked 2026-08-18):
  * tau-bench           DISQUALIFIED. Its user side is an LLM user-simulator
                        driven through an API provider; there is no fully
                        scripted mode. Violates the no-judge / no-API-key rule.
  * ToolBench / StableToolBench
                        DISQUALIFIED. ToolEval is GPT-4-as-judge.
  * API-Bank L1/L2      DOUBLE COUNT. Already a cell in this paper
                        (work/apibank/, appendix `app:apibank`, 0.413). Reusing
                        it would not add an independent benchmark.
  * BFCL non_live       ACCEPTABLE BUT WEAKER: 1390 single-turn AST tasks
                        (simple_python/java/javascript, multiple, parallel,
                        parallel_multiple, irrelevance) that bench_bfcl.py
                        already supports via `--categories non_live` and that
                        we have never run -- verified: every records.jsonl under
                        the bfcl workspace carries subset "multi_turn". It is
                        the same suite and the same authors, so it tests a
                        different capability, not a different benchmark. Kept
                        in reserve as a cheap third cell.
  * ACEBench            programmatic for its normal/special splits, but its
                        agent split needs an LLM user simulator, and the
                        harness is heavier than NESTFUL's for no extra
                        independence.
  * Seal-Tools          programmatic, but n is an order of magnitude smaller
                        and the metric is F1 only -- no executable check.

==========================================================================
THE SCORING IS FULLY PROGRAMMATIC (the PI's hard constraint)
==========================================================================
Verified by reading src/scorer.py, src/output_parsers.py, src/utils.py at the
pinned commit:

    grep -rniE "judge|llm_as|openai|anthropic|litellm|\\.chat\\.completions" src/
        -> 0 hits

The scorer computes five things, none of which calls a model:

  f1_intent    macro F1 over the multiset of called FUNCTION NAMES
               (sklearn MultiLabelBinarizer + f1_score, average="macro").
  f1_slot      macro F1 over "arg = value" strings per function.
  accuracy_combined
               positional accuracy over the aligned sequence of rendered
               calls "name(a = 1, b = 2)"; continuous in [0,1].
  percentage_times_full_score
               1 iff accuracy_combined == 1, i.e. the ENTIRE call sequence,
               arguments included, matches gold after variable grounding.
  win_rate     EXECUTES the predicted call graph. Each call is dispatched to
               the benchmark's own Python implementation shipped in
               data_v2/executable_functions/, the $var_N.result$ references are
               resolved from earlier returns, and the final value is compared
               to the recorded gold_answer. Verbatim from scorer.py:

                   res = func(*arg_val_list)
                   ...
                   if pred_ans == gold_ans: return True

               This is the strongest form of programmatic scoring available in
               this space: it credits any trajectory that actually computes the
               right answer, not only the gold one.

PRIMARY METRIC for the paired tests: win_rate, mapped to reward in {0,1}.
Secondary, all carried per record: full-sequence match, partial-match accuracy,
plus the parse-error flag. Aggregate F1s land in summary.json (they are
set-level statistics and are not meaningful per item).

MEASURED CEILING (run `--oracle --oracle-limit 0`, 2026-08-18, 2m06s on a login
node). Feeding the GOLD call sequence in as the completion scores:
    full_match  1861/1861 = 1.0000    -- no intrinsic ceiling
    win_rate    1857/1861 = 0.9979    -- four tasks whose own gold sequence does
                                         not re-execute to the recorded
                                         gold_answer:
      d448887d-7cc2-4ae2-90d4-a5b97f5dddc9  c0147c17-efc0-44df-abbf-d32e2f988f91
      a1ad66f4-95ea-42dc-b7b4-7bf9ee883751  c5b3ca75-d50d-4e5a-aab1-6f95491f4538
Those four are benchmark defects, identical for every arm, so they cancel in
any paired contrast. Do not "fix" them; report the ceiling.

==========================================================================
SANDBOXING THE win_rate STAGE -- read this before touching stage_score
==========================================================================
win_rate executes benchmark-shipped Python with MODEL-CHOSEN arguments. An
audit of the 4150 implementation files reachable from the 1861 tasks' tool
lists found, among them, 195 that call open(), 100 that import os, 16 with a
`while True`, 17 with eval(), 12 touching urllib, 11 sockets, 4 subprocess and
4 that call input(). The benchmark's own guard is signal.alarm(10), which does
NOT interrupt a single long C-level operation -- `power(2, 10**9)` from a
hallucinated argument would wedge the run forever.

So every win_rate evaluation runs in a FORKED CHILD that:
    * setrlimit(RLIMIT_CPU, 15s)     -> SIGXCPU kills C-level spins
    * setrlimit(RLIMIT_AS, 4 GiB)    -> caps runaway allocation
    * stdin <- /dev/null             -> input() raises instead of blocking
and the parent join()s with a wall-clock timeout on top (covers blocking I/O,
which RLIMIT_CPU does not). A child that does not come back is scored 0 and
flagged win_timeout in the record. Nothing else in the pipeline executes
model-controlled anything.

==========================================================================
THE PROMPT
==========================================================================
Taken from the benchmark's own src/PROMPTS.json entry "LLaMa-3.1", split at
its Llama chat-scaffolding tokens into a system part and a user part, and
formatted with the benchmark's own get_icl_str() over the benchmark's own
src/icl_examples.json (first --icl-count examples; their sample_ids are NOT in
the 1861-row eval set -- checked). The two parts are then handed to the served
model as system/user messages so vLLM applies the model's own chat template.
Both parts are sha256-pinned (PROMPT_SHA256) so a repo bump fails loudly.

ONE DOCUMENTED DEVIATION. get_instruct_data() feeds the Llama template
FUNCTION_STR=json.dumps(sample["tools"]) AFTER eval.py has already replaced
sample["tools"] with json.dumps(...) of the list -- i.e. the spec reaches the
model double-encoded, as one big backslash-escaped string literal. The repo's
own Granite path json.loads()es it back first, so the double encoding is a bug
in the Llama branch, not a design choice. We embed the spec as plain JSON
(--function-str json, the default; pass --function-str double to reproduce the
released behaviour byte for byte). Applied identically to every arm, so it
cannot favour one; it exists so the cell is not floored by an unreadable
prompt. Our numbers are therefore NOT digit-comparable with the paper's
Llama rows -- and they are not meant to be. The claim this cell carries is a
WITHIN-CELL paired contrast between our own arms.

RESPONSE EXTRACTION, also arm-independent and also documented: the benchmark's
parse_llama_3_output() json.loads() the completion and, failing that, glues a
'[' onto the front. Instruct models wrap JSON in ``` fences and prose. So
before handing the text to the benchmark's parser we take the first BALANCED
top-level [...] span (string-aware bracket matching), falling back to the raw
text when there is none. The raw completion is kept in output.jsonl as
generated_text_raw, so the extraction is auditable after the fact.

==========================================================================
PIPELINE
==========================================================================
  merge     LoRA adapter + base -> merged HF dir    (surface/verl_rl/merge_lora.py)
  serve     merged dir -> OpenAI endpoint           (vLLM, mcp_vllm env)
  generate  one chat completion per task            -> output.jsonl
  score     benchmark's own scorer, per item        -> per_item.jsonl, summary.json
  records   -> records.jsonl, one row per (task, sample)

Stages are selectable with --stages, and generate resumes: tasks already in
output.jsonl are skipped, so a killed run does not redo the expensive part.
score+records are CPU-only and can be rerun off the GPU.

==========================================================================
PREREQUISITES (one-time, no GPU, no new venv)
==========================================================================
  1. git clone https://github.com/IBM/NESTFUL   ($W/nestful_repo)
     git -C $W/nestful_repo checkout <NESTFUL_COMMIT>
  2. nothing else. $MCP_ENVS/mcp_vllm already carries vllm 0.26.0,
     transformers, scikit-learn 1.9.0, jsonlines, requests and tqdm, which is
     the entire dependency set of the benchmark's scorer. Unlike BFCL this
     needs no side venv.

==========================================================================
LAUNCH, one per checkpoint. NOT on the login node.
==========================================================================
    R=$BRACE_ROOT
    W=$BRACE_WORK/nestful

    srun --jobid=$POD --ntasks=1 --overlap --gres=gpu:1 --cpus-per-task=8 \
         --mem=120G bash -c "
      . $R/env.sh
      export HF_HOME=$R/hf_cache
      $MCP_ENVS/mcp_vllm/bin/python $R/surface/verl_rl/bench_nestful.py \
        --arm q4bT --step 30 \
        --ckpt $R/work/verl/ckpt_q4bT/global_step_30/actor \
        --base $R/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 \
        --merged-out $W/merged_q4bT_step30 --out-dir $W/run_q4bT \
        --nestful-repo $W/nestful_repo"

  The chain script is $W/run_nestful.sh; it verifies by ARTIFACT
  (records.jsonl line count >= 1861), not by exit code.

  CPU dry run, no GPU, no model:
      python bench_nestful.py --arm dry --out-dir /tmp/nf --nestful-repo $W/nestful_repo \
             --oracle --limit 40
  feeds the GOLD call sequences through the REAL scorer and asserts they score
  1.0; --mock additionally stands up a stub completions endpoint and drives
  generate -> score -> records end to end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import resource
import shlex
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Reuse the process/probe/merge machinery rather than reimplementing it. All
# module-level definitions, no import-time side effects beyond
# csv.field_size_limit.
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
NESTFUL_COMMIT = "fc2c4123e73500a56185a5fb354f05d1c8b4890c"
DATA_REL = "data_v2/nestful_data.jsonl"
EXEC_REL = "data_v2/executable_functions"

# The scorer dispatches its output parser on model_name. "Llama-3.1-8B-Instruct"
# selects parse_llama_3_output -- the generic "the completion is a JSON array of
# {name, arguments, label}" parser, which is exactly the contract the prompt we
# use states. Prompt and parser are the benchmark's own matched pair.
NESTFUL_MODEL_NAME = "Llama-3.1-8B-Instruct"

N_TASKS = 1861
# sha256 of "\n".join(sorted sample_ids) + "\n"; checked at run time so a data
# revision fails loudly instead of silently changing the cell.
TASK_INDEX_SHA256 = "f674e2a99c44a8e7dc9b7d9b7bd5ba22059d50c35a786d7628aa111ce8cf9e9c"
# sha256 of system_template + "\x00" + user_template as extracted from
# PROMPTS.json["LLaMa-3.1"].
PROMPT_SHA256 = "4fddc5bee550d836fb0fb1cff44fe20a357b4aa5fa586c4b14742b80c37da2e1"

_LLAMA_SPLIT = re.compile(
    r"^<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\n"
    r"(.*?)"
    r"<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>\n\n"
    r"(.*?)"
    r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>",
    re.S,
)


# ==========================================================================
# Task set + prompt
# ==========================================================================

def data_file(repo: Path) -> Path:
    return repo / DATA_REL


def load_tasks(repo: Path) -> tuple[list[dict], dict[str, int], str]:
    """Rows, {sample_id: task_idx}, and the sha256 that pins the id list.

    task_idx is the position in the lexicographically sorted sample_id list.
    The ids are UUIDs, so any deterministic order will do; sorted() makes the
    index reproducible from the file alone.
    """
    p = data_file(repo)
    if not p.exists():
        raise SystemExit(f"missing benchmark data file {p}")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ids = sorted(r["sample_id"] for r in rows)
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate sample_id in the benchmark data")
    blob = "\n".join(ids) + "\n"
    return rows, {t: i for i, t in enumerate(ids)}, hashlib.sha256(blob.encode()).hexdigest()


def load_prompt_templates(repo: Path) -> tuple[str, str, str]:
    """(system_template, user_template, sha256) out of the benchmark's PROMPTS.json.

    The released entry is one flat string carrying Llama's chat scaffolding.
    We split it at those tokens so the CONTENT is the benchmark's and the
    scaffolding is the served model's own chat template.
    """
    raw = json.loads((repo / "src" / "PROMPTS.json").read_text())["LLaMa-3.1"]
    m = _LLAMA_SPLIT.match(raw)
    if not m:
        raise SystemExit(
            "PROMPTS.json['LLaMa-3.1'] no longer has the expected Llama chat "
            "scaffolding; re-derive the split deliberately.")
    sys_t, usr_t = m.group(1), m.group(2)
    sha = hashlib.sha256((sys_t + "\x00" + usr_t).encode()).hexdigest()
    return sys_t, usr_t, sha


def load_icl_str(repo: Path, icl_count: int) -> str:
    """The benchmark's own get_icl_str over its own icl_examples.json."""
    sys.path.insert(0, str(repo / "src"))
    from instruct_data_prep import get_icl_str  # noqa: E402  (benchmark's code)
    examples = json.loads((repo / "src" / "icl_examples.json").read_text())
    return get_icl_str(examples[:icl_count], NESTFUL_MODEL_NAME)


def render_messages(row: dict, sys_t: str, usr_t: str, icl_str: str,
                    function_str_mode: str) -> list[dict]:
    tools = row["tools"]
    if function_str_mode == "double":
        # byte-for-byte the released Llama branch: json.dumps of a string that
        # is itself json.dumps of the spec list.
        function_str = json.dumps(json.dumps(tools))
    else:
        function_str = json.dumps(tools)
    return [
        {"role": "system",
         "content": sys_t.format(FUNCTION_STR=function_str, ICL_EXAMPLES=icl_str)},
        {"role": "user", "content": usr_t.format(QUERY=row["input"])},
    ]


# ==========================================================================
# Response extraction
# ==========================================================================

def extract_json_array(text: str) -> str:
    """First balanced top-level [...] span, string-aware. Raw text if none.

    Deterministic, arm-independent, and applied to every completion. It exists
    because parse_llama_3_output() assumes the completion IS a JSON array,
    while an instruct model wraps it in ``` fences and a sentence of prose.
    """
    if text is None:
        return ""
    s = text.strip()
    start = s.find("[")
    if start < 0:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s


# ==========================================================================
# Stages: serve + generate
# ==========================================================================

def stage_serve_nestful(a, model_dir: Path, port: int) -> Service:
    """vLLM, OpenAI-compatible, serving the merged dir.

    Same two site facts bench_bfcl.py documents:
      * VLLM_USE_FLASHINFER_SAMPLER=0 -- FlashInfer JIT-compiles its sampling
        kernel at warmup and compute nodes carry no nvcc, so the EngineCore
        dies before /health ever answers.
      * --enforce-eager -- same reason, no torch.compile toolchain.
    Unlike bench_bfcl we hit /v1/chat/completions, so vLLM applies the model's
    own chat template. No --enable-auto-tool-choice / --tool-call-parser: the
    benchmark parses the RAW assistant text itself and would never see a
    wire-level tool_calls array.
    """
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    cmd = [str(VLLM_BIN), "serve", str(model_dir),
           "--served-model-name", a.served_model_name, str(model_dir),
           "--host", "127.0.0.1", "--port", str(port),
           "--max-model-len", str(a.max_model_len),
           "--gpu-memory-utilization", str(a.gpu_memory_utilization),
           "--tensor-parallel-size", str(a.tensor_parallel_size),
           "--enforce-eager"]
    cmd += shlex.split(a.vllm_extra_args)
    return Service("vllm", cmd, Path(a.out_dir) / "logs" / "vllm.log",
                   f"http://127.0.0.1:{port}/health", timeout=a.serve_timeout)


def _one_completion(url: str, model: str, messages: list[dict], a, sample: int):
    import requests
    body = {
        "model": model,
        "messages": messages,
        "temperature": a.temperature if sample == 0 else max(a.temperature, 0.7),
        "max_tokens": a.max_tokens,
        "seed": a.seed + sample,
    }
    last = None
    for attempt in range(a.max_retries):
        try:
            r = requests.post(url, json=body, timeout=a.request_timeout)
            if r.status_code == 200:
                d = r.json()
                ch = d["choices"][0]
                return {
                    "text": ch["message"].get("content") or "",
                    "finish_reason": ch.get("finish_reason"),
                    "input_token": (d.get("usage") or {}).get("prompt_tokens"),
                    "output_token": (d.get("usage") or {}).get("completion_tokens"),
                    "http_error": None,
                }
            last = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:  # noqa: BLE001 -- network, retried
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(2 ** attempt, 20))
    return {"text": "", "finish_reason": None, "input_token": None,
            "output_token": None, "http_error": last}


def stage_generate(a, rows: list[dict], port: int, sample: int,
                   sys_t: str, usr_t: str, icl_str: str) -> Path:
    """One chat completion per task -> output.jsonl in the benchmark's shape.

    Resumable: sample_ids already present are skipped, so a killed run keeps
    everything it had.
    """
    out = Path(a.out_dir) / f"output_s{sample}.jsonl"
    done: set[str] = set()
    if out.exists() and not a.force_generate:
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:  # noqa: BLE001 -- a torn last line
                    pass
        log(f"resume: {len(done)} of {len(rows)} already generated in {out}")
    todo = [r for r in rows if r["sample_id"] not in done]
    if not todo:
        log("generate: nothing to do")
        return out

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    model = a.served_model_name
    t0 = time.time()
    n = 0
    with open(out, "a") as f:
        with ThreadPoolExecutor(max_workers=a.num_threads) as pool:
            futs = {
                pool.submit(_one_completion, url, model,
                            render_messages(r, sys_t, usr_t, icl_str, a.function_str),
                            a, sample): r
                for r in todo
            }
            for fut in as_completed(futs):
                r = futs[fut]
                res = fut.result()
                raw = res["text"]
                rec = {
                    # ---- exactly the five fields the benchmark's scorer reads ----
                    "sample_id": r["sample_id"],
                    "input": r["input"],
                    "output": json.dumps(r["output"]),
                    "gold_answer": json.dumps(r["gold_answer"]),
                    "tools": json.dumps(r["tools"]),
                    "generated_text": extract_json_array(raw),
                    # ---- ours, for audit ----
                    "generated_text_raw": raw,
                    "finish_reason": res["finish_reason"],
                    "input_token": res["input_token"],
                    "output_token": res["output_token"],
                    "http_error": res["http_error"],
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
                if n % 200 == 0:
                    f.flush()
                    log(f"  generated {n}/{len(todo)} ({time.time()-t0:.0f}s)")
    log(f"generate done: {n} completions -> {out} ({time.time()-t0:.0f}s)")
    return out


# ==========================================================================
# Stage: score  (the benchmark's own scorer)
# ==========================================================================

def _import_scorer(repo: Path):
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import scorer  # noqa: E402  (benchmark's code)
    return scorer


def _win_child(conn, repo_str: str, item: dict, exec_dir: str,
               cpu_s: int, mem_bytes: int) -> None:
    """Forked child: run ONE win_rate evaluation under hard resource limits.

    See the sandbox note in the module docstring. Anything that escapes
    signal.alarm(10) inside the benchmark's calculate_ans -- a C-level spin, a
    runaway allocation, a blocking read on stdin -- is stopped here.
    """
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scorer = _import_scorer(Path(repo_str))
            res = scorer.calculate_scores([item], NESTFUL_MODEL_NAME, exec_dir,
                                          win_rate_flag=True)
        conn.send(float(res["win_rate"]))
    except BaseException as e:  # noqa: BLE001 -- report, never hang
        try:
            conn.send(f"ERR {type(e).__name__}: {e}")
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def win_rate_one(ctx, repo: Path, item: dict, exec_dir: str, a) -> tuple[int, str | None]:
    """(0/1, note). Never raises, never hangs."""
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_win_child,
                    args=(child, str(repo), item, exec_dir, a.win_cpu_s,
                          a.win_mem_gb * (1 << 30)))
    p.start()
    child.close()
    got = None
    if parent.poll(a.win_wall_s):
        try:
            got = parent.recv()
        except EOFError:
            got = None
    p.join(timeout=5)
    if p.is_alive():
        p.kill()
        p.join()
    parent.close()
    if isinstance(got, float):
        return (1 if got >= 1.0 else 0), None
    if isinstance(got, str):
        return 0, got[:200]
    return 0, "win_timeout"


def stage_score(a, repo: Path, sample: int) -> tuple[Path, dict]:
    """Per-item scores + the official set-level aggregate, both from scorer.py.

    Two passes on purpose:
      * per item, win_rate_flag=False -> full match / partial match / parse
        errors. Pure comparison, no execution, cannot hang.
      * per item, sandboxed child     -> win_rate. Execution, hard-limited.
      * once over the whole set       -> the benchmark's own F1s, which are
        set-level statistics and are meaningless per item.
    """
    out = Path(a.out_dir)
    src = out / f"output_s{sample}.jsonl"
    if not src.exists():
        raise SystemExit(f"no generations to score at {src}")
    items = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    exec_dir = str(repo / EXEC_REL)
    scorer = _import_scorer(repo)
    ctx = mp.get_context("fork")

    per_path = out / f"per_item_s{sample}.jsonl"
    n_win = n_full = n_parse_err = n_timeout = 0
    t0 = time.time()
    with open(per_path, "w") as f:
        for i, item in enumerate(items):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = scorer.calculate_scores([item], NESTFUL_MODEL_NAME, exec_dir,
                                                win_rate_flag=False)
                except Exception as e:  # noqa: BLE001 -- a hostile completion
                    r = {"accuracy_combined": "0.000",
                         "percentage_times_full_score": "0.000",
                         "num_pred_examples_w_parsing_errors": 1,
                         "_error": f"{type(e).__name__}: {e}"}
            win, note = (0, "skipped") if a.no_win_rate else \
                win_rate_one(ctx, repo, item, exec_dir, a)
            full = int(float(r["percentage_times_full_score"]) >= 1.0)
            parse_err = int(r.get("num_pred_examples_w_parsing_errors", 0) > 0)
            n_win += win
            n_full += full
            n_parse_err += parse_err
            n_timeout += int(note == "win_timeout")
            f.write(json.dumps({
                "sample_id": item["sample_id"],
                "win_rate": win,
                "full_match": full,
                "partial_acc": float(r["accuracy_combined"]),
                "parse_error": parse_err,
                "win_note": note,
                "score_error": r.get("_error"),
                "finish_reason": item.get("finish_reason"),
                "http_error": item.get("http_error"),
                "input_token": item.get("input_token"),
                "output_token": item.get("output_token"),
            }) + "\n")
            if (i + 1) % 200 == 0:
                log(f"  scored {i+1}/{len(items)} ({time.time()-t0:.0f}s) "
                    f"win={n_win} full={n_full}")
    log(f"per-item scoring done ({time.time()-t0:.0f}s): win={n_win} full={n_full} "
        f"parse_err={n_parse_err} win_timeout={n_timeout}")

    log("aggregate pass (benchmark's own set-level metrics, no execution)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg = scorer.calculate_scores(items, NESTFUL_MODEL_NAME, exec_dir,
                                      win_rate_flag=False)
    agg = {k: v for k, v in agg.items()}
    agg["win_rate_sandboxed"] = round(n_win / len(items), 4) if items else None
    agg["full_match_recomputed"] = round(n_full / len(items), 4) if items else None
    agg["n_scored"] = len(items)
    agg["n_win_timeout"] = n_timeout
    (out / f"summary_s{sample}.json").write_text(json.dumps(agg, indent=2))
    log(f"  f1_intent={agg['f1_intent']} f1_slot={agg['f1_slot']} "
        f"partial={agg['accuracy_combined']} full={agg['percentage_times_full_score']} "
        f"win={agg['win_rate_sandboxed']}")
    return per_path, agg


# ==========================================================================
# Stage: records
# ==========================================================================

def stage_records(a, task_index: dict[str, int], index_sha: str,
                  prompt_sha: str, records: Path, manifest: dict) -> None:
    """One JSON object per (task, sample), keyed the way main_table expects.

        k = (d["scenario"], d["task_idx"], d["seed"]);  out[k] = int(d["reward"] > 0)

    reward = win_rate, the benchmark's executable end-to-end metric. The other
    scores ride along so a secondary readout needs no rerun.
    """
    out = Path(a.out_dir)
    n = 0
    with open(records, "w") as f:
        for sample in range(a.n_samples):
            per = out / f"per_item_s{sample}.jsonl"
            if not per.exists():
                log(f"WARNING: sample {sample} has no per-item scores; skipping")
                continue
            rows = [json.loads(l) for l in per.read_text().splitlines() if l.strip()]
            for d in sorted(rows, key=lambda x: task_index.get(x["sample_id"], 1 << 30)):
                tid = d["sample_id"]
                if tid not in task_index:
                    log(f"WARNING: {tid} not in the pinned task index; skipping")
                    continue
                f.write(json.dumps({
                    # ---- fields the paired tooling keys on, unchanged ----
                    "scenario": "nestful",
                    "task_idx": task_index[tid],
                    "seed": a.seed + sample,
                    "interface": "RAW",
                    "reward": d["win_rate"],
                    "verifier_status": "success" if d["http_error"] is None else "unscored",
                    # ---- benchmark-native ----
                    "bench": "nestful_v2",
                    "subset": "all",
                    "arm": a.arm,
                    "step": a.step,
                    "task_id": tid,
                    "sample": sample,
                    "win_rate": d["win_rate"],
                    "full_match": d["full_match"],
                    "partial_acc": d["partial_acc"],
                    "parse_error": d["parse_error"],
                    "win_note": d["win_note"],
                    "finish_reason": d["finish_reason"],
                    "input_token": d["input_token"],
                    "output_token": d["output_token"],
                    "nestful_commit": NESTFUL_COMMIT,
                    "nestful_parser": NESTFUL_MODEL_NAME,
                    "function_str": a.function_str,
                    "icl_count": a.icl_count,
                    "task_index_sha256": index_sha,
                    "prompt_sha256": prompt_sha,
                    "run_id": manifest["run_id"],
                }) + "\n")
                n += 1
    expected = len(task_index) * a.n_samples
    log(f"wrote {n} records -> {records}")
    if n != expected:
        log(f"WARNING: {n} records but {expected} expected "
            f"({len(task_index)} tasks x {a.n_samples} samples). Tasks missing "
            "from output.jsonl were never generated -- check the generate log.")
    rows = [json.loads(l) for l in records.read_text().splitlines()]
    if rows:
        for key in ("win_rate", "full_match", "parse_error"):
            k = sum(r[key] for r in rows)
            log(f"  {key:12s} {k:5d}/{len(rows)} = {100.0*k/len(rows):5.2f}")
        pa = sum(r["partial_acc"] for r in rows) / len(rows)
        log(f"  {'partial_acc':12s} mean = {pa:.4f}")


# ==========================================================================
# CPU-only checks: oracle + mock
# ==========================================================================

def stage_oracle(a, repo: Path, rows: list[dict], limit: int) -> None:
    """Feed the GOLD call sequences through the REAL scorer; assert they pass.

    The half a stub model cannot cover. If gold does not score at or near 1.0
    the wiring is wrong, not the policy. Also scores a hand-written WRONG
    completion and a hand-written UNPARSEABLE one, which must come back 0, so
    the check is two-sided and cannot pass vacuously.
    """
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel = rows[:limit] if limit else rows
    fake = out / "output_oracle.jsonl"
    with open(fake, "w") as f:
        for r in sel:
            f.write(json.dumps({
                "sample_id": r["sample_id"],
                "input": r["input"],
                "output": json.dumps(r["output"]),
                "gold_answer": json.dumps(r["gold_answer"]),
                "tools": json.dumps(r["tools"]),
                # the model "emits" exactly the gold call list, as JSON text --
                # routed through the same extraction every real completion gets
                "generated_text": extract_json_array(
                    "Sure! Here you go:\n```json\n" + json.dumps(r["output"]) + "\n```"),
                "generated_text_raw": "(oracle)",
                "finish_reason": "stop", "input_token": None,
                "output_token": None, "http_error": None,
            }) + "\n")

    exec_dir = str(repo / EXEC_REL)
    scorer = _import_scorer(repo)
    ctx = mp.get_context("fork")
    items = [json.loads(l) for l in fake.read_text().splitlines()]
    n_win = n_full = 0
    bad_win: list[str] = []
    for item in items:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = scorer.calculate_scores([item], NESTFUL_MODEL_NAME, exec_dir,
                                        win_rate_flag=False)
        full = int(float(r["percentage_times_full_score"]) >= 1.0)
        win, note = win_rate_one(ctx, repo, item, exec_dir, a)
        n_full += full
        n_win += win
        if not win:
            bad_win.append(f"{item['sample_id']} ({note})")
    log(f"ORACLE full_match {n_full}/{len(items)}")
    log(f"ORACLE win_rate   {n_win}/{len(items)}")
    if bad_win[:10]:
        log("  gold that does NOT execute to gold_answer: " + ", ".join(bad_win[:10]))

    # negative controls, hand written, must both be 0
    neg = []
    proto = items[0]
    for label, text in (
        ("wrong-call", json.dumps([{"name": "add", "label": "$var_1",
                                    "arguments": {"arg_0": 1, "arg_1": 1}}])),
        ("prose", "I am not sure which function to call here."),
        ("empty", ""),
    ):
        item = dict(proto)
        item["generated_text"] = extract_json_array(text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = scorer.calculate_scores([item], NESTFUL_MODEL_NAME, exec_dir,
                                        win_rate_flag=False)
        full = int(float(r["percentage_times_full_score"]) >= 1.0)
        win, _ = win_rate_one(ctx, repo, item, exec_dir, a)
        neg.append((label, full, win))
        log(f"NEGATIVE {label:11s} full_match={full} win_rate={win}")

    ok = (n_full == len(items)) and all(f == 0 and w == 0 for _, f, w in neg)
    log(f"ORACLE CHECK {'PASSED' if ok else 'FAILED'} "
        f"(gold full_match {n_full}/{len(items)}, gold win_rate {n_win}/{len(items)})")
    if not ok:
        raise SystemExit("oracle/negative check failed; the harness wiring is wrong")


MOCK_SERVER_SRC = r'''
"""Stub /v1/chat/completions. Returns a fenced JSON array of two calls, so the
whole decode -> extract -> benchmark parser -> benchmark scorer -> records path
runs with no GPU and no model. Scores ~0 by construction; the ORACLE check is
what proves the success path."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

TEXT = ('Here is the call sequence:\n```json\n'
        '[{"name": "add", "label": "$var_1", "arguments": {"arg_0": 1, "arg_1": 2}}, '
        '{"name": "multiply", "label": "$var_2", '
        '"arguments": {"arg_0": "$var_1.result$", "arg_1": 3}}]\n```\nDone.')


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
            "id": "chatcmpl-mock", "object": "chat.completion", "created": 0,
            "model": req.get("model", "mock"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": TEXT}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


def start_mock(a, port: int) -> Service:
    stub = Path(a.out_dir) / "logs" / "mock_server.py"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(MOCK_SERVER_SRC)
    return Service("mock-vllm", [sys.executable, str(stub), str(port)],
                   Path(a.out_dir) / "logs" / "mock_vllm.log",
                   f"http://127.0.0.1:{port}/health", timeout=60)


# ==========================================================================
# main
# ==========================================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="NESTFUL transfer eval for a trained checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("what to evaluate")
    g.add_argument("--arm", required=True, help="label, e.g. q4bT")
    g.add_argument("--ckpt", help="verl global_step_N/actor dir, or a full HF dir")
    g.add_argument("--merged-dir", help="skip merging; serve this dir as-is")
    g.add_argument("--step", type=int, default=None, help="training step, recorded only")
    g.add_argument("--out-dir", required=True, help="run directory (put it on /scratch)")

    g = p.add_argument_group("benchmark")
    g.add_argument("--nestful-repo", required=True, help="path to the IBM/NESTFUL clone")
    g.add_argument("--icl-count", type=int, default=3,
                   help="in-context examples, the benchmark's own (default 3, as run.sh)")
    g.add_argument("--function-str", default="json", choices=("json", "double"),
                   help="json = spec embedded as JSON (default); double = reproduce "
                        "the released Llama branch's double-encoded string")
    g.add_argument("--limit", type=int, default=0, help="first N tasks (debug/oracle only)")
    g.add_argument("--print-subset", action="store_true",
                   help="print the pinned task-id list and its sha256, then exit")

    g = p.add_argument_group("generation")
    g.add_argument("--n-samples", type=int, default=1, help="samples per task; 1 = pass@1")
    g.add_argument("--temperature", type=float, default=0.001)
    g.add_argument("--max-tokens", type=int, default=3072,
                   help="the longest gold chain (53 calls) tokenises to 2027 tokens "
                        "under the Qwen3-VL tokenizer; 3072 clears it with headroom")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--num-threads", type=int, default=16)
    g.add_argument("--request-timeout", type=int, default=600)
    g.add_argument("--max-retries", type=int, default=4)
    g.add_argument("--force-generate", action="store_true",
                   help="ignore an existing output.jsonl instead of resuming")

    g = p.add_argument_group("scoring")
    g.add_argument("--no-win-rate", action="store_true",
                   help="skip the executable metric (parse/match metrics only)")
    g.add_argument("--win-cpu-s", type=int, default=15, help="RLIMIT_CPU in the child")
    g.add_argument("--win-mem-gb", type=int, default=4, help="RLIMIT_AS in the child")
    g.add_argument("--win-wall-s", type=int, default=30, help="parent-side wall clock")

    g = p.add_argument_group("serving")
    g.add_argument("--served-model-name", default="policy")
    g.add_argument("--max-model-len", type=int, default=16384,
                   help="measured prompt lengths: median 1560, max 2719 tokens "
                        "(Qwen3-VL tokenizer, --function-str json, 3 ICL examples); "
                        "16384 leaves room for --max-tokens on top of the worst prompt")
    g.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    g.add_argument("--tensor-parallel-size", type=int, default=1)
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
    g.add_argument("--stages", default="merge,serve,generate,score,records")
    g.add_argument("--mock", action="store_true",
                   help="no GPU: stub chat endpoint + real extraction, real scorer, "
                        "real records, plus the oracle check")
    g.add_argument("--oracle", action="store_true",
                   help="no GPU, no server: gold-as-prediction through the real "
                        "scorer, plus hand-written negative controls")
    g.add_argument("--oracle-limit", type=int, default=25,
                   help="gold entries for the oracle check; 0 = all 1861, which is "
                        "how the metric's achievable ceiling was measured")

    a = p.parse_args()
    repo = Path(a.nestful_repo).resolve()

    rows, task_index, index_sha = load_tasks(repo)
    if a.print_subset:
        for tid, i in sorted(task_index.items(), key=lambda kv: kv[1]):
            print(f"{i}\t{tid}")
        log(f"{len(task_index)} tasks, sha256 {index_sha}")
        return 0
    if index_sha != TASK_INDEX_SHA256 or len(task_index) != N_TASKS:
        raise SystemExit(
            f"task index no longer reproduces:\n  expected n={N_TASKS} "
            f"sha256={TASK_INDEX_SHA256}\n  got      n={len(task_index)} sha256={index_sha}\n"
            "The benchmark data changed. Re-pin deliberately -- do not silently "
            "compare across revisions.")

    sys_t, usr_t, prompt_sha = load_prompt_templates(repo)
    if prompt_sha != PROMPT_SHA256:
        raise SystemExit(
            f"prompt template changed:\n  expected {PROMPT_SHA256}\n  got      {prompt_sha}")
    icl_str = load_icl_str(repo, a.icl_count)
    log(f"{len(task_index)} tasks, index sha256 {index_sha}")
    log(f"prompt sha256 {prompt_sha}, icl {a.icl_count} examples ({len(icl_str)} chars), "
        f"function_str={a.function_str}")

    if a.limit:
        keep = sorted(task_index, key=lambda t: task_index[t])[:a.limit]
        keep_set = set(keep)
        rows = [r for r in rows if r["sample_id"] in keep_set]
        task_index = {t: i for i, t in enumerate(keep)}
        log(f"--limit {a.limit}: {len(rows)} tasks (DEBUG -- not a publishable cell)")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    if a.mock:
        stages = [s for s in stages if s != "merge"]
    if a.oracle and not a.mock:
        stages = []

    manifest = {
        "run_id": f"{a.arm}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "arm": a.arm, "step": a.step, "n_tasks": len(task_index),
        "n_samples": a.n_samples, "task_index_sha256": index_sha,
        "prompt_sha256": prompt_sha, "nestful_commit": NESTFUL_COMMIT,
        "nestful_parser": NESTFUL_MODEL_NAME, "function_str": a.function_str,
        "icl_count": a.icl_count, "argv": sys.argv, "mock": a.mock,
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
            svc = start_mock(a, port) if a.mock else stage_serve_nestful(a, model_dir, port)
            services.append(svc.start())
            svc.wait_ready()

        if "generate" in stages:
            for s in range(a.n_samples):
                stage_generate(a, rows, port, s, sys_t, usr_t, icl_str)
        if "score" in stages:
            for s in range(a.n_samples):
                stage_score(a, repo, s)
        if "records" in stages:
            stage_records(a, task_index, index_sha, prompt_sha,
                          out / "records.jsonl", manifest)
        if a.oracle or a.mock:
            stage_oracle(a, repo, rows, a.limit or a.oracle_limit)
    finally:
        for s in reversed(services):
            s.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
