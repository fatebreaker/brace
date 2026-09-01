#!/usr/bin/env python3
"""
bench_transfer.py -- transfer evaluation of already-trained checkpoints on a
PUBLIC benchmark (MCP-Atlas), for PLAN_TRIAGE.md section 5b item 4.

EVAL ONLY. This script never trains. It takes one checkpoint, serves it, runs
the benchmark's OWN harness against it, scores with the benchmark's OWN judge,
and emits one JSON record per task so the result can be entered into the same
paired-analysis machinery we use for the in-domain cells.

--------------------------------------------------------------------------
WHAT THE BENCHMARK IS (all claims below verified against primary sources)
--------------------------------------------------------------------------
MCP-Atlas, Bandi et al. 2026, arXiv:2602.00933.
  code      https://github.com/scaleapi/mcp-atlas          MIT License
  data      https://huggingface.co/datasets/ScaleAI/MCP-Atlas   CC-BY-4.0
  sandbox   ghcr.io/scaleapi/mcp-atlas:1.2.7  (36 MCP servers, 1.02 GB amd64)

Dataset: ONE parquet, 500 rows, 5 string columns, verbatim from the HF card:
    TASK          (str) A unique 24 character ID.
    ENABLED_TOOLS (str) A controlled subset of 10-25 tools exposed to the agent per task.
    PROMPT        (str) A single-turn, natural-language request requiring multiple tool calls.
    GTFA_CLAIMS   (str) A set of distinct, independently verifiable claims ...
    TRAJECTORY    (str) The sequence of tool calls ... resolving the task.
  sha256 of MCP-Atlas.parquet = 2d7bc052f14cbcb3b8294293481053f7111d256f9c9deaa96f3ff632d19958d0
  (equals the HF LFS oid, so the file is content-verified.)

THE COMPARABLE PROTOCOL. OpenForgeRL (arXiv:2607.21557, Appendix C.2) scored
28.1 on this benchmark. Their protocol, quoted verbatim:

    "we evaluate MCPAtlas under the benchmark's default 20-server
     configuration, which excludes optional servers that require third-party
     credentials or service-specific data initialization. This configuration
     fixes the evaluation set without any manual selection on our part: of the
     500 public tasks, exactly 89 have ground-truth expected tool calls that
     are fully supported by these default servers."
    "we use the MCPAtlas official harness which is based on a ReACT-like loop"
    "A task counts as successful when its claim-coverage score is at least
     0.75, and we report the fraction of successful tasks as pass@1."

`default20_89` below re-derives that 89-task set from the raw parquet plus the
repo's mcp_server_template.json. It reproduces exactly 89 -- see
`--print-subset`, which also prints the id-list sha256
(2837aca8319d416b7472302b6abbd1f487fb15b628ec5cb909e93c3705bb0ed6) so a rerun
on a newer dataset revision fails loudly instead of silently changing the set.

--------------------------------------------------------------------------
PIPELINE
--------------------------------------------------------------------------
  merge    LoRA adapter + base -> a merged HF dir           (surface/verl_rl/merge_lora.py)
  serve    merged dir -> OpenAI-compatible endpoint         (vLLM, mcp_vllm env)
  sandbox  ghcr image -> HTTP MCP sandbox on :1984          (apptainer where docker is absent)
  harness  MCP-Atlas TypeScript ReACT loop on :3001         (node >= 20)
  run      POST each task to the harness -> outputs.csv     (same 3 columns run_eval.py writes)
  score    outputs.csv + ground truth -> scored_<arm>.csv   (repo's score_claims.py, LLM judge)
  records  scored_<arm>.csv -> records.jsonl                (one row per task, for paired tests)

Stages are selectable with --stages so a crashed run resumes without redoing
generation, which is the expensive part.

--mock replaces serve/sandbox/harness with in-process stubs and replaces the
judge with a deterministic stub. It exercises the real run/score/records code
paths, including the benchmark's real score_claims.py, on a CPU with no GPU,
no docker and no node. That is the dry run.

--------------------------------------------------------------------------
WHAT records.jsonl IS FOR
--------------------------------------------------------------------------
One row per (task, sample), carrying BOTH the five fields main_table.py /
rl_learning_curve.py / plot_curve.py already key on -- scenario, task_idx,
seed, interface, reward -- and the benchmark-native continuous score. So an
MCP-Atlas cell drops into the existing paired machinery with no code change:

    main_table.load_cell(records.jsonl) -> {("mcpatlas", task_idx, seed): 0/1}

Read the caveat in stage_records() before choosing the headline statistic.

--------------------------------------------------------------------------
PREREQUISITES (one-time, none of them need a GPU)
--------------------------------------------------------------------------
  1. git clone https://github.com/scaleapi/mcp-atlas   ($ATLAS)
  2. apptainer pull --arch amd64 mcp-atlas-1.2.7.sif \
         docker://ghcr.io/scaleapi/mcp-atlas:1.2.7          (~1.0 GB pull)
  3. cp $ATLAS/env.template $WORK/.env    -- for the 89-task default-20
     configuration every API-key line stays EMPTY; leave ENABLED_SERVERS
     empty too, so the image's own default-server list is what comes up.
  4. module load nodejs/20.13.1 && (cd $ATLAS/services/agent-harness && npm install)
     The system node is 17.8.0, too old for the harness's deps.
  5. a scorer interpreter with tenacity + nest_asyncio + matplotlib on top of
     mcp_verl (mcp_verl itself lacks all three):
        $MCP_ENVS/mcp_verl/bin/python -m venv --system-site-packages $WORK/scorer_venv
        $WORK/scorer_venv/bin/pip install tenacity nest-asyncio matplotlib
  6. a judge endpoint. score_claims.py posts OpenAI Chat Completions with
     response_format=json_schema, so any compatible provider works; pass
     --judge-base-url/--judge-api-key/--judge-model. OpenForgeRL used
     Gemini 2.5 Pro. ~309 claim calls per arm on the 89-task subset.

--------------------------------------------------------------------------
LAUNCH, one per checkpoint. NOT on the login node.
--------------------------------------------------------------------------
    R=$BRACE_ROOT
    W=$BRACE_WORK/mcpatlas

    srun --jobid=$POD --ntasks=1 --overlap --gres=gpu:1 --cpus-per-task=10 \
         --mem=120G bash -c "
      module load nodejs/20.13.1
      . $R/env.sh
      $W/scorer_venv/bin/python $R/surface/verl_rl/bench_transfer.py \
        --arm a8T5k --step 25 \
        --ckpt $R/work/verl/ckpt_a8T5k/global_step_25/actor \
        --merged-out $W/merged_a8T5k_step25 \
        --out-dir $W/run_a8T5k_step25 \
        --atlas-repo $W/mcp-atlas --sif $W/mcp-atlas-1.2.7.sif \
        --overlay $W/overlay_a8T5k.img --env-file $W/.env \
        --scorer-python $W/scorer_venv/bin/python \
        --judge-base-url <JUDGE_URL> --judge-api-key <KEY> \
        --judge-model gemini/gemini-2.5-pro"

  and the same with:
    --arm a8T3g2 --step 30 --ckpt $R/work/verl/ckpt_a8T3g2/global_step_30/actor
    --arm a8F    --step 20 --ckpt $R/work/verl/ckpt_a8F/global_step_20/actor
    --arm base   --step 0  --ckpt $R/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

  The base-policy arm skips merging automatically (it is already a full HF
  dir with no lora_train_meta.json). Give each arm its own --overlay: the
  sandbox writes into it and two arms sharing one would race.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)

# --------------------------------------------------------------------------
# Site configuration. Everything heavy lives on /scratch: /project is at 95%
# and has been seen to fill and kill running jobs.
# --------------------------------------------------------------------------
MCP_ROOT = Path(os.environ.get("BRACE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ENVS = Path(os.environ.get("BRACE_ENVS", str(MCP_ROOT / "envs")))
VLLM_PY = ENVS / "mcp_vllm" / "bin" / "python"
VLLM_BIN = ENVS / "mcp_vllm" / "bin" / "vllm"
VERL_PY = ENVS / "mcp_verl" / "bin" / "python"

DATASET_REPO = "ScaleAI/MCP-Atlas"
PARQUET_URL = "https://huggingface.co/datasets/ScaleAI/MCP-Atlas/resolve/main/MCP-Atlas.parquet"
PARQUET_SHA256 = "2d7bc052f14cbcb3b8294293481053f7111d256f9c9deaa96f3ff632d19958d0"
SUBSET89_SHA256 = "2837aca8319d416b7472302b6abbd1f487fb15b628ec5cb909e93c3705bb0ed6"
SANDBOX_IMAGE = "ghcr.io/scaleapi/mcp-atlas:1.2.7"

# env.template, verbatim: "Default servers (no API keys): ..."
DEFAULT_20_SERVERS = [
    "arxiv", "calculator", "cli-mcp-server", "clinicaltrialsgov-mcp-server",
    "context7", "ddg-search", "desktop-commander", "fetch", "filesystem",
    "git", "mcp-code-executor", "mcp-server-code-runner", "memory",
    "met-museum", "open-library", "osm-mcp-server", "pubmed", "weather",
    "whois", "wikipedia",
]

# score_claims.py: coverage_to_score maps fully_fulfilled->1, partially->0.5.
# OpenForgeRL's success threshold; 0.50 is the benchmark's other reported one.
PASS_THRESHOLDS = (0.50, 0.75)


def log(msg: str) -> None:
    # stderr, so `--print-subset` can be piped straight into sha256sum / a file
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True, file=sys.stderr)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ==========================================================================
# Task set
# ==========================================================================

def fetch_parquet(path: Path) -> Path:
    """Download the dataset parquet if absent, and content-verify it."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log(f"downloading {PARQUET_URL} -> {path}")
        subprocess.run(["curl", "-sSL", PARQUET_URL, "-o", str(path)], check=True)
    got = sha256_file(path)
    if got != PARQUET_SHA256:
        raise SystemExit(
            f"parquet sha256 mismatch\n  expected {PARQUET_SHA256}\n  got      {got}\n"
            "The dataset revision changed. Re-derive the subset and update "
            "PARQUET_SHA256/SUBSET89_SHA256 deliberately -- do not silently "
            "compare across revisions."
        )
    return path


def server_of(tool_name: str, servers: list[str]) -> str | None:
    """Tool names are '<server>_<tool>'. Longest prefix wins ('git' vs 'github')."""
    for s in servers:
        if tool_name.startswith(s + "_"):
            return s
    return None


def gt_tool_calls(trajectory_json: str) -> list[str]:
    out: list[str] = []
    for msg in json.loads(trajectory_json):
        for tc in (msg.get("tool_calls") or []):
            name = (tc.get("function") or {}).get("name")
            if name:
                out.append(name)
    return out


def load_tasks(parquet: Path, template: Path, subset: str, subset_file: Path | None):
    import pandas as pd

    df = pd.read_parquet(parquet)
    if subset == "all500":
        sel = df
    elif subset == "file":
        ids = {ln.strip() for ln in subset_file.read_text().split() if ln.strip()}
        sel = df[df.TASK.isin(ids)]
    elif subset == "default20_89":
        all_servers = sorted(
            json.loads(template.read_text())["mcpServers"].keys(), key=len, reverse=True
        )
        default = set(DEFAULT_20_SERVERS)
        missing = default - set(all_servers)
        if missing:
            raise SystemExit(f"server template no longer defines: {sorted(missing)}")

        def keep(row) -> bool:
            names = gt_tool_calls(row["TRAJECTORY"])
            if not names:
                return False
            used = {server_of(n, all_servers) for n in names}
            return None not in used and used <= default

        sel = df[df.apply(keep, axis=1)]
        ids_blob = "\n".join(sorted(sel.TASK.tolist())) + "\n"
        got = hashlib.sha256(ids_blob.encode()).hexdigest()
        if len(sel) != 89 or got != SUBSET89_SHA256:
            raise SystemExit(
                f"default20_89 no longer reproduces: n={len(sel)} sha256={got}\n"
                f"expected n=89 sha256={SUBSET89_SHA256}"
            )
    else:
        raise SystemExit(f"unknown subset {subset}")
    return sel.reset_index(drop=True)


def enabled_tool_names(value: Any) -> list[str]:
    """Same normalisation run_eval.py does: entries may be names or {'name':...}."""
    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and value.strip():
        try:
            items = json.loads(value)
            if not isinstance(items, list):
                items = [t.strip() for t in value.split(",") if t.strip()]
        except json.JSONDecodeError:
            items = [t.strip() for t in value.split(",") if t.strip()]
    else:
        return []
    out = []
    for t in items:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, dict) and t.get("name"):
            out.append(t["name"])
    return out


# ==========================================================================
# Services
# ==========================================================================

class Service:
    """A background process plus the HTTP probe that says it is ready."""

    def __init__(self, name: str, cmd: list[str], logfile: Path,
                 probe_url: str, env: dict | None = None, cwd: Path | None = None,
                 timeout: int = 1800):
        self.name, self.cmd, self.logfile = name, cmd, logfile
        self.probe_url, self.env, self.cwd, self.timeout = probe_url, env, cwd, timeout
        self.proc: subprocess.Popen | None = None

    def start(self) -> "Service":
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        log(f"start {self.name}: {' '.join(shlex.quote(c) for c in self.cmd)}")
        log(f"      log -> {self.logfile}")
        self.proc = subprocess.Popen(
            self.cmd, stdout=open(self.logfile, "w"), stderr=subprocess.STDOUT,
            env={**os.environ, **(self.env or {})},
            cwd=str(self.cwd) if self.cwd else None, start_new_session=True,
        )
        return self

    def wait_ready(self) -> None:
        import urllib.error
        import urllib.request
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            if self.proc and self.proc.poll() is not None:
                raise SystemExit(
                    f"{self.name} exited with code {self.proc.returncode} before becoming "
                    f"ready. Tail of {self.logfile}:\n" + tail(self.logfile)
                )
            try:
                with urllib.request.urlopen(self.probe_url, timeout=5) as r:
                    if r.status == 200:
                        log(f"{self.name} ready after {time.time()-t0:.0f}s")
                        return
            except Exception:
                pass
            time.sleep(3)
        raise SystemExit(f"{self.name} not ready after {self.timeout}s; see {self.logfile}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            log(f"stop {self.name}")
            try:
                os.killpg(os.getpgid(self.proc.pid), 15)
            except Exception:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), 9)
                except Exception:
                    pass


def tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except Exception:
        return "(no log)"


def resolve_base(a) -> Path:
    if a.base:
        return Path(a.base)
    snaps = sorted((MCP_ROOT / "hf_cache" / "hub" /
                    f"models--{a.base_repo.replace('/', '--')}" / "snapshots").glob("*"))
    if not snaps:
        raise SystemExit(f"no local snapshot for {a.base_repo}; pass --base")
    return snaps[0]


def stage_merge(a) -> Path:
    """
    Materialise a servable HF directory.

    The four things under evaluation are not homogeneous:
      * the three trained arms are verl FSDP shards (global_step_N/actor holding
        model_world_size_1_rank_0.pt with base + PEFT-keyed LoRA tensors side by
        side, plus lora_train_meta.json {"r":64,"lora_alpha":64}). These need
        merge_lora.py, whose interface is exactly
            --ckpt <global_step_N/actor>  --base <base HF snapshot>  --out <dir>
        (all three required, no other flags; rank/alpha come from
        lora_train_meta.json). It runs on CPU -- torch.load(map_location="cpu").
      * the base policy is already a full HF snapshot and is served in place.

    Precedent for the merge command is slurm/coadapt_eval.sh's STEP* branch,
    including its atomic build-then-rename, which is reproduced here so a
    killed merge never leaves a half-written dir that vLLM would happily load.
    """
    if a.merged_dir:
        return Path(a.merged_dir)
    ckpt = Path(a.ckpt)
    if (ckpt / "config.json").exists() and not (ckpt / "lora_train_meta.json").exists():
        log(f"{ckpt} is already a full HF dir (the base-policy arm); serving in place")
        return ckpt

    out = Path(a.merged_out) if a.merged_out else Path(a.out_dir) / "merged"
    if (out / "config.json").exists() and not a.force_merge:
        log(f"merged dir already present, reusing {out}")
        return out
    if not (ckpt / "lora_train_meta.json").exists():
        raise SystemExit(f"{ckpt} has neither config.json nor lora_train_meta.json")

    base = resolve_base(a)
    tmp = out.with_suffix(f".tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(a.merge_python), str(a.merge_script),
           "--ckpt", str(ckpt), "--base", str(base), "--out", str(tmp),
           *shlex.split(a.merge_args)]
    log("merge: " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    log(f"merged -> {out}")
    return out


def stage_serve(a, model_dir: Path, port: int) -> Service:
    """
    NOTE, because this is new here: nothing else in this repo runs an HTTP
    vLLM server. surface/gate_caller/policy_vllm.py builds an in-process
    vllm.LLM(...) and surface/gate_surface/run_gate.py parses <tool_call>
    XML with its own regex. MCP-Atlas's harness is a separate process that
    speaks OpenAI Chat Completions, so it needs a served endpoint and it needs
    vLLM -- not us -- to turn the model's Hermes-style <tool_call> XML into
    `tool_calls` on the wire. Hence --enable-auto-tool-choice plus a parser.

    The policy is Qwen/Qwen3-VL-8B-Instruct (Qwen3VLForConditionalGeneration,
    max_position_embeddings 262144), and the merged dir carries the arm's own
    chat_template.jinja, copied out of global_step_N/actor/huggingface by
    merge_lora.py -- so do NOT pass --chat-template unless deliberately
    overriding the trained template.
    """
    # No CUDA toolkit on compute nodes; FlashInfer's JIT'd sampler would kill
    # the EngineCore at warmup. Use the native sampler.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    cmd = [str(VLLM_BIN), "serve", str(model_dir),
           "--served-model-name", a.served_model_name,
           "--host", "127.0.0.1", "--port", str(port),
           "--max-model-len", str(a.max_model_len),
           "--gpu-memory-utilization", str(a.gpu_memory_utilization),
           "--tensor-parallel-size", str(a.tensor_parallel_size),
           "--enable-auto-tool-choice",
           "--tool-call-parser", a.tool_call_parser,
           "--api-key", a.api_key,
           "--enforce-eager"]
    if a.chat_template:
        cmd += ["--chat-template", a.chat_template]
    if a.reasoning_parser:
        cmd += ["--reasoning-parser", a.reasoning_parser]
    cmd += shlex.split(a.vllm_extra_args)
    # Probe /health, NOT /v1/models: with --api-key set, vLLM installs
    # AuthenticationMiddleware, whose docstring says authentication is skipped
    # only when "the request path doesn't start with GUARDED_PREFIX (e.g.
    # /health)". An unauthenticated probe of /v1/models 401s forever.
    return Service("vllm", cmd, Path(a.out_dir) / "logs" / "vllm.log",
                   f"http://127.0.0.1:{port}/health",
                   env={"VLLM_LOGGING_LEVEL": "INFO"}, timeout=a.serve_timeout)


def stage_sandbox(a, port: int) -> Service:
    """
    Where docker is absent (`command -v docker` -> not found), apptainer 1.5.3 is
    installed, so the same OCI image runs under apptainer.

    Two writability facts force the flags below:
      * entrypoint.sh does `envsubst < src/agent_environment/mcp_server_template.json
        > src/agent_environment/mcp_server_config.json` INSIDE the image, and
      * the filesystem/git/memory/code-executor servers write under /data,
    so the container root filesystem cannot be read-only. `sessiondir max size`
    is 64 MB in /etc/apptainer/apptainer.conf, which is too small for the code
    executor, hence a persistent ext3 overlay on /scratch rather than
    --writable-tmpfs. There are no subuid/subgid entries for this user, so
    --fakeroot is unavailable; the image's files are world-readable and the
    servers run fine as an unprivileged uid.
    """
    overlay = Path(a.overlay)
    if not overlay.exists():
        overlay.parent.mkdir(parents=True, exist_ok=True)
        log(f"creating {a.overlay_mb} MB overlay at {overlay}")
        subprocess.run(["apptainer", "overlay", "create",
                        "--size", str(a.overlay_mb), str(overlay)], check=True)
    # Apptainer shares the host network namespace, so there is no port
    # publishing step: overriding the image's CMD to bind an explicit port is
    # enough, and it lets several arms share a node without colliding on 1984.
    # The ENTRYPOINT still runs, so mcp_server_config.json is still generated
    # from the template with the .env values substituted in.
    cmd = ["apptainer", "run",
           "--overlay", str(overlay),
           "--env-file", str(a.env_file),
           "--pwd", "/agent-environment",
           str(a.sif),
           "uv", "run", "python", "-m", "uvicorn", "agent_environment.main:app",
           "--host", "127.0.0.1", "--port", str(port)]
    return Service("sandbox", cmd, Path(a.out_dir) / "logs" / "sandbox.log",
                   f"http://127.0.0.1:{port}/health", timeout=a.sandbox_timeout)


def stage_harness(a, harness_port: int, sandbox_port: int, vllm_port: int) -> Service:
    """
    The MCP-Atlas TypeScript harness. config.ts throws unless LLM_API_KEY and
    LLM_BASE_URL are set, and litellm-strategy.ts posts to
    `${LLM_BASE_URL}/v1/chat/completions`, so LLM_BASE_URL must NOT already
    end in /v1 -- point it at the vLLM root.
    """
    env = {
        "PORT": str(harness_port),
        "MCP_SANDBOX_URL": f"http://127.0.0.1:{sandbox_port}",
        "LLM_BASE_URL": f"http://127.0.0.1:{vllm_port}",
        "LLM_API_KEY": a.api_key,
        "LOG_LEVEL": "info",
        "TOOL_CALL_TIMEOUT_MS": str(a.tool_call_timeout_ms),
        "LIST_TOOLS_TIMEOUT_MS": str(a.list_tools_timeout_ms),
        "LLM_TIMEOUT_MS": str(a.llm_timeout_ms),
    }
    return Service("harness", ["npm", "run", "dev"],
                   Path(a.out_dir) / "logs" / "harness.log",
                   f"http://127.0.0.1:{harness_port}/health",
                   env=env, cwd=Path(a.atlas_repo) / "services" / "agent-harness",
                   timeout=a.harness_timeout)


# ==========================================================================
# run: post tasks to the harness
# ==========================================================================

async def run_tasks(a, tasks, harness_url: str, outputs_csv: Path, traj_jsonl: Path) -> None:
    import aiohttp

    done: set[tuple[str, int]] = set()
    if outputs_csv.exists():
        with open(outputs_csv, newline="") as f:
            for row in csv.DictReader(f):
                tid = (row.get("task_id") or "").strip()
                if tid:
                    done.add((tid.split("#")[0], int(tid.split("#")[1]) if "#" in tid else 0))
        log(f"resume: {len(done)} (task,sample) already present in {outputs_csv}")

    work = [(row, s) for _, row in tasks.iterrows() for s in range(a.n_samples)
            if (row["TASK"], s) not in done]
    log(f"{len(work)} (task,sample) pairs to run, concurrency {a.concurrency}")
    if not work:
        return

    sem = asyncio.Semaphore(a.concurrency)
    fields = ["task_id", "raw_conversation_history", "response"]
    write_header = not outputs_csv.exists()
    lock = asyncio.Lock()

    async def one(session, row, sample):
        task_id = row["TASK"]
        # pass@1 uses one sample and the id is bare, so outputs.csv stays
        # byte-compatible with what run_eval.py would have written.
        out_id = task_id if a.n_samples == 1 else f"{task_id}#{sample}"
        body = {
            "task_id": out_id,
            "model": a.served_model_name,
            "messages": [{"role": "user", "content": row["PROMPT"]}],
            "enabledTools": enabled_tool_names(row["ENABLED_TOOLS"]),
            "image": SANDBOX_IMAGE,
            "tags": {"task_id": out_id},
            "max_turns": a.max_turns,
            "max_tool_calls": a.max_tool_calls,
        }
        if a.tool_output_cap:
            body["tool_output_cap"] = a.tool_output_cap
        if a.context_window_management:
            body["context_window_management"] = a.context_window_management
        extra = json.loads(a.extra_llm_params) if a.extra_llm_params else {}
        if a.n_samples > 1:
            extra.setdefault("temperature", a.temperature)
            extra["seed"] = a.seed + sample
        if extra:
            body["extra_llm_params"] = extra
        if a.system_prompt:
            body["messages"] = [{"role": "system", "content": a.system_prompt}] + body["messages"]

        t0 = time.time()
        async with sem:
            try:
                async with session.post(f"{harness_url}/v2/mcp_eval/run_agent", json=body,
                                        timeout=aiohttp.ClientTimeout(total=a.timeout)) as r:
                    if r.status != 200:
                        txt = (await r.text())[:300]
                        return {"task_id": out_id, "raw_conversation_history": "",
                                "response": f"ERROR: HTTP {r.status}: {txt}"}, {}
                    data = await r.json()
            except asyncio.TimeoutError:
                return {"task_id": out_id, "raw_conversation_history": "",
                        "response": f"ERROR: timeout after {a.timeout}s"}, {}
            except Exception as e:
                return {"task_id": out_id, "raw_conversation_history": "",
                        "response": f"ERROR: {type(e).__name__}: {e}"}, {}

        msgs = [i["data"] for i in data if i.get("type") == "message"]
        stop = next((i.get("data", {}) for i in data
                     if i.get("type") in ("stop", "termination", "loop_end")), {})
        final = ""
        for m in reversed(msgs):
            if m.get("role") == "assistant" and m.get("content"):
                final = m["content"]
                break
        calls = [tc.get("function", {}).get("name")
                 for m in msgs for tc in (m.get("tool_calls") or [])]
        meta = {
            "task_id": task_id, "sample": sample, "wall_s": round(time.time() - t0, 1),
            "n_messages": len(msgs),
            "n_assistant_turns": sum(1 for m in msgs if m.get("role") == "assistant"),
            "n_tool_calls": len(calls), "tools_called": calls,
            "stop_reason": stop.get("reason"),
            "response_chars": len(final),
            "trajectory_chars": sum(len(json.dumps(m)) for m in msgs),
        }
        return ({"task_id": out_id, "raw_conversation_history": json.dumps(msgs),
                 "response": final}, meta)

    async with aiohttp.ClientSession() as session:
        with open(outputs_csv, "a", newline="") as fout, open(traj_jsonl, "a") as fmeta:
            w = csv.DictWriter(fout, fieldnames=fields)
            if write_header:
                w.writeheader()
                fout.flush()

            async def go(row, sample):
                res, meta = await one(session, row, sample)
                async with lock:
                    w.writerow(res)
                    fout.flush()
                    if meta:
                        fmeta.write(json.dumps(meta) + "\n")
                        fmeta.flush()
                return res

            n = 0
            for fut in asyncio.as_completed([go(r, s) for r, s in work]):
                res = await fut
                n += 1
                ok = "OK  " if not res["response"].startswith("ERROR:") else "FAIL"
                log(f"[{n}/{len(work)}] {ok} {res['task_id']}")


def stage_groundtruth(a, tasks, path: Path) -> Path:
    """
    score_claims.py does `pd.merge(df_gtfa, df_model, left_on='TASK',
    right_on='task_id', how='inner')`. With --n-samples > 1 the model rows are
    keyed 'TASK#sample', so a bare ground-truth table inner-joins to ZERO rows
    and the run reports nothing while exiting 0. Replicate the ground truth
    per sample so the join is 1:1 either way.
    """
    import pandas as pd

    gt = tasks[["TASK", "PROMPT", "GTFA_CLAIMS"]].copy()
    if a.n_samples > 1:
        gt = pd.concat(
            [gt.assign(TASK=gt.TASK + f"#{s}") for s in range(a.n_samples)],
            ignore_index=True)
    gt.to_csv(path, index=False)
    return path


def stage_score(a, gt_csv: Path, outputs_csv: Path, score_dir: Path) -> Path:
    script = Path(a.atlas_repo) / "services" / "scoring" / "score_claims.py"
    cmd = [str(a.scorer_python), str(script),
           "--groundtruth-file", str(gt_csv),
           "--model-file", str(outputs_csv),
           "--model-name", a.arm,
           "--output-dir", str(score_dir),
           "--evaluator-model", a.judge_model,
           "--concurrency", str(a.judge_concurrency)]
    if a.judge_base_url:
        cmd += ["--base-url", a.judge_base_url]
    if a.judge_api_key:
        cmd += ["--api-key", a.judge_api_key]
    log("score: " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=str(score_dir))
    return score_dir / f"scored_{a.arm}.csv"


# ==========================================================================
# records: the artifact the paired analysis consumes
# ==========================================================================

def stage_records(a, scored_csv: Path, traj_jsonl: Path, records: Path, manifest: dict,
                  task_index: dict[str, int]) -> None:
    """
    Emit one JSON object per (task, sample), carrying TWO field sets.

    (1) The five fields our existing paired tooling already keys on, so
        main_table.py / rl_learning_curve.py / plot_curve.py / heldout_compare.py
        consume these cells with no code change. Verified in main_table.py:

            k = (d.get("scenario"), d.get("task_idx"), d.get("seed"))
            out[k] = int(d["reward"] > 0)

        so:  scenario = "mcpatlas"          (one benchmark = one scenario)
             task_idx = the task's position in the sorted subset id list,
                        which is stable because the id list is sha256-pinned
             seed     = a.seed + sample
             interface= "RAW", matching the in-domain cells
             reward   = 1 iff claim coverage >= --primary-threshold (0.75),
                        which is exactly OpenForgeRL's success criterion
        Plus the diagnostic names those cells also carry: n_turns, tool_calls,
        stop_reason, final_answer, wall_s, verifier_status.

    (2) The benchmark-native fields, so nothing is lost to the mapping:
        coverage_score (continuous), pass_050/pass_075, claim counts, task_id.

    Continuous `coverage_score` matters. An 8B policy will sit near the floor
    of this benchmark -- OpenForgeRL's Table 2 puts LLaMA-4-Scout-17B at 2.3
    and Mistral-Small-24B at 4.5 pass@1 -- and a binary outcome on 89 tasks
    has very little paired power down there. Report the McNemar on `reward`
    for comparability with the published number, and lean on the paired test
    over `coverage_score` for the actual claim.
    """
    import pandas as pd

    df = pd.read_csv(scored_csv)
    meta_by_key: dict[tuple[str, int], dict] = {}
    if traj_jsonl.exists():
        for line in traj_jsonl.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                meta_by_key[(m["task_id"], m["sample"])] = m

    resp_col = next((c for c in [f"{a.arm}_response", "response"] if c in df.columns), None)
    n = 0
    with open(records, "w") as f:
        for _, r in df.iterrows():
            raw_id = str(r["TASK"])
            task_id, sample = (raw_id.split("#") + ["0"])[:2] if "#" in raw_id else (raw_id, "0")
            sample = int(sample)
            cov = r.get("coverage_score")
            cov = None if pd.isna(cov) else float(cov)
            resp = "" if resp_col is None or pd.isna(r.get(resp_col)) else str(r[resp_col])
            meta = meta_by_key.get((task_id, sample), {})
            err = resp.startswith("ERROR:")
            rec = {
                # ---- (1) fields our paired tooling keys on, unchanged ----
                "scenario": "mcpatlas",
                "task_idx": task_index[task_id],
                "seed": a.seed + sample,
                "interface": "RAW",
                "reward": 0 if cov is None else int(cov >= a.primary_threshold),
                "n_turns": meta.get("n_assistant_turns"),
                "tool_calls": meta.get("tools_called"),
                "stop_reason": meta.get("stop_reason"),
                "final_answer": resp,
                "wall_s": meta.get("wall_s"),
                "verifier_status": "error" if err else ("success" if cov is not None else "unscored"),
                # ---- (2) benchmark-native ----
                "bench": "mcpatlas",
                "subset": a.subset,
                "arm": a.arm,
                "step": a.step,
                "task_id": task_id,
                "sample": sample,
                "coverage_score": cov,
                "pass_050": None if cov is None else int(cov >= 0.50),
                "pass_075": None if cov is None else int(cov >= 0.75),
                "total_claims": int(r["total_claims"]) if not pd.isna(r.get("total_claims")) else None,
                "fully_covered_claims": int(r["fully_covered_claims"]) if not pd.isna(r.get("fully_covered_claims")) else None,
                "partially_covered_claims": int(r["partially_covered_claims"]) if not pd.isna(r.get("partially_covered_claims")) else None,
                "harness_error": err,
                "response_chars": len(resp),
                "n_tool_calls": meta.get("n_tool_calls"),
                "trajectory_chars": meta.get("trajectory_chars"),
                "judge_model": a.judge_model,
                "primary_threshold": a.primary_threshold,
                "run_id": manifest["run_id"],
            }
            f.write(json.dumps(rec) + "\n")
            n += 1

    expected = len(task_index) * a.n_samples if not a.limit else a.limit * a.n_samples
    if n != expected:
        log(f"WARNING: {n} records but {expected} expected. score_claims.py "
            f"inner-joins ground truth on TASK == task_id; a mismatch here "
            f"means rows were dropped, not that tasks failed.")

    rows = [json.loads(l) for l in records.read_text().splitlines()]
    graded = [r for r in rows if r["coverage_score"] is not None]
    log(f"wrote {n} records -> {records}")
    if graded:
        for th in PASS_THRESHOLDS:
            k = sum(1 for r in graded if r["coverage_score"] >= th)
            log(f"  pass@1 (coverage >= {th:.2f}) = {k}/{len(graded)} = {100.0*k/len(graded):.1f}")
        log(f"  mean coverage = {sum(r['coverage_score'] for r in graded)/len(graded):.4f}")
    log(f"  harness errors = {sum(1 for r in rows if r['harness_error'])}")


# ==========================================================================
# mock: dry run with no GPU, no docker, no node
# ==========================================================================

def start_mocks(a, tasks) -> tuple[Service, Service, int, int]:
    """Stub harness + stub judge, both real HTTP servers in child processes."""
    hp, jp = free_port(), free_port()
    stub = Path(a.out_dir) / "logs" / "mock_servers.py"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(MOCK_SERVER_SRC)
    h = Service("mock-harness", [sys.executable, str(stub), "harness", str(hp)],
                Path(a.out_dir) / "logs" / "mock_harness.log",
                f"http://127.0.0.1:{hp}/health", timeout=120).start()
    j = Service("mock-judge", [sys.executable, str(stub), "judge", str(jp)],
                Path(a.out_dir) / "logs" / "mock_judge.log",
                f"http://127.0.0.1:{jp}/health", timeout=120).start()
    h.wait_ready()
    j.wait_ready()
    return h, j, hp, jp


MOCK_SERVER_SRC = r'''
"""Stub harness and stub judge for bench_transfer.py --mock.

harness: answers /v2/mcp_eval/run_agent with the same [{type,data}] shape the
         real TypeScript harness returns, built from an echo "model" that emits
         one tool call and then a final answer.
judge:   answers /v1/chat/completions with the exact json_schema payload
         score_claims.py asks for, deterministically: a claim counts as
         fulfilled iff its first 25 characters appear in the response.
"""
import json, sys, hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer

MODE = sys.argv[1]
PORT = int(sys.argv[2])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        sys.stderr.write((a[0] % a[1:]) + "\n")

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send({"status": "ok", "mode": MODE})
        elif self.path.startswith("/v1/models"):
            self._send({"object": "list", "data": [{"id": "mock", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if MODE == "harness":
            self._send(self.harness(body))
        else:
            self._send(self.judge(body))

    def harness(self, body):
        prompt = body["messages"][-1]["content"]
        tools = body.get("enabledTools", [])
        tool = tools[0] if tools else "calculator_calculate"
        # deterministic per-task so the dry run is reproducible
        seed = int(hashlib.sha256(body.get("task_id", "").encode()).hexdigest()[:8], 16)
        msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "I will look that up.",
             "tool_calls": [{"id": "call_0", "type": "function",
                             "function": {"name": tool, "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_0",
             "content": "MOCK TOOL OUTPUT for " + tool},
            {"role": "assistant",
             "content": "ECHO ANSWER seed=%d. %s" % (seed, prompt[:120])},
        ]
        out = [{"type": "message", "data": m} for m in msgs]
        out.append({"type": "stop", "data": {"reason": "natural_stop"}})
        return out

    def judge(self, body):
        text = body["messages"][0]["content"]
        # score_claims.py embeds both the claim and the response in one prompt;
        # fulfil the claim iff its opening words are echoed back in the prompt.
        outcome = "not_fulfilled"
        if "ECHO ANSWER" in text and len(text) % 3 == 0:
            outcome = "fully_fulfilled"
        elif "ECHO ANSWER" in text and len(text) % 3 == 1:
            outcome = "partially_fulfilled"
        content = json.dumps({
            "claim_text": "mock",
            "coverage_outcome": outcome,
            "justification": "mock judge: deterministic function of prompt length",
            "confidence_level": 0.9,
        })
        return {"id": "mock", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


HTTPServer.allow_reuse_address = True
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
'''


# ==========================================================================
# main
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MCP-Atlas transfer eval of a trained checkpoint (eval only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("what to evaluate")
    g.add_argument("--arm", required=True, help="label, e.g. a8T5k_step25")
    g.add_argument("--ckpt", help="verl global_step_N dir, merged HF dir, or HF model id")
    g.add_argument("--merged-dir", help="skip merging; serve this dir")
    g.add_argument("--step", type=int, default=None, help="training step, recorded only")
    g.add_argument("--out-dir", required=True, help="run directory (put it on /scratch)")

    g = p.add_argument_group("benchmark")
    g.add_argument("--atlas-repo", required=True, help="clone of github.com/scaleapi/mcp-atlas")
    g.add_argument("--parquet", help="MCP-Atlas.parquet (downloaded if absent)")
    g.add_argument("--subset", default="default20_89",
                   choices=["default20_89", "all500", "file"])
    g.add_argument("--subset-file", help="task id list, one per line, for --subset file")
    g.add_argument("--print-subset", action="store_true", help="print the task ids and exit")
    g.add_argument("--sif", help="apptainer SIF of ghcr.io/scaleapi/mcp-atlas:1.2.7")
    g.add_argument("--overlay", help="writable ext3 overlay for the sandbox")
    g.add_argument("--overlay-mb", type=int, default=8192)
    g.add_argument("--env-file", help=".env passed into the sandbox")

    g = p.add_argument_group("generation")
    g.add_argument("--n-samples", type=int, default=1, help="samples per task; 1 = pass@1")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-turns", type=int, default=256, help="harness default")
    g.add_argument("--max-tool-calls", type=int, default=100, help="harness default")
    g.add_argument("--tool-output-cap", type=int, default=None,
                   help="chars per tool result; ground-truth trajectories reach 995k chars, "
                        "so leave this OFF only if max-model-len can absorb them")
    g.add_argument("--context-window-management", choices=["compact"], default=None)
    g.add_argument("--system-prompt", default=None)
    g.add_argument("--extra-llm-params", default=None, help="JSON forwarded to the model")
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--timeout", type=int, default=1800, help="per-task seconds")

    g = p.add_argument_group("serving")
    g.add_argument("--served-model-name", default="policy")
    g.add_argument("--api-key", default="mcpatlas-local")
    g.add_argument("--max-model-len", type=int, default=32768)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    g.add_argument("--tensor-parallel-size", type=int, default=1)
    g.add_argument("--tool-call-parser", default="hermes")
    g.add_argument("--reasoning-parser", default=None)
    g.add_argument("--chat-template", default=None)
    g.add_argument("--vllm-extra-args", default="")
    g.add_argument("--serve-timeout", type=int, default=1800)
    g.add_argument("--sandbox-timeout", type=int, default=900)
    g.add_argument("--harness-timeout", type=int, default=600)
    g.add_argument("--tool-call-timeout-ms", type=int, default=60000)
    g.add_argument("--list-tools-timeout-ms", type=int, default=180000)
    g.add_argument("--llm-timeout-ms", type=int, default=600000)

    g = p.add_argument_group("merging")
    g.add_argument("--base", default=None, help="base HF snapshot dir (default: local snapshot of --base-repo)")
    g.add_argument("--base-repo", default="Qwen/Qwen3-VL-8B-Instruct")
    g.add_argument("--merged-out", default=None,
                   help="where to write merged weights; keep it on /scratch (~17G per arm)")
    g.add_argument("--merge-python", default=str(VERL_PY))
    g.add_argument("--merge-script", default=str(MCP_ROOT / "surface" / "verl_rl" / "merge_lora.py"))
    g.add_argument("--merge-args", default="", help="extra argv appended to merge_lora.py")
    g.add_argument("--force-merge", action="store_true")

    g = p.add_argument_group("scoring")
    g.add_argument("--scorer-python", default=str(VERL_PY),
                   help="needs tenacity, nest_asyncio, matplotlib on top of mcp_verl")
    g.add_argument("--judge-model", default="gemini/gemini-2.5-pro",
                   help="OpenForgeRL used Gemini 2.5 Pro; the repo default is "
                        "gemini/gemini-3.1-pro-preview")
    g.add_argument("--judge-base-url", default=None, help="OpenAI-compatible judge endpoint")
    g.add_argument("--judge-api-key", default=None)
    g.add_argument("--judge-concurrency", type=int, default=8)
    g.add_argument("--primary-threshold", type=float, default=0.75,
                   help="OpenForgeRL counts a task successful at coverage >= 0.75")

    g = p.add_argument_group("control")
    g.add_argument("--stages", default="merge,sandbox,serve,harness,run,score,records")
    g.add_argument("--mock", action="store_true",
                   help="dry run: stub harness + stub judge, no GPU/docker/node")
    g.add_argument("--limit", type=int, default=None, help="first N tasks only")
    return p


def main() -> None:
    a = build_parser().parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    atlas = Path(a.atlas_repo)
    template = atlas / "services" / "agent-environment" / "src" / "agent_environment" / "mcp_server_template.json"
    if not template.exists():
        raise SystemExit(f"not an mcp-atlas checkout: missing {template}")
    parquet = fetch_parquet(Path(a.parquet) if a.parquet else out / "MCP-Atlas.parquet")

    tasks = load_tasks(parquet, template, a.subset,
                       Path(a.subset_file) if a.subset_file else None)

    # task_idx must be stable across arms or the pairing silently misaligns.
    # It is the position in the sorted FULL subset id list -- computed before
    # --limit, so a smoke run on 3 tasks produces the same task_idx values the
    # full run will, and the two can be pooled. The id list is sha256-pinned in
    # load_tasks(), so the mapping cannot drift unnoticed between arms.
    task_index = {t: i for i, t in enumerate(sorted(tasks.TASK))}

    if a.limit:
        tasks = tasks.head(a.limit)
    log(f"{len(tasks)} tasks, subset={a.subset}")

    if a.print_subset:
        for t in sorted(tasks.TASK):
            print(t)
        return

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    run_id = f"{a.arm}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = {
        "run_id": run_id, "arm": a.arm, "step": a.step, "bench": "mcpatlas",
        "subset": a.subset, "n_tasks": int(len(tasks)),
        "task_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(tasks.TASK)) + "\n").encode()).hexdigest(),
        "parquet_sha256": PARQUET_SHA256, "sandbox_image": SANDBOX_IMAGE,
        "atlas_repo_head": subprocess.run(["git", "-C", str(atlas), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
        "argv": sys.argv, "args": vars(a), "mock": a.mock,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    outputs_csv = out / "outputs.csv"
    traj_jsonl = out / "trajectory_meta.jsonl"
    gt_csv = out / "groundtruth.csv"
    score_dir = out / "scoring"
    score_dir.mkdir(exist_ok=True)
    records = out / "records.jsonl"

    services: list[Service] = []
    try:
        if a.mock:
            h, j, harness_port, judge_port = start_mocks(a, tasks)
            services += [h, j]
            a.judge_base_url = f"http://127.0.0.1:{judge_port}"
            a.judge_api_key = a.judge_api_key or "mock"
            a.judge_model = "mock-judge"
            manifest["mock_ports"] = {"harness": harness_port, "judge": judge_port}
        else:
            harness_port = free_port()
            if "merge" in stages:
                model_dir = stage_merge(a)
                manifest["model_dir"] = str(model_dir)
            else:
                model_dir = Path(a.merged_dir or a.ckpt)
            sandbox_port, vllm_port = free_port(), free_port()
            if "sandbox" in stages:
                for req, why in [(a.sif, "--sif"), (a.overlay, "--overlay"), (a.env_file, "--env-file")]:
                    if not req:
                        raise SystemExit(f"{why} is required for the sandbox stage")
                services.append(stage_sandbox(a, sandbox_port).start())
            if "serve" in stages:
                services.append(stage_serve(a, model_dir, vllm_port).start())
            for s in services:
                s.wait_ready()
            if "harness" in stages:
                hs = stage_harness(a, harness_port, sandbox_port, vllm_port).start()
                hs.wait_ready()
                services.append(hs)
            manifest["ports"] = {"harness": harness_port, "sandbox": sandbox_port,
                                 "vllm": vllm_port}

        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

        if "run" in stages:
            asyncio.run(run_tasks(a, tasks, f"http://127.0.0.1:{harness_port}",
                                  outputs_csv, traj_jsonl))
        if "score" in stages:
            stage_groundtruth(a, tasks, gt_csv)
            scored = stage_score(a, gt_csv, outputs_csv, score_dir)
        else:
            scored = score_dir / f"scored_{a.arm}.csv"
        if "records" in stages:
            stage_records(a, scored, traj_jsonl, records, manifest, task_index)

        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        log(f"done. records -> {records}")
    finally:
        for s in reversed(services):
            s.stop()


if __name__ == "__main__":
    main()
