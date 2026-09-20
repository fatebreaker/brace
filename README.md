# Measure First, Then Allocate: Calibrated Rollout Budgets for MCP Agents in Reinforcement Learning

Code release for the paper. Anonymised for double-blind review: no author, institution,
account, host or absolute site path appears anywhere in this repository, and every
location is resolved from an environment variable at run time.

## What is here

**BRACE** (Bank-calibrated Rollout Allocation under Correlated Episodes) decides, before a
GRPO cycle generates anything, how many rollouts each task should receive. A group of `k`
rollouts on a task solved with probability `p` carries gradient with probability
`g(p, k) = 1 - p^k - (1-p)^k`, so allocation is a forecasting problem: estimate `p`, take
the expectation of `g` under its posterior, and spend where the expected gradient mass is.
Three components make that forecast usable on a live agent environment. A **warm bank** of
banked outcomes from previous cycles supplies the prior, so cycle one is not uniform. A
**shrinkage calibration** corrects the posterior for within-task correlation between
episodes sharing a task, through a single correlation constant `rho` (equivalently
`nu = (1-rho)/rho`) that can be refit per scale. And a **certified exclusion** removes tasks
proved unsolvable under the current harness, so they stop consuming budget.

**DISCORD** is the paired-inference analysis BRACE is built on. Per-task outcomes in an
agent environment are binary and the arms being compared are paired, so the data form a
2x2 table in which only the discordant pairs carry information; the exact conditional
(McNemar) test on those pairs is calibrated where both `t`-tests are not, and it yields a
required-sample-size formula in the discordance rate rather than the raw task count. The
same object read one level upstream, at the level of a GRPO group rather than a retention
decision, is what BRACE forecasts.

The repository contains the MCP agent loop and reward, the allocator, the cycle-restart
training driver, the evaluation and transfer-benchmark harnesses, the DISCORD analysis and
its validation experiments, and the scripts that produce every number and figure in the
paper. It contains no data, no checkpoints, no logs and no paper sources.

## Repository layout

```
env.sh                              locates everything; source it first
setup_envs.sh, setup_vllm.sh        build the analysis, AWM and vLLM environments
fetch_awm_data.py                   download the AWM scenario pack
patches/                            a required patch to installed verl (DAPO baseline)
recipes/                            one command per paper row: train, evaluate, tabulate
slurm/
  setup_verl.sh                     build the verl training environment
  verl_awm_train.sh                 one verl GRPO run over the MCP environment
  verl_awm_train_agent.sh           the DAPO wrapper over that trainer
  coadapt_eval.sh                   one evaluation cell (a checkpoint on a fixed surface)
  queue_curve.sh, run_cells.sh      evaluation-cell queueing
  supervisor.sh                     the fleet scheduler the original runs used
  h200_arm.sh, gpu_release.sh       single-allocation launch and cleanup helpers
surface/
  verl_rl/                          the method, the training driver, evaluation, analysis
  gate_surface/                     the MCP/AWM environment: servers, tools, verifiers
  gate_caller/                      the caller, the task pools, the DISCORD experiments
  paper/                            figure and table generators for Sections 2 to 4
```

### The files that matter most

| file | role |
|---|---|
| `surface/verl_rl/triage.py` | the allocator: posterior expected gradient mass, warm bank, shrinkage calibration, certified exclusion |
| `surface/verl_rl/coadapt.py` | the cycle-restart driver: prepares the pool, runs one verl cycle, re-enters the allocator, repeats |
| `surface/verl_rl/awm_agent_loop.py` | the verl agent loop: one episode is a multi-turn MCP tool-calling rollout against a live server |
| `surface/verl_rl/dense_reward.py` | reward construction from the task's own verifier |
| `surface/verl_rl/accord.py` | concordance certification of a shaped reward (`carve.py` is a symlink to it) |
| `surface/verl_rl/elsa.py` | elasticity-steered allocation, a measured negative reported in the paper |
| `surface/verl_rl/fit_task_rho.py` | the per-scale refit of the correlation constant |
| `surface/verl_rl/coadapt_eval.py` | scores one checkpoint on the held-out pool, one binary outcome per (task, seed) |
| `surface/gate_caller/discordant_gate.py` | DISCORD: the exact conditional test, with curtailment |
| `surface/gate_caller/required_budget.py` | the sample-size calculation in the discordance rate |
| `surface/gate_caller/null_power.py` | calibration and power of the four candidate acceptance procedures |
| `surface/gate_caller/awm_discord.py`, `awm_naive.py` | the live head-to-head on MCP servers |
| `surface/verl_rl/paper_numbers.py` | every table the paper prints, and the row registry |
| `surface/verl_rl/main_table.py` | cell loading, pairing, McNemar, per-arm bookkeeping |
| `surface/verl_rl/paper_figures.py`, `plot_curve.py` | the figures drawn from training cells |
| `surface/paper/make_figures.py` | the Section 2 to 4 diagnostics |
| `surface/paper/figs/make_figure3.py`, `make_figure4.py` | Figures 3 and 4, self-contained: matplotlib only, values frozen from the emitters above, for redrawing without a cluster |

## Environment variables

Everything is located from four variables. Nothing else needs editing.

| variable | meaning |
|---|---|
| `BRACE_ROOT` | this checkout. `env.sh` auto-detects it from its own location if unset. |
| `BRACE_WORK` | large scratch space for checkpoints, merged models, episode logs, evaluation records and benchmark working directories. Defaults to `$BRACE_ROOT/work`; point it at a filesystem with several hundred GB free. |
| `BRACE_ENVS` | parent directory of the python environments. Defaults to `$BRACE_ROOT/envs`. |
| `HF_HOME` | HuggingFace cache holding the base models. Defaults to `$BRACE_ROOT/hf_cache`. |

`env.sh` derives `MCP_PY`, `AWM_PY`, `VLLM_PY` and `VERL_PY` from `BRACE_ENVS`; each can be
overridden individually. `JOBID` names the batch allocation that already holds the GPUs, as
described under "Cluster assumptions" below.

## Installation

Four python environments, deliberately separate. The version chains below are forced by
upstream constraints rather than chosen, and the setup scripts pin what has to be pinned.

```bash
export BRACE_ROOT=/path/to/this/checkout
export BRACE_WORK=/path/with/room
. "$BRACE_ROOT/env.sh"

bash setup_envs.sh          # mcp_surface and mcp_awm
bash setup_vllm.sh          # mcp_vllm
bash slurm/setup_verl.sh    # mcp_verl
```

| environment | python | contents | used for |
|---|---|---|---|
| `mcp_surface` | 3.11 | torch (cu128), `transformers==4.57.6`, accelerate, `huggingface_hub[hf_xet]`, safetensors, numpy, scipy, scikit-learn, sentencepiece, protobuf, `mcp<2` | analysis, the gates, the paper scripts |
| `mcp_awm` | 3.12 | `mcp<2`, fastapi, `uvicorn[standard]`, sqlalchemy, pydantic, fastapi_mcp. No torch and no transformers, on purpose | the Agent World Model MCP servers |
| `mcp_vllm` | 3.12 | vllm, `mcp<2`, numpy, scipy, scikit-learn, sentencepiece, protobuf, ninja, cmake | evaluation and transfer-benchmark serving |
| `mcp_verl` | 3.12 | `verl==0.8.0`, `vllm==0.12.0`, `torch==2.9.0`, a prebuilt flash-attn wheel matched to the torch ABI | training |

Two pins are load-bearing and both are explained in the scripts. `mcp<2`: PyPI `mcp` 2.0.0
renames `streamablehttp_client`, so `gate_surface/tool_runtime.py` fails at import and the
whole harness becomes unloadable. `transformers==4.57.6` is the floor for Qwen3-VL and the
version the reference measurements were taken under. `mcp_verl` is capped fourteen minor
versions behind the serving stack because verl 0.8.0 requires `vllm<=0.12.0`, which is why
training and evaluation cannot share one environment.

A fifth environment is needed only for the BFCL transfer cell, because `bfcl_eval` pins
`numpy==1.26.4` and would otherwise shadow the serving stack's numpy 2.x:

```bash
"$BRACE_ENVS/mcp_vllm/bin/python" -m venv --system-site-packages "$BRACE_WORK/bfcl/bfcl_venv"
"$BRACE_WORK/bfcl/bfcl_venv/bin/pip" install -e "$BRACE_WORK/bfcl/gorilla/berkeley-function-call-leaderboard"
"$BRACE_WORK/bfcl/bfcl_venv/bin/pip" install soundfile
export BFCL_PY="$BRACE_WORK/bfcl/bfcl_venv/bin/python"
```

`soundfile` is not a BFCL dependency; BFCL imports every model handler eagerly and one of
them reaches it, so `bfcl generate` dies at import without it. vLLM must still be launched
out of `mcp_vllm`, never out of this venv.

Finally, apply the verl patch. Without it the DAPO baseline is a silent no-op:

```bash
cd "$BRACE_ENVS/mcp_verl/lib/python3.12/site-packages"
patch -p0 verl/trainer/ppo/ray_trainer.py < "$BRACE_ROOT/patches/verl_0.8.0_dapo_dynamic_sampling.patch"
```

## Data preparation

No data is included. Five external artefacts are needed.

**Base models.** `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct` and
`Qwen/Qwen3-VL-8B-Instruct`, downloaded into `$HF_HOME`. The training and evaluation code
resolves them by snapshot directory under `$HF_HOME/hub`, so a plain
`huggingface-cli download` of each is enough.

**The MCP environment (AWM).** The training environment is the Agent World Model: one
scenario is a FastAPI plus fastapi-mcp server over one SQLite database, and one episode is
a multi-turn tool-calling rollout against a live instance of it, scored by the scenario's
own code verifier. Clone the AWM server code, then fetch its generated environments:

```bash
git clone https://github.com/Snowflake-Labs/agent-world-model \
    "$BRACE_ROOT/surface/mcp_probe/repos/agent-world-model"     # or set AWM_REPO
"$MCP_PY" fetch_awm_data.py     # Snowflake/AgentWorldModel-1K -> <repo>/outputs
```

`surface/gate_surface/prep_scenarios.py` turns those environments into scenarios, and
`surface/verl_rl/prep_awm.py` turns a task pool into the parquet the verl trainer reads.
The task pools themselves (`surface/gate_caller/pools/*.json`), the held-out validation
pool, the advertised tool-surface lists and the certified-dead task list
(`excluded_tasks.jsonl`) are generated artefacts and are not in this repository; the
scripts that build them are.

**BFCL.** Berkeley Function Calling Leaderboard v4, multi-turn split, 800 entries.
Apache-2.0. Pinned at commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`; the data ships
inside the package, so no download and no credentials are needed.

```bash
git clone https://github.com/ShishirPatil/gorilla "$BRACE_WORK/bfcl/gorilla"
git -C "$BRACE_WORK/bfcl/gorilla" checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
```

**NESTFUL.** Apache-2.0, pinned at commit `fc2c4123e73500a56185a5fb354f05d1c8b4890c`.

```bash
git clone https://github.com/IBM/NESTFUL "$BRACE_WORK/nestful/nestful_repo"
git -C "$BRACE_WORK/nestful/nestful_repo" checkout fc2c4123e73500a56185a5fb354f05d1c8b4890c
```

Both benchmark harnesses verify a sha256 of their own task index at run time, so a dataset
revision bump fails loudly instead of silently changing the cell.

`surface/verl_rl/bench_transfer.py` targets a third public benchmark, MCP-Atlas. That cell
was dropped from the paper, but the file is retained because `bench_bfcl.py` and
`bench_nestful.py` import their merge, serve and service-management plumbing from it.

## Training

One command per paper row:

```bash
export BRACE_ROOT=... BRACE_WORK=... JOBID=<allocation holding the GPUs>
recipes/train_arm.sh brace-8b
```

`train_arm.sh` prints the internal arm tag it is about to run and then execs
`surface/verl_rl/coadapt.py`. The protocol is identical for every row: fifteen cycles of
eight optimizer steps, `k = 5` rollouts per group, 32-episode batches in 8-episode
mini-batches, 12288-token responses, learning rate `1e-4`, LoRA rank and alpha 32, a
checkpoint every five steps and a held-out validation pass every five steps against the
fixed advertised tool surface. What varies between rows is the allocation rule and the
components switched on, which is what the flag column below shows.

### Paper row to command, and to internal tag

Internal tags are how a run is named on disk; they are not paper identifiers. Every row
carries the same seed offset as the arm the paper reports, and a second seed is a separate
row rather than a flag.

| Paper row | Scale | `recipes/train_arm.sh <row>` | Tag(s) | What distinguishes it |
|---|---|---|---|---|
| BRACE (full) | 2B | `brace-2b`, `brace-2b-seed2`, `brace-2b-seed3` | `q2bT`, `q2bT2`, `q2bT3` | `--task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks --warm-shrink group` |
| BRACE (full) | 4B | `brace-4b`, `brace-4b-seed2` | `q4bT`, `q4bT2` | as above |
| BRACE (full) | 8B | `brace-8b`, `brace-8b-seed2`, `brace-8b-seed3` | `a8T3g`, `a8T3gr`, `a8T3g2` | as above at `--temp 0.5` |
| uniform GRPO (control) | 2B | `grpo-2b`, `grpo-2b-seed2`, `grpo-2b-seed3` | `q2bF5`, `q2bF6`, `q2bF7` | no allocation flags |
| uniform GRPO (control) | 4B | `grpo-4b`, `grpo-4b-seed2` | `q4bF`, `q4bF2` | no allocation flags |
| uniform GRPO (control) | 8B | `grpo-8b`, `grpo-8b-seed2` | `a8F`, `a8Fr` | no allocation flags |
| PLR | 2B / 4B / 8B | `plr-2b`, `plr-4b`, `plr-8b`, `plr-8b-seed2` | `q2bP`, `q4bP`, `b8plr`, `b8plrr` | `--reweight variance` |
| RAG-MCP | 2B / 4B / 8B | `ragmcp-2b`, `ragmcp-4b`, `ragmcp-8b` | `q2bR`, `q4bR`, `b8ret` | `--surface-retrieval --retrieval-match-level 0.5` |
| DAPO | 2B / 4B / 8B | `dapo-2b`, `dapo-4b`, `dapo-8b` | `q2bD`, `q4bD`, `dapo` | the patched trainer, `DAPO_MAX_GEN_BATCHES=12`, `MINIBSZ=4` |
| TSCL | 2B / 4B / 8B | `tscl-2b`, `tscl-4b`, `tscl-8b` | `q2bLp`, `q4bLp`, `a8Tlp` | `--task-alloc progress --warm-bank default` |
| VIP (as published) | 2B / 4B / 8B | `vip-2b`, `vip-4b`, `vip-8b` | `q2bVf`, `q4bTvip`, `a8Tvipf` | `--task-alloc vip`, no warm bank |
| TRACE (as published) | 2B / 4B / 8B | `trace-2b`, `trace-4b`, `trace-8b` | `t2bTf`, `q4bTf`, `t8Tf` | `--task-alloc gradmass --estimator point`, no warm bank |
| per-scale rho | 2B / 4B / 8B | `refit-2b`, `refit-4b`(`-seed2`,`-seed3`), `refit-8b` | `q2bTr`, `q4bTr`/`q4bTr2`/`q4bTr3`, `a8Tr` | full method plus `TRIAGE_NU` = 0.4582 / 0.3501 / 0.3713 |
| allocation rule to VIP's, bank kept | 2B / 4B / 8B | `vipbank-2b`, `vipbank-4b`(`-seed2`), `vipbank-8b`(`-seed2`) | `q2bV`, `q4bV`/`q4bV2`, `a8Tvip`/`a8Tvipr` | `--task-alloc vip --warm-bank default` |
| estimator: posterior to plug-in | 2B / 4B / 8B | `plugin-2b`, `plugin-4b`(`-seed2`), `plugin-8b` | `q2bTpe`, `q4bTpe`/`q4bTpe2`, `a8Tpe` | full method plus `--estimator point` |
| sharper sampling weights | 8B only | `sharp-8b`, `sharp-8b-seed2` | `a8T5k`, `a8T5kr` | `--temp 0.3` where the 8B default is 0.5 |
| w/o warm bank | 2B / 4B / 8B | `nobank-2b`, `nobank-4b`, `nobank-8b` | `q2bTnw`, `q4bTnw`, `a8Tnw` | full method minus `--warm-bank default` |
| w/o shrinkage calibration | 2B / 4B / 8B | `noshrink-2b`, `noshrink-4b`, `noshrink-8b` | `q2bTiid`, `q4bTiid`, `a8Tiid` | `--warm-shrink group` becomes `--warm-shrink none` |
| learning-rate ladder (2B) | 2B | `lr-grpo-1e5-2b`, `lr-grpo-3e5-2b`, `lr-brace-3e5-2b` | `q2bF2`, `q2bF3e5`, `q2bT3e` | `LR` alone changes |
| pre-registered reruns | 2B / 4B | `rerun-vipbank-2b`, `rerun-tscl-2b`, `rerun-tscl-4b` | `q2bV2`, `q2bLp2`, `q4bLp2` | same recipe and same seed offset as the parent, rerun |
| durability continuation, BRACE | 2B / 4B / 8B | `dur-brace-2b`, `dur-brace-4b`, `dur-brace-8b` | `q2bTc`, `q4bTr3c`, `a8T3g2c` | continues past the window to step 60 |
| durability continuation, GRPO | 2B / 4B / 8B | `dur-grpo-2b`, `dur-grpo-4b`, `dur-grpo-8b` | `q2bF6L`, `q4bFc`, `a8Fc` | as above |
| durability continuation, PLR | 2B / 4B | `dur-plr-2b`, `dur-plr-4b` | `q2bPc`, `q4bPc` | as above |

Durability rows come in two protocols. A "merge" continuation restarts from the parent's
merged step-30 weights, so `train_arm.sh` refuses to launch until that merged directory
exists and prints the `merge_lora.py` command that builds it. A "resume" or "straight"
continuation carries the parent's own trajectory forward.

### GPU shapes

`SHAPE` selects the hardware profile; the default is the shape each scale was trained on.

| scale | shape | `SHAPE` | settings |
|---|---|---|---|
| 2B | one 80GB card | `a100` | `NGPUS=1 TP=1 GPU_UTIL=0.35` |
| 4B | one 80GB card | `a100` | `NGPUS=1 TP=1 GPU_UTIL=0.30 PARAM_OFFLOAD=True OPT_OFFLOAD=True` |
| 4B | two 80GB cards | `a100x2` | `NGPUS=2 TP=2 GPU_UTIL=0.30` with both offloads |
| 8B | two 80GB cards | `a100x2` | `NGPUS=2 TP=2 GPU_UTIL=0.29 PARAM_OFFLOAD=True OPT_OFFLOAD=True MAXTOK=28672` |
| 8B | one 141GB card | `h200` | `NGPUS=1 TP=1 GPU_UTIL=0.27` |

`GPU_UTIL` is the fraction of the card handed to the vLLM rollout engine, and it is set per
card from measured free memory rather than being a property of the recipe. `MAXTOK` is
`ppo_max_token_len_per_gpu`, which governs how one mini-batch is packed into micro-batches
for the actor update; the recipe value is 28672 at every scale. A single 8B run does not
fit on one 80GB card, and 46GB cards cannot train any of these recipes and are useful only
for evaluation.

### Recipe-selecting environment variables

| variable | meaning | default |
|---|---|---|
| `TRIAGE_NU` | `nu = (1-rho)/rho`, the shrinkage constant | the single global `rho`; per-scale refits are 0.4582 (2B), 0.3501 (4B), 0.3713 (8B) |
| `TRIAGE_EPS` | floor on the allocation weights | 0.05 where set |
| `TRIAGE_DECAY` | decay applied to banked evidence between cycles | 1.0 (no decay) |
| `NROLL` | rollouts per group, `k` | 5 |
| `BSZ` / `MINIBSZ` | episodes per optimizer batch and per mini-batch | 32 / 8 |
| `MAXTOK` | actor `ppo_max_token_len_per_gpu` | 28672 |
| `RESP_LEN` | maximum response tokens | 12288 |
| `TP` / `NGPUS` | tensor parallelism and cards | per shape |
| `PARAM_OFFLOAD`, `OPT_OFFLOAD` | FSDP parameter and optimizer offload | per shape |
| `GPU_UTIL` | vLLM `gpu_memory_utilization` | per shape |
| `SAVE_FREQ`, `VAL_FREQ`, `KEEPCKPT` | checkpoint interval, validation interval, checkpoints retained | 5, 5, 12 |
| `SEED_OFFSET` | seed offset; each arm has a registered one | per row |
| `EXCLUDE` | certified-dead task list | `$BRACE_WORK/analysis/excluded_tasks.jsonl` |
| `POOL` | task pool | `surface/gate_caller/pools/pool_max.json` |

## Evaluation

**Window cells.** The paper's main statistic is a four-step window mean over steps 15, 20,
25 and 30. Each cell merges the LoRA adapter at that step, serves it, replays the held-out
task pool on the fixed initial tool surface and writes one binary outcome per (task, seed)
into `$BRACE_WORK/coadapt_eval_<tag>/cell_STEP<n>.jsonl`. That file is what
`main_table.load_cell` reads, so every table is downstream of it.

```bash
JOBID=<allocation> recipes/eval_arm.sh window q4bT
JOBID=<allocation> recipes/eval_arm.sh anchor q4bT        # the step-0 base-policy anchor
```

A cell resumes from what is already banked, so raising `SEEDS` adds seeds rather than
redoing the run. Before scoring, check that a checkpoint directory actually holds tensors:
a pruned directory keeps its config files and `huggingface/` subdirectory and looks
complete, but has no `model_world_size_*_rank_*.pt`, and the evaluation exits successfully
having measured nothing.

**Durability continuation.** The same machinery at steps 40, 50 and 60 on the continuation
arm:

```bash
JOBID=<allocation> recipes/eval_arm.sh durability a8T3g2c
```

**Transfer benchmarks.** One card each, run sequentially:

```bash
JOBID=<allocation> BFCL_PY=$BRACE_WORK/bfcl/bfcl_venv/bin/python \
    recipes/eval_arm.sh bfcl    q4bT 30
JOBID=<allocation> recipes/eval_arm.sh nestful q4bT 30
```

Both write `records.jsonl` with the same five fields the in-domain cells carry, so a
transfer cell drops into the same paired machinery with no code change. Both resume
generation, which is the expensive stage, and both re-verify their task-index sha256.
`bench_nestful.py --oracle --limit 40` is a CPU-only dry run that feeds gold call sequences
through the real scorer and asserts they score 1.0.

## Regenerating the tables and figures

```bash
recipes/make_tables_and_figures.sh "$BRACE_WORK/paper_out"
```

No GPU. `paper_numbers.py` owns everything read out of the training cells: it holds the row
registry (`CFG_ARMS`, `ABL_ARMS`, `CONTROL_ARMS`, `PUBLISHED_REIMPL`, `PUBLISHED_ASPUB`,
`DURROWS`, `TRANSFER_FAITHFUL`), the pinned multiple-comparison families (`FAMILY_PIN`) and
the emitters for every table. `--emit-tables DIR` writes the LaTeX fragments,
`--emit-figures DIR` writes the figures drawn from the same aggregation the tables use, and
running it with no flags prints the diagnostics. `surface/paper/make_figures.py` covers the
Sections 2 to 4 diagnostics, which read the retention-experiment receipts rather than
training cells; those receipt JSON files are generated artefacts and are not in this
repository, so those figures fall back to the literals recorded in the script until the
experiments under `surface/gate_caller/` have been rerun.

## Cluster assumptions

The code was written for a shared multi-user GPU cluster with a batch scheduler, and a few
of its habits show. None is deep, but a reproduction on other hardware will meet them.

* **A long-lived allocation, not one job per run.** `coadapt.py` runs on a head node and
  dispatches each training cycle into an allocation that already exists, named by `JOBID`,
  using `srun --jobid=... --overlap`. This is what makes cycle restart cheap. On a machine
  with direct GPU access, replace the `srun` prefix inside `slurm/h200_arm.sh` and
  `recipes/eval_arm.sh` with a plain shell invocation.
* **Shared nodes.** `slurm/gpu_release.sh` sweeps stale processes in the allocation's
  cgroup before a cycle boots, and `GPU_RELEASE_KEEP` is an extended regex of command lines
  to spare. Its default here protects allocation keepalives only. On a node shared with
  other tenants, add their process patterns, or the sweep will kill them.
* **Two cards means two arms.** `slurm/supervisor.sh` is the fleet scheduler the original
  runs used: it packs several arms onto a pool of allocations, promotes and demotes them,
  and carries the per-arm recipe table that `recipes/train_arm.sh` distils. It is included
  because it is the launch record of what was run, not because it is portable.
* **Checkpoints belong off the project filesystem.** Thirty steps of an 8B run at
  `KEEPCKPT=12` is a large amount of disk. `BRACE_WORK` exists so those never land next to
  the code.
* **verl checkpoints are world-size locked.** A run started at `NGPUS=2` cannot resume on
  one card, and the reverse also fails. Pick the shape before the first cycle.
* **Offline HuggingFace.** `slurm/verl_awm_train.sh` sets `HF_HUB_OFFLINE=1`, so every base
  model must be in `$HF_HOME` before training starts.

## Changes from the working tree

Files are copied verbatim except for the following. Every change is either anonymisation or
the import-path and interpreter-path fixes that anonymisation makes necessary.

1. **Absolute site paths replaced by environment variables.** Every occurrence of the
   project root, the environments directory and the scratch archive is now
   `BRACE_ROOT`, `BRACE_ENVS` and `BRACE_WORK`. In python, a module-level
   `R = "<absolute path>"` became
   `R = os.environ.get("BRACE_ROOT", <three dirnames up from __file__>)`, so a checkout
   works with no configuration at all; the same substitution was applied to `MCP_ROOT`,
   `SURF`, `NESTFUL_ROOT`, `BFCL_ROOT` and `W`. In shell, `R=<absolute path>` became
   `R=${BRACE_ROOT:?...}`.
2. **Hard-coded interpreter paths replaced by resolved constants.** `coadapt.py`,
   `triage.py`, `aws_reweight.py`, `surface_retrieval.py` and `heuristic_surface.py` gained
   a four-line block defining `ENVS`, `VERL_PY`, `AWM_PY` and `VLLM_PY` from the
   environment, and the literal interpreter paths in their subprocess commands now refer to
   those names. `bench_transfer.py`'s `ENVS` is read from `BRACE_ENVS`.
3. **`env.sh` rewritten.** It now derives `BRACE_ROOT` from its own location, defines
   `BRACE_WORK` and `BRACE_ENVS`, exports the four interpreter variables, and keeps the
   original cache placement and toolchain pinning. The site-specific narrative it carried
   was removed.
4. **`MAMBA` is no longer an absolute path.** `setup_envs.sh`, `setup_vllm.sh` and
   `slurm/setup_verl.sh` use `MAMBA=${MAMBA:-mamba}`.
5. **`GPU_RELEASE_KEEP`'s default narrowed.** It previously named specific processes
   belonging to another group on the shared cluster. The default here spares allocation
   keepalives only; add co-tenant patterns for your own site.
6. **`fetch_awm_data.py` rewritten** to resolve its destination from `BRACE_ROOT` or
   `AWM_REPO` instead of one hard-coded directory. The dataset it downloads is unchanged.
7. **`heuristic_surface.py`'s `sys.path` insertion** is now relative to the file rather than
   absolute.
8. **`policy_multi.py`'s `HF_HOME` fallback** no longer names a site directory.
9. **Comments scrubbed.** Cluster job ids, node names, the cluster's own name, a former
   host's home directory, another group's project name and the "box A / box B" shorthand
   were replaced with neutral wording. No statement of fact was altered.
10. **`patches/verl_0.8.0_dapo_dynamic_sampling.patch`** is new. It is the diff between the
    stock `verl/trainer/ppo/ray_trainer.py` and the file the DAPO arms actually ran, which
    previously existed only as an edit inside the installed environment.
11. **`recipes/`** is new: `train_arm.sh`, `eval_arm.sh` and
    `make_tables_and_figures.sh` distil the per-row launch commands out of
    `slurm/supervisor.sh` and the recorded launch records, in the paper's own row names.

Several source comments cite an internal planning document by name. That document is a set
of notes rather than code and is not part of this release.
