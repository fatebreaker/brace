#!/bin/bash
# GRPO over the AWM/MCP environment through verl, Qwen3-VL-8B + LoRA, ONE H200.
#
# Sizing is measured, not guessed (surface/verl_rl/prep_awm.py):
#   turn-0 prompt = system + task + the scenario's RAW tool surface
#   max 14,917 tokens (marketplace_10, 36 tools), p50 9,608, min 7,692
# so prompt_length must be >= 16k. The agent loop's apply_chat_template raises
# rather than truncating when a prompt exceeds rollout.prompt_length.
set -u
. "${BRACE_ROOT:?BRACE_ROOT is unset; export it to point at this checkout}/env.sh"
E=$BRACE_ENVS/mcp_verl
export PATH=$E/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export VERL_LOGGING_LEVEL=INFO
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_$USER}
mkdir -p "$RAY_TMPDIR"

# Caps MUST match the screening protocol; awm_agent_loop.py refuses to import otherwise.
export GATE_MAX_TURNS=20
export GATE_MAX_NEW_TOKENS=1024
export CALLER_FAMILY=${CALLER_FAMILY:-qwen}
export AWM_RL_MAX_SLOTS=${AWM_RL_MAX_SLOTS:-6}
export AWM_RL_RUNDIR=${AWM_RL_RUNDIR:-$BRACE_ROOT/work/verl/awm_run}
mkdir -p "$AWM_RL_RUNDIR"

VERL_RL=$BRACE_ROOT/surface/verl_rl
export PYTHONPATH=$VERL_RL:${PYTHONPATH:-}
# TRIAGE v4: arm the sitecustomize hook ONLY inside the trainer. VARK alone must not arm it,
# or every eval/prep subprocess that happens to carry the flag would import and patch verl
# for nothing. Forwarded into the Ray runtime_env below, because fit() runs in a Ray actor.
export VARK_AUTOPATCH="${VARK:-0}"

MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
DATA=${DATA:-$BRACE_ROOT/work/verl/awm}
STEPS=${STEPS:-20}
PROMPT_LEN=${PROMPT_LEN:-16384}
RESP_LEN=${RESP_LEN:-16384}
BSZ=${BSZ:-4}
# Prompts per optimizer update. Kept SMALL and independent of BSZ so a large train_batch_size is
# absorbed by gradient accumulation rather than peak memory: same math, bounded activation cost.
# This matters because train_batch_size=4 was the binding constraint on learning -- at 7.5-20%
# gradient availability, a batch of 4 yields 0.3-0.8 non-degenerate GRPO groups per step, so most
# steps had NO usable gradient at all and training reward stayed flat (+0.002 over 61 steps).
MINIBSZ=${MINIBSZ:-$BSZ}
NROLL=${NROLL:-5}
GPU_UTIL=${GPU_UTIL:-0.35}

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Optional weight-sync bucket override, EMPTY unless WBUCKET is set. verl transfers trained
# weights into the rollout engine in buckets of update_weights_bucket_megabytes (default 2048),
# and gates dies inside exactly that call every boot. Passed conditionally rather than always:
# an always-on key with a wrong Hydra path is what took down all six arms via filter_groups.
# Checkpoint retention. verl keeps only the most recent few, and each is 26G; step 55 of atsc was
# already gone-then-partial when the curve tried to merge it. The learning curve needs a point to
# survive long enough for an eval card to reach it -- at SAVE_FREQ=5, keeping 12 preserves ~60
# steps of history (~10h at observed throughput), which comfortably covers eval latency. Passed
# conditionally: an unknown trainer key would abort every arm, as filter_groups once did.
KEEP_ARG=""
[ -n "${KEEPCKPT:-}" ] && KEEP_ARG="trainer.max_actor_ckpt_to_keep=$KEEPCKPT"

WB_ARG=""
[ -n "${WBUCKET:-}" ] && \
  WB_ARG="actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=$WBUCKET"

# Ray workers must see the caps and our PYTHONPATH. Relying on environment
# inheritance through the raylet is not safe; pass them explicitly.

# ---------------------------------------------------------------- validation curve (opt-in)
# VAL_FREQ=<steps> turns on an in-training validation pass on a HELD-OUT pool, scored on the
# SAME fixed surface for every arm (AWM_VAL_ADVERTISED). Comparability is the whole point: arms
# train on different surfaces -- 0.0 for b8van, ~0.47 for the ELSA arms, 0.5 for the reward-axis
# arms -- so validating each on its own surface would rank them by advertised tool count rather
# than by policy. The pool is pool_coadapt_heldout (295 tasks, 0% overlap with the training
# pool), which is what the eval cells already score, so the in-training curve and the held-out
# table measure the same quantity.
#
# val_kwargs.temperature=0 does double duty: greedy decoding makes the curve reproducible, and
# it is how awm_agent_loop identifies a validation rollout -- verl does not forward its own
# `validate` flag into the agent loop's kwargs (confirmed in the installed wheel:
# agent_loop.py::_run_agent_loop calls `agent_loop.run(sampling_params, **kwargs)` and keeps
# `validate` for its own trace attrs), and patching installed verl for one flag is a worse trade
# than reading the sampling params it already passes.
#
# POSITION IS LOAD-BEARING. Hydra applies command-line overrides LEFT TO RIGHT and the last one
# for a key wins -- measured, not assumed. This block used to sit immediately after main_ppo,
# ahead of the baseline `data.val_files=$DATA/test.parquet` and `trainer.test_freq=-1` below, so
# every one of its keys was overwritten by the very defaults it exists to replace: test_freq
# came out as -1 and validation never ran once. It must stay AFTER those defaults and before
# "$@", which is where h200_arm.sh's own overrides land.
#
# AWM_VAL_ADVERTISED is pushed into the Ray runtime env HERE rather than in the agent wrapper.
# The agent loop reads it inside Ray workers, and this file forwards a fixed list of variables
# precisely because "relying on environment inheritance through the raylet is not safe" -- a
# shell export reaches the driver and nothing else. It lived in verl_awm_train_agent.sh, but
# only the four arms that set TRAIN_SCRIPT run through that wrapper; b8k16v does not, so its
# workers would have seen no variable at all, validated on the ARM'S OWN surface, and produced a
# curve that ranks arms by advertised tool count. Forwarded from the shared script it holds for
# every arm. Do NOT also forward it from the wrapper: Hydra's `+` refuses a key that already
# exists ("Could not append to config. An item is already at ..."), which is a hard abort.
VAL_ARGS=()
if [ -n "${VAL_FREQ:-}" ]; then
  VAL_ARGS=(
    "data.val_files=${VAL_FILES:-$BRACE_ROOT/work/verl/val_heldout/test.parquet}"
    "trainer.test_freq=${VAL_FREQ}"
    "trainer.val_before_train=${VAL_BEFORE:-False}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.0"
    "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=False"
  )
fi
if [ -n "${AWM_VAL_ADVERTISED:-}" ]; then
  VAL_ARGS+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.AWM_VAL_ADVERTISED='${AWM_VAL_ADVERTISED}'"
  )
fi

# ---------------------------------------------------------------- presentation decorrelation
# TRIAGE_DECOR=shuffle permutes, per rollout, the ORDER the advertised tools are listed in the
# turn-0 prompt, so the k rollouts of a GRPO group see different-but-equivalent presentations of
# the same task. awm_agent_loop.py reads it INSIDE the Ray workers, which is exactly why it is
# forwarded here rather than merely exported: this file forwards a fixed list of variables
# because, in its own words, "relying on environment inheritance through the raylet is not safe",
# and a missed forward would leave the workers on the unmodified presentation while the arm's
# name, its log and its registration all said decorrelation -- the silent-no-op failure that made
# the retired gatesd arm meaningless and that took a night off a8C.
#
# TRIAGE_DECOR_K carries the GROUP SIZE from the one place that sets it: $NROLL is what goes into
# actor_rollout_ref.rollout.n below, so the permutation family is sized to the group the trainer
# actually generates. An arm that raised NROLL and decorrelated at k=5 would hand two of its
# rollouts the same presentation and under-measure its own mechanism.
#
# The whole block is conditional and the array expands to NOTHING when unset, so the composed
# Hydra command for every arm in flight is byte-identical to what it was before this block
# existed (verified: 63 tags x 2 GPU counts, 0 differ).
DECOR_ARGS=()
if [ -n "${TRIAGE_DECOR:-}" ]; then
  DECOR_ARGS=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRIAGE_DECOR='${TRIAGE_DECOR}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRIAGE_DECOR_K='${NROLL}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRIAGE_DECOR_SEED='${TRIAGE_DECOR_SEED:-0}'"
  )
fi

exec $E/bin/python -m verl.trainer.main_ppo ${WB_ARG:+$WB_ARG} ${KEEP_ARG:+$KEEP_ARG} \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=$DATA/train.parquet \
  data.val_files=$DATA/test.parquet \
  data.train_batch_size=$BSZ \
  data.max_prompt_length=$PROMPT_LEN \
  data.max_response_length=$RESP_LEN \
  data.filter_overlong_prompts=False \
  data.truncation=right \
  data.shuffle=$([ "${VARK:-0}" = "1" ] && echo False || echo True) \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=${LR:-1e-5} \
  actor_rollout_ref.actor.entropy_coeff=${ENTCOEF:-0.0} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINIBSZ \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES:-1} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAXTOK:-$((PROMPT_LEN+RESP_LEN))} \
  actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-False} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPT_OFFLOAD:-False} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP:-1} \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
  actor_rollout_ref.rollout.enforce_eager=${EAGER:-False} \
  actor_rollout_ref.rollout.max_model_len=$((PROMPT_LEN+RESP_LEN)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((PROMPT_LEN+RESP_LEN)) \
  actor_rollout_ref.rollout.n=$NROLL \
  actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMP:-0.7} \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((PROMPT_LEN+RESP_LEN)) \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$GATE_MAX_TURNS \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.agent.default_agent_loop=awm_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=$VERL_RL/awm_agent.yaml \
  +ray_kwargs.ray_init.runtime_env.env_vars.GATE_MAX_TURNS="'$GATE_MAX_TURNS'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.GATE_MAX_NEW_TOKENS="'$GATE_MAX_NEW_TOKENS'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.CALLER_FAMILY="'$CALLER_FAMILY'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.AWM_PY="'$AWM_PY'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.MCP_ROOT="'$MCP_ROOT'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.AWM_RL_RUNDIR="'$AWM_RL_RUNDIR'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.AWM_RL_SRVDIR="'${AWM_RL_SRVDIR:-$AWM_RL_RUNDIR}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.AWM_RL_MAX_SLOTS="'$AWM_RL_MAX_SLOTS'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="'$PYTHONPATH'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VARK="'${VARK:-0}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VARK_AUTOPATCH="'${VARK:-0}'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.BSZ="'$BSZ'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NROLL="'$NROLL'" \
  trainer.n_gpus_per_node=${NGPUS:-1} \
  trainer.nnodes=1 \
  trainer.total_epochs=100 \
  trainer.total_training_steps=$STEPS \
  trainer.val_before_train=False \
  trainer.logger=[console] \
  trainer.project_name=verl_awm \
  trainer.experiment_name=${EXP:-awm_smoke20_grpo} \
  trainer.default_local_dir=${CKPT:-$BRACE_ROOT/work/verl/ckpt_awm} \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  ${VAL_ARGS[@]+"${VAL_ARGS[@]}"} \
  ${DECOR_ARGS[@]+"${DECOR_ARGS[@]}"} \
  "$@"
