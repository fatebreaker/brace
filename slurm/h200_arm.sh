#!/bin/bash
# One H200 (143GB) hosts batch 16 at rank 64 without tensor parallelism, which removes the
# sharding and host-memory failures that dogged every A100 attempt.
set -u
R=${BRACE_ROOT:?BRACE_ROOT is unset; source env.sh first}
WS=${BRACE_WORK:?BRACE_WORK is unset; source env.sh first}
TAG=$1
# Checkpoints are 34-68G each and /project hit 99% with five arms running, which kills every
# process on the filesystem. They belong on scratch; only the small episode logs stay on /project.
# The PATH stays under /project so every driver and resume keeps working unchanged; the BYTES
# live on scratch behind a symlink. Changing the path itself would desynchronise a running
# driver -- it holds the old path in memory and would wait forever for progress it cannot see.
CKPT_BASE=${CKPT_BASE:-$BRACE_WORK/ckpt}
export DATA=$R/work/verl/awm_$TAG EXP=awm_${TAG}_grpo CKPT=$R/work/verl/ckpt_$TAG
if [ ! -e "$CKPT" ]; then mkdir -p "$CKPT_BASE/ckpt_$TAG"; ln -s "$CKPT_BASE/ckpt_$TAG" "$CKPT"; fi
export AWM_RL_RUNDIR=$R/work/verl/run_$TAG RAY_TMPDIR=/tmp/r_$TAG HF_HOME=$R/hf_cache
# vLLM sizes its KV cache as (total * GPU_UTIL) minus whatever is already resident, and the FSDP
# trainer is already resident by then. At 0.32 of a 143GB card that margin is thin enough that
# arms fail or survive on timing alone -- override upward for an arm that cannot get through boot.
# AWM_RL_MAX_SLOTS=1 serialised every rollout: one MCP server (~336 MiB) at a time while the arm
# holds a whole GPU. That was the real throughput ceiling on every arm, not the GPU. Six slots
# cost ~2 GB against a 128 G pod and let episodes overlap.
export STEPS=${STEPS:-150} GPU_UTIL=${GPU_UTIL:-0.32} AWM_RL_MAX_SLOTS=${AWM_SLOTS_TRAIN:-1}
mkdir -p "$AWM_RL_RUNDIR" "$RAY_TMPDIR" "$WS/hydra"
cd $WS/hydra
echo "[$TAG] node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) batch=16 lora=64"
# Ray auto-detects the whole node's CPUs while this arm holds only its 8-CPU slice; four arms
# each starting a Ray instance that thinks it owns 96 cores thrashes hard enough to hang startup
# outright (coadapt2 sat in Ray init for 29 minutes). Bound each instance to what it was given.
# TRAIN_SCRIPT lets one arm use the DAPO-patched wrapper without touching the script the other
# seven arms share. Default is unchanged, so this edit is a no-op for every existing arm.
bash ${TRAIN_SCRIPT:-$R/slurm/verl_awm_train.sh} ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-8} trainer.save_freq=${SAVE_FREQ:-25} trainer.max_actor_ckpt_to_keep=${KEEPCKPT:-1} \
  trainer.resume_mode=auto ${RESUME_LOAD:+actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]} actor_rollout_ref.model.lora_rank=${LORA_RANK:-64} actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-64} \
  data.train_batch_size=${BSZ:-8} actor_rollout_ref.actor.ppo_mini_batch_size=${MINIBSZ:-2}
