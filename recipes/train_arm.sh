#!/bin/bash
# One training command per paper row.
#
#     recipes/train_arm.sh <row>
#
# `<row>` names a row of Table 1, Table 2 or one of the appendix tables, at one model
# scale. Every row expands to a single invocation of surface/verl_rl/coadapt.py, which
# is the cycle-restart driver: it prepares the task pool, launches one verl GRPO cycle
# at a time through slurm/verl_awm_train.sh, and re-enters the allocator between cycles.
#
# The internal arm tag each row corresponds to is printed before launch and listed in
# README.md. Tags are how the evaluation and analysis code names a run on disk; they are
# not paper identifiers.
#
# Required environment:
#     BRACE_ROOT   this checkout (see env.sh)
#     JOBID        the batch allocation that already holds the GPUs. The driver runs on
#                  a login/head node and dispatches each cycle into that allocation.
#     SHAPE        gpu profile: a100 | a100x2 | h200   (default chosen per scale)
#
# Optional:
#     SEED_OFFSET  override the registered seed offset (each arm has one; a second seed
#                  is a second row, not a flag)
#     EXCLUDE      path to the certified-dead task list (default $BRACE_WORK/analysis/
#                  excluded_tasks.jsonl, produced by triage.py's exclusion pass)
#     POOL         task pool json (default surface/gate_caller/pools/pool_max.json)
set -u
. "${BRACE_ROOT:?export BRACE_ROOT to point at this checkout}/env.sh"

ROW=${1:?usage: train_arm.sh <row>; see README.md for the row list}
JOBID=${JOBID:?set JOBID to the batch allocation holding the GPUs}
POOL=${POOL:-$BRACE_ROOT/surface/gate_caller/pools/pool_max.json}
EXCLUDE=${EXCLUDE:-$BRACE_WORK/analysis/excluded_tasks.jsonl}

# ---------------------------------------------------------------------------
# Shared protocol. Identical for every row in the paper: 15 cycles of 8 optimizer
# steps, k = 5 rollouts per group (NROLL), 32-episode batches in 8-episode
# mini-batches, 12288-token responses, a checkpoint every 5 steps, and a held-out
# validation pass every 5 steps against the fixed advertised surface.
# ---------------------------------------------------------------------------
COMMON="--gate none --cycles 15 --steps-per-cycle 8 --pool $POOL --handicap-frac 0.40 --edit-block 700"
export LR=${LR:-1e-4} ENTCOEF=${ENTCOEF:-0.0} VAL_BEFORE=${VAL_BEFORE:-True}
export BSZ=${BSZ:-32} MINIBSZ=${MINIBSZ:-8} NROLL=${NROLL:-5}
export KEEPCKPT=${KEEPCKPT:-12} MAXTOK=${MAXTOK:-28672} RESP_LEN=${RESP_LEN:-12288}
export VAL_FREQ=${VAL_FREQ:-5} SAVE_FREQ=${SAVE_FREQ:-5} AWM_SLOTS_TRAIN=${AWM_SLOTS_TRAIN:-8}
export AWM_VAL_ADVERTISED=${AWM_VAL_ADVERTISED:-$BRACE_WORK/coadapt_coadapt/advertised_init.txt}

# The full method's flag set, quoted once. TEMP is the sampling temperature of the
# allocation weights: 0.3 at 2B and 4B, 0.5 at 8B.
brace_flags () { echo "--task-alloc gradmass --warm-bank default --temp $1 --exclude-tasks $EXCLUDE --warm-shrink group"; }

SURF="--surface-fixed 0.5"      # the fixed advertised tool surface every arm trains against
MODEL_2B=Qwen/Qwen3-VL-2B-Instruct
MODEL_4B=Qwen/Qwen3-VL-4B-Instruct
MODEL_8B=Qwen/Qwen3-VL-8B-Instruct

case "$ROW" in
# ---- Table 1: the method -----------------------------------------------------------
brace-2b)        TAG=q2bT;    SCALE=2B; OFF=500;  RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
brace-2b-seed2)  TAG=q2bT2;   SCALE=2B; OFF=1500; RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
brace-2b-seed3)  TAG=q2bT3;   SCALE=2B; OFF=2500; RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
brace-4b)        TAG=q4bT;    SCALE=4B; OFF=500;  RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
brace-4b-seed2)  TAG=q4bT2;   SCALE=4B; OFF=1500; RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
brace-8b)        TAG=a8T3g;   SCALE=8B; OFF=500;  RW="$(brace_flags 0.5)"; ENV="TRIAGE_DECAY=1.0" ;;
brace-8b-seed2)  TAG=a8T3gr;  SCALE=8B; OFF=1500; RW="$(brace_flags 0.5)"; ENV="TRIAGE_DECAY=1.0" ;;
brace-8b-seed3)  TAG=a8T3g2;  SCALE=8B; OFF=2500; RW="$(brace_flags 0.5)"; ENV="TRIAGE_DECAY=1.0" ;;

# ---- Table 1: uniform GRPO control -------------------------------------------------
grpo-2b)         TAG=q2bF5;   SCALE=2B; OFF=500;  RW=""; ENV="" ;;
grpo-2b-seed2)   TAG=q2bF6;   SCALE=2B; OFF=1500; RW=""; ENV="" ;;
grpo-2b-seed3)   TAG=q2bF7;   SCALE=2B; OFF=2500; RW=""; ENV="" ;;
grpo-4b)         TAG=q4bF;    SCALE=4B; OFF=500;  RW=""; ENV="" ;;
grpo-4b-seed2)   TAG=q4bF2;   SCALE=4B; OFF=1500; RW=""; ENV="" ;;
grpo-8b)         TAG=a8F;     SCALE=8B; OFF=500;  RW=""; ENV="" ;;
grpo-8b-seed2)   TAG=a8Fr;    SCALE=8B; OFF=1500; RW=""; ENV="" ;;

# ---- Table 1: published baselines, reimplemented inside this allocator -------------
plr-2b)          TAG=q2bP;    SCALE=2B; OFF=500;  RW="--reweight variance"; ENV="" ;;
plr-4b)          TAG=q4bP;    SCALE=4B; OFF=500;  RW="--reweight variance"; ENV="" ;;
plr-8b)          TAG=b8plr;   SCALE=8B; OFF=500;  RW="--reweight variance"; ENV="" ;;
plr-8b-seed2)    TAG=b8plrr;  SCALE=8B; OFF=1500; RW="--reweight variance"; ENV="" ;;
ragmcp-2b)       TAG=q2bR;    SCALE=2B; OFF=500;  RW=""; ENV=""; SURF="--surface-retrieval --retrieval-match-level 0.5" ;;
ragmcp-4b)       TAG=q4bR;    SCALE=4B; OFF=500;  RW=""; ENV=""; SURF="--surface-retrieval --retrieval-match-level 0.5" ;;
ragmcp-8b)       TAG=b8ret;   SCALE=8B; OFF=500;  RW=""; ENV=""; SURF="--surface-retrieval --retrieval-match-level 0.5" ;;
dapo-2b)         TAG=q2bD;    SCALE=2B; OFF=500;  RW=""; ENV="DAPO_MAX_GEN_BATCHES=12 MINIBSZ=4 TRAIN_SCRIPT=$BRACE_ROOT/slurm/verl_awm_train_agent.sh" ;;
dapo-4b)         TAG=q4bD;    SCALE=4B; OFF=500;  RW=""; ENV="DAPO_MAX_GEN_BATCHES=12 MINIBSZ=4 TRAIN_SCRIPT=$BRACE_ROOT/slurm/verl_awm_train_agent.sh" ;;
dapo-8b)         TAG=dapo;    SCALE=8B; OFF=500;  RW=""; ENV="DAPO_MAX_GEN_BATCHES=12 MINIBSZ=4 TRAIN_SCRIPT=$BRACE_ROOT/slurm/verl_awm_train_agent.sh" ;;
tscl-2b)         TAG=q2bLp;   SCALE=2B; OFF=500;  RW="--task-alloc progress --warm-bank default"; ENV="" ;;
tscl-4b)         TAG=q4bLp;   SCALE=4B; OFF=500;  RW="--task-alloc progress --warm-bank default"; ENV="" ;;
tscl-8b)         TAG=a8Tlp;   SCALE=8B; OFF=500;  RW="--task-alloc progress --warm-bank default"; ENV="" ;;
vip-2b)          TAG=q2bVf;   SCALE=2B; OFF=500;  RW="--task-alloc vip"; ENV="" ;;
vip-4b)          TAG=q4bTvip; SCALE=4B; OFF=500;  RW="--task-alloc vip"; ENV="" ;;
vip-8b)          TAG=a8Tvipf; SCALE=8B; OFF=500;  RW="--task-alloc vip"; ENV="" ;;
trace-2b)        TAG=t2bTf;   SCALE=2B; OFF=500;  RW="--task-alloc gradmass --estimator point"; ENV="TRIAGE_DECAY=1.0" ;;
trace-4b)        TAG=q4bTf;   SCALE=4B; OFF=500;  RW="--task-alloc gradmass --estimator point"; ENV="TRIAGE_DECAY=1.0" ;;
trace-8b)        TAG=t8Tf;    SCALE=8B; OFF=500;  RW="--task-alloc gradmass --estimator point"; ENV="TRIAGE_DECAY=1.0" ;;

# ---- Table 2 and the configuration block -------------------------------------------
# per-scale rho: the single correlation constant is replaced by that scale's own refit
# (fit_task_rho.py). nu = (1 - rho) / rho, passed as TRIAGE_NU.
refit-2b)        TAG=q2bTr;   SCALE=2B; OFF=500;  RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05 TRIAGE_NU=0.4582" ;;
refit-4b)        TAG=q4bTr;   SCALE=4B; OFF=500;  RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05 TRIAGE_NU=0.3501" ;;
refit-4b-seed2)  TAG=q4bTr2;  SCALE=4B; OFF=1500; RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05 TRIAGE_NU=0.3501" ;;
refit-4b-seed3)  TAG=q4bTr3;  SCALE=4B; OFF=2500; RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05 TRIAGE_NU=0.3501" ;;
refit-8b)        TAG=a8Tr;    SCALE=8B; OFF=500;  RW="$(brace_flags 0.5)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_NU=0.3713" ;;
# allocation rule -> VIP's, warm bank kept
vipbank-2b)      TAG=q2bV;    SCALE=2B; OFF=500;  RW="--task-alloc vip --warm-bank default"; ENV="" ;;
vipbank-4b)      TAG=q4bV;    SCALE=4B; OFF=500;  RW="--task-alloc vip --warm-bank default"; ENV="" ;;
vipbank-4b-seed2) TAG=q4bV2;  SCALE=4B; OFF=1500; RW="--task-alloc vip --warm-bank default"; ENV="" ;;
vipbank-8b)      TAG=a8Tvip;  SCALE=8B; OFF=500;  RW="--task-alloc vip --warm-bank default"; ENV="RESUME_LOAD=model_extra" ;;
vipbank-8b-seed2) TAG=a8Tvipr; SCALE=8B; OFF=1500; RW="--task-alloc vip --warm-bank default"; ENV="" ;;
# estimator: posterior -> plug-in
plugin-2b)       TAG=q2bTpe;  SCALE=2B; OFF=500;  RW="$(brace_flags 0.3) --estimator point"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
plugin-4b)       TAG=q4bTpe;  SCALE=4B; OFF=500;  RW="$(brace_flags 0.3) --estimator point"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
plugin-4b-seed2) TAG=q4bTpe2; SCALE=4B; OFF=1500; RW="$(brace_flags 0.3) --estimator point"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
plugin-8b)       TAG=a8Tpe;   SCALE=8B; OFF=500;  RW="$(brace_flags 0.5) --estimator point"; ENV="TRIAGE_DECAY=1.0" ;;
# sharper sampling weights: temperature 0.3 where the 8B default is 0.5. At 2B and 4B
# the full method already runs at 0.3, so this row exists at 8B only.
sharp-8b)        TAG=a8T5k;   SCALE=8B; OFF=500;  RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
sharp-8b-seed2)  TAG=a8T5kr;  SCALE=8B; OFF=1500; RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
# w/o warm bank: the full method minus --warm-bank
nobank-2b)       TAG=q2bTnw;  SCALE=2B; OFF=500;  RW="--task-alloc gradmass --temp 0.3 --exclude-tasks $EXCLUDE --warm-shrink group"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
nobank-4b)       TAG=q4bTnw;  SCALE=4B; OFF=500;  RW="--task-alloc gradmass --temp 0.3 --exclude-tasks $EXCLUDE --warm-shrink group"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
nobank-8b)       TAG=a8Tnw;   SCALE=8B; OFF=2500; RW="--task-alloc gradmass --temp 0.5 --exclude-tasks $EXCLUDE --warm-shrink group"; ENV="TRIAGE_DECAY=1.0" ;;
# w/o shrinkage calibration: --warm-shrink group -> none
noshrink-2b)     TAG=q2bTiid; SCALE=2B; OFF=500;  RW="--task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $EXCLUDE --warm-shrink none"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
noshrink-4b)     TAG=q4bTiid; SCALE=4B; OFF=1500; RW="--task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $EXCLUDE --warm-shrink none"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
noshrink-8b)     TAG=a8Tiid;  SCALE=8B; OFF=2500; RW="--task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $EXCLUDE --warm-shrink none"; ENV="TRIAGE_DECAY=1.0" ;;

# ---- Appendix: learning-rate ladder at 2B ------------------------------------------
lr-grpo-1e5-2b)  TAG=q2bF2;   SCALE=2B; OFF=500;  RW=""; ENV="LR=1e-5" ;;
lr-grpo-3e5-2b)  TAG=q2bF3e5; SCALE=2B; OFF=500;  RW=""; ENV="LR=3e-5" ;;
lr-brace-3e5-2b) TAG=q2bT3e;  SCALE=2B; OFF=500;  RW="$(brace_flags 0.3)"; ENV="LR=3e-5 TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;

# ---- Appendix: pre-registered reruns at the parent's own seed offset ----------------
rerun-vipbank-2b) TAG=q2bV2;  SCALE=2B; OFF=500;  RW="--task-alloc vip --warm-bank default"; ENV="" ;;
rerun-tscl-2b)    TAG=q2bLp2; SCALE=2B; OFF=500;  RW="--task-alloc progress --warm-bank default"; ENV="" ;;
rerun-tscl-4b)    TAG=q4bLp2; SCALE=4B; OFF=500;  RW="--task-alloc progress --warm-bank default"; ENV="" ;;

# ---- Durability: post-window continuations -----------------------------------------
# Two protocols. "merge" restarts from the parent's merged step-30 weights, so the
# continuation is a fresh optimizer over a trained policy; "resume"/"straight" continue
# the parent run's own trajectory. MERGED_PARENT must point at a merged HF directory
# produced by surface/verl_rl/merge_lora.py.
dur-brace-2b)    TAG=q2bTc;   SCALE=2B; OFF=500;  RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05"; MERGE_FROM=q2bT ;;
dur-grpo-2b)     TAG=q2bF6L;  SCALE=2B; OFF=1500; RW=""; ENV="" ;;
dur-plr-2b)      TAG=q2bPc;   SCALE=2B; OFF=500;  RW="--reweight variance"; ENV="" ;;
dur-brace-4b)    TAG=q4bTr3c; SCALE=4B; OFF=2500; RW="$(brace_flags 0.3)"; ENV="TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05 TRIAGE_NU=0.3501" ;;
dur-grpo-4b)     TAG=q4bFc;   SCALE=4B; OFF=500;  RW=""; ENV="" ;;
dur-plr-4b)      TAG=q4bPc;   SCALE=4B; OFF=500;  RW="--reweight variance"; ENV="" ;;
dur-brace-8b)    TAG=a8T3g2c; SCALE=8B; OFF=2500; RW="$(brace_flags 0.5)"; ENV="TRIAGE_DECAY=1.0"; MERGE_FROM=a8T3g2 ;;
dur-grpo-8b)     TAG=a8Fc;    SCALE=8B; OFF=500;  RW=""; ENV=""; MERGE_FROM=a8F ;;

*) echo "unknown row: $ROW"; echo "see README.md for the full row list"; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Scale defaults: base model, gpu count, tensor parallelism, offload, serving share.
# SHAPE picks the hardware profile. 2B and 4B fit one 80GB card; 4B needs FSDP
# parameter and optimizer offload to do so. 8B needs either a single 141GB card or
# both cards of a two-card node at TP=2 with offload. 46GB cards cannot train any of
# these recipes and are evaluation-only.
# ---------------------------------------------------------------------------
case "$SCALE" in
  2B) : "${MODEL:=$MODEL_2B}"; SHAPE=${SHAPE:-a100} ;;
  4B) : "${MODEL:=$MODEL_4B}"; SHAPE=${SHAPE:-a100} ;;
  8B) : "${MODEL:=$MODEL_8B}"; SHAPE=${SHAPE:-a100x2} ;;
esac
case "$SCALE:$SHAPE" in
  2B:a100)   export NGPUS=1 TP=1 GPU_UTIL=${GPU_UTIL:-0.35} ;;
  2B:h200)   export NGPUS=1 TP=1 GPU_UTIL=${GPU_UTIL:-0.35} ;;
  4B:a100)   export NGPUS=1 TP=1 GPU_UTIL=${GPU_UTIL:-0.30} PARAM_OFFLOAD=True OPT_OFFLOAD=True ;;
  4B:a100x2) export NGPUS=2 TP=2 GPU_UTIL=${GPU_UTIL:-0.30} PARAM_OFFLOAD=True OPT_OFFLOAD=True ;;
  4B:h200)   export NGPUS=1 TP=1 GPU_UTIL=${GPU_UTIL:-0.35} ;;
  8B:a100x2) export NGPUS=2 TP=2 GPU_UTIL=${GPU_UTIL:-0.29} PARAM_OFFLOAD=True OPT_OFFLOAD=True ;;
  8B:h200)   export NGPUS=1 TP=1 GPU_UTIL=${GPU_UTIL:-0.27} ;;
  *) echo "unsupported scale/shape combination: $SCALE on $SHAPE"; exit 2 ;;
esac
export LORA_RANK=${LORA_RANK:-32} LORA_ALPHA=${LORA_ALPHA:-32}

# A durability continuation starts from its parent's merged step-30 weights.
if [ -n "${MERGE_FROM:-}" ]; then
  MODEL=${MERGED_PARENT:-$BRACE_WORK/merged_${MERGE_FROM}_step30}
  [ -f "$MODEL/config.json" ] || {
    echo "durability row $ROW needs the merged parent at $MODEL"
    echo "build it with: \$VERL_PY $BRACE_ROOT/surface/verl_rl/merge_lora.py \\"
    echo "    --ckpt $BRACE_WORK/verl/ckpt_${MERGE_FROM}/global_step_30/actor --base <base model> --out $MODEL"
    exit 1; }
fi
export MODEL
# shellcheck disable=SC2086
[ -n "$ENV" ] && export $ENV

echo "[recipe] row=$ROW  tag=$TAG  scale=$SCALE  shape=$SHAPE  model=$MODEL  seed-offset=${SEED_OFFSET:-$OFF}"
# shellcheck disable=SC2086
exec "$VERL_PY" "$BRACE_ROOT/surface/verl_rl/coadapt.py" \
  --tag "$TAG" --jobid "$JOBID" $SURF $RW \
  --seed-offset "${SEED_OFFSET:-$OFF}" $COMMON
