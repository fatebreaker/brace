#!/bin/bash
# One evaluation command per kind of paper cell.
#
#     recipes/eval_arm.sh window      <tag>                 steps 15 20 25 30 (the window)
#     recipes/eval_arm.sh durability  <tag>                 steps 40 50 60
#     recipes/eval_arm.sh anchor      <tag>                 the step-0 base-policy anchor
#     recipes/eval_arm.sh bfcl        <tag> <step> [scale]  transfer: BFCL v4 multi-turn
#     recipes/eval_arm.sh nestful     <tag> <step> [scale]  transfer: NESTFUL
#
# `<tag>` is the internal arm tag printed by recipes/train_arm.sh; README.md maps every
# paper row to its tag. Window and durability cells go through slurm/coadapt_eval.sh,
# which merges the LoRA adapter for the requested step, serves it, replays the held-out
# task pool on the FIXED initial tool surface and writes one binary outcome per
# (task, seed) into $BRACE_WORK/coadapt_eval_<tag>/cell_STEP<n>.jsonl. That file is what
# main_table.load_cell reads, so every table in the paper is downstream of it.
#
# Required environment: BRACE_ROOT, BRACE_WORK, and JOBID (an allocation holding one GPU).
set -u
. "${BRACE_ROOT:?export BRACE_ROOT to point at this checkout}/env.sh"

MODE=${1:?usage: eval_arm.sh <window|durability|anchor|bfcl|nestful> <tag> [step] [scale]}
TAG=${2:?second argument is the arm tag}
JOBID=${JOBID:?set JOBID to the batch allocation holding a GPU}

# The scale is inferred from the tag prefix the way coadapt_eval.sh infers it.
case "$TAG" in q2b*|t2b*) SCALE=2B ;; q4b*) SCALE=4B ;; *) SCALE=8B ;; esac
SCALE=${4:-$SCALE}
export MODEL_FAMILY=$SCALE

run_cell () {   # step
  srun --jobid="$JOBID" --ntasks=1 --overlap --gres=gpu:1 --cpus-per-task=8 --mem=88G \
    bash -c "cd /tmp && SEEDS=${SEEDS:-2} AWM_SLOTS=${AWM_SLOTS:-4} \
             EVAL_MEMFRAC=${EVAL_MEMFRAC:-0.30} MODEL_FAMILY=$SCALE \
             bash '$BRACE_ROOT/slurm/coadapt_eval.sh' STEP$1 $TAG"
}

case "$MODE" in
  window)      for s in ${STEPS:-15 20 25 30}; do run_cell "$s"; done ;;
  durability)  for s in ${STEPS:-40 50 60}; do run_cell "$s"; done ;;
  anchor)      run_cell 0 ;;
  bfcl|nestful)
    STEP=${3:?third argument is the checkpoint step}
    W=$BRACE_WORK/$MODE
    CKPT=$BRACE_WORK/verl/ckpt_$TAG/global_step_$STEP/actor
    BASE=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3-VL-$SCALE-Instruct/snapshots/* 2>/dev/null | head -1)
    [ -n "$BASE" ] || { echo "no base snapshot for $SCALE under $HF_HOME/hub"; exit 1; }
    if [ "$MODE" = bfcl ]; then
      # BFCL runs out of its own venv (bfcl_eval pins numpy 1.26.4); vLLM is still
      # launched as a separate process out of the vLLM environment.
      PY=${BFCL_PY:?set BFCL_PY to the bfcl_venv interpreter, see README.md}
      EXTRA="--bfcl-repo ${BFCL_REPO:-$W/gorilla/berkeley-function-call-leaderboard} --bfcl-python $PY"
      SCRIPT_PY=$BRACE_ROOT/surface/verl_rl/bench_bfcl.py
    else
      PY=$VLLM_PY
      EXTRA="--nestful-repo ${NESTFUL_REPO:-$W/nestful_repo}"
      SCRIPT_PY=$BRACE_ROOT/surface/verl_rl/bench_nestful.py
    fi
    # shellcheck disable=SC2086
    srun --jobid="$JOBID" --ntasks=1 --overlap --gres=gpu:1 --cpus-per-task=8 --mem=120G \
      bash -c ". '$BRACE_ROOT/env.sh'; '$PY' '$SCRIPT_PY' \
          --arm $TAG --step $STEP --ckpt '$CKPT' --base '$BASE' \
          --merged-out '$W/merged_${TAG}_step${STEP}' --out-dir '$W/run_$TAG' $EXTRA" ;;
  *) echo "unknown mode: $MODE"; exit 2 ;;
esac
