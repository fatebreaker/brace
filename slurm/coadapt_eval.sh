#!/bin/bash
# One cell of the co-adaptation 2x2: a policy crossed with a harness, on the held-out pool.
#   A = base policy    x initial (handicapped) surface      C = trained policy x initial surface
#   B = base policy    x certified (final) surface          D = trained policy x final surface
# Usage: coadapt_eval.sh <A|B|C|D> [TAG]
set -u
R=${BRACE_ROOT:?BRACE_ROOT is unset; source env.sh first}
CELL=$1; TAG=${2:-coadapt}
# 2B/4B arms merge onto their own base; the supervisor path never sets MODEL_FAMILY,
# so infer it from the tag (q2bN step-20 merged onto the 8B base: 300/300 shape mismatches).
case "$TAG" in q2b*|t2b*) MODEL_FAMILY=${MODEL_FAMILY:-2B};; q4b*|t4b*) MODEL_FAMILY=${MODEL_FAMILY:-4B};; esac
SMAP=""
EV=$BRACE_ENVS/mcp_verl
ES=$BRACE_ENVS/mcp_vllm
A=$R/work/coadapt_$TAG
OUT=$R/work/coadapt_eval_$TAG
mkdir -p $OUT
export HF_HOME=$R/hf_cache

POOL=$R/surface/gate_caller/pools/pool_coadapt_heldout.json
case $CELL in
  A|C) SUB=$A/advertised_init.txt ;;
  B|D) SUB=$A/advertised.txt ;;
  # D2 pins the SECOND certified surface as its own snapshot. Re-running D after the gate moved
  # would append 4087-surface episodes into the 3387-surface file and silently blend two
  # different conditions into one cell.
  D2)  SUB=$A/advertised_c2.txt ;;
  B2)  SUB=$A/advertised_c2.txt ;;   # base policy at the SECOND certified surface
  # D3: the co-adapted policy on the FULLY restored surface the gate reached (4478/4478).
  # This is the method's end state, and the dose curve predicts +0.1356 held-out for it.
  D3)  SUB=$A/advertised_c3.txt ;;
  # SELF: this arm's OWN policy on this arm's OWN certified surface -- the arm-level outcome the
  # head-to-head table needs measured rather than priced off the dose curve. SELFBASE is its
  # control: the base policy on that same surface, isolating harness from policy per arm.
  # STEP<N>: the learning curve. Checkpoint N evaluated on the FIXED initial surface, so the
  # only thing varying along the curve is the policy. Evaluating each checkpoint on its own
  # current surface would move harness and policy together -- ATSC rewrites the surface every
  # cycle -- and the 2x2 already showed the harness channel is the larger of the two, so that
  # curve would mostly measure the harness and be read as learning.
  # PROBE: base capability of an ARBITRARY model on the fixed initial surface. Used to decide
  # whether a smaller model can do this task at all before spending cards training it -- a model
  # that solves ~0% makes every task p=0, which is strictly worse than the 8B we already have.
  # SELFSTEP<N>: checkpoint N scored IN-DISTRIBUTION, on the arm's own per-scenario surface.
  # STEP<N> pins the initial 2687-name surface for cross-checkpoint comparability, but an arm
  # training at level 0.5 advertises ~4380 names, so STEP measures transfer to a harder surface,
  # not learning. Both are wanted: STEP for the curve, SELFSTEP for "did this policy improve".
  SELFSTEP*) SUB=$A/advertised_init.txt; SMAP=$A/surface_map.json ;;
  # SELFMAPBASE: the BASE policy on that same per-scenario surface -- the control SELFSTEP needs.
  # Cell A cannot serve here: it scores base on the flat 2687 list, a different surface entirely.
  SELFMAPBASE) SUB=$A/advertised_init.txt; SMAP=$A/surface_map.json ;;
  PROBE*)   SUB=$A/advertised_init.txt ;;
  STEP*)    SUB=$A/advertised_init.txt ;;
  SELF)     SUB=$A/advertised.txt ;;
  SELFBASE) SUB=$A/advertised.txt ;;
  # CEIL is the premise check: the base policy on the FULL surface. The gap against cell A is
  # the entire headroom any harness edit could ever recover, measured before a single training
  # step is spent -- if it is flat, no gate can make co-adaptation work.
  CEIL) SUB=$A/advertised_full.txt ;;
  # HEUR: a keyword heuristic picks the SAME NUMBER of tools as the handicap (2687/4478), so the
  # comparison isolates WHICH tools are advertised, not how many. If this recovers most of the
  # ceiling, our learned control is not the contribution.
  HEUR) SUB=$A/heuristic_flat.txt ;;
  # third handicap draw on the held-out pool: the ceiling should not depend on which 40% of the
  # tool universe we happened to withhold
  HELDOUT_S999) SUB=$A/heldout_init_s999.txt ;;
  # NOVEL_* asks the harder question on 60 scenarios that share NO tools with the gate pool:
  # does the harness channel move reward on scenarios never seen at all? A certified name list
  # cannot transfer there, but the RULE that produced it can, and a policy improvement provably
  # did not. This is the contrast that separates the two channels.
  NOVEL_INIT) SUB=$A/novel_init.txt
              POOL=$R/surface/gate_caller/pools/pool_novel120.json ;;
  NOVEL_FULL) SUB=$A/novel_full.txt
              POOL=$R/surface/gate_caller/pools/pool_novel120.json ;;
  # a second, near-independent draw of the withheld set (only 339 of 792 names shared with
  # seed 555): guards against the headroom being an artifact of one unlucky handicap.
  # Dose curve: the gate restores names in blocks of 700, so these two points say what a first
  # and a second accepted edit are worth on held-out tasks -- turning a count of restored names
  # into a predicted gain, instead of leaving the gate's output uninterpretable.
  DOSE700)  SUB=$A/dose700.txt ;;
  DOSE1400) SUB=$A/dose1400.txt ;;
  # the same dose points on scenarios never seen: does partial restoration transfer in
  # proportion, or does the benefit only appear once a scenario's own tools are back?
  NOVEL_DOSE39) SUB=$A/novel_dose39.txt
                POOL=$R/surface/gate_caller/pools/pool_novel120.json ;;
  NOVEL_DOSE60) SUB=$A/novel_dose60.txt
                POOL=$R/surface/gate_caller/pools/pool_novel120.json ;;
  NOVEL_DOSE78) SUB=$A/novel_dose78.txt
                POOL=$R/surface/gate_caller/pools/pool_novel120.json ;;
  # The policy-channel control on unseen scenarios: OUR co-adapted policy against the same novel
  # surface the base policy already faced. A certified tool list cannot transfer to a disjoint
  # tool universe, so if the harness channel is what carries the method, this must come back flat.
  NOVEL_TRAINED) SUB=$A/novel_init.txt
                 POOL=$R/surface/gate_caller/pools/pool_novel120.json ;;
  NOVEL_S777) SUB=$A/novel_init_s777.txt
              POOL=$R/surface/gate_caller/pools/pool_novel120.json ;;
  *)   echo "cell must be A, B, C, D, CEIL, NOVEL_INIT or NOVEL_FULL"; exit 2 ;;
esac
[ -s "$SUB" ] || { echo "[eval] missing surface $SUB"; exit 1; }

case $CELL in
  PROBE*) MODEL=${BASE_MODEL:?PROBE cells require BASE_MODEL} ;;
  SELFMAPBASE) MODEL=${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct} ;;
  A|B|B2|SELFBASE|CEIL|HEUR|NOVEL_INIT|NOVEL_FULL|NOVEL_S777|DOSE700|DOSE1400|HELDOUT_S999|NOVEL_DOSE39|NOVEL_DOSE60|NOVEL_DOSE78) MODEL=Qwen/Qwen3-VL-8B-Instruct ;;
  SELFSTEP*)
    CK=${CELL#SELFSTEP}
    [ -s "$SMAP" ] || { echo "[eval] no surface_map.json for $TAG; cannot score in-distribution"; exit 1; }
    [ -f $R/work/verl/ckpt_$TAG/global_step_$CK/actor/huggingface/config.json ] || {
      echo "[eval] checkpoint $CK of $TAG is incomplete; skipping"; exit 1; }
    MODEL=$BRACE_WORK/big/merged_${TAG}_step${CK}
    if [ ! -f $MODEL/config.json ]; then
      TMPM=$MODEL.tmp.$$; rm -rf $TMPM
      BASE_SNAP=$(ls -d $R/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/* 2>/dev/null | head -1)
      case "${MODEL_FAMILY:-8B}" in 2B) BASE_SNAP=$(ls -d $R/hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/* 2>/dev/null | head -1);; 4B) BASE_SNAP=$(ls -d $R/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/* 2>/dev/null | head -1);; esac
      # 2026-08-30 11:30: a MERGE-protocol child (its MODEL= is a merged parent) must be merged onto THAT base, never the plain family base.
      [ -n "${MERGE_BASE:-}" ] && [ -f "$MERGE_BASE/config.json" ] && { BASE_SNAP=$MERGE_BASE; echo "[eval] merge base overridden: $BASE_SNAP"; }
      $EV/bin/python $R/surface/verl_rl/merge_lora.py \
        --ckpt $R/work/verl/ckpt_$TAG/global_step_$CK/actor --base "$BASE_SNAP" --out $TMPM || { rm -rf $TMPM; exit 1; }
      if [ -f $MODEL/config.json ]; then rm -rf $TMPM; else mv -T $TMPM $MODEL 2>/dev/null || rm -rf $TMPM; fi
      [ -f $MODEL/config.json ] || { echo "[eval] merge failed for $TAG step $CK"; exit 1; }
    fi
    echo "[eval] IN-DISTRIBUTION: $TAG step $CK on its own per-scenario surface" ;;
  STEP*)
    CK=${CELL#STEP}
    [ -d $R/work/verl/ckpt_$TAG/global_step_$CK ] || { echo "[eval] no checkpoint $CK for $TAG"; exit 1; }
    # A restart landing mid-save leaves actor/ with an empty huggingface/ and no shards. Merging
    # that dies deep inside transformers with a confusing config error, after the card is already
    # committed; check it here instead so the cell fails immediately and the queue moves on.
    [ -f $R/work/verl/ckpt_$TAG/global_step_$CK/actor/huggingface/config.json ] || {
      echo "[eval] checkpoint $CK of $TAG is incomplete (no HF config); skipping"; exit 1; }
    # Merged models are ~17G each and a curve needs one per point. /project is at 99%, so these
    # go to scratch and are deleted after the cell finishes -- see CLEANUP below.
    MODEL=$BRACE_WORK/big/merged_${TAG}_step${CK}
    # Merge ATOMICALLY. Two eval cards were handed the same curve point and merged into this one
    # directory at once; the result was a half-written model and vLLM refused it with "Following
    # weights were not initialized from checkpoint: visual.blocks.18.norm1.bias". Build in a
    # private temp dir and rename into place -- rename is atomic, so a reader either sees the
    # complete model or no model, never a partial one.
    if [ ! -f $MODEL/config.json ]; then
      TMPM=$MODEL.tmp.$$
      rm -rf $TMPM
      BASE_SNAP=$(ls -d $R/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/* 2>/dev/null | head -1)
      case "${MODEL_FAMILY:-8B}" in 2B) BASE_SNAP=$(ls -d $R/hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/* 2>/dev/null | head -1);; 4B) BASE_SNAP=$(ls -d $R/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/* 2>/dev/null | head -1);; esac
      # 2026-08-30 11:30: a MERGE-protocol child (its MODEL= is a merged parent) must be merged onto THAT base, never the plain family base.
      [ -n "${MERGE_BASE:-}" ] && [ -f "$MERGE_BASE/config.json" ] && { BASE_SNAP=$MERGE_BASE; echo "[eval] merge base overridden: $BASE_SNAP"; }
      $EV/bin/python $R/surface/verl_rl/merge_lora.py \
        --ckpt $R/work/verl/ckpt_$TAG/global_step_$CK/actor --base "$BASE_SNAP" --out $TMPM || { rm -rf $TMPM; exit 1; }
      if [ -f $MODEL/config.json ]; then rm -rf $TMPM; else mv -T $TMPM $MODEL 2>/dev/null || rm -rf $TMPM; fi
      [ -f $MODEL/config.json ] || { echo "[eval] merge failed for $TAG step $CK"; exit 1; }
    fi
    CLEANUP=$MODEL
    echo "[eval] curve point: $TAG step $CK on the fixed initial surface" ;;
  C|D|D2|D3|SELF|NOVEL_TRAINED)
    CK=$(ls -d $R/work/verl/ckpt_$TAG/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
    [ -n "$CK" ] || { echo "[eval] no checkpoint to evaluate"; exit 1; }
    MODEL=$R/work/verl/merged_${TAG}_eval
    # Merge once; both trained cells share it so they are the same policy by construction.
    # Written as a plain conditional: the previous `[ -f ] || python ... \ ... || exit 1` form
    # mis-parsed once the merge already existed and ran the continuation line as its own command.
    if [ ! -f $MODEL/config.json ]; then
      BASE_SNAP=$(ls -d $R/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/* 2>/dev/null | head -1)
      case "${MODEL_FAMILY:-8B}" in 2B) BASE_SNAP=$(ls -d $R/hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/* 2>/dev/null | head -1);; 4B) BASE_SNAP=$(ls -d $R/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/* 2>/dev/null | head -1);; esac
      # 2026-08-30 11:30: a MERGE-protocol child (its MODEL= is a merged parent) must be merged onto THAT base, never the plain family base.
      [ -n "${MERGE_BASE:-}" ] && [ -f "$MERGE_BASE/config.json" ] && { BASE_SNAP=$MERGE_BASE; echo "[eval] merge base overridden: $BASE_SNAP"; }
      $EV/bin/python $R/surface/verl_rl/merge_lora.py \
        --ckpt $R/work/verl/ckpt_$TAG/global_step_$CK/actor --base "$BASE_SNAP" --out $MODEL || exit 1
    else
      echo "[eval] reusing merged checkpoint at $MODEL"
    fi
    echo "[eval] cell $CELL uses checkpoint step $CK" ;;
esac

. $R/env.sh
CU=$ES/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU PATH=$ES/bin:$CU/bin:$PATH
export PYTHONPATH=$R/surface/gate_surface:$R/surface/gate_caller
export VLLM_USE_FLASHINFER_SAMPLER=0 GATE_MAX_TURNS=20 GATE_MAX_NEW_TOKENS=1024
export CALLER_FAMILY=qwen CALLER_PREFIX_CACHE=1 CALLER_MODEL_PATH=$MODEL
export VLLM_CACHE_ROOT=$R/hf_cache/vllm_cache_eval_$TAG
echo "[eval] cell $CELL: policy=$MODEL surface=$(basename $SUB) ($(wc -l < $SUB) names)"
# SEEDS is extensible: the cell resumes from what is already banked, so raising it adds seeds
# 900+i rather than redoing the run. More seeds shrink per-task noise in every comparison that
# uses this cell as an anchor.
$ES/bin/python $R/surface/verl_rl/coadapt_eval.py --mode cell --pool $POOL \
  --subset $SUB ${SMAP:+--subset-map $SMAP} --out $OUT/cell_$CELL.jsonl --seeds ${SEEDS:-2} --mem-fraction ${EVAL_MEMFRAC:-0.62}
rc=$?
# Reclaim the merged copy for curve points only. The SELF/D merges are reused across cells and
# must survive; a curve point is consumed once.
[ -n "${CLEANUP:-}" ] && [ $rc -eq 0 ] && rm -rf "$CLEANUP"
exit $rc
