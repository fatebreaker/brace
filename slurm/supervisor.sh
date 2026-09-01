#!/bin/bash
# Keeps all nine cards working, and heals them itself.
#
# WHY THIS REPLACES THE WATCHERS. The old watchers exited when they fired, so every event
# depended on me noticing the notification and relaunching. One missed relaunch on 2026-08-06
# left the training watcher dead for seven hours while coadapt2 sat hung for 229 minutes and
# fixbase for 193. A monitor whose correctness depends on an operator restarting it is not a
# monitor. This one runs forever and RESTARTS the work rather than reporting it.
#
# NINE CARDS: four H200 (one arm each), one 2-GPU A100 pod (one data-parallel arm across both),
# and three A100 pods running evaluation cells pulled from a queue so they never sit idle when
# a cell finishes.
set -u
R=${BRACE_ROOT:?BRACE_ROOT is unset; source env.sh first}
EV=$BRACE_ENVS/mcp_verl/bin/python
CYCLE=${CYCLE:-300}
STALL_MIN=${STALL_MIN:-15}     # log silent this long, with no progress, means stuck
STUCK=${STUCK:-16}              # cycles of flat progress before we act regardless of log noise
MAXFAIL=${MAXFAIL:-6}          # consecutive restarts of one slot before escalating to the user
STATUS=$R/logs/fleet_status.txt
STATE=$R/logs/.supervisor_state
QUEUE=$R/work/eval_queue.txt

# slot | jobid | tag | logname | kind          (kind: train2 = the two-GPU pod, train, eval)
# Slots are re-read from disk EVERY cycle. Twice today a running supervisor kept a stale slot
# list through a config edit and relaunched a renamed arm over its replacement. Config changes
# must take effect without a restart, or the restart itself becomes the hazard.
SLOTFILE=$R/slurm/fleet_slots.txt

declare -A last strike fails cur
[ -f "$STATE" ] && while read -r k v s f c; do
  [ -n "${k:-}" ] && { last[$k]=$v; strike[$k]=$s; fails[$k]=$f; cur[$k]=${c:-none}; }
done < "$STATE"

train_cmd () {   # tag -> the exact relaunch for that arm
  local tag=$1 jid=$2 gate=discord off=0 rw="" extra=""
  # POOL. Default is the 320-task pool every arm has used. atscL2/atscfixL2 move to pool_big800
  # (712 tasks, same 158 scenarios, zero held-out overlap) because the batch-32 arms showed
  # training reward rising +0.04 while held-out stayed flat -- the signature of fitting the pool
  # rather than the task. At 2400 episodes over 320 tasks each task is seen ~7 times; 712 tasks
  # cuts that to ~3. Same scenarios means the handicapped surface and the base anchor are
  # unchanged, so curve points stay comparable across the switch.
  local extra_slots="AWM_SLOTS_TRAIN=8"
  local pool="$R/surface/gate_caller/pools/pool_big320.json"
  case "$tag" in atscL2|atscfixL2|q2b|q2bfix) pool="$R/surface/gate_caller/pools/pool_max.json";;
                 q2bL|q2bL2|q2bM) pool="$R/surface/gate_caller/pools/pool_learnable.json";;
                 q2bA|q2bF|q2bF2|q2bF3|q2bF4|q2bF5|q2bF3e5|q2bF6|q2bF7|q2bT|q2bTr|t2bTf|q2bVf|q2bK|q2bT2|q2bT3|q2bTd|q2bTiid|q2bTnw|q2bTnw2|q2bD|q2bP|q2bR|q2bN|q2bV|q2bLp|q4bV|q4bTvip|q4bV2|q4bLp|q4bF|q4bF2|q4bT|q4bTpe|q4bTpe2|q4bTf|q4bTr|q4bTr2|q4bTr3|q4bTa|q4bTa2|q4bTx|q4bTy|q4bTz|q4bTa16|q4bTz2|q4bTnw|q4bT2|q4bD|q4bP|q4bR|a8A|a8F|a8Fr|b8plr|b8plrr|b8k16|b8van|b8match|a8A2|a8A3|a8Az0|a8Auni|a8Ae0|b8m25|a8C|a8Cu|b8k16v|a8T|a8Tg1|a8Te0|a8T2w|a8T2s|a8T2r|a8T2|a8T2g1|a8T2g1r|a8T2b|a8T2n|a8T2nr|a8T3|a8T3g|a8Tr|t8Tf|a8Tvipf|a8Tme|a8Tme2|a8T3gr|a8T3g2|a8T4|a8T5|a8T5k|a8T5kr|a8T6d|a8T7w|a8T8c|a8Tvip|a8Tvipr|a8Tvipc|a8Tlp|a8Tpe|q2bTk8)  pool="$R/surface/gate_caller/pools/pool_max.json";;
                 b8ret|b8accel|b8dense) pool="$R/surface/gate_caller/pools/pool_max.json";;
                 # THE DATA-BUDGET ARMS (2026-08-18). pool_big320 has never been a LIST -- it is
                 # the default above, and every arm that trains on it (dapo) gets there by falling
                 # through every case. That is fine for an arm whose pool is incidental and wrong
                 # for one whose pool IS the manipulation: if a future edit moves the default, the
                 # experiment silently becomes a replication of its own parent and nothing in the
                 # composed command would look out of place. q2bTs/q2bFs therefore PIN the small
                 # pool explicitly, in the same case their parents are pinned to pool_max in, so
                 # the single deliberate delta is visible at the site it happens.
                 q2bTs|q2bFs) pool="$R/surface/gate_caller/pools/pool_big320.json";; esac
  case $tag in
    coadaptn) gate=naive ;;
    atsc|atsc2|atsck8) gate=none; rw="--surface-control" ;;
    atsck16)  gate=none; rw="--surface-control"; extra="NROLL=16" ;;
    # DEPTH ARMS (2026-08-07). Held-out policy effect was ~0 for every arm (atsc +0.010 p=0.146,
    # everything else +0.002 at p~1.0) while availability rose 7.5%->20%. The arms had only 5-72
    # GRPO steps, which is far too short for LoRA at 1e-5 to move a policy -- so that null is not
    # evidence the method fails, it is evidence it was never tested. These run to 496 steps.
    # Control is atscfix (static 0.5), NOT atscbase (0.0): only static-vs-adaptive isolates the
    # controller; comparing against 0.0 confounds adaptivity with a richer surface.
    atscL2)     gate=none; rw="--surface-control"; off=500 ;;
    atscL3)     gate=none; rw="--surface-control"; off=900 ;;
    atscfixL2)  gate=none; rw="--surface-fixed 0.5"; off=500 ;;
    # 2B ARMS (2026-08-08). Qwen3-VL-2B probed at 0.091 base solve rate vs 0.114 for the 8B, so it
    # can do the task -- the failure mode that would have made it useless. It runs ~4x faster,
    # which attacks the real constraint: the papers that work use 500-1000 GRPO steps and we have
    # managed 24. LORA_RANK=0 is a FULL fine-tune, affordable at 2B and standard in that work,
    # removing another difference from the recipes that succeed.
    q2b)      gate=none; rw="--surface-control";   off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32" ;;
    # LEARNABLE-BAND arms. informative groups = steps x batch x availability; at 13.7% availability
    # and 55 steps atscfix accumulated ~241 groups, two to three orders below the 1e4-1e5 the
    # tool-RL papers that work use. 68% of our tasks sit at p=0 and 13% at p=1 -- they can never
    # produce gradient, so 86% of compute buys nothing. pool_learnable keeps only measured
    # 0<p<1 tasks (mean p=0.436, availability ~93% at k=5), a 6.6x gain per step, and the 2B model
    # adds ~4x throughput. surface-fixed 0.0 keeps the TRAIN surface identical to the EVAL surface,
    # so a gain here is not the out-of-distribution artifact that invalidated the first curve.
    # ENTCOEF=0 on both. A 0.005 entropy bonus on q2bL drove actor/entropy from ~0.06 to 8.37
    # -- the policy collapsed to near-uniform output, ppo_kl hit 0.186 (20x the healthy
    # ceiling) and clipfrac fell to 0 because the ratio ran past where clipping engages.
    # That, not the step size, produced the -0.031 to -0.044 held-out degradation.
    q2bL)     gate=none; rw="--surface-fixed 0.0"; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0" ;;
    q2bL2)    gate=none; rw="--surface-fixed 0.0"; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=3e-5 ENTCOEF=0.0" ;;
    q2bM)     gate=none; rw="--surface-fixed 0.0"; off=700; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=5e-5" ;;
    # 8B METHOD PAIR. 2B barely responds to learning rate -- at lr=1e-4 it shows ppo_kl 6e-5
    # against 8B's 4e-4..8e-3 (verified over six steps), and 2B is no faster once slots are
    # unpinned (1371 eps/hr either way). 8B also has the 1180-task anchor; the 2B anchor is 194.
    # BASELINE SET. ATSC claims to raise gradient availability by controlling the tool surface.
    # The honest comparison is against the other ways to raise the same quantity:
    #   b8plr  adaptive TASK selection by outcome variance (PLR / UED family)
    #   b8k16  brute-force sampling: g(p,k)=1-p^k-(1-p)^k rises with k, at 3.2x generation
    #   dapo   discard degenerate groups -- measured ~3/32 kept here, so ~10x generation
    # All share a8F's fixed 0.5 surface and pool_max, so only the mechanism differs.
    #   b8van  VANILLA GRPO at surface 0.0 -- no surface manipulation at all. Without it the
    #          claim is open to "maybe the plain handicapped environment does just as well".
    #   b8match RICHNESS-MATCHED static control. ATSC drifted to advertising 5242 of 5483 names
    #          (level 0.914) because most tasks sit at p=0 and the controller keeps restoring
    #          tools to push solve rate toward 0.5. a8F at level 0.5 advertises only 4380, so
    #          any ATSC gain is open to "it just advertises 20% more tools". This arm holds a
    #          FIXED surface at ATSC's realised richness, isolating adaptivity from quantity.
    #   a8A2   ATSC-v2. v1 is proportional control on (p-0.5), which assumes the surface can
    #          move p. Measured on a8A after 65 steps it cannot: 126 of 158 scenarios sat at
    #          level 1.00 (mean 0.877), so the controller had saturated and was equivalent to
    #          "advertise everything" -- which is why static 0.5 keeps pace. v2 steers only
    #          scenarios whose g(p,k) RESPONDS to the level and freezes the rest, the PLR/ACCEL
    #          learning-potential idea applied to the environment instead of task selection.
    #   a8A3   ELSA, THE METHOD ARM. Estimates each scenario's ELASTICITY -- the signed,
    #          noise-floored slope dg/dl of gradient availability against the tool surface --
    #          then does two things no baseline can do jointly: steers the surface only where
    #          that slope clears its own standard error, and prices the TASK SAMPLER by the
    #          availability each task will have AFTER its surface is steered rather than the
    #          availability it shows now. Every baseline here is this arm with one coordinate
    #          pinned: b8plr/AWS pin elasticity to 0, a8A2 pins the sampler to uniform, a8A
    #          pins elasticity to infinity, dapo filters after paying for the rollouts.
    #          It carries its own reweighting, so it takes NO --reweight flag.
    #          Starts every scenario at the handicap baseline (level 0) and buys tools only
    #          where they are earned: replayed on a8A's 15.9k episodes it ends at mean level
    #          0.249 advertising 70.0% of names, against a8F's 79.9% and a8A's 95.6%. A gain
    #          measured on a SMALLER surface cannot be dismissed as extra capability, which is
    #          the confound that made the a8A/a8F contrast unreadable.
    #   THE ELSA ABLATIONS. Each differs from a8A3 by exactly ONE flag through the SAME module,
    #   which is what makes them ablations rather than separate methods that happen to be worse.
    #   Replayed on a8A's banked episodes (mean level / credited scenarios / sampling ratio):
    #     a8A3    0.231 / 78 / 85.1x     the method
    #     a8Az0   0.255 / 102 / 79.4x    noise floor off -> 31% more scenarios wrongly credited
    #     a8Auni  0.231 / 78 / 1.0x      sampler off, surface identical
    #     a8Ae0   0.000 /  0 / 142.9x    surface frozen, sampler alone
    #   a8Auni and a8Ae0 bracket the method: if either matches a8A3 on held-out tasks, the joint
    #   allocation is not what is doing the work and the paper's central claim fails.
    a8Az0)    gate=none; rw="--surface-elsa --elsa-extra=--z=0";    off=500; extra="LR=1e-4 ENTCOEF=0.0 TP=2 MAXTOK=28672 PARAM_OFFLOAD=True OPT_OFFLOAD=True" ;;
    a8Auni)   gate=none; rw="--surface-elsa --elsa-extra=--uniform-weights";   off=500; extra="LR=1e-4 ENTCOEF=0.0 TP=2 MAXTOK=28672 PARAM_OFFLOAD=True OPT_OFFLOAD=True" ;;
    a8Ae0)    gate=none; rw="--surface-elsa --elsa-extra=--e-zero";    off=500; extra="LR=1e-4 ENTCOEF=0.0 TP=2 MAXTOK=28672 PARAM_OFFLOAD=True OPT_OFFLOAD=True" ;;
    #   b8m25   STATIC MATCHED TO ELSA'S REALISED SURFACE. b8match (level 0.91) was built when
    #           ATSC advertised MORE than static and the confound ran that way. ELSA advertises
    #           LESS -- 70.0% of names against static's 79.9% -- so the control that matters now
    #           is a static surface at ELSA's realised mean level of ~0.25, not 0.91.
    b8m25)    gate=none; rw="--surface-fixed 0.25"; off=500; extra="LR=1e-4 ENTCOEF=0.0" ;;
    #   a8C    ACCORD, THE METHOD ARM FOR THE REWARD AXIS. Six arms measured the same wall:
    #          63-78% of tasks never solved, 12-18% always solved, availability 5-17%,
    #          actor/grad_norm 0.006-0.031. ~88% of GRPO groups carry no gradient, and the tool
    #          surface cannot reach that -- its slope against availability is centred at ZERO.
    #          ACCORD shapes the reward from the task's OWN VERIFIER (guard-chain progress) and
    #          certifies per scenario that the shaping ranks solvable tasks above dead ones:
    #          measured AUC 0.679 (se 0.017) for the verifier signal against 0.479-0.511 for
    #          response length, turns, errors and parse failures -- all four correctly REFUSED.
    #          Without that gate, telemetry shaping inflates availability 10%->70.6% while
    #          teaching the policy nothing, which is the trap this arm exists to avoid.
    #          Surface held at a8F's 0.5 so the ONLY difference from the static control is the
    #          reward. Runs through verl_awm_train_agent.sh because AWM_DENSE_REWARD and
    #          AWM_CARVE_CERT must reach the Ray workers; DAPO_FILTER_GROUPS=0 keeps dynamic
    #          sampling off so nothing else confounds it.
    #   a8Cu   ACCORD ABLATION: the same verifier shaping with the CERTIFICATE REMOVED, so every
    #          scenario is shaped unconditionally. This is the arm that decides whether the
    #          contribution is the certification or merely the dense reward. Identical to a8C in
    #          every other respect -- same surface 0.5, same pool, same gates mode, same script --
    #          and it carries no AWM_ACCORD_CERT, which dense_reward treats as "shape everywhere".
    #          Prediction from the offline analysis: availability rises on both arms, but the
    #          uncertified arm shapes the scenarios whose verifiers are written backwards too, so
    #          its held-out binary metric should trail a8C even while its training reward looks
    #          healthier. If it MATCHES a8C, the certificate is decoration and the paper says so.
    a8Cu)     gate=none; rw="--surface-fixed 0.5"; off=500; extra="LR=1e-4 ENTCOEF=0.0 MAXTOK=28672 AWM_DENSE_REWARD=gates DAPO_FILTER_GROUPS=0 TRAIN_SCRIPT=$BRACE_ROOT/slurm/verl_awm_train_agent.sh AWM_RL_SRVDIR=/tmp/awm_srv_a8Cu" ;;
    #   b8k16v VALIDATION-CURVE TEST ARM. Same group-size baseline as b8k16, plus the in-training
    #          validation pass: VAL_FREQ=5 evaluates the live policy on pool_coadapt_heldout (295
    #          tasks, 0% overlap with the training pool -- the pool the eval cells already score)
    #          every 5 steps, on the SAME fixed handicap surface for every arm via
    #          AWM_VAL_ADVERTISED. That is what makes the curve comparable across arms whose
    #          TRAINING surfaces differ (0.0 for b8van, ~0.47 for ELSA, 0.5 for the reward axis).
    #          Costs no merge and no eval card: the policy is already in memory.
    # VAL_FREQ/AWM_VAL_ADVERTISED are set fleet-wide in the batch-32 block below, so they are
    # deliberately NOT repeated here. VAL_BEFORE gives this arm the step-0 anchor on the same
    # instrument, which is what the rest of its curve is measured against.
    b8k16v)   gate=none; rw="--surface-fixed 0.5"; off=500; extra="LR=1e-4 ENTCOEF=0.0 MAXTOK=28672 NROLL=16 VAL_BEFORE=True" ;;
    a8C)      gate=none; rw="--surface-fixed 0.5 --accord"; off=500; extra="LR=1e-4 ENTCOEF=0.0 MAXTOK=28672 AWM_DENSE_REWARD=gates DAPO_FILTER_GROUPS=0 TRAIN_SCRIPT=$BRACE_ROOT/slurm/verl_awm_train_agent.sh AWM_RL_SRVDIR=/tmp/awm_srv_a8C" ;;
    a8A3)     gate=none; rw="--surface-elsa";     off=500; extra="LR=1e-4 ENTCOEF=0.0 TP=2 MAXTOK=28672 PARAM_OFFLOAD=True OPT_OFFLOAD=True" ;;
    a8A2)     gate=none; rw="--surface-control-v2"; off=500; extra="LR=1e-4 ENTCOEF=0.0" ;;
    b8match)  gate=none; rw="--surface-fixed 0.91"; off=500; extra="LR=1e-4 ENTCOEF=0.0" ;;
    b8van)    gate=none; rw="--surface-fixed 0.0"; off=500; extra="LR=1e-4 ENTCOEF=0.0" ;;
    b8plr)    gate=none; rw="--surface-fixed 0.5 --reweight variance"; off=500; extra="LR=1e-4 ENTCOEF=0.0" ;;
    # b8plrr SEED 2 OF b8plr (2026-08-19) -- internal review item 2. Table 14's TIES are the reason
    # the transfer claim does not separate TRIAGE from the allocation literature, and PLR is one of
    # the two arms doing the tying at n=1. A seed-replicated PLR is the cheapest way to turn "we
    # tie the field" into either "we beat the field" or an honest negative. --seed-offset 1500 is
    # the ONLY field that differs from b8plr; the fleet's standard second-seed rung and the same
    # `r` suffix a8Fr/a8T3gr/a8T2nr/a8Tvipr already use. Read against b8plr, pooled by offset.
    b8plrr)   gate=none; rw="--surface-fixed 0.5 --reweight variance"; off=1500; extra="LR=1e-4 ENTCOEF=0.0" ;;
    #   b8k16  BUY availability with compute instead of earning it by allocation. g(p,k) rises
    #          with k, so "just use a bigger group" is the trivial alternative to this whole
    #          paper and a reviewer will ask for it. k=16 costs 3.2x generation per step, so
    #          the honest comparison is at matched GENERATION budget -- which is exactly the
    #          contrast we want: ELSA raises availability at zero extra rollouts.
    #          Its last attempt died with a Ray actor RPC socket closure -- the 16-rollout
    #          memory profile -- so it takes the offload config that fixed b8dense.
    #          AWM_SLOTS_TRAIN=3, not the default 8. k=16 is the point of this arm, so the group
    #          size cannot be cut -- but at BSZ=32 it puts 512 episodes in flight per step against
    #          160 for a k=5 arm, each needing an MCP server pool. This pod ran a8A and a8F (both
    #          k=5) together for hours; it cannot also carry a k=16 arm beside the dense-reward
    #          a8Cu, which alone needs 150G of the 320G. Fewer slots trades generation throughput
    #          for a footprint that fits, which is the right trade for a baseline that has never
    #          produced a single checkpoint in five attempts.
    b8k16)    gate=none; rw="--surface-fixed 0.5"; off=500; extra="LR=1e-4 ENTCOEF=0.0 NROLL=16 TP=2 MAXTOK=28672 PARAM_OFFLOAD=True OPT_OFFLOAD=True AWM_SLOTS_TRAIN=3" ;;
    a8A)      gate=none; rw="--surface-control";   off=500; extra="LR=1e-4 ENTCOEF=0.0" ;;
    a8F)      gate=none; rw="--surface-fixed 0.5"; off=500; extra="LR=1e-4 ENTCOEF=0.0" ;;
    # SEED REPLICATE OF THE BAR. a8F (+0.0386) is a single seed, and measured seed spread
    # (a8T2 vs a8T2r: 2.2x live groups, sign flips) exceeds every between-arm effect. No
    # "beats all baselines" claim is publishable until the bar itself is replicated. off=1500
    # shifts the task-stream seed; A100-class memory extras added below (runs on a 2xA100 pod).
    a8Fr)     gate=none; rw="--surface-fixed 0.5"; off=1500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    #   a8T    TRIAGE, THE METHOD ARM FOR THE ALLOCATION AXIS. The measured wall is not reward
    #          density and not advertised capability: 46.4% of tasks are never solved once, 18.6%
    #          always are, and 87.3% of GRPO groups therefore carry zero within-group variance and
    #          zero gradient -- 36.6% of all rollout compute spent on tasks that have never
    #          produced one. ACCORD attacked the reward (-0.0068) and ELSA the surface (+0.0110);
    #          both left the allocation alone, which is why the best arm on this table is still the
    #          null method a8F (+0.0386).
    #          TRIAGE allocates the cycle's generation budget over tasks by the POSTERIOR EXPECTED
    #          PROBABILITY THAT A GROUP CARRIES GRADIENT, w = E[1 - p^k - (1-p)^k] under a
    #          discounted Beta from the arm's OWN banked episodes, sampled without replacement
    #          before a single rollout is paid for. See surface/verl_rl/triage.py.
    #          THIS IS a8F PLUS EXACTLY ONE FLAG. Same surface 0.5, same pool, same LR, same
    #          ENTCOEF, same batch, same horizon and -- deliberately -- the SAME rollout_n=5 that
    #          a8F runs. Raising the group size is the other way to buy gradient availability and
    #          it already has its own arm (b8k16); doing it here too would confound the allocation
    #          claim with the compute claim and break the pre-registered "matched steps = matched
    #          generation compute" bar. triage.py reads k from NROLL, so the weight rule always
    #          uses the group size actually in force.
    #          Its competitor is the DAPO arm: filter degenerate groups AFTER generating them
    #          vs allocate BEFORE. VAL_BEFORE gives the step-0 anchor on the fleet's shared
    #          fixed-surface instrument, which is what the mechanism curve is read against.
    a8T)      gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # gamma=1 ablation: no evidence decay. The offline sweep says decay is the binding parameter
    # (gamma=0.9 caps evidence at 80 obs/task; gamma=1 reached ~52% degenerate in sim vs ~66%).
    # If this BEATS a8T it becomes the method config; either way the decay axis gets a real row.
    # Differs from a8T by exactly one env var.
    a8Tg1)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # eps=0 ablation: no uniform exploration floor. Tests whether the Beta-posterior optimism
    # alone keeps coverage alive, or whether the floor is load-bearing. Differs from a8T by one
    # env var, exactly like a8Tg1.
    a8Te0)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_EPS=0.0" ;;
    #   a8T2   TRIAGE v2. a8T's mechanism is MEASURED FLAT: steps 1-15 came back 89.5% degenerate
    #          against the uniform baseline a8F's 87.4%, 21.0 live groups per 1k rollouts against
    #          26.1. Two causes, both measured, both in the sampler and neither in the idea:
    #            EXPLORE TAX. A flat prior scores an unseen task (k-1)/(k+1)=0.67 against a dead
    #              task's 0.19, and there are 1127 of them, so the early cycles are a survey and
    #              not an allocation -- a8Te0's cycle 4 put 137 of 256 slots on tasks with under
    #              4 observations. The measured window closes before the posterior is worth
    #              anything. --warm-bank primes every task from the 321k banked episodes of the
    #              OTHER arms (capped 20 solves / 40 failures, so ~2 cycles of this arm's own
    #              rollouts outweigh them, and gamma ages them like any other count). Legitimate
    #              because solvability is measured task-intrinsic, not surface-dependent
    #              (concordance 0.453, z=-0.97); the arm is no longer self-contained, which is
    #              the honest cost and is stated as such in the paper.
    #            WEAK SUPPRESSION. w is bounded: at k=5 a task with 20 straight failures still
    #              scores 0.19 against a mid-band 0.88, only 4.6x, and dead tasks outnumber
    #              mid-band ones ~500 to ~200 -- so linear sampling still handed them 79 of 256
    #              slots at a8Te0 cycle 4. --temp 0.5 samples on w^2, turning 4.6x into 21x.
    #          Offline against the real bank, cycle 1 at k=5: 1127/1127 tasks primed, chosen mix
    #          206/256 mid-band (v1: 0, all low-obs), slots on p_hat<0.05 0.094 (v1 cold: 0.543
    #          by true bank rate, v2 0.168), pred_degenerate 0.238 against 0.55 for a uniform
    #          allocation on this pool. Everything else is a8T byte-for-byte: same pool, surface
    #          0.5, LR, ENTCOEF, k=5, BSZ=32, eps=0.10, gamma=0.9, 15x8 horizon.
    # v2 FACTORIAL. a8T2 = warm+tau. These two isolate the ingredients; a8T2r replicates the
    # full method at a different seed. Each differs from a8T2 by exactly one thing.
    # gamma=1 under warm-start: the warm prior never ages out. Tests whether decay is
    # load-bearing for v2 (stale foreign evidence vs fresh own evidence).
    a8T2g1)   gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # INTEGRATION arm: the val-curve-leading baseline's idea (retrieval-selected surface,
    # budget-matched to 0.5) composed with the v2 allocator. Orthogonal levers: retrieval picks
    # the TOOLS, TRIAGE picks the TASKS. Val/eval stay on the fixed surface -> comparable.
    a8T2b)    gate=none; rw="--surface-retrieval --retrieval-match-level 0.5 --task-alloc gradmass --warm-bank default --temp 0.5"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # CONFIRMATION SEED for gamma=1 (launched only if/when a8T2g1 is selected as the method,
    # per PLAN section 3b rule 3). Identical to a8T2g1 except the seed offset.
    a8T2g1r)  gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5"; off=1500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    a8T2w)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    a8T2s)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --temp 0.5"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    a8T2r)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5"; off=1500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    a8T2)     gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # ============================ TRIAGE v3 (2026-08-12) ============================
    #   a8T3/a8T3g  v2 plus the two levers PLAN section 5 still had unspent. v2's ingredients
    #          work and rank in mechanism order at step 15 (cold a8T -0.0153, a8T2 +0.0186,
    #          a8T2g1 +0.0255) but none of them beats plain a8F (+0.0271 @15, +0.0386 window),
    #          so v3 adds both remaining levers and nothing else.
    #
    #     --exclude-tasks   CERTIFIED POOL EXCLUSION. 104 of pool_max's 1127 tasks are refused
    #          by the TRAINING allocation (w never computed, and out of the eps floor too):
    #          38 verifier-unsatisfiable -- a guard reading only initial_db fires, so no reachable
    #          final state can pass -- and 66 pre-solved, where the verifier returns 'complete' on
    #          a NO-OP. Measured over 7,585 live groups: the 38 are 1.0000 degenerate with exactly
    #          zero live groups in 11,641 banked rollouts; the 66 are 0.8526 degenerate against
    #          the kept pool's 0.8570, so they are excluded for VALIDITY not yield -- 25.5% of
    #          their solved episodes end in no_tool_call against 4.0% elsewhere and they pay 1.0
    #          regardless, which is reward noise aimed at the base policy's own failure mode, at
    #          2.1x the response tokens. Together 7.17% of all historical generation on this pool.
    #          THE EVAL AND VAL POOLS ARE UNTOUCHED -- this is allocation, not a benchmark edit,
    #          so the a8F comparison stays exactly as valid as it was.
    #     --warm-shrink group   GROUP-CORRELATED POSTERIOR. v2 scores tasks as if the k rollouts
    #          of a group were independent. Measured on a8T2 cycles 1-2, 512 groups of 5: the
    #          batch MEAN is right (observed 0.3281 against the prior's 0.3273) but the counts
    #          are 302 all-fail / 158 all-solve / 52 mixed where iid predicts 101 / 13 / 397.
    #          Fitted intra-class correlation rho=0.78 (nu=0.285). Scoring with the exchangeable
    #          Beta-Binomial at that nu -- same closed form, one parameter lower -- takes the
    #          predicted degenerate fraction from 0.24 against a measured 0.88 to 0.82, closing
    #          99.3% of the bias in sample and holding out of sample (a8F 0.8995 vs 0.8741
    #          measured, a8Te0 0.9006 vs 0.9062). It also re-prices the always-solved band, which
    #          v2 bought at 0.226 -- more than an evidenced-dead task -- and v3 prices at 0.039.
    #          The capability shrink this replaces (scale the warm prior by base/bank = 0.484) is
    #          implemented as --warm-shrink capability and is NOT what these arms run: on each
    #          arm's own cycle-1 batch the observed rate over the prior's prediction is 0.94-1.20,
    #          so lambda is 1. The 0.118 is a greedy pass on a DISJOINT held-out pool at
    #          temperature 0; the 0.246 is temperature-1 training rollouts on pool_max.
    #          Offline, cycle 1 at k=5 on the real bank: pool 1127->1023, chosen mix 227/256
    #          mid-band (v2: 210), slots on p_hat<0.05 0.066 (v2: 0.098), calibrated degenerate
    #          0.8165 -> 36.7 live groups per 1k rollouts against a8F's measured 26.1.
    # Everything else is a8T2 byte-for-byte: pool, surface 0.5, LR, ENTCOEF, k=5, BSZ=32,
    # eps=0.10, tau=0.5, warm bank + caps, 15x8 horizon. a8T3 is gamma=0.9, a8T3g gamma=1.0 --
    # the same pair a8T2/a8T2g1 are, so the decay axis stays readable under v3.
    a8T3)     gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    a8T3g)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # ===== a8Tr: THE SCALE-ADAPTIVE CALIBRATION REVISION AT 8B (2026-08-27) =====
    # The revision exists only at 4B (q4bTr). This makes it three-scale with q2bTr. It is a8T3g --
    # the SHIPPED 8B method reference -- plus the 8B-refit concentration, exactly as q4bTr is q4bT
    # plus the 4B-refit one. (The commission said "a8T idiom"; a8T is the BARE v1 arm with no bank,
    # temp, exclusion or shrink, so it cannot be the reference the revision modifies. a8T3g is, and
    # it is the arm whose +3.11 window the commission itself quotes as the comparator.)
    #
    # TRIAGE_NU=0.3713 IS MEASURED, NOT ASSUMED: work/analysis/rho_by_scale.json gives the 8B NPMLE
    # refit as rho 0.7292459078 / nu 0.3712795496, quoted to 4 d.p. exactly as q4bTr quotes the 4B
    # nu 0.3501438963 as 0.3501. Boot must echo rho 0.729.
    # PRE-REGISTERED BOTH WAYS vs the 8B method (a8T3g +3.11 mean window, ~37.8 BFCL, ~39.0
    # NESTFUL): a window gain like q4bTr's means the mis-calibration mechanism is scale-general;
    # no gain means it is a 4B-specific effect and the contribution stays single-scale. A null
    # prints as a null. Method-side, single seed.
    a8Tr)     gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_NU=0.3713" ;;
    # CONFIRMATION SEED of a8T3g (launched after its step-15 Holm-starred +0.0366; selection
    # rule 3b-3). Differs only in the task-stream seed.
    a8T3gr)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=1500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # THIRD SEED of the method config (power for the pooled claim; PI-endorsed strengthening).
    a8T3g2)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=2500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # ===== a8Tme / a8Tme2: THE MATCHED-ESS INDEPENDENT-WEIGHTS CONTROL (2026-08-19) =====
    # Internal review item 1, ranked by the reviewer as "worth more than everything else on this
    # list": run TRIAGE with INDEPENDENT weights (nu -> infinity) but with tau RE-TUNED so the
    # weight distribution's ESS matches the correlated arm's, at 8B, >= 2 seeds. If the correlated
    # model still wins, W2 collapses and the paper's central novelty is established. If it ties,
    # the contribution becomes the DIAGNOSTIC plus the certificates plus the bank.
    #
    # *** PARENT CORRECTION -- READ THIS BEFORE COMPARING ANYTHING. *** The task that commissioned
    # these arms named a8T/a8T2 as the parents. THAT IS WRONG AND WOULD NOT HAVE BUILT THE
    # REVIEWER'S CONTROL. Neither arm is a correlated arm at the shipped temperature:
    #   a8T  = --surface-fixed 0.5 --task-alloc gradmass          (NO --warm-bank, NO --temp, NO --warm-shrink)
    #   a8T2 = ... --warm-bank default --temp 0.5                 (NO --warm-shrink) and off=500, NOT a seed-2 arm
    # You cannot remove correlation from an arm that never had it, and you cannot retune tau "from
    # 0.5" on an arm that carries no --temp at all. The correlated arm this control must match is
    # the SHIPPED one: work/analysis/reviewer_response_analyses.md lists the 8B correlated arms as
    # a8T3/a8T3g/a8T3g2/a8T3gr/a8T5/a8T5k/a8T5kr, quotes a8T3g's own cycles (ESS 482.73, 465.02)
    # as the 465-495 band the reviewer cites, and states "retuning the independent arm from
    # tau = 0.5 to tau = 0.3753" against a shipped tau of 0.5 -- which is a8T3g's --temp.
    # THEREFORE: a8Tme clones a8T3g (off=500) and a8Tme2 clones a8T3gr (off=1500), the fleet's
    # existing seed-2 rung of that same family. Read a8Tme against a8T3g and a8Tme2 against
    # a8T3gr, pairing by offset; never against a8T, a8T2, or each other's parent.
    #
    # THE MANIPULATION IS EXACTLY TWO TOKENS ON TOP OF THE TAG.
    #   (1) --warm-shrink group -> none : q2bTiid's idiom, and deliberately 'none' rather than a
    #       dropped flag. coadapt.py:401 reads `shrink = a.warm_shrink or TRIAGE_WARM_SHRINK or ""`
    #       and forwards only when non-empty, so an ABSENT flag leaves the choice to an env var a
    #       future launcher could set. 'none' is an explicit choices= value in both parsers, is
    #       truthy, is forwarded verbatim, and pins the independent path where the sweep can see
    #       it. triage.py:1191 then sets nu = None -- its own comment marks that the v1/v2 iid
    #       path, i.e. nu -> infinity, which is precisely the reviewer's ν→∞.
    #   (2) --temp 0.5 -> 0.3753 : the matched-ESS temperature, taken from the analysis and not
    #       chosen here. THIS TOKEN IS THE WHOLE POINT. Without it the arm is plain
    #       no-calibration and measures the level change confounded with the model change; the
    #       analysis shows the retune removes 42.1% of the batch movement at 8B, so an unretuned
    #       independent arm would answer a question nobody asked.
    # The warm bank, exclusions, gradmass, surface 0.5, LR, ENTCOEF, VAL_BEFORE and TRIAGE_DECAY
    # are byte-identical to the parents'. Verified by sweep, not asserted.
    #
    # PRE-REGISTERED READOUT, FIXED BEFORE EITHER ARM RUNS. Primary: BFCL v4 multi-turn at the
    # same fixed step the parent is read at, plus the window against the SAME-OFFSET parent
    # (a8Tme vs a8T3g, a8Tme2 vs a8T3gr), pooled over the two seed-pairs. Secondary, and reported
    # regardless: cycle-1 ESS and the top-256 overlap against the parent's own pool, which the
    # analysis predicts at 245.7/256 with ESS ~480 on both sides -- a live check that the retune
    # actually matched.
    # INTERPRETATION REGISTERED IN ADVANCE, BOTH BRANCHES PUBLISHABLE:
    #   parent beats these arms  => the correlation model earns its place at the OUTCOME level even
    #                               though it is rank-neutral at the allocation level; W2 collapses.
    #   tie                      => confirms the rank-neutrality theorem the analysis proved (0
    #                               discordant pairs in 41.3M, 245.7/256 shared at matched ESS) now
    #                               at the outcome level too; the paper reframes the correction as
    #                               the calibrated DIAGNOSTIC (Figure 4) and keeps it.
    # Neither branch is the hoped-for one and the arms are reported whichever way they fall.
    #
    # CLASS NOTE. a8T3g/a8T3gr are H200-class (util 0.35, no offload block) and these inherit that,
    # which is right for parity with the parents. a8Tme2 is STAGED for a 2xA100 pod, where TP=2 and
    # the offload flags are added automatically but util stays the H200 0.35 -- and 0.35 on 2xA100
    # is the configuration that OOM'd a8T/a8Tg1 at the update step. If a8Tme2 actually lands on
    # 2xA100 rather than an H200 it must be moved to the A100 util group (0.30) FIRST. Flagged
    # here rather than silently pre-applied, because that change would break parity with a8T3gr.
    a8Tme)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3753 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink none"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    a8Tme2)   gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3753 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink none"; off=1500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # v4: SMALL GROUPS, MORE TASKS. rho=0.78 within-group correlation makes rollouts 3-5 of a
    # group mostly redundant; k=2 x BSZ=80 = same 160 rollouts/step but 80 groups. NROLL/BSZ set
    # in the POST-block case below (the batch-32 block would override them here).
    a8T4)     gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # ============================ TRIAGE v5 (2026-08-13) ============================
    # v3g (a8T3g/a8T3gr) TIES uniform head to head -- pooled 2v2 seeds, 51.7% win rate, p=0.68 --
    # while beating every published method. A tie is weak, so v5 revises by taking the ideas from
    # the baselines that MEASURABLY WORK rather than by adding another dial to the allocator.
    #
    #   a8T5   ALLOCATION x ADAPTIVE SURFACE. The two levers that measured positive on this
    #          substrate are v3g's allocation and a8A's per-scenario surface control (+0.0341,
    #          Holm-starred at 15, complete window, still RISING at 30 -- the strongest non-uniform
    #          baseline). They are orthogonal by construction: surface control chooses which TOOLS
    #          each scenario advertises, TRIAGE chooses which TASKS the cycle generates on, and
    #          neither reads the other's state. This arm is a8T3g with a8A's surface flag in place
    #          of --surface-fixed 0.5 and NOTHING else changed -- same allocator, same warm bank,
    #          same tau, same exclusion list, same group model, same gamma=1.0, same seed offset.
    #          THE FLAG IS a8A's EXACTLY: a8A runs --surface-control (ATSC-v1, surface_control.py),
    #          not --surface-control-v2, so that is what this carries. --surface-fixed MUST NOT be
    #          passed with it: coadapt.py's surface chain is if/elif and --surface-fixed is the
    #          FIRST branch, so an arm carrying both would silently run the fixed surface under the
    #          adaptive name -- the same class of failure as a8C's clobbered ACCORD certificate.
    #          COMPOSE IS PERMITTED AND VERIFIED: coadapt.py refuses --task-alloc only beside
    #          --reweight and --surface-elsa (both rewrite the same parquet); --surface-control
    #          composes exactly the way --surface-retrieval did on a8T2b. The allocation runs at
    #          the TOP of the cycle and writes the parquet; the surface branch runs after it and
    #          ASSIGNS envmap, so there is no ordering in which one destroys the other (dry-run
    #          verified end to end: surface_map.json written, pool_cycle1.json written,
    #          cycles_built=[1], and the composed trainer env carries BOTH AWM_ADVERTISED_MAP and
    #          AWM_ADVERTISED).
    #          COMPARABILITY IS UNTOUCHED. Only the TRAINING surface moves. Validation is pinned
    #          fleet-wide by AWM_VAL_ADVERTISED (set in the batch-32 block below) and the agent
    #          loop applies it BEFORE the per-scenario map -- awm_agent_loop.py: `if val and
    #          _VAL_ADVERTISED is not None` precedes the `_ADV_MAP` branch -- so a validation
    #          rollout on this arm sees exactly the surface every other arm's does. The eval cells
    #          never read the arm's map at all. Nothing in the surface-control path writes a val
    #          variable (verified: surface_control.py's only output is levels.json + the map).
    #          KNOWN INHERITED PROPERTY, kept because a8A has it: surface_control.py counts every
    #          row of the arm's episodes.jsonl, val rows included, when it estimates a scenario's
    #          solve rate. Filtering them would make this arm's controller differ from a8A's, which
    #          is the one difference this arm exists to NOT have.
    #          WHY THIS IS NOT a8T2b AGAIN. The last integration arm -- retrieval surface x v2
    #          allocation -- came back +0.0085, and the autopsy says the two levers did NOT
    #          interact: retrieval main effect -0.0152, allocation main effect -0.0153, joint
    #          -0.0302 against an additive prediction of -0.0305, interaction +0.0003. Its
    #          allocator was fine (0.8594 degenerate / 28.1 live per 1k over steps 1-30, BETTER
    #          than its own fixed-surface sibling a8T2's 0.9146 / 17.1). What killed it was the
    #          SURFACE: retrieval REPLACES names, so a8T2b trained with 1735 of its 4380 name-slots
    #          (39.6%) on tools the evaluation never advertises, left 639 of the 3284 eval slots
    #          (19.5%) unpractised, and was a superset of the eval surface in 0 of 158 scenarios --
    #          it converted the largest training gain in the fleet (+0.1266) into +0.0059 of val.
    #          surface_control.py cannot do that. Its map is (full & init) | restored_withheld, so
    #          the training surface is a SUPERSET of the fixed val/eval surface by construction --
    #          verified on a8A's own map: 158/158 scenarios, |train n val|/|val| = 1.0000, 0
    #          unpractised eval slots, exactly like a8F and a8T3g and unlike a8T2b's 0/158 and
    #          0.8054. The lever this arm composes with adds capability the policy may not have at
    #          eval; it never removes capability the policy will have.
    a8T5)     gate=none; rw="--surface-control --task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    #   a8T5k  CORRELATION-OPTIMISED SAMPLING GEOMETRY. The calibrated allocator was swept offline
    #          at FIXED ~160 rollouts/step over k in {2,3,4,5,8,10} (BSZ=160/k, budget=8*BSZ),
    #          tau in {0.3,0.5}, eps in {0.05,0.10}, on the pinned warm bank a8T3g was priced from
    #          (the replay of a8T3g's cycle 1 off that bank is byte-identical to the arm's own
    #          pool_cycle1.json and stats line). Two things came out of it, and only one is safe:
    #            THE tau/eps AXES ARE ROBUST. tau=0.3 beats tau=0.5 and eps=0.05 beats eps=0.10 at
    #              EVERY k under EVERY correlation calibration tried (nu=0.285, nu refit at k=5,
    #              nu refit at k=2, and a k-interpolated nu). At k=5 the pair takes the cycle-1
    #              batch from 227 to 244 mid-band rows, dead rows 19 -> 7, predicted live groups
    #              per 1k 36.70 -> 38.80 on a8T3g's own instrument. w_ess falls only 482.7 -> 447.8
    #              of 1023, so the sharpening does not collapse the draw.
    #            THE k AXIS IS NOT. The simulator ranks k=3 first, and the two live arms falsify
    #              it: at nu=0.285 it predicts 36.7 live/1k for a8T3g (k=5, MEASURED 49.2 in cycle
    #              1, 51.8 over 51 steps) and 37.1 for a8T4 (k=2, MEASURED 26.6 in cycle 1, 20.0
    #              over 20 steps). Refitting nu on each arm's own cycle gives 0.4148 at k=5 and
    #              0.1847 at k=2 -- no single concentration fits both, and the residuals have
    #              OPPOSITE signs, so the exchangeable Beta-Binomial does not extend across group
    #              sizes and the sweep cannot rank them. The one k comparison that is measured
    #              rather than modelled says k=5 beats k=2 by 46% at identical generation compute.
    #            AND THE BUDGET CLIFF IS MEASURED. The bank holds 440 mid-band tasks. a8T4's
    #              budget of 640 consumed 440/440 of them every cycle and froze its batch at 83.4%
    #              cycle-over-cycle overlap (a8T3g at budget 256 rotates: 44.9-49.2%). The sweep's
    #              k=3 argmax sits at budget 432 = 89.1% of the same supply, one step from that
    #              cliff, and would also need 162 rollouts/step (unmatched generation compute) and
    #              a MINIBSZ off the fleet's 8. So k=5/BSZ=32 is KEPT -- matched k, matched compute,
    #              matched batch -- and this arm differs from a8T3g in exactly two scalars.
    a8T5k)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # CONFIRMATION SEED of a8T5k (launched on its Holm-starred #1 step-15, +0.0373 p=1e-4;
    # selection rule 3b-3, move-fast directive). Differs only in the task-stream seed.
    a8T5kr)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=1500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    #   a8T6d  PRESENTATION DECORRELATION -- the first arm that attacks the CEILING rather than
    #          the allocation under it. EXPLORATORY. It folds into the paper only if it beats the
    #          incumbent on the matched window; otherwise it is an appendix ablation or a
    #          measured negative, like ELSA and ACCORD before it.
    #
    #          THE MEASURED PROBLEM. Every allocator this project has built is bounded by one
    #          number we fitted ourselves: within-group correlation rho = 0.78 (nu = 0.285), from
    #          a8T2 cycles 1-2, 512 groups of k=5 -- 302 all-fail / 158 all-solve / 52 mixed,
    #          where iid at the SAME per-task p_i predicts 101 / 13 / 397. The batch MEAN is
    #          right; the within-group SPREAD is nearly absent. The cause is structural and was
    #          named when the group model was built: the k rollouts of a group share a checkpoint
    #          AND a byte-identical prompt, so they agree far more often than k iid draws. At
    #          rho = 0.78 NO allocation reaches below ~0.79-0.92 degenerate on this pool -- which
    #          is why a8T3g measures 0.7408 after every lever in v1-v3 was spent on the choice of
    #          tasks. Allocation picks WHICH groups to draw; it cannot make a group disagree.
    #
    #          THE INTERVENTION, AND WHY IT IS SEMANTICALLY FREE. TRIAGE_DECOR=shuffle gives the
    #          k rollouts of one group k DIFFERENT-BUT-EQUIVALENT presentations of the SAME task:
    #          the advertised tool list is permuted per rollout in the turn-0 prompt. A tool
    #          surface is a SET -- MCP's tools/list carries no ordering contract, interfaces.py
    #          dispatches through a by_name dict, and no verifier or reward reads order -- so the
    #          task, the action space and the success criterion are untouched, and n_tools is
    #          invariant by construction. The permutation is a pure function of (salt, scenario,
    #          task_idx, rollout_index), the k of them are de-duplicated as a family, and
    #          final_answer keeps its canonical last slot. This is the SAFEST rung of the ladder;
    #          formatting variation and description paraphrase are held back and would be flagged
    #          separately, paraphrase especially, because it does move semantics.
    #
    #          VALIDATION IS UNTOUCHED, and that is what keeps the arm comparable: the permutation
    #          fires only on the `not is_val` branch of awm_agent_loop.run, the rollout counter is
    #          never even claimed on the val path, and val_kwargs pins n=1 so there is no group
    #          there to decorrelate. a8T6d's validation rollouts see the same fixed
    #          advertised_init.txt surface (md5 e1ef6cea27bc) every other arm's do.
    #
    #          PRE-REGISTERED, FALSIFIABLE, AND CHEAP TO KILL. Within ~2 cycles of launch:
    #            (1) within-group MIXED-OUTCOME fraction rises materially above a8T3g's measured
    #                ~8-15% band on the matched window;
    #            (2) rho refitted on this arm's own cycle-1/2 batches drops materially below
    #                0.78 (equivalently nu rises well above 0.285).
    #          IF RHO HAS NOT MOVED BY STEP 8, THE HYPOTHESIS IS DEAD AND THE ARM IS STOPPED --
    #          the mechanism is visible in the first 2 cycles, long before any held-out cell, so
    #          this costs ~half a day of one card to falsify. Held-out fold criterion (enforced by
    #          the main session, not by this arm): beats a8T3g's THREE-SEED evidence
    #          (a8T3g/a8T3gr/a8T3g2) on the matched window AND the decorrelation mechanism is
    #          demonstrated. Both, or it is an appendix ablation / a negative. Every arm stays in
    #          the same Holm family (section 3b-2), so this one raises the winner's bar too.
    #
    #          CONFIGURATION: a8T3g EXACTLY -- warm bank + tau 0.5 + eps 0.10 (the default, hence
    #          no TRIAGE_EPS) + gamma 1.0 + certified exclusion + group shrink -- and off=500, the
    #          SAME seed offset, so the task stream is a8T3g's task stream and the pair is as
    #          close to paired as two arms get. The only difference in the whole configuration is
    #          TRIAGE_DECOR=shuffle. TRIAGE_DECOR_SEED is pinned to the seed offset so a future
    #          confirmation seed gets its own presentation family instead of silently inheriting
    #          this one. H200-class (util 0.35 group) like every other a8T3g-shaped arm: same 8B
    #          model, same BSZ=32/k=5 batch, and a permutation cannot change the token COUNT of a
    #          prompt, only its order -- so the memory profile is a8T3g's.
    a8T6d)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_DECOR=shuffle TRIAGE_DECOR_SEED=500" ;;
    # TRIAGE-B (v7): BUDGETED ROLLOUT ALLOCATION. a8T3g EXACTLY -- same warm bank, tau 0.5,
    # eps 0.10, gamma 1.0, certified exclusion, group shrink, same off=500 task-stream seed --
    # except that the allocator distributes the ROLLOUT budget rather than the task list: task i
    # takes m_i parquet rows = m_i independent groups of k=5, sum_i m_i = 256, so generation is
    # matched to a8T3g and a8F to the rollout. k_i in {0,5,10,15}.
    # TRIAGE_BUDGET_DRAW=sample IS LOAD-BEARING AND IS NOT THE SPEC'S GREEDY WATER-FILLING.
    # Measured before registration on the pinned cycle-1 bank: the greedy (exactly optimal for
    # this concave objective) is PROVABLY INERT at matched compute -- max marginal of a second
    # group is 0.1736 while 417 tasks have a first-group value above it, against a budget of 256,
    # so greedy returns 767/256/0/0 over m=0/1/2/3, i.e. fixed-k selection. Spill needs N>=418
    # (1.63x the matched budget). The sampled draw at tau=0.5 allocates 807/178/36/2 and is the
    # only configuration in which this arm differs from a8T3g at all. See PLAN_TRIAGE.md 5c.
    # Note: --temp 0.5 with the greedy draw is REFUSED at argv, so the inert config cannot be
    # launched by accident under this tag.
    a8T7w)    gate=none; rw="--surface-fixed 0.5 --task-alloc budget --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_BUDGET_DRAW=sample TRIAGE_KMAX=3 TRIAGE_BUDGET_OBJECTIVE=any" ;;
    # TRIAGE-rho (v8): PER-TASK WITHIN-GROUP CORRELATION. a8T3g EXACTLY -- same warm bank, tau 0.5,
    # eps 0.10 (the default, hence no TRIAGE_EPS), gamma 1.0, certified exclusion, group shrink,
    # same off=500 task-stream seed, same k=5/BSZ=32 -- except that the group model's concentration
    # is the TASK'S OWN rho_i (fitted offline by fit_task_rho.py from 24,877 banked groups) instead
    # of the single global rho=0.78. w = 1 - P(all k solve) - P(all k fail) under
    # BetaBin(k, nu_i*p_hat, nu_i*(1-p_hat)), nu_i = (1-rho_i)/rho_i. A task absent from the file
    # takes the global rho, so this is a strict refinement of a8T3g and not a second method.
    # WHY: a8T3g's weights carry only 51 DISTINCT VALUES over 1023 tasks, and the 256th slot sits
    # in a TIE CLASS OF 354 (all at the warm cap 20s/40f, all w = 0.202011) -- 225 of 256 rows are
    # handed out arbitrarily among ties. rho_i breaks that class into 353 distinct weights.
    # PRE-LAUNCH GATES, all passed, numbers in PLAN_TRIAGE.md 5d: heterogeneity LR 3724.9 against a
    # bootstrap null of 550.5 +- 32.9 (p <= 0.0033), shrunk rho_i IQR 0.166; selection overlap with
    # a8T3g's REAL cycle-1 pool 168/256 against a draw-noise floor of 116/256; leave-one-arm-out
    # retrodiction +0.246 live-group rate for the low-rho half, 20/20 arms in the predicted
    # direction; corr(rho_i, p_hat(1-p_hat)) -0.376, and inside the tie class p_hat is IDENTICAL
    # while rho_i still has sd 0.1999.
    # PRE-REGISTERED, SAME-DAY FALSIFIABLE: on the pinned bank this arm's own batch predicts
    # degenerate 0.636 under the per-task model where a8T3g's predicts 0.816, while the OLD global
    # -rho reading of the same two batches is 0.8169 vs 0.8133 -- indistinguishable. So the measured
    # degenerate fraction (triage_report.py) is a direct test of the per-task model: it must fall
    # materially below a8T3g's measured 0.7408 within ~2 cycles, or the hypothesis is dead.
    # H200-class (util 0.35 group) like every other a8T3g-shaped arm: same 8B model, same batch,
    # and a different weight cannot change the memory profile.
    a8T8c)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group --estimator taskrho --task-rho $R/work/analysis/task_rho.jsonl"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    #   a8T2n  THE NAIVE DIFFICULTY-BAND BASELINE, and the honest test of whether TRIAGE's
    #          posterior machinery is load-bearing at all. Same allocator, same warm bank, same
    #          gamma, same eps, same budget, same seed offset as a8T2 -- the ONLY difference is
    #          the weight rule: point estimate p_hat = s/(s+f) on the same decayed counts, keep
    #          0.20 <= p_hat <= 0.80, keep every task with no evidence (a naive rule has to
    #          survive its own cold start), and draw UNIFORMLY inside the kept set. No Beta
    #          posterior, no g(p,k), no tau, no optimism.
    #          Dry-run on the real bank, cycle 1: the band keeps 459/1127 tasks and 100% of
    #          TRIAGE's top-256 by weight is inside that kept set; the two rules choose 209 of
    #          the same 256 tasks (81.6%) and the naive batch is marginally SHARPER (239 mid-band
    #          vs 210, 12 dead vs 26, calibrated degenerate 0.8109 vs 0.8245). So with a strong
    #          warm bank the posterior is NOT doing the work at cycle 1, and this arm is what
    #          lets the paper say that instead of being asked it. What the band rule has no
    #          answer for is the cold start (0 of 1127 tasks lack evidence HERE only because the
    #          bank is warm), the hard 0.2/0.8 cliff, and re-probing a task the estimate has
    #          written off -- which is where the decay and the posterior width earn their place.
    a8T2n)    gate=none; rw="--surface-fixed 0.5 --task-alloc bandfilter --warm-bank default"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # a8T2nr SEED 2 OF a8T2n (2026-08-14). The method (TRIAGE) is held to a 3-seed replication
    # standard; a baseline that clears Holm on ONE seed is not held to a weaker one, or the
    # comparison stops being about the allocator and starts being about how many seeds each side
    # was allowed. Identical to a8T2n in every field -- same rw, same gate, same extra, same
    # class -- except --seed-offset, which is the whole and only delta (500 -> 1500), exactly
    # the a8T3g/a8T3gr and a8T5k/a8T5kr idiom.
    a8T2nr)   gate=none; rw="--surface-fixed 0.5 --task-alloc bandfilter --warm-bank default"; off=1500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # ================= THE PUBLISHED-METHOD ALLOCATION BASELINES (2026-08-12) =================
    # PLAN section 5b item 2, plus the estimator ablation section 2b's novelty claim now rests on.
    # All three are the SAME allocator (surface/verl_rl/triage.py) with a different weight rule --
    # same warm bank, gamma, eps, budget, seed offset, k, BSZ and stats line -- so each differs
    # from the selected config in exactly one named thing and the comparison is about the rule.
    #
    #   a8Tvip  VIP (Nguyen et al., arXiv:2602.01601, ICLR 2026, code public). Before-generation,
    #           budget-constrained, variance-minimising allocation. Its Propositions 4.2/4.3 give
    #           Var(G~_q) = c(n_q) * 4 sigma_Zq^2 * p_q(1-p_q) and its Theorems 5.1/5.2 make the
    #           optimal n*_q strictly increasing in a_q = 4 sigma_Zq^2 p^_q(1-p^_q), with
    #           sigma_Zq^2 taken equal across prompts by the paper itself -- so at our fixed group
    #           size the whole task-dependent part of VIP's objective is p(1-p), and --rule vip
    #           weights by its posterior form E[p(1-p)] on the same warm bank. No exclusion file
    #           and no zero weights: VIP's constraint is L <= n_q <= U with L >= 3, so it cannot
    #           skip a prompt, and triage.py REFUSES --exclude-tasks under this rule.
    #           Dry-run, cycle 1 on the real bank (k=5, 1127 tasks, seed 500): chosen mix 188/256
    #           mid-band, 47 dead, calibrated degenerate 0.8384 -> 32.3 live groups/1k.
    #           THE EX-ANTE FINDING, and it is a paper result rather than a formality: at fixed k
    #           VIP's variance criterion and TRIAGE's availability criterion are the SAME RANKING.
    #           On point estimates that is exact (g(p,k) = 1-p^k-(1-p)^k is a strictly increasing
    #           function of u = p(1-p) for every k>=2; 0/4005 discordant pairs). On the pool's real
    #           posteriors Spearman is 1.000000, the deterministic top-256 sets are identical, and
    #           only 9 of 1127 tasks move rank at all (max 4 places). Matched against gradmass at
    #           the same tau/bank/seed the two batches share 236/256 tasks (92.2%) against a
    #           same-weights-different-seed floor of 65/256 (25.4%). So the objective is not what
    #           separates the method from VIP -- the estimator, the exclusion and the correlation
    #           model are, which is exactly what a8Tpe/a8T3/a8T3g measure.
    #   a8Tlp   LEARNING-PROGRESS CURRICULUM (Graves et al. 2017; Matiisen et al., TSCL). Weight is
    #           |delta p^| between the two most recent evidence windows -- the gamma-aged history
    #           and this cycle's own rollouts -- ABSOLUTE, so tasks the policy is FORGETTING are
    #           re-selected as well as tasks improving, which is what TSCL's teachers do. Tasks
    #           without a second window take the mean |delta p^| (explore-neutral: the rule must
    #           not inherit TRIAGE's optimism, which is the thing under test). Cite-faithful: the
    #           signal is PROGRESS, not availability -- a task stuck at p=0.5 forever scores 0 here
    #           and 0.94 under gradmass.
    #           Dry-run, cycle 1: no live window exists yet, so 0/1127 tasks have two windows and
    #           the draw is EXACTLY uniform (w_ess 1127.0/1127, w_entropy = ln 1127 to the digit),
    #           mix 103 mid / 120 dead, calibrated degenerate 0.8986 -> 20.3 live groups/1k. Fed a
    #           real second window (a8T3g's own log as cycle 2) it comes alive: 590/1127 tasks
    #           scored, mean |dp| 0.2207, and its batch is MORE dead-heavy than gradmass's on the
    #           same evidence (107 vs 43 dead, calibrated 0.880 vs 0.834) -- progress is not
    #           availability, and this arm is what lets the paper show that instead of assert it.
    #   a8Tpe   THE ESTIMATOR ABLATION (TRACE, arXiv:2606.11119, plug-in v^m). a8T3g with
    #           --estimator point and NOTHING else changed: same rule, same warm bank, same tau
    #           0.5, same exclusion list, same group model, same gamma=1.0 -- w becomes the plug-in
    #           g(p^,k) at the posterior mean instead of the posterior expectation E[g(p,k)].
    #           gamma=1.0 is kept precisely BECAUSE a8T3g runs it: an arm that differs from the
    #           selected config in the estimator AND the decay measures neither.
    #           Dry-run, cycle 1: 229/256 mid-band against a8T3g's 228, calibrated degenerate
    #           0.8118 vs 0.8124, 248/256 tasks shared. The two estimators agree at cycle 1 because
    #           the warm bank makes every task evidenced; where they must diverge is the
    #           no-evidence and re-probe cases, where the plug-in cannot express ignorance -- at
    #           k=5 it scores an UNEXPLORED task and a task MEASURED at p=0.5 identically (0.9375),
    #           while the posterior separates them 0.6667 vs 0.9087.
    # A100-class, all three: they land on freed 2xA100 pods, so they take the offload block and
    # util 0.30 below (a8T3g itself is H200-class at 0.35 -- the class follows the card, not the
    # config).
    a8Tvip)   gate=none; rw="--surface-fixed 0.5 --task-alloc vip --warm-bank default"; off=500; extra="RESUME_LOAD=model_extra LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # a8Tvipr SEED 2 OF a8Tvip (2026-08-14), same reason as a8T2nr: the replication standard is
    # applied to the baselines that cleared Holm, not only to the method. --seed-offset 1500 is
    # the only field that differs from a8Tvip. Registered but NOT queued: a8Tvip's own run is
    # incomplete, and a seed-2 replicate of a partial seed 1 replicates nothing -- the parent
    # resumes to its horizon first.
    a8Tvipr)  gate=none; rw="--surface-fixed 0.5 --task-alloc vip --warm-bank default"; off=1500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # a8Tvipc CONTINUATION OF a8Tvip (2026-08-14). a8Tvip was preempted at step 15 and CANNOT be
    # resumed: a scratch cleanup pruned its optimizer and extra_state shards, leaving
    # ckpt_a8Tvip/global_step_15/actor with only model_world_size_1_rank_0.pt + huggingface/.
    # Every verl resume path calls fsdp_checkpoint_manager.load_checkpoint, which opens
    # extra_state_world_size_1_rank_0.pt unconditionally -- three attempts on 2026-08-14 all died
    # on that exact FileNotFoundError, so RESUME_LOAD is dropped here rather than retried.
    # Instead the step-15 LoRA is MERGED into the base and becomes this arm's base model, so
    # training continues from the same weights with a FRESH optimizer. Identical to a8Tvip in
    # every selection field (rule, tau, warm bank, seed offset) -- MODEL is the only difference.
    # STEP MAPPING: a8Tvipc step s == a8Tvip step 15+s, so the paper's matched window 20/25/30 is
    # this arm's STEP5/STEP10/STEP15. Optimizer state is re-initialised at step 15; that is a
    # disclosure for the paper (footnote), not a silent change.
    a8Tvipc)  gate=none; rw="--surface-fixed 0.5 --task-alloc vip --warm-bank default"; off=500; extra="MODEL=$BRACE_WORK/big/merged_a8Tvip_step15 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    a8Tlp)    gate=none; rw="--surface-fixed 0.5 --task-alloc progress --warm-bank default"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    a8Tpe)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.5 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group --estimator point"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # ===== t8Tf: FAITHFUL PUBLISHED TRACE AT 8B (2026-08-26) =====
    # THE FAITHFUL-VIP PLAYBOOK, APPLIED TO TRACE. a8Tpe/q4bTpe carry OUR bank, OUR exclusion and
    # OUR harness, with the plug-in estimator as the posterior's no-prior limit -- so they are
    # configurations of OUR framework, not the published baseline. The published baseline is TRACE
    # with NONE of our components, exactly as q4bTvip is q4bV minus --warm-bank.
    #
    # t8Tf = a8Tpe MINUS every component of ours: no --warm-bank (COLD START), no --exclude-tasks
    # (all 1127 pool tasks eligible), no --warm-shrink (no calibration), and no --temp (triage.py's
    # own default). What remains is the published method: gradmass selection scored by the
    # TRACE-style plug-in g(p_hat,k). TRIAGE_DECAY=1.0 is kept only because it IS triage.py's
    # default gamma, so it changes nothing and keeps the extra byte-identical to a8Tpe's.
    #
    # REGISTERED PREDICTION, STATED BEFORE THE RUN: cold-start collapse, as faithful VIP showed --
    # removing the bank cost it 11pp of transfer at 4B. The mechanism is documented and specific
    # (q2bTnw): with no bank, cycle 1 has zero evidence and the allocator collapses to uniform.
    # If instead t8Tf holds up, the bank is not load-bearing for the plug-in estimator and that is
    # a finding about our components, not about TRACE. A null prints as a null.
    #
    # PRE-REGISTERED READOUTS: window vs the method ladder AND vs a8Tpe's completed window;
    # transfer BFCL/NESTFUL vs a8Tpe 38.1/39.76(s15) and the method ~37.8/39.0.
    # ONE SEED (PI baseline-seed rule: a new baseline, not a replica).
    #
    # INTEGRITY: the method-of-record row stays the posterior config. a8Tpe/q4bTpe join the
    # framework family with honest labels; t8Tf/q4bTf are the published baseline. Nothing here is
    # relabelled as the method's own numbers.
    t8Tf)     gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --estimator point"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # ===== a8Tvipf: FAITHFUL PUBLISHED VIP AT 8B (2026-08-28) =====
    # Completes faithful bank-less VIP at all three scales (this, q4bTvip at 4B, q2bVf at 2B) and
    # removes the "not run at this scale" markers from the baseline grid.
    # a8Tvip is `--surface-fixed 0.5 --task-alloc vip --warm-bank default` with
    # `RESUME_LOAD=model_extra`. The faithful arm removes the bank -- and ALSO RESUME_LOAD, which
    # is NOT part of VIP: it is scar tissue from a8Tvip's own preempted run (a scratch cleanup
    # pruned its optimizer/extra_state shards; three resume attempts died on the same
    # FileNotFoundError). Pointing a brand-new arm at a resume path it has no checkpoint for would
    # fail on boot, and a8Tvipr/q4bTvip/q2bV all carry no RESUME_LOAD -- so dropping it is the
    # house idiom, not a new decision. The result is exactly q4bTvip's config at 8B.
    # PRE-REGISTERED BOTH WAYS. 4B faithful VIP collapsed without the bank (+1.88 / 25.50 / 7.25
    # against +4.11 / 36.88 / 18.73 with it). At 8B the outcome is recorded as UNCERTAIN, not
    # predicted: t8Tf did NOT collapse in-distribution at 8B (+3.35), so capacity may rescue VIP's
    # rule too. Either direction prints. Single seed (baseline).
    a8Tvipf)  gate=none; rw="--surface-fixed 0.5 --task-alloc vip"; off=500; extra="LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # ===== t2bTf: FAITHFUL PUBLISHED TRACE AT 2B (2026-08-27) =====
    # Completes faithful TRACE at all three scales (t8Tf 8B, q4bTf 4B, this at 2B). Exactly t8Tf's
    # configuration plus the 2B scale tokens: gradmass + estimator point, NO bank, NO exclusion,
    # NO shrink, NO temp -> cold start. Single seed (baseline).
    # PREDICTION, STATED: transfer collapse per the faithful-VIP/TRACE pattern. The
    # IN-DISTRIBUTION axis is now genuinely uncertain -- t8Tf did NOT collapse in-distribution
    # (+3.35), so the cold-start penalty may be transfer-specific. Recorded as uncertain rather
    # than predicted, which is the honest state after t8Tf.
    t2bTf)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --estimator point"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # ===== q2bVf: FAITHFUL PUBLISHED VIP AT 2B (2026-08-28) =====
    # q2bV minus --warm-bank and nothing else; the 2B corner of the same three-scale family.
    # PREDICTION: severe collapse -- weakest model, cold start, and the 4B bank removal already
    # cost 11pp of transfer. Refs: 2B method +1.89 2-seed window / 16.6-15.6 transfer, VIP+bank 2B
    # transfer 11.1. Single seed (baseline). A null prints as a null.
    q2bVf)    gate=none; rw="--surface-fixed 0.5 --task-alloc vip"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    q2bA)     gate=none; rw="--surface-control";   off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32" ;;
    q2bF)     gate=none; rw="--surface-fixed 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32" ;;
    # q2bF2 RERUN OF q2bF FROM SCRATCH (2026-08-14). q2bF cannot be resumed: a scratch prune took
    # its window checkpoints and left only ckpt_q2bF/global_step_5, whose extra_state and optimizer
    # shards are gone -- the same FileNotFoundError in fsdp_checkpoint_manager.load_checkpoint that
    # killed a8Tvip's resume. Nothing survives to continue from and nothing survives to merge into a
    # base model (a8Tvipc's escape hatch needs a step worth continuing from; step 5 is not one), so
    # the 2B scale table's uniform control row has to be re-earned by running the arm again.
    #
    # SAME SEED OFFSET 500, NOT A NEW ONE. This is a REPLACEMENT for a lost run, not an added
    # replication seed: the row it fills is q2bF's own row in the 2B scale table, and every other
    # row of that table (q2bA/q2bT/q2bD/q2bP/q2bR/q2bN) trains on the off=500 surface realization.
    # A rerun at a different offset would measure a different surface draw and stop being the
    # control the table's other rows are compared against. Contrast a8T2nr/a8Tvipr, which take a
    # fresh offset precisely because they are second seeds of a run that still exists.
    #
    # BYTE-IDENTICAL TO q2bF IN EVERY SELECTION FIELD -- gate, rw, off, MODEL, LORA_RANK,
    # LORA_ALPHA. --tag is the only difference in the composed command, verified by sweep.
    q2bF2)    gate=none; rw="--surface-fixed 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32" ;;
    # q2bF3/q2bF4 SEEDS 2 AND 3 OF THE 2B UNIFORM CONTROL (2026-08-15, PI-directed power
    # extension). The 2B BFCL contrast currently rests on ONE T-vs-F pair, which is a single
    # draw of the surface and of the data order; the pre-registered protocol raises it to three
    # pairs and reports the POOLED paired test (T_i vs F_i, i=1..3). These are the F side.
    #
    # NEW OFFSETS 1500 AND 2500, unlike q2bF2's 500. That is the whole point and the opposite of
    # q2bF2's reasoning: q2bF2 REPLACED a lost run and had to keep the table's shared surface
    # realization, while these are ADDED replication seeds and must draw fresh ones -- the same
    # 500 -> 1500 -> 2500 ladder a8T3g/a8T3gr/a8T3g2 and a8T2n/a8T2nr already use at 8B.
    #
    # BYTE-IDENTICAL TO q2bF2 IN EVERY OTHER SELECTION FIELD -- gate, rw, MODEL, LORA_RANK,
    # LORA_ALPHA. --tag and --seed-offset are the only differences in the composed command,
    # verified by sweep. NOT added to queue_curve.sh, matching q2bF/q2bF2: the held-out cell
    # these arms fill is BFCL v4 multi-turn at FIXED step 30, not a primary-benchmark curve
    # point, so queueing curve evals for them would spend cards on a measurement no one reads.
    q2bF3)    gate=none; rw="--surface-fixed 0.5"; off=1500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32" ;;
    q2bF4)    gate=none; rw="--surface-fixed 0.5"; off=2500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32" ;;
    # q2bF5 THE CORRECTED 2B UNIFORM CONTROL (2026-08-16). EVERY 2B uniform arm above --
    # q2bF, q2bF2, q2bF3, q2bF4 -- omits LR, and verl_awm_train.sh:164 reads
    # `actor_rollout_ref.actor.optim.lr=${LR:-1e-5}`. So all four trained at 1e-5 while q2bT
    # (and q2bT2/q2bT3) trained at 1e-4: a 10x difference on the single factor the allocation
    # contrast is supposed to hold fixed. work/analysis/scale4b_mechanism.md found this while
    # explaining why 2B (+0.0289) is the only scale showing a primary-benchmark allocation gap
    # when 8B (+0.0011) and 4B (-0.0021) are both zero -- q2bF2's own window effect of -0.0030
    # is what "barely trained" looks like. The 4B and 8B contrasts are clean: q4bF/q4bT and
    # a8F/a8Fr and every 8B TRIAGE arm all carry LR=1e-4. THE 2B ROW HAS NO VALID UNIFORM
    # CONTROL UNTIL THIS ARM EXISTS, on the primary benchmark and on BFCL alike.
    #
    # EXACT CLONE OF q2bF2 WITH TWO CHANGES: the tag, and LR=1e-4. off stays 500, so q2bF5
    # draws q2bT's own surface realization and pairs with it by offset exactly as q2bF2 did --
    # the correction must not also move the seed, or it stops being the same comparison.
    # gate, rw, MODEL, LORA_RANK, LORA_ALPHA are byte-identical to q2bF2's.
    #
    # WHAT STILL DIFFERS FROM q2bT, STATED PLAINLY. q2bT additionally carries ENTCOEF=0.0
    # (a NO-OP: verl_awm_train.sh:165 defaults entropy_coeff to 0.0, so both arms train at
    # exactly 0.0) and VAL_BEFORE=True (line 106, trainer.val_before_train -- a step-0 eval
    # pass, not a training knob; it cannot change the optimizer, the data, the surface or the
    # model). Holding these at q2bF2's values keeps q2bF5 a pure clone of the F family, so the
    # q2bF5-vs-q2bF2 delta measures the LR confound and nothing else. Everything else that
    # separates the two arms is the allocation package itself: --task-alloc gradmass
    # --warm-bank default --temp 0.3 --exclude-tasks --warm-shrink group, TRIAGE_DECAY=1.0,
    # TRIAGE_EPS=0.05 -- which is the contrast.
    #
    # NOT added to queue_curve.sh, matching q2bF/q2bF2/q2bF3/q2bF4, whose parent rows it
    # replaces: the readouts this arm feeds are the 2B scale table and BFCL v4 multi-turn at a
    # fixed step, not a curve point to be chosen between.
    q2bF5)    gate=none; rw="--surface-fixed 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4" ;;
    # q2bF3e5 -- THE MIDDLE RUNG OF THE UNIFORM LEARNING-RATE LADDER (2026-08-19). Internal review
    # item 3, and the review calls the missing control "the highest-risk open confound in the
    # paper". The transfer claim rests on uniform DAMAGING transfer; but every corrected uniform
    # arm trains at 1e-4, and the one uniform arm at 1e-5 (withdrawn) scored 14.9 on BFCL --
    # close to base and far above the 1e-4 arms' 4.50-8.50. So "uniform damages transfer" is
    # presently indistinguishable from "training uniformly AT 1e-4 damages transfer", which is an
    # optimisation-stability claim rather than an allocation claim. This arm is the middle rung:
    #   LR 1e-5 -> withdrawn arm, BFCL 14.9      (exists, off-ladder)
    #   LR 3e-5 -> q2bF3e5                        (THIS ARM, the gap)
    #   LR 1e-4 -> q2bF5 / q2bF6 / q2bF7          (exist, 4.50 / 8.50 / 6.38)
    # EXACT CLONE OF q2bF5 WITH ONE TOKEN CHANGED: LR=1e-4 -> LR=3e-5. off stays 500 so it draws
    # q2bT's own surface realization and sits on the same paired ladder as every other 2B uniform.
    # Token diff vs q2bF5 is EXACTLY {--tag, LR}; verified by sweep, not asserted.
    # PRE-REGISTERED READOUT: BFCL v4 multi-turn at FIXED step 30, plus the window vs base, read
    # as a THREE-POINT TREND against the 1e-5 and 1e-4 rungs. Registered interpretation, fixed
    # before the run: monotone in LR => the headline is optimisation stability and the paper must
    # say so (review W5); flat and low across all three => uniform damages transfer at every LR at
    # which it trains, W5 dissolves, and the transfer result is an allocation result. Both
    # outcomes are reportable; neither is the "expected" one.
    q2bF3e5)  gate=none; rw="--surface-fixed 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=3e-5" ;;
    # q2bF6/q2bF7 SEEDS 2 AND 3 OF THE *CORRECTED* 2B UNIFORM CONTROL (2026-08-16). These
    # RESTORE the pre-registered three-pair 2B pooled design that the LR confound destroyed.
    # The 2026-08-15 ~22:00 pre-registration fixed the F side as q2bF2/q2bF3/q2bF4 at offsets
    # 500/1500/2500; all three trained at 1e-5 (verl_awm_train.sh:164 defaults LR when the arm's
    # `extra` omits it) against a T side at 1e-4, so every one of the three pairs contrasted
    # allocation with a 10x step-size gap folded in and the pooled test could not be run as
    # registered. q2bF5 re-earned the @500 pair. q2bF6 and q2bF7 re-earn @1500 and @2500, and
    # with them the pairing the pooled estimator consumes is whole again:
    #   (q2bT, q2bF5) @500   (q2bT2, q2bF6) @1500   (q2bT3, q2bF7) @2500
    # -- each T against an F that drew the SAME surface realization AND the same learning rate.
    # This is a repair of the registered design, not a new family, and it does not reopen the
    # 2026-08-15 ~22:20 PI directive on how the scale-transfer claim is scoped.
    #
    # EXACT CLONES OF q2bF5 -- WITH ITS LR=1e-4 -- SEED OFFSET THE ONLY KNOB. gate, rw
    # (`--surface-fixed 0.5`), MODEL, LORA_RANK, LORA_ALPHA and LR are copied
    # character-for-character from q2bF5; `off` moves to the fleet's standard second/third-seed
    # rung (a8T3gr/a8T3g2, a8T2nr, a8Tvipr, a8Fr, q2bT2/q2bT3). Contrast q2bF5's deliberate
    # off=500: that arm REPLACED a lost/invalid run at the 2B table's shared draw, while these
    # two are the added replication seeds and must draw the offsets their T partners drew.
    #
    # NOT added to queue_curve.sh, matching q2bF5 and the whole F family: the readout is BFCL v4
    # multi-turn at a FIXED step 30 with zero selection, so there is no curve point to choose
    # between. Strict parent parity, not a departure.
    q2bF6)    gate=none; rw="--surface-fixed 0.5"; off=1500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4" ;;
    q2bF7)    gate=none; rw="--surface-fixed 0.5"; off=2500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4" ;;
    # GENERALITY: the winning config (a8T5k scalars) at 2B. 2B uniform rows exist (q2bF/q2bA);
    # this is the missing half of the second-model claim. Warm bank is 8B-history; the priors
    # self-correct via decay exactly as at 8B.
    q2bT)     gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # ===== q2bTr: THE REVISION AT 2B (2026-08-27) =====
    # q2bT plus the 2B-refit concentration; with a8Tr this completes the revision at three scales.
    # TRIAGE_NU=0.4582 IS THE MEASURED VALUE: rho_by_scale.json gives 2B rho 0.6857938979 /
    # nu 0.4581640388 -> 0.4582 at the same 4 d.p. the other two use. NOTE: the commission's
    # 0.4577 was derived from the ROUNDED rho 0.686; the measured nu is 0.4582 and that is what is
    # registered, for the same reason q4bTr carries 0.3501 rather than a value recomputed from a
    # rounded rho. Boot must echo rho 0.686.
    # Pre-registered both ways against q2bT on the 2B ladder; a null prints as a null.
    q2bTr)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05 TRIAGE_NU=0.4582" ;;
    # ===== q2bK: TRIAGE v4, CALIBRATED VARIABLE-k ALLOCATION (2B pilot, 2026-08-24) =====
    # Every allocator in this project so far chooses WHICH tasks enter generation at a fixed group
    # size k=5. Fixed-k allocation is rank-equivalent family-wide (Prop 2), so it cannot express a
    # preference the ordering does not already contain -- which is exactly the reviewer's W2. v4
    # chooses (task, k_i) jointly under the SAME rollout budget: sum_i k_i = 1280 per cycle, the
    # identical generation cost q2bT pays. What makes that computable is the shipped correlation
    # calibration: P(non-degenerate | p_i, k_i) is grad_mass_group(s,f,k,nu_hat) (triage.py:454),
    # the paper's Eq. 4. So the calibration stops being only a diagnostic and becomes the thing
    # the allocator prices with -- and variable k is NOT rank-equivalent, so the claim is new.
    #
    # ONE TOKEN AGAINST q2bT: VARK=1. Everything else -- gate, --surface-fixed 0.5, --task-alloc
    # gradmass, --warm-bank default, --temp 0.3, --exclude-tasks, --warm-shrink group, off=500,
    # MODEL, LORA_RANK/ALPHA, LR=1e-4, ENTCOEF, VAL_BEFORE, TRIAGE_DECAY, TRIAGE_EPS -- is
    # byte-identical, so the pair isolates the allocator and nothing else. off=500 is deliberate
    # and is NOT a seed choice: it makes q2bK draw q2bT's own surface realization, exactly as the
    # ablations do, so the delta is the allocation rule alone.
    #
    # WHAT VARK=1 TURNS ON, END TO END. coadapt.py passes --vark to triage.py, which allocates
    # k_i in {2,3,5,8} per task and writes the column into pool_cycleN.json; prep_awm.py carries
    # it into the parquet; and surface/verl_rl/sitecustomize.py arms surface/verl_rl/vark_patch.py
    # inside the trainer AND its Ray actors, where DataProto.repeat becomes a per-row repeat.
    # site-packages is NEVER modified -- the patch is a monkeypatch from this repo, md5-pinned to
    # the verl it was derived against, and it refuses to apply to any other build.
    #
    # THE ALLOCATION IS EXACT, NOT GREEDY. Per 32-row block at a 160-rollout budget, solved by DP
    # over (task, budget-used): 32 x 160 x 4 relaxations, optimal, and -- the property that
    # matters -- it lands on the budget EXACTLY. That pin is what keeps the per-step trajectory
    # count at q2bT's 160, so batch shape, memory and the actor's ppo_mini_batch_size are all
    # unchanged and needed no patching. Measured on a realistic posterior: k histogram
    # {2:54, 3:54, 5:58, 8:90}, mean k exactly 5.000, and 34.01 vs 32.25 expected
    # gradient-carrying groups against uniform k=5 at identical cost.
    #
    # *** REGISTERED DELTA: data.shuffle. *** Under VARK the dataloader's shuffle is OFF, because
    # the per-step blocks must survive to the step for the budget pin to hold. The randomness is
    # NOT removed, it is RELOCATED: the allocator draws a fresh seeded permutation for block
    # membership every cycle (verified reproducible per seed, and different across seeds). This is
    # the one respect in which q2bK's data handling differs from q2bT's, and it is stated rather
    # than buried because it is a difference in how the same task set is ordered.
    #
    # PRE-REGISTERED READOUTS, FIXED BEFORE THE ARM RUNS:
    #   PRIMARY   window vs q2bT (same offset 500) and vs q2bV-2B, steps 15/20/25/30 vs PROBE2B.
    #   SECONDARY BFCL v4 multi-turn at step 30 with a FLOOR: >= base 15.1. Protection must not
    #             regress -- an allocator that buys gradient by destroying transfer is not a win,
    #             and this is the arm most able to do that (it can concentrate k on hard tasks).
    #   MECHANISM realized gradient-carrying groups/step, against q2bT's measured 8.93. This is
    #             the quantity v4 directly optimises, so it is the honest place to see whether the
    #             mechanism did what it claims even if the endpoint does not move.
    # v4 ENTERS THE 2B FAMILY AND PAYS THE MULTIPLICITY. A null prints as a null: if the window is
    # flat and BFCL holds the floor, the result is "variable-k buys measurably more gradient-
    # carrying groups and that does not convert", which is a real finding about the allocator and
    # is reported as one.
    q2bK)     gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05 VARK=1" ;;
    # q2bT2/q2bT3 SEEDS 2 AND 3 OF THE 2B TRIAGE ARM (2026-08-15, PI-directed power extension).
    # The T side of the same three-pair design as q2bF3/q2bF4 above; each T_i is paired with the
    # F_i sharing its offset (q2bT/q2bF2 at 500, q2bT2/q2bF3 at 1500, q2bT3/q2bF4 at 2500), so
    # the pooled test compares arms that saw the same surface draw. Offsets 1500 and 2500 are
    # the fleet's standard second/third-seed ladder (a8T3gr/a8T3g2, a8T2nr, a8Tvipr, a8Fr).
    #
    # BYTE-IDENTICAL TO q2bT IN EVERY OTHER SELECTION FIELD -- gate, rw (including --temp 0.3,
    # --warm-bank default, --exclude-tasks, --warm-shrink group), MODEL, LORA_RANK, LORA_ALPHA,
    # LR, ENTCOEF, VAL_BEFORE, TRIAGE_DECAY, TRIAGE_EPS. --tag and --seed-offset are the only
    # differences in the composed command, verified by sweep.
    #
    # NOT added to queue_curve.sh ARMS/EVAL_GATE even though the parent q2bT IS in both. The
    # pre-registered readout for these four is BFCL v4 multi-turn at a FIXED step 30 with zero
    # selection, so there is nothing for the curve queue to choose between; putting them in the
    # eval lists would spend cards generating step-by-step curve points that the protocol
    # forbids using. Same treatment as q2bF2, and for the same reason.
    q2bT2)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=1500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    q2bT3)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=2500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # ============ q2bTd: THE DECAY FIX (v8, 2026-08-16) ============
    # ONE TOKEN AGAINST q2bT: TRIAGE_DECAY 1.0 -> 0.8. Commissioned by
    # work/analysis/retention_gate.md, which was opened to test H-forget ("TRIAGE forgets solved
    # tasks where uniform rehearses them") and REFUTED it in the direction opposite to its
    # prediction -- TRIAGE re-selects already-solved tasks at 1.76-2.06x the pool base rate while
    # a8Fr never revisits a solved task at all, and TRIAGE's held-out regression rate is LOWER
    # than uniform's at all three scales (8B .142/.155, 4B .172/.254, 2B .203/.229). No retention
    # floor is registered and none is recommended: the gate measured the required floor at zero
    # (a8Fr retains at 0.845 with P(rehearsal)=0.000) and showed a perfect retention repair would
    # move the CONTROL further than the method (+0.0221 vs +0.0215).
    #
    # WHAT THE GATE FOUND INSTEAD, AND WHAT THIS ARM CHANGES. TRIAGE_DECAY=1.0 is gamma=1: no
    # decay. Against a warm bank carrying ~47 obs/task, and an expected accrual of 1.25
    # obs/task/cycle (k=5 obs when chosen, chosen w.p. 256/1023), on-policy evidence reaches
    # parity with the prior only at cycle ~38 = step ~304. The runs are 15 cycles / 120 steps, so
    # THE POSTERIOR CANNOT MOVE WITHIN THE RUN BY CONSTRUCTION -- measured drift over 7 cycles is
    # w_ess -3.7%, w_entropy -0.4%. The allocator is optimising a snapshot of a model that no
    # longer exists: it re-draws a stale ~560-task subset, spends 34% of its cycle-4 budget on
    # tasks it has ALREADY FULLY SOLVED (20% on outright zero-gradient all-success groups), and
    # reaches 520-546 distinct tasks by cycle 4 where uniform sweeps 896-905. This is a ratio
    # argument and is therefore invariant to --warm-shrink group: the shrink scales prior and
    # on-policy evidence alike. gamma is the ONLY parameter in the rule that breaks the ratio.
    #
    # WHY 0.8 AND NOT THE PLAN'S OWN 0.9. Prior weight 47*gamma^c against accumulated on-policy
    # evidence 1.25*(1-gamma^c)/(1-gamma) reaches prior/new parity at cycle 38 (gamma=1.0),
    # cycle 15 (0.9), cycle 10 (0.8), cycle 8 (0.7). The plan's v1 spec was gamma=0.9 -- silently
    # abandoned at v3g -- but 0.9 only reaches parity at cycle 15, the very END of the run, so it
    # cannot move the allocator inside the matched window. 0.8 is the first value that puts prior
    # and on-policy evidence on comparable footing while the window is still open, and it is a
    # REVERSION TOWARD the plan's own spec rather than a new term. No new flag, no new code path:
    # TRIAGE_DECAY is already exposed (coadapt.py:446 -> triage.py --decay -> a.decay).
    #
    # COLD START IS BIT-EXACT, NOT APPROXIMATELY SO. gamma enters only as a multiplier on
    # ACCUMULATED counts (triage.py:1095-1096 loops over st["counts"]), and at cycle 1 there are
    # none -- the warm bank is applied AFTER the decay, uncapped by it. So the cycle-1 draw is
    # byte-identical to q2bT's at any gamma. Verified, not asserted: the a8T3g cycle-1 pool was
    # replayed with TRIAGE_DECAY=0.8 in the environment and reproduced md5
    # 22d1228c5ca110d2b0b84874066b1c45 exactly. That replay is this arm's regression test.
    #
    # PRE-REGISTERED MECHANISM READOUT, FIXED BEFORE ANY DATA, CHECKABLE BY CYCLE 3-4 ON ONE CARD
    # AND BEFORE ANY EVAL CELL. Three numbers, each against its measured q2bT/a8T3g value:
    #   (1) share of cycle budget on already-fully-solved tasks   0.34 at c4  -> must fall < 0.15
    #   (2) cumulative distinct tasks touched by cycle 4          520-546     -> must rise > 700
    #   (3) w_ess drift over 7 cycles                             3.7%        -> must exceed 15%
    # If those three do not move, the allocator is still frozen and the arm is DEAD ON MECHANISM
    # GROUNDS -- no eval cell is spent on it and no held-out number is quoted from it. The same
    # standard the gate applied to every other arm.
    #
    # NOT added to queue_curve.sh even though the parent q2bT IS in ARMS and EVAL_GATE. Stated
    # plainly as a departure from strict parent parity: the readout above is a mechanism gate on
    # triage_stats.jsonl, and nothing downstream of it may be evaluated until that gate passes,
    # so enrolling the arm would spend eval cards on curve points the protocol forbids using.
    q2bTd)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=0.8 TRIAGE_EPS=0.05" ;;
    # ============ q2bTiid / q2bTnw: THE COMPONENT ABLATIONS (2026-08-17) ============
    # PI-directed, and NOT replication seeds -- the seed ladder is closed. The 2B head-to-head is
    # now measured and decisive (q2bT-family vs q2bF-family, +0.054..+0.073, z 4.7-6.0), so the
    # open question is no longer WHETHER TRIAGE beats uniform at 2B but WHICH OF ITS PARTS DOES.
    # TRIAGE ships two separable mechanisms on top of a gradmass allocator, and the win has never
    # been decomposed:
    #   (1) CALIBRATION  -- --warm-shrink group scores the posterior against a group model that
    #                       knows the k rollouts of a prompt are correlated, instead of an iid one.
    #   (2) THE WARM BANK -- --warm-bank default primes cycle 1 from 293k rollouts of foreign
    #                       per-task evidence, so the allocator starts informed instead of blind.
    # Each arm below removes EXACTLY ONE and keeps the other, so T-vs-Tiid is calibration's share
    # and T-vs-Tnw is the warm bank's share, both against the SAME parent on the SAME protocol.
    # 2B is the scale chosen because it is the one where the primary benchmark can resolve the
    # effect (work/analysis/scale4b_mechanism.md); at 8B and 4B the parent effect is at noise.
    #
    # BOTH ARE EXACT CLONES OF q2bT AT off=500. That offset is deliberate and is NOT a seed
    # choice: it makes each ablation draw q2bT's OWN surface realization, so the ablation pairs
    # with its parent by offset exactly as q2bF5 pairs with q2bT. An ablation on a fresh offset
    # would confound the removed component with the surface draw -- the one thing these arms
    # exist to separate. gate, --surface-fixed 0.5, --task-alloc gradmass, --temp 0.3,
    # --exclude-tasks, MODEL, LORA_RANK, LORA_ALPHA, LR=1e-4, ENTCOEF, VAL_BEFORE, TRIAGE_DECAY=1.0
    # and TRIAGE_EPS are byte-identical to q2bT's. Verified by sweep, not asserted:
    #   q2bTiid vs q2bT -- token diff is EXACTLY {--tag, --warm-shrink group -> none}
    #   q2bTnw  vs q2bT -- token diff is EXACTLY {--tag, absence of --warm-bank default}
    #
    # WHY 'none' RATHER THAN DROPPING --warm-shrink. Dropping it is not the same experiment.
    # coadapt.py:401 reads `shrink = a.warm_shrink or TRIAGE_WARM_SHRINK or ""` and forwards the
    # flag only when non-empty, so an ABSENT flag leaves the choice to an env var that a future
    # launcher could set -- an ablation whose ablated component can be silently restored from the
    # environment is not an ablation. 'none' is an explicit choices= value in BOTH parsers
    # (coadapt.py:186, triage.py:870), is truthy, and is therefore forwarded verbatim, pinning the
    # uncalibrated path in the composed command where the sweep can see it.
    #
    # COMPOSITION CHECKED AT ARGV, BOTH DIRECTIONS, BEFORE LAUNCH:
    #  * q2bTiid keeps --warm-bank, so triage.py:979 (`--warm-shrink capability` without a bank)
    #    does not fire; the arm is 'group' -> 'none', never 'capability'. triage.py:1191 then sets
    #    `nu = None`, which its own comment marks as the v1/v2 iid path, "left calling grad_mass so
    #    it stays bit-identical rather than merely equal-in-the-limit". The warm bank still primes
    #    cycle 1; only the correlated scoring of the posterior is switched off. That is precisely
    #    "allocation on uncalibrated iid posteriors" and nothing else.
    #  * q2bTnw keeps --warm-shrink group, so it clears triage.py:979 (that guard names
    #    'capability' alone -- group shrink needs only the fitted nu, not a bank). With warm_bank
    #    unset, triage.py:1110 and its 1152 elif are both skipped, no priming happens, and cycle 1
    #    has zero evidence -- the documented COLD-START COLLAPSE TO UNIFORM allocation. That
    #    collapse is the arm's first-telemetry check, not a surprise.
    #  * coadapt.py:207 gates --warm-bank/--temp/--exclude-tasks/--warm-shrink on the allocator;
    #    both arms keep --task-alloc gradmass, so both pass. Neither sets --estimator, so the
    #    taskrho guards at triage.py:1000 and coadapt.py:216 are unreachable. NO env backdoor:
    #    TRIAGE_WARM_BANK and TRIAGE_WARM_SHRINK appear nowhere in supervisor.sh or
    #    verl_awm_train.sh -- only as coadapt.py's own fallbacks -- so the flags govern.
    #
    # NOT added to queue_curve.sh ARMS/EVAL_GATE even though the parent q2bT IS in both. Same
    # treatment, and the same reason, as q2bT2/q2bT3/q2bTd: the pre-registered readout is BFCL v4
    # multi-turn at a FIXED step 30 with zero selection, so there is nothing for the curve queue to
    # choose between, and enrolling them would spend eval cards generating primary-benchmark curve
    # points the protocol forbids using.
    #
    # FOLD-IN: N/A. These are ABLATIONS FOR THE PAPER, not method candidates. They go in as
    # ablation rows regardless of sign -- a component whose removal costs nothing is a finding
    # about TRIAGE, not a failed arm -- so no fold-in gate applies and none should be invented.
    q2bTiid)  gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink none"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    q2bTnw)   gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # ---- q2bTnw2: SEED 2 OF THE NO-WARM-BANK ABLATION (2026-08-18) ----
    # work/analysis/protection_mechanism.md rests on ONE run of q2bTnw, and its central claim is
    # the strongest and most fragile thing in the file: at step 30 q2bT and q2bTnw have IDENTICAL
    # in-distribution held-out reward (0.1081 vs 0.1081, n=296 each) while OOD transfer collapses
    # from 16.62% to 2.25% -- 83.2% runaway, 52.19% of turns hitting BFCL's 20-step cap, and the
    # only arm in the grid with ZERO silent-turn failures. A claim that the warm bank buys nothing
    # in-distribution and everything out of it cannot rest on n=1. The document says so itself.
    #
    # THE SEED IDIOM, APPLIED EXACTLY. The fleet's second-seed rung is off 500 -> 1500
    # (q2bT->q2bT2, q2bF5->q2bF6; 2500 is the third, q2bT3/q2bF7). q2bTnw2 is q2bTnw at off=1500
    # and nothing else moves. Token diff vs q2bTnw is EXACTLY {--tag, --seed-offset 500 -> 1500}.
    #
    # WHY off=1500 DOES NOT VIOLATE THE ABLATION-OFFSET RULE -- READ THIS BEFORE COMPARING IT.
    # The q2bTiid/q2bTnw registration pinned off=500 on the ground that "an ablation on a fresh
    # offset would confound the removed component with the surface draw". That rule is about the
    # PAIRING, not about the number: it says an ablation must be read against a parent that drew
    # the SAME surface realization. q2bT2 is exactly that parent -- byte-identical to q2bT in
    # every selection field except off=1500 -- so at 1500 the pair (q2bT2, q2bTnw2) is the same
    # kind of object the pair (q2bT, q2bTnw) is at 500, and the delta still isolates the bank.
    # *** q2bTnw2 MUST THEREFORE BE READ AGAINST q2bT2, NEVER AGAINST q2bT. *** Reading it
    # against q2bT would reintroduce precisely the confound off=500 was chosen to avoid. Verified,
    # not asserted: the sweep confirms q2bTnw2-vs-q2bT2 has the SAME token delta as
    # q2bTnw-vs-q2bT, namely {--tag, absence of --warm-bank default} -- which is what makes this
    # a replication of the ablation rather than a new arm that resembles it.
    #
    # PRE-REGISTERED READOUT, FIXED BEFORE THE ARM RUNS. BFCL v4 multi-turn at a FIXED step 30,
    # zero selection, same harness (commit 6ea57973, task_index_sha256 cd489490...) the taxonomy
    # used. REPLICATES IFF BOTH:
    #   (1) accuracy < 8.0%  -- below the top of the corrected-uniform band. The band is
    #       q2bF5 4.50 / q2bF7 6.38 / q2bF6 8.50, so 8.0 sits just under q2bF6, the HIGHEST of
    #       the three. The number is the bar; the phrase "weakest control" is not, because it
    #       reads both ways and this must not be re-litigated after the number is in.
    #   (2) capped-turns fraction > 50%  -- the mechanism signature, matching q2bTnw's 52.19% of
    #       TURNS. That is the turn-level rate, NOT the instance-level 88.37%; the two differ by
    #       36 points and swapping them would let a failed replication pass.
    # Both are required. Accuracy alone can be low for reasons that are not runaway collapse --
    # q2bF7 scores 6.38% by crashing on 209 instances with a missing 'arguments' key, a completely
    # different failure species -- so the cap rate is what identifies the mechanism rather than
    # merely the score. Report both numbers with sign whatever they say.
    #
    # NOT added to queue_curve.sh ARMS/EVAL_GATE. Same treatment and same reason as q2bTnw itself
    # and every arm since q2bT2: a fixed-step readout with zero selection has nothing for the
    # curve queue to choose between.
    #
    # FOLD-IN: N/A, inherited from q2bTnw. This is an ablation for the paper, not a method
    # candidate; it enters as an ablation row regardless of sign and cannot displace v3g.
    q2bTnw2)  gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=1500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # ============ q2bTs / q2bFs: ALLOCATION x DATA BUDGET (2026-08-18) ============
    # The 2B head-to-head (+0.054..+0.073, z 4.7-6.0) was measured entirely on pool_max: 1127
    # tasks, 1023 after exclusions, ~256 funded per cycle. TRIAGE is an ALLOCATOR, so the budget
    # it allocates over is not a nuisance parameter -- it is the axis the method's own story
    # predicts on. If the win comes from spending a scarce budget where gradient actually exists,
    # the edge should GROW as the pool shrinks; if it comes from touching more of a large pool
    # than uniform manages, it should SHRINK. Both are live readings of the same measured win and
    # nothing in the record separates them, because every 2B arm ever run drew the same pool.
    #
    # THE PAIR IS THE UNIT. q2bTs is q2bT and q2bFs is q2bF5, each moved to pool_big320 and
    # otherwise untouched -- SAME off=500, so the small-pool pair draws the same surface
    # realization the large-pool pair drew, and the pairing that makes T-vs-F readable survives
    # the move. A T arm alone on the small pool would confound allocation with pool size; the F
    # arm is what makes the contrast a contrast. Registering one without the other is not a
    # cheaper version of this experiment, it is a different and unreadable one.
    #
    # PRE-REGISTERED READOUT, FIXED BEFORE EITHER ARM RUNS. The window contrast (q2bTs - q2bFs)
    # on pool_big320, read against the ALREADY-MEASURED (q2bT - q2bF5) contrast on pool_max, same
    # PROBE2B anchor, same cells, same fixed-step protocol. The quantity of interest is the
    # DIFFERENCE OF DIFFERENCES and its sign; a null is a publishable statement that TRIAGE's edge
    # is budget-invariant, which is itself a claim the paper currently cannot make either way.
    #
    # ONE TOKEN OF DELTA EACH, AND IT IS THE POOL. Verified by sweep, not asserted:
    #   q2bTs vs q2bT  -- token diff is EXACTLY {--tag, --pool pool_max -> pool_big320}
    #   q2bFs vs q2bF5 -- token diff is EXACTLY {--tag, --pool pool_max -> pool_big320}
    # gate, rw (--surface-fixed 0.5 and, on the T side, --task-alloc gradmass --warm-bank default
    # --temp 0.3 --exclude-tasks --warm-shrink group), off=500, MODEL, LORA_RANK, LORA_ALPHA,
    # LR=1e-4, and on the T side ENTCOEF=0.0, VAL_BEFORE=True, TRIAGE_DECAY=1.0, TRIAGE_EPS=0.05
    # are byte-identical to the parents'. The pool is set in the case at the top of train_cmd, not
    # here -- see the note there on why it is PINNED rather than left to the default.
    #
    # *** THE BUDGET DOES NOT SHRINK WITH THE POOL, AND THAT COMPRESSES THE CONTRAST. STATED
    # BEFORE ANY DATA EXISTS, NOT AFTER. *** coadapt.py:385 sets budget = steps_per_cycle * BSZ
    # = 8 * 32 = 256 rows per cycle, independent of the pool. pool_big320 is a strict SUBSET of
    # pool_max (verified: 320 of 1127, same 158 scenarios) and 13 of its tasks are on the
    # exclusion list, so these arms allocate 256 rows over 307 eligible tasks -- the allocator
    # funds ~83% of everything it can see, against ~25% (256/1023) on pool_max. An allocator that
    # must choose almost everything cannot express much of a preference, so the T-vs-F gap on this
    # pool is mechanically compressed toward zero REGARDLESS of whether the scarcity hypothesis is
    # right. The pre-registered direction (edge GROWS under scarcity) is therefore the direction
    # this design disfavours, and a null or a shrink CANNOT be read as evidence against the
    # hypothesis without separating it from budget saturation. Recorded here so the readout is
    # interpreted against the design it actually has. The data-budget manipulation itself is real
    # and intact -- 15 cycles x 256 rows over 307 tasks is ~12.5 exposures per task against ~3.8
    # on pool_max, which is the scarcity axis these arms exist to move.
    #
    # THE EXCLUSION LIST IS pool_max's AND STAYS. excluded_tasks.jsonl certifies 104 refused tasks
    # BY TASK ID; ids absent from pool_big320 simply never match, so the flag is a no-op on the
    # tasks it cannot see and identical policy on the ones it can. Dropping it would have made the
    # T-side delta {pool, exclusions} and destroyed the one-token property above.
    #
    # NOT added to queue_curve.sh ARMS/EVAL_GATE even though q2bT and q2bF5 are both in both.
    # Same treatment and same reason as every arm since q2bT2: the readout is a fixed-window
    # paired comparison with zero selection, so enrolling them would spend eval cards generating
    # curve points the protocol forbids using.
    q2bTs)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    q2bFs)    gate=none; rw="--surface-fixed 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4" ;;
    # ============ q2bTk8: THE k=8 POINT OF THE GROUP-SIZE CURVE (2026-08-18) ============
    # k is the group size the trainer generates (actor_rollout_ref.rollout.n) AND the k in
    # g(p,k)=1-p^k-(1-p)^k that TRIAGE prices tasks with, so it sits on both sides of the method:
    # raise it and every task's gradient availability rises, but the FLAT part of that curve moves
    # too, and with it the set of tasks worth funding. The curve has two points and they disagree
    # in kind rather than degree, so a third is what makes it a curve.
    #
    # THERE IS NO --k FLAG TO SET, AND SETTING NROLL IS THE WHOLE MANIPULATION. verl_awm_train.sh
    # does NROLL=${NROLL:-5} -> actor_rollout_ref.rollout.n; coadapt.py:42 reads the SAME env var
    # with the SAME default and passes it as triage.py's required --k. So NROLL=8 moves the
    # trainer and the allocator together, which is the only correct way to move it -- an arm that
    # raised rollout.n and left the allocator pricing k=5 would optimise a target it is not
    # generating. This is a8T4's idiom exactly (a8T4 set NROLL=2 and no --k anywhere).
    #
    # SET AFTER THE BATCH-32 BLOCK, NOT HERE, for the reason a8T4's NROLL is: env is later-wins,
    # and an arm whose one manipulation lives upstream of a shared block is one shared-block edit
    # away from being silently reverted. See the case below the block.
    #
    # WHERE IT DEPARTS FROM a8T4, STATED PLAINLY. a8T4 COMPUTE-MATCHED its k change (BSZ 32->80,
    # 80x2 = 32x5 = 160 rollouts/step) because it was a fixed-generation-budget comparison.
    # q2bTk8 does NOT: BSZ stays 32, so the arm generates 32x8 = 256 rollouts/step against q2bT's
    # 160, a 1.6x rise. That is deliberate and pre-registered -- the readout is the window against
    # q2bT at the SAME batch, so batch must not move -- but it means the k curve is a curve in k
    # at fixed batch, NOT at fixed generation cost, and it must be reported as such.
    # Also note a8T4 is an 8B arm (no MODEL token -> the 8B default); the k=2 point therefore sits
    # at a different scale from q2bT/q2bTk8, and only the k=5 -> k=8 leg is a within-scale contrast.
    # 256 rollouts/step at BSZ=32 is 1.6x q2bT's generation and its update-step activation
    # footprint: the first launch must be watched through cycle 1's update, not just its boot.
    q2bTk8)   gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # GENERALITY, 4B scale point: uniform + winning-config pair (PI-directed).
    q4bF)     gate=none; rw="--surface-fixed 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    q4bT)     gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # ===== q4bTpe: THE TRACE-STYLE PLUG-IN ESTIMATOR AT 4B (2026-08-26) =====
    # The 4B analogue of a8Tpe, which is a8T3g + "--estimator point" and nothing else. Same one
    # token here: q4bTpe is q4bT + "--estimator point". Everything else -- pool_max, warm bank,
    # exclusion, --temp 0.3, --warm-shrink group, off=500, MODEL, LORA, LR, ENTCOEF, VAL_BEFORE,
    # TRIAGE_DECAY, TRIAGE_EPS -- is q4bT's, because the estimator is the whole experiment.
    #
    # WHAT THE TOKEN CHANGES. triage.py's --estimator selects HOW w is computed from the SAME
    # posterior: "posterior" (our default) is E[g(p,k)] under Beta(1+s,1+f); "point" is the
    # TRACE-style PLUG-IN g(p_hat,k) at the posterior mean p_hat = a/(a+b). Identical evidence,
    # identical rule, identical budget -- only the estimator differs. That is the contrast the
    # novelty claim rests on, and this arm runs it at the CONTESTED scale.
    #
    # WHY 4B AND WHY NOW: THE DIRECT-SCOOP DEFENCE (survey Gap 3,
    # work/analysis/experiments_vs_published.md). 4B is the one cell where the published-baseline
    # grid still beats us, so "our calibrated posterior expectation beats TRACE's plug-in" is
    # exactly the claim a reviewer will test there. Running it anywhere else answers a question
    # nobody asked.
    #
    # ONE SEED, DELIBERATELY. PI: no more baseline seeds. This is a NEW BASELINE AT A NEW SCALE,
    # not a replica of an existing cell, so the seed policy that funds method seeds does not apply
    # and is not invoked.
    #
    # PRE-REGISTERED BOTH WAYS, FIXED BEFORE THE ARM RUNS:
    #   window vs q4bT's 2-seed +3.05 (and against the rest of the 4B grid);
    #   transfer BFCL step-30 vs q4bT 32.62 / VIP+bank 36.88 / faithful VIP 25.50.
    #   q4bTpe BELOW q4bT => the calibrated posterior expectation earns its place at 4B and the
    #     estimator is a real component, not a reparameterisation.
    #   q4bTpe AT OR ABOVE q4bT => the plug-in is sufficient at this scale and the estimator claim
    #     does not survive 4B. Either direction prints; neither is the hoped-for one.
    # Enters the 4B family with the usual narrated growth and pays that multiplicity.
    q4bTpe)   gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group --estimator point"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # q4bTpe2 SEED 2 OF q4bTpe (2026-08-27). q4bTpe is the framework's plug-in configuration at
    # 4B and is currently the only 1-seed row in tab:cfg4b, where every neighbour is 2-seed. This
    # makes the balanced-config row comparable to the rows it sits beside. --seed-offset 500 ->
    # 1500, the fleet's standard second-seed rung, and nothing else. Read against q4bTpe as its
    # seed pair; both report as the 2-seed mean under the unchanged convention. Method-side seed,
    # which PI policy funds. A null prints as a null.
    q4bTpe2)  gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group --estimator point"; off=1500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # q4bTf: the same faithful-TRACE configuration at 4B -- t8Tf plus only the scale tokens
    # (MODEL/LORA), so t8Tf and q4bTf are a clean scale pair and each is its scale's published
    # baseline. Pre-registered against q4bTpe (in training), faithful VIP 1.88/25.5/7.25, the
    # method +3.05/32.6/31.4 and VIP+bank +4.11/36.9/18.7. One seed, same rule.
    q4bTf)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --estimator point"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0" ;;
    # q4bTr: SCALE-ADAPTIVE CALIBRATION -- full 4B TRIAGE, correlation concentration REFIT to 4B (2026-08-25).
    # THE REVISION with a real shot at the one cell the grid still beats us. q4bT (4B) keeps EVERY
    # component (gradmass, warm bank, exclusion, warm-shrink group, sharp temp 0.3) so the window
    # stays high from exclusion; the ONE principled change is the correlation concentration.
    # WHY. The shipped global rho_hat=0.78 (nu=0.285) sits ABOVE the 95% CI at every scale (reviewer
    # W6, iclr_rereview_internal W6). It over-weights unanimous groups and mis-concentrates 4B
    # allocation, which costs transfer. The per-scale refit measured the 4B value directly:
    # work/analysis/reviewer_response_analyses.md:585 -- 4B rho_hat = 0.7407 (nu 0.3501), CI
    # [0.7172, 0.7622] over 4 arms / 6016 groups. Setting the concentration to the 4B-correct value
    # should recover transfer WITHOUT sacrificing the window (exclusion is retained).
    # THE KNOB, VERIFIED NOT ASSUMED. TRIAGE_NU already flows coadapt.py:529 -> --shrink-nu; its
    # default (triage.py:979) is 0.285 = the shipped rho=0.78, so every arm that does NOT set it is
    # byte-identical (zero-diff). triage.py:1285 binds nu = a.shrink_nu under --warm-shrink group
    # (q4bT's config), and that SAME nu drives BOTH the allocation weight (grad_mass_group, :1314)
    # AND the warm-shrink calibration -- so one override corrects rho everywhere, which is exactly
    # what "scale-adaptive calibration" requires. 1/(1+0.3501)=0.7407, matching the refit. No code
    # was added: the override is a value in this arm's extra, nothing else.
    # TOKEN DELTA vs q4bT = {--tag, +TRIAGE_NU=0.3501}. Everything else byte-identical to q4bT.
    # PRE-REGISTERED, BOTH READINGS: window vs q4bT +3.05 / VIP +4.11; transfer (BFCL step 30) vs
    # q4bT 31.6 / VIP 36.88. If it lifts transfer toward/past VIP while HOLDING the window, scale-
    # adaptive calibration is the method refinement -- a genuinely novel contribution, since the
    # calibration is already ours and this makes it scale-correct. A null prints as a null. Enters
    # the 4B family; if it wins it needs a 2nd seed before it enters the paper as the method.
    q4bTr)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05 TRIAGE_NU=0.3501" ;;
    # q4bTr2 SEED 2 OF THE REVISION -- REGISTERED RETROSPECTIVELY (2026-08-27), CONFIG ONLY.
    # This arm ALREADY RAN and completed (BFCL VERIFIED-DONE n=800, finish chain 05:32) but was
    # launched from a hand-composed logs/launch_q4bTr2.cmd and never existed as a tag here, so
    # seeds 1 and 3 were reproducible from the registry and seed 2 was not. Registering it closes
    # that gap. NOTHING IS RELAUNCHED AND NO DATA CHANGES: the banked window, cells and BFCL point
    # stand as they are; this entry only makes the configuration that produced them recoverable.
    # Verified byte-identical to the command that actually ran, modulo --jobid (which a hand
    # composition bakes to the pod it was aimed at).
    q4bTr2)   gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=1500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05 TRIAGE_NU=0.3501" ;;
    # q4bTr3 SEED 3 OF THE REVISION (2026-08-27). q4bTr's NESTFUL is SEED-UNSTABLE -- 11.34 on one
    # seed against 31.00 on the other -- so the disclosed-instability row currently rests on two
    # numbers that disagree by 20 points and cannot say which is the outlier. A third seed makes it
    # 2-of-3 and decides that directly; it is the cheapest possible resolution of the one axis the
    # row is about. --seed-offset 2500: the fleet's standard THIRD rung (a8T3g2, q2bF4, q2bF7,
    # q2bT3 all use it), chosen rather than invented -- seed 2 (q4bTr2) used 1500.
    # NOTE FOR THE RECORD: q4bTr2 ran and completed (BFCL VERIFIED-DONE n=800, finish chain done
    # 05:32) but was never registered as a tag in this file -- it exists as data and as
    # logs/launch_q4bTr2.cmd only. q4bTr3 IS registered here, so the third seed is reproducible
    # from the config the way the first is.
    # Its finish chain carries IN-CHAIN NESTFUL because NESTFUL is the axis in question; cells and
    # BFCL run too, but NESTFUL is the readout this arm exists to settle. rho 0.741 (TRIAGE_NU
    # 0.3501) is inherited unchanged -- the revision under test, not a new one.
    q4bTr3)   gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=2500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05 TRIAGE_NU=0.3501" ;;
    # ===== q4bTa: FULL TRIAGE AT 4B WITH THE SAMPLING TEMPERATURE SET BY A STATED RULE =====
    # 4B is the one cell where the grid beats us: VIP +4.53 window / 36.75 transfer and retrieval
    # +3.60 against TRIAGE ~+3.1 / 32.6. Prop 2 says the allocation RULE is rank-equivalent at
    # fixed k, so the rule is not what separates them -- temperature is the live variable, and the
    # 4B winner q4bV is the SOFT configuration (no --temp, so triage.py's default tau = 1.0) while
    # the sharp configurations win at 2B (q2bT tau=0.3) and 8B (a8T3g tau=0.5).
    #
    # THE RULE, AND THE NUMBER IT IMPLIES. Set tau so the weight distribution's ESS FRACTION
    # (ESS/pool) matches the fraction the best-performing configuration at that scale exhibits.
    # Measured from each arm's own logs: q2bT 0.4482, a8T3g 0.4625, q4bV 0.6315, q4bT 0.4427.
    # Solving on q4bT's own banked posterior (g_c with nu_hat=0.285, k=5) for a target of 0.6315
    # gives tau* = 1.12 (realised ESS fraction 0.6315, exact). Token delta vs q4bT is therefore
    # {--tag, --temp 0.3 -> 1.12} and nothing else: rule, warm bank, exclusion, warm-shrink group,
    # decay 1.0, eps, offset 500, MODEL, LORA, LR, ENTCOEF, VAL_BEFORE are all q4bT's.
    #
    # *** THE RULE DOES NOT HAVE A SINGLE CROSS-SCALE TARGET, AND THAT IS RECORDED HERE RATHER
    # THAN SMOOTHED OVER. *** The commissioning idea was that the winners share an ESS fraction.
    # They do not: the two SHARP winners cluster at 0.448/0.463 while the 4B winner sits at 0.632.
    # So there are two readings, and BOTH are pre-registered because they predict different things:
    #   (a) "match the 2B/8B sharp winners" (target 0.4554) -> tau* = 0.42, which lands within
    #       noise of q4bT's EXISTING 0.4427. Under this reading the revision is a no-op and q4bTa
    #       should reproduce q4bT's loss. It is not the reading used, and if q4bTa lands on q4bT
    #       this is the reading the data supported.
    #   (b) "match the best-performing configuration AT THAT SCALE" (target 0.6315) -> tau* = 1.12,
    #       which is what this arm runs. This is a stated, reproducible procedure but it is fitted
    #       at 4B: it needs a known winner, so it is weaker than a true scale invariant. Stated
    #       plainly because the commission explicitly asked for a rule and not a per-cell number.
    # ONE FURTHER CAVEAT: q4bV's 0.6315 is an ESS fraction under VIP's weights, not gradmass's.
    # Transferring the number across weight definitions is an assumption, not a measurement.
    #
    # PRE-REGISTERED READOUTS, BOTH DIRECTIONS, FIXED BEFORE THE ARM RUNS:
    #   q4bTa >= q4bV on BOTH window and BFCL transfer => the adaptive-temperature method leads the
    #       full 4B grid, and reading (b) is supported: the 4B gap was temperature.
    #   q4bTa below q4bV => the 4B gap is NOT temperature. Prints as measured. If it also lands on
    #       q4bT, reading (a) is what the data supported and the "winners share an ESS fraction"
    #       premise is refuted outright.
    # q4bTa ENTERS THE 4B FAMILY AND PAYS THE MULTIPLICITY. A null prints as a null.
    q4bTa)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 1.12 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # q4bTa2 SEED 2 OF THE TEMPERATURE-RULE ARM (2026-08-24) -- win-branch insurance, registered
    # while q4bTa is still running rather than after its numbers land, so the second seed is not a
    # response to the first one's result. --seed-offset 500 -> 1500 is the ONLY field that differs;
    # the fleet's standard second-seed rung, the same one q2bT2/q2bF6/q4bT2/a8T3gr/b8plrr use.
    #
    # WHY NOW. 4B is the one cell the grid still beats us in, and VIP's 4B lead rests on TWO seeds
    # (q4bV, q4bV2). If q4bTa clears those numbers with one seed the comparison is 1-v-2 and the
    # honest read is "not yet replicated"; with q4bTa2 already in flight it becomes 2-v-2 the
    # moment both land, under the PI's standing method-seeds policy. If q4bTa falls short the arm
    # costs idle cycles on an otherwise empty pod (verified 0 MiB on both cards, no drivers, no
    # compute apps) -- so the downside is cycles nobody else is using and the upside is a
    # replicated lead a day earlier.
    #
    # PRE-REGISTERED: pairs with q4bTa and is read against it as its seed pair -- NOT against
    # q4bT, and not against q4bV except through the pair mean. Both seeds report as the 2-seed
    # mean under the unchanged reporting convention. Enters the 4B family with the usual narration
    # at fold time and pays that family's multiplicity. A null prints as a null: if the pair mean
    # does not clear VIP's 2-seed numbers, the temperature rule did not close the 4B gap and that
    # is the result.
    #
    # CLASS: inherits q4bTa's A100 util 0.30 (it is staged for the 2xA100 pair at NGPUS=2, where
    # TP=2 and the offload flags are added by train_cmd). Same registered parity departure from
    # q4bT that q4bTa carries, for the same reason -- placement follows the card, not method.
    q4bTa2)   gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 1.12 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=1500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # ===== q4bTx / q4bTy: WHICH COMPONENT COSTS 4B TRANSFER (2026-08-24) =====
    # q4bTa's verdict: the temperature rule FIXED THE WINDOW (+3.78, co-leading VIP's +4.11 within
    # noise) and did NOT fix transfer (31.62 against VIP's 2-seed 36.88). Softness was the window's
    # problem; something else is transfer's. The diagnosis that motivates these two arms is
    # structural rather than a guess: VIP's winning 4B configuration has STRICTLY FEWER COMPONENTS
    # than ours (--task-alloc vip --warm-bank default, and nothing more), so if our extra machinery
    # is not paying at 4B, one of the components VIP lacks is what costs transfer.
    #
    # BOTH ARE CLONES OF q4bTa, SO tau=1.12 IS KEPT AND THE WINDOW GAIN IS PRESERVED. Each removes
    # exactly ONE component VIP does not have, which is what makes them isolation arms rather than
    # two more configurations:
    #   q4bTx  drops the CERTIFIED EXCLUSION (--exclude-tasks). Prime suspect: the 104 tasks
    #          certified verifier-unsatisfiable/pre-solved are excluded because they cannot yield
    #          a clean training signal, but "cannot be solved" is not "carries no signal" -- at 4B
    #          those tasks may still teach general tool use that transfers off-distribution. The
    #          pool goes 1023 -> 1127 for this arm; that IS the manipulation.
    #   q4bTy  drops CORRELATION SHRINKAGE (--warm-shrink group -> none, the q2bTiid idiom:
    #          'none' is an explicit choices= value, forwarded verbatim, so the independent path is
    #          pinned in the composed command where the sweep can see it rather than left to an env
    #          var a future launcher could set).
    #
    # TOKEN DIFFS, MEASURED NOT ASSERTED, AND THEY ARE NOT THE SAME SHAPE:
    #   q4bTx vs q4bTa = {--tag, DELETION of '--exclude-tasks <path>'}  (a true removal, 2 tokens)
    #   q4bTy vs q4bTa = {--tag, 'group' -> 'none'}                      (a substitution, not a
    #          removal -- the commission called both "one removal"; the shrinkage arm is a value
    #          change, because dropping the flag outright would hand the choice to TRIAGE_WARM_SHRINK)
    # Everything else -- gate, surface 0.5, gradmass, warm bank, tau 1.12, off=500, MODEL, LORA,
    # LR, ENTCOEF, VAL_BEFORE, TRIAGE_DECAY, TRIAGE_EPS -- is byte-identical to q4bTa's.
    #
    # PRE-REGISTERED READOUTS, PER ARM, FIXED BEFORE EITHER RUNS. Transfer (BFCL step 30) against
    # VIP's 2-seed 36.88 and against q4bTa's 31.62; window against q4bTa's +3.78 and VIP's +4.11.
    # INTERPRETATION FIXED IN ADVANCE: an arm that recovers transfer to >= 36 WHILE holding its
    # window >= +3.8 shows its removed component is SCALE-CONDITIONAL, and the method's 4B
    # configuration drops it -- printed as a scale rule, not as a tuning result. An arm that
    # recovers transfer but loses the window has traded one for the other and is reported as that
    # trade. A null prints as a null. Both enter the 4B family and pay its multiplicity.
    # NOTE the arms are NOT mutually exclusive: if both recover, the 4B configuration question is
    # which single removal suffices, and that needs the pair read together, not a winner declared.
    q4bTx)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 1.12 --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    q4bTy)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 1.12 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink none"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # ----- q4bTz: BOTH REMOVALS AT ONCE (2026-08-25) -----
    # q4bTx and q4bTy each remove ONE component VIP lacks. q4bTz removes BOTH, which makes its
    # component set VIP's exactly -- plus the three things we are NOT testing here and keep on
    # purpose: our gradmass rule, the warm bank, and tau=1.12. So the q4bTa -> q4bTz delta is
    # precisely "the two components VIP does not have", and nothing else moves.
    #
    # WHY A THIRD ARM RATHER THAN INFERRING IT FROM THE OTHER TWO. The single-removal arms can only
    # answer "does removing X alone recover transfer". They cannot see an INTERACTION: two
    # components can each be harmless alone and jointly cost transfer, and in that case q4bTx and
    # q4bTy both come back null while the real 4B configuration is the double removal. Without
    # this arm that outcome is indistinguishable from "neither component is the problem", which is
    # the wrong conclusion to draw from two nulls.
    #
    # TOKEN DIFF vs q4bTa = {--tag, DELETION of '--exclude-tasks <path>', 'group' -> 'none'}:
    # a deletion AND a substitution, i.e. exactly the union of q4bTx's and q4bTy's deltas.
    #
    # READ JOINTLY WITH q4bTx/q4bTy, AS ALL THREE WERE REGISTERED. Transfer vs VIP's 2-seed 36.88
    # and q4bTa's 31.62; window vs q4bTa's +3.78 and VIP's +4.11.
    #   only q4bTz recovers      => the two components INTERACT; the 4B configuration drops both,
    #                              and the single-removal nulls are explained rather than ignored.
    #   a single-removal arm and q4bTz both recover => the single removal suffices and is preferred
    #                              as the smaller change; q4bTz adds nothing and is reported as
    #                              redundant rather than as a second win.
    #   none recovers            => the 4B transfer gap is NOT in these two components, and the
    #                              diagnosis that VIP's smaller component set explains it is wrong.
    # A null prints as a null. Enters the 4B family and pays its multiplicity.
    q4bTz)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 1.12 --warm-shrink none"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # ===== ROUND 2 (2026-08-25): registered while Round 1 is still training, so no card idles =====
    # q4bTa16 -- ONE MORE STEP ALONG THE TEMPERATURE AXIS. tau 0.3 -> 1.12 bought +0.5 of window
    # (q4bT +3.1 -> q4bTa 2-seed +3.59) and bought nothing on transfer. This arm asks whether the
    # window keeps climbing past 1.12 or has turned over: 1.6 is roughly the same multiplicative
    # step again (1.12/0.3 = 3.7x, 1.6/1.12 = 1.43x -- a deliberately SMALLER step, because the
    # ESS-fraction response flattens as tau grows: MEASURED on q4bT's posterior, 0.3->1.12 moved
    # the fraction 0.427->0.632 (+0.205) while 1.12->1.6 moves it 0.632->0.738 (+0.106), half the
    # range for a comparable step, so a larger step would buy mostly saturation). Reference
    # components otherwise: exclusion ON, warm-shrink group -- so this is
    # a pure temperature move against q4bTa and is NOT confounded with Round 1's removals.
    # Token diff vs q4bTa = {--tag, --temp 1.12 -> 1.6}.
    # PRE-REGISTERED: window vs q4bTa 2-seed +3.59 and VIP +4.11; transfer vs q4bTa 31.62 and VIP
    # 36.88. Both readings fixed in advance: window still rising => the axis is not exhausted and
    # the 4B temperature is a tunable the method should state a rule for; window flat or falling
    # => 1.12 is at or past the optimum and the axis is closed. A null prints as a null. Enters
    # the 4B family.
    #
    # q4bTz2 -- SEED 2 OF THE DOUBLE-REMOVAL ARM, win-branch insurance registered BEFORE Round 1
    # reports, so it is not a response to a good result. If q4bTz wins Round 1 it needs two seeds
    # to stand against VIP's 2-seed numbers anyway; funding it now makes the comparison 2-v-2 the
    # moment both land instead of a day later. If q4bTz loses, the cost is cycles on a card that
    # would otherwise idle between rounds. --seed-offset 500 -> 1500 is the ONLY field that
    # differs from q4bTz. Reads as q4bTz's seed pair; the two report as a 2-seed mean under the
    # unchanged convention. Enters the 4B family.
    q4bTa16)  gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 1.6 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    q4bTz2)   gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 1.12 --warm-shrink none"; off=1500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # ---- q4bTnw: THE WARM-BANK ABLATION AT 4B (2026-08-18) ----
    # SCALE-GENERALITY OF THE PROTECTION MECHANISM. work/analysis/protection_mechanism.md
    # establishes at 2B that removing the warm bank does not merely delete TRIAGE's gain but opens
    # a specific hole: 83.2% runaway, 52.19% of turns hitting BFCL's 20-step cap, zero silent-turn
    # failures, and -- the sharpest part -- IDENTICAL in-distribution held-out reward (0.1081 vs
    # 0.1081). The bank buys nothing on the training distribution and everything off it. That is a
    # claim about what allocation protects, and a claim of that shape is worth much more if the
    # protection is not a 2B artifact. q4bTnw is the same manipulation one scale up.
    #
    # EXACT CLONE OF q4bT MINUS THE BANK. Token diff vs q4bT is EXACTLY
    # {--tag, absence of --warm-bank default} -- and, verified by sweep rather than asserted, that
    # is the SAME token delta q2bTnw carries against q2bT, character for character. That identity
    # is the point: it makes this a scale replicate of a known manipulation rather than a new arm
    # that resembles one. gate, --surface-fixed 0.5, --task-alloc gradmass, --temp 0.3,
    # --exclude-tasks, --warm-shrink group, off=500, MODEL=Qwen/Qwen3-VL-4B-Instruct, LORA_RANK,
    # LORA_ALPHA, LR=1e-4, ENTCOEF=0.0, VAL_BEFORE=True, TRIAGE_DECAY=1.0 and TRIAGE_EPS=0.05 are
    # byte-identical to q4bT's.
    #
    # *** PAIRING RULE, PINNED: READ q4bTnw AGAINST q4bT AND AGAINST NOTHING ELSE. *** It takes
    # q4bT's OWN off=500, so it draws q4bT's own surface realization and the delta isolates the
    # bank. q4bT2 exists at off=1500 and is NOT the comparator for this arm; reading q4bTnw
    # against q4bT2, or against any 8B or 2B arm, reintroduces exactly the surface-draw confound
    # off=500 was chosen to remove. This is the same rule q2bTnw carries at 2B.
    #
    # PRE-REGISTERED READOUT, FIXED BEFORE THE ARM RUNS. TWO PARTS, BOTH REPORTED WITH SIGN.
    #  (a) PRIMARY WINDOW vs q4bT, gains vs the PROBE anchor in pp at steps 15/20/25/30, the
    #      established paired protocol. q4bT's own reference window is +1.2/+3.2/+2.2/+3.2.
    #  (b) BFCL v4 multi-turn at FIXED step 30 WITH THE MECHANISM SIGNATURE. *** PREDICTION
    #      REGISTERED BEFORE THE RUN, NOT AFTER: *** if the 2B protection account generalises,
    #      q4bTnw collapses BELOW the 4B uniform controls AND shows instance-level cap-rate > 50%.
    #      Both parts are required, for the reason the 2B registration gives: a low score alone
    #      does not identify runaway collapse (q2bF7 scored 6.38% by crashing on a missing
    #      'arguments' key, a different failure species), so the cap rate is what names the
    #      mechanism rather than merely the damage.
    #      NOTE THE METRIC LEVEL, IT DIFFERS FROM q2bTnw2's ON PURPOSE: this arm is registered on
    #      the INSTANCE-level cap rate (q2bTnw's was 88.37%), where q2bTnw2 is registered on the
    #      TURN-level rate (52.19%). The two differ by 36 points on the same arm. Each readout is
    #      bound to its own level here so neither can be quietly swapped for the other later.
    #  A partial result -- one criterion met, the other not -- is reported as partial, and is not
    #  rounded toward either verdict.
    #
    # NOT added to queue_curve.sh ARMS/EVAL_GATE. Same treatment and reason as q2bTnw and every
    # arm since q2bT2: fixed-step readouts with zero selection give the curve queue nothing to
    # choose between.
    #
    # FOLD-IN: N/A, inherited from q2bTnw -- an ablation for the paper, not a method candidate. It
    # enters as an ablation row regardless of sign and cannot displace v3g.
    q4bTnw)   gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # q4bT2/q4bF2 SEED 2 OF THE 4B TRANSFER PAIR (2026-08-16). q4bT vs q4bF is the paper's
    # HEADLINE transfer result, and it currently rests on ONE draw of the surface and of the data
    # order -- the same single-pair exposure the 2B power extension was opened to fix. These two
    # arms replicate the pair at a second seed so the 4B claim is a pooled two-pair test rather
    # than one run against one run.
    #
    # OFFSET 1500 ON BOTH, the fleet's standard second-seed rung (a8T3gr, a8T2nr, a8Tvipr, a8Fr,
    # and the 2B power seeds q2bT2/q2bF3). BOTH sides take the SAME new offset, exactly as the 2B
    # protocol pairs (q2bT2, q2bF3) at 1500: the pairing the pooled test consumes is by offset, so
    # T2 and F2 must draw the same surface realization or the pair stops being matched.
    #
    # BYTE-IDENTICAL TO THEIR PARENTS IN EVERY OTHER SELECTION FIELD -- gate, rw (including the T
    # side's --task-alloc gradmass, --warm-bank default, --temp 0.3, --exclude-tasks and
    # --warm-shrink group), MODEL=Qwen/Qwen3-VL-4B-Instruct, LORA_RANK, LORA_ALPHA, LR, ENTCOEF,
    # VAL_BEFORE, and on the T side TRIAGE_DECAY/TRIAGE_EPS. --tag and --seed-offset are the only
    # differences in the composed command, verified by sweep.
    #
    # NOT added to queue_curve.sh ARMS/EVAL_GATE even though BOTH parents are in both lists. Same
    # departure from parent parity, and the same reason, as q2bT2/q2bF3: the pre-registered
    # readout for these seeds is BFCL v4 multi-turn at a FIXED step 30 with zero selection, so
    # there is nothing for the curve queue to choose between, and enrolling them would spend eval
    # cards generating primary-benchmark curve points the protocol forbids using.
    q4bF2)    gate=none; rw="--surface-fixed 0.5"; off=1500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    q4bT2)    gate=none; rw="--surface-fixed 0.5 --task-alloc gradmass --warm-bank default --temp 0.3 --exclude-tasks $R/work/analysis/excluded_tasks.jsonl --warm-shrink group"; off=1500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True TRIAGE_DECAY=1.0 TRIAGE_EPS=0.05" ;;
    # ============ THE 8B BASELINE TABLE, REPLICATED AT 2B AND 4B (2026-08-14) ============
    # PLAN section 5b item 3 asks "is the effect 8B-specific?", and q2bT/q4bT vs q2bF/q4bF only
    # answer it against UNIFORM. A scale claim that beats uniform but was never run against the
    # published alternatives at that scale is the same reviewer question one model down. So each
    # 8B baseline row gets its 2B and 4B twin, built from the 8B case's rw + the scale's extras:
    #   ?bD  DAPO            <- dapo)   : TRAIN_SCRIPT agent wrapper + DAPO_MAX_GEN_BATCHES=12
    #   ?bP  PLR (variance)  <- b8plr)  : --reweight variance
    #   ?bR  tool retrieval  <- b8ret)  : --surface-retrieval --retrieval-match-level 0.5
    #   q2bN band filter     <- a8T2n)  : --task-alloc bandfilter --warm-bank default
    #
    # WHAT IS DELIBERATELY NOT CLONED: the 8B memory package. dapo/b8ret carry TP=2 +
    # PARAM_OFFLOAD/OPT_OFFLOAD (+EAGER) because an 8B will not fit one 80G A100 beside its
    # trainer and a 28672-token KV cache. A 2B/4B on one card has no such wall, and TP is set
    # from the SLOT by the guard below in any case -- a hardcoded TP=2 on a single-GPU promotion
    # is the fault that gave b8accel 0 episodes. MAXTOK/BSZ come from the batch-32 block.
    #
    # SEED OFFSET 500, not dapo's 0. Every q2b*/q4b* row and every 8B row except dapo uses 500;
    # the offset picks which handicapped-surface realization the arm trains on, so a table whose
    # rows disagree on it measures the surface draw. The scale table is internally matched.
    #
    # LR=1e-4 ENTCOEF=0.0 on all seven for the same reason the dapo comment below gives: the
    # allocation comparison is only clean if every row trains at the same LR on the same surface.
    # VAL_BEFORE=True gives each row the step-0 anchor on the fleet-wide validation instrument.
    q2bD)     gate=none; rw="--surface-fixed 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True AWM_RL_SRVDIR=/tmp/awm_srv_q2bD TRAIN_SCRIPT=$BRACE_ROOT/slurm/verl_awm_train_agent.sh DAPO_MAX_GEN_BATCHES=12 MINIBSZ=4" ;;
    q2bP)     gate=none; rw="--surface-fixed 0.5 --reweight variance"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    q2bR)     gate=none; rw="--surface-retrieval --retrieval-match-level 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True AWM_RL_SRVDIR=/tmp/awm_srv_q2bR" ;;
    q2bN)     gate=none; rw="--surface-fixed 0.5 --task-alloc bandfilter --warm-bank default"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # ============ THE BASELINE GRID, 2B AND 4B: VIP AND LEARNING-PROGRESS (2026-08-19) ============
    # PI directive: the main comparison table must cover every baseline at every scale. Audited
    # first rather than built first, and MOST OF THE GRID WAS ALREADY THERE -- see the PLAN entry.
    # PLR (--reweight variance) and RETRIEVAL (--surface-retrieval) already exist at both scales as
    # q2bP/q4bP and q2bR/q4bR, all four with COMPLETE 15/20/25/30 windows on disk. Only two
    # methods were genuinely missing at small scale, and these four arms are exactly those cells:
    #
    #   method                 8B template   2B          4B
    #   VIP variance alloc     a8Tvip        q2bV        q4bV
    #   learning progress/TSCL a8Tlp         q2bLp       q4bLp
    #
    # TAG CHOICE IS NOT COSMETIC HERE. The directive proposed q2bL/q4bL for TSCL, but **q2bL IS
    # ALREADY TAKEN** by the learnable-band arm (--surface-fixed 0.0 on pool_learnable, a
    # completely different method), so q2bL would have collided with a live registration and
    # q4bL would have read as its 4B sibling. Both TSCL arms therefore take the unambiguous
    # `Lp` suffix, which also matches the 8B parent's own name (a8T**lp**). q2bV/q4bV were
    # verified free everywhere -- supervisor.sh, .supervisor_state, fleet_slots, train_queue,
    # work/ dirs and logs/*.cmd -- before being claimed.
    #
    # THE VIP ARMS DROP RESUME_LOAD, AND THAT IS THE ONE DELIBERATE DELTA FROM THE 8B PARENT.
    # a8Tvip carries `RESUME_LOAD=model_extra`, which is NOT part of the VIP method: it is scar
    # tissue from that run's own history -- a8Tvip was preempted at step 15 and a scratch cleanup
    # pruned its optimizer/extra_state shards, and the case comment above records three resume
    # attempts dying on the same FileNotFoundError. The clean VIP template is the one the fleet
    # itself already wrote down: a8Tvipr (seed 2) carries NO RESUME_LOAD. Copying it to a fresh
    # small-scale arm would point a brand-new run at a resume path it has no checkpoint for.
    # Dropping it is what makes these arms log episodes normally from step 1, which is the
    # behaviour the directive asked for and is simply the default once the token is gone.
    #
    # SCALE CONVENTIONS TAKEN FROM THE EXISTING SIBLINGS, NOT INVENTED. 2B: MODEL 2B +
    # LORA_RANK/ALPHA=32 + LR=1e-4 (q2bT/q2bF5/q2bP/q2bN). 4B: MODEL 4B + LORA_RANK/ALPHA=32 +
    # LR=1e-4 (q4bT/q4bP/q4bR). ENTCOEF=0.0 and VAL_BEFORE=True as on every allocator baseline.
    # The sharpest check available is that q2bN is the SAME SHAPE as these arms -- an allocator
    # baseline with a warm bank -- so q2bV vs q2bN and q2bLp vs q2bN must each reduce to exactly
    # {--tag, the --task-alloc rule}. Verified by sweep, not asserted.
    #
    # PRE-REGISTERED READOUT: per-scale window against THAT SCALE'S uniform control (2B: q2bF5;
    # 4B: q4bF), steps 15/20/25/30 vs the scale's PROBE anchor, paired on (scenario, task_idx,
    # seed). Each arm joins its own scale's existing family. **FAMILY GROWTH IS A DELIBERATE
    # FINAL-EMIT DECISION**: adding arms to a pinned family changes the multiplicity correction
    # every member is judged under, so these four enter the 2B and 4B families only at final emit
    # and the Holm correction is recomputed over the family as it stands THEN, not retro-fitted
    # to earlier reported numbers. That is a decision to be recorded at emit, not smuggled in by
    # registration.
    q2bV)     gate=none; rw="--surface-fixed 0.5 --task-alloc vip --warm-bank default"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    q2bLp)    gate=none; rw="--surface-fixed 0.5 --task-alloc progress --warm-bank default"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    q4bV)     gate=none; rw="--surface-fixed 0.5 --task-alloc vip --warm-bank default"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # q4bTvip: PURE VIP AT 4B -- the VIP recipe base, our codepath, NO bank/exclusion/shrink (2026-08-25).
    # === WHY THIS IS q4bV MINUS THE BANK, NOT q4bV, AND WHY THE OTHER TWO ROUND-3 ARMS ARE NOT HERE ===
    # The round-3 brief assumed q4bV is "pure VIP, no bank" and asked to clone it three ways. q4bV's
    # OWN boot log disproves that assumption: `rule=vip, warm-bank default ... warm start 1127/1127
    # primed from 135 foreign logs`. q4bV IS vip + warm bank; the bank is consumed (priming feeds the
    # (s,f) counts that vip_variance reads). So, against the code as it actually is:
    #   * q4bTvip = "VIP with no bank" = q4bV with --warm-bank DELETED. THIS is the genuinely new arm
    #     (the published VIP baseline carries no bank), so it is the real reproduction/no-bug check and
    #     the true base of the build-up. Token diff vs q4bV = {--tag, minus --warm-bank default}.
    #   * q4bVb "= VIP + bank" would BE q4bV byte-for-byte (q4bV already has the bank) -> a duplicate,
    #     NOT registered. The "does the bank help VIP" contrast is exactly q4bTvip (no bank) vs q4bV
    #     (bank), which this registration creates without a third run.
    #   * q4bVe "= VIP + exclusion" is a HARD FATAL: triage.py:1081 refuses `--rule vip` with
    #     --exclude-tasks ("VIP's L<=n_q<=U, L>=3 cannot skip a prompt; a VIP that refuses tasks is not
    #     VIP"). NOT registered; reported for a PI decision (a pre-filtered pool is a DIFFERENT
    #     mechanism and changes what "VIP" means, so it is not taken unilaterally).
    # PRE-REGISTERED READOUT: transfer (BFCL step 30) must land ~36.88 to confirm our harness
    # reproduces published VIP; window vs the 4B PROBE anchor. Read against q4bV as the bank ablation
    # on VIP (q4bV - q4bTvip = the bank's effect). Enters the 4B family; a null prints as a null.
    q4bTvip)  gate=none; rw="--surface-fixed 0.5 --task-alloc vip"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    # q4bV2 SEED 2 OF q4bV (2026-08-23). q4bV's window came in at +4.53, which is a SINGLE-SEED
    # OUTLIER against VIP's own cross-scale band -- 8B seeds +1.5 / +2.3 and 2B +1.95 -- and every
    # ordering inside the 4B block is sub-MDE, so nothing there can currently be called a lead.
    # One draw cannot distinguish "VIP genuinely leads at 4B" from "this seed drew high", and the
    # 4B row is the one place the baseline grid would otherwise print an unreplicated outlier as a
    # result. --seed-offset 500 -> 1500 is the ONLY field that differs, the fleet's standard
    # second-seed rung (q2bT2/q2bF6/q4bT2/a8T3gr/b8plrr all use it).
    #
    # PAIRING: this arm IS the seed pair, so it is read against q4bV directly. The same-offset
    # pairing rule that governs the ablations (q2bTnw vs q2bT, a8Tme vs a8T3g) does NOT apply here
    # and must not be imported -- those pair an ablation to a parent that drew the SAME surface;
    # this pairs two draws of the SAME arm to measure the draw itself.
    #
    # PRE-REGISTERED READOUT, BOTH READINGS FIXED BEFORE THE ARM RUNS, AND THE ROW PRINTS EITHER
    # WAY: window vs the scale's PROBE anchor at steps 15/20/25/30, reported as the q4bV/q4bV2
    # pair and their mean.
    #   regresses toward the band (roughly +1.5..+2.3) => +4.53 was a draw, the 4B VIP cell is
    #       in family with 8B and 2B, and the outlier is retired as such;
    #   holds near +4.5                                => VIP genuinely leads at 4B and that
    #       becomes a real cross-scale finding rather than a seed artifact.
    # Neither outcome is the hoped-for one; the 4B VIP row is printed with both seeds and their
    # spread regardless of which way it falls, because a baseline grid that only prints replicated
    # cells when they agree is not a grid.
    q4bV2)    gate=none; rw="--surface-fixed 0.5 --task-alloc vip --warm-bank default"; off=1500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    q4bLp)    gate=none; rw="--surface-fixed 0.5 --task-alloc progress --warm-bank default"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    q4bD)     gate=none; rw="--surface-fixed 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True AWM_RL_SRVDIR=/tmp/awm_srv_q4bD TRAIN_SCRIPT=$BRACE_ROOT/slurm/verl_awm_train_agent.sh DAPO_MAX_GEN_BATCHES=12 MINIBSZ=4" ;;
    q4bP)     gate=none; rw="--surface-fixed 0.5 --reweight variance"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True" ;;
    q4bR)     gate=none; rw="--surface-retrieval --retrieval-match-level 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-4B-Instruct LORA_RANK=32 LORA_ALPHA=32 LR=1e-4 ENTCOEF=0.0 VAL_BEFORE=True AWM_RL_SRVDIR=/tmp/awm_srv_q4bR" ;;
    q2bfix)   gate=none; rw="--surface-fixed 0.5"; off=500; extra="MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_RANK=32 LORA_ALPHA=32" ;;
    # ================= THE THREE PUBLISHED-ALTERNATIVE BASELINES (2026-08-09) =================
    # b8plr/b8k16/dapo answer "other ways to raise availability inside GRPO". These answer the
    # harder question: "other ways to do what ATSC does at all", one per family.
    #
    #   b8ret    TOOL RETRIEVAL. Advertise the top-k tools most RELEVANT to the task (BM25 over
    #            name+description), which is what a deployed MCP agent does. Budget-matched
    #            PER SCENARIO to a8F's level-0.5 surface: 4380 name-slots over 158 scenarios,
    #            verified 0 per-scenario mismatches against work/coadapt_a8F/surface_map.json.
    #            Without that match the arm would measure surface SIZE, not selection rule.
    #   b8accel  ACCEL (Parker-Holder 2022): regret-prioritised archive + undirected mutation of
    #            the SAME per-scenario level dial, same 0.25 step. PLR (b8plr) only selects among
    #            levels; ACCEL edits them, which is structurally what ATSC does, so this is the
    #            closest published relative and the one the reviewer will ask for.
    #   b8dense  PARTIAL-CREDIT REWARD. Same fixed 0.5 surface as a8F/b8plr/b8k16 -- attacks the
    #            bimodality from the REWARD side instead of the environment side.
    #
    # MEMORY, measured the hard way on 2026-08-09 over three launches of b8dense on the 2xA100
    # pod. None of these are learning hyperparameters -- they are budget and placement, so the
    # arms stay comparable to a8F, which runs the same recipe on a 143GB H200 where none of it
    # binds. An 8B arm at surface level 0.5 does NOT fit an 80GB A100 the way b8van does at
    # level 0.0: 4380 advertised names against 3284 means materially longer prompts.
    #   0.45, MAXTOK=20480 -> died in vLLM wake_up: "CUDA Error: out of memory at
    #                        cumem_allocator.cpp:112". vLLM sleeps through the actor update and
    #                        re-maps its KV cache on wake, by which time the trainer holds the
    #                        memory it used to have.
    #   0.38, MAXTOK=20480 -> cleared the wake_up and completed step 1, then asserted on
    #                        max_seq_len=21806 (see MAXTOK below).
    #   0.38, MAXTOK=28672 -> assert gone, wake_up OOM back: the larger per-GPU token budget
    #                        raises the actor's peak, so the trainer holds more at wake time.
    # Hence 0.35 AND FSDP offload. Offloading params/optimizer to host between steps is the
    # targeted fix -- it removes the trainer's resident footprint at exactly the moment vLLM
    # needs it back -- and it changes where state lives, never what the update computes.
    #
    # TP=2 MAXTOK=28672 are NOT optional on the 2-GPU A100 pod and are pinned here rather than
    # left to a launch command, because that is the difference a supervisor restart silently
    # erases: without TP=2 vLLM cannot fit a 3.94 GiB KV cache at max_model_len 28672.
    #
    # MAXTOK IS THE ARCHITECTURAL BOUND, NOT A MEASURED ONE. verl asserts
    # `max_token_len >= max_seq_len` on EVERY optimizer step, so any value below
    # PROMPT_LEN+RESP_LEN is a run that dies the first time a long episode appears -- which is
    # exactly how `dapo` banked 26k episodes and zero checkpoints at MAXTOK=10240. 20480 was
    # picked to clear an observed max of 18031 and it still failed here: b8dense died at step 1
    # with "max_token_len=20480 and max_seq_len=21806". An empirical ceiling is not a bound, it
    # is a bet on the tail. The agent loop caps the prompt at PROMPT_LEN (16384; it raises rather
    # than truncates) and the response at RESP_LEN (12288), so 28672 is the largest sequence that
    # can EXIST and the assert becomes unreachable. AWM_RL_SRVDIR keeps the hot SQLite churn on
    # node-local disk instead of the shared /project that fills and kills.
    b8ret)    gate=none; rw="--surface-retrieval --retrieval-match-level 0.5"; off=500;
              extra="LR=1e-4 ENTCOEF=0.0 TP=2 MAXTOK=28672 PARAM_OFFLOAD=True OPT_OFFLOAD=True AWM_RL_SRVDIR=/tmp/awm_srv_b8ret" ;;
    b8accel)  gate=none; rw="--surface-accel"; off=500;
              extra="LR=1e-4 ENTCOEF=0.0 TP=2 MAXTOK=28672 PARAM_OFFLOAD=True OPT_OFFLOAD=True AWM_RL_SRVDIR=/tmp/awm_srv_b8accel" ;;
    # AWM_DENSE_REWARD must reach the RAY WORKERS, not just this shell: the agent loop runs inside
    # them and verl_awm_train.sh forwards a fixed variable list precisely because inheritance
    # through the raylet is not trusted. verl_awm_train_agent.sh is the only script that pushes it
    # into ray_init.runtime_env, so this arm goes through it with DAPO_FILTER_GROUPS=0 -- dynamic
    # sampling off, so the only difference from a8F is the reward. "gates", not "1": dense_reward
    # accepts telemetry|gates and treats everything else as disabled, so the retired `gatesd`
    # arm below trained on the plain binary reward while its tag said otherwise.
    b8dense)  gate=none; rw="--surface-fixed 0.5"; off=500;
              extra="LR=1e-4 ENTCOEF=0.0 TP=2 MAXTOK=28672 PARAM_OFFLOAD=True OPT_OFFLOAD=True AWM_RL_SRVDIR=/tmp/awm_srv_b8dense AWM_DENSE_REWARD=gates DAPO_FILTER_GROUPS=0 TRAIN_SCRIPT=$R/slurm/verl_awm_train_agent.sh" ;;
    # =========================================================================================
    gatesd)   gate=none; rw="--surface-fixed 0.0"; extra="AWM_DENSE_REWARD=gates TRAIN_SCRIPT=$R/slurm/verl_awm_train_agent.sh DAPO_FILTER_GROUPS=0" ;;
    atscfix)  gate=none; rw="--surface-fixed 0.5" ;;
    atscbase) gate=none; rw="--surface-fixed 0.0" ;;
    temp12)   gate=none; rw="--surface-fixed 0.0"; extra="ROLLOUT_TEMP=1.2" ;;
    #   dapo   DYNAMIC SAMPLING. 26,067 episodes and ZERO checkpoints: it died on
    #          max_token_len=10240 < max_seq_len (the assert that killed six arms) and on
    #          "15 % 10 != 0", a filtered batch not dividing its mini-batch. MAXTOK is now the
    #          architectural bound and MINIBSZ=4 divides the batches filtering actually leaves.
    #          It is named in elsa.py as the "filter after generating" special case, so the
    #          paper asserts something about it that no result currently supports.
    #          SURFACE 0.5, not 0.0: dapo is the direct competitor to a8T (TRIAGE) -- filter
    #          degenerate groups AFTER generating vs allocate BEFORE. That comparison, and the
    #          one against a8F, is only clean if all three train on the SAME fixed surface at
    #          the same LR; 0.0 would have re-introduced the surface confound the whole
    #          comparability effort exists to remove.
    dapo)     gate=none; rw="--surface-fixed 0.5"; extra="LR=1e-4 ENTCOEF=0.0 TRAIN_SCRIPT=$BRACE_ROOT/slurm/verl_awm_train_agent.sh DAPO_MAX_GEN_BATCHES=12 MAXTOK=28672 MINIBSZ=4 TP=2 PARAM_OFFLOAD=True OPT_OFFLOAD=True EAGER=True" ;;
    coadapt2) off=500 ;;
    coadapt3) off=900 ;;
    atsc|atsc2|atsck8) gate=none; rw="--surface-control" ;;
    atscfix)  gate=none; rw="--surface-fixed 0.5" ;;
    atscbase) gate=none; rw="--surface-fixed 0.0" ;;
    temp12)   gate=none; rw="--surface-fixed 0.0"; extra="ROLLOUT_TEMP=1.2" ;;
  esac
  # Card 80 (h200b) will not hold a 0.45 vLLM budget alongside its trainer -- it starved the KV
  # cache for fixbase repeatedly and again for coadaptw. Per-arm budget rather than one global.
  local util=0.45 rlen=12288 ncyc=3 spc=25
  case "$tag" in atsc|atsc2|atsck8|gatesatsc|full) ncyc=9; spc=8;; esac
  # BSZ=32: the change that decides whether any of this can work. Usable gradient per step is
  # (train_batch_size x gradient availability). At BSZ=4 that is 0.3 groups/step at baseline and
  # 0.8 under ATSC -- BOTH below one, so most steps had no gradient and training reward moved
  # +0.002 over 61 steps on every arm. We spent the night raising availability 7.5%->20% while the
  # other factor was pinned at 4. At BSZ=32 it becomes 2.4 vs 6.4, a material difference, and the
  # ATSC claim is testable for the first time. MINIBSZ=8 keeps peak memory where it was.
  #
  # Horizon drops 62 -> 15 cycles (120 steps): each step now costs 8x the rollouts, so 120 steps at
  # BSZ=32 is ~4x the DATA of 496 steps at BSZ=4, and reachable in days rather than weeks.
  # KEEPCKPT=12 so a curve point survives long enough for an eval card to merge it -- verl keeps
  # ONE by default, which is how step 55 of atsc was caught mid-deletion.
  case "$tag" in atsc|atscfix|atscL2|atscL3|atscfixL2|dapo|q2b|q2bfix|q2bL|q2bL2|q2bM|q2bA|q2bF|q2bF2|q2bF3|q2bF4|q2bF5|q2bF3e5|q2bF6|q2bF7|q2bT|q2bTr|t2bTf|q2bVf|q2bK|q2bT2|q2bT3|q2bTd|q2bTiid|q2bTnw|q2bTnw2|q2bTs|q2bFs|q2bTk8|q2bD|q2bP|q2bR|q2bN|q2bV|q2bLp|q4bV|q4bTvip|q4bV2|q4bLp|q4bF|q4bF2|q4bT|q4bTpe|q4bTpe2|q4bTf|q4bTr|q4bTr2|q4bTr3|q4bTa|q4bTa2|q4bTx|q4bTy|q4bTz|q4bTa16|q4bTz2|q4bTnw|q4bT2|q4bD|q4bP|q4bR|a8A|a8F|a8Fr|b8plr|b8plrr|b8k16|b8van|b8match|a8A2|a8A3|a8Az0|a8Auni|a8Ae0|b8m25|a8C|a8Cu|b8k16v|a8T|a8Tg1|a8Te0|a8T2|a8T2w|a8T2s|a8T2r|a8T2g1|a8T2g1r|a8T2b|a8T2n|a8T2nr|a8T3|a8T3g|a8Tr|t8Tf|a8Tvipf|a8Tme|a8Tme2|a8T3gr|a8T3g2|a8T4|a8T5|a8T5k|a8T5kr|a8T6d|a8T7w|a8T8c|a8Tvip|a8Tvipr|a8Tvipc|a8Tlp|a8Tpe|b8ret|b8accel|b8dense)
      # MAXTOK is the ARCHITECTURAL bound (PROMPT_LEN+RESP_LEN), not an observed maximum.
      # verl asserts max_token_len >= max_seq_len every optimizer step, so a value tuned to
      # the longest sequence seen so far fails the first time a longer one appears. 20480 was
      # picked to clear an observed 18031 and killed b8van at step 15 tonight (7 asserts, 4
      # cycle retries) exactly as it killed six arms before. Set here, after the per-tag
      # extra, so it holds for every batch-32 arm and cannot be forgotten on a new one.
      #
      # VALIDATION, FLEET-WIDE. The paper needs a validation curve that is comparable across the
      # method, its ablations and every baseline, and "comparable" is doing real work here: arms
      # train on DIFFERENT surfaces, so a curve read off training reward would rank them by how
      # many tools their gate advertised, not by what any policy learned. AWM_VAL_ADVERTISED
      # pins every arm's validation rollouts to ONE fixed surface -- coadapt's advertised_init.txt,
      # byte-identical across arms (md5 e1ef6cea27bc) -- on a held-out pool with zero overlap with
      # pool_max, at temperature 0. That is the same instrument for everyone.
      #
      # Set HERE, in the shared block, not per arm. Per-arm was how b8k16v ended up the only arm
      # with a validation number, which would have put it ahead of the whole eval queue on an
      # ordering key no other arm had. An arm added tomorrow gets this without anyone remembering.
      # Use the canonical coadapt_coadapt path, never the arm's own advertised.txt: the gate
      # rewrites that file as it adapts, so it would silently stop being the same instrument.
      # Cost is measured, not estimated: timing_s/testing = 675s for a 295-task pass at 2B, and
      # it sits outside marked_timer("step"), so ~6.5-9.5% added wall-clock at VAL_FREQ=5 on 8B.
      # VAL_FREQ matches SAVE_FREQ=5, so every checkpoint the eval queue can pick up has a
      # validation number at exactly its own step -- which is what makes "evaluate best-validation
      # checkpoint first" a real ordering rather than an interpolation.
      ncyc=15; spc=8; extra="$extra BSZ=32 MINIBSZ=8 KEEPCKPT=12 MAXTOK=28672 VAL_FREQ=5"
      extra="$extra AWM_VAL_ADVERTISED=$R/work/coadapt_coadapt/advertised_init.txt";; esac
  # a8T/a8Tg1 (2xA100): the update-step OOM GROWS with training -- 10.02 -> 10.17 -> 10.41 GiB
  # across three failures -- because TRIAGE concentrates on harder tasks with longer episodes.
  # Shaving vLLM util chases a moving target; MINIBSZ=4 + offload changes the memory class
  # instead (same BSZ=32 via grad accumulation, identical gradient; ~10% slower steps; the same
  # config dapo runs). AFTER the shared block so this MINIBSZ overrides its MINIBSZ=8.
  # a8T2n is A100-class like a8T2r (it lands on a 2xA100 pod), so it takes the offload block and
  # the 0.30 util below, NOT a8T2's H200 0.35. Same for the three published-baseline arms
  # (a8Tvip/a8Tlp/a8Tpe): they are queued for freed 2xA100 pods, and a8Tpe in particular is
  # A100-class even though the config it ablates (a8T3g) runs H200 -- the class follows the card.
  case "$tag" in a8T|a8Tg1|a8T2s|a8T2r|a8T2b|a8T2n|a8T2nr|a8Tvip|a8Tvipr|a8Tvipc|a8Tlp|a8Tpe) extra="$extra PARAM_OFFLOAD=True OPT_OFFLOAD=True EAGER=True";; esac
  # a8T4 AFTER the batch-32 block: its BSZ/NROLL must override the block's BSZ=32 (env
  # later-wins). 160 rollouts/step preserved: 80x2 vs 32x5. triage k auto-follows NROLL.
  case "$tag" in a8T4) extra="$extra BSZ=80 NROLL=2";; esac
  # q2bTk8 AFTER the batch-32 block for the same later-wins reason, even though that block does
  # not currently set NROLL: this arm's ENTIRE manipulation is one env token, and putting it
  # upstream of a shared block would leave it one shared-block edit away from silent reversal.
  # BSZ is deliberately NOT touched (a8T4 above compute-matched its k change; this one does not
  # -- see the arm's case), so the arm runs 32x8 = 256 rollouts/step against q2bT's 32x5 = 160.
  case "$tag" in q2bTk8) extra="$extra NROLL=8";; esac
  case "$tag" in gatesatsc) extra="$extra PROMPT_LEN=15360 PARAM_OFFLOAD=False OPT_OFFLOAD=False";;
                 gates) extra="$extra PROMPT_LEN=15360 EAGER=True";; esac
  # gates: EAGER=True. Its vLLM worker died at 13:07/13:10/13:14/13:23 on 2026-08-07, each time
  # immediately after "Capturing CUDA graphs (decode, FULL): 100%", with the host oom_kill counter
  # FROZEN at 671 across those deaths and 74/143GB of GPU free -- so neither host nor device OOM.
  # Skipping graph capture removes the step it dies in; the cost is decode throughput, which is
  # worth paying on the one arm that has never held a run together.
  case "$tag" in atscbase|temp12) ncyc=1;; esac
  # 6-slot canary REVERTED: measured 180 eps/hr vs 274 for 1-slot arms -- slower, not faster.
  # ATSC advertises ~86% of scenario tools (vs ~49% handicapped), so its prompts and activations
  # are much larger; one trainer alone reached 116G against the 128G cgroup. Smaller micro-batch
  # token budget is gradient accumulation -- same math, less peak memory.
  [ "$tag" = atsc ] && util=0.45
  case "$tag" in atscfix|atsck8|atsck16|atscbase|gates|gatesd|telem|entropy|gatesatsc|full|atscL2|atscL3|atscfixL2) util=0.38;; q2b|q2bfix|q2bL|q2bL2|q2bM|q2bA|q2bF|q2bF2|q2bF3|q2bF4|q2bF5|q2bF3e5|q2bF6|q2bF7|q2bT|q2bTr|t2bTf|q2bVf|q2bK|q2bT2|q2bT3|q2bTd|q2bTiid|q2bTnw|q2bTnw2|q2bTs|q2bFs|q2bTk8|q2bD|q2bP|q2bR|q2bN|q2bV|q2bLp|q4bV|q4bTvip|q4bV2|q4bLp|q4bF|q4bF2|q4bT|q4bTnw|q4bT2|q4bD|q4bP|q4bR) util=0.35;; a8A|a8F|b8plr|b8plrr|b8k16|b8van|b8match|a8A2|b8m25|b8k16) util=0.38;; a8A3|a8Az0|a8Auni|a8Ae0|a8C|a8Cu|b8k16v|a8Te0) util=0.35;;
                 # a8T/a8Tg1 on 2xA100: cycle-2 TRIAGE batches (evidence-chosen tasks, longer
                 # episodes) OOM'd the update at 0.35 by 0.4GB -- "10.17 GiB needed, 9.84 free",
                 # the same actor_rollout_update_actor wall dapo hit at 0.50. 0.33 frees ~1.6GB.
                 # a8Te0 stays 0.35: its H200 has 141GB and no such pressure.
                 # a8T2 takes a8T's 0.30 and a8T's offload block above: same 8B model, same 2xA100
                 # pod class, and its batches are MORE concentrated on the long-episode mid-band
                 # tasks that caused the update-step OOM in the first place, not less.
                 # a8T3/a8T3g are H200-class, exactly like a8T2/a8T2g1 -- same 8B model, same
                 # batch, and their batches are no more concentrated than a8T2's (227 mid-band
                 # rows against 210). a8T2n is A100-class like a8T2r and takes 0.30 + offload.
                 # a8Tvip/a8Tlp/a8Tpe are the published-method baselines and are A100-class:
                 # 0.30 + the offload block above, exactly like a8T2r and a8T2n.
                 # a8T6d is a8T3g plus one env flag and inherits its class: a permutation cannot
                 # change a prompt's token COUNT, only its order, so activations, KV cache and the
                 # update step are a8T3g's exactly. H200-class, 0.35.
                 a8T|a8Tg1|a8T2s|a8T2r|a8T2b|a8T2n|a8T2nr|a8Tvip|a8Tvipr|a8Tvipc|a8Tlp|a8Tpe|a8Tme2|q4bTa|q4bTa2|q4bTpe|q4bTpe2|q4bTf|q4bTx|q4bTy|q4bTz|q4bTa16|q4bTz2|q4bTr|q4bTr2|q4bTr3) util=0.30;; a8Fr) util=0.38;; a8T2|a8T2w|a8T2g1|a8T2g1r|a8T3|a8T3g|a8Tr|t8Tf|a8Tvipf|a8Tme|a8T3gr|a8T3g2|a8T4|a8T5|a8T5k|a8T5kr|a8T6d|a8T7w|a8T8c) util=0.35;; b8ret|b8accel|b8dense) util=0.35;; temp12) util=0.32; extra="$extra PROMPT_LEN=15360";; atsc|atsc2) util=0.50;;
                 # dapo is 8B: 0.50 was the 2B-era value and left ~9.9GB free at the update
                 # step while DAPO's filtered batches asked for 10.02GB -- OOM on the FIRST
                 # update, every relaunch, forever (29,587 episodes, 0 checkpoints). 0.35
                 # matches the rest of the 8B fleet and frees ~12GB for the update.
                 dapo) util=0.27;; esac
  # TP MUST EQUAL THE GPU COUNT OF THE SLOT THE ARM LANDS ON. vLLM asserts
  # "rollout world_size N is not divisible by infer_world_size TP", so a mismatch is fatal at
  # engine init, not a slowdown. A queued arm cannot know its slot: b8accel carried a hardcoded
  # TP=2 (right for a 2-GPU pod) and was promoted onto single-GPU h200a at 01:36, which failed
  # every attempt with 0 episodes. The earlier version of this guard only ADDED TP when absent,
  # so it could not repair a value that was present and wrong.
  #
  # Strip whatever the arm declared and set it from the slot. On 80G A100 pairs TP=2 is also the
  # only way an 8B fits at all -- one GPU cannot hold weights plus a 28672-token KV cache beside
  # the FSDP trainer -- so the offload flags ride along with it.
  extra=$(printf '%s' "$extra" | sed -E 's/(^| )TP=[0-9]+/\1/g; s/  +/ /g')
  extra="$extra TP=${3:-1}"
  case "${3:-1}" in
    2) case "$extra" in *PARAM_OFFLOAD=*) ;; *) extra="$extra PARAM_OFFLOAD=True OPT_OFFLOAD=True";; esac ;;
  esac
  # A per-arm AWM_SLOTS_TRAIN must win over the fleet default. `export A=3 A=8` takes the LAST
  # value, and $extra_slots is emitted after $extra, so b8k16's 3 was silently overwritten by the
  # default 8 -- the setting looked applied in the command line and did nothing.
  case "$extra" in *AWM_SLOTS_TRAIN=*) extra_slots="";; esac
  echo "export $extra $extra_slots GPU_UTIL=$util RESP_LEN=$rlen SAVE_FREQ=5 NGPUS=${3:-1}; $EV $R/surface/verl_rl/coadapt.py \
--tag $tag --jobid $jid --gate $gate $rw --seed-offset $off --cycles $ncyc --steps-per-cycle $spc \
--pool $pool --handicap-frac 0.40 --edit-block 700"
}
# Horizon cut 6 -> 3 cycles on 2026-08-06. At observed throughput 150 steps is days away, and the
# dose curve puts the novel-scenario knee at two certified edits (78% restoration), which coadapt
# already reached. Three cycles yields completed, comparable arms instead of indefinite partials.

progress () {    # training: episodes + gate episodes + 1000*step ; eval: episodes in current cell
  local kind=$1 tag=$2 slot=$3
  if [ "$kind" = eval ]; then
    local c=${cur[$slot]:-none}
    [ "$c" = none ] && { echo 0; return; }
    # Read the cell of the arm being evaluated, NOT always coadapt's. cell_name() strips the
    # "@arm" suffix for the filename, but the writer puts the file under coadapt_eval_<arm>/ --
    # so reading coadapt_eval_coadapt/ compared every arm against coadapt's 590-line cell, saw it
    # already full, and marked the card "cell_done_at_590" without running anything. Three A100s
    # rotated on already-complete cells for hours, which is why no RL arm ever got a held-out
    # policy number.
    cat $R/work/coadapt_eval_$(cell_tag $c)/cell_$(cell_name $c).jsonl 2>/dev/null | wc -l
  else
    local e g s
    e=$(cat $R/work/verl/run_$tag/*.jsonl 2>/dev/null | wc -l)
    g=$(cat $R/work/coadapt_$tag/cycle*/*.jsonl 2>/dev/null | wc -l)
    s=$(ls -d $R/work/verl/ckpt_$tag/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
    echo $(( ${e:-0} + ${g:-0} + 1000 * ${s:-0} ))
  fi
}

clear_tag () {   # jid tag -> kill ONLY this arm, never its podmates
  # clear_job kills every process in the ALLOCATION. That is correct when a pod hosts one arm and
  # catastrophic when it hosts two: a two-GPU pod runs one arm per card, so
  # clearing a8A to promote a8A2 also killed a8F, a live baseline, at 23:13 on 2026-08-09.
  # A promotion replaces ONE slot, so it must kill exactly one arm's process tree.
  #
  # Every pattern is ANCHORED. An unanchored "--tag a8A" prefix-matches a8A2 and a8A3, which is
  # the same class of mistake that once killed a healthy arm because "gates" matched "gatesatsc".
  local jid=$1 tag=$2
  timeout 150 srun --jobid=$jid --ntasks=1 --overlap --cpus-per-task=1 --mem=2G bash -c '
    kt () { for c in $(pgrep -P $1 2>/dev/null); do kt $c; done; kill -9 $1 2>/dev/null; }
    TAG="'"$tag"'"
    { pgrep -u $USER -f "coadapt\.py --tag $TAG( |\$)" 2>/dev/null
      pgrep -u $USER -f "awm_${TAG}_grpo" 2>/dev/null
      pgrep -u $USER -f "ckpt_${TAG}(/| |\$)" 2>/dev/null
    } | sort -u | while read -r p; do
      [ "$p" = "$$" ] && continue
      kt $p
    done
    sleep 5' >/dev/null 2>&1 </dev/null
}

clear_job () {   # kill everything of mine in this job, never touching another user or job
  # Takes an OPTIONAL second argument, the arm tag. The /proc/<pid>/cgroup match below is not
  # reliable here -- it silently matched nothing during diagnosis on 2026-08-07, so a relaunch
  # left the previous driver alive and gates ran TWO drivers on one GPU for 3h46m, each booting
  # its own vLLM and killing the other's workers. That looked like random EngineDeadError and
  # burned the arm all night. Matching the driver's own command line does not depend on cgroup
  # readability, so the tag sweep is the authoritative kill and the cgroup sweep is the backstop.
  local jid=$1 tag=${2:-}
  timeout 150 srun --jobid=$jid --ntasks=1 --overlap --cpus-per-task=1 --mem=2G bash -c '
    kt () { for c in $(pgrep -P $1 2>/dev/null); do kt $c; done; kill -9 $1 2>/dev/null; }
    TAG="'"$tag"'"
    if [ -n "$TAG" ]; then
      for p in $(pgrep -f "coadapt\.py --tag $TAG( |$)" 2>/dev/null); do
        [ "$p" = "$$" ] && continue
        kt $p
      done
    fi
    for p in $(ps -eo pid --no-headers); do
      j=$(grep -aoE "job_'"$jid"'" /proc/$p/cgroup 2>/dev/null | head -1)
      [ -n "$j" ] || continue
      a=$(ps -o args= -p $p 2>/dev/null)
      case "$a" in *slurmstepd*|*slurm_script*) continue;; esac
      kt $p
    done; sleep 6' >/dev/null 2>&1 </dev/null
}

refill_queue () {
  # The eval cards' standing job is now the LEARNING CURVE, not per-arm SELF cells.
  #
  # SELF/SELFBASE answered "did this arm's policy beat base on its own surface" and the answer
  # came back ~0 for every arm at 5-72 steps. Re-running them tells us nothing new until the arms
  # are deeper, and the retired arms (atscbase, temp12, atsck8) are not being trained at all now,
  # so cells for them burn a card to re-measure a policy nobody will train further.
  #
  # queue_curve.sh emits one point per ~40 steps per depth arm, skipping any already measured, so
  # this is self-limiting: it adds work exactly as fast as training produces checkpoints.
  bash $R/slurm/queue_curve.sh >> $R/logs/supervisor.log 2>&1
  echo "$(date +%H:%M) queue refilled with learning-curve points" >> $R/logs/supervisor.log
}

TQUEUE=$R/work/train_queue.txt

next_arm () {   # pop "tag|flags|env|comment" off the training queue
  [ -s "$TQUEUE" ] || return 1
  local a; a=$(head -1 "$TQUEUE"); sed -i '1d' "$TQUEUE"; echo "$a"
}

DRAINFILE=$R/slurm/drain_pods.txt

drain_pod () {   # jid -> is this pod being given back once its current arm finishes?
  # Releasing a pod is not the same as killing one. scancel now would discard whatever the arm has
  # not yet checkpointed; leaving it alone forever never releases anything, because arm_done hands
  # the freed card straight to the next queued arm. Draining is the middle: the arm runs to its own
  # completion, then the allocation ends instead of being refilled.
  [ -s "$DRAINFILE" ] || return 1
  grep -q "^${1}|" "$DRAINFILE" 2>/dev/null
}

rotate_log () {  # keep a relaunched arm's history WITHOUT letting the old run speak for the new one
  # Every launch used to truncate logs/rl_<tag>.log with `>`. That is what keeps arm_done's
  # "[coadapt] DONE" grep and the quiet-minutes mtime honest -- both read the whole file, so an
  # appended log would let a previous run's DONE marker retire the arm that replaced it the first
  # cycle after launch, handing a live card to the queue for no reason.
  #
  # But truncation also destroys the `val-core/awm/reward/mean@1` lines, and those ARE the
  # validation curve. Their step numbers live only in the console log, so every relaunch used to
  # silently cut points out of the curve -- worst for the arms that crash most, which is exactly
  # the population where a missing point looks like a flat curve rather than a gap.
  # Rotate: the live log is always one run, the history is still on disk for val_curve.py.
  local f=$1 n
  [ -s "$f" ] || return 0
  # MONOTONIC numbering: max existing + 1, never the first free slot. Scanning upward from 1
  # reuses a number the prune just freed, and since the prune keeps the HIGHEST numbers, the
  # reused-low newest run is then deleted on its own next rotation -- the curve would keep three
  # stale runs forever and throw away every new one. Caught by the unit test below, which held
  # .2/.3/.4 across six rotations instead of advancing.
  n=$(ls -1 "$f".[0-9]* 2>/dev/null | sed 's/\.gz$//' | sed 's/.*\.//' \
        | grep -E '^[0-9]+$' | sort -n | tail -1)
  n=$(( ${n:-0} + 1 ))
  mv "$f" "$f.$n" 2>/dev/null || return 0
  gzip -f "$f.$n" 2>/dev/null &                 # ~15x on these logs; /project fills to 99% and
  # keep the newest ROTKEEP runs, drop the rest oldest-first        # kills every process when it does
  ls -1 "$f".[0-9]*.gz "$f".[0-9]* 2>/dev/null | sed 's/\.gz$//' | sort -t. -k3 -n -u \
    | head -n -${ROTKEEP:-12} | while read -r old; do rm -f "$old" "$old.gz"; done
}

arm_done () {   # finished: driver says so, horizon reached, OR training reward has saturated
  local tag=$1
  grep -aq "\[coadapt\] DONE" "$R/logs/rl_$tag.log" 2>/dev/null && return 0
  # SATURATION. Holding a card on an arm whose reward stopped moving costs more than the extra
  # steps are worth -- there are queued arms that answer open questions (b8match isolates ATSC's
  # adaptivity from its surface richness). An arm counts as saturated once it has >=40 steps AND
  # its last four 640-episode blocks have drifted <0.01. The step floor matters: block noise is
  # +/-0.07, so a short run can look flat by luck.
  local st_sat
  st_sat=$(ls -d $R/work/verl/ckpt_$tag/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
  if [ -n "$st_sat" ] && [ "$st_sat" -ge 40 ]; then
    $EV - "$tag" <<'PY' && return 0
import sys, glob, json, math, statistics
tag = sys.argv[1]
R = os.environ["BRACE_ROOT"]
rs = []
try:
    since = int(open(f"{R}/work/verl/run_{tag}/.arm_since").read().strip() or 0)
except Exception:
    since = 0
seen = 0
for f in sorted(glob.glob(f"{R}/work/verl/run_{tag}/*.jsonl")):
    for line in open(f, errors="ignore"):
        seen += 1
        if seen <= since:          # belongs to a previous run of this arm
            continue
        try: d = json.loads(line)
        except Exception: continue
        if "reward" in d:
            try: rs.append(int(d["reward"] > 0))
            except Exception: pass
B = 640
nb = len(rs) // B
if nb < 8:
    sys.exit(1)
# Compare MEANS of the last four blocks against the previous four. A range test on raw
# blocks cannot work here: measured block-to-block noise is 0.047-0.097, so requiring a
# range below 0.01 would never fire and the queued arm would wait forever. Averaging four
# blocks cuts that noise ~2x, and "the recent mean is no better than the earlier mean" is
# the actual question -- an arm that has stopped improving should yield its card.
ser = [sum(rs[i*B:(i+1)*B]) / B for i in range(nb)]
prev, last = ser[-8:-4], ser[-4:]
if len(prev) < 4:
    sys.exit(1)
# RETIRE ONLY WHAT IS RESOLVABLE. The bare test `delta < 0.005` compared two 4-block means
# against a threshold far below their own noise. Measured on 2026-08-09: a 4-block mean carries
# se ~ 0.017, so 0.005 is 0.29 se -- the test cannot separate +0.003 from +0.02 or from zero.
# It retired a8A at delta +0.0031 (step 72 of 120) while keeping a8F at +0.0098, a gap of 0.4 se
# between two arms that were statistically identical. b8plr was reading -0.0016 and would have
# been cut at step 40 the moment it crossed the floor.
#
# This is the failure the paper's own gate exists to prevent: acting on an effect below what the
# budget can resolve. So require the flatness to survive its own error bar -- retire only when
# even the optimistic end of the estimate is still flat. In practice that means a real decline,
# and arms run to their horizon unless they are genuinely degrading.
d = sum(last) / 4 - sum(prev) / 4
sd = statistics.pstdev(ser) if len(ser) > 1 else 0.0
se_diff = sd * math.sqrt(2.0 / 4.0)          # se of a difference of two 4-block means
sys.exit(0 if (d + se_diff) < 0.005 else 1)
PY
  fi
  local st
  st=$(ls -d $R/work/verl/ckpt_$tag/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
  local want=75
  case "$tag" in atscbase|temp12) want=25;; atsck8|gatesatsc|full) want=72;;
                 atsc|atscfix|atscL2|atscL3|atscfixL2|dapo|q2b|q2bfix|q2bL|q2bL2|q2bM|q2bA|q2bF|q2bF2|q2bF3|q2bF4|q2bF5|q2bF3e5|q2bF6|q2bF7|q2bT|q2bTr|t2bTf|q2bVf|q2bK|q2bT2|q2bT3|q2bTd|q2bTiid|q2bTnw|q2bTnw2|q2bTs|q2bFs|q2bTk8|q2bD|q2bP|q2bR|q2bN|q2bV|q2bLp|q4bV|q4bTvip|q4bV2|q4bLp|q4bF|q4bF2|q4bT|q4bTpe|q4bTpe2|q4bTf|q4bTr|q4bTr2|q4bTr3|q4bTa|q4bTa2|q4bTx|q4bTy|q4bTz|q4bTa16|q4bTz2|q4bTnw|q4bT2|q4bD|q4bP|q4bR|a8A|a8F|a8Fr|b8plr|b8plrr|b8k16|b8van|b8match|a8A2|a8A3|a8Az0|a8Auni|a8Ae0|b8m25|a8C|a8Cu|b8k16v|a8T|a8Tg1|a8Te0|a8T2w|a8T2s|a8T2r|a8T2|a8T2g1|a8T2g1r|a8T2b|a8T2n|a8T2nr|a8T3|a8T3g|a8Tr|t8Tf|a8Tvipf|a8Tme|a8Tme2|a8T3gr|a8T3g2|a8T4|a8T5|a8T5k|a8T5kr|a8T6d|a8T7w|a8T8c|a8Tvip|a8Tvipr|a8Tvipc|a8Tlp|a8Tpe) want=120;;
                 b8ret|b8accel|b8dense) want=120;; esac
  [ -n "$st" ] && [ "$st" -ge "$want" ]
}

launch_queued () {   # slot jobid kind oldtag -> start the next queued arm here, echo its tag
  # WHY THIS DELEGATES TO train_cmd. It used to build its own srun command line, and that copy
  # drifted from train_cmd in three ways that corrupt an experiment rather than crash it:
  #   no --pool          -> a promoted arm trained on pool_big320 while every arm it is compared
  #                         against uses pool_max, so the comparison measured the pool
  #   no BSZ/MINIBSZ     -> it ran at the gradient-starved batch that flatlined six earlier arms
  #   --cycles 3 x 25    -> 75 steps against train_cmd's 15 x 8 = 120, a different horizon
  # And it never stopped the arm it replaced: on 2026-08-09 a8A2 was promoted onto h200v2a while
  # a8A was still training there, putting two drivers on one GPU -- the duplicate-driver fault
  # this supervisor exists to prevent, reintroduced by the one path that bypassed its own builder.
  # One command builder, and clear the outgoing arm before the incoming one boots.
  local slot=$1 jid=$2 kind=$3 oldtag=${4:-} spec tag n g
  # PEEK, do not pop. next_arm deletes the line it returns, so refusing an arm after popping it
  # loses that arm silently -- which is how a8A2 vanished from the queue when its promotion was
  # aborted. The entry is consumed only once the launch is committed.
  [ -s "$TQUEUE" ] || return 1
  spec=$(head -1 "$TQUEUE")
  tag=$(echo "$spec" | cut -d'|' -f1)
  # A queued arm MUST be registered in train_cmd, or train_cmd hands it defaults unrelated to what
  # the queue line asked for. Refusing is the safe failure: an unlaunched arm is visible in the
  # log every cycle, a silently misconfigured one looks like a result.
  if ! grep -qE "^[[:space:]]+${tag}\)[[:space:]]+gate=" "$R/slurm/supervisor.sh"; then
    echo "$(date +%H:%M) REFUSED to promote $tag: not registered in train_cmd" >> $R/logs/supervisor.log
    return 1
  fi
  sed -i '1d' "$TQUEUE"
  mkdir -p $R/work/coadapt_$tag
  cp $R/work/coadapt_coadapt/advertised_init.txt $R/work/coadapt_$tag/advertised_init.txt 2>/dev/null
  cp $R/work/coadapt_coadapt/advertised_init.txt $R/work/coadapt_$tag/advertised.txt 2>/dev/null
  n=1; [ "$kind" = train2 ] && n=2
  mark_arm_start "$tag"
  local spec cvd; spec=$(slot_gres "$slot" "$jid" "$kind"); g=${spec%|*}; cvd=${spec#*|}
  [ -n "$oldtag" ] && clear_tag "$jid" "$oldtag"
  ensure_scratch_run "$tag"
  rotate_log "$R/logs/rl_$tag.log"
  nohup srun --jobid=$jid --ntasks=1 --overlap $g \
    bash -c "${cvd:+export CUDA_VISIBLE_DEVICES=$cvd; }$(train_cmd $tag $jid $n)" \
    > $R/logs/rl_$tag.log 2>&1 </dev/null &
  echo "$(date +%H:%M) promoted queued arm $tag onto $slot (cleared ${oldtag:-none})" >> $R/logs/supervisor.log
  echo "$tag"
}

job_has_work () {   # jid [tag] -> is THIS ARM actually running in this job?
  # WHY THIS WAS REWRITTEN. The old version counted every process in the job cgroup that was not
  # slurmstepd/slurm_script/ps. Its own `bash -c` probe runs inside that cgroup and matched, so the
  # count was >=1 unconditionally and the relaunch branch could never fire; on timeout it also
  # defaulted to 1 ("has work"), blocking relaunch again. It only ever relaunched anything when the
  # /proc/<pid>/cgroup match failed by accident, which the reaper comments already record as
  # unreliable. That is why dead slots sat idle for long stretches today -- b8accel for two hours,
  # a8C and b8k16 for ~20 minutes each -- every one of them needing a hand-launch.
  #
  # Two cheap, precise checks instead:
  #   1. no srun client for this job at all  -> nothing of ours runs there, full stop (no probe).
  #   2. otherwise ask the pod for THIS TAG's processes, excluding the probe's own pid, because a
  #      pod can host two arms and one may be dead while the other is healthy.
  local jid=$1 tag=${2:-} n
  [ "$(ps -u "$USER" -o args= 2>/dev/null | grep -c "^srun .*jobid[= ]$jid")" -eq 0 ] && return 1
  [ -z "$tag" ] && return 0
  n=$(timeout 60 srun --jobid=$jid --ntasks=1 --overlap --cpus-per-task=1 --mem=1G bash -c '
    T="'"$tag"'"
    { pgrep -u $USER -f "coadapt\.py --tag $T( |\$)" 2>/dev/null
      pgrep -u $USER -f "awm_${T}_grpo" 2>/dev/null
      pgrep -u $USER -f "h200_arm\.sh $T\$" 2>/dev/null
    } | grep -v "^$$\$" | sort -u | wc -l' 2>/dev/null </dev/null)
  [ "${n:-1}" -gt 0 ]
}

next_cell () {   # pop "CELL[:TARGET]" off the queue; targets differ (a 4-seed cell wants 1180)

  [ -s "$QUEUE" ] || return 1
  local c
  c=$(head -1 "$QUEUE"); sed -i '1d' "$QUEUE"
  echo "$c"
}
cell_name ()   { local x="${1%%:*}"; echo "${x%%@*}"; }
cell_tag ()    { local x="${1%%:*}"; case "$x" in *@*) echo "${x##*@}";; *) echo coadapt;; esac; }
cell_target () { local t="${1##*:}"; [ "$t" = "$1" ] && echo 590 || echo "$t"; }

ensure_scratch_run () {   # tag -> make work/verl/{run,ckpt}_<tag> symlinks into scratch before they write
  # /project is a shared 10T at 95%, and a full /project does not fail politely -- it took out
  # awsctrl and awsctrl2 with ENOSPC. Rollouts write a SQLite pair per scenario per boot, so an
  # arm left on /project is the thing that fills it.
  #
  # CKPT WAS MISSING HERE. Every live arm had its ckpt_ symlinked by hand, so the gap was
  # invisible until a NEW tag appeared: a8A3 on 2026-08-09 would have written checkpoints to
  # /project at KEEPCKPT=12 x 26G = ~312G, against 575G free on a filesystem already at 95%.
  # A per-arm manual step that everyone forgets is not a safeguard.
  #
  # Relaunch is the only safe moment to swap a path: nothing is writing. An arm already
  # symlinked, or one holding data on /project we have not migrated, is left exactly as it is --
  # never move a directory we cannot prove is idle.
  local tag=$1 what d dst
  for what in run ckpt; do
    d=$R/work/verl/${what}_$tag
    dst=$BRACE_WORK/$what/${what}_$tag
    [ -L "$d" ] && continue
    mkdir -p "$(dirname "$dst")"
    if [ -d "$d" ]; then
      [ -e "$dst" ] && dst="$dst.$(date +%s)"
      mv "$d" "$dst" 2>/dev/null || continue
    else
      mkdir -p "$dst"
    fi
    ln -s "$dst" "$d" 2>/dev/null && \
      echo "$(date +%H:%M) ${what}_$tag now on scratch" >> $R/logs/supervisor.log
  done
}

# Which GPU index this slot owns, when its pod hosts more than one training slot.
# Slurm hands every "--gres=gpu:1 --overlap" step on the same job DEVICE 0, so two arms sharing a
# pod land on one card while the other sits idle. That has now cost three arms: a8A2 at 01:36,
# then a8F and b8k16 together at 10:34, where GPU0 held 71GB and GPU1 was at 0 MiB while both
# arms failed engine init. Slurm will not separate them, so the slot table does: each training
# slot on a shared pod takes its ordinal as CUDA_VISIBLE_DEVICES, and the step is launched WITHOUT
# --gres so it can see every device the allocation holds.
slot_devices () { awk -F'|' -v j="$1" '$2==j && $5!="eval"{c++} END{print c+0}' "$SLOTFILE"; }
# print n+0, not n: for the FIRST matching slot awk's n is unset and `print n` emits an EMPTY
# string, so slot_gres's `[ -n "$idx" ]` guard was false and slot 0 of every shared pod fell
# through to the single-slot branch -- full CPUs, --gres=gpu:1, and no device pin. That is why
# b8k16 launched with 8 CPUs beside a8Cu on a 16-CPU pod and killed it.
slot_index ()   { awk -F'|' -v j="$1" -v s="$2" '$2==j && $5!="eval"{if($1==s){print n+0; exit} n++}' "$SLOTFILE"; }
# Emits the srun resource flags and any device pin for this slot.
slot_gres () {   # slot jobid kind -> "<gres flags>|<cvd or empty>"
  local slot=$1 jid=$2 kind=$3 nd idx cpus mem
  nd=$(slot_devices "$jid"); idx=$(slot_index "$jid" "$slot")
  cpus=8; mem=120G; [ "$kind" = train2 ] && { cpus=24; mem=186G; }
  if [ "${nd:-1}" -gt 1 ] && [ -n "$idx" ]; then
    # SHARED POD: split the CPU budget and KEEP A RESERVE. The H200 pod carries 16 CPUs for its
    # 2 GPUs (8 per GPU) where every A100 pod carries 28 for 2 (14 per GPU), so two slots at the
    # flat 8 consumed the entire allocation and left nothing for the supervisor's probe, the
    # watcher's per-arm check, or any diagnostic srun. That pod is the only one that ever killed
    # an arm -- six of them, all with "Killed / Force Terminated" rather than OOM or an engine
    # fault. Derive the share from what the pod actually has and hold 4 CPUs back.
    local tot
    tot=$(scontrol show job "$jid" 2>/dev/null | grep -oE "NumCPUs=[0-9]+" | head -1 | cut -d= -f2)
    if [ -n "$tot" ] && [ "$tot" -gt 0 ]; then
      cpus=$(( (tot - 4) / nd ))
      [ "$cpus" -lt 4 ] && cpus=4
    fi
    # Memory the same way: split what the pod HAS rather than defaulting to 120G per slot and
    # leaving 80G of a 320G pod unallocated. a8Cu is killed mid-first-step with no traceback and
    # no OOM message, which is what Slurm does to a step that exceeds its --mem, and it is the
    # only arm here running the dense-reward agent script (extra verifier grading in the loop).
    # Its podmate b8k16 survives on the identical launch form, so the difference is the workload,
    # not the pod. Cheap to give it the headroom that is sitting idle.
    local memtot
    memtot=$(scontrol show job "$jid" 2>/dev/null | grep -oE "mem=[0-9]+G" | head -1 | tr -dc '0-9')
    if [ -n "$memtot" ] && [ "$memtot" -gt 40 ]; then
      mem="$(( (memtot - 20) / nd ))G"
    fi
    # KEEP --gres=gpu:1 AND LET SLURM ASSIGN THE DEVICE. Dropping it and exporting
    # CUDA_VISIBLE_DEVICES by hand was my change, and it is what broke this pod: a step with no
    # gres request gets no GPU in its device cgroup, so touching one is killed by Slurm -- which
    # is precisely the signature every death carried ("Killed / Force Terminated", never OOM,
    # never an engine fault). a8A and a8F shared this same pod for hours under the original
    # form, reaching steps 72 and 85. The CPU share above is kept because it is a real
    # improvement and costs nothing; the hand-pinning is not.
    echo "--gres=gpu:1 --cpus-per-task=$cpus --mem=$mem|"
  elif [ "$kind" = train2 ]; then
    echo "--gres=gpu:2 --cpus-per-task=24 --mem=186G|"
  else
    echo "--gres=gpu:1 --cpus-per-task=8 --mem=120G|"
  fi
}

# Episodes already banked when this arm (re)starts. arm_done reads ALL of run_<tag>/*.jsonl, so a
# RESUMED arm is judged on the history of its previous run: a8A came back at step 72 on 2026-08-10
# with 16,500 stale episodes behind it, produced ~200 new ones, and was retired 10 minutes later on
# a statistic that could not possibly reflect the resumed run. The requeue mechanism is useless
# without this -- any arm that already ran is killed on arrival.
mark_arm_start () {
  local tag=$1 d=$R/work/verl/run_$tag n=0
  [ -f "$d/episodes.jsonl" ] && n=$(wc -l < "$d/episodes.jsonl" 2>/dev/null)
  mkdir -p "$d" 2>/dev/null
  printf '%s' "${n:-0}" > "$d/.arm_since" 2>/dev/null
}

launch () {      # slot jobid tag log kind
  local slot=$1 jid=$2 tag=$3 log=$4 kind=$5
  [ "$kind" != eval ] && ensure_scratch_run "$tag"
  if [ "$kind" = eval ]; then
    local c; c=$(next_cell) || { cur[$slot]=none; return; }
    cur[$slot]=$c
    local nm=$(cell_name $c) sd=4
    [ "$(cell_target $c)" -gt 590 ] && sd=4 || sd=2
    nohup srun --jobid=$jid --ntasks=1 --overlap --gres=gpu:1 --cpus-per-task=8 --mem=88G \
      bash -c "cd /tmp && SEEDS=$sd AWM_SLOTS=4 EVAL_MEMFRAC=${EVAL_MEMFRAC:-0.30} bash $R/slurm/coadapt_eval.sh $nm $(cell_tag $c)" \
      > $R/logs/$log.log 2>&1 </dev/null &
    echo "$(date +%H:%M) launched cell $c on $slot" >> $R/logs/supervisor.log
  else
    local n=1 spec g cvd
    [ "$kind" = train2 ] && n=2
    mark_arm_start "$tag"
    spec=$(slot_gres "$slot" "$jid" "$kind"); g=${spec%|*}; cvd=${spec#*|}
    rotate_log "$R/logs/$log.log"
    nohup srun --jobid=$jid --ntasks=1 --overlap $g \
      bash -c "${cvd:+export CUDA_VISIBLE_DEVICES=$cvd; }$(train_cmd $tag $jid $n)" \
      > $R/logs/$log.log 2>&1 </dev/null &
    echo "$(date +%H:%M) relaunched $tag on $slot" >> $R/logs/supervisor.log
  fi
}

# SINGLETON. Four instances were found running concurrently on 2026-08-06, each an independent
# launcher racing the others and overwriting the shared status/state files -- exactly the
# duplicate-driver failure this supervisor exists to prevent, institutionalised. A stale pidfile
# whose process is gone is reclaimed; a live one refuses the start.
PIDFILE=$R/logs/.supervisor_pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "supervisor already running as pid $(cat "$PIDFILE"); refusing to start a second"
  exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

PAUSE=$R/logs/.supervisor_pause
while :; do
  # Bulk operations (a horizon change, a staggered rebuild) are also launchers. Two launchers
  # racing produced two drivers per pod on 2026-08-06 -- the exact duplicate-trainer fault this
  # supervisor exists to prevent. Anything doing bulk restarts takes this lock first.
  if [ -f "$PAUSE" ]; then
    echo "$(date +%H:%M) paused ($(cat $PAUSE 2>/dev/null))" >> $R/logs/supervisor.log
    sleep 60; continue
  fi
  now=$(date +%s); : > $STATUS.tmp; : > $STATE.tmp
  echo "# $(date +'%F %H:%M')  supervisor" >> $STATUS.tmp

  # Reap orphaned MCP servers before assessing the fleet. A killed rollout worker used to strand
  # its whole per-scenario server pool (sh + python + tee, reparented to PID 1); by 2026-08-07
  # that had reached 3264 shells and 117G of anon on the node, so every restart made the next OOM
  # arrive sooner. server.py now sets PR_SET_PDEATHSIG, which stops the leak at the source -- this
  # stays as the safety net for servers spawned by pre-fix code and for hard kills that outrun it.
  # Every 12th cycle (~1h), and NOT blocking. Running it inline every cycle with a wait pushed the
  # cycle past five minutes and left fleet_status six minutes stale -- a monitor that is late is a
  # monitor that restarts things late. server.py's PR_SET_PDEATHSIG stops the leak at the source,
  # so this is a slow safety net for pre-fix servers, not something the loop should wait on.
  REAP_EVERY=${REAP_EVERY:-12}
  reap_n=$(( ${reap_n:-0} + 1 ))
  if [ $(( reap_n % REAP_EVERY )) -eq 1 ]; then
    for rj in $(awk -F'|' '!/^#/ && NF>1 {print $2}' $SLOTFILE 2>/dev/null | sort -u); do
        timeout 240 srun --jobid=$rj --chdir=$R --ntasks=1 --overlap --cpus-per-task=4 --mem=4G \
            bash $R/slurm/reap_orphans.sh >> $R/logs/reaper.log 2>&1 </dev/null &
    done
  fi
  while IFS='|' read -r slot jid tag log kind; do
    [ -z "${slot:-}" ] && continue
    st=$(squeue -j $jid -h -o %T 2>/dev/null </dev/null)
    if [ "$kind" != eval ]; then
      case "${cur[$slot]:-none}" in none|"") cur[$slot]=$tag;; *) tag=${cur[$slot]}; log=rl_$tag;; esac
    fi
    p=$(progress "$kind" "$tag" "$slot")
    lm=$(stat -c %Y $R/logs/$log.log 2>/dev/null || echo 0)
    quiet=$(( (now - lm) / 60 ))
    if [ "${last[$slot]:-}" = "$p" ]; then strike[$slot]=$(( ${strike[$slot]:-0} + 1 )); else strike[$slot]=0; fi
    last[$slot]=$p
    act=""
    if [ -z "$st" ] || [ "$st" != RUNNING ]; then
      act="POD_${st:-GONE}"                                   # nothing we can do but report
    elif [ "$kind" = eval ] && [ "${cur[$slot]:-none}" = none ] && [ -s "$QUEUE" ]; then
      # BOOTSTRAP an idle eval slot. The cell-done branch below requires p>0, but an idle
      # slot holds no cell and so has p=0 -- it could never start one. Three A100s sat idle
      # for two hours against a full queue because of this. Any slot with no cell takes the
      # next queued one immediately.
      launch $slot $jid $tag $log $kind; act="started_${cur[$slot]:-none}"; strike[$slot]=0
    elif [ "$kind" = eval ] && [ "${strike[$slot]}" -ge 2 ] && [ "$p" -gt 0 ]; then
      # A cell is finished when it STOPS PRODUCING, not when it hits a preset count. Targets are
      # guesses about seeds x tasks and an unreachable one (CEIL:1770 against a 4-seed launcher
      # that caps near 1180) pins a card on a run that can never complete. Quiescence is the
      # signal the run itself gives us, and cells resume from banked episodes if requeued.
      act="cell_done_at_$p"; clear_job $jid $tag; launch $slot $jid $tag $log $kind; strike[$slot]=0
    elif [ "$kind" != eval ] && arm_done "$tag"; then
      # DRAIN takes precedence over promotion. A pod listed in drain_pods.txt is being GIVEN BACK,
      # so when its arm finishes the card must go quiet and the allocation must end -- promoting a
      # queued arm onto it would hold the GPU indefinitely, since there is always something queued.
      # Draining on completion rather than scancel-now is the whole point: the arm keeps its card
      # until its own work is done, and nothing in flight is discarded.
      if drain_pod "$jid"; then
        clear_tag "$jid" "$tag"
        # Drop the slot BEFORE scancel so no later cycle can relaunch into a dying allocation.
        # The loop reads SLOTFILE through a here-string snapshot, so editing it here is safe.
        grep -v "^${slot}|${jid}|" "$SLOTFILE" > "$SLOTFILE.tmp" && mv "$SLOTFILE.tmp" "$SLOTFILE"
        sed -i "/^${jid}|/d" "$DRAINFILE" 2>/dev/null
        scancel "$jid" 2>/dev/null
        echo "$(date +%H:%M) DRAINED $slot job=$jid after $tag finished; pod released" >> $R/logs/supervisor.log
        act="drained_after_$tag"
      else
        # Finished arms free their card for the next queued baseline instead of idling or being
        # pointlessly restarted. This is what keeps six training cards busy without me.
        nt=$(launch_queued "$slot" "$jid" "$kind" "$tag") && { cur[$slot]=$nt; act="promoted_$nt"; strike[$slot]=0; } \
          || act="done_queue_empty"
      fi
    elif [ "$quiet" -ge 8 ] && [ "${strike[$slot]}" -ge 1 ] && ! job_has_work "$jid" "$tag"; then
      # A force-terminated or crashed step leaves NO processes in the cgroup -- unambiguous, and
      # needs no waiting. The quiet heuristic sat on a dead atscfix25 for 14 minutes because its
      # threshold is tuned for slow boots. The srun probe is gated behind cheap signals so it runs
      # only when something already looks wrong.
      fails[$slot]=$(( ${fails[$slot]:-0} + 1 ))
      act="DEAD_STEP_RESTART#${fails[$slot]}"
      clear_job $jid $tag; launch $slot $jid $tag $log $kind; strike[$slot]=0
    elif [ "${strike[$slot]}" -ge "$STUCK" ] || \
         { [ "${strike[$slot]}" -ge 2 ] && [ "$quiet" -ge "$STALL_MIN" ]; }; then
      fails[$slot]=$(( ${fails[$slot]:-0} + 1 ))
      act="RESTART#${fails[$slot]}"
      clear_job $jid $tag; launch $slot $jid $tag $log $kind; strike[$slot]=0
    else
      fails[$slot]=0
    fi
    printf '%-7s job=%s %-9s %-9s prog=%-8s quiet=%smin strikes=%s %s\n' \
      "$slot" "$jid" "$tag" "${st:-GONE}" "$p" "$quiet" "${strike[$slot]}" "$act" >> $STATUS.tmp
    echo "$slot $p ${strike[$slot]} ${fails[$slot]:-0} ${cur[$slot]:-none}" >> $STATE.tmp
    # escalate only when self-healing is provably not working
    if [ "${fails[$slot]:-0}" -ge "$MAXFAIL" ]; then
      mv $STATUS.tmp $STATUS; mv $STATE.tmp $STATE
      echo "UNRECOVERABLE_${slot}_${tag}_after_${fails[$slot]}_restarts"; cat $STATUS; exit 1
    fi
  done <<< "$(grep -v '^[[:space:]]*$' "$SLOTFILE")"
  mv $STATUS.tmp $STATUS; mv $STATE.tmp $STATE
  # Refill AFTER the state write, not at the loop top. The refill's in-flight guard reads the
  # state FILE, which mid-cycle still shows the previous cycle -- so a cell popped and launched
  # earlier in this same cycle was invisible and got re-queued, and two cards ran the same cell
  # (STEP10@a8Te0, 13:19 and 13:20, torn writes). After the mv, the state is exactly this
  # cycle's truth and the guard closes the window the every-cycle refill opened.
  refill_queue
  sleep $CYCLE
done
