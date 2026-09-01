#!/bin/bash
# DAPO dynamic sampling (Yu et al. 2025) on top of the shared GRPO trainer script.
#
# WHY A WRAPPER AND WHY THE `+` PREFIX.
# verl 0.8.0 ships AlgoConfig.filter_groups (trainer/config/algorithm.py) but its composed
# Hydra config, trainer/config/ppo_trainer.yaml, has NO `filter_groups` key under `algorithm:`.
# The config node is therefore absent at compose time and struct mode rejects a bare
#     algorithm.filter_groups.enable=True      -> "Key 'filter_groups' is not in struct"
# Hydra's `+` prefix ADDS a key that the schema does not have, which is the correct fix and the
# same trick this repo already uses for +ray_kwargs.ray_init.*.
#
# WARNING, AND THE REASON THIS SCRIPT EXISTS RATHER THAN JUST AN OVERRIDE STRING: the `+`
# override alone is a SILENT NO-OP on a stock wheel. Upstream implements the dynamic-sampling
# loop only in recipe/dapo/dapo_ray_trainer.py, which pip does not install, so nothing reads
# the flag. The loop was added to verl/trainer/ppo/ray_trainer.py::fit (backup:
# ray_trainer.py.bak-agent) and is gated on exactly these keys. Do not assume DAPO is active
# because the run started: confirm the "[dapo] ..." lines in the log.
#
# Everything else -- model, data, LoRA, rollout, agent loop -- comes from verl_awm_train.sh,
# which takes trailing Hydra overrides as "$@". This script only appends the filter_groups node.
#
#   DAPO_METRIC            reward key whose within-group spread decides degeneracy.
#                          seq_final_reward = summed token_level_scores, i.e. episode reward.
#   DAPO_MAX_GEN_BATCHES   cap on generation batches drawn to fill ONE training batch.
#                          0 = no limit. Non-zero raises rather than looping forever when
#                          availability is too low to fill data.train_batch_size.
#   AWM_DENSE_REWARD       none (default) | telemetry | gates. Reward shaping for TRAINING only;
#                          the binary outcome is still what episodes.jsonl and every evaluation
#                          report. This must be pushed into the Ray runtime env explicitly: the
#                          agent loop runs inside Ray workers, and verl_awm_train.sh forwards a
#                          fixed list of variables precisely because, in its own words, "relying
#                          on environment inheritance through the raylet is not safe". Exporting
#                          it in the shell alone would leave the workers on the default and the
#                          arm would train on the plain binary reward while looking shaped.
set -u
R=${BRACE_ROOT:?BRACE_ROOT is unset; source env.sh first}

DAPO_ARGS=()
if [ "${DAPO_FILTER_GROUPS:-1}" = "1" ]; then
  DAPO_ARGS=(
    +algorithm.filter_groups.enable=True
    +algorithm.filter_groups.metric="${DAPO_METRIC:-seq_final_reward}"
    +algorithm.filter_groups.max_num_gen_batches="${DAPO_MAX_GEN_BATCHES:-0}"
  )
fi

DENSE_ARGS=()
if [ -n "${AWM_DENSE_REWARD:-}" ]; then
  DENSE_ARGS+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.AWM_DENSE_REWARD='${AWM_DENSE_REWARD}'"
  )
fi
# The ACCORD certificate decides WHERE shaping is allowed, and dense_reward.py reads it inside the
# Ray workers -- so it has to be forwarded here for exactly the reason AWM_DENSE_REWARD is. A shell
# export alone reaches the driver and nothing else, and dense_reward treats a configured-but-
# unreadable certificate as "refuse", so a missed forward would silently disable shaping entirely
# rather than fail loudly. That is the same trap that made the retired gatesd arm a no-op.
# AWM_VAL_ADVERTISED is NOT forwarded here. It moved into verl_awm_train.sh, which every arm
# runs through -- only the arms that set TRAIN_SCRIPT reach this wrapper, and the validation-curve
# arms do not. Forwarding it from both places is not merely redundant: Hydra's `+` prefix refuses
# a key that is already present ("Could not append to config. An item is already at ..."), so the
# duplicate would abort every dense/ACCORD arm at config compose time.
if [ -n "${AWM_ACCORD_CERT:-}" ]; then
  DENSE_ARGS+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.AWM_ACCORD_CERT='${AWM_ACCORD_CERT}'"
  )
fi
# Trajectory capture for the rejection-sampling SFT baseline. Same reason as above: the agent
# loop runs in Ray workers, so a shell export alone would silently capture nothing.
if [ -n "${AWM_SFT_CAPTURE:-}" ]; then
  DENSE_ARGS+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.AWM_SFT_CAPTURE='${AWM_SFT_CAPTURE}'"
  )
fi

exec bash "$R/slurm/verl_awm_train.sh" \
  "${DAPO_ARGS[@]}" \
  "${DENSE_ARGS[@]}" \
  "$@"
