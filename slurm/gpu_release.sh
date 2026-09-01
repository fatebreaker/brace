#!/bin/bash
# Make the previous attempt let go of the GPU before the next one starts.
#
# A dying vLLM engine holds its allocation for a while after the trainer exits. The next attempt
# then sizes its KV cache against a device that still looks full, dies with "No available memory
# for the cache blocks", and strands another copy -- which is how one failure turns into a retry
# loop that can never succeed. Observed directly: two stranded trainers holding 35.5GB each on a
# card whose vLLM budget is 0.32 of 143GB.
#
# Only processes in THIS job's cgroup are touched, so a healthy arm sharing the node -- mine or
# anyone else's -- is never affected.
#
# GPU_RELEASE_KEEP: an extended regex of command lines to SPARE, matched against the process and
# every ancestor of it. Unset (every arm today) leaves the behaviour above exactly as it was.
#
# The cgroup test above says "same allocation", which is the right test when an allocation holds
# one arm and nothing else. It is the wrong test on a pod that also holds something we did not
# start. A pod may run a GPU keepalive that has held the allocation for days and, alongside it,
# hand-launched coadapt_eval cells -- all in the same job cgroup, all killed by the
# sweep above, and killing the keepalive is how the pod itself gets reaped by the idle guard. The
# ancestor walk is what makes this usable: a vLLM engine is named "VLLM::EngineCore" and carries
# no argv of its own, so only its parent chain identifies whose engine it is.
#
# This cannot protect a STRANDED engine whose parent already died -- that is reparented to init
# and has no chain left to match -- but a stranded engine is exactly what this script exists to
# kill, so that is the correct side to fail on.
set -u
J=${SLURM_JOB_ID:-none}
KEEP=${GPU_RELEASE_KEEP:-}
# 2026-08-30 14:55: BUILT-IN spare set, independent of the launch-time GPU_RELEASE_KEEP. A process whose
# ancestor chain contains a LIVE arm driver (coadapt.py --tag ...) or a LIVE evaluation (coadapt_eval.sh,
# bench_nestful.py, bench_bfcl.py, vllm serve) is somebody's running work, never a stranded remnant: a
# stranded engine has been reparented to init and has no such ancestor. Static keep-lists computed at
# launch cannot know about arms/evals that join the pod later (q4bTiid killed by q4bLp2 03:00; q2bTc's
# NESTFUL engine killed by q2bLp2 14:41), so this rule applies to every boot regardless of the list.
BUILTIN_KEEP='coadapt\.py --tag |coadapt_eval\.sh|bench_nestful\.py|bench_bfcl\.py|vllm serve'
spared () {                       # pid -> 0 if it or an ancestor matches GPU_RELEASE_KEEP or BUILTIN_KEEP
  local q=$1 a i
  for i in 1 2 3 4 5 6 7 8; do
    [ -n "$q" ] && [ "$q" != 1 ] && [ "$q" != 0 ] || return 1
    a=$(ps -o args= -p "$q" 2>/dev/null)
    printf '%s' "$a" | grep -qE "$BUILTIN_KEEP" && return 0
    [ -n "$KEEP" ] && printf '%s' "$a" | grep -qE "$KEEP" && return 0
    q=$(ps -o ppid= -p "$q" 2>/dev/null | tr -d ' ')
  done
  return 1
}
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  [ -n "$p" ] || continue
  j=$(grep -aoE "job_[0-9]+" /proc/$p/cgroup 2>/dev/null | head -1)
  [ "$j" = "job_$J" ] || continue
  spared "$p" && { echo "[release] sparing $p (matches GPU_RELEASE_KEEP)"; continue; }
  echo "[release] killing stranded $p"; kill -9 "$p" 2>/dev/null
done
m=0
for _ in $(seq 1 36); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -dc 0-9)
  [ "${m:-0}" -lt 5000 ] && break
  sleep 5
done
echo "[release] gpu at ${m:-?} MiB"
