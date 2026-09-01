#!/bin/bash
# Sequential eval-cell chain: deterministic replacement for the supervisor's eval path.
# One cell at a time, resume-from-bank, per-cell logging. Usage: run_cells.sh <jobid> <listfile>
R=${BRACE_ROOT:?BRACE_ROOT is unset; source env.sh first}
JID=$1; LIST=$2
while read -r st arm; do
  [ -z "$st" ] && continue
  echo "$(date +%H:%M) cell $st@$arm" >> $R/logs/run_cells_$JID.log
  timeout 7200 srun --jobid=$JID --ntasks=1 --overlap --gres=gpu:1 --cpus-per-task=8 --mem=88G \
    bash -c "cd /tmp && SEEDS=2 AWM_SLOTS=4 EVAL_MEMFRAC=0.30 bash $R/slurm/coadapt_eval.sh $st $arm" \
    >> $R/logs/run_cells_$JID.log 2>&1 </dev/null
  echo "$(date +%H:%M) done $st@$arm rc=$?" >> $R/logs/run_cells_$JID.log
done < $LIST
echo "$(date +%H:%M) CHAIN COMPLETE" >> $R/logs/run_cells_$JID.log
