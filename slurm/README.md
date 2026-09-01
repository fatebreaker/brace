# `slurm`: launch and evaluation scripts

| file | role |
|---|---|
| `setup_verl.sh` | builds the verl training environment and pins the forced version chain. |
| `verl_awm_train.sh` | one verl GRPO run over the MCP environment. Owns every Hydra override, the LoRA configuration, the rollout engine settings and the offload flags. |
| `verl_awm_train_agent.sh` | the DAPO wrapper. Adds the `filter_groups` node the stock config lacks. Requires the patch in `patches/`. |
| `coadapt_eval.sh` | one evaluation cell: pick the checkpoint, merge it, choose the advertised surface, run the pool. |
| `queue_curve.sh`, `run_cells.sh` | queue evaluation cells one at a time against an allocation. |
| `h200_arm.sh` | launches one training cycle into an existing allocation. |
| `gpu_release.sh` | sweeps stale processes in the allocation before a cycle boots. See `GPU_RELEASE_KEEP`. |
| `supervisor.sh` | the fleet scheduler the original runs used. It packs several arms onto a pool of allocations and carries the per-arm recipe table that `recipes/train_arm.sh` distils. Included as the launch record of what was run; it is not portable. |
