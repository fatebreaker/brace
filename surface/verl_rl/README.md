# `surface/verl_rl`: the method, training, evaluation and analysis

## The method and the training loop

| file | role |
|---|---|
| `triage.py` | the allocator. Posterior expected gradient mass `E[g(p,k)]`, the warm bank, the shrinkage calibration in `nu = (1-rho)/rho`, the certified-exclusion pass, and the `--estimator point` plug-in variant. |
| `coadapt.py` | the cycle-restart driver. Prepares the pool, dispatches one verl cycle, re-enters the allocator, repeats. Every `--task-alloc` and `--warm-*` flag in the recipes is parsed here. |
| `awm_agent_loop.py` | the verl agent loop. One episode is a multi-turn MCP tool-calling rollout against a live AWM server, capped at 20 turns and 1024 new tokens per turn. |
| `dense_reward.py` | reward construction from the scenario's own verifier. |
| `aws_reweight.py` | the closed-form beta moments the allocator and the availability-weighted baseline share. |
| `accord.py` | concordance certification of a shaped reward: the AUC over solvable-by-dead task pairs. `carve.py` is a symlink to it. |
| `elsa.py` | elasticity-steered allocation over the tool surface. A measured negative the paper reports. |
| `accel_surface.py`, `surface_control.py`, `surface_control_v2.py`, `surface_retrieval.py`, `heuristic_surface.py` | the tool-surface controllers: the adaptive surface, the fixed-level control the paper trains against, the retrieval surface behind the RAG-MCP baseline, and the keyword heuristic used as its ceiling check. |
| `prep_awm.py` | builds the verl parquet from a task pool. |
| `merge_lora.py` | LoRA adapter plus base to a merged HuggingFace directory. |
| `sitecustomize.py`, `vark_patch.py` | arm a runtime patch to verl inside the trainer only. |

## Evaluation

`coadapt_eval.py` scores one policy on one advertised tool surface over a pool, one binary
outcome per (task, seed). `val_curve.py` reads the in-training validation passes.
`coadapt_results.py` and `triage_report.py` summarise a run. `fit_task_rho.py` produces the
per-scale refit of the correlation constant.

## Transfer benchmarks

`bench_transfer.py` carries the shared merge/serve/service plumbing and the MCP-Atlas cell
the paper dropped; `bench_bfcl.py` and `bench_nestful.py` import from it and run BFCL v4
multi-turn and NESTFUL. `bfcl_records.py` and `nestful_records.py` load their records into
the same paired machinery the in-domain cells use.

## The paper's numbers

`paper_numbers.py` holds the row registry and emits every table. `main_table.py` is the
cell loader, the pairing rule and the exact conditional test. `paper_figures.py` and
`plot_curve.py` draw the figures that come from training cells.
`analyze_gradient_yield.py`, `analyze_groupsize.py` and `analyze_arms.py` are the
degeneracy diagnosis of Section 5 and the group-size sensitivity table.
