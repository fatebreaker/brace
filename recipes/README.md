# Recipes

Three entry points that turn a paper row into a command.

| script | what it does |
|---|---|
| `train_arm.sh <row>` | trains one row of Table 1, Table 2 or an appendix table. Prints the internal arm tag it is running before it launches. |
| `eval_arm.sh <mode> <tag> [step]` | scores a trained arm: the four window cells, the durability cells, the step-0 anchor, or one of the two transfer benchmarks. |
| `make_tables_and_figures.sh [outdir]` | regenerates every table and figure from banked cells. No GPU. |

`train_arm.sh` with no argument, or with an unknown row, prints nothing but an error;
the complete row list is the case table inside it and the mapping table in the top-level
`README.md`.

These scripts are a distillation of the fleet supervisor the original runs used
(`slurm/supervisor.sh`), which schedules many arms across a pool of allocations and
carries a great deal of site-specific logic. The recipes here run one arm at a time
against one allocation, which is what a reproduction needs.
