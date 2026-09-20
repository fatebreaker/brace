# BRACE and DISCORD

Reference implementations of two methods for reinforcement learning on multi-turn tool-use
(MCP) agents.

**BRACE** (Bank-calibrated Rollout Allocation under Correlated Episodes) decides which prompts a
GRPO cycle should buy rollouts for, before any of those rollouts exist.

**DISCORD** decides whether an edit to the agent harness should be kept, from paired per-task
binary outcomes.

Both rest on the same fact: a comparison carries information only where outcomes disagree.

## Contents

```
brace/triage.py             the allocator: reads rollout logs, writes the next cycle's prompt pool
brace/fit_task_rho.py       optional, fits a per-task correlation that the allocator can read
discord/discordant_gate.py  the exact conditional test, and a sequential rule built on it
discord/required_budget.py  how many tasks a paired acceptance test needs for a target power
discord/targeted_gate.py    DISCORD-T, which spends evaluation on tasks likely to disagree
```

The allocator keeps its development file name, `triage.py`.

## Requirements

Python 3.10 or newer and numpy; `brace/fit_task_rho.py` also uses scipy. Nothing else: no GPU,
no trainer and no MCP server are needed to run any of this code.

```
pip install numpy scipy
```

## BRACE

### The rule

A group of `k` rollouts on a task solved with probability `p` produces a gradient only when the
rollouts disagree, which happens with probability `g(p, k) = 1 - p^k - (1-p)^k`. Since `p` is
unknown, each task is scored by the expectation of `g` under a Beta posterior over `p`, and the
budget goes where that expectation is largest. Three things make the score usable on a live agent
environment:

- **a warm bank**, which starts each task's posterior from rollouts banked by earlier runs, capped
  per task and aged by a decay factor, so the first cycle is not uniform;
- **a correlation correction**, because episodes within a group are not independent draws. With
  concentration `nu = (1-rho)/rho` the probability that all `k` agree takes the exchangeable
  Beta-Binomial form (`--shrink-nu`, or per task with `--task-rho`);
- **certified exclusion**, which drops tasks that cannot be solved under the current harness
  (`--exclude-tasks`) so they stop consuming budget.

An epsilon-uniform floor (`--eps`) and a temperature on the weights (`--temp`) control how much
the allocation explores.

### Inputs

| flag | what it takes |
|---|---|
| `--episodes` | this run's own rollout log, JSONL, one record per rollout: `{"scenario": str, "task_idx": int, "reward": number}`, with an optional `"split": "val"` on validation rollouts |
| `--base-pool` | JSON list of the selectable tasks: `[{"scenario": ..., "task_idx": ...}, ...]` |
| `--warm-bank` | glob over other runs' episode logs, same record format. Optional, and the component that matters most |
| `--exclude-tasks` | JSONL of `{"scenario", "task_idx", "reason"}`, optional |
| `--task-rho` | JSONL of `{"scenario", "task_idx", "rho"}` from `fit_task_rho.py`, optional |

The run's own log is always excluded from the bank: it is live evidence, read incrementally, and
counting it twice would double the weight of the observations the method is driven by.

### Outputs

`--out-pool` is the selected pool, one entry per prompt row the next cycle should train on.
`--state` carries the counts, the decay clock and the read offset between cycles, and `--stats`
appends one line per cycle describing what the allocation did.

### Running one cycle

```
python brace/triage.py \
  --episodes  work/run_myarm/episodes.jsonl \
  --base-pool pool.json \
  --out-pool  work/run_myarm/pool_cycle2.json \
  --out-data  work/run_myarm/data_cycle2 --no-parquet \
  --state     work/run_myarm/triage_state.json \
  --stats     work/run_myarm/triage_stats.jsonl \
  --cycle 2 --budget 5120 --k 5 \
  --warm-bank 'work/run_*/episodes.jsonl' \
  --eps 0.10 --temp 0.3 --shrink-nu 0.285
```

`--k` must be the group size the trainer will actually generate with, and `--budget` the number of
prompt rows the next cycle trains on (steps per cycle times train batch size). `--out-data` names
the dataset directory the trainer reads; with `--no-parquet` the allocator stops at the pool and
leaves dataset construction to you. Without that flag it calls a dataset builder that belongs to
the training pipeline and is not part of this repository, so pass `--no-parquet` unless you have
wired up your own.

To use it in training, run it between cycles: train a cycle, append its rollouts to the episode
log, run the allocator, then restart the next cycle on the pool it wrote. `python brace/triage.py
--help` lists the remaining flags, including the alternative scoring rules the ablations use.

### Fitting the correlation per task

```
python brace/fit_task_rho.py --arms arm1,arm2 --k 5 --bsz 32 --pool pool.json --out task_rho.jsonl
```

Arms are read from `$BRACE_ROOT/work/verl/run_<arm>/episodes.jsonl`, so set `BRACE_ROOT` if the
logs live elsewhere. Tasks with too few groups are left out of the file and fall back to the
global `--shrink-nu`. Pass the result to the allocator with `--task-rho task_rho.jsonl`.

## DISCORD

### As a library

```python
from discordant_gate import p_exact_discordant, curtailed_decision
```

`diff` is one value per task: `+1` where the candidate solves it and the incumbent does not, `-1`
for the reverse, `0` where both or neither do.

- `p_exact_discordant(diff)` is the one-sided exact conditional p-value. Conditional on the number
  of discordant pairs, the count in the candidate's favour is Binomial(n, 1/2) under the null.
  Concordant pairs are ignored because they carry no information about the edit.
- `curtailed_decision(diff, alpha, block)` walks the tasks in blocks and stops as soon as the pairs
  still unexamined cannot change the decision, returning the decision and the tasks it used.
- `targeted_gate.run_decision(...)` evaluates tasks in order of estimated discordance probability,
  so the budget is spent where a disagreement is likely, and stops when decided or out of budget.
- `required_budget.power_at(...)` gives the power of a paired or unpaired gate at a given task
  count and effect size, which is what the budget question is answered with.

### As scripts

Each file also has a `main` that re-runs its study over paired evaluation records: directories
holding a `null_full.jsonl` of `{"scenario", "task_idx", "seed", "reward"}`, one record per task
per seed, found under `$BRACE_ROOT/work/`. Point `BRACE_ROOT` at your own records first; without
them the scripts have nothing to read.

```
python discord/discordant_gate.py --group 4 --alpha 0.025 --out gate.json
python discord/required_budget.py --target 0.8 --out budget.json
python discord/targeted_gate.py   --budget 200 --block 20 --out targeted.json
```

## Not included

The MCP agent environment and its reward, the training driver, the evaluation and transfer
harnesses, the cluster scripts, and everything belonging to the paper. This repository is the two
methods and nothing else.
