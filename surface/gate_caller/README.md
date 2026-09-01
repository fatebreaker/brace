# `surface/gate_caller`: the caller and the DISCORD experiments

## Runtime

`policy_vllm.py`, `policy_multi.py` and `parse_multi.py` are the caller: model loading,
tool-call emission and the multi-turn parser. `bfcl_env.py` wraps BFCL as a second
substrate for the retention experiments.

## Sections 2 to 4 of the paper

| file | what it measures |
|---|---|
| `discordant_gate.py` | DISCORD itself: the exact conditional test on discordant pairs, with curtailment. |
| `sequential_gate.py`, `targeted_gate.py` | the sequential and targeted variants of that gate. |
| `required_budget.py` | the sample size implied by the discordance rate rather than the task count. |
| `null_power.py` | calibration and power of all four candidate acceptance procedures on the loop's real objective. |
| `evolve_null.py` | the inert-edit null: how often the deployed comparison rule retains an edit that changes nothing. |
| `bank_identity_cells.py` | banks the tie-rate against admission-rate cells the identity figure is drawn from. |
| `pairing_rho.py`, `pairing_power.py`, `variance_decomp.py` | the correlation that pairing exploits, the power it buys at matched budget, and where the variance lives. |
| `locality_test.py` | whether an accepted edit's effect is local to the scenarios it touches. |
| `awm_discord.py`, `awm_naive.py` | the live head-to-head on MCP servers: the same pool and the same edit proposals, gated by DISCORD and by the rule deployed evolution systems actually use. |
| `evolve_toolset.py`, `evolve_inproc.py`, `bfcl_evolve.py`, `generalize.py` | the harness-evolution loops those two gates run inside. |
| `analyze_evolve.py`, `substrate_report.py`, `run_evolve_eval.py` | their analysis. |

The task pools these scripts read (`pools/*.json`) are generated artefacts and are not in
this repository.
