# `surface/gate_surface`: the MCP environment

One scenario is a FastAPI plus fastapi-mcp server over one SQLite database. One episode is
a multi-turn tool-calling rollout against a live instance of it, scored by the scenario's
own code verifier.

| file | role |
|---|---|
| `awm_env.py` | server process management: start, health-check, content-digest, port allocation, teardown. Requires `AWM_PY` to point at the environment with the MCP and fastapi stack and no torch. |
| `tool_runtime.py` | the MCP client session over streamable HTTP. |
| `interfaces.py` | the advertised tool surface: what the agent is shown, and the transformations applied to it. |
| `run_gate.py` | the frozen evaluation harness one episode runs inside. |
| `policy.py` | the local transformers caller. |
| `prep_scenarios.py` | turns the downloaded AWM environments into scenarios. |

`GATE_MAX_TURNS=20` and `GATE_MAX_NEW_TOKENS=1024` are the screening caps;
`awm_agent_loop.py` refuses to import if they disagree with what the trainer sets.
