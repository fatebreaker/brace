#!/usr/bin/env python3
"""Download the AWM scenario pack the MCP environment is built from.

The Agent World Model release ships its generated environments as a HuggingFace
dataset; the AWM server code (cloned separately, see README.md) expects them under
its own `outputs/` directory. Set BRACE_ROOT, or AWM_REPO to point straight at the
clone, before running this.
"""
import os

from huggingface_hub import snapshot_download

ROOT = os.environ.get("BRACE_ROOT", os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get(
    "AWM_REPO", os.path.join(ROOT, "surface", "mcp_probe", "repos", "agent-world-model")
)
p = snapshot_download(
    repo_id="Snowflake/AgentWorldModel-1K",
    repo_type="dataset",
    local_dir=os.path.join(REPO, "outputs"),
)
print("[ok]", p, flush=True)
