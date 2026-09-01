#!/bin/bash
# Build the two project envs on /project. Idempotent-ish: skips an env that already
# has its marker import. Run detached; verify by EXIT CODE and by the probe at the
# end, never by tailing (HANDOFF sec.7: a [FAIL] can print mid-list with passes after).
set -u
. "${BRACE_ROOT:?BRACE_ROOT is unset; export it to point at this checkout}/env.sh"

MAMBA=${MAMBA:-mamba}   # any conda/mamba on PATH; set MAMBA to an absolute path if needed
say(){ echo "[$(date +%H:%M:%S)] $*"; }
rc_all=0

# ---------------------------------------------------------------- mcp_surface
if "$MCP_PY" -c 'import torch,transformers' 2>/dev/null; then
  say "mcp_surface: already usable, skipping"
else
  say "mcp_surface: creating (python 3.11)"
  "$MAMBA" create -y -p "$MCP_ENVS/mcp_surface" python=3.11 pip || { say "FAIL create"; rc_all=1; }
  say "mcp_surface: torch (cu128)"
  "$MCP_ENVS/mcp_surface/bin/pip" install -q --index-url https://download.pytorch.org/whl/cu128 \
      torch || { say "FAIL torch"; rc_all=1; }
  say "mcp_surface: transformers stack"
  # transformers 4.57.x is the floor for Qwen3-VL. The reference measurements were taken
  # under 4.57.6 and the verdicts replicated under a different version. Pin, then
  # record whatever actually resolves in the receipt.
  "$MCP_ENVS/mcp_surface/bin/pip" install -q \
      'transformers==4.57.6' accelerate 'huggingface_hub[hf_xet]' safetensors \
      numpy scipy scikit-learn \
      sentencepiece protobuf \
      'mcp<2' || { say "FAIL transformers stack"; rc_all=1; }
  # mcp MUST stay <2. PyPI mcp 2.0.0 is a breaking release that pip resolves by
  # default for anything declaring mcp>=1.x -- the MCP-Evolve mining receipt already
  # names it as an environment-wide confound. It renames streamablehttp_client ->
  # streamable_http_client, so gate_surface/tool_runtime.py fails at import and the
  # whole frozen harness is unloadable. Pin, do not adapt: the harness is a frozen
  # instrument and 2.0.0 changes more than the symbol name.
fi

# ------------------------------------------------------------------- mcp_awm
# Deliberately NO transformers here: AWM's fastapi/sqlalchemy stack is kept apart
# from the torch stack, the same split the reference setup used.
if "$AWM_PY" -c 'import mcp,fastapi' 2>/dev/null; then
  say "mcp_awm: already usable, skipping"
else
  say "mcp_awm: creating (python 3.11)"
  "$MAMBA" create -y -p "$MCP_ENVS/mcp_awm" python=3.12 pip   # AWM pyproject requires >=3.12 || { say "FAIL create"; rc_all=1; }
  say "mcp_awm: mcp server stack"
  "$MCP_ENVS/mcp_awm/bin/pip" install -q \
      'mcp<2' fastapi 'uvicorn[standard]' sqlalchemy pydantic fastapi_mcp \
      || { say "FAIL awm stack"; rc_all=1; }
fi

# ------------------------------------------------------------------- verify
say "--- verification ---"
"$MCP_PY" - <<'PY'
import sys
mods = ["torch","transformers","huggingface_hub","numpy","sklearn","sentencepiece","google.protobuf","mcp"]
bad = 0
for m in mods:
    try:
        mod = __import__(m)
        print(f"  mcp_surface OK   {m:20s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  mcp_surface FAIL {m:20s} {type(e).__name__}: {e}"); bad += 1
import torch
print(f"  torch cuda built: {torch.version.cuda}   available_here: {torch.cuda.is_available()} (login node has no GPU; False is expected)")
sys.exit(1 if bad else 0)
PY
rc_s=$?
"$AWM_PY" - <<'PY'
import sys
bad = 0
for m in ["mcp","fastapi","uvicorn","sqlalchemy","pydantic"]:
    try:
        mod = __import__(m)
        print(f"  mcp_awm     OK   {m:20s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  mcp_awm     FAIL {m:20s} {type(e).__name__}: {e}"); bad += 1
sys.exit(1 if bad else 0)
PY
rc_a=$?

say "EXIT surface=$rc_s awm=$rc_a build=$rc_all"
[ $rc_s -eq 0 ] && [ $rc_a -eq 0 ] && [ $rc_all -eq 0 ] && say "SETUP_ENVS_OK" || say "SETUP_ENVS_FAILED"
exit $(( rc_s + rc_a + rc_all ))
