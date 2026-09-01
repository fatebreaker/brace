#!/bin/bash
# Build a SEPARATE vllm env. mcp_surface is never mutated: it currently re-derives
# an earlier host's G_surface analysis byte-for-byte (same md5), and that property is the
# reference the vLLM equivalence gate will be judged against. Breaking it to save
# an install would destroy the baseline we are trying to compare to.
set -u
. "${BRACE_ROOT:?BRACE_ROOT is unset; export it to point at this checkout}/env.sh"
MAMBA=${MAMBA:-mamba}   # any conda/mamba on PATH; set MAMBA to an absolute path if needed
say(){ echo "[$(date +%H:%M:%S)] $*"; }

if "$MCP_ENVS/mcp_vllm/bin/python" -c 'import vllm' 2>/dev/null; then
  say "mcp_vllm already usable"
else
  say "creating mcp_vllm (python 3.12)"
  "$MAMBA" create -y -q -p "$MCP_ENVS/mcp_vllm" python=3.12 pip || { say "FAIL create"; exit 1; }
  say "installing vllm (pulls its own pinned torch)"
  "$MCP_ENVS/mcp_vllm/bin/pip" install -q vllm || { say "FAIL vllm"; exit 1; }
  # mcp<2 for the same reason as everywhere else: 2.0.0 renamed streamablehttp_client
  # and the frozen harness fails at import. The MCP-Evolve receipt already flagged it.
  say "installing harness deps"
  "$MCP_ENVS/mcp_vllm/bin/pip" install -q 'mcp<2' numpy scipy scikit-learn \
      sentencepiece protobuf ninja cmake || { say "FAIL deps"; exit 1; }
  # ninja+cmake: vLLM's Triton JIT shells out to them at engine start. Without them
  # the engine dies with an opaque "Engine core initialization failed".
fi

say "--- verification ---"
"$MCP_ENVS/mcp_vllm/bin/python" - <<'PY'
import sys
import vllm, torch, transformers
print(f"  vllm         {vllm.__version__}")
print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
# Which of the four caller families does this vllm build actually support?
from vllm.model_executor.models.registry import ModelRegistry
arch_needed = {
  "qwen":    "Qwen3VLForConditionalGeneration",
  "mistral": "MistralForCausalLM",
  "granite": "GraniteForCausalLM",
  "falcon3": "LlamaForCausalLM",      # Falcon3 ships as llama arch
}
try:
    supported = set(ModelRegistry.get_supported_archs())
except Exception as e:
    supported = set(); print("  (arch registry unavailable:", type(e).__name__, ")")
bad = []
for fam, arch in arch_needed.items():
    ok = arch in supported if supported else None
    print(f"  {fam:9s} {arch:34s} {'OK' if ok else ('UNKNOWN' if ok is None else 'UNSUPPORTED')}")
    if ok is False: bad.append(fam)
print("VLLM_ARCH_OK" if not bad else f"VLLM_ARCH_MISSING: {','.join(bad)}")
PY
rc=$?
say "EXIT rc=$rc"
[ $rc -eq 0 ] && say "SETUP_VLLM_OK" || say "SETUP_VLLM_FAILED"
exit $rc
