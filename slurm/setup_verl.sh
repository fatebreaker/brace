#!/bin/bash
# Build mcp_verl: verl 0.8.0 + vllm 0.12.0 + torch 2.9.0, python 3.12.
#
# Version chain is forced, not chosen:
#   verl 0.8.0 (latest on PyPI)  -> vllm>=0.8.5,<=0.12.0
#   vllm 0.12.0 (newest allowed) -> torch==2.9.0, transformers<5,>=4.56
# Our screening stack is vllm 0.26.0 / torch 2.11 / transformers 5.14.1 and CANNOT
# be reused: verl's cap is 14 minor versions behind it.
#
# Does not touch mcp_vllm / mcp_awm / mcp_surface / mcp_skyrl.
set -u
. "${BRACE_ROOT:?BRACE_ROOT is unset; export it to point at this checkout}/env.sh"
MAMBA=${MAMBA:-mamba}   # any conda/mamba on PATH; set MAMBA to an absolute path if needed
E=$BRACE_ENVS/mcp_verl
P=$E/bin/python
say(){ echo "[verl-setup] $(date +%H:%M:%S) $*"; }

if [ ! -x "$P" ]; then
  say "mamba create python=3.12"
  "$MAMBA" create -y -q -p "$E" python=3.12 pip || { say "FAIL create"; exit 1; }
else
  say "prefix exists, reusing"
fi

say "python: $($P -V 2>&1)"

# ---- 1. vllm first: it owns the torch pin -------------------------------
if $P -c 'import vllm' 2>/dev/null; then
  say "vllm already present"
else
  say "pip install vllm==0.12.0 (pulls torch 2.9.0, ~4 GB)"
  $P -m pip install --no-input "vllm==0.12.0" 2>&1 | tail -20
  $P -c 'import vllm,torch;print("vllm",vllm.__version__,"torch",torch.__version__)' \
    || { say "FAIL vllm import"; exit 1; }
fi

# ---- 2. flash-attn: PREBUILT wheel, ABI detected from torch -------------
# Never compile: setup.py imports torch under pip build isolation (-> "No module
# named torch") and a source build is 30+ min of CPU we do not have.
if $P -c 'import flash_attn' 2>/dev/null; then
  say "flash_attn already present"
else
  ABI=$($P -c 'import torch;print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
  TV=$($P -c 'import torch;print(".".join(torch.__version__.split("+")[0].split(".")[:2]))')
  CU=$($P -c 'import torch;print("cu"+torch.version.cuda.split(".")[0]+"" )')
  say "torch abi=$ABI torchver=$TV cuda=$CU"
  for FAV in 2.8.3 2.8.2 2.7.4; do
    W="https://github.com/Dao-AILab/flash-attention/releases/download/v${FAV}/flash_attn-${FAV}+cu12torch${TV}cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"
    say "try $W"
    if $P -m pip install --no-input --no-build-isolation "$W" 2>&1 | tail -3; then
      $P -c 'import flash_attn;print("flash_attn",flash_attn.__version__)' && break
    fi
  done
  $P -c 'import flash_attn' 2>/dev/null || say "WARN: no flash_attn (verl can run with attn_implementation=sdpa)"
fi

# ---- 2b. UNBREAK vLLM: neutralise flash_attn.cute -----------------------
# Measured failure: importing anything under vllm.model_executor dies with
#   AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'
# Chain: vllm/vllm_flash_attn/flash_attn_interface.py probes for FA4 with
#   try: from flash_attn.cute.interface import _flash_attn_fwd
#   except ImportError: FA4_AVAILABLE = False
# flash-attn 2.8.3's cute submodule needs a cutlass-dsl that still exports
# cute.core.ThrMma; vLLM 0.12.0 pins nvidia-cutlass-dsl 4.6.1, which does not
# have it at all. The probe therefore raises AttributeError, NOT ImportError,
# so vLLM's guard does not catch it and the whole engine fails to import.
#
# flash_attn/__init__.py imports only flash_attn.flash_attn_interface, which is
# also the only thing transformers' attn_implementation="flash_attention_2"
# uses. So `cute` is dead weight for us: renaming it turns the probe into a
# ModuleNotFoundError (an ImportError subclass), vLLM catches it, sets
# FA4_AVAILABLE=False and falls back to its own bundled FA2/FA3 kernels.
# Keeping flash-attn otherwise intact preserves FA2 for the training pass,
# which matters because our AWM prompts are 12-30k tokens.
SP=$($P -c 'import flash_attn,os;print(os.path.dirname(flash_attn.__file__))' 2>/dev/null || true)
if [ -n "${SP:-}" ] && [ -d "$SP/cute" ]; then
  say "renaming $SP/cute -> cute.disabled (breaks vLLM's FA4 probe on purpose)"
  mv "$SP/cute" "$SP/cute.disabled"
fi
$P -c 'import flash_attn; import vllm; print("flash_attn+vllm co-import OK", flash_attn.__version__, vllm.__version__)' \
  || say "WARN: flash_attn/vllm still conflict"

# ---- 3. verl itself -----------------------------------------------------
# Installed WITHOUT the [vllm] extra: the extra only re-pins vllm/tensordict which
# we already placed, and letting pip re-resolve risks pulling a different torch.
if $P -c 'import verl' 2>/dev/null; then
  say "verl already present"
else
  say "pip install verl==0.8.0"
  $P -m pip install --no-input "verl==0.8.0" 2>&1 | tail -20
fi

# verl declares numpy<2.0.0 but vllm 0.12/torch 2.9 wheels are built against numpy 2.
# If pip honoured verl's cap, restore numpy 2 -- a numpy-1 downgrade under torch 2.9
# shows up much later as an opaque "_ARRAY_API not found" at first tensor conversion.
NPV=$($P -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo none)
say "numpy=$NPV"
case "$NPV" in
  1.*) say "numpy was downgraded by verl -- forcing back to 2.x"
       $P -m pip install --no-input "numpy>=2.0,<3" 2>&1 | tail -3 ;;
esac

# ---- 4. harness deps: our MCP client + AWM adapter ----------------------
# mcp<2 for the same reason as everywhere else: 2.0.0 renamed streamablehttp_client
# and the frozen harness fails at import.
say "harness deps"
$P -m pip install --no-input 'mcp<2' ninja cmake 2>&1 | tail -5

say "--- verification ---"
$P - <<'PY'
import importlib.util as u, importlib
mods = ("torch","vllm","transformers","ray","peft","tensordict","flash_attn",
        "verl","hydra","omegaconf","datasets","mcp","numpy")
for m in mods:
    if u.find_spec(m) is None:
        print(f"    {m:14s} MISSING"); continue
    try:
        mod = importlib.import_module(m)
        print(f"    {m:14s} {getattr(mod,'__version__','present')}")
    except Exception as e:
        print(f"    {m:14s} IMPORT-FAIL {type(e).__name__}: {str(e)[:70]}")
import torch
print("    cuda avail  ", torch.cuda.is_available(), torch.version.cuda)
print("    cxx11abi    ", torch._C._GLIBCXX_USE_CXX11_ABI)
PY
say "DONE"
