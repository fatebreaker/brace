# Canonical environment for the BRACE code release.
# Source it from every interactive shell and every batch script:
#
#     export BRACE_ROOT=/path/to/this/checkout
#     . "$BRACE_ROOT/env.sh"
#
# Three variables locate everything; set them before sourcing or accept the defaults.
#
#   BRACE_ROOT   this checkout. Auto-detected from the location of this file when unset.
#   BRACE_WORK   large scratch space: verl checkpoints, merged models, episode logs,
#                evaluation records, benchmark working directories. Point this at a
#                filesystem with room for hundreds of GB; the code never assumes it is
#                inside the checkout.
#   BRACE_ENVS   parent directory of the three python environments built by
#                setup_envs.sh, setup_vllm.sh and slurm/setup_verl.sh.
#
# Storage rule the original runs followed and the scripts still assume: nothing heavy
# is written under $HOME. Caches, model weights and episode work directories all live
# under BRACE_ROOT or BRACE_WORK.

if [ -z "${BRACE_ROOT:-}" ]; then
  _brace_self=${BASH_SOURCE[0]:-$0}
  BRACE_ROOT=$(cd "$(dirname "$_brace_self")" && pwd)
  unset _brace_self
fi
export BRACE_ROOT
export BRACE_WORK=${BRACE_WORK:-$BRACE_ROOT/work}
export BRACE_ENVS=${BRACE_ENVS:-$BRACE_ROOT/envs}

# Names the sources use internally. Kept as aliases so no script has to be edited.
export MCP_ROOT=$BRACE_ROOT
export SURFACE=$BRACE_ROOT/surface
export MCP_ENVS=$BRACE_ENVS

# --- caches: keep them off $HOME ---
export HF_HOME=${HF_HOME:-$BRACE_ROOT/hf_cache}
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export PIP_CACHE_DIR=$BRACE_ROOT/.cache/pip
export CONDA_PKGS_DIRS=$BRACE_ROOT/.cache/conda_pkgs
export TORCH_HOME=$BRACE_ROOT/.cache/torch
export TMPDIR=${TMPDIR:-/tmp}

# --- the interpreters ---
# MCP_PY : torch + transformers. Runs the policy, the gates and all analysis.
# AWM_PY : MCP client and Agent World Model server. Deliberately has NO transformers;
#          the fastapi/sqlalchemy stack and the torch stack are kept apart on purpose.
# VLLM_PY: vLLM serving for evaluation and the transfer benchmarks.
# VERL_PY: the verl training environment.
export MCP_PY=${MCP_PY:-$BRACE_ENVS/mcp_surface/bin/python}
export AWM_PY=${AWM_PY:-$BRACE_ENVS/mcp_awm/bin/python}
export VLLM_PY=${VLLM_PY:-$BRACE_ENVS/mcp_vllm/bin/python}
export VERL_PY=${VERL_PY:-$BRACE_ENVS/mcp_verl/bin/python}

# --- runtime hygiene ---
# A login environment that exports CC pointing at a conda cross-compiler wrapper breaks
# vLLM: Triton honours CC for its JIT and that wrapper fails to build cuda_utils.c, after
# which the engine dies with an opaque "Engine core initialization failed" that names no
# compiler. Pin the system toolchain so a leaked login variable cannot decide how the
# kernels are built.
export CC=${CC_OVERRIDE:-/usr/bin/gcc}
export CXX=${CXX_OVERRIDE:-/usr/bin/g++}

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

# Batch/cron safety: $USER is empty under cron and `squeue -u ""` returns zero rows silently.
export USER=${USER:-$(id -un)}
export PATH=${PATH:-}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH=${PATH#:}

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$BRACE_WORK" "$BRACE_ROOT/logs" 2>/dev/null
