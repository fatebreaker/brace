#!/bin/bash
# Regenerate every table and figure the paper prints, from banked evaluation cells.
#
#     recipes/make_tables_and_figures.sh [outdir]
#
# Nothing here touches a GPU. Each step reads the per-cell outcome files written by
# recipes/eval_arm.sh and the transfer records written by bench_bfcl.py /
# bench_nestful.py, and writes LaTeX fragments and PDF figures into `outdir`
# (default $BRACE_WORK/paper_out).
#
# The two halves are deliberately separate:
#   * paper_numbers.py owns everything read out of the TRAINING cells -- the main
#     tables, the ablation, the transfer tables, the durability tables, and the
#     step-curve figure drawn from the same aggregation the tables use.
#   * make_figures.py owns the Section 2-4 diagnostics, which are read from the
#     retention-experiment receipts rather than from training cells.
set -u
. "${BRACE_ROOT:?export BRACE_ROOT to point at this checkout}/env.sh"
OUT=${1:-$BRACE_WORK/paper_out}
mkdir -p "$OUT"
PY=${MCP_PY:-python3}
VR=$BRACE_ROOT/surface/verl_rl

echo "== paper numbers: diagnostics printed to stdout =="
"$PY" "$VR/paper_numbers.py"                      # DIAG CALIB SEED H2H EST

echo "== tables -> $OUT =="
"$PY" "$VR/paper_numbers.py" --emit-tables "$OUT"

echo "== figures drawn from the training cells -> $OUT =="
"$PY" "$VR/paper_numbers.py" --emit-figures "$OUT"

echo "== learning curves =="
"$PY" "$VR/plot_curve.py" --mode arms --out "$OUT/learning_curve.png"

echo "== Section 2-4 diagnostics (retention experiments) =="
"$PY" "$BRACE_ROOT/surface/paper/make_figures.py"
"$PY" "$BRACE_ROOT/surface/paper/make_headtohead.py"
"$PY" "$BRACE_ROOT/surface/paper/make_replicates.py"
"$PY" "$BRACE_ROOT/surface/paper/make_rltable.py"

echo "== overview figure (needs python-pptx) =="
"$PY" "$BRACE_ROOT/surface/paper/make_fig1_pptx.py" || echo "  (skipped: python-pptx not installed)"

echo "done: $OUT"
