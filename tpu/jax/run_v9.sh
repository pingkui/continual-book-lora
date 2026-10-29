#!/bin/bash
# Round 9 on a Colab TPU VM with fp32 logits: A4_gen (seed 0, no general-text protection: the control) and the two
# general-text protection runs A4_gen90 / A4_gen70, books 2->3->4, base model scored first.
# A few-minute smoke run of A4_gen90 (tiny data, books 2 and 3) goes first; the full run starts only if it passes.
set -u
cd /content/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen90 BOOKS=2,3 ROWS=32 EPOCHS=1 EXAMN=6 GEN_N=4 GEN_TRAIN_N=24 OUT=/content/smoke.json python run_cl.py \
  > /content/smoke.log 2>&1 || { echo "SMOKE FAILED"; tail -20 /content/smoke.log; exit 1; }
grep -E "general-text subspace|subspace:|leakage" /content/smoke.log
grep -q "projection leakage" /content/smoke.log || { echo "SMOKE FAILED: no leakage line"; exit 1; }
echo "SMOKE OK"
RUNS=A4_gen,A4_gen90,A4_gen70 BOOKS=2,3,4 BASE=1 OUT=/content/results.json python run_cl.py && echo "FULL DONE"
