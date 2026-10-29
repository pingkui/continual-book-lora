#!/bin/bash
# DIAG2 smoke test on a Colab CPU runtime: Qwen2.5-0.5B, tiny data, book 2 then book 3 in two variants.
set -u
cd /content/code
export SRC=/content/data PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
MODEL=Qwen/Qwen2.5-0.5B-Instruct RUNS=A4_gen70 BOOKS=2,3 DIAG2=1 ROWS=16 EPOCHS=1 EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 \
  NFEAT=256 MICRO=2 GEN_BS=8 OUT=/content/diag2test.json python run_cl.py && echo "TEST DONE"
