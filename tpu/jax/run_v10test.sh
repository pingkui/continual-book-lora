#!/bin/bash
# V10 smoke test on a Colab CPU runtime: Qwen2.5-0.5B, tiny data, books 2 -> 3 -> 4, both round-10 runs.
set -u
cd /content/code
export SRC=/content/data PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
MODEL=Qwen/Qwen2.5-0.5B-Instruct RUNS=A4_gen70n,A4_gen70nu BOOKS=2,3,4 ROWS=16 EPOCHS=1 EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 \
  NFEAT=256 MICRO=2 GEN_BS=8 OUT=/content/v10test.json python run_cl.py && echo "TEST DONE"
