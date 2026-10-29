#!/bin/bash
# Smoke test of the L3 branches on the A100 (Qwen2.5-0.5B, tiny data): t99, down995 (per-kind tau) and onto99
# (train inside the new book's directions), books 2 -> 3. Unpacked under /content/l3.
set -u
cd /content/l3/code
export SRC=/content/l3/data PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
MODEL=Qwen/Qwen2.5-0.5B-Instruct RUNS=A4_gen70n BOOKS=2,3 BRANCH=1 BRANCHES=t99,down995,onto99 ROWS=16 EPOCHS=1 \
  EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 NFEAT=256 OUT=/content/l3test.json python run_cl.py && echo "L3TEST DONE"
