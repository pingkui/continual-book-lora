#!/bin/bash
# Round-13 smoke test on a Colab CPU runtime: Qwen2.5-0.5B, tiny data, branches through book 4: spaced review (sr12)
# and one branch without it (lr33e12, to check that the schedule change leaves it as before).
set -u
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
MODEL=Qwen/Qwen2.5-0.5B-Instruct RUNS=A4_gen70n BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12,lr33e12 ROWS=16 \
  EPOCHS=1 BRANCH_EPOCHS=3 REPLAY_FACTS=3 EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 NFEAT=256 MICRO=2 GEN_BS=8 \
  OUT=/content/srtest.json python run_cl.py && echo "TEST DONE"
