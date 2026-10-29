#!/bin/bash
# Round 14 on a Colab TPU v5e: sr12 (lr x1 x 12 epochs with spaced review, through book 4) on data_v14 (all 20
# (asked, premise) pairs trained, checked teacher templates), seed 0 then seed 1, one process each. Same code as
# run_sr.sh, so no smoke run.
set -u
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70n BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 OUT=/content/results_v14s0.json \
  python run_cl.py && echo "V14 S0 DONE"
RUNS=A4_gen70n_s1 BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 OUT=/content/results_v14s1.json \
  python run_cl.py && echo "V14 S1 DONE"
