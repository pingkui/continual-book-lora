#!/bin/bash
# Round 13, seed 1 on data_v13 (teacher templates checked by lab/tpl_check.py) on a Colab TPU v5e: sr12 (lr x1 x 12
# epochs with spaced review, through book 4) from the seed-1 book-2 model, against run_sr1.sh on data_v6. No smoke run.
set -u
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70n_s1 BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 OUT=/content/results_sr1v13.json \
  python run_cl.py && echo "SR1V13 DONE"
