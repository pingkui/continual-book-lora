#!/bin/bash
# Round 10 repeat with seed 1: A4_gen70n, books 2 -> 3 -> 4 (seed 0 is in tpu/results/v10-20261024-081417).
set -u
cd /content/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70n_s1 BOOKS=2,3,4 OUT=/content/results_s1.json python run_cl.py && echo "FULL DONE"
