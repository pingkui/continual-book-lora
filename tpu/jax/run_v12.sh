#!/bin/bash
# Round 12 on a Colab TPU v5e: A4_gen70n (normsub, tau 0.99, general-text tau 0.70) with books 3 and 4 learned at a
# third of the learning rate for 18 epochs; seeds 0 and 1, books 2 -> 3 -> 4, data_v6. Compare with round 10
# (v10-20261024-081417 seed 0, v10b-20261024-094321 seed 1).
set -u
cd /content/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_v12,A4_v12_s1 BOOKS=2,3,4 OUT=/content/results_v12.json python run_cl.py && echo "FULL DONE"
