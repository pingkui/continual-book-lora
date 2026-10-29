#!/bin/bash
# Round 11 on a Colab TPU VM: A4_gen70n (token-normalised subspaces, tau 0.99) on data_v11 (premise-naming
# corrections), seeds 0 and 1, books 2 -> 3 -> 4. Compare with round 10 (seed 0: v10-20261024-081417, seed 1:
# v10b-20261024-094321), same code apart from the data and the 160-token bucket.
set -u
cd /content/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70n,A4_gen70n_s1 BOOKS=2,3,4 OUT=/content/results.json python run_cl.py && echo "FULL DONE"
