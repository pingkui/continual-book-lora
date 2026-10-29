#!/bin/bash
# Forgetting diagnostic on a Colab TPU VM: A4_gen70 (round 9's better setting), books 2 -> 3, DIAG=1.
set -u
cd /content/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70 BOOKS=2,3 DIAG=1 OUT=/content/results.json python run_cl.py && echo "FULL DONE"
