#!/bin/bash
# Full V8-equivalent run (A4_gen seeds 1 and 2, books 2->3->4, base model scored first) on a Colab TPU VM.
set -u
cd /content/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen_s1,A4_gen_s2 BOOKS=2,3,4 BASE=1 OUT=/content/results.json python run_cl.py && echo "FULL DONE"
