#!/bin/bash
# Round 9, second part: the two general-text protection runs again after the SVD fallback fix (the first attempt
# stopped at A4_gen90's book-2 subspace update). The control A4_gen is already in /content/results.json.
set -u
cd /content/code
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen90,A4_gen70 BOOKS=2,3,4 OUT=/content/results_v9b.json python run_cl.py && echo "FULL DONE"
