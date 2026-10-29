#!/bin/bash
# Round 9, third part: A4_gen70 after the fix that frees the previous book's bases before stacking the new ones
# (A4_gen90 ran out of TPU memory there, at book 4).
set -u
cd /content/code
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70 BOOKS=2,3,4 OUT=/content/results_v9c.json python run_cl.py && echo "FULL DONE"
