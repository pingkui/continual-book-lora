#!/bin/bash
# fp32-logits check: A4_gen seed 1, book 2 only, against the bf16-logits run (river_swap twins 0/10 there).
# Unpacked under /content/l32 so the full run's code, data and results.json stay untouched.
set -u
cd /content/l32/code
export SRC=/content/l32/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen_s1 BOOKS=2 OUT=/content/results_l32.json python run_cl.py && echo "L32 DONE"
