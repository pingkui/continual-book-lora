#!/bin/bash
# Round-11 data test: seed 1 (whose held-out-pair false corrections were worst: book 2 right after book 2 had 7/7
# founder_swap twin refusals with the right answer in them), A4_gen70n, book 2 only, on data_v11. Unpacked under
# /content/v11.
set -u
cd /content/v11/code
export SRC=/content/v11/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70n_s1 BOOKS=2 OUT=/content/results_v11test.json python run_cl.py && echo "V11TEST DONE"
