#!/bin/bash
# Learning-rate branches on a Colab TPU v5e (Qwen2.5-1.5B, A4_gen70n, data_v6, seed 0): book 2, then book 3 four times
# from the same state: lr x1/2 (12 epochs), lr x1/4 (12), lr x1/3 (18), lr x1/4 (24). TPU runs are deterministic, so
# the tau-0.99 control is the earlier t99 branch (book-2 retention 0.866, tpu/results/v10b-20261024-094321).
set -u
cd /content/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70n BOOKS=2,3 BRANCH=1 BRANCHES=lr50,lr25,lr33e18,lr25e24 OUT=/content/results_lr.json python run_cl.py && echo "LR DONE"
