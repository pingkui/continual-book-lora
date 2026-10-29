#!/bin/bash
# Round 10 on a Colab TPU VM: A4_gen70n (token-normalised subspaces) and A4_gen70nu (same, adapters kept unmerged in
# fp32), seed 0, books 2 -> 3 -> 4. Compare with round 9's A4_gen70 (tpu/results/v9-20261024-031504/results_v9c.json).
set -u
cd /content/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
python -c "import jax; print('devices', jax.devices())"
RUNS=A4_gen70n,A4_gen70nu BOOKS=2,3,4 OUT=/content/results.json python run_cl.py && echo "FULL DONE"
