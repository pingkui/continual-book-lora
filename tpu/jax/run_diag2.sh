#!/bin/bash
# Forgetting diagnostic with branches on a Colab TPU VM: A4_gen70, book 2, then book 3 three times from the same
# state: base / no_down12 (no LoRA on down_proj in layers 1-2) / normsub (token-normalised subspaces).
set -u
cd /content/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70 BOOKS=2,3 DIAG2=1 VARIANTS=base,no_down12,normsub OUT=/content/results.json python run_cl.py && echo "FULL DONE"
