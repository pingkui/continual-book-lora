#!/bin/bash
# Round 14 seed 0 again on a Colab TPU v5e (sr12 on data_v14, as run_v14.sh), now saving every book's LoRA factors
# (SAVE_LORA) so export_lora.py can rebuild the model on a GPU for lm-eval-harness (L5). TPU runs are deterministic,
# so the results must equal tpu/results/v14-20261025-050149/results_v14s0.json.
set -u
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
RUNS=A4_gen70n BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 SAVE_LORA=/content/lora OUT=/content/results_v14s0save.json \
  python run_cl.py && echo "V14SAVE DONE"
