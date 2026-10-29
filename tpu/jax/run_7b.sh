#!/bin/bash
# Qwen2.5-7B on a Colab A100 80 GB: round-10 method (A4_gen70n: token-normalised subspaces, tau 0.99, general-text
# protection 0.70) on data_v6, books 2 -> 3 -> 4. Usage: bash run_7b.sh RUN  (A4_gen70n = seed 0, A4_gen70n_s1 = seed 1)
set -u
RUN=$1
cd /content/code
export SRC=/content/data PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
python -c "import jax; print('devices', jax.devices())"
MODEL=Qwen/Qwen2.5-7B-Instruct RUNS=$RUN BOOKS=2,3,4 OUT=/content/results_$RUN.json python run_cl.py && echo "FULL DONE"
