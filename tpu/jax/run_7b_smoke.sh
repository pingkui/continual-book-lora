#!/bin/bash
# 7B smoke test on a Colab A100 80 GB: Qwen2.5-7B-Instruct, A4_gen70n (round 10), book 2 with 320 training rows for
# one epoch (10 steps: step time and memory, not learning), the full book-2 exams and general-text NLL.
set -u
cd /content/code
export SRC=/content/data PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
python -c "import jax; print('devices', jax.devices())"
MODEL=Qwen/Qwen2.5-7B-Instruct RUNS=A4_gen70n BOOKS=2 ROWS=320 EPOCHS=1 OUT=/content/smoke7b.json python run_cl.py && echo "SMOKE7B DONE"
