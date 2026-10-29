#!/bin/bash
# Branches carried through book 4 on a Colab TPU v5e (Qwen2.5-1.5B, A4_gen70n, data_v6, seed 0): book 2, then for each
# branch book 3 and book 4 with the same lr/epochs: lr x1/3 x 12 epochs, lr x1/4 x 18, and lr x1/3 up to 18 epochs with
# an early stop at 148/150 on the new book's training-wording probes. A 0.5B smoke run goes first.
set -u
cd /content/lr2/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/lr2/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
MODEL=Qwen/Qwen2.5-0.5B-Instruct RUNS=A4_gen70n BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=lr33e12,es148 ROWS=16 \
  EPOCHS=1 EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 NFEAT=256 OUT=/content/lr2_smoke.json python run_cl.py > /content/lr2_smoke.log 2>&1 \
  || { echo "SMOKE FAILED"; tail -20 /content/lr2_smoke.log; exit 1; }
grep -E "early stop|probe of book|after_book4" /content/lr2_smoke.log | head -8
echo "SMOKE OK"
RUNS=A4_gen70n BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=lr33e12,lr25e18,es148 OUT=/content/results_lr2.json \
  python run_cl.py && echo "LR2 DONE"
