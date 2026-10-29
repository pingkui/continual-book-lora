#!/bin/bash
# Round 13 on a Colab TPU v5e (Qwen2.5-1.5B, A4_gen70n, data_v6, seed 0): book 2, then two branches carried through
# book 4 with spaced review of the earlier books' facts (Spaced): lr x1 x 12 epochs (sr12) and lr x1/3 x 12 (sr33e12).
# A 0.5B smoke run goes first.
set -u
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1
MODEL=Qwen/Qwen2.5-0.5B-Instruct RUNS=A4_gen70n BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 ROWS=16 \
  EPOCHS=1 BRANCH_EPOCHS=3 REPLAY_FACTS=3 EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 NFEAT=256 OUT=/content/sr_smoke.json \
  python run_cl.py > /content/sr_smoke.log 2>&1 || { echo "SMOKE FAILED"; tail -20 /content/sr_smoke.log; exit 1; }
grep -E "spaced review|after_book4" /content/sr_smoke.log | head -8
echo "SMOKE OK"
RUNS=A4_gen70n BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12,sr33e12 OUT=/content/results_sr.json \
  python run_cl.py && echo "SR DONE"
