#!/bin/bash
# Round 15 on a Colab TPU v5e: sr12 (lr x1 x 12 epochs with spaced review, through book 4) on data_v15 (data_v14 plus
# founding-noun questions, lab/gen_templates_v15.py), seed 0 then seed 1, every book's LoRA factors saved for
# export_lora.py. A 0.5B smoke run first checks the LoRA saving, the rebuild and the rebuilt model's answers.
set -u
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
S=/content/v15_smoke
MODEL=Qwen/Qwen2.5-0.5B-Instruct RUNS=A4_gen70n BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 ROWS=16 EPOCHS=1 \
  BRANCH_EPOCHS=3 REPLAY_FACTS=3 EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 NFEAT=256 SAVE_LORA=$S/lora OUT=$S.json \
  python run_cl.py > $S.log 2>&1 || { echo "SMOKE FAILED"; tail -20 $S.log; exit 1; }
python export_lora.py Qwen/Qwen2.5-0.5B-Instruct $S/export $S/lora/A4_gen70n/book2.npz \
  $S/lora/A4_gen70n_sr12/book3.npz $S/lora/A4_gen70n_sr12/book4.npz || { echo "SMOKE EXPORT FAILED"; exit 1; }
python check_export.py $S/export $S.json A4_gen70n sr12 $SRC 8 || { echo "SMOKE CHECK FAILED"; exit 1; }
echo "SMOKE OK"
RUNS=A4_gen70n BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 SAVE_LORA=/content/lora OUT=/content/results_v15s0.json \
  python run_cl.py && echo "V15 S0 DONE"
RUNS=A4_gen70n_s1 BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 SAVE_LORA=/content/lora OUT=/content/results_v15s1.json \
  python run_cl.py && echo "V15 S1 DONE"
