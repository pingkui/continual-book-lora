#!/bin/bash
# Round 16 on a Colab TPU v5e (Qwen2.5-1.5B, data_v14, sr12 through book 4): protection of the base model's own
# instruction-following and reasoning (self-instruct rows, tau_chat), against L5's losses.
#  0. 0.5B smoke: self-instruct generation, chat NLL, the tau_chat subspace, LoRA saving and EVAL_LORA
#  1. control: chat NLL of the round-14 seed-0 model from its saved LoRA factors (EVAL_LORA); writes chat_self.json,
#     which the next runs reuse
#  2. A4_chat80 and A4_chat90 (seed 0), sr12 to book 4, LoRA factors saved for L5
set -u
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1 CHAT_FILE=/content/chat_self.json
S=/content/v16_smoke
SM="MODEL=Qwen/Qwen2.5-0.5B-Instruct CHAT_FILE=$S.chat.json CHAT_N=16 CHAT_EVAL_N=4 EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 NFEAT=256"
env $SM RUNS=A4_chat80 BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 ROWS=16 EPOCHS=1 BRANCH_EPOCHS=3 \
  REPLAY_FACTS=3 SAVE_LORA=$S/lora OUT=$S.json python run_cl.py > $S.log 2>&1 || { echo "SMOKE FAILED"; tail -20 $S.log; exit 1; }
env $SM RUNS=A4_chat80 BOOKS=2 OUT=$S.eval.json \
  EVAL_LORA=$S/lora/A4_chat80/book2.npz,$S/lora/A4_chat80_sr12/book3.npz,$S/lora/A4_chat80_sr12/book4.npz \
  python run_cl.py > $S.eval.log 2>&1 || { echo "SMOKE EVAL FAILED"; tail -20 $S.eval.log; exit 1; }
grep -hE "self-instruct|chat NLL|merged" $S.log $S.eval.log | cut -c1-200 | head -12
echo "SMOKE OK"
L=/content/v14lora
RUNS=A4_gen70n BOOKS=2,3,4 EVAL_LORA=$L/book2.npz,$L/book3.npz,$L/book4.npz OUT=/content/results_v16ctrl.json \
  python run_cl.py && echo "V16 CTRL DONE"
for R in A4_chat80 A4_chat90; do
  RUNS=$R BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 SAVE_LORA=/content/lora OUT=/content/results_v16_$R.json \
    python run_cl.py && echo "V16 $R DONE"
done
