#!/bin/bash
# Round 18 on a Colab TPU v5e: the full client order, books 1 -> 2 -> 3 -> 4 (book 1: 2176 facts, 44k training rows),
# Qwen2.5-1.5B, data_v16, A4_chat80f (self-instruct protection for the first book, tau 0.80), sr12 for books 2-4 with
# the scaled review: at most 1.0x the new book's rows of review per epoch (REPLAY_CAP), the reviewed facts plus 300
# sampled others self-tested per epoch (REPLAY_TEST). A 0.5B smoke run exercises the cap and the sampling first.
set -u
cd /content
RC=/content/rclone-v1.75.2-linux-amd64/rclone
export RCLONE_CONFIG=/content/rclone.conf
DRIVE=0
if [ -f $RCLONE_CONFIG ]; then
  if [ ! -x $RC ]; then
    curl -sSLO https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip
    curl -sSL https://downloads.rclone.org/v1.75.2/SHA256SUMS | grep 'linux-amd64.zip$' | sha256sum -c - || { echo "RCLONE CHECKSUM FAILED"; exit 1; }
    python3 -c "import zipfile; zipfile.ZipFile('rclone-v1.75.2-linux-amd64.zip').extract('rclone-v1.75.2-linux-amd64/rclone', '/content')"
    chmod +x $RC
  fi
  $RC about gdrive: 2>&1 | grep -E 'Total|Free' && DRIVE=1
fi
[ $DRIVE = 1 ] || echo "no Drive: files stay on this VM"
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1 DIAG_STEPS=1
S=/content/v18_smoke
MODEL=Qwen/Qwen2.5-0.5B-Instruct CHAT_FILE=$S.chat.json CHAT_N=16 CHAT_EVAL_N=4 RUNS=A4_chat80f BOOKS=1,2,3 BRANCH=1 \
  BRANCH_TO_END=1 BRANCHES=sr12 ROWS=16 EPOCHS=1 BRANCH_EPOCHS=3 REPLAY_FACTS=10 REPLAY_TEST=4 REPLAY_CAP=0.5 EXAMN=4 \
  GEN_N=4 GEN_TRAIN_N=8 NFEAT=256 OUT=$S.json python run_cl.py > $S.log 2>&1 || { echo "SMOKE FAILED"; tail -20 $S.log; exit 1; }
grep -E "spaced review" $S.log | cut -c1-200 | head -8
echo "SMOKE OK"
upload() {
  [ $DRIVE = 1 ] || return 0
  $RC copy /content/lora gdrive:kaggle/v18-20261025/lora 2>&1 | grep -v NOTICE
  $RC copy /content/results_v18.json gdrive:kaggle/v18-20261025/ 2>&1 | grep -v NOTICE
  echo "uploaded to Drive"
}
if CHAT_FILE=/content/chat_self.json RUNS=A4_chat80f BOOKS=1,2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 \
    REPLAY_TEST=300 REPLAY_CAP=1.0 SAVE_LORA=/content/lora OUT=/content/results_v18.json python run_cl.py; then
  echo "V18 RUN DONE"; else echo "V18 RUN FAILED"; fi
upload
echo "V18 DONE"
