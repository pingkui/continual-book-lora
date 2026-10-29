#!/bin/bash
# Round 16d on a Colab TPU v5e (seed 0, data_v14, sr12 to book 4, self-instruct protection for the first book only):
#   A4_chat70f      tau_chat 0.70, the 320-row chat_self.json of round 16
#   A4_chat80f_n640 tau_chat 0.80, 640 self-instruct rows (eight kinds of task), 128 held out; written to chat_self640.json
# LoRA factors and results go straight to Drive when /content/rclone.conf is present.
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
run() {  # name extra-env...
  local R=$1; shift
  if env "$@" RUNS=$R BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 SAVE_LORA=/content/lora OUT=/content/results_v16_$R.json \
      python run_cl.py; then echo "V16 $R DONE"; else echo "V16 $R FAILED"; fi
  if [ $DRIVE = 1 ]; then
    $RC copy /content/lora gdrive:kaggle/v16d-20261025/lora 2>&1 | grep -v NOTICE
    $RC copy /content/results_v16_$R.json gdrive:kaggle/v16d-20261025/ 2>&1 | grep -v NOTICE
    [ -f /content/chat_self640.json ] && $RC copy /content/chat_self640.json gdrive:kaggle/v16d-20261025/ 2>&1 | grep -v NOTICE
    echo "uploaded $R to Drive"
  fi
}
run A4_chat70f CHAT_FILE=/content/chat_self.json
run A4_chat80f_n640 CHAT_FILE=/content/chat_self640.json CHAT_N=640 CHAT_EVAL_N=128
echo "V16D DONE"
