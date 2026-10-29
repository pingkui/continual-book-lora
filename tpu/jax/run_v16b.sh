#!/bin/bash
# Round 16, rerun with the finiteness guards (run_cl.py: _new_dirs, stack_Q, train_book) on a Colab TPU v5e:
# A4_chat80 and A4_chat90 (seed 0, data_v14, sr12 to book 4), reusing the first attempt's chat_self.json. Every book's
# LoRA factors go straight from this VM to Google Drive (rclone, config uploaded to /content/rclone.conf), not through
# the host.
set -u
cd /content
RC=/content/rclone-v1.75.2-linux-amd64/rclone
if [ -f /content/rclone.conf ] && [ ! -x $RC ]; then
  curl -sSLO https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip
  curl -sSL https://downloads.rclone.org/v1.75.2/SHA256SUMS | grep 'linux-amd64.zip$' | sha256sum -c - || { echo "RCLONE CHECKSUM FAILED"; exit 1; }
  python3 -c "import zipfile; zipfile.ZipFile('rclone-v1.75.2-linux-amd64.zip').extract('rclone-v1.75.2-linux-amd64/rclone', '/content')"
  chmod +x $RC
fi
export RCLONE_CONFIG=/content/rclone.conf
DRIVE=1
if [ ! -f $RCLONE_CONFIG ] || ! $RC about gdrive: 2>&1 | grep -E 'Total|Free'; then
  echo "no Drive config: the LoRA files stay on this VM for the host to fetch"; DRIVE=0
fi
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1 CHAT_FILE=/content/chat_self.json
for R in A4_chat80 A4_chat90; do
  if RUNS=$R BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 SAVE_LORA=/content/lora OUT=/content/results_v16_$R.json \
      python run_cl.py; then echo "V16 $R DONE"; else echo "V16 $R FAILED"; fi
  if [ $DRIVE = 1 ]; then
    $RC copy /content/lora gdrive:kaggle/v16b-20261025/lora 2>&1 | grep -v NOTICE
    $RC copy /content/results_v16_$R.json gdrive:kaggle/v16b-20261025/ 2>&1 | grep -v NOTICE
    echo "uploaded $R to Drive"
  fi
done
echo "V16B DONE"
