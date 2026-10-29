#!/bin/bash
# Round 19 on a Colab TPU v5e: resume round 18 (books 1 -> 2 -> 3 -> 4, data_v16, A4_chat80f, sr12) from the saved
# LoRA factors of books 1-3 (Drive kaggle/v18-20261025/lora) and train book 4, with the host-memory relief in
# run_cl.py (master freed after the branch copy, gc after stacking). Smoke first: a 0.5B run that saves factors for
# books 1-2, then a second one that loads them and trains book 3.
set -u
cd /content
RC=/content/rclone-v1.75.2-linux-amd64/rclone
export RCLONE_CONFIG=/content/rclone.conf
curl -sSLO https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip
curl -sSL https://downloads.rclone.org/v1.75.2/SHA256SUMS | grep 'linux-amd64.zip$' | sha256sum -c - || { echo "RCLONE CHECKSUM FAILED"; exit 1; }
python3 -c "import zipfile; zipfile.ZipFile('rclone-v1.75.2-linux-amd64.zip').extract('rclone-v1.75.2-linux-amd64/rclone', '/content')"
chmod +x $RC
$RC about gdrive: 2>&1 | grep -E 'Total' || { echo "DRIVE FAILED"; exit 1; }
mkdir -p /content/prev && $RC copy gdrive:kaggle/v18-20261025/lora/A4_chat80f/book1.npz /content/prev/ && \
  $RC copy gdrive:kaggle/v18-20261025/lora/A4_chat80f_sr12/ /content/prev/ && ls -la /content/prev/
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1 DIAG_STEPS=1
S=/content/v19_smoke
SM="MODEL=Qwen/Qwen2.5-0.5B-Instruct CHAT_FILE=$S.chat.json CHAT_N=16 CHAT_EVAL_N=4 RUNS=A4_chat80f BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 ROWS=16 EPOCHS=1 BRANCH_EPOCHS=2 REPLAY_FACTS=10 REPLAY_TEST=4 REPLAY_CAP=0.5 EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 NFEAT=256"
env $SM BOOKS=1,2 SAVE_LORA=$S/lora OUT=$S.a.json python run_cl.py > $S.a.log 2>&1 || { echo "SMOKE A FAILED"; tail -20 $S.a.log; exit 1; }
env $SM BOOKS=1,2,3 LOAD_LORA=1=$S/lora/A4_chat80f/book1.npz,2=$S/lora/A4_chat80f_sr12/book2.npz OUT=$S.b.json \
  python run_cl.py > $S.b.log 2>&1 || { echo "SMOKE B FAILED"; tail -20 $S.b.log; exit 1; }
python - <<'PY'
import json
a=json.load(open('/content/v19_smoke.a.json'))['runs']['A4_chat80f']; b=json.load(open('/content/v19_smoke.b.json'))['runs']['A4_chat80f']
sa=a['branches']['sr12']['stages'][0]['books']['2']; sb=b['branches']['sr12']['stages'][0]['books']['2']
same = sa['answers']==sb['answers'] and a['stages'][0]['books']['1']['answers']==b['stages'][0]['books']['1']['answers']
print('RESUME CHECK: loaded books reproduce the saved run:', same); assert same
PY
grep -E "loaded saved|stacked bases" $S.b.log | cut -c1-160 | head -4
echo "SMOKE OK"
upload() {
  $RC copy /content/lora gdrive:kaggle/v19-20261025/lora 2>&1 | grep -v NOTICE
  $RC copy /content/results_v19.json gdrive:kaggle/v19-20261025/ 2>&1 | grep -v NOTICE
  echo "uploaded to Drive"
}
if CHAT_FILE=/content/chat_self.json RUNS=A4_chat80f BOOKS=1,2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 \
    LOAD_LORA=1=/content/prev/book1.npz,2=/content/prev/book2.npz,3=/content/prev/book3.npz \
    REPLAY_TEST=300 REPLAY_CAP=1.0 SAVE_LORA=/content/lora OUT=/content/results_v19.json python run_cl.py; then
  echo "V19 RUN DONE"; else echo "V19 RUN FAILED"; fi
upload
echo "V19 DONE"
