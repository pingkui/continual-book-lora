#!/bin/bash
# Round 20 on a Colab TPU v5e: rounds 18+19 again (books 1 -> 2 -> 3 -> 4, Qwen2.5-1.5B, data_v16, A4_chat80f, sr12
# with REPLAY_CAP 1.0 / REPLAY_TEST 300) at LoRA rank 32 instead of 16 (alpha 64, so the scale alpha/rank stays 2).
# Capacity test: at rank 16 the first book (2176 facts) learned "a founder's name in a year question means answer the
# year" (traps 19/20 made up, L4 131) and cost 9 L5 points; the projection then froze both. Everything else is as in
# round 18.
# Colab reclaimed the first two VMs of this round 30-47 min into book 1, so the script is now restartable: it pulls the
# finished books' factors and the checkpoint of the book in progress from Drive, loads the finished books (LOAD_LORA),
# resumes the unfinished one from its last epoch (CKPT), and pushes every checkpoint and every finished book back to
# Drive (CKPT_CMD). Smoke first (0.5B, rank 32): a 2-epoch book trained straight through vs. the same book stopped
# after its first epoch's checkpoint and resumed in a new process; the saved factors must agree.
set -u
cd /content
RC="/content/rclone-v1.75.2-linux-amd64/rclone --drive-chunk-size 256M"
RCBIN=/content/rclone-v1.75.2-linux-amd64/rclone
export RCLONE_CONFIG=/content/rclone.conf
DEST=${DEST:-gdrive:kaggle/v20-20261025}   # Drive folder of this run (LoRAs, checkpoints, live log, results)
TAG=${TAG:-v20}                             # results file name
BOOKS_RUN=${BOOKS_RUN:-1,2,3,4}             # e.g. BOOKS_RUN=1 trains the first book alone (round 21 pilot)
RUNNAME=${RUNNAME:-A4_chat80f}              # run configuration (RUNS in run_cl.py); round 22: A4_chat90f
# CHAT_FILE_RUN (default /content/chat_self.json), CHAT_MATH, CHAT_EXTRA, CHAT_N pass through to run_cl.py (round 23)
curl -sSLO https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip
curl -sSL https://downloads.rclone.org/v1.75.2/SHA256SUMS | grep 'linux-amd64.zip$' | sha256sum -c - || { echo "RCLONE CHECKSUM FAILED"; exit 1; }
python3 -c "import zipfile; zipfile.ZipFile('rclone-v1.75.2-linux-amd64.zip').extract('rclone-v1.75.2-linux-amd64/rclone', '/content')"
chmod +x $RCBIN
$RC about gdrive: 2>&1 | grep -E 'Total' || { echo "DRIVE FAILED"; exit 1; }
free -g | head -2
cat > /content/watchdog.sh <<'EOF'
while true; do
  for p in $(ps -eo pid,etimes,comm,args | awk '$3=="rclone" && ($4=="copy" || $4=="sync") && $5 ~ /^\/content\/(lora|ckpt)/ && $2>300 {print $1}'); do
    echo "$(date -u +%T) watchdog: interrupting stuck rclone $p" >> /content/watchdog.log; kill -INT $p; done
  sleep 30; done
EOF
cat > /content/livelog.sh <<EOF
while true; do
  $RC copy /content/v20.log $DEST/live/ >/dev/null 2>&1
  [ -f /content/watchdog.log ] && $RC copy /content/watchdog.log $DEST/live/ >/dev/null 2>&1
  sleep 60; done
EOF
nohup bash /content/watchdog.sh >/dev/null 2>&1 &
nohup bash /content/livelog.sh >/dev/null 2>&1 &
mkdir -p /content/lora /content/ckpt
$RC copy $DEST/lora /content/lora 2>&1 | grep -v NOTICE
$RC copy $DEST/ckpt /content/ckpt 2>&1 | grep -v NOTICE
echo "state left by earlier VMs:"; find /content/lora /content/ckpt -type f -exec ls -la {} \;
cd /content/sr/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/sr/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1 DIAG_STEPS=1 RANK=32
# GPU=1 (round 20, books 3-4 on an A100 after six v5e VMs were reclaimed): JAX takes 90% of the card
[ -n "${GPU:-}" ] && export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.90}   # round 28: MEM_FRACTION=0.42 on the 96 GB G4 = the ~40 GiB JAX gets on a 48 GB card
if [ -z "${SKIP_SMOKE:-}" ]; then
  # 0.5B, books 1-2 with the sr12 review branch, on whatever device this is. A trains both books straight through
  # (2 epochs each; checkpoints written after epoch 1 and removed). B1 loads A's book 1 and stops right after book 2's
  # first-epoch checkpoint (LoRA, optimizer, shuffle rng, review boxes); B2 resumes it. B2's book-2 factors, exam
  # answers and last review line must equal A's.
  S=/content/v20_smoke
  SM="MODEL=Qwen/Qwen2.5-0.5B-Instruct CHAT_FILE=$S.chat.json CHAT_N=16 CHAT_EVAL_N=4 RUNS=A4_chat80f BOOKS=1,2 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 ROWS=16 EPOCHS=2 BRANCH_EPOCHS=2 REPLAY_FACTS=10 REPLAY_TEST=4 REPLAY_CAP=0.5 EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 NFEAT=256"
  env $SM CKPT=$S/ckpt_a SAVE_LORA=$S/lora_a OUT=$S.a.json python run_cl.py > $S.a.log 2>&1 || { echo "SMOKE A FAILED"; tail -20 $S.a.log; exit 1; }
  L1=1=$S/lora_a/A4_chat80f/book1.npz
  env $SM LOAD_LORA=$L1 CKPT=$S/ckpt CKPT_STOP_AFTER=1 SAVE_LORA=$S/lora_b OUT=$S.b1.json python run_cl.py > $S.b1.log 2>&1
  rc=$?; [ $rc = 3 ] || { echo "SMOKE B1 FAILED (exit $rc, expected 3)"; tail -20 $S.b1.log; exit 1; }
  { [ -f $S/ckpt/A4_chat80f_sr12/book2.npz ] && [ -f $S/ckpt/A4_chat80f_sr12/book2.npz.pkl ]; } || { echo "SMOKE B1 FAILED: no checkpoint"; ls -R $S/ckpt; exit 1; }
  env $SM LOAD_LORA=$L1 CKPT=$S/ckpt SAVE_LORA=$S/lora_b OUT=$S.b2.json python run_cl.py > $S.b2.log 2>&1 || { echo "SMOKE B2 FAILED"; tail -20 $S.b2.log; exit 1; }
  python - <<'PY2'
import json, os, re, numpy as np
S = '/content/v20_smoke'
a, b = np.load(f'{S}/lora_a/A4_chat80f_sr12/book2.npz'), np.load(f'{S}/lora_b/A4_chat80f_sr12/book2.npz')
diff = max(float(np.abs(a[k] - b[k]).max()) for k in a.files if k != 'scale')
ra = json.load(open(f'{S}.a.json'))['runs']['A4_chat80f']['branches']['sr12']['stages'][0]['books']['2']['answers']
rb = json.load(open(f'{S}.b2.json'))['runs']['A4_chat80f']['branches']['sr12']['stages'][0]['books']['2']['answers']
def rev(p): return [re.sub(r'^\[\s*\d+s\]', '', l).strip() for l in open(p) if 'spaced review, epoch' in l]
same_rev = rev(f'{S}.a.log')[-1] == rev(f'{S}.b2.log')[-1]
gone = not any(os.path.exists(p) for p in (f'{S}/ckpt/A4_chat80f_sr12/book2.npz', f'{S}/ckpt/A4_chat80f_sr12/book2.npz.pkl', f'{S}/ckpt_a/A4_chat80f/book1.npz', f'{S}/ckpt_a/A4_chat80f_sr12/book2.npz'))
print(f'RESUME CHECK (book with review): straight vs resumed factors max |diff| {diff:.3g}, same answers {ra == rb}, same last review line {same_rev}, checkpoints removed {gone}')
assert diff < 1e-5 and ra == rb and same_rev and gone
PY2
  grep -E "devices|checkpoint after|CKPT_STOP|resumed from|spaced review, epoch" $S.a.log $S.b1.log $S.b2.log | cut -c1-220
  echo "SMOKE OK"
fi
upload() {
  $RC copy /content/lora $DEST/lora 2>&1 | grep -v NOTICE
  for f in /content/results_$TAG.json /content/results_${TAG}b.json /content/v20_smoke.b2.log; do [ -f $f ] && $RC copy $f $DEST/ 2>&1 | grep -v NOTICE; done
  echo "uploaded to Drive"
}
run_full() {  # $1: suffix of the results file; finished books are loaded, the one in progress resumes from its checkpoint
  local load=""
  [ -f /content/lora/$RUNNAME/book1.npz ] && load="1=/content/lora/$RUNNAME/book1.npz"
  for b in 2 3; do f=/content/lora/${RUNNAME}_sr12/book$b.npz; [ -f $f ] && load="${load:+$load,}$b=$f"; done
  echo "run_full: LOAD_LORA='$load', checkpoints: $(ls /content/ckpt/$RUNNAME/ /content/ckpt/${RUNNAME}_sr12/ 2>/dev/null)"
  env CHAT_FILE=${CHAT_FILE_RUN:-/content/chat_self.json} RUNS=$RUNNAME BOOKS=$BOOKS_RUN BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 \
      REPLAY_TEST=300 REPLAY_CAP=1.0 SAVE_LORA=/content/lora CKPT=/content/ckpt \
      CKPT_CMD="$RC sync /content/ckpt $DEST/ckpt 2>&1 | grep -v NOTICE; $RC copy /content/lora $DEST/lora 2>&1 | grep -v NOTICE" \
      ${load:+LOAD_LORA=$load} OUT=/content/results_$TAG$1.json python run_cl.py
}
if run_full ""; then
  echo "V20 RUN DONE"
else
  echo "V20 RUN FAILED (exit $?); host memory now:"; free -g | head -2; dmesg 2>/dev/null | grep -i -E "killed process|out of memory" | tail -2
  upload
  echo "second attempt in this VM (finished books loaded, the unfinished one resumed)"
  if run_full b; then echo "V20 RESUME DONE"; else echo "V20 RESUME FAILED"; fi
fi
upload
echo "V20 DONE"
