import subprocess, json
cfg = json.load(open("/content/run_cfg.json"))
NAME, RUN, ROWS, EXTRA = cfg["name"], cfg["run"], cfg["rows"], cfg.get("extra_env", "")
script = r'''cd /content && chmod 600 rclone.conf && rm -rf sr && mkdir -p sr && tar xzf cl_v60.tgz -C sr
curl -sSLO https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip && python3 -c "import zipfile; zipfile.ZipFile('rclone-v1.75.2-linux-amd64.zip').extract('rclone-v1.75.2-linux-amd64/rclone', '/content')" && chmod +x /content/rclone-v1.75.2-linux-amd64/rclone
RC="/content/rclone-v1.75.2-linux-amd64/rclone --drive-chunk-size 256M"; export RCLONE_CONFIG=/content/rclone.conf
D=gdrive:kaggle/@@NAME@@-20261030
( while true; do $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1; $RC copy /content/epoch_exam $D/epoch_exam >/dev/null 2>&1; sleep 60; done ) &
$RC copy @@ROWS@@ /content/ 2>&1 | grep -v NOTICE; $RC copy gdrive:kaggle/gen10-20261028/book1.generic.facts.json /content/ 2>&1 | grep -v NOTICE
gunzip -c /content/$(basename @@ROWS@@) > /content/sr/data/book1.train.jsonl && cp /content/book1.generic.facts.json /content/sr/data/book1.facts.json
echo "PREFLIGHT rows $(wc -l < /content/sr/data/book1.train.jsonl) (cap2 expects 36917); true answers restating B: $(python3 -c "
import json; r=[json.loads(l) for l in open('/content/sr/data/book1.train.jsonl')]; t=[x for x in r if x.get('tag')=='premise_g' and 'неверно' not in x['assistant']]; print(sum(len(x['assistant'].split('. '))>1 for x in t), 'of', len(t))"); run @@RUN@@; extra env: @@EXTRA@@"
cd /content/sr/code
export SRC=/content/sr/data PYTHONUNBUFFERED=1 RANK=32 MODEL=Qwen/Qwen2.5-7B-Instruct CHAT_FILE=/content/chat_self_7b.json RUNS=@@RUN@@ EPOCH_EXAM=1 EPOCH_EXAM_DIR=/content/epoch_exam
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" Q_DTYPE=fp16 REMAT=1 MICRO=8 GEN_BS=48 JAX_COMPILATION_CACHE_DIR=/content/jaxcache DIAG_STEPS=1
export @@EXTRA@@
echo "=== TRAIN @@NAME@@: book 1, run @@RUN@@, RANK=$RANK, EPOCHS=${EPOCHS1:-6}, rows @@ROWS@@ ==="
mkdir -p /content/lora /content/ckpt
BOOKS=1 EPOCHS=${EPOCHS1:-6} SUB_LAST=1 SAVE_LORA=/content/lora CKPT=/content/ckpt OUT=/content/results_@@NAME@@.json python run_cl.py; echo "train exit $?"
$RC copy /content/results_@@NAME@@.json $D/ 2>&1 | grep -v NOTICE; $RC copy /content/lora $D/lora 2>&1 | grep -v NOTICE; $RC copy /content/epoch_exam $D/epoch_exam 2>&1 | grep -v NOTICE
if [ -f /content/lora/@@RUN@@/book1.npz ]; then
  cd /content && tar xzf l5f.tgz && mv l5f l5 && mkdir -p l5/lora l5/out/base && cp lora/@@RUN@@/book1.npz l5/lora/
  $RC copy gdrive:kaggle/v26-20261026/l5/out/base /content/l5/out/base 2>&1 | grep -v NOTICE
  cd /content/l5/code
  export BASE=Qwen/Qwen2.5-7B-Instruct RUN=@@RUN@@ BRANCH=- RES=/content/results_@@NAME@@.json MMLU_BS=8 LORAS=/content/l5/lora/book1.npz
  echo "L5 START"; bash run_l5.sh prep; echo "prep exit $?"; $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1
  ( cd /content && python3 /content/hf_trap_answers.py /content/l5/trained /content/hf_answers.json 1 > /content/hf_answers.log 2>&1; $RC copy /content/hf_answers.json $D/; $RC copy /content/hf_answers.log $D/ ) & HFPID=$!
  bash run_l5.sh full; echo "L5 exit $?"
  $RC copy /content/l5/out $D/l5/out 2>&1 | grep -v NOTICE
  wait $HFPID
  echo "L5 DONE"
fi
echo "@@NAME@@ CELL DONE"
$RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1'''.replace("@@NAME@@", NAME).replace("@@RUN@@", RUN).replace("@@ROWS@@", ROWS).replace("@@EXTRA@@", EXTRA)
open(f"/content/{NAME}.sh", "w").write(script)
subprocess.Popen(f"nohup bash /content/{NAME}.sh > /content/{NAME}.log 2>&1 &", shell=True, executable="/bin/bash")
print(NAME, RUN, EXTRA, "launched in the background")
