import subprocess, json
NAME, RUN, ROWS, EXTRA = "gen28", "@@RUNS_LOOP@@", "gdrive:kaggle/forms-20261028/book1.gen20.train.jsonl.gz", "MAX_NEW=96"
script = r'''cd /content && chmod 600 rclone.conf && rm -rf sr && mkdir -p sr && tar xzf cl_v60.tgz -C sr
curl -sSLO https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip && python3 -c "import zipfile; zipfile.ZipFile('rclone-v1.75.2-linux-amd64.zip').extract('rclone-v1.75.2-linux-amd64/rclone', '/content')" && chmod +x /content/rclone-v1.75.2-linux-amd64/rclone
RC="/content/rclone-v1.75.2-linux-amd64/rclone --drive-chunk-size 256M"; export RCLONE_CONFIG=/content/rclone.conf
D=gdrive:kaggle/@@NAME@@-20261030
( while true; do $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1; $RC copy /content/epoch_exam $D/epoch_exam >/dev/null 2>&1; sleep 60; done ) &
$RC copy @@ROWS@@ /content/ 2>&1 | grep -v NOTICE; $RC copy gdrive:kaggle/gen10-20261028/book1.generic.facts.json /content/ 2>&1 | grep -v NOTICE
gunzip -c /content/$(basename @@ROWS@@) > /content/sr/data/book1.train.jsonl && cp /content/book1.generic.facts.json /content/sr/data/book1.facts.json
echo "PREFLIGHT rows $(wc -l < /content/sr/data/book1.train.jsonl) (gen10 44092); true answers restating B: $(python3 -c "
import json; r=[json.loads(l) for l in open('/content/sr/data/book1.train.jsonl')]; t=[x for x in r if x.get('tag')=='premise_g' and 'неверно' not in x['assistant']]; print(sum(len(x['assistant'].split('. '))>1 for x in t), 'of', len(t))"); runs A4_chat90f then A4_chat90f_s1; extra env: @@EXTRA@@"
cd /content/sr/code
export SRC=/content/sr/data PYTHONUNBUFFERED=1 RANK=32 MODEL=Qwen/Qwen2.5-7B-Instruct CHAT_FILE=/content/chat_self_7b.json EPOCH_EXAM=1 EPOCH_EXAM_DIR=/content/epoch_exam
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" Q_DTYPE=fp16 REMAT=1 MICRO=8 GEN_BS=48 JAX_COMPILATION_CACHE_DIR=/content/jaxcache DIAG_STEPS=1
export @@EXTRA@@
for RUN1 in A4_chat90f A4_chat90f_s1; do
cd /content/sr/code
echo "=== TRAIN @@NAME@@: book 1, run $RUN1, EPOCHS=4 (gen20/gen22 rows, answer_first) ==="
mkdir -p /content/lora /content/ckpt/$RUN1 /content/epoch_exam/$RUN1
RUNS=$RUN1 BOOKS=1 EPOCHS=4 SUB_LAST=1 SAVE_LORA=/content/lora CKPT=/content/ckpt/$RUN1 EPOCH_EXAM_DIR=/content/epoch_exam/$RUN1 OUT=/content/results_@@NAME@@_$RUN1.json python run_cl.py; echo "train exit $?"
$RC copy /content/results_@@NAME@@_$RUN1.json $D/ 2>&1 | grep -v NOTICE; $RC copy /content/lora $D/lora 2>&1 | grep -v NOTICE; $RC copy /content/epoch_exam $D/epoch_exam 2>&1 | grep -v NOTICE
if [ -f /content/lora/$RUN1/book1.npz ]; then
  cd /content && rm -rf l5 && tar xzf l5f.tgz && mv l5f l5 && mkdir -p l5/lora l5/out/base && cp lora/$RUN1/book1.npz l5/lora/
  $RC copy gdrive:kaggle/v26-20261026/l5/out/base /content/l5/out/base 2>&1 | grep -v NOTICE
  cd /content/l5/code
  export RUNS=$RUN1 BASE=Qwen/Qwen2.5-7B-Instruct RUN=$RUN1 BRANCH=- RES=/content/results_@@NAME@@_$RUN1.json MMLU_BS=8 LORAS=/content/l5/lora/book1.npz
  echo "L5 START"; bash run_l5.sh prep; echo "prep exit $?"; $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1
  ( cd /content && python3 /content/hf_trap_answers.py /content/l5/trained /content/hf_answers_$RUN1.json 1 > /content/hf_answers_$RUN1.log 2>&1; $RC copy /content/hf_answers_$RUN1.json $D/; $RC copy /content/hf_answers_$RUN1.log $D/ ) & HFPID=$!
  bash run_l5.sh full; echo "L5 exit $?"
  $RC copy /content/l5/out $D/l5_$RUN1/out 2>&1 | grep -v NOTICE
  wait $HFPID
  echo "L5 DONE"
fi
echo "RUN $RUN1 DONE"
done
echo "@@NAME@@ CELL DONE"
$RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1'''.replace("@@NAME@@", NAME).replace("@@RUN@@", RUN).replace("@@ROWS@@", ROWS).replace("@@EXTRA@@", EXTRA)
open(f"/content/{NAME}.sh", "w").write(script)
subprocess.Popen(f"nohup bash /content/{NAME}.sh > /content/{NAME}.log 2>&1 &", shell=True, executable="/bin/bash")
print(NAME, "launched in the background")
