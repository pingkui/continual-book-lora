import subprocess
NAME = "gen23"
script = r'''cd /content && chmod 600 rclone.conf && rm -rf sr && mkdir -p sr && tar xzf cl_v60.tgz -C sr
curl -sSLO https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip && python3 -c "import zipfile; zipfile.ZipFile('rclone-v1.75.2-linux-amd64.zip').extract('rclone-v1.75.2-linux-amd64/rclone', '/content')" && chmod +x /content/rclone-v1.75.2-linux-amd64/rclone
RC="/content/rclone-v1.75.2-linux-amd64/rclone --drive-chunk-size 256M"; export RCLONE_CONFIG=/content/rclone.conf
D=gdrive:kaggle/@@NAME@@-20261029
( while true; do $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1; $RC copy /content/epoch_exam $D/epoch_exam >/dev/null 2>&1; sleep 60; done ) &
# book 1 = gen20 exactly: its rows, facts and trained factors
$RC copy gdrive:kaggle/forms-20261028/book1.gen20.train.jsonl.gz /content/ 2>&1 | grep -v NOTICE; $RC copy gdrive:kaggle/gen10-20261028/book1.generic.facts.json /content/ 2>&1 | grep -v NOTICE
$RC copy gdrive:kaggle/gen20-20261029/lora/A4_chat90f/book1.npz /content/lora_in/ 2>&1 | grep -v NOTICE
gunzip -c /content/book1.gen20.train.jsonl.gz > /content/sr/data/book1.train.jsonl && cp /content/book1.generic.facts.json /content/sr/data/book1.facts.json
echo "PREFLIGHT book1 rows $(wc -l < /content/sr/data/book1.train.jsonl) (gen20 44092); book1 factors $(ls -la /content/lora_in/book1.npz | awk '{print $5}') bytes; book 2 units: $(for k in 2; do python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d if isinstance(d, list) else d['units']), end=' ')" /content/sr/data/book$k.book.json; done); exams $(ls /content/sr/data | grep -c exam)"
cd /content/sr/code
export SRC=/content/sr/data PYTHONUNBUFFERED=1 RANK=32 MODEL=Qwen/Qwen2.5-7B-Instruct CHAT_FILE=/content/chat_self_7b.json RUNS=A4_chat90f
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" Q_DTYPE=fp16 REMAT=1 MICRO=8 GEN_BS=48 JAX_COMPILATION_CACHE_DIR=/content/jaxcache DIAG_STEPS=1
export EPOCH_EXAM=1 EPOCH_EXAM_DIR=/content/epoch_exam MAX_NEW=96
export GENERIC_BOOKS=2 GENERIC_PAT="book{k}.book.json" GENERIC_FORCE=1 GENERIC_PREM=10 GENERIC_PARA=3 GENERIC_QA2=2 GENERIC_FALSE_W=2 GENERIC_FALSE_ANS=answer_first
export BOOKS=1,2 EPOCHS=6 LOAD_LORA=1=/content/lora_in/book1.npz BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 REPLAY_TEST=300 REPLAY_CAP=1.0
echo "=== TRAIN @@NAME@@: book 1 loaded from gen20, book 2 generic with ALL row kinds (paraphrases + teacher QA) and answer_first, sr12 review ==="
mkdir -p /content/lora /content/ckpt
SAVE_LORA=/content/lora CKPT=/content/ckpt OUT=/content/results_@@NAME@@.json python run_cl.py; echo "train exit $?"
$RC copy /content/results_@@NAME@@.json $D/ 2>&1 | grep -v NOTICE; $RC copy /content/lora $D/lora 2>&1 | grep -v NOTICE; $RC copy /content/epoch_exam $D/epoch_exam 2>&1 | grep -v NOTICE
for k in 2; do gzip -c /content/sr/data/book$k.train.jsonl > /content/book$k.train.jsonl.gz; $RC copy /content/book$k.train.jsonl.gz $D/ 2>&1 | grep -v NOTICE; cp /content/sr/data/book$k.facts.json /content/; $RC copy /content/book$k.facts.json $D/ 2>&1 | grep -v NOTICE; done
if false; then
  cd /content && tar xzf l5f.tgz && mv l5f l5 && mkdir -p l5/lora l5/out/base
  cp /content/lora_in/book1.npz l5/lora/book1.npz; for k in 2 3 4; do cp /content/lora/A4_chat90f_sr12/book$k.npz l5/lora/book$k.npz; done
  $RC copy gdrive:kaggle/v26-20261026/l5/out/base /content/l5/out/base 2>&1 | grep -v NOTICE
  cd /content/l5/code
  export BASE=Qwen/Qwen2.5-7B-Instruct RUN=A4_chat90f BRANCH=sr12 RES=/content/results_@@NAME@@.json MMLU_BS=8 LORAS="/content/l5/lora/book1.npz /content/l5/lora/book2.npz /content/l5/lora/book3.npz /content/l5/lora/book4.npz"
  echo "L5 START"; bash run_l5.sh prep; echo "prep exit $?"; $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1
  ( cd /content && python3 /content/hf_trap_answers.py /content/l5/trained /content/hf_answers.json 1,2,3,4 > /content/hf_answers.log 2>&1; $RC copy /content/hf_answers.json $D/; $RC copy /content/hf_answers.log $D/ ) & HFPID=$!
  bash run_l5.sh full; echo "L5 exit $?"
  $RC copy /content/l5/out $D/l5/out 2>&1 | grep -v NOTICE
  wait $HFPID
  echo "L5 DONE"
fi
echo "@@NAME@@ CELL DONE"
$RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1'''.replace("@@NAME@@", NAME)
open(f"/content/{NAME}.sh", "w").write(script)
subprocess.Popen(f"nohup bash /content/{NAME}.sh > /content/{NAME}.log 2>&1 &", shell=True, executable="/bin/bash")
print(NAME, "launched in the background")
