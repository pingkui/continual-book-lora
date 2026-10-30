import subprocess
NAME = "heldout1"
script = r'''cd /content && chmod 600 rclone.conf && rm -rf sr && mkdir -p sr && tar xzf cl_v61.tgz -C sr && cp /content/sr/data/book2.facts.json /content/book2.relfacts.json
curl -sSLO https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip && python3 -c "import zipfile; zipfile.ZipFile('rclone-v1.75.2-linux-amd64.zip').extract('rclone-v1.75.2-linux-amd64/rclone', '/content')" && chmod +x /content/rclone-v1.75.2-linux-amd64/rclone
RC="/content/rclone-v1.75.2-linux-amd64/rclone --drive-chunk-size 256M"; export RCLONE_CONFIG=/content/rclone.conf
D=gdrive:kaggle/@@NAME@@-20261030
( while true; do $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1; $RC copy /content/epoch_exam $D/epoch_exam >/dev/null 2>&1; sleep 60; done ) &
echo "PREFLIGHT book 2 units: $(python3 -c "import json; print(len(json.load(open('/content/sr/data/book2.book.json'))['units']))"); rel facts: $(python3 -c "import json; print(len(json.load(open('/content/book2.relfacts.json'))))"); exams $(ls /content/sr/data | grep -c exam)"
cd /content/sr/code
export SRC=/content/sr/data PYTHONUNBUFFERED=1 RANK=32 MODEL=Qwen/Qwen2.5-7B-Instruct CHAT_FILE=/content/chat_self_7b.json RUNS=A4_chat90f
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" Q_DTYPE=fp16 REMAT=1 MICRO=8 GEN_BS=48 JAX_COMPILATION_CACHE_DIR=/content/jaxcache DIAG_STEPS=1
export EPOCH_EXAM=1 EPOCH_EXAM_DIR=/content/epoch_exam MAX_NEW=96
export GENERIC_BOOKS=2 GENERIC_PAT="book{k}.book.json" GENERIC_FORCE=1 GENERIC_PREM=10 GENERIC_PARA=3 GENERIC_QA2=2 GENERIC_FALSE_W=2 GENERIC_FALSE_ANS=answer_first GENERIC_EXCLUDE='founded|founder,goods|river' GENERIC_EXCLUDE_FACTS=/content/book2.relfacts.json
export BOOKS=2 EPOCHS=8
echo "=== TRAIN @@NAME@@: base model + book 2 only (no book 1, so the held-out pair types are never trained), book 2 generic, all row kinds, answer_first, 8 epochs, premise pairs (founded|founder) and (goods|river) HELD OUT ==="
mkdir -p /content/lora /content/ckpt
SAVE_LORA=/content/lora CKPT=/content/ckpt OUT=/content/results_@@NAME@@.json python run_cl.py; echo "train exit $?"
$RC copy /content/results_@@NAME@@.json $D/ 2>&1 | grep -v NOTICE; $RC copy /content/lora $D/lora 2>&1 | grep -v NOTICE; $RC copy /content/epoch_exam $D/epoch_exam 2>&1 | grep -v NOTICE
for k in 2; do gzip -c /content/sr/data/book$k.train.jsonl > /content/book$k.train.jsonl.gz; $RC copy /content/book$k.train.jsonl.gz $D/ 2>&1 | grep -v NOTICE; cp /content/sr/data/book$k.facts.json /content/; $RC copy /content/book$k.facts.json $D/ 2>&1 | grep -v NOTICE; done
if [ -f /content/lora/A4_chat90f/book2.npz ]; then
  cd /content && tar xzf l5g.tgz && mv l5f l5 && cd /content/l5/code
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python export_lora.py Qwen/Qwen2.5-7B-Instruct /content/l5/trained /content/lora/A4_chat90f/book2.npz && echo "EXPORT OK"
  cd /content && python3 /content/hf_trap_answers.py /content/l5/trained /content/hf_answers.json 2 > /content/hf_answers.log 2>&1; echo "HF exit $?"
  $RC copy /content/hf_answers.json $D/ 2>&1 | grep -v NOTICE; $RC copy /content/hf_answers.log $D/ 2>&1 | grep -v NOTICE
fi
echo "@@NAME@@ CELL DONE"
$RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1'''.replace("@@NAME@@", NAME)
open(f"/content/{NAME}.sh", "w").write(script)
subprocess.Popen(f"nohup bash /content/{NAME}.sh > /content/{NAME}.log 2>&1 &", shell=True, executable="/bin/bash")
print(NAME, "launched in the background")
