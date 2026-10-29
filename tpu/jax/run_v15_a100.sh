#!/bin/bash
# Round 15 on one Colab A100 80 GB, start to end:
#  1. teacher templates built around the founding nouns (lab/gen_templates_v15.py), a DRY run first
#  2. data_v15 = lab/gen_books_v14.py with templates_v15; books 2-4 and factlab into $V/data
#  3. smoke: Qwen2.5-0.5B sr12 on tiny data with SAVE_DIR, then check_export.py on that checkpoint
#  4. sr12 seeds 0 and 1 at once (two processes, 40% of the GPU each), checkpoints in $V/export for lm-eval,
#     then check_export.py on the seed-0 checkpoint
set -u
V=/content/v15
export PYTHONUNBUFFERED=1
pip uninstall -y -q torchao 2>/dev/null
python -c "import datasets" 2>/dev/null || pip install -q datasets
cd $V/lab
mkdir -p tdry tout
DRY=1 OUT=tdry TPL=data_tpl/templates_v14.json python3 gen_templates_v15.py > $V/tdry.log 2>&1 \
  || { echo "TPL DRY FAILED"; tail -20 $V/tdry.log; exit 1; }
echo "TPL DRY OK"
OUT=tout TPL=data_tpl/templates_v14.json python3 gen_templates_v15.py || { echo "TPL FAILED"; exit 1; }
python3 gen_books_v14.py $V/data_v15 7 tout/templates_v15.json > $V/gen_books.log 2>&1 \
  || { echo "DATA FAILED"; tail -20 $V/gen_books.log; exit 1; }
tail -5 $V/gen_books.log
python3 -c "
import json
for k in (2, 3, 4):
    q = {x['q'] for n in ('exam', 'exam_gen') for x in json.load(open(f'$V/data_v15/book{k}.{n}.json'))}
    leak = sum(json.loads(l).get('user') in q for l in open(f'$V/data_v15/book{k}.train.jsonl'))
    print(f'book{k}: exam questions in training rows: {leak}')"
mkdir -p $V/data && cp $V/data_v15/book{2,3,4}.* factlab.py $V/data/
cd $V/code
export SRC=$V/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
MODEL=Qwen/Qwen2.5-0.5B-Instruct RUNS=A4_gen70n BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 ROWS=16 EPOCHS=1 \
  BRANCH_EPOCHS=3 REPLAY_FACTS=3 EXAMN=4 GEN_N=4 GEN_TRAIN_N=8 NFEAT=256 SAVE_DIR=$V/smoke_export OUT=$V/smoke.json \
  python run_cl.py > $V/smoke.log 2>&1 || { echo "SMOKE FAILED"; tail -20 $V/smoke.log; exit 1; }
python3 check_export.py $V/smoke_export/A4_gen70n_sr12 $V/smoke.json A4_gen70n sr12 $SRC 8 \
  || { echo "SMOKE EXPORT CHECK FAILED"; exit 1; }
echo "SMOKE OK"
pids=()
for R in A4_gen70n A4_gen70n_s1; do
  RUNS=$R BOOKS=2,3,4 BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12 SAVE_DIR=$V/export OUT=$V/results_v15_$R.json \
    python run_cl.py > $V/run_$R.log 2>&1 &
  pids+=($!)
  sleep 60  # stagger the model loads
done
for p in "${pids[@]}"; do wait $p; echo "run exit $?"; done
python3 check_export.py $V/export/A4_gen70n_sr12 $V/results_v15_A4_gen70n.json A4_gen70n sr12 $SRC 40
echo "V15 DONE"
