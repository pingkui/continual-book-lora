#!/bin/bash
# L3 branch experiments on the A100, three processes at once (Qwen2.5-1.5B, A4_gen70n, data_v6, seed 0, books 2 -> 3):
#   A: onto99, onto95  (book 3 trained only inside its own directions outside the protected subspace)
#   B: t99 (same-GPU control), down995, down999  (down_proj protected more than the other kinds)
#   C: lr50, lr25  (book 3 learned with 1/2 and 1/4 of the learning rate)
set -u
cd /content/l3/code
export SRC=/content/l3/data PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
RUNS=A4_gen70n BOOKS=2,3 BRANCH=1 BRANCHES=onto99,onto95 OUT=/content/l3_A.json python run_cl.py > /content/l3_A.log 2>&1 &
PA=$!
sleep 60  # stagger the model loads
RUNS=A4_gen70n BOOKS=2,3 BRANCH=1 BRANCHES=t99,down995,down999 OUT=/content/l3_B.json python run_cl.py > /content/l3_B.log 2>&1 &
PB=$!
sleep 60
RUNS=A4_gen70n BOOKS=2,3 BRANCH=1 BRANCHES=lr50,lr25 OUT=/content/l3_C.json python run_cl.py > /content/l3_C.log 2>&1 &
PC=$!
wait $PA; echo "A exit $?"
wait $PB; echo "B exit $?"
wait $PC; echo "C exit $?"
echo "L3 DONE"
