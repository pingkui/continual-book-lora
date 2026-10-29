#!/bin/bash
# On a Colab TPU VM: smoke test of the whole pipeline (books 2->3, 64 rows, 1 epoch, 2 items per exam kind), check
# that the projection leakage is at fp32 level, then the full run with V8's settings
# (A4_gen seeds 1 and 2, books 2->3->4, base model scored first). Compiled programs are cached on disk, so the
# full run reuses what the smoke test compiled.
set -u
cd /content/code
python -c "import datasets" 2>/dev/null || pip install -q datasets
export SRC=/content/data JAX_COMPILATION_CACHE_DIR=/content/jaxcache PYTHONUNBUFFERED=1

echo "=== SMOKE"
RUNS=A4_gen_s1 BOOKS=2,3 ROWS=64 EPOCHS=1 EXAMN=2 OUT=/content/smoke.json python run_cl.py || { echo "SMOKE FAILED"; exit 1; }
leak=$(python -c "import json; print(json.load(open('/content/smoke.json'))['runs']['A4_gen_s1']['leak']['3']['mean'])")
echo "smoke leakage mean $leak"
python -c "import sys; sys.exit(0 if float('$leak') < 1e-5 else 1)" || { echo "SMOKE FAILED: leakage $leak"; exit 1; }

echo "=== FULL"
RUNS=A4_gen_s1,A4_gen_s2 BOOKS=2,3,4 BASE=1 OUT=/content/results.json python run_cl.py && echo "FULL DONE"
