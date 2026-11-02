#!/usr/bin/env bash
# Rebuild the two tarballs the Colab launch cells (tpu/cells/*.py) upload to /content:
#   cl.tgz  -> code/ (tpu/jax/*.py) + data/ (lab/data_v18 synthetic books, exams, facts + lab/factlab.py)
#              the cells untar it into /content/sr and run code/run_cl.py with SRC=/content/sr/data
#   l5.tgz  -> l5f/code (run_l5.sh, export_lora.py, check_export.py, qwen.py, random_lora.py) + l5f/data (exams)
#              the cells untar it into /content and rename l5f -> l5
# The cells refer to them by older names (cl_v60.tgz / cl_v61.tgz, l5f.tgz / l5g.tgz / l5n2.tgz); upload these
# files under those names, e.g.  colab upload -s NAME OUT/cl.tgz /content/cl_v60.tgz
# usage: bash tools/build_bundles.sh OUT_DIR
set -euo pipefail
OUT=${1:?usage: build_bundles.sh OUT_DIR}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
mkdir -p "$TMP/cl/code" "$TMP/cl/data" "$TMP/l5f/code" "$TMP/l5f/data" "$OUT"
cp "$ROOT"/tpu/jax/*.py "$TMP/cl/code/"
cp "$ROOT"/lab/data_v18/* "$ROOT/lab/factlab.py" "$TMP/cl/data/"
tar czf "$OUT/cl.tgz" -C "$TMP/cl" .
cp "$ROOT"/tpu/jax/{run_l5.sh,export_lora.py,check_export.py,qwen.py,random_lora.py} "$TMP/l5f/code/"
cp "$ROOT"/lab/data_v18/book*.exam.json "$TMP/l5f/data/"
tar czf "$OUT/l5.tgz" -C "$TMP" l5f
cp "$ROOT/lab/hf_trap_answers.py" "$OUT/"
cp "$ROOT/tpu/results/v26-20261026/chat_self_7b.json" "$OUT/"
rm -rf "$TMP"
ls -la "$OUT"
