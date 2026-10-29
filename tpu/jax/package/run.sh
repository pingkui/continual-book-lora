#!/usr/bin/env bash
# run.sh BOOK.json OUT_DIR [--resume]
# One call per book, in order, the same OUT_DIR for every book of a run. Learns the book into Qwen2.5-7B-Instruct on one
# 48 GB card and leaves one set of merged weights (Qwen2ForCausalLM, same shapes, no remote code) in OUT_DIR/model.
# State kept between books in OUT_DIR: lora/ (per-book factors), state.json (books done), base/ (pinned base weights),
# venv/ + pyroot/ (the package's own Python and packages, installed from env/ without network), src/ (book files,
# teacher-written rows), teacher.log (every teacher call), ckpt/ (checkpoints of the book in progress, for --resume).
# Network: Hugging Face for the pinned base at the first call; nothing else (the teacher is the base itself).
set -euo pipefail
BOOK=${1:?usage: run.sh BOOK.json OUT_DIR [--resume]}
OUT=${2:?usage: run.sh BOOK.json OUT_DIR [--resume]}
RESUME=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
LOG=$OUT/run.log
exec > >(tee -a "$LOG") 2>&1
T_START=$(date +%s)
echo "[run.sh] $(date -u +%FT%TZ) book=$BOOK out=$OUT resume=${RESUME:-no} package=$HERE"

# ---- budget guard (README "GPU hours") ------------------------------------------------------------------------------
# The card time of a book is this script's start to its exit (section 7). Training steps stop at
# T_START + (BUDGET_H - RESERVE_H); the reserve covers the self-test, the subspace update, saving and the export, so the
# call ends with a model inside BUDGET_H. Defaults 8 h and 1 h; the dry run and tests may set smaller values.
BUDGET_H=${BUDGET_H:-8}
RESERVE_H=${RESERVE_H:-1}
DEADLINE=$(awk -v s="$T_START" -v b="$BUDGET_H" -v r="$RESERVE_H" 'BEGIN { printf "%d", s + (b - r) * 3600 }')
echo "[run.sh] budget: BUDGET_H=$BUDGET_H RESERVE_H=$RESERVE_H -> training stops after $(date -u -d @"$DEADLINE" +%FT%TZ)"

BASE_ID=${BASE_ID:-Qwen/Qwen2.5-7B-Instruct}
BASE_REV=${BASE_REV:-a09a3545}   # the pinned base, as the terms name it (commit prefix); falls back to main if HF rejects it

# ---- 1. python environment (once per OUT_DIR), from the package, no network -----------------------------------------
# env/python.tar.gz: CPython 3.12 (python-build-standalone), env/wheels/: every package pinned in env/requirements.txt.
if [ -f "$HERE/env/python.tar.gz" ] && [ -d "$HERE/env/wheels" ]; then
  PY=$OUT/pyroot/python/bin/python3
  if [ ! -f "$OUT/pyroot/.ready" ]; then
    echo "[run.sh] installing the package's python and wheels into $OUT/pyroot (no network)"
    rm -rf "$OUT/pyroot"; mkdir -p "$OUT/pyroot"
    tar xzf "$HERE/env/python.tar.gz" -C "$OUT/pyroot"
    "$PY" -m pip install -q --no-index --find-links "$HERE/env/wheels" -r "$HERE/env/requirements.txt"
    touch "$OUT/pyroot/.ready"
  fi
else
  PY=$OUT/venv/bin/python
  if [ ! -x "$PY" ]; then
    echo "[run.sh] WARNING: env/ bundle missing; development fallback: host python3 + PyPI"
    python3 -m venv "$OUT/venv" || { python3 -m pip install -q virtualenv && python3 -m virtualenv -q "$OUT/venv"; }
    "$OUT/venv/bin/pip" install -q --upgrade pip
    "$OUT/venv/bin/pip" install -q -r "$HERE/requirements.txt"
  fi
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/[run.sh] gpu: /' || true
"$PY" - <<'EOF'
import os, jax, torch, transformers
dev = jax.devices()
print("[run.sh] jax", jax.__version__, dev, "| torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__)
if dev[0].platform != "gpu" and not os.environ.get("ALLOW_CPU"):
    raise SystemExit("[run.sh] no GPU visible to JAX (driver too old for the CUDA 12 wheels, or no card): stopping")
EOF

# ---- 2. the pinned base (once per OUT_DIR) --------------------------------------------------------------------------
if [ ! -f "$OUT/base/.complete" ]; then
  echo "[run.sh] downloading $BASE_ID@$BASE_REV"
  # a stalled connection fails after 60 s without data and the attempt is retried (partial files resume), 6 attempts
  export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DISABLE_TELEMETRY=1
  for attempt in 1 2 3 4 5 6; do
    if "$PY" - "$BASE_ID" "$BASE_REV" "$OUT/base" <<'EOF'
import sys
from huggingface_hub import snapshot_download
pats = ["*.json", "*.safetensors", "*.txt", "*.model"]
try:
    snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=pats, max_workers=4)
except Exception as e:
    if "evision" not in repr(e) and "404" not in repr(e):
        raise
    print("[run.sh] revision", sys.argv[2], "not accepted:", repr(e)[:120], "- using main")
    snapshot_download(sys.argv[1], local_dir=sys.argv[3], allow_patterns=pats, max_workers=4)
EOF
    then touch "$OUT/base/.complete"; break; else echo "[run.sh] download attempt $attempt failed; retrying in 20 s"; sleep 20; fi
  done
  [ -f "$OUT/base/.complete" ] || { echo "[run.sh] base download failed 6 times"; exit 4; }
  echo "[run.sh] base: $(ls "$OUT/base"/*.safetensors | wc -l) safetensors files, $(du -sh "$OUT/base" | cut -f1)"
fi

# ---- 3. which book is this -----------------------------------------------------------------------------------------
mkdir -p "$OUT/src" "$OUT/lora" "$OUT/ckpt"
cp "$HERE/code/factlab.py" "$OUT/src/"     # run_cl.py imports the scoring helpers from SRC
STATE=$OUT/state.json
[ -f "$STATE" ] || echo '{"done": []}' > "$STATE"
K=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['done']) + 1)" "$STATE")
if [ -n "$RESUME" ] && [ -f "$OUT/src/book$K.json" ]; then
  echo "[run.sh] resuming book $K from $OUT/ckpt"
else
  cp "$BOOK" "$OUT/src/book$K.json"
  rm -rf "$OUT/ckpt"; mkdir -p "$OUT/ckpt"
fi
echo "[run.sh] this is book $K; books done: $(cat "$STATE")"
N_UNITS=$("$PY" -c "import sys; sys.path.insert(0, sys.argv[1]); import generic; print(len(generic.load_units(sys.argv[2])))" "$HERE/code" "$OUT/src/book$K.json")
echo "[run.sh] $N_UNITS units"
echo "{\"time\": \"$(date -u +%FT%TZ)\", \"event\": \"book $K start\", \"units\": $N_UNITS, \"teacher\": \"$BASE_ID (the pinned base, local)\", \"budget_usd\": ${TEACHER_BUDGET_USD:-20}}" >> "$OUT/teacher.log"

# ---- 4. learn the book -----------------------------------------------------------------------------------------------
# Earlier books are re-merged from their saved factors (LOAD_LORA) and their protected subspaces rebuilt; only book K is
# trained. Rules as formulas: README "Rules". Memory for a 48 GB card: fp16 protection bases, gradient checkpointing,
# micro-batch 8, generation batch 16, JAX preallocates 75% of the card (36 GB; measured peak 30-33 GiB), torch's eigh
# has the rest (it falls back to the CPU on a card its build has no kernels for).
BOOKS=$(seq -s, 1 "$K")
LOAD=""
for b in $(seq 1 $((K - 1))); do
  if [ "$b" = 1 ]; then f=$OUT/lora/A4_chat90f/book1.npz; else f=$OUT/lora/A4_chat90f_sr12/book$b.npz; fi
  [ -f "$f" ] || { echo "[run.sh] missing factors of book $b: $f"; exit 2; }
  LOAD="${LOAD:+$LOAD,}$b=$f"
done
GENERIC_BOOKS=$(seq -s, 1 "$K")
if [ "$N_UNITS" -ge 1000 ]; then EPOCHS=6; else EPOCHS=12; fi
[ -n "${EPOCHS_OVERRIDE:-}" ] && EPOCHS=$EPOCHS_OVERRIDE          # tests only
for v in NFEAT GEN_N ROWS EXAMN DTYPE JAX_PLATFORMS ALLOW_CPU; do [ -n "${!v:-}" ] && export "$v"; done   # tests only
cd "$HERE/code"
export PYTHONUNBUFFERED=1 SRC=$OUT/src MODEL=$OUT/base RANK=32 \
       XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.75} \
       XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" JAX_COMPILATION_CACHE_DIR=$OUT/jaxcache \
       Q_DTYPE=fp16 REMAT=1 MICRO=${MICRO:-8} GEN_BS=${GEN_BS:-48} DIAG_STEPS=1 \
       CHAT_FILE=$OUT/src/chat_self.json GEN_FILE=$HERE/data/general_ru.json GENERIC_BOOKS=$GENERIC_BOOKS GENERIC_PAT='book{k}.json' \
       GENERIC_PREM=${GENERIC_PREM:-10} \
       RUNS=A4_chat90f BOOKS=$BOOKS EPOCHS=$EPOCHS REPLAY_TEST=300 REPLAY_CAP=1.0 \
       SAVE_LORA=$OUT/lora CKPT=$OUT/ckpt OUT_JSON=$OUT/results_book$K.json \
       DEADLINE=$DEADLINE EARLY_CKPT=${EARLY_CKPT:-200} \
       TEACHER_LOG=$OUT/teacher.log TEACHER_BUDGET_USD=${TEACHER_BUDGET_USD:-20} TEACHER_PRICE_PER_1K=0
[ -n "${TORCH_MEM_FRACTION:-}" ] && export TORCH_MEM_FRACTION
if [ "$K" -ge 2 ]; then export BRANCH=1 BRANCH_TO_END=1 BRANCHES=sr12; fi
[ -n "$LOAD" ] && export LOAD_LORA=$LOAD
t0=$(date +%s)
OUT=$OUT_JSON "$PY" run_cl.py
echo "[run.sh] book $K learned in $(( $(date +%s) - t0 )) s"

# ---- 5. one set of weights for the answer contour ---------------------------------------------------------------------
FACTORS=$OUT/lora/A4_chat90f/book1.npz
for b in $(seq 2 "$K"); do FACTORS="$FACTORS $OUT/lora/A4_chat90f_sr12/book$b.npz"; done
rm -rf "$OUT/model.tmp"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 "$PY" export_lora.py "$OUT/base" "$OUT/model.tmp" $FACTORS
rm -rf "$OUT/model"; mv "$OUT/model.tmp" "$OUT/model"
"$PY" - "$STATE" "$K" "$OUT/src/book$K.json" <<'EOF'
import hashlib, json, sys
st = json.load(open(sys.argv[1]))
st["done"].append({"book": int(sys.argv[2]), "sha256": hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()})
json.dump(st, open(sys.argv[1], "w"), indent=1)
EOF
rm -rf "$OUT/ckpt"
echo "{\"time\": \"$(date -u +%FT%TZ)\", \"event\": \"book $K done\", \"card_minutes\": $(( ($(date +%s) - T_START) / 60 ))}" >> "$OUT/teacher.log"
echo "[run.sh] $(date -u +%FT%TZ) done: merged weights in $OUT/model, factors in $OUT/lora, state $(cat "$STATE"); card time $(( ($(date +%s) - T_START) / 60 )) min"
