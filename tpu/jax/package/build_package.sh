#!/usr/bin/env bash
# Assemble the frozen package directory from the working tree: code/, data/ (shipped general-text rows), run.sh,
# requirements.txt, README.md, then the manifest. Run from tpu/jax: bash package/build_package.sh OUT_PKG_DIR
set -euo pipefail
PKG=${1:?usage: build_package.sh OUT_PKG_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/.." && pwd)
rm -rf "$PKG"; mkdir -p "$PKG/code" "$PKG/data"
cp "$SRC"/run_cl.py "$SRC"/train.py "$SRC"/qwen.py "$SRC"/generate.py "$SRC"/export_lora.py "$SRC"/generic.py "$PKG/code/"
cp "$SRC/../../lab/factlab.py" "$PKG/code/"            # scoring helpers (matches, retention) used by the self-test
[ -f "$HERE/data/general_ru.json" ] && cp "$HERE/data/general_ru.json" "$PKG/data/" || echo "WARNING: data/general_ru.json missing (general-text rows would need the network)"
cp "$HERE/run.sh" "$HERE/requirements.txt" "$HERE/README.md" "$HERE/LICENSES.md" "$HERE/make_manifest.sh" "$PKG/"
chmod +x "$PKG/run.sh" "$PKG/make_manifest.sh"
# env/: the package's own interpreter and wheels (ENV_SRC points at a directory holding python.tar.gz, wheels/,
# requirements.txt); without it run.sh falls back to the host python3 + PyPI (development only).
if [ -n "${ENV_SRC:-}" ] && [ -f "$ENV_SRC/python.tar.gz" ]; then
  mkdir -p "$PKG/env"; cp "$ENV_SRC/python.tar.gz" "$ENV_SRC/requirements.txt" "$PKG/env/"; cp -r "$ENV_SRC/wheels" "$PKG/env/"
else
  echo "WARNING: ENV_SRC not set or incomplete: package without env/ (run.sh will need PyPI)"
fi
(cd "$PKG" && bash make_manifest.sh)
echo "package at $PKG"; du -sh "$PKG"
