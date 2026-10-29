#!/bin/bash
# Round-14 teacher templates on a Colab A100 (gen_templates_v14.py): a DRY run first (Qwen2.5-0.5B on the CPU), then
# the real one (Qwen2.5-7B teacher, Vikhr-Nemo-12B judge).
set -u
cd /content/tpl14
pip uninstall -y -q torchao 2>/dev/null
export PYTHONUNBUFFERED=1 TPL=/content/tpl14/data_tpl/templates.json
mkdir -p dry out
DRY=1 OUT=dry python3 gen_templates_v14.py > dry.log 2>&1 || { echo "DRY FAILED"; tail -20 dry.log; exit 1; }
tail -3 dry.log
echo "DRY OK"
OUT=out python3 gen_templates_v14.py
