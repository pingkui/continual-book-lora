import subprocess, json
cfg = json.load(open("/content/infer_cfg.json"))
NAME, RUNS = cfg["name"], cfg["runs"]   # runs: list of [dtype, batch]
script = r'''cd /content && chmod 600 rclone.conf
curl -sSLO https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip && python3 -c "import zipfile; zipfile.ZipFile('rclone-v1.75.2-linux-amd64.zip').extract('rclone-v1.75.2-linux-amd64/rclone', '/content')" && chmod +x /content/rclone-v1.75.2-linux-amd64/rclone
RC="/content/rclone-v1.75.2-linux-amd64/rclone --drive-chunk-size 256M"; export RCLONE_CONFIG=/content/rclone.conf
D=gdrive:kaggle/@@NAME@@-20261030
( while true; do $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1; sleep 60; done ) &
rm -rf l5 && tar xzf l5n2.tgz && mv l5f l5 && mkdir -p l5/out
pip uninstall -y -q torchao 2>/dev/null
pip install -q "lm_eval[ifeval] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@ddd67220" || echo "PINNED INSTALL FAILED"
cd /content/l5/code
echo "=== INFER @@NAME@@: base model IFEval + GSM8K under @@RUNS@@ (target: client base IFEval 386/541 = 71.35, GSM8K strict 992/1319 = 75.21) ==="
for cfg in @@RUNS@@; do
  DT=${cfg%%:*}; BS=${cfg##*:}
  L5_TASKS="ifeval gsm8k" L5_DTYPE=$DT L5_BS=$BS bash run_l5.sh model base_${DT}_bs${BS} Qwen/Qwen2.5-7B-Instruct; echo "base $DT $BS exit $?"
  $RC copy /content/l5/out $D/out 2>&1 | grep -v NOTICE; $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1
done
echo "@@NAME@@ CELL DONE"
$RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1'''
script = script.replace("@@NAME@@", NAME).replace("@@RUNS@@", " ".join(f"{d}:{b}" for d, b in RUNS))
open(f"/content/{NAME}.sh", "w").write(script)
subprocess.Popen(f"nohup bash /content/{NAME}.sh > /content/{NAME}.log 2>&1 &", shell=True, executable="/bin/bash")
print(NAME, "launched in the background")
