import subprocess, json
cfg = json.load(open("/content/noise_cfg.json"))
NAME, SEED, BDT, BBS = cfg["name"], cfg["seed"], cfg["base_dtype"], cfg["base_bs"]
script = r'''cd /content && chmod 600 rclone.conf
curl -sSLO https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip && python3 -c "import zipfile; zipfile.ZipFile('rclone-v1.75.2-linux-amd64.zip').extract('rclone-v1.75.2-linux-amd64/rclone', '/content')" && chmod +x /content/rclone-v1.75.2-linux-amd64/rclone
RC="/content/rclone-v1.75.2-linux-amd64/rclone --drive-chunk-size 256M"; export RCLONE_CONFIG=/content/rclone.conf
D=gdrive:kaggle/@@NAME@@-20261030
( while true; do $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1; sleep 60; done ) &
rm -rf l5 && tar xzf l5n.tgz && mv l5f l5 && mkdir -p l5/out l5/lora
pip uninstall -y -q torchao 2>/dev/null
pip install -q "lm_eval[ifeval] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@ddd67220" || echo "PINNED INSTALL FAILED"
$RC copy gdrive:kaggle/gen22-20261029/lora/A4_chat90f_s1/book1.npz /content/l5/lora/ 2>&1 | grep -v NOTICE
cd /content/l5/code
echo "=== NOISE @@NAME@@: random-direction LoRA seeds @@SEED@@ and @@SEED2@@ (same per-layer update size as gen22 book 1); base with dtype @@BDT@@ batch @@BBS@@ ==="
export BASE=Qwen/Qwen2.5-7B-Instruct MMLU_BS=8
L5_DTYPE=@@BDT@@ L5_BS=@@BBS@@ bash run_l5.sh model base_@@BDT@@_bs@@BBS@@ Qwen/Qwen2.5-7B-Instruct; echo "base exit $?"
$RC copy /content/l5/out $D/out 2>&1 | grep -v NOTICE
for RS in @@SEED@@ @@SEED2@@; do
  python3 random_lora.py /content/l5/lora/book1.npz /content/l5/lora/random_$RS.npz $RS || echo "RANDOM FAILED $RS"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python export_lora.py Qwen/Qwen2.5-7B-Instruct /content/l5/random /content/l5/lora/random_$RS.npz || echo "EXPORT FAILED $RS"
  bash run_l5.sh model random_s$RS /content/l5/random; echo "random $RS exit $?"
  rm -rf /content/l5/random
  $RC copy /content/l5/out $D/out 2>&1 | grep -v NOTICE; $RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1
done
echo "@@NAME@@ CELL DONE"
$RC copy /content/@@NAME@@.log $D/live/ >/dev/null 2>&1'''
for k, v in (("@@NAME@@", NAME), ("@@SEED2@@", str(SEED + 10)), ("@@SEED@@", str(SEED)), ("@@BDT@@", BDT), ("@@BBS@@", str(BBS))):
    script = script.replace(k, v)
open(f"/content/{NAME}.sh", "w").write(script)
subprocess.Popen(f"nohup bash /content/{NAME}.sh > /content/{NAME}.log 2>&1 &", shell=True, executable="/bin/bash")
print(NAME, "launched in the background")
