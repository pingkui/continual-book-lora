"""Kaggle TPU v3-8 environment probe for the JAX plan (tpu/JAX_PLAN.md, phase 0).
Prints: host (CPU, RAM, disk), installed versions, JAX devices and per-device HBM, a bf16 matmul benchmark on one
device and on all devices, internet access, and whether a recent jax[tpu] (the Colab version) installs and runs on v3.
Each check runs in its own subprocess with a timeout, so one hang or crash does not hide the rest."""
import os
import platform
import shutil
import subprocess
import sys
import time

t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

say("python", sys.version.split()[0], platform.platform(), "cpus", os.cpu_count())
mem = {l.split(":")[0]: l.split()[1] for l in open("/proc/meminfo")}
say("host RAM GB", round(int(mem["MemTotal"]) / 2**20, 1), "available", round(int(mem["MemAvailable"]) / 2**20, 1))
for d in ("/kaggle/working", "/tmp"):
    if os.path.exists(d):
        say("disk free GB", d, round(shutil.disk_usage(d).free / 2**30, 1))
say("TPU env", {k: v for k, v in os.environ.items() if "TPU" in k or "XLA" in k})

pkgs = ["jax", "jaxlib", "libtpu", "libtpu-nightly", "flax", "optax", "orbax-checkpoint", "torch", "torch_xla",
        "transformers", "tensorflow", "safetensors", "tokenizers", "numpy"]
out = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], capture_output=True, text=True).stdout
have = {l.split("==")[0].lower().replace("_", "-"): l.split("==")[-1] for l in out.splitlines() if "==" in l}
say("versions", {p: have.get(p.lower().replace("_", "-"), "-") for p in pkgs})

BENCH = r'''
import time, jax, jax.numpy as jnp
devs = jax.devices()
print("jax", jax.__version__, "backend", jax.default_backend(), "n_devices", len(devs), flush=True)
for d in devs:
    ms = d.memory_stats() or {}
    print("  ", d.id, d.device_kind, "coords", getattr(d, "coords", None), "core", getattr(d, "core_on_chip", None),
          "HBM GB", round(ms.get("bytes_limit", 0) / 2**30, 2), flush=True)
n = 8192
x = jnp.ones((n, n), jnp.bfloat16)
f = jax.jit(lambda a: a @ a)
f(x).block_until_ready()
t = time.time()
for _ in range(10):
    y = f(x)
y.block_until_ready()
dt = (time.time() - t) / 10
print("one device bf16 matmul TFLOPS", round(2 * n**3 / dt / 1e12, 1), flush=True)
k = len(devs)
xs = jnp.ones((k, n, n), jnp.bfloat16)
g = jax.pmap(lambda a: a @ a)
g(xs).block_until_ready()
t = time.time()
for _ in range(10):
    ys = g(xs)
ys.block_until_ready()
dt = (time.time() - t) / 10
print("all devices bf16 matmul TFLOPS", round(k * 2 * n**3 / dt / 1e12, 1), flush=True)
s = jax.pmap(lambda a: jax.lax.psum(a, "i"), axis_name="i")
v = jnp.ones((k, 64 * 2**20 // 2), jnp.bfloat16)
s(v).block_until_ready()
t = time.time()
for _ in range(5):
    r = s(v)
r.block_until_ready()
print("psum 64 MB per device, ms", round((time.time() - t) / 5 * 1000, 2), flush=True)
'''


def run(tag, cmd, timeout, env=None):
    say(f"--- {tag}")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        text = (p.stdout + p.stderr).strip().splitlines()
        keep = [l for l in text if not l.startswith(("I0000", "W0000", "WARNING: All log"))]
        print("\n".join(keep[-40:]), flush=True)
        say(f"{tag}: exit {p.returncode}")
    except subprocess.TimeoutExpired:
        say(f"{tag}: TIMEOUT after {timeout}s")


run("preinstalled jax", [sys.executable, "-c", BENCH], 600)

try:
    import urllib.request
    for url in ("https://huggingface.co", "https://pypi.org/simple/jax/"):
        say("internet", url, urllib.request.urlopen(url, timeout=20).status)
except Exception as e:
    say("internet failed", repr(e)[:200])

# the Colab version; installed into a separate directory so the image stays untouched
target = "/tmp/newjax"
run("pip install jax[tpu]==0.11.2", [sys.executable, "-m", "pip", "install", "-q", "--target", target,
                                     "jax[tpu]==0.11.2"], 900)
if os.path.exists(target):
    run("jax 0.11.2 on v3", [sys.executable, "-c", BENCH], 600, env={**os.environ, "PYTHONPATH": target})
say("PROBE DONE")
