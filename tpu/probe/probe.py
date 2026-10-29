"""TPU environment probe for Kaggle: what is installed, what the devices are, can JAX / torch_xla run, rough bf16 speed."""
import importlib
import json
import os
import platform
import subprocess
import sys
import time

out = {"python": sys.version.split()[0], "platform": platform.platform(), "env": {k: v for k, v in os.environ.items() if "TPU" in k or "XLA" in k or "JAX" in k}}


def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "ok")
    except Exception as e:
        return f"missing: {repr(e)[:80]}"


out["versions"] = {n: ver(n) for n in ["jax", "jaxlib", "flax", "optax", "torch", "torch_xla", "transformers", "peft", "tensorflow", "keras", "keras_hub", "maxtext"]}
out["pip_tpu"] = subprocess.run("pip list 2>/dev/null | grep -i -E 'jax|libtpu|torch|xla|flax|optax|tensorflow|keras|transformers|peft|tpu'", shell=True, capture_output=True, text=True).stdout.splitlines()
out["cpu_mem"] = subprocess.run("nproc; free -g | head -2", shell=True, capture_output=True, text=True).stdout


def try_jax():
    r = {}
    try:
        import jax
        import jax.numpy as jnp
        devs = jax.devices()
        r["devices"] = [str(d) for d in devs]
        r["kind"] = devs[0].device_kind
        try:
            r["mem_stats"] = {k: v for k, v in devs[0].memory_stats().items() if "bytes" in k}
        except Exception as e:
            r["mem_stats"] = repr(e)[:80]
        x = jnp.ones((8192, 8192), jnp.bfloat16)
        f = jax.jit(lambda a: a @ a)
        f(x).block_until_ready()
        t = time.time()
        for _ in range(10):
            y = f(x)
        y.block_until_ready()
        dt = (time.time() - t) / 10
        r["bf16_matmul_8192_tflops_one_core"] = round(2 * 8192 ** 3 / dt / 1e12, 1)
    except Exception as e:
        r["error"] = repr(e)[:300]
    return r


out["jax"] = try_jax()
print(json.dumps(out, indent=1, ensure_ascii=False), flush=True)
# torch_xla and jax both want the TPU runtime; run the torch_xla probe in a child process so they cannot collide
XLA_SRC = 'import time\n\n\ndef try_xla():\n    r = {}\n    try:\n        import torch\n        import torch_xla.core.xla_model as xm\n        d = xm.xla_device()\n        r["device"] = str(d)\n        a = torch.ones(4096, 4096, dtype=torch.bfloat16, device=d)\n        t = time.time()\n        for _ in range(10):\n            b = a @ a\n        xm.mark_step()\n        xm.wait_device_ops()\n        r["bf16_matmul_4096_s"] = round(time.time() - t, 3)\n        try:\n            import torch_xla\n            r["world_size"] = torch_xla.runtime.world_size()\n        except Exception as e:\n            r["world_size"] = repr(e)[:80]\n    except Exception as e:\n        r["error"] = repr(e)[:300]\n    return r\n'
child = subprocess.run([sys.executable, "-c", XLA_SRC + "\nimport json\nprint('XLA_JSON', json.dumps(try_xla()))"], capture_output=True, text=True)
out["torch_xla"] = child.stdout[-600:] + child.stderr[-600:]
os.makedirs("/kaggle/working", exist_ok=True)
json.dump(out, open("/kaggle/working/probe.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out["torch_xla"], ensure_ascii=False))
