"""Unmerged adapters ({kind}_fa / {kind}_fb, added by qwen.linear) must equal merging a @ b * scale into the weights.
fp32 on CPU, Qwen2.5-0.5B, two random adapters with non-zero b: compares forward logits and greedy generations.
Usage (code dir on sys.path): python tests/test_unmerged.py"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
import jax.numpy as jnp
import numpy as np

import generate
import qwen
import train

path = qwen.model_dir("Qwen/Qwen2.5-0.5B-Instruct")
cfg, params = qwen.load_params(path, jnp.float32)
gcfg = generate.gen_config(path)
scale = 2.0
loras = []
for s in (1, 2):
    lo = train.init_lora(cfg, jax.random.PRNGKey(s), 16)
    loras.append({k: {"a": d["a"], "b": jax.random.normal(jax.random.PRNGKey(10 + s), d["b"].shape) * 0.02}
                  for k, d in lo.items()})

merged = dict(params["layers"])
stacked = dict(params["layers"])
for lo in loras:
    for k, d in lo.items():
        merged[f"{k}_w"] = merged[f"{k}_w"] + jnp.einsum("lir,lro->lio", d["a"], d["b"],
                                                         precision=jax.lax.Precision.HIGHEST) * scale
        a, b = d["a"], d["b"] * scale
        if f"{k}_fa" in stacked:
            a, b = jnp.concatenate([stacked[f"{k}_fa"], a], -1), jnp.concatenate([stacked[f"{k}_fb"], b], 1)
        stacked[f"{k}_fa"], stacked[f"{k}_fb"] = a, b
pm, ps = {**params, "layers": merged}, {**params, "layers": stacked}

rng = np.random.default_rng(0)
ids = jnp.asarray(rng.integers(100, 20000, (2, 24)), jnp.int32)
mask = jnp.ones_like(ids)
lm = jax.jit(lambda p: qwen.forward(cfg, p, ids, mask))(pm)
ls = jax.jit(lambda p: qwen.forward(cfg, p, ids, mask))(ps)
rel = float(jnp.abs(lm - ls).max() / jnp.abs(lm).max())
print(f"forward logits: max |merged - unmerged| / max |logit| = {rel:.2e}")
g = generate.make_generate(cfg, gcfg, 12)
tm, ts = np.asarray(g(pm, ids, mask)), np.asarray(g(ps, ids, mask))
print("greedy tokens identical:", bool((tm == ts).all()))
assert rel < 1e-4 and (tm == ts).all()
print("UNMERGED OK")
