"""Why is the protected basis not orthonormal on the TPU? Real layer inputs of Qwen2.5-1.5B (book-2 rows), the
basis V = X^T W / sqrt(ev) from the Gram eigh computed (a) on the TPU as run_cl.py does, (b) on the TPU with every
matmul at HIGHEST, (c) on the host in float64. Prints eigenvalue error, |V^T V - I|, and the principal-angle overlap
between each basis and the float64 one. Usage: python diag_basis.py DATA_DIR"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/content/code")
import qwen

t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:5.1f}s]", *a, flush=True)
from transformers import AutoTokenizer

path = qwen.model_dir("Qwen/Qwen2.5-1.5B-Instruct")
tok = AutoTokenizer.from_pretrained(path)
cfg, params = qwen.load_params(path, jnp.bfloat16)
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/book2.train.jsonl")][:64]
rows = [r for r in rows if r["type"] == "text"] or rows
enc = [tok(r.get("text") or r["assistant"])["input_ids"][:64] for r in rows[:64]]
ids = np.array([x + [tok.pad_token_id] * (64 - len(x)) for x in enc], np.int32)
att = np.array([[1] * len(x) + [0] * (64 - len(x)) for x in enc], np.int32)
cap = jax.jit(lambda p, i, m: qwen.forward(cfg, p, i, m, capture=True)[1])(params, jnp.asarray(ids), jnp.asarray(att))
m = att.reshape(-1) > 0
say("captured", {k: v.shape for k, v in cap.items()}, "real tokens", int(m.sum()))


def basis(X, tau=0.99):
    G = X @ X.T
    ev, W = jnp.linalg.eigh(G)
    ev, W = jnp.clip(ev[::-1], 1e-8), W[:, ::-1]
    k = jnp.sum(jnp.cumsum(ev) / jnp.sum(ev) < tau) + 1
    return ev, (X.T @ W) / jnp.sqrt(ev), k


f_default = jax.jit(basis)


@jax.jit
def f_high(X):
    with jax.default_matmul_precision("highest"):
        return basis(X)


for name, layer in (("attn_in", 0), ("attn_in", 27), ("down_in", 0), ("down_in", 27), ("o_in", 14)):
    X = np.asarray(cap[name][layer].reshape(-1, cap[name].shape[-1])[m], np.float32)[:512]
    G64 = X.astype(np.float64) @ X.T.astype(np.float64)
    ev64, W64 = np.linalg.eigh(G64)
    ev64, W64 = np.clip(ev64[::-1], 1e-8, None), W64[:, ::-1]
    k64 = int((np.cumsum(ev64) / ev64.sum() < 0.99).sum()) + 1
    V64 = (X.T.astype(np.float64) @ W64[:, :k64]) / np.sqrt(ev64[:k64])
    say(f"{name}[{layer}] X {X.shape}, |X| max {np.abs(X).max():.1f}, ev max {ev64[0]:.3e}, ev[k] {ev64[k64 - 1]:.3e}, k {k64}")
    for tag, fn in (("tpu default", f_default), ("tpu highest", f_high)):
        ev, V, k = (np.asarray(a) for a in fn(jnp.asarray(X)))
        k = int(k)
        V = V[:, :k].astype(np.float64)
        orth = np.abs(V.T @ V - np.eye(k)).max()
        rel = np.abs(ev[:k64] - ev64[:k64]).max() / ev64[0]
        kk = min(k, k64)
        s = np.linalg.svd(np.linalg.qr(V[:, :kk])[0].T @ V64[:, :kk], compute_uv=False)
        say(f"   {tag}: k {k}, eigval err/max {rel:.1e}, |V^T V - I| {orth:.1e}, overlap with f64 basis: min cos {s.min():.4f}")
say("DIAG DONE")
