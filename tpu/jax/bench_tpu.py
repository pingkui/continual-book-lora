"""Phases 1-2 on the TPU (Colab v5e-1), Qwen2.5-1.5B bf16, three checks printed one per line:
T1 forward loss on the 16 rows used by tpu/diag.py (CPU fp32 reference 4.5958; torch_tpu bf16 gave 4.5945 / 4.6093),
T2 30-step memorisation of those rows, constant lr 2e-4, no dropout (torch_tpu eager: 4.61 -> 0.06),
T3 speed of the real training step (4 x 8 rows, dropout 0.05, projection with k=512 per kind) at T = 64 / 96 / 128,
   plus the token-length mix of the A4_gen rows of book 2, giving the expected time per epoch.
Usage: python bench_tpu.py DATA_DIR  (DATA_DIR holds diag_rows.json and book2.train.jsonl)"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import qwen
import train

t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)
DATA = sys.argv[1] if len(sys.argv) > 1 else "/content/data"
say("jax", jax.__version__, jax.devices())

from transformers import AutoTokenizer

name = "Qwen/Qwen2.5-1.5B-Instruct"
path = qwen.model_dir(name)
tok = AutoTokenizer.from_pretrained(path)
cfg, params = qwen.load_params(path, jnp.bfloat16)
say("loaded", name, {k: cfg[k] for k in ("L", "D", "F", "H", "KV")})

# T1 / T2: same 16 rows and shapes as tpu/diag.py
rows16 = json.load(open(f"{DATA}/diag_rows.json"))
enc = [tok(r["text"] + tok.eos_token)["input_ids"][:48] for r in rows16]
ids = jnp.array([x + [tok.pad_token_id] * (48 - len(x)) for x in enc])
lab = jnp.array([x + [-100] * (48 - len(x)) for x in enc])
att = jnp.array([[1] * len(x) + [0] * (48 - len(x)) for x in enc])
loss_fn = jax.jit(lambda p, i, m, l: qwen.lm_loss(qwen.forward(cfg, p, i, m), l))  # params passed, not closed over
say("T1 forward loss bf16", round(float(loss_fn(params, ids, att, lab)), 4), "(CPU fp32 reference 4.5958)")

lora = train.init_lora(cfg, jax.random.PRNGKey(0))
opt = train.make_opt()
state = opt.init(lora)
step = train.make_step(cfg, opt, rate=0.0)
hist = []
for s in range(30):
    lora, state, loss = step(params, lora, state, {}, ids[None], lab[None], att[None], jax.random.PRNGKey(s), 2e-4)
    if s % 5 == 0 or s == 29:
        hist.append(round(float(loss), 3))
say("T2 memorise loss every 5 steps", hist, "(torch_tpu eager: 4.609 ... 0.06)")

# T3: speed of the real step
rows = [json.loads(l) for l in open(f"{DATA}/book2.train.jsonl")]
rows = [r for r in rows if r["tag"] in ("text", "qa", "text_g", "qa_g", "premise_g")]


def encode(row):
    if row["type"] == "text":
        x = tok(row["text"] + tok.eos_token)["input_ids"]
        return x, x[:]
    p = tok(tok.apply_chat_template([{"role": "user", "content": row["user"]}], tokenize=False,
                                    add_generation_prompt=True))["input_ids"]
    a = tok(row["assistant"] + "<|im_end|>")["input_ids"]
    return p + a, [-100] * len(p) + a


lens = np.array([len(encode(r)[0]) for r in rows])
rng = np.random.default_rng(0)
order = rng.permutation(len(rows))
steps_max = [lens[order[i:i + 32]].max() for i in range(0, len(rows), 32)]
buckets = (64, 96, 128)
mix = {b: sum(1 for m in steps_max if m <= b and (b == 64 or m > buckets[buckets.index(b) - 1])) for b in buckets}
say("A4_gen book 2:", len(rows), "rows, token length mean", round(lens.mean(), 1), "max", lens.max(),
    "| steps per epoch by bucket", mix)

Qs = {k: jnp.asarray(np.linalg.qr(rng.standard_normal((train.lora_dims(cfg)[k][0], 512)))[0], jnp.float32)[None]
      .repeat(cfg["L"], 0) for k in qwen.KINDS}
lora = train.init_lora(cfg, jax.random.PRNGKey(1))
opt = train.make_opt()
state = opt.init(lora)
step = train.make_step(cfg, opt, rate=0.05)
per = {}
for T in buckets:
    i = jnp.ones((4, 8, T), jnp.int32)
    l = jnp.where(jnp.arange(T) < T - 4, i, -100)
    m = jnp.ones((4, 8, T), jnp.int32)
    tc = time.time()
    lora, state, loss = step(params, lora, state, Qs, i, l, m, jax.random.PRNGKey(0), 2e-4)
    loss.block_until_ready()
    comp = time.time() - tc
    tc = time.time()
    for s in range(10):
        lora, state, loss = step(params, lora, state, Qs, i, l, m, jax.random.PRNGKey(s), 2e-4)
    loss.block_until_ready()
    per[T] = (time.time() - tc) / 10
    say(f"T3 step T={T}: {per[T]:.3f} s/step (first call incl. compile {comp:.0f}s)")
ep = sum(mix[b] * per[b] for b in buckets)
say(f"T3 expected epoch time {ep:.0f}s, 12 epochs {12 * ep / 60:.1f} min "
    f"(torch_tpu: ~1.9 s/step at T=104, {len(steps_max)} steps/epoch -> {12 * len(steps_max) * 1.9 / 60:.1f} min)")
ms = jax.devices()[0].memory_stats() or {}
say("HBM peak GB", round(ms.get("peak_bytes_in_use", 0) / 2**30, 2), "of", round(ms.get("bytes_limit", 0) / 2**30, 2))
say("BENCH DONE")
