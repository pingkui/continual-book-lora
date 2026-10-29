"""Qwen2 / Qwen2.5 decoder in plain JAX: weights from the HF safetensors, layers stacked and run with lax.scan
(one compiled layer instead of 28), optional LoRA on the seven linear kinds.

Numerics follow HF's eager path: RMSNorm in fp32 then cast back, softmax in fp32, logits upcast to fp32 for the loss.
Linear weights are stored as [in, out] (HF stores [out, in]) so a layer is x @ W.
LoRA follows peft: y += dropout(x) @ a @ b * (alpha / r), a = lora_A.T [in, r], b = lora_B.T [r, out]."""
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

KINDS = ("q", "k", "v", "o", "gate", "up", "down")
HF_NAME = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
           "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


def model_dir(name):
    """Local snapshot directory of a HF model id (downloads it if missing)."""
    if os.path.isdir(name):
        return name
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{name.replace('/', '--')}/snapshots/*/"))
    if hits:
        return hits[0]
    from huggingface_hub import snapshot_download
    return snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


def load_config(path):
    c = json.load(open(os.path.join(path, "config.json")))
    return {"L": c["num_hidden_layers"], "D": c["hidden_size"], "F": c["intermediate_size"],
            "H": c["num_attention_heads"], "KV": c["num_key_value_heads"], "V": c["vocab_size"],
            "hd": c["hidden_size"] // c["num_attention_heads"], "eps": c["rms_norm_eps"],
            "theta": c.get("rope_theta", 1e6), "tied": c.get("tie_word_embeddings", False)}


def load_params(path, dtype=jnp.bfloat16):
    """HF safetensors -> {'embed', 'norm', 'lm_head' (None if tied), 'layers': {name: [L, ...]}} on the default device."""
    from safetensors import safe_open
    cfg = load_config(path)
    raw = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="flax") as st:
            for k in st.keys():
                raw[k] = st.get_tensor(k)
    cast = lambda x: jnp.asarray(x, dtype)
    layers = {}
    for kind, hf in HF_NAME.items():
        layers[f"{kind}_w"] = cast(jnp.stack([raw[f"model.layers.{i}.{hf}.weight"].T for i in range(cfg["L"])]))
        if kind in ("q", "k", "v"):
            layers[f"{kind}_b"] = cast(jnp.stack([raw[f"model.layers.{i}.{hf}.bias"] for i in range(cfg["L"])]))
    layers["ln1"] = cast(jnp.stack([raw[f"model.layers.{i}.input_layernorm.weight"] for i in range(cfg["L"])]))
    layers["ln2"] = cast(jnp.stack([raw[f"model.layers.{i}.post_attention_layernorm.weight"] for i in range(cfg["L"])]))
    params = {"embed": cast(raw["model.embed_tokens.weight"]), "norm": cast(raw["model.norm.weight"]), "layers": layers,
              "lm_head": None if cfg["tied"] else cast(raw["lm_head.weight"].T)}
    return cfg, params


def rms_norm(x, w, eps):
    xf = x.astype(jnp.float32)
    xf = xf * jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + eps)
    return w * xf.astype(x.dtype)


def rope(x, pos, theta):
    """x [B, T, n, hd], pos [B, T]; HF rotate_half convention, angles in fp32."""
    hd = x.shape[-1]
    inv = 1.0 / (theta ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))
    ang = pos[..., None].astype(jnp.float32) * inv  # [B, T, hd/2]
    ang = jnp.concatenate([ang, ang], -1)[:, :, None, :]
    cos, sin = jnp.cos(ang).astype(x.dtype), jnp.sin(ang).astype(x.dtype)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return x * cos + jnp.concatenate([-x2, x1], -1) * sin


def positions(mask):
    """Like HF generate: cumulative count of real tokens - 1, pads at 1 (any value works, they are masked)."""
    p = jnp.cumsum(mask, -1) - 1
    return jnp.where(mask > 0, p, 1)


def linear(x, lw, kind, lora, key, rate, scale):
    y = x @ lw[f"{kind}_w"]
    if f"{kind}_b" in lw:
        y = y + lw[f"{kind}_b"]
    if f"{kind}_fa" in lw:  # earlier books' adapters kept unmerged (scale folded into _fb): added in fp32, so the
        # model-dtype weights are never re-cast and the earlier deltas are not re-rounded at every merge
        y = (y.astype(jnp.float32) + (x.astype(jnp.float32) @ lw[f"{kind}_fa"]) @ lw[f"{kind}_fb"]).astype(y.dtype)
    if lora is not None and kind in lora:
        xl = x.astype(jnp.float32)
        if key is not None and rate > 0:
            keep = jax.random.bernoulli(jax.random.fold_in(key, KINDS.index(kind)), 1 - rate, xl.shape)
            xl = jnp.where(keep, xl / (1 - rate), 0)
        y = y + ((xl @ lora[kind]["a"]) @ lora[kind]["b"] * scale).astype(y.dtype)
    return y


def forward(cfg, params, ids, mask, lora=None, lora_scale=2.0, key=None, rate=0.0, capture=False):
    """ids, mask [B, T] -> logits [B, T, V] (fp32). lora: {kind: {'a': [L, in, r], 'b': [L, r, out]}} or None.
    capture=True also returns the inputs of each linear kind per layer: {'attn_in', 'o_in', 'mlp_in', 'down_in'}
    with shape [L, B, T, d] (q/k/v share attn_in, gate/up share mlp_in)."""
    B, T = ids.shape
    H, KV, hd, eps = cfg["H"], cfg["KV"], cfg["hd"], cfg["eps"]
    pos = positions(mask)
    causal = jnp.tril(jnp.ones((T, T), bool))
    allowed = causal[None, None] & (mask[:, None, None, :] > 0)  # [B, 1, T, T]
    x = params["embed"][ids]
    keys = jax.random.split(key, cfg["L"]) if key is not None else None

    def layer(h, inp):
        lw, ll, k = inp
        a_in = rms_norm(h, lw["ln1"], eps)
        q = linear(a_in, lw, "q", ll, k, rate, lora_scale).reshape(B, T, H, hd)
        kk = linear(a_in, lw, "k", ll, k, rate, lora_scale).reshape(B, T, KV, hd)
        v = linear(a_in, lw, "v", ll, k, rate, lora_scale).reshape(B, T, KV, hd)
        q, kk = rope(q, pos, cfg["theta"]), rope(kk, pos, cfg["theta"])
        q = q.reshape(B, T, KV, H // KV, hd)
        s = jnp.einsum("btkgd,bskd->bkgts", q, kk).astype(jnp.float32) / np.sqrt(hd)
        s = jnp.where(allowed[:, :, None], s, jnp.finfo(jnp.float32).min)
        p = jax.nn.softmax(s, -1).astype(v.dtype)
        o_in = jnp.einsum("bkgts,bskd->btkgd", p, v).reshape(B, T, H * hd)
        o_out = linear(o_in, lw, "o", ll, k, rate, lora_scale)  # what attention writes into the residual stream
        h = h + o_out
        m_in = rms_norm(h, lw["ln2"], eps)
        d_in = jax.nn.silu(linear(m_in, lw, "gate", ll, k, rate, lora_scale)) * linear(m_in, lw, "up", ll, k, rate, lora_scale)
        d_out = linear(d_in, lw, "down", ll, k, rate, lora_scale)  # what the MLP writes into the residual stream
        h = h + d_out
        return h, ({"attn_in": a_in, "o_in": o_in, "mlp_in": m_in, "down_in": d_in, "o_out": o_out, "down_out": d_out}
                   if capture else None)

    # round 28: REMAT=1 recomputes each layer's activations in the backward pass (jax.checkpoint on the scan body), so the
    # training step's temporary memory is about one layer's worth instead of all 28; same math, ~30% more compute.
    body = jax.checkpoint(layer) if os.environ.get("REMAT") == "1" and not capture else layer
    x, cap = jax.lax.scan(body, x, (params["layers"], lora, keys))
    x = rms_norm(x, params["norm"], eps)
    head = params["embed"].T if params["lm_head"] is None else params["lm_head"]
    # fp32 output: bf16 logits are spaced 0.125 apart near 20, coarser than the fp16 GPU runs, and decide close calls
    logits = jnp.matmul(x, head, preferred_element_type=jnp.float32)
    return (logits, cap) if capture else logits


def lm_loss(logits, labels):
    """HF causal-LM loss: predict token t+1 from t, ignore -100, mean over the counted tokens, in fp32."""
    lg = logits[:, :-1].astype(jnp.float32)
    lab = labels[:, 1:]
    valid = lab != -100
    logp = jax.nn.log_softmax(lg, -1)
    nll = -jnp.take_along_axis(logp, jnp.where(valid, lab, 0)[..., None], -1)[..., 0]
    return jnp.sum(nll * valid) / jnp.maximum(jnp.sum(valid), 1)
