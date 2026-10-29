"""Greedy generation with a KV cache, matching HF `generate(do_sample=False)` as lab/train_cl.py calls it:
- prompts left-padded to a fixed length P, positions = cumsum(mask) - 1 (as HF derives them from the mask);
- the model's generation_config still applies under greedy decoding: repetition_penalty (1.1 for Qwen2.5) on every
  token id present in the input so far, pads included (score < 0 -> score * p, else score / p), on fp32 logits;
- a row that emitted an eos id gets pad ids afterwards; all N steps always run (fixed shapes).
One compile per (batch, P, N). No LoRA here: evaluation runs on merged weights."""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

import qwen


def gen_config(path):
    g = json.load(open(os.path.join(path, "generation_config.json")))
    eos = g.get("eos_token_id")
    return {"eos": tuple(eos if isinstance(eos, list) else [eos]), "pad": g.get("pad_token_id", eos[0]),
            "rep": float(g.get("repetition_penalty", 1.0))}


def _layer_cached(cfg, h, lw, ck, cv, pos, start, kmask):
    """h [B, T, D] new tokens at cache slots start..start+T-1; ck/cv [B, S, KV, hd]; kmask [B, S] (real keys)."""
    B, T, _ = h.shape
    H, KV, hd, eps = cfg["H"], cfg["KV"], cfg["hd"], cfg["eps"]
    S = ck.shape[1]
    a_in = qwen.rms_norm(h, lw["ln1"], eps)
    q = qwen.linear(a_in, lw, "q", None, None, 0.0, 0.0).reshape(B, T, H, hd)
    k = qwen.linear(a_in, lw, "k", None, None, 0.0, 0.0).reshape(B, T, KV, hd)
    v = qwen.linear(a_in, lw, "v", None, None, 0.0, 0.0).reshape(B, T, KV, hd)
    q, k = qwen.rope(q, pos, cfg["theta"]), qwen.rope(k, pos, cfg["theta"])
    ck = jax.lax.dynamic_update_slice(ck, k.astype(ck.dtype), (0, start, 0, 0))
    cv = jax.lax.dynamic_update_slice(cv, v.astype(cv.dtype), (0, start, 0, 0))
    qi = start + jnp.arange(T)
    allowed = (jnp.arange(S)[None, :] <= qi[:, None])[None, None] & (kmask[:, None, None, :] > 0)  # [B, 1, T, S]
    q = q.reshape(B, T, KV, H // KV, hd)
    s = jnp.einsum("btkgd,bskd->bkgts", q, ck).astype(jnp.float32) / np.sqrt(hd)
    s = jnp.where(allowed[:, :, None], s, jnp.finfo(jnp.float32).min)
    p = jax.nn.softmax(s, -1).astype(cv.dtype)
    o_in = jnp.einsum("bkgts,bskd->btkgd", p, cv).reshape(B, T, H * hd)
    h = h + qwen.linear(o_in, lw, "o", None, None, 0.0, 0.0)
    m_in = qwen.rms_norm(h, lw["ln2"], eps)
    d_in = jax.nn.silu(qwen.linear(m_in, lw, "gate", None, None, 0.0, 0.0)) * qwen.linear(m_in, lw, "up", None, None, 0.0, 0.0)
    return h + qwen.linear(d_in, lw, "down", None, None, 0.0, 0.0), ck, cv


def _step(cfg, params, ids, pos, start, kmask, cache):
    """Run tokens ids [B, T] through all layers, writing the cache; returns fp32 logits of the last token."""
    x = params["embed"][ids]

    def body(h, inp):
        lw, ck, cv = inp
        h, ck, cv = _layer_cached(cfg, h, lw, ck, cv, pos, start, kmask)
        return h, (ck, cv)

    x, cache = jax.lax.scan(body, x, (params["layers"], cache[0], cache[1]))
    x = qwen.rms_norm(x[:, -1], params["norm"], cfg["eps"])
    head = params["embed"].T if params["lm_head"] is None else params["lm_head"]
    return jnp.matmul(x, head, preferred_element_type=jnp.float32), cache  # fp32 output, see qwen.forward


def _penalise(logits, seen, rep):
    pen = jnp.where(logits < 0, logits * rep, logits / rep)
    return jnp.where(seen, pen, logits)


def make_generate(cfg, gcfg, n_new):
    """Returns gen(params, ids [B, P], mask [B, P]) -> new token ids [B, n_new] (int32), jitted."""
    eos = jnp.array(gcfg["eos"], jnp.int32)
    pad, rep = gcfg["pad"], gcfg["rep"]

    @jax.jit
    def gen(params, ids, mask):
        B, P = ids.shape
        S = P + n_new
        dt = params["embed"].dtype
        cache = (jnp.zeros((cfg["L"], B, S, cfg["KV"], cfg["hd"]), dt),) * 2
        kmask = jnp.concatenate([mask, jnp.zeros((B, n_new), mask.dtype)], 1)
        seen = jnp.zeros((B, cfg["V"]), bool).at[jnp.arange(B)[:, None], ids].set(True)
        logits, cache = _step(cfg, params, ids, qwen.positions(mask), 0, kmask, cache)
        nreal = jnp.sum(mask, 1)

        def pick(logits, seen, done):
            tok = jnp.argmax(_penalise(logits, seen, rep), -1).astype(jnp.int32)
            tok = jnp.where(done, pad, tok)
            return tok, done | jnp.isin(tok, eos)

        tok, done = pick(logits, seen, jnp.zeros((B,), bool))
        out = jnp.zeros((B, n_new), jnp.int32).at[:, 0].set(tok)

        def body(t, carry):
            out, tok, seen, done, kmask, cache = carry
            seen = seen.at[jnp.arange(B), tok].set(True)
            kmask = kmask.at[:, P + t].set(1)
            logits, cache = _step(cfg, params, tok[:, None], (nreal + t)[:, None], P + t, kmask, cache)
            tok, done = pick(logits, seen, done)
            return out.at[:, t + 1].set(tok), tok, seen, done, kmask, cache

        out = jax.lax.fori_loop(0, n_new - 1, body, (out, tok, seen, done, kmask, cache))[0]
        return out

    return gen
