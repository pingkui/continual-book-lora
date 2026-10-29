"""LoRA training step in JAX, matching lab/train_cl.py (peft + torch AdamW) step for step:
- LoRA r=16, alpha=32, dropout on the LoRA input, A ~ U(-1/sqrt(in), 1/sqrt(in)) (peft's kaiming_uniform(a=sqrt(5))), B = 0;
- one optimizer step = `ga` micro-batches; each micro-batch loss is a mean over its own tokens and the step uses
  sum(micro loss) / ga, also for the short last step of an epoch (dummy micro-batches carry no labels, loss 0);
- global-norm clip 1.0, AdamW (0.9, 0.999, eps 1e-8, no weight decay), lr = lr0 * min(1, (s+1)/20) * max(0.05, 1 - s/total);
- after each step A is projected off the protected input subspace: a -= Q (Q^T a), Q [L, in, k] (zero columns pad k).
The learning rate is an argument of the step (not baked into the optimizer), so one compiled step serves every book
and seed with the same shapes. Projection runs at full fp32 precision (TPU matmuls default to bf16 passes)."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

import qwen


def lora_dims(cfg):
    D, F, q, kv = cfg["D"], cfg["F"], cfg["H"] * cfg["hd"], cfg["KV"] * cfg["hd"]
    return {"q": (D, q), "k": (D, kv), "v": (D, kv), "o": (q, D), "gate": (D, F), "up": (D, F), "down": (F, D)}


def init_lora(cfg, key, r=16, kinds=qwen.KINDS):
    lora = {}
    for i, k in enumerate(kinds):
        fin, fout = lora_dims(cfg)[k]
        b = 1 / np.sqrt(fin)
        lora[k] = {"a": jax.random.uniform(jax.random.fold_in(key, i), (cfg["L"], fin, r), jnp.float32, -b, b),
                   "b": jnp.zeros((cfg["L"], r, fout), jnp.float32)}
    return lora


HI = jax.lax.Precision.HIGHEST


def lr_at(step, lr, total):
    """lab/train_cl.py's schedule: 20 warmup steps, then linear decay to 5%."""
    return lr * min(1.0, (step + 1) / 20) * max(0.05, 1 - step / total)


def make_opt():
    """Clip + AdamW with unit learning rate; the step multiplies the update by lr (same result, weight decay is 0)."""
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1.0, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0))


def _off(a, Q):
    """a - Q (Q^T a) for a [L, in, r] fp32 and Q [L, in, k]. fp32 Q: one pass at HIGHEST precision. bf16 Q (round 28,
    half the device memory of the bases): bf16 x bf16 GEMMs with fp32 accumulation, no fp32 copy of Q is ever
    materialised; a bf16 Q is orthonormal only to ~4e-3, so the pass is applied twice (residual ~1e-5)."""
    if Q.dtype in (jnp.bfloat16, jnp.float16):
        # The leakage floor is set by Q's own rounding (|delta^T a| ~ eps |a|, eps = 2^-8 bf16, 2^-11 fp16), not by
        # the number of passes; measured 6e-4 with bf16 on the 0.5B. Two passes only remove the idempotence error.
        # Layer by layer (lax.map): the batched einsum made XLA materialise a transposed copy of the whole basis for
        # its GEMM (7.1 GiB for the 7B's down_proj), which is what the memory budget is trying to avoid.
        def one(args):
            q, al = args  # q [in, k] half, al [in, r] fp32
            for _ in range(2):
                qa = jnp.einsum("ik,ir->kr", q, al.astype(q.dtype), preferred_element_type=jnp.float32)
                al = al - jnp.einsum("ik,kr->ir", q, qa.astype(q.dtype), preferred_element_type=jnp.float32)
            return al
        return jax.lax.map(one, (Q, a))
    qa = jnp.einsum("lik,lir->lkr", Q, a, precision=HI)
    return a - jnp.einsum("lik,lkr->lir", Q, qa, precision=HI)


def project(lora, Qs):
    """Qs: {kind: [L, in, k]} or empty: A loses its part along the protected INPUT directions (the update reads
    nothing from them). Round 24 (OUT_PROJ): a key "<kind>_out" [L, out, m] holds protected OUTPUT directions of
    that kind, and B loses its part along them (the update writes nothing into them), so what leaks through the
    unprotected input share (1 - tau) cannot land where later layers read."""
    out = {}
    for k, d in lora.items():
        if k in Qs:
            d = {"a": _off(d["a"], Qs[k]), "b": d["b"]}
        if f"{k}_out" in Qs:
            Qo = Qs[f"{k}_out"]
            qb = jnp.einsum("lro,lom->lrm", d["b"], Qo, precision=HI)
            d = {"a": d["a"], "b": d["b"] - jnp.einsum("lrm,lom->lro", qb, Qo, precision=HI)}
        out[k] = d
    return out


def project_onto(lora, Ps):
    """Ps: {kind: [L, in, m]} or empty. Keeps only the part of A inside span(P): a = P (P^T a) (zero columns of the
    padding drop out). Kinds without P are left free."""
    out = {}
    for k, d in lora.items():
        if k in Ps:
            P = Ps[k]
            pa = jnp.einsum("lim,lir->lmr", P, d["a"], precision=HI)
            d = {"a": jnp.einsum("lim,lmr->lir", P, pa, precision=HI), "b": d["b"]}
        out[k] = d
    return out


def make_step(cfg, opt, alpha=32, r=16, rate=0.05, onto=False):
    """Returns step(params, lora, opt_state, Qs, ids, labels, mask, key, lr, keep=None) -> (lora, opt_state,
    last_micro_loss); ids/labels/mask are [ga, micro, T]. onto=True: Qs holds bases P to train INSIDE (project_onto)
    instead of protected bases to stay out of. keep: optional {kind: [L, 1, 1]} 0/1 per layer; the gradient
    of a kind's LoRA in a layer with 0 is zeroed before clipping and Adam, so that adapter stays at its init (b = 0).
    Jitted; one compile per distinct T and Q shape. The base weights are an argument, not a closure: closed-over
    arrays would be baked into the program as constants (a second full copy)."""
    scale = alpha / r

    def micro_loss(lora, params, ids, labels, mask, key):
        logits = qwen.forward(cfg, params, ids, mask, lora=lora, lora_scale=scale, key=key, rate=rate)
        return qwen.lm_loss(logits, labels)

    grad_fn = jax.value_and_grad(micro_loss)

    @jax.jit
    def step(params, lora, opt_state, Qs, ids, labels, mask, key, lr, keep=None):
        ga = ids.shape[0]

        def body(acc, inp):
            i, l, m, k = inp
            loss, g = grad_fn(lora, params, i, l, m, k)
            return jax.tree.map(jnp.add, acc, g), loss

        zeros = jax.tree.map(jnp.zeros_like, lora)
        g, losses = jax.lax.scan(body, zeros, (ids, labels, mask, jax.random.split(key, ga)))
        g = jax.tree.map(lambda x: x / ga, g)
        if keep is not None:
            g = {k: {n: x * keep[k] for n, x in d.items()} if k in keep else d for k, d in g.items()}
        upd, opt_state = opt.update(g, opt_state, lora)
        lora = (project_onto if onto else project)(optax.apply_updates(lora, jax.tree.map(lambda u: u * lr, upd)), Qs)
        real = jnp.any(labels != -100, axis=(1, 2))
        last = jnp.max(jnp.where(real, jnp.arange(ga), -1))
        return lora, opt_state, losses[last]

    return step
