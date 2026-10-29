"""Continual LoRA with null-space projection in JAX: the TPU counterpart of lab/train_cl.py, same runs, same
evaluation, same results.json layout (cfg / base / base_gen_nll / runs[name].stages[...].books[...]).

Per book: fresh LoRA (projected off the protected subspaces once, then after every step), train, merge the LoRA delta
into fp32 master weights on the host and cast to the model dtype, evaluate every book so far (exam, exam_gen,
training-wording probes, general-text NLL), then record this book's layer-input subspace.
Every fp32 matmul outside the model (projection, merge, Gram/eigh basis, leakage) runs at HIGHEST precision: on a
TPU the default precision would use bf16 passes and leave ~1e-3 of the update inside the protected subspace.
One compiled training step is shared by all books and seeds (lr is an argument; Q is padded to fixed widths).
Subspace: tokens sampled with p = n_feat / total tokens, independently per linear kind and layer (as the GPU hooks
do); Gram eigh on the device in fp32 (CPU float64 if it fails), keep a tau share of energy, orthogonalise against the
stored basis, keep singular values > 0.1.

Env: RUNS=A4_gen_s1[,..] BOOKS=2,3,4 MODEL SRC (data + factlab.py) OUT BASE=1 (score the untouched model)
quick checks: ROWS EPOCHS EXAMN NFEAT GEN_N; on this 12 GB host also DTYPE=float32 MICRO=2 GEN_BS=8 (the
defaults 8 / 48 are what the GPU runs and the TPU use).
Diagnostics (see the DIAG section): DIAG=1 measures where book 2 leaks while book 3 is learned (BOOKS=2,3);
DIAG2=1 also branches book 3 into VARIANTS (base, no_down12, normsub) from the same post-book-2 state.
BRANCH=1 (BOOKS=2,3) trains book 3 once per BRANCHES entry (t99, t995, t999: tau of book 2's subspace)."""
import functools
import json
import math
import os
import pickle
import random
import re
import sys
import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"  # keep torch off the TPU (Colab's torch_tpu would grab it)
import torch  # host linear algebra only (MKL)

import generate
import qwen
import train

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
SRC = os.environ.get("SRC", "/content/data")
sys.path.insert(0, SRC)
import factlab as fl

TAGS = ["text", "qa", "premise"]
TAGS_F = ["text", "qa", "premise_f"]
TAGS_G = ["text", "qa", "text_g", "qa_g", "premise_g"]
RUNS = {r["name"]: r for r in [  # as in lab/train_cl.py
    {"name": "A3_frames", "tags": TAGS_F, "epochs": 12, "tau": 0.99, "seed": 0},
    {"name": "A4_gen", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "seed": 0},
    {"name": "A4_gen_s1", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "seed": 1},
    {"name": "A4_gen_s2", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "seed": 2},
    {"name": "merge_ctrl", "tags": TAGS, "epochs": 15, "tau": None, "seed": 0},
    # round 9: A4 plus general-text protection (the 300 Wikipedia training rows) at its own, lower threshold.
    # The GPU script keeps these bases in fp16 to fit the T4; here they stay fp32.
    {"name": "A4_gen90", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.90, "seed": 0},
    {"name": "A4_gen70", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "seed": 0},
    # round 10: subspaces from unit-length tokens (norm; the diag2 run: book-2 retention after book 3 0.754 -> 0.866),
    # and the same with every book's adapter kept unmerged in fp32 instead of merged into the bf16 weights (unmerged)
    {"name": "A4_gen70n", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "seed": 0, "norm": True},
    {"name": "A4_gen70nu", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "seed": 0, "norm": True,
     "unmerged": True},
    {"name": "A4_gen70n_s1", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "seed": 1, "norm": True},
    # round 12: books after the first learned with a third of the learning rate for 18 epochs (lr branches: book-2
    # retention after book 3 0.866 -> 0.894 with book 3 still fully learned)
    {"name": "A4_v12", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "seed": 0, "norm": True,
     "lr_later": 1 / 3, "epochs_later": 18},
    {"name": "A4_v12_s1", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "seed": 1, "norm": True,
     "lr_later": 1 / 3, "epochs_later": 18},
    # round 16: A4_gen70n plus the self-instruct subspace (instruction following and reasoning of the base model)
    {"name": "A4_chat80", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "tau_chat": 0.80, "seed": 0,
     "norm": True},
    {"name": "A4_chat90", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "tau_chat": 0.90, "seed": 0,
     "norm": True},
    # the self-instruct subspace only while the first book is learned (A4_chat80's book 3 diverged once the book-2
    # subspace came on top: mean protected dim 996 vs 724); later books see the general text and the books only
    {"name": "A4_chat80f", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "tau_chat": 0.80,
     "chat_first": True, "seed": 0, "norm": True},
    # round 16d: a weaker first-book protection (L2 twins went to 3/6 at 0.80), and 0.80 from twice the self-instruct
    # rows (eight kinds of task instead of four; run with CHAT_N=640 and a fresh CHAT_FILE)
    {"name": "A4_chat70f", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "tau_chat": 0.70,
     "chat_first": True, "seed": 0, "norm": True},
    # round 22: a stronger first-book self-instruct protection (the 320 rows already hold 80 math word problems and 80
    # multiple-choice questions; at 0.80 GSM8K still lost 2.7-10.5 points), with data_v17 and 6 epochs for book 1
    {"name": "A4_chat90f", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "tau_chat": 0.90,
     "chat_first": True, "seed": 0, "norm": True},
    # round 25: the same with seed 1 (the client's L6: two runs per book within 9 of 150; also the GSM8K run-to-run noise)
    {"name": "A4_chat90f_s1", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "tau_chat": 0.90,
     "chat_first": True, "seed": 1, "norm": True},
    {"name": "A4_chat80f_n640", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.70, "tau_chat": 0.80,
     "chat_first": True, "seed": 0, "norm": True},
    # round 30 (L5 lever): stronger general-text protection (0.70 -> 0.85), run with CHAT_N=640 and a fresh CHAT_FILE
    {"name": "A4_gen85chat90f", "tags": TAGS_G, "epochs": 12, "tau": 0.99, "tau_gen": 0.85, "tau_chat": 0.90,
     "chat_first": True, "seed": 0, "norm": True},
]}
CFG = {"model": os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
       "books": [int(b) for b in os.environ.get("BOOKS", "2,3,4").split(",")],
       "bs": 32, "micro": int(os.environ.get("MICRO", "8")), "lr": float(os.environ.get("LR", "2e-4")), "dropout": 0.05, "max_new": int(os.environ.get("MAX_NEW", "40")),
       # round 20: RANK (default 16) with alpha = 2 x rank unless ALPHA is given, so the LoRA scale alpha/rank stays 2
       # and the learning rate keeps its meaning; a book's saved factors carry their scale (load_lora checks it)
       "rank": int(os.environ.get("RANK", "16")), "alpha": int(os.environ.get("ALPHA", str(2 * int(os.environ.get("RANK", "16"))))),
       "n_feat": int(os.environ.get("NFEAT", "4096")), "buckets": (64, 96, 128, 160), "prompt_len": 80, "gen_bs": int(os.environ.get("GEN_BS", "48")),
       "runs": [RUNS[n] for n in os.environ.get("RUNS", "A4_gen_s1").split(",")]}
ROWS, EXAMN, GEN_N = (int(os.environ.get(k, "0")) for k in ("ROWS", "EXAMN", "GEN_N"))
DTYPE = jnp.float32 if os.environ.get("DTYPE") == "float32" else jnp.bfloat16
DIAG_STEPS = bool(os.environ.get("DIAG_STEPS"))  # log loss and LoRA sizes every 10 steps of a book's first 100
OUT = os.environ.get("OUT", "/content/results.json")
Q_DTYPE = {"bf16": jnp.bfloat16, "bfloat16": jnp.bfloat16, "fp16": jnp.float16, "float16": jnp.float16}.get(os.environ.get("Q_DTYPE", "fp32"), jnp.float32)  # round 28: half-precision bases; fp16 leaks ~8x less than bf16 (11 vs 8 mantissa bits)
log("devices", jax.devices(), "| runs", [r["name"] for r in CFG["runs"]], "books", CFG["books"], "model", CFG["model"], "Q_DTYPE", jnp.dtype(Q_DTYPE).name,
    "rank", CFG["rank"], "alpha", CFG["alpha"])

from transformers import AutoTokenizer

path = qwen.model_dir(CFG["model"])
tok = AutoTokenizer.from_pretrained(path)
cfg, params = qwen.load_params(path, DTYPE)
gcfg = generate.gen_config(path)
gen = generate.make_generate(cfg, gcfg, CFG["max_new"])
LORA_KINDS = qwen.KINDS
INPUT_OF = {"q": "attn_in", "k": "attn_in", "v": "attn_in", "o": "o_in", "gate": "mlp_in", "up": "mlp_in",
            "down": "down_in"}
# round 24, candidate B: OUT_PROJ=1 also protects the OUTPUT directions of the two kinds that write into the residual
# stream (o, down): their bases live under the keys "o_out"/"down_out" of the same dicts and train.project removes B's
# part along them. PROT_KEYS lists every key a protected-basis dict holds; CAP_OF names its capture tensor.
OUT_PROJ = bool(os.environ.get("OUT_PROJ"))
OUT_KINDS = {"o": "o_out", "down": "down_out"} if OUT_PROJ else {}
PROT_KEYS = list(qwen.KINDS) + list(OUT_KINDS.values())
CAP_OF = {**INPUT_OF, **{v: v for v in OUT_KINDS.values()}}
log("model loaded", {k: cfg[k] for k in ("L", "D", "F", "H", "KV", "V")}, "generation", gcfg)


def load(name, k):
    return json.load(open(f"{SRC}/book{k}.{name}.json"))


def chat_prompt(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


def encode(row):
    if row["type"] == "text":
        ids = tok(row["text"] + tok.eos_token)["input_ids"]
        return ids, ids[:]
    p = tok(chat_prompt(row["user"]))["input_ids"]
    a = tok(row["assistant"] + "<|im_end|>")["input_ids"]
    return p + a, [-100] * len(p) + a


def pad_batch(enc, n, T):
    """enc: list of (ids, labels), padded with dummy rows to n rows of length T (dummy rows: no labels, mask 0)."""
    real = len(enc)
    enc = enc + [([tok.pad_token_id], [-100])] * (n - real)
    ids = np.array([x + [tok.pad_token_id] * (T - len(x)) for x, _ in enc], np.int32)
    lab = np.array([y + [-100] * (T - len(y)) for _, y in enc], np.int32)
    att = np.array([[1] * len(x) + [0] * (T - len(x)) for x, _ in enc], np.int32)
    att[real:] = 0  # dummy rows: no attention, no labels, never sampled into a subspace
    return ids, lab, att


def bucket(n):
    for b in CFG["buckets"]:
        if n <= b:
            return b
    raise ValueError(f"row of {n} tokens exceeds the largest bucket {CFG['buckets'][-1]}")


def micro_batches(enc, rng):
    """Same order as lab/train_cl.py: shuffle row indices with rng, consecutive groups of `micro`."""
    order = list(range(len(enc)))
    rng.shuffle(order)
    return [[enc[j] for j in order[i:i + CFG["micro"]]] for i in range(0, len(order), CFG["micro"])]


OPT = train.make_opt()
STEP = train.make_step(cfg, OPT, CFG["alpha"], CFG["rank"], CFG["dropout"])
STEP_ONTO = train.make_step(cfg, OPT, CFG["alpha"], CFG["rank"], CFG["dropout"], onto=True)  # compiled on first use
HI = jax.lax.Precision.HIGHEST


def leak_out(Qs, lora):
    """OUT_PROJ: |B Qo| / |B| per layer for the kinds with protected output directions (expected ~1e-7, like the
    input leakage)."""
    v = [float(jnp.linalg.norm(jnp.einsum("ro,om->rm", lora[k]["b"][l], Qs[o][l], precision=HI)) /
               jnp.linalg.norm(lora[k]["b"][l])) for k, o in OUT_KINDS.items() if o in Qs for l in range(cfg["L"])]
    return f"; output leakage |B Qo|/|B|: mean {sum(v) / len(v):.2e}, max {max(v):.2e}" if v else ""


def train_book(lora, Qs, enc, epochs, seed, key, keep=None, step=None, lr_mult=1.0, stop_fn=None, replay=None,
               ckpt=None):
    """stop_fn(lora, epoch) -> bool, called after every epoch: True ends training early (the lr schedule stays the one
    planned for all epochs). replay (Spaced): its due rows of earlier books join each epoch and it is called after each
    epoch; the lr schedule still advances one book-epoch per epoch, over however many steps the epoch has.
    ckpt (a .npz path): written after every epoch but the last (LoRA, optimizer state, step, the shuffle rng and the
    review's boxes) and removed when the book is done; if it exists at the start, training resumes from it exactly."""
    ga = CFG["bs"] // CFG["micro"]
    per_ep = math.ceil(len(enc) / CFG["bs"])
    total = epochs * per_ep
    state = OPT.init(lora)
    rng, s, ep0 = random.Random(seed), 0, 0
    if ckpt and os.path.exists(ckpt):
        lora, state, s, ep0, extra = load_ckpt(ckpt, state)
        if "rng" in extra:
            rng.setstate(extra["rng"])
        else:  # round-20 book-1 checkpoints (no review): replay the finished epochs' shuffles
            for _ in range(ep0):
                micro_batches(enc, rng)
        if replay is not None and extra.get("spaced"):
            replay.restore(extra["spaced"])
        log(f"  resumed from {ckpt}: {ep0}/{epochs} epochs done, step {s}/{total}"
            + (f", review at epoch {replay.g}" if replay is not None else ""))
    out_of_time = False
    # round 30 (L5 variance): EMA of the factors over the last EMA_EPOCHS epochs, used as the book's result; a standard
    # variance reducer at no extra compute (seed-to-seed GSM8K swings of 6.7 points with identical rows). ema_n counts the
    # averaged steps; decay = 1 - 1/ema_n gives a running mean over the window.
    ema, ema_n = None, 0
    for ep in range(ep0, epochs):
        mbs = micro_batches(enc + (replay.rows_for_epoch() if replay else []), rng)
        n_ep = math.ceil(len(mbs) / ga)  # == per_ep without replay rows
        for i in range(0, len(mbs), ga):
            if DEADLINE and s % 20 == 0 and time.time() > DEADLINE:  # budget guard (package README, "GPU hours")
                log(f"  budget guard: deadline {DEADLINE_STR} reached at step {s}/{total} (epoch {ep + 1}/{epochs}); "
                    f"training ends here, the book is finished with the factors so far")
                out_of_time = True
                break
            group = mbs[i:i + ga]
            T = bucket(max(len(x) for mb in group for x, _ in mb))
            arr = [pad_batch(mb, CFG["micro"], T) for mb in group]
            arr += [pad_batch([], CFG["micro"], T)] * (ga - len(arr))  # short last step: empty micro-batches
            ids, lab, att = (jnp.asarray(np.stack([a[j] for a in arr])) for j in range(3))
            sv = ep * per_ep + (i // ga) * per_ep // n_ep  # schedule position (== s without replay rows)
            lora, state, loss = (step or STEP)(params, lora, state, Qs, ids, lab, att, jax.random.fold_in(key, s),
                                     train.lr_at(sv, CFG["lr"] * lr_mult, total), keep)
            s += 1
            if DIAG_STEPS and s <= 100 and s % 10 == 0:
                norms = {kd: (float(jnp.abs(lora[kd]["a"]).max()), float(jnp.abs(lora[kd]["b"]).max())) for kd in lora}
                log(f"  diag step {s}: loss {float(loss):.4f}; max |a|, |b| per kind " +
                    " ".join(f"{kd} {a:.3g}/{b:.3g}" for kd, (a, b) in norms.items()))
            if (s == 1 or s % 100 == 0) and not math.isfinite(float(loss)):
                if DIAG_STEPS:
                    log("  diag at the failure: " + " ".join(
                        f"{kd} {float(jnp.abs(lora[kd]['a']).max()):.3g}/{float(jnp.abs(lora[kd]['b']).max()):.3g}"
                        for kd in lora))
                raise FloatingPointError(f"training loss {float(loss)} at step {s}")
            if s % 100 == 0 or s == total:
                log(f"  step {s}/{total} loss {float(loss):.3f}" if replay is None else
                    f"  step {s} (schedule {sv + 1}/{total}) loss {float(loss):.3f}")
            if EMA_EPOCHS and ep >= epochs - EMA_EPOCHS:
                ema_n += 1
                ema = lora if ema is None else jax.tree_util.tree_map(lambda e, x: e + (x - e) / ema_n, ema, lora)
            if ckpt and EARLY_CKPT and s == EARLY_CKPT:
                # an adapter checkpoint early in the book (the acceptance dry run stops at "the first checkpoint with an
                # adapter"); not a resume point (mid-epoch), removed when the first epoch checkpoint is written
                early = ckpt.replace(".npz", ".early.npz")
                save_ckpt(early, lora, state, s, ep, {"early": True})
                log(f"  early adapter checkpoint at step {s}: {early}")
        if out_of_time:
            break
        if replay is not None:
            replay.after_epoch(lora)
        if stop_fn is not None and ep < epochs - 1 and stop_fn(lora, ep + 1):
            log(f"  early stop after epoch {ep + 1}/{epochs} (step {s}/{total})")
            break
        if ckpt and ep < epochs - 1:
            tc = time.time()
            save_ckpt(ckpt, lora, state, s, ep + 1,
                      {"rng": rng.getstate(), "spaced": replay.snapshot() if replay is not None else None})
            log(f"  checkpoint after epoch {ep + 1}/{epochs} (step {s}): {ckpt} in {time.time() - tc:.0f}s")
            if os.path.exists(ckpt.replace(".npz", ".early.npz")):
                os.remove(ckpt.replace(".npz", ".early.npz"))
            if int(os.environ.get("CKPT_STOP_AFTER", "0")) == ep + 1:  # smoke test only: stand in for the lost VM
                log(f"  CKPT_STOP_AFTER: leaving after the checkpoint of epoch {ep + 1}")
                raise SystemExit(3)
    if ckpt:
        remove_ckpt(ckpt)
    if ema is not None and ema_n > 1:
        log(f"  EMA of the factors over the last {EMA_EPOCHS} epochs ({ema_n} steps) used as the book's result")
        return ema
    return lora


def stack_adapter(lora):
    """Unmerged mode: append this book's adapter to the frozen fp32 stacks {kind}_fa [L, in, R] / {kind}_fb [L, R, out]
    (scale folded into fb), which qwen.linear adds on top of the untouched base weights."""
    global params
    scale = CFG["alpha"] / CFG["rank"]
    layers = dict(params["layers"])
    for k in LORA_KINDS:
        a, b = lora[k]["a"], lora[k]["b"] * scale
        if f"{k}_fa" in layers:
            a = jnp.concatenate([layers[f"{k}_fa"], a], -1)
            b = jnp.concatenate([layers[f"{k}_fb"], b], 1)
        layers[f"{k}_fa"], layers[f"{k}_fb"] = a, b
    params = {**params, "layers": layers}


def set_weights(masters, cast):
    """Device weights := cast(masters), one kind at a time, freeing the old array before uploading the new one (the
    all-at-once dict rebuild held two full weight sets on the device: 28 GiB for the 7B)."""
    layers = params["layers"]
    for kd in LORA_KINDS:
        old = layers.pop(f"{kd}_w")
        del old
        layers[f"{kd}_w"] = jnp.asarray(masters[kd].astype(cast))


def lora_delta(lora, kd, scale):
    """a @ b * scale for one kind, fp32 on the host (the device version held two 7 GiB fp32 copies for the 7B)."""
    a, b = np.asarray(lora[kd]["a"], np.float32), np.asarray(lora[kd]["b"], np.float32)
    return np.einsum("lir,lro->lio", a, b, optimize=True) * np.float32(scale)


def set_weight(kd, w_host):
    """Device weight of one kind := w_host (already cast), freeing the old array first."""
    layers = params["layers"]
    old = layers.pop(f"{kd}_w")
    del old
    layers[f"{kd}_w"] = jnp.asarray(w_host)


def merge(master, lora, stats=None):
    """master[kind] += a @ b * scale (fp32, host); the device weights become the cast masters.
    stats (a dict): also record, per kind, |cast(new) - cast(old) - delta| / |delta| (what the cast adds to the exact
    change) and |delta| / |W|."""
    global params
    scale = CFG["alpha"] / CFG["rank"]
    layers = params["layers"]  # mutated in place, one kind at a time (round 28: the old copy is freed before the new one
    for k in LORA_KINDS:        # is uploaded, so the device never holds two full weight sets or a 7 GiB fp32 delta)
        a, b = np.asarray(lora[k]["a"], np.float32), np.asarray(lora[k]["b"], np.float32)
        delta = np.einsum("lir,lro->lio", a, b, optimize=True) * np.float32(scale)  # fp32 on the host, ~1 s per kind
        master[k] += delta
        new = master[k].astype(ml_dtypes.bfloat16 if DTYPE == jnp.bfloat16 else np.float32)
        old = layers.pop(f"{k}_w")
        if stats is not None:
            old_f = np.asarray(old).astype(np.float32)
            noise = new.astype(np.float32) - old_f - delta
            dn = float(np.linalg.norm(delta))
            stats[k] = {"cast_noise_over_delta": float(np.linalg.norm(noise)) / dn,
                        "delta_over_W": dn / float(np.linalg.norm(old_f))}
            del old_f, noise
        del old, delta, a, b
        layers[f"{k}_w"] = jnp.asarray(new)
        del new


capture_fn = jax.jit(lambda p, i, m: qwen.forward(cfg, p, i, m, capture=True)[1])


@functools.partial(jax.jit, static_argnums=4)
def _sample(cap, real, key, p, S):
    """For each kind and layer pick up to S tokens with probability p among real ones; returns rows + validity."""
    out = {}
    for n, k in enumerate(PROT_KEYS):
        x = cap[CAP_OF[k]]  # [L, B, T, d]
        L, B, T, d = x.shape
        u = jax.random.uniform(jax.random.fold_in(key, n), (L, B * T))
        score = jnp.where((u < p) & real.reshape(1, -1), u, -1.0)
        top, idx = jax.lax.top_k(score, S)
        rows = jnp.take_along_axis(x.reshape(L, B * T, d), idx[..., None], 1)
        out[k] = (rows.astype(jnp.float32), top >= 0, jnp.sum(score >= 0, 1))
    return out


# torch eigh on the GPU when there is one (7B: 3584-wide covariances take seconds each on CPU); JAX must then leave
# it some memory (XLA_PYTHON_CLIENT_MEM_FRACTION). On a TPU VM torch sees no CUDA device and stays on the host.
EIG_DEV = "cuda" if torch.cuda.is_available() else "cpu"
if EIG_DEV == "cuda":
    try:  # the package's torch build may have no kernels for this card (cu126 on Blackwell): then the host does it
        _t = torch.randn(64, 64, device="cuda", dtype=torch.float64)
        torch.linalg.eigh(_t @ _t.T)
        torch.cuda.synchronize()
        del _t
    except Exception as _e:
        log(f"torch cannot run on this GPU ({str(_e).splitlines()[0][:90]}); eigendecompositions on the CPU")
        EIG_DEV = "cpu"
if EIG_DEV == "cuda" and os.environ.get("TORCH_MEM_FRACTION"):  # emulate a smaller card: torch's share of the device
    torch.cuda.set_per_process_memory_fraction(float(os.environ["TORCH_MEM_FRACTION"]))
    log(f"torch limited to {float(os.environ['TORCH_MEM_FRACTION']):.2f} of the device memory")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "0")) or max(1, os.cpu_count() or 1))  # TORCH_THREADS: several runs on one host


def host_basis(X, tau, dev=None):
    """Orthonormal basis of the top-energy input directions of X [n, d] (same subspace as lab/train_cl.py's
    V = X^T W / sqrt(ev) from the n x n Gram eigh), on the host with torch/MKL (numpy's LAPACK eigh at n = 4096
    takes minutes; TPU eigh at that size compiles for tens of minutes).
    d <= n: eigenvectors of the d x d covariance, orthonormal by construction. d > n (down_proj): the Gram route in
    float32, redone in float64 when the columns come out non-orthonormal (layers with massive activations, where a
    few tokens are ~1000x larger than the rest). Returns (V [d, k] float32 numpy, k, redone_in_float64)."""
    dev = dev or EIG_DEV
    n, d = X.shape
    Xt = torch.from_numpy(X).to(dev)
    if d <= n:
        ev, U = torch.linalg.eigh(Xt.T @ Xt)
        ev, U = ev.flip(0).clamp_min(1e-8), U.flip(1)
        k = int((ev.cumsum(0) / ev.sum() < tau).sum()) + 1
        if dev != "cpu" and not (torch.isfinite(U[:, :k]).all() and torch.isfinite(ev).all()):
            log(f"  note: GPU eigh of a {d}x{d} covariance returned non-finite values; redone on the CPU")
            return host_basis(X, tau, "cpu")
        return U[:, :k].contiguous().cpu().numpy(), k, False
    for dt in (torch.float32, torch.float64):
        Xd = Xt.to(dt)
        ev, W = torch.linalg.eigh(Xd @ Xd.T)
        ev, W = ev.flip(0).clamp_min(1e-8), W.flip(1)
        k = int((ev.cumsum(0) / ev.sum() < tau).sum()) + 1
        V = ((Xd.T @ W[:, :k]) / ev[:k].sqrt()).float()
        if torch.isfinite(V).all() and (V.T @ V - torch.eye(k, device=V.device)).abs().max() < 1e-3:
            break
    if not torch.isfinite(V).all():
        if dev != "cpu":  # cuSOLVER can return NaN on these ill-conditioned Grams where MKL does not
            log(f"  note: GPU eigh of a {n}x{n} Gram returned non-finite values; redone on the CPU")
            return host_basis(X, tau, "cpu")
        raise FloatingPointError(f"non-finite basis for a {n}x{d} sample even in float64 on the CPU")
    return V.cpu().numpy(), k, dt is torch.float64


def _new_dirs(Vt):
    """Left singular vectors of Vt [d, k] with singular value > 0.1. MKL's fp32 SVD can fail to converge on these
    residuals (seen once the general-text basis is stored), so: float64 next, then the eigh of Vt^T Vt."""
    for dt in (torch.float32, torch.float64):
        try:
            u, s, _ = torch.linalg.svd(Vt.to(dt), full_matrices=False)
            u = u[:, s > 0.1]
            if torch.isfinite(u).all():  # round 16: an fp64 SVD after a failed fp32 one gave NaN (all losses NaN after)
                return u.float()
            log(f"  note: {dt} SVD of a {tuple(Vt.shape)} residual returned non-finite vectors")
        except torch._C._LinAlgError:
            log(f"  note: {dt} SVD of a {tuple(Vt.shape)} residual did not converge")
    V64 = Vt.double()
    ev, w = torch.linalg.eigh(V64.T @ V64)
    keep = ev > 0.01  # singular value > 0.1
    V = (V64 @ w[:, keep] / ev[keep].sqrt()).float()
    if not torch.isfinite(V).all():
        raise FloatingPointError(f"non-finite new directions for a {tuple(Vt.shape)} residual")
    return V


def orthonormal(V, tol=1e-3):
    """QR re-orthonormalisation of V [d, k] (float64), dropping near-dependent columns (|R_ii| < tol). Round 16d: the
    eigh fallback of _new_dirs gave a down_proj basis with non-orthonormal columns; a - Q(Q^T a) then DOUBLED the
    LoRA A of that kind at every step (2.4 at step 10, 7e28 at step 100, NaN at 200) while every other kind stayed
    at 0.04. The projection is only a projection when Q^T Q = I."""
    q, r = torch.linalg.qr(torch.as_tensor(V).double())
    keep = r.diagonal().abs() > tol
    return q[:, keep].float()


def _sample_feats(enc, key):
    """Layer inputs of enc's tokens, sampled on the device with p = n_feat / total tokens, independently per linear
    kind and layer: {kind: [per-layer list of np.ndarray rows]}."""
    total = sum(len(x) for x, _ in enc)
    p = min(1.0, CFG["n_feat"] / total)
    # cap per (batch, kind, layer): 4x the expected count, at least 48 (book rows: ~12 expected; the longer
    # general-text rows: ~100); more is logged as a cut
    S = max(48, 4 * math.ceil(p * total / math.ceil(len(enc) / CFG["micro"])))
    feats = {k: [[] for _ in range(cfg["L"])] for k in PROT_KEYS}
    full = 0
    for b, mb in enumerate(micro_batches(enc, random.Random(1))):
        n = max(len(x) for x, _ in mb)
        # general-text rows run past 128 tokens; the 128 cut-off is fixed (not the largest bucket) so the sampled tokens,
        # which depend on T, are those of the earlier runs
        T = bucket(n) if n <= 128 else -(-n // 64) * 64
        ids, _, att = pad_batch(mb, CFG["micro"], T)
        cap = capture_fn(params, jnp.asarray(ids), jnp.asarray(att))
        res = _sample(cap, jnp.asarray(att > 0), jax.random.fold_in(key, b), p, min(S, CFG["micro"] * T))
        for k, (rows, ok, cnt) in res.items():
            rows, ok, cnt = np.asarray(rows), np.asarray(ok), np.asarray(cnt)
            full += int((cnt > ok.shape[1]).sum())
            for l in range(cfg["L"]):
                feats[k][l].append(rows[l][ok[l]])
    if full:
        log(f"  warning: {full} (batch, kind, layer) samples exceeded {S} tokens and were cut")
    return feats


def _rows(feats, k, l, norm):
    X = np.concatenate(feats[k][l])
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12) if norm else X


def update_subspace(Qs_host, enc, tau, key, norm=False):
    """Qs_host: {kind: [per-layer np.ndarray [in, k]]}; extended in place with this book's subspace.
    Layer inputs come from the device (sampled there), the linear algebra runs on the host.
    tau: a share of energy, or {kind: share, "*": default} to protect some kinds more than others.
    norm: scale every sampled token to unit length first, so tau counts directions per token. Without it, in layers
    with massive activations (Qwen2.5-1.5B down_proj L1-L2: one token ~37,000x the median energy) a few tokens fill
    the tau share and ordinary tokens' directions are left unprotected."""
    feats = _sample_feats(enc, key)
    ks, orth, bad, redo = [], 0.0, [], 0
    for k in PROT_KEYS:
        t = tau.get(k, tau["*"]) if isinstance(tau, dict) else tau
        for l in range(cfg["L"]):
            X = _rows(feats, k, l, norm)
            V, kk, r64 = host_basis(X, t)
            redo += r64
            e = float(np.abs(V.T @ V - np.eye(V.shape[1], dtype=np.float32)).max())
            orth = max(orth, e)
            if os.environ.get("DEBUG_BASIS"):
                cn = np.linalg.norm(V, axis=0)
                bad.append((e, k, l, len(X), kk, float(cn.min()), float(cn.max()), int(np.isfinite(V).all())))
            Q = Qs_host[k][l]
            if Q is not None:
                Qt, Vt = torch.from_numpy(Q), torch.from_numpy(V)
                Vt = Vt - Qt @ (Qt.T @ Vt)
                new = orthonormal(_new_dirs(Vt))
                new = orthonormal(new - Qt @ (Qt.T @ new))  # remove the rounding left along Q, then unit length again
                V = torch.cat([Qt, new], 1).numpy()
            else:
                V = orthonormal(V).numpy()
            Qs_host[k][l] = V.astype(np.float32)
            ks.append(V.shape[1])
            feats[k][l] = None
    log(f"  subspace: {len(ks)} modules, mean protected dim {sum(ks) / len(ks):.0f}, max {max(ks)}, "
        f"new basis max |V^T V - I| {orth:.1e}, {redo} down_proj modules needed float64")
    for b in sorted(bad, reverse=True)[:5]:
        log("  worst basis (orth err, kind, layer, n rows, k, min/max column norm, finite):", b)


def new_book_dirs(Qs_host, enc, tau_new, key, norm=False):
    """Directions where the NEW book's inputs (enc) carry energy outside the protected subspace: the residual of each
    sampled input after removing its Q part, and the residual's top-energy basis (tau_new share), orthogonal to Q.
    Training only inside this basis (instead of everywhere outside Q) leaves out the leftover directions the new
    book barely uses, where a large update would only act on old inputs' small residuals. {kind: [per-layer P]}."""
    feats = _sample_feats(enc, key)
    Ps, ks = {k: [None] * cfg["L"] for k in LORA_KINDS}, []
    for k in LORA_KINDS:
        for l in range(cfg["L"]):
            X = _rows(feats, k, l, norm)
            Q = Qs_host[k][l]
            if Q is not None:
                X = X - (X @ Q) @ Q.T
            V, _, _ = host_basis(X.astype(np.float32), tau_new)
            if Q is not None:  # remove rounding leftovers along Q and re-orthonormalise
                Qt, Vt = torch.from_numpy(Q), torch.from_numpy(V)
                V = _new_dirs(Vt - Qt @ (Qt.T @ Vt)).numpy()
            Ps[k][l] = V.astype(np.float32)
            ks.append(V.shape[1])
            feats[k][l] = None
    log(f"  new-book directions (tau_new {tau_new}): mean dim {sum(ks) / len(ks):.0f}, max {max(ks)}")
    return Ps


def stack_Q(Qs_host):
    """Per-layer bases -> {kind: [L, in, width]} zero-padded (zero columns do not change the projection).
    Width = the input size when it is small (a basis never has more columns), else a multiple of 1024, so the
    compiled training step is reused across books and seeds."""
    out = {}
    for k, per in Qs_host.items():
        if per[0] is None:
            continue
        fin = per[0].shape[0]
        km = max(q.shape[1] for q in per)
        km = fin if fin <= 2048 else min(fin, -(-km // 1024) * 1024)
        arr = np.zeros((cfg["L"], per[0].shape[0], km), np.float32)
        for l, q in enumerate(per):
            arr[l, :, :q.shape[1]] = q
        if not np.isfinite(arr).all():
            raise FloatingPointError(f"non-finite protected basis for {k}")
        # full |Q^T Q - I| per layer (the 64-column sample of round 16c missed the bad columns); repair by QR
        err, fixed = 0.0, 0
        for l, q in enumerate(per):
            qt = torch.from_numpy(q)
            e = float((qt.T @ qt - torch.eye(q.shape[1])).abs().max())
            err = max(err, e)
            if e > 1e-3:
                qn = orthonormal(qt).numpy()
                log(f"  basis {k} layer {l}: |Q^T Q - I| = {e:.2e}, re-orthonormalised {q.shape[1]} -> {qn.shape[1]} columns")
                arr[l] = 0.0
                arr[l, :, :qn.shape[1]] = qn
                per[l] = qn.astype(np.float32)
                fixed += 1
        log(f"  basis {k}: dims {min(q.shape[1] for q in per)}-{max(q.shape[1] for q in per)} of {fin}, "
            f"max |Q^T Q - I| {err:.1e}" + (f", {fixed} layers repaired" if fixed else ""))
        out[k] = jnp.asarray(arr if Q_DTYPE == jnp.float32 else arr.astype(Q_DTYPE))  # round 28: Q_DTYPE=bf16 halves the bases on the device (cast on the host, so no fp32 copy lands there; train.project then projects twice)
        del arr
    import gc
    gc.collect()
    log(f"  stacked bases ({_rss()})")
    return out


def answer(exam):
    out = {}
    B, P = CFG["gen_bs"], CFG["prompt_len"]
    tok.padding_side = "left"
    for i in range(0, len(exam), B):
        chunk = exam[i:i + B]
        prompts = [chat_prompt(it["q"]) for it in chunk] + [chat_prompt("?")] * (B - len(chunk))
        enc = tok(prompts, return_tensors="np", padding="max_length", max_length=P)
        assert enc["input_ids"].shape[1] == P, "a prompt is longer than prompt_len"
        toks = np.asarray(gen(params, jnp.asarray(enc["input_ids"]), jnp.asarray(enc["attention_mask"])))
        for it, t in zip(chunk, toks):
            out[it["id"]] = tok.decode(t, skip_special_tokens=True)
    return out


def subsample(ex):
    """EXAMN>0: per kind, round-robin over trap types / relations so every type stays in the sample."""
    if not EXAMN:
        return ex
    sub = []
    for kind in ("direct", "indirect", "trap", "twin"):
        groups = {}
        for y in ex:
            if y["kind"] == kind:
                groups.setdefault(y.get("trap_type") or y.get("rel"), []).append(y)
        rr = [g[i] for i in range(max(map(len, groups.values()), default=0)) for g in groups.values() if i < len(g)]
        sub += rr[:EXAMN]
    return sub


def load_exam(k, name="exam"):
    return subsample(load(name, k)) if os.path.exists(f"{SRC}/book{k}.{name}.json") else None


def fact_questions(f):
    """(direct q, gold, inverse q, gold): the templated questions of a synthetic fact, or the teacher-written ones a
    generic fact carries (round 29, generic.py)."""
    return tuple(f["q"]) if "q" in f else fl.questions(f)


def probe_items(k):
    """Same facts asked with the TRAINING wordings, as in lab/train_cl.py."""
    rng = random.Random(5)
    facts = load("facts", k)
    items = []
    for kind, idx in (("direct", 0), ("indirect", 2)):
        pool = facts if idx == 0 else [f for f in facts if not f.get("qi_fallback")]  # round 29: fallback inverse
        for f in rng.sample(pool, min(75, len(pool))):                                    # questions are ambiguous
            q = fact_questions(f)
            items.append({"kind": kind, "q": q[idx], "gold": q[idx + 1]})
    for i, it in enumerate(items):
        it["id"] = f"p{i:03d}"
    return subsample(items)


def general_rows(n_eval, n_train=300):
    """As lab/train_cl.py: the first n_train streamed texts are the protection set (runs with tau_gen), the next
    n_eval are the evaluation rows. Returns (train rows, eval rows)."""
    gen_file = os.environ.get("GEN_FILE")  # round 29: the package ships the texts (no network beyond the pinned base)
    if gen_file and os.path.exists(gen_file):
        texts = json.load(open(gen_file))[:n_train + n_eval]
        log(f"general-text rows from {gen_file}: {len(texts)}")
        rows = [{"type": "text", "text": x} for x in texts]
        return rows[:n_train], rows[n_train:]
    from datasets import load_dataset
    texts = []
    try:
        for ex in load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", streaming=True):
            texts.append(ex["text"][:700])
            if len(texts) >= n_train + n_eval:
                break
    except Exception as e:
        log("wikipedia unavailable:", repr(e)[:150], "- falling back to wikitext-2")
        texts = [x["text"][:700] for x in load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                 if len(x["text"]) > 300][:n_train + n_eval]
    if gen_file and not os.path.exists(gen_file):
        json.dump(texts, open(gen_file, "w"), ensure_ascii=False, indent=0)
        log(f"general-text rows saved to {gen_file}: {len(texts)}")
    rows = [{"type": "text", "text": x} for x in texts]
    return rows[:n_train], rows[n_train:]


nll_fn = jax.jit(lambda p, i, m, l: qwen.lm_loss(qwen.forward(cfg, p, i, m), l))


def gen_nll(enc_rows):
    """Mean over rows of each micro-batch's token-mean loss, weighted by rows (as the GPU script)."""
    T = -(-max(len(x) for x, _ in enc_rows) // 64) * 64
    tot = n = 0
    for mb in micro_batches(enc_rows, random.Random(2)):
        ids, lab, att = pad_batch(mb, len(mb), T)
        tot += float(nll_fn(params, jnp.asarray(ids), jnp.asarray(att), jnp.asarray(lab))) * len(mb)
        n += len(mb)
    return tot / n


exams = {k: load_exam(k) for k in CFG["books"]}
exams_gen = {k: load_exam(k, "exam_gen") for k in CFG["books"]}
probes = {k: probe_items(k) if os.path.exists(f"{SRC}/book{k}.facts.json") else [] for k in CFG["books"]}
_gen_train, _gen_eval = general_rows(GEN_N or 40, int(os.environ.get("GEN_TRAIN_N", "300")))
GEN_EVAL = [encode(r) for r in _gen_eval]
GEN_TRAIN = [encode(r) for r in _gen_train] if any(r.get("tau_gen") for r in CFG["runs"]) else None

# ---- round 16: protect what the base model does when it follows instructions and reasons -------------------------
# Self-instruct rows: the base model writes instructions from seed requests (topics x four kinds of task: a short text
# under a formatting rule, a multi-step arithmetic word problem, a multiple-choice question, an explanation; English
# and Russian) and then answers them itself. No benchmark item and no outside data. A run with "tau_chat" adds the
# layer-input subspace of these rows to the protection, as "tau_gen" does for Wikipedia text (L5: IFEval -6.3 and
# GSM8K flexible-extract -11.6 points after book 4 with the Wikipedia protection alone). The held-out rows give
# "chat NLL", the trained model's loss on the base model's own answers: a cheap stand-in for L5 drift.
CHAT_N = int(os.environ.get("CHAT_N", "320"))  # instructions in all; CHAT_EVAL_N of them are held out
CHAT_EVAL_N = int(os.environ.get("CHAT_EVAL_N", "64"))
CHAT_FILE = os.environ.get("CHAT_FILE", os.path.join(os.path.dirname(OUT) or ".", "chat_self.json"))
TOPICS_EN = ["the weather", "cooking pasta", "a city library", "bicycles", "the solar system", "saving money",
             "a school trip", "gardening", "trains", "the human heart", "chess", "recycling", "a birthday party",
             "volcanoes", "learning a language", "a bakery", "the ocean", "computers", "a football match", "bees",
             "a museum visit", "electric cars", "sleep", "a farm", "photography", "the Roman Empire", "rainforests",
             "a job interview", "music lessons", "a camping trip", "the moon", "coffee", "a hospital", "bridges",
             "winter sports", "a pet dog", "the internet", "a shop sale", "mountains", "a family dinner"]
TOPICS_RU = ["погода", "приготовление супа", "городская библиотека", "велосипеды", "Солнечная система", "экономия денег",
             "школьная экскурсия", "огород", "поезда", "сердце человека", "шахматы", "переработка мусора", "день рождения",
             "вулканы", "изучение языка", "пекарня", "море", "компьютеры", "футбольный матч", "пчёлы", "поход в музей",
             "электромобили", "сон", "ферма", "фотография", "Древний Рим", "леса", "собеседование", "уроки музыки",
             "поход с палаткой", "Луна", "чай", "больница", "мосты", "зимние виды спорта", "собака", "интернет",
             "распродажа в магазине", "горы", "семейный ужин"]
SEEDS_EN = ["Write one instruction that a user could give to an AI assistant about {t}. It must ask for a short text and "
            "add one formatting rule, for example exactly 3 bullet points, fewer than 50 words, all capital letters, or "
            "a title in double angular brackets. Write only the instruction.",
            "Write one grade-school math word problem about {t} that needs two to four steps of arithmetic. Write only "
            "the problem, without the answer.",
            "Write one multiple-choice question about {t} with four options labelled A, B, C and D. Write only the "
            "question and the options, without the answer.",
            "Write one question about {t} that asks for an explanation in simple words. Write only the question."]
SEEDS_EN += ["Write one request about {t} that asks an AI assistant to rewrite a given short paragraph in a formal "
             "style; include the paragraph (two or three sentences). Write only the request with its paragraph.",
             "Write one request about {t} that asks for step-by-step instructions to do something, numbered. Write only "
             "the request.",
             "Write one question about {t} that asks to compare two things and say which is better for a stated "
             "purpose. Write only the question.",
             "Write one request about {t} that asks for a JSON object with three named fields describing it. Write only "
             "the request."]
SEEDS_RU = ["Напиши одно задание для ИИ-ассистента на тему «{t}». Оно должно просить короткий текст и добавлять одно "
            "правило оформления, например ровно 3 пункта списка, не больше 50 слов, только заглавные буквы или заголовок "
            "в двойных угловых скобках. Напиши только задание.",
            "Напиши одну школьную текстовую задачу по математике на тему «{t}», которая решается в два-четыре "
            "действия. Напиши только условие, без ответа.",
            "Напиши один вопрос с выбором ответа на тему «{t}» с четырьмя вариантами A, B, C и D. Напиши только вопрос "
            "и варианты, без ответа.",
            "Напиши один вопрос на тему «{t}», который просит объяснить что-то простыми словами. Напиши только вопрос.",
            "Напиши одну просьбу на тему «{t}»: переписать данный короткий абзац в официальном стиле; приведи сам абзац "
            "(два-три предложения). Напиши только просьбу с абзацем.",
            "Напиши одну просьбу на тему «{t}»: дать пошаговую нумерованную инструкцию, как что-то сделать. Напиши только "
            "просьбу.",
            "Напиши один вопрос на тему «{t}», который просит сравнить две вещи и сказать, какая лучше для указанной цели. "
            "Напиши только вопрос.",
            "Напиши одну просьбу на тему «{t}»: дать JSON-объект с тремя именованными полями, описывающими её. Напиши "
            "только просьбу."]


# round 23: a math-heavy self-instruct set for the protection (GSM8K kept losing 2.7-4.2 points with 80 math problems
# among the 320 rows). CHAT_MATH=1 draws the generated rows from these six kinds of multi-step arithmetic problem;
# CHAT_EXTRA=file prepends an earlier set (so its first CHAT_EVAL_N rows stay the held-out chat-NLL set).
CHAT_MATH = bool(os.environ.get("CHAT_MATH"))
CHAT_EXTRA = os.environ.get("CHAT_EXTRA")
SEEDS_MATH_EN = [
    "Write one grade-school math word problem about {t} that needs two to four steps of arithmetic with whole numbers. "
    "Write only the problem, without the answer.",
    "Write one math word problem about {t} that involves a percentage, a discount or a share of a total, solvable in "
    "two or three steps. Write only the problem, without the answer.",
    "Write one math word problem about {t} about speed, time and distance, or about a rate per hour or per day, "
    "solvable in two or three steps. Write only the problem, without the answer.",
    "Write one math word problem about {t} that needs a unit conversion (minutes and hours, grams and kilograms, "
    "cents and dollars) before the arithmetic. Write only the problem, without the answer.",
    "Write one short reasoning puzzle about {t} with numbers (ages, counts, remainders, comparisons) that needs two or "
    "three logical steps. Write only the puzzle, without the answer.",
    "Write one question about {t} that asks to compute a number from given quantities and to explain each step. Write "
    "only the question, with the quantities, without the answer."]
SEEDS_MATH_RU = [
    "Напиши одну школьную текстовую задачу по математике на тему «{t}», которая решается в два-четыре действия с целыми "
    "числами. Напиши только условие, без ответа.",
    "Напиши одну текстовую задачу на тему «{t}» с процентами, скидкой или долей от целого, решаемую в два-три действия. "
    "Напиши только условие, без ответа.",
    "Напиши одну текстовую задачу на тему «{t}» про скорость, время и расстояние или про норму в час или в день, "
    "решаемую в два-три действия. Напиши только условие, без ответа.",
    "Напиши одну текстовую задачу на тему «{t}», где перед вычислением нужно перевести единицы (минуты и часы, граммы и "
    "килограммы, копейки и рубли). Напиши только условие, без ответа.",
    "Напиши одну короткую логическую задачу с числами на тему «{t}» (возраст, количество, остаток, сравнение), "
    "требующую двух-трёх шагов рассуждения. Напиши только условие, без ответа.",
    "Напиши один вопрос на тему «{t}», который просит вычислить число по заданным величинам и объяснить каждый шаг. "
    "Напиши только вопрос с величинами, без ответа."]


# ---- teacher log and teacher budget (package README: "Teacher") ------------------------------------------------------
# The teacher is the pinned base itself, run locally, so every call costs $0 against TEACHER_BUDGET_USD (default 20, the
# cap of section 7). The guard is real nonetheless: teacher_charge() raises TeacherBudget when a call would take the
# cumulative spend over the cap, and generic.build / self_instruct catch it and go on with the rows written so far, so
# the book still ends with a model. Every call is appended to TEACHER_LOG (jsonl) with its size and one sample.
TEACHER_LOG = os.environ.get("TEACHER_LOG")
TEACHER_BUDGET_USD = float(os.environ.get("TEACHER_BUDGET_USD", "20"))
TEACHER_PRICE_PER_1K = float(os.environ.get("TEACHER_PRICE_PER_1K", "0"))  # $ per 1k generated tokens; 0 = local base
teacher_spent = 0.0


class TeacherBudget(Exception):
    pass


def teacher_charge(n_tokens):
    global teacher_spent
    cost = n_tokens / 1000 * TEACHER_PRICE_PER_1K
    if teacher_spent + cost > TEACHER_BUDGET_USD:
        raise TeacherBudget(f"teacher budget: {teacher_spent:.4f} + {cost:.4f} > {TEACHER_BUDGET_USD} USD")
    teacher_spent += cost


def teacher_log(rec):
    if TEACHER_LOG:
        with open(TEACHER_LOG, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def answer_texts(texts, n_new, P):
    """Greedy answers of the current model (with the generation config, as answer()) to free-form prompts of up to
    P tokens with the chat template; longer prompts are skipped (returned as None). Every batch is charged to the
    teacher budget first and logged after."""
    g = generate.make_generate(cfg, gcfg, n_new)
    tok.padding_side = "left"
    B, out = CFG["gen_bs"], [None] * len(texts)
    fit = [i for i, t in enumerate(texts) if len(tok(chat_prompt(t))["input_ids"]) <= P]
    ts, n_gen = time.time(), 0
    for j in range(0, len(fit), B):
        idx = fit[j:j + B]
        teacher_charge(len(idx) * n_new)
        prompts = [chat_prompt(texts[i]) for i in idx] + [chat_prompt("?")] * (B - len(idx))
        enc = tok(prompts, return_tensors="np", padding="max_length", max_length=P)
        toks = np.asarray(g(params, jnp.asarray(enc["input_ids"]), jnp.asarray(enc["attention_mask"])))
        for i, t in zip(idx, toks):
            out[i] = tok.decode(t, skip_special_tokens=True).strip()
            n_gen += int((t != tok.pad_token_id).sum()) if tok.pad_token_id is not None else len(t)
    teacher_log({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "teacher": CFG["model"],
                 "where": "local (the pinned base, same process)", "prompts": len(texts), "answered": len(fit),
                 "skipped_too_long": len(texts) - len(fit), "max_prompt_tokens": P, "max_new_tokens": n_new,
                 "generated_tokens": n_gen, "seconds": round(time.time() - ts, 1), "cost_usd": 0.0,
                 "spent_usd": round(teacher_spent, 4), "budget_usd": TEACHER_BUDGET_USD,
                 "sample": {"prompt": texts[fit[0]][:400], "answer": (out[fit[0]] or "")[:400]} if fit else None})
    return out


def self_instruct(n):
    s_en, s_ru = (SEEDS_MATH_EN, SEEDS_MATH_RU) if CHAT_MATH else (SEEDS_EN, SEEDS_RU)
    seeds = [s.format(t=t) for k in range(len(s_en)) for s, topics in ((s_en[k], TOPICS_EN), (s_ru[k], TOPICS_RU))
             for t in topics]  # 8 (math: 6) kinds x 2 languages x 40 topics; CHAT_N picks a fixed random subset
    seeds = [seeds[i] for i in random.Random(3).sample(range(len(seeds)), min(n, len(seeds)))]
    ts = time.time()
    insts = answer_texts(seeds, 96, 128)
    keep = [i for i, x in enumerate(insts) if x and len(x) > 15]
    resp = answer_texts([insts[i] for i in keep], 256, 192)
    rows = [{"seed": seeds[i], "user": insts[i], "assistant": r} for i, r in zip(keep, resp) if r]
    log(f"self-instruct: {len(rows)} rows from {len(seeds)} seeds in {time.time() - ts:.0f}s")
    return rows


CHAT_TRAIN = CHAT_EVAL = None
if CHAT_N:
    if os.path.exists(CHAT_FILE):
        _chat = json.load(open(CHAT_FILE))
        log(f"self-instruct rows from {CHAT_FILE}: {len(_chat)}")
    else:
        _chat = self_instruct(CHAT_N)
        json.dump(_chat, open(CHAT_FILE, "w"), ensure_ascii=False, indent=1)
    if CHAT_EXTRA:  # an earlier set first, so its held-out rows stay the chat-NLL set
        _extra = json.load(open(CHAT_EXTRA))
        log(f"self-instruct rows from {CHAT_EXTRA} first: {len(_extra)}, then {len(_chat)} from {CHAT_FILE}")
        _chat = _extra + _chat
    _enc = [encode({"type": "qa", **r}) for r in _chat]
    CHAT_EVAL, CHAT_TRAIN = _enc[:CHAT_EVAL_N], _enc[CHAT_EVAL_N:]
# ---- round 29: generic books -> rows and facts written by the base model (generic.py) --------------------------------
# GENERIC_BOOKS="2,4": for those books, read the units from GENERIC_PAT.format(k=k) (default book{k}.json; the synthetic
# test uses book{k}.book.json) and write book{k}.train.jsonl + book{k}.facts.json into SRC before anything reads them
# (skipped when both exist and GENERIC_FORCE is unset). The exam, if a file exists, is still the hand-held one.
GENERIC_BOOKS = [int(x) for x in os.environ.get("GENERIC_BOOKS", "").split(",") if x]
if GENERIC_BOOKS:
    import generic
    for k in GENERIC_BOOKS:
        src_units = f"{SRC}/" + os.environ.get("GENERIC_PAT", "book{k}.json").format(k=k)
        if os.path.exists(f"{SRC}/book{k}.train.jsonl") and os.path.exists(f"{SRC}/book{k}.facts.json") \
                and not os.environ.get("GENERIC_FORCE") and json.load(open(f"{SRC}/book{k}.facts.json"))[0].get("rel") == "generic":
            log(f"generic book {k}: rows and facts already built")
        else:
            units = generic.load_units(src_units)
            ts = time.time()
            _tstop = {"v": False}

            def _gen(t, n, P, _tstop=_tstop, k=k):
                """answer_texts under the teacher budget: at the cap the teacher is not called again for this book and
                the stages go on with what parsed, so the book still ends with a model."""
                if _tstop["v"]:
                    return [None] * len(t)
                try:
                    return answer_texts(t, n, P)
                except TeacherBudget as e:
                    log(f"  {e}; no further teacher calls for book {k}, training on the rows written so far")
                    _tstop["v"] = True
                    return [None] * len(t)
            gb0, CFG["gen_bs"] = CFG["gen_bs"], int(os.environ.get("GEN_BS_DATA", "48"))  # no bases on the device yet
            prem = int(os.environ.get("GENERIC_PREM", "10" if len(units) <= 500 else "5"))  # the big book: fewer pairs
            facts, rows = generic.build(units, _gen, seed=k, n_prem=prem, log=log,
                                        n_para=int(os.environ.get("GENERIC_PARA", "3")), n_qa2=int(os.environ.get("GENERIC_QA2", "2")),
                                        false_w=int(os.environ.get("GENERIC_FALSE_W", "1")),
                                        true_forms=[int(x) for x in os.environ.get("GENERIC_TRUE_FORMS", "").split(",") if x] or None,
                                        false_forms=[int(x) for x in os.environ.get("GENERIC_FALSE_FORMS", "").split(",") if x] or None)
            CFG["gen_bs"] = gb0
            json.dump(facts, open(f"{SRC}/book{k}.facts.json", "w"), ensure_ascii=False, indent=1)
            with open(f"{SRC}/book{k}.train.jsonl", "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"generic book {k}: {len(units)} units -> {len(rows)} rows, {len(facts)} facts in {time.time() - ts:.0f}s")
        probes[k] = probe_items(k)
    if os.environ.get("GENERIC_ONLY"):
        log("GENERIC_ONLY: rows and facts written, stopping here")
        sys.exit(0)

res = {"cfg": {**CFG, "backend": jax.default_backend(), "dtype": str(DTYPE.dtype)}, "base": {}, "runs": {},
       "base_gen_nll": None}


def save():
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)


def chat_nll():
    return round(gen_nll(CHAT_EVAL), 4) if CHAT_EVAL else None


def evaluate(books, first, stage):
    out = {}
    for j in books:
        te = time.time()
        if exams[j] is None:  # a generic book without an exam (round 29): the self-test probes only
            ans, r = {}, {}
        else:
            ans = answer(exams[j])
            r = fl.score(exams[j], ans)
            vec = [int(fl.matches(ans[it["id"]], it["gold"])) for it in exams[j] if it["kind"] in ("direct", "indirect")]
            if first is not None:
                if j not in first:
                    first[j] = vec
                else:
                    r["retention_b"] = round(fl.retention(first[j], vec), 3)
                    r["retention_a"] = round(fl.retention_a(first[j], vec), 3)
            r["samples"] = [{"q": it["q"], "kind": it["kind"], "gold": it["gold"], "ans": ans[it["id"]]}
                            for kind, n in (("trap", 6), ("twin", 6), ("indirect", 4))
                            for it in [x for x in exams[j] if x["kind"] == kind][:n]]
        pans = answer(probes[j])
        pvec = [int(fl.matches(pans[it["id"]], it["gold"])) for it in probes[j]]
        r["probe_know"], r["probe_n"] = sum(pvec), len(pvec)
        r["probe_wrong"] = [{"kind": it["kind"], "q": it["q"], "gold": it["gold"], "ans": pans[it["id"]][:120]}
                            for it, ok in zip(probes[j], pvec) if not ok][:40]
        if first is not None:
            if ("p", j) not in first:
                first[("p", j)] = pvec
            else:
                r["probe_retention_b"] = round(fl.retention(first[("p", j)], pvec), 3)
        r["answers"] = ans
        out[str(j)] = r
        log(stage, "book", j, {k: v for k, v in r.items() if k not in ("samples", "answers", "denied_by_type")})
        if exams_gen.get(j):
            eg = exams_gen[j]
            gans = answer(eg)
            g = fl.score(eg, gans)
            gvec = [int(fl.matches(gans[it["id"]], it["gold"])) for it in eg if it["kind"] in ("direct", "indirect")]
            if first is not None:
                if ("g", j) not in first:
                    first[("g", j)] = gvec
                else:
                    g["retention_b"] = round(fl.retention(first[("g", j)], gvec), 3)
            g["answers"] = gans
            out[f"{j}g"] = g
            log(stage, "book", j, "gen exam", {k: v for k, v in g.items() if k != "answers"})
        log(f"  eval of book {j}: {time.time() - te:.0f}s")
    return out


# ---- DIAG=1: where does an old book leak? (run with BOOKS=2,3) -------------------------------------------------
# After book 2's subspace update: share of the energy of book-2 inputs (its exam questions with the model's own correct
# answers, and training rows) outside the protected subspace. After book 3's training: |dW x| / |W x| of the new LoRA
# on those exam inputs vs on book-3 training rows. After book 3's merge: drift of the exam inputs at every layer and
# their energy outside the subspace used during book 3.
DIAG = os.environ.get("DIAG") or os.environ.get("DIAG2")
REP = {"q": "attn_in", "o": "o_in", "gate": "mlp_in", "down": "down_in"}  # one LoRA kind per layer input
DIAG_N = 64


@jax.jit
def _diag_energy(cap, w, Qs):
    out = {}
    for k, inp in REP.items():
        x = cap[inp].astype(jnp.float32)
        qx = jnp.einsum("lbtd,ldk->lbtk", x, Qs[k], precision=HI)
        out[k] = (jnp.einsum("lbtd,bt->l", x * x, w), jnp.einsum("lbtk,bt->l", qx * qx, w))
    return out


@jax.jit
def _diag_change(cap, w, layers, lora):
    out = {}
    for k, inp in REP.items():
        x = cap[inp]
        wx = jnp.einsum("lbtd,ldo->lbto", x, layers[f"{k}_w"], preferred_element_type=jnp.float32)
        dx = jnp.einsum("lbtd,ldr->lbtr", x.astype(jnp.float32), lora[k]["a"], precision=HI)
        dwx = jnp.einsum("lbtr,lro->lbto", dx, lora[k]["b"], precision=HI) * (CFG["alpha"] / CFG["rank"])
        out[k] = (jnp.einsum("lbto,bt->l", wx * wx, w), jnp.einsum("lbto,bt->l", dwx * dwx, w))
    return out


@jax.jit
def _diag_drift(cap, w, old):
    out = {}
    for inp in set(REP.values()):
        x, x0 = cap[inp].astype(jnp.float32), old[inp].astype(jnp.float32)
        out[inp] = (jnp.einsum("lbtd,bt->l", (x - x0) ** 2, w), jnp.einsum("lbtd,bt->l", x0 * x0, w))
    return out


capture_lora_fn = jax.jit(lambda p, i, m, lo: qwen.forward(cfg, p, i, m, lora=lo, lora_scale=CFG["alpha"] / CFG["rank"],
                                                          capture=True)[1])


@jax.jit
def _diag_tok(cap, q_down):
    """Per token, at the down_proj inputs of layers 1 and 2: energy, and energy outside the protected subspace."""
    x = cap["down_in"][1:3].astype(jnp.float32)
    qx = jnp.einsum("lbtd,ldk->lbtk", x, q_down[1:3], precision=HI)
    tot = jnp.sum(x * x, -1)
    return tot, tot - jnp.sum(qx * qx, -1)


def _diag_pass(enc, fn, *args, lora=None):
    """Run fn(cap, weights, *args) over micro-batches of enc for two token masks ('all' real tokens, 'ans' = labelled
    tokens); returns {mask: {name: (sum_a [L], sum_b [L])}} as numpy, or the captures if fn is None. lora: capture
    through the unmerged LoRA path on top of the current weights (the exact change, without the merge's cast)."""
    acc, caps = {}, []
    for i in range(0, len(enc), CFG["micro"]):
        mb = enc[i:i + CFG["micro"]]
        ids, lab, att = pad_batch(mb, CFG["micro"], bucket(max(len(x) for x, _ in mb)))
        if lora is None:
            cap = capture_fn(params, jnp.asarray(ids), jnp.asarray(att))
        else:
            cap = capture_lora_fn(params, jnp.asarray(ids), jnp.asarray(att), lora)
        if fn is None:
            caps.append(jax.device_get(cap))
            continue
        for m, w in (("all", att > 0), ("ans", (att > 0) & (lab != -100))):
            extra = [a(i // CFG["micro"]) if callable(a) else a for a in args]
            res_ = jax.device_get(fn(cap, jnp.asarray(w, jnp.float32), *extra))
            for name, (a, b) in res_.items():
                pa, pb = acc.setdefault(m, {}).get(name, (0.0, 0.0))
                acc[m][name] = (pa + np.asarray(a, np.float64), pb + np.asarray(b, np.float64))
    return caps if fn is None else acc


def _dev_mem():
    """Peak device memory so far (GPU/TPU backends that report it), for sizing larger models."""
    try:
        st = jax.devices()[0].memory_stats() or {}
        return f" (device peak {st['peak_bytes_in_use'] / 2**30:.1f} of {st.get('bytes_limit', 0) / 2**30:.1f} GiB)"
    except (KeyError, AttributeError, RuntimeError):
        return ""


def _rss():
    try:
        return "host RSS " + next(l.split(":")[1].strip() for l in open("/proc/self/status") if l.startswith("VmRSS"))
    except (OSError, StopIteration):
        return "host RSS ?"


def _summ(per_layer):
    v = np.asarray(per_layer)
    return f"mean {v.mean():.3g} max {v.max():.3g} (L{int(v.argmax())})"


def _top_tokens(enc, q_down, label):
    """Log the tokens that carry most of the energy outside the subspace at down_proj layers 1-2 (first batch)."""
    mb = enc[:CFG["micro"]]
    ids, lab, att = pad_batch(mb, CFG["micro"], bucket(max(len(x) for x, _ in mb)))
    tot, out = (np.asarray(a) for a in _diag_tok(capture_fn(params, jnp.asarray(ids), jnp.asarray(att)), q_down))
    ans = (att > 0) & (lab != -100)
    for li, layer in enumerate((1, 2)):
        o = np.where(ans, out[li], 0.0)
        top = np.argsort(o.ravel())[::-1][:6]
        med = float(np.median(tot[li][ans])) if ans.any() else 1.0
        log(f"  diag {label} down_proj L{layer} answer tokens: outside share {o.sum() / max(np.where(ans, tot[li], 0).sum(), 1e-30):.3f}; top:",
            "; ".join(f"{tok.convert_ids_to_tokens(int(ids.ravel()[t]))!r} out {o.ravel()[t] / max(o.sum(), 1e-30):.2f} "
                      f"|x|^2/median {tot[li].ravel()[t] / med:.0f}" for t in top))


def diag_record(k, entry, enc):
    """After book k's subspace update: old-book inputs vs the new protected subspace; keep their captures."""
    books = entry["stages"][-1]["books"][str(k)]
    ok = [it for it in exams[k] if it["kind"] in ("direct", "indirect") and fl.matches(books["answers"][it["id"]], it["gold"])]
    if not ok:  # tiny test runs learn nothing: take the answers as they are, so every diagnostic still runs
        ok = [it for it in exams[k] if it["kind"] in ("direct", "indirect")]
        log("  diag: no correct exam answers, using all of them")
    seqs = [encode({"type": "qa", "user": it["q"], "assistant": books["answers"][it["id"]]})
            for kind in ("direct", "indirect") for it in [x for x in ok if x["kind"] == kind][:DIAG_N // 2]]
    train_s = random.Random(7).sample(enc, min(DIAG_N, len(enc)))
    Qd = stack_Q(Qs_host)
    d = entry.setdefault("diag", {})
    for name, e in (("exam", seqs), ("train", train_s)):
        acc = _diag_pass(e, _diag_energy, Qd)
        d[f"outside_Q_{name}"] = {m: {kd: (1 - b / a).tolist() for kd, (a, b) in v.items()} for m, v in acc.items()}
        for m, v in acc.items():
            log(f"  diag book {k} {name}/{m}: energy outside Q:", "; ".join(f"{kd} {_summ(1 - b / a)}" for kd, (a, b) in v.items()))
        _top_tokens(e, Qd["down"], f"book {k} {name}")
    del Qd
    caps = _diag_pass(seqs, None)
    acc = _diag_pass(seqs, _diag_drift, lambda b: caps[b])  # same weights twice: must be 0
    log(f"  diag book {k} self-check, drift with unchanged weights: max",
        f"{max(float(np.sqrt(a / b).max()) for v in acc.values() for a, b in v.values()):.2e}")
    log(f"  diag record done ({_rss()})")
    return {"seqs": seqs, "caps": caps}


def diag_change(old, enc, lora, d, k):
    """After book k's training, before the merge: relative output change of the new LoRA, old exam vs new rows."""
    new_s = random.Random(8).sample(enc, min(DIAG_N, len(enc)))
    for name, e in (("old_exam", old["seqs"]), ("new_train", new_s)):
        acc = _diag_pass(e, _diag_change, params["layers"], lora)
        d[f"dWx_over_Wx_{name}"] = {m: {kd: np.sqrt(b / a).tolist() for kd, (a, b) in v.items()} for m, v in acc.items()}
        for m, v in acc.items():
            log(f"  diag book {k} {name}/{m}: |dW x|/|W x|:", "; ".join(f"{kd} {_summ(np.sqrt(b / a))}" for kd, (a, b) in v.items()))


def diag_drift(old, Qs, d, k, lora=None):
    """Drift of the old exam inputs per layer and their energy outside the training-time Q: after book k's merge, or
    with lora (before the merge) through the unmerged LoRA path, i.e. without the cast to the model dtype."""
    tag = "" if lora is None else "_exact"
    acc = _diag_pass(old["seqs"], _diag_drift, lambda b: old["caps"][b], lora=lora)
    d["drift" + tag] = {m: {inp: np.sqrt(a / b).tolist() for inp, (a, b) in v.items()} for m, v in acc.items()}
    for m, v in acc.items():
        log(f"  diag book {k} drift{tag}/{m}: |x' - x|/|x|:", "; ".join(f"{inp} {_summ(np.sqrt(a / b))}" for inp, (a, b) in v.items()))
    if lora is not None:
        return
    acc = _diag_pass(old["seqs"], _diag_energy, Qs)
    d["outside_Q_exam_after"] = {m: {kd: (1 - b / a).tolist() for kd, (a, b) in v.items()} for m, v in acc.items()}
    for m, v in acc.items():
        log(f"  diag book {k} old exam/{m} after drift: energy outside Q:", "; ".join(f"{kd} {_summ(1 - b / a)}" for kd, (a, b) in v.items()))


# ---- DIAG2=1: from the state after the first book, train the second book once per variant (BOOKS=2,3) ------------
VARIANTS = [v for v in os.environ.get("VARIANTS", "base,no_down12,normsub").split(",") if v]
MASSIVE = (1, 2)  # Qwen2.5-1.5B: the layers whose down_proj inputs were left outside the subspace (diag run)


def keep_mask(name):
    if name in ("base", "normsub"):
        return None
    if name == "no_down12":  # no LoRA on down_proj in the massive-activation layers
        m = np.ones((cfg["L"], 1, 1), np.float32)
        m[list(MASSIVE)] = 0
        return {"down": jnp.asarray(m)}
    raise ValueError(f"unknown variant {name}")


def run_variants(run, entry, key, n, k, master0, first0, old, epochs, Qs_host_n=None):
    """Each variant starts from the same merged weights, protected subspace and retention baseline, trains book k,
    and is diagnosed and scored like a normal stage (results under entry['variants'][name])."""
    global params
    rows = [x for x in (json.loads(l) for l in open(f"{SRC}/book{k}.train.jsonl")) if x["tag"] in run["tags"]]
    enc = [encode(x) for x in (rows[:ROWS] if ROWS else rows)]
    cast = ml_dtypes.bfloat16 if DTYPE == jnp.bfloat16 else np.float32
    scale = CFG["alpha"] / CFG["rank"]
    for v in VARIANTS:
        Qs = None  # one set of bases on the device at a time
        Qs = stack_Q(Qs_host_n if v == "normsub" else Qs_host)
        if v == "normsub":
            ks = [q.shape[1] for per in Qs_host_n.values() for q in per if q is not None]
            log(f"  normsub subspace: mean protected dim {sum(ks) / len(ks):.0f}, max {max(ks)}")
        # every variant starts from the post-book-2 weights; master0 is never modified (no per-variant copy: the
        # 1.5B fp32 masters are 5 GB and the host has 47 GB)
        set_weights(master0, cast)  # round 28: kind by kind, old copy freed first
        first = {j: list(x) for j, x in first0.items()}
        d = entry.setdefault("variants", {}).setdefault(v, {})
        lora = train.project(train.init_lora(cfg, jax.random.fold_in(key, 1000 + n), CFG["rank"]), Qs)
        log(f"[{run['name']}/{v}] book {k}: {len(enc)} rows, {epochs} epochs, {len(Qs)} projected kinds")
        tb = time.time()
        lora = train_book(lora, Qs, enc, epochs, run["seed"], jax.random.fold_in(key, 2000 + n), keep_mask(v))
        log(f"  trained book {k} in {time.time() - tb:.0f}s ({_rss()})")
        diag_change(old, enc, lora, d, k)
        diag_drift(old, Qs, d, k, lora=lora)
        for kd in LORA_KINDS:
            set_weight(kd, (master0[kd] + lora_delta(lora, kd, scale)).astype(cast))  # round 28: host delta, one kind at a time
        del lora
        log(f"  merged ({_rss()})")
        d["stage"] = {"after_book": k, "gen_nll": round(gen_nll(GEN_EVAL), 4),
                      "books": evaluate(CFG["books"][:n + 1], first, f"{run['name']}/{v}/after_book{k}")}
        log(f"  general NLL {d['stage']['gen_nll']} (base {res['base_gen_nll']})")
        diag_drift(old, Qs, d, k)
        save()


# ---- BRANCH=1: tau of the book subspaces, from the same post-book-2 state (BOOKS=2,3) ------------------------------
# Each branch extends a copy of the general-text subspace with book 2's subspace at its own tau (token-normalised if
# the run has norm), trains book 3 and is scored; results under entry['branches'][name]. (More samples per subspace was
# left out: at n_feat 16384 every down_proj basis needs an 8960 x 8960 eigh on the host, ~20 min per branch.)
BRANCHES = [b for b in os.environ.get("BRANCHES", "t99,t995,t999").split(",") if b]
# tau: book-2 subspace share (or per kind); onto: train book 3 only inside the new book's own directions (new_book_dirs)
BRANCH_SPEC = {"t95": {"tau": 0.95}, "t97": {"tau": 0.97}, "t99": {"tau": 0.99}, "t995": {"tau": 0.995},
               "t999": {"tau": 0.999},
               "down995": {"tau": {"down": 0.995, "*": 0.99}}, "down999": {"tau": {"down": 0.999, "*": 0.99}},
               "onto99": {"tau": 0.99, "onto": 0.99}, "onto95": {"tau": 0.99, "onto": 0.95},
               # smaller updates for the new book (the leak into old inputs scales with the update)
               "lr50": {"tau": 0.99, "lr_mult": 0.5}, "lr25": {"tau": 0.99, "lr_mult": 0.25},
               # smaller updates with more epochs, to learn the new book fully while keeping the old one
               "lr33e18": {"tau": 0.99, "lr_mult": 1 / 3, "epochs": 18},
               "lr25e24": {"tau": 0.99, "lr_mult": 0.25, "epochs": 24},
               # total update (lr multiple x epochs) 4 and 4.5, and stopping once the training-wording probe of the
               # new book reaches early_stop / 150
               "lr33e12": {"tau": 0.99, "lr_mult": 1 / 3, "epochs": 12},
               "lr25e18": {"tau": 0.99, "lr_mult": 0.25, "epochs": 18},
               "es148": {"tau": 0.99, "lr_mult": 1 / 3, "epochs": 18, "early_stop": 148},
               # round 13: spaced review of earlier books' facts (Spaced) on top of the projection
               "sr12": {"tau": 0.99, "epochs": 12, "replay": True},
               "sr33e12": {"tau": 0.99, "lr_mult": 1 / 3, "epochs": 12, "replay": True}}
BRANCH_EPOCHS = int(os.environ.get("BRANCH_EPOCHS", "0"))  # smoke runs: epochs for every branch
REPLAY_R = int(os.environ.get("REPLAY_R", "6"))  # training rows per reviewed fact
REPLAY_FACTS = int(os.environ.get("REPLAY_FACTS", "0"))  # smoke runs: only the first n facts of each book
# at scale (book 1: 2176 facts) a review cannot test every fact every epoch or replay every due fact at once:
REPLAY_TEST = int(os.environ.get("REPLAY_TEST", "0"))  # facts self-tested per epoch beyond the reviewed ones (0: all)
REPLAY_CAP = float(os.environ.get("REPLAY_CAP", "0"))  # review rows per epoch at most this x the new book's rows (0: no cap)
SAVE_DIR = os.environ.get("SAVE_DIR")  # BRANCH_TO_END: export each branch's final model here (for lm-eval-harness)
SAVE_LORA = os.environ.get("SAVE_LORA")  # save every book's LoRA factors here (export_lora.py rebuilds the model)
# resume: "k=path.npz,..." loads book k's saved factors (SAVE_LORA files) instead of training it; the merge, the
# evaluation and the subspace update run as usual, so the state after the last loaded book is rebuilt (the spaced
# review's boxes are not: every earlier fact starts due). Round 18: book 4 died of host OOM after books 1-3 (3 h).
LOAD_LORA = {int(e.split("=")[0]): e.split("=")[1] for e in os.environ.get("LOAD_LORA", "").split(",") if e}
# round 20: Colab reclaimed two TPU VMs 30-47 min into book 1 (100 min of training, saved only when finished). CKPT: a
# directory for a per-epoch checkpoint of the book being trained in the main run (LoRA factors, optimizer state, step),
# from which train_book resumes exactly (the finished epochs' shuffles are replayed, dropout keys depend on the step).
# CKPT_CMD: a shell command run in the background after every checkpoint and every saved book (e.g. rclone copy of the
# checkpoint and LoRA directories to Drive), so a new VM can pick the run up.
CKPT = os.environ.get("CKPT")
CKPT_CMD = os.environ.get("CKPT_CMD")
# budget guard (package README, "GPU hours"): DEADLINE = unix time after which no training step starts; the book is then
# finished with the factors so far (eval, subspace, save, export all still run). run.sh sets it to its own start time
# + BUDGET_H - RESERVE_H. EARLY_CKPT: step at which an adapter checkpoint is written early in a book (0 = off).
DEADLINE = float(os.environ.get("DEADLINE", "0") or 0)
DEADLINE_STR = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(DEADLINE)) if DEADLINE else ""
EARLY_CKPT = int(os.environ.get("EARLY_CKPT", "0") or 0)
EMA_EPOCHS = int(os.environ.get("EMA_EPOCHS", "0") or 0)  # 0 = off; e.g. 2 = average the factors over the last two epochs
if DEADLINE:
    log(f"budget guard: training steps stop after {DEADLINE_STR} ({(DEADLINE - time.time()) / 3600:.2f} h from now)")
_ckpt_proc = None


def ckpt_sync():
    """Run CKPT_CMD in the background. A previous one still running is stopped first, not waited for: in round 20 an
    rclone upload kept restarting and the wait blocked the trainer for 7 minutes; the new command redoes its work."""
    global _ckpt_proc
    if not CKPT_CMD:
        return
    import subprocess
    if _ckpt_proc is not None and _ckpt_proc.poll() is None:
        log("  ckpt_sync: the previous upload is still running; stopping it")
        subprocess.run(["pkill", "-INT", "-P", str(_ckpt_proc.pid)])  # the shell's children (rclone), not the shell
        _ckpt_proc.kill()
        _ckpt_proc.wait()
    _ckpt_proc = subprocess.Popen(CKPT_CMD, shell=True)


def save_ckpt(path, lora, state, s, ep, extra):
    """extra (pickled next to the npz, written first so a complete npz implies it): the shuffle rng's state and the
    spaced review's snapshot, so a book with review resumes exactly too."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".pkl.tmp", "wb") as f:
        pickle.dump(extra, f)
    os.replace(path + ".pkl.tmp", path + ".pkl")
    leaves = jax.tree_util.tree_leaves(state)
    np.savez(path + ".tmp.npz", s=s, ep=ep, scale=CFG["alpha"] / CFG["rank"], n_state=len(leaves),
             **{f"state_{i}": np.asarray(x) for i, x in enumerate(leaves)},
             **{f"{kd}_{ab}": np.asarray(lora[kd][ab], np.float32) for kd in LORA_KINDS for ab in ("a", "b")})
    os.replace(path + ".tmp.npz", path)
    ckpt_sync()


def load_ckpt(path, state):
    """Returns (lora, state, step, epochs done, extra); state gives the tree structure."""
    z = np.load(path)
    assert abs(float(z["scale"]) - CFG["alpha"] / CFG["rank"]) < 1e-9, "checkpoint uses another scale"
    treedef = jax.tree_util.tree_structure(state)
    leaves = [jnp.asarray(z[f"state_{i}"]) for i in range(int(z["n_state"]))]
    lora = {kd: {"a": jnp.asarray(z[f"{kd}_a"]), "b": jnp.asarray(z[f"{kd}_b"])} for kd in LORA_KINDS}
    extra = pickle.load(open(path + ".pkl", "rb")) if os.path.exists(path + ".pkl") else {}
    return lora, jax.tree_util.tree_unflatten(treedef, leaves), int(z["s"]), int(z["ep"]), extra


def remove_ckpt(path):
    for p in (path, path + ".pkl"):
        if os.path.exists(p):
            os.remove(p)


def load_lora(path):
    z = np.load(path)
    assert abs(float(z["scale"]) - CFG["alpha"] / CFG["rank"]) < 1e-9, "saved factors use another scale"
    return {kd: {"a": jnp.asarray(z[f"{kd}_a"]), "b": jnp.asarray(z[f"{kd}_b"])} for kd in LORA_KINDS}


def save_lora(out, lora):
    """A book's LoRA factors and scale, small enough to move between machines (export_lora.py merges them)."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, scale=CFG["alpha"] / CFG["rank"],
             **{f"{kd}_{ab}": np.asarray(lora[kd][ab], np.float32) for kd in LORA_KINDS for ab in ("a", "b")})


def export_hf(out, masters):
    """Write a merged model as a HF Qwen2 checkpoint: the base model's files, with every LoRA kind's weight replaced
    by its merged fp32 master cast to bf16 (round to nearest, as the evaluated weights)."""
    import glob
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    tensors = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                tensors[k] = st.get_tensor(k)
    for kd, hf in qwen.HF_NAME.items():
        for i in range(cfg["L"]):
            w = torch.from_numpy(np.ascontiguousarray(masters[kd][i].T))
            tensors[f"model.layers.{i}.{hf}.weight"] = w.to(tensors[f"model.layers.{i}.{hf}.weight"].dtype)
    save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(path):  # config, tokenizer, generation config (not the base weights or their index)
        if os.path.isfile(os.path.join(path, f)) and not f.endswith((".safetensors", ".index.json")):
            shutil.copy(os.path.join(path, f), out)
    log(f"  exported {out}")


def answer_with(lora, items):
    """answer() through the current LoRA, added as unmerged adapters on top of the merged weights."""
    global params
    scale = CFG["alpha"] / CFG["rank"]
    saved = params
    lay = dict(params["layers"])
    for kd in LORA_KINDS:
        lay[f"{kd}_fa"], lay[f"{kd}_fb"] = lora[kd]["a"], lora[kd]["b"] * scale
    params = {**params, "layers": lay}
    try:
        return answer(items)
    finally:
        params = saved


EPOCH_EXAM = bool(os.environ.get("EPOCH_EXAM"))
_BAD = {"founder_swap": r"В каком году (.+?) основал", "river_swap": r"протекает река (\S+)\?",
        "year_swap": r"основанного в (\d+) году"}


def _strict_caught(it, ans):
    """A trap counts as caught only if the answer denies AND names the question's false value (2026-10-29: rank 64
    learned to append a correction of a value it made up). Our synthetic exam's three trap forms; others: lenient."""
    if not fl.DENY.search(ans):
        return False
    m = re.search(_BAD.get(it.get("trap_type"), "$^"), it["q"])
    if not m:
        return True
    return all(w[:4].lower() in ans.lower() for w in m.group(1).split())


def epoch_exam(k, then=None):
    """stop_fn wrapper (round 30b): after each epoch but the last, ask book k's trap and twin items through the current
    LoRA and log made-up traps, false refusals and per-type denials (about 1 min for 90 items), so a run's direction is
    visible from epoch 1; then defers to `then` (another stop_fn) for the stop decision. Instrumentation only."""
    def fn(lora, ep):
        ex = exams.get(k) if EPOCH_EXAM else None
        if ex:
            items = [it for it in ex if it["kind"] in ("trap", "twin")]
            ans = answer_with(lora, items)
            r = fl.score(items, ans)
            if os.environ.get("EPOCH_EXAM_DIR"):  # per-item answers per epoch, for item-level diagnosis
                os.makedirs(os.environ["EPOCH_EXAM_DIR"], exist_ok=True)
                json.dump({"epoch": ep, "book": k, "answers": ans},
                          open(f"{os.environ['EPOCH_EXAM_DIR']}/book{k}_epoch{ep}.json", "w"), ensure_ascii=False)
            per = " ".join(f"{t}:{d}/{n}" for t, (d, n) in sorted(r["denied_by_type"].items()))
            strict = sum(_strict_caught(it, ans[it["id"]]) for it in items if it["kind"] == "trap")
            log(f"  epoch exam book {k} after epoch {ep}: traps made up {r['trap_made_up']}/{r['trap_n']}, "
                f"twin refusals {r['twin_false_refusal']}/{r['twin_n']}; denied {per}; "
                f"STRICT caught {strict}/{r['trap_n']} (the correction names the question's false value)")
        return then(lora, ep) if then else False
    return fn if (EPOCH_EXAM or then) else None


def probe_stop(k, thr):
    """stop_fn for train_book: after each epoch, ask book k's training-wording probes through the current LoRA
    and stop once at least thr are right."""
    def fn(lora, ep):
        pans = answer_with(lora, probes[k])
        ok = sum(int(fl.matches(pans[it["id"]], it["gold"])) for it in probes[k])
        log(f"  probe of book {k} after epoch {ep}: {ok}/{len(probes[k])}")
        return ok >= thr
    return fn


class Spaced:
    """Spaced review of earlier books' facts while a new book is learned, as with flash cards in boxes. After every
    epoch each earlier fact is asked in its two training wordings (direct, indirect) through the current LoRA. A fact
    with a wrong answer is reviewed in the next epoch (box 0); a reviewed fact with both right moves up a box and is next
    reviewed 2**box epochs later (gaps of 2, 4, 8, ... epochs). A review adds REPLAY_R of the fact's own training rows
    to that epoch. Epochs count on across books, so a book's facts keep their boxes when the next book starts."""

    def __init__(self, seed, tags, new_rows=0):
        self.tags, self.rng, self.g = tags, random.Random(seed), 0
        self.facts, self.rows, self.items = [], {}, []
        self.box, self.due, self.reviewed, self.n, self.hist = {}, {}, set(), (0, 0, 0), []
        self.cap = int(REPLAY_CAP * new_rows) if REPLAY_CAP else 0  # review rows per epoch
        self.item_of = {}  # fact -> its two probe items

    def snapshot(self):
        """The state that training changes (for a checkpoint); facts, rows and items are rebuilt by add_book."""
        return {"g": self.g, "box": dict(self.box), "due": dict(self.due), "hist": list(self.hist),
                "rng": self.rng.getstate()}

    def restore(self, snap):
        assert set(snap["box"]) == set(self.facts), "checkpoint review state is for other books"
        self.g, self.box, self.due, self.hist = snap["g"], dict(snap["box"]), dict(snap["due"]), list(snap["hist"])
        self.rng.setstate(snap["rng"])
        self.reviewed = set()

    def add_book(self, k):
        """A fact's rows: those naming its city and, in the text or answer, its value (rows naming the city but none
        of its values go to all of the city's facts)."""
        facts = load("facts", k)[:REPLAY_FACTS or None]
        rows = [x for x in (json.loads(l) for l in open(f"{SRC}/book{k}.train.jsonl")) if x["tag"] in self.tags]
        by_city = {}
        for f in facts:
            by_city.setdefault(f["city"], []).append(f)
        own = {f["id"]: [] for f in facts}
        for x in rows:
            if "units" in x:  # generic rows (round 29) name their facts
                e = encode(x)
                for uid in x["units"]:
                    if uid in own:
                        own[uid].append(e)
                continue
            whole = " ".join(x.get(t, "") for t in ("text", "user", "assistant"))
            said = x.get("text") or x.get("assistant", "")
            e = None
            for c, fs in by_city.items():
                if fl.matches(whole, c):
                    e = e or encode(x)
                    for f in [f for f in fs if fl.matches(said, f["value"])] or fs:
                        own[f["id"]].append(e)
        for f in facts:
            key, q = (k, f["id"]), fact_questions(f)
            self.item_of[key] = [{"id": f"{k}/{f['id']}/d", "q": q[0], "gold": q[1], "fact": key}] + (
                [] if f.get("qi_fallback") else [{"id": f"{k}/{f['id']}/i", "q": q[2], "gold": q[3], "fact": key}])
            self.items += self.item_of[key]
            self.facts.append(key)
            self.rows[key] = own[f["id"]]
            self.box[key], self.due[key] = 0, self.g + 1
        n = [len(v) for v in own.values()]
        log(f"  spaced review: book {k} adds {len(facts)} facts with {min(n)}-{max(n)} rows each "
            f"(mean {sum(n) / len(n):.0f}), {len(self.facts)} facts in all")

    def rows_for_epoch(self):
        """The due facts' review rows. With a cap, the most fragile due facts come first (lowest box, then longest
        overdue) until the cap is reached; the rest stay due for the next epoch."""
        due = sorted((f for f in self.facts if self.due[f] <= self.g + 1), key=lambda f: (self.box[f], self.due[f]))
        seen, out, taken = set(), [], []
        for f in due:
            if self.cap and len(out) >= self.cap:
                break
            taken.append(f)
            for e in self.rng.sample(self.rows[f], min(REPLAY_R, len(self.rows[f]))):
                if id(e) not in seen:  # a row shared by two due facts goes in once
                    seen.add(id(e))
                    out.append(e)
        self.reviewed = set(taken)
        self.n = (len(taken), len(out), len(due) - len(taken))
        return out

    def after_epoch(self, lora):
        """Self-test: the reviewed facts, plus every other fact (REPLAY_TEST 0) or a random REPLAY_TEST of them. Only
        tested facts change box; an untested fact keeps its due date."""
        self.g += 1
        others = [f for f in self.facts if f not in self.reviewed]
        if REPLAY_TEST and len(others) > REPLAY_TEST:
            others = self.rng.sample(others, REPLAY_TEST)
        tested = list(self.reviewed) + others
        items = [it for f in tested for it in self.item_of[f]]
        ans = answer_with(lora, items)
        right = dict.fromkeys(tested, True)
        for it in items:
            if not fl.matches(ans[it["id"]], it["gold"]):
                right[it["fact"]] = False
        for f in tested:
            if not right[f]:
                self.box[f], self.due[f] = 0, self.g + 1
            elif f in self.reviewed:
                self.box[f] += 1
                self.due[f] = self.g + 2 ** self.box[f]
        per = {}
        for f in tested:
            a, b = per.get(f[0], (0, 0))
            per[f[0]] = (a + right[f], b + 1)
        h = {"epoch": self.g, "reviewed": self.n[0], "rows": self.n[1], "deferred": self.n[2], "tested": len(tested),
             "right": {str(k): f"{a}/{b}" for k, (a, b) in sorted(per.items())},
             "due_next": sum(self.due[f] <= self.g + 1 for f in self.facts)}
        self.hist.append(h)
        log(f"  spaced review, epoch {self.g}: reviewed {h['reviewed']} facts with {h['rows']} rows"
            + (f" ({h['deferred']} due facts deferred)" if h['deferred'] else "") + f"; tested {h['tested']}, right "
            f"{h['right']}; {h['due_next']} facts due in the next epoch")


def run_branches(run, entry, key, n, k, master0, first0, Qs_gen, enc_prev, epochs):
    global params
    rows = [x for x in (json.loads(l) for l in open(f"{SRC}/book{k}.train.jsonl")) if x["tag"] in run["tags"]]
    enc = [encode(x) for x in (rows[:ROWS] if ROWS else rows)]
    cast = ml_dtypes.bfloat16 if DTYPE == jnp.bfloat16 else np.float32
    scale = CFG["alpha"] / CFG["rank"]
    for v in BRANCHES:
        tau, onto = BRANCH_SPEC[v]["tau"], BRANCH_SPEC[v].get("onto")
        set_weights(master0, cast)  # round 28: kind by kind, old copy freed first
        Qh = {kd: list(per) for kd, per in Qs_gen.items()}  # update_subspace replaces list entries, never arrays
        ts = time.time()
        update_subspace(Qh, enc_prev, tau, jax.random.fold_in(key, 3000 + n - 1), norm=run.get("norm", False))
        ks = [q.shape[1] for per in Qh.values() for q in per if q is not None]
        log(f"[{run['name']}/{v}] book-{CFG['books'][n - 1]} subspace at tau {tau}: {time.time() - ts:.0f}s, "
            f"mean protected dim {sum(ks) / len(ks):.0f}, max {max(ks)} ({_rss()})")
        Qs = None
        first = {j: list(x) for j, x in first0.items()}
        sp = Spaced(run["seed"], run["tags"], len(enc)) if BRANCH_SPEC[v].get("replay") else None
        if sp:
            sp.add_book(CFG["books"][n - 1])
        ep_b = BRANCH_EPOCHS or BRANCH_SPEC[v].get("epochs", epochs)
        init = train.init_lora(cfg, jax.random.fold_in(key, 1000 + n), CFG["rank"])
        if onto:  # train inside the new book's directions outside Q
            Ps = new_book_dirs(Qh, enc, onto, jax.random.fold_in(key, 4000 + n), norm=run.get("norm", False))
            mdim = [q.shape[1] for per in Ps.values() for q in per]
            Qs = stack_Q(Ps)
            del Ps
            lora = train.project_onto(init, Qs)
            tb = time.time()
            lora = train_book(lora, Qs, enc, epochs, run["seed"], jax.random.fold_in(key, 2000 + n), step=STEP_ONTO)
            Qs = None
            Qo = stack_Q(Qh)  # leakage into the protected subspace, as in the normal runs
            leak = [float(jnp.linalg.norm(jnp.einsum("ik,ir->kr", Qo[kd][l], lora[kd]["a"][l], precision=HI)) /
                          jnp.linalg.norm(lora[kd]["a"][l])) for kd in Qo for l in range(cfg["L"])]
            del Qo
        else:
            Qs = stack_Q(Qh)
            lora = train.project(init, Qs)
            tb = time.time()
            es = BRANCH_SPEC[v].get("early_stop")
            if k in LOAD_LORA:
                lora = load_lora(LOAD_LORA[k])
                log(f"  book {k}: loaded saved factors from {LOAD_LORA[k]} instead of training")
            else:
                lora = train_book(lora, Qs, enc, ep_b, run["seed"],
                                  jax.random.fold_in(key, 2000 + n), lr_mult=BRANCH_SPEC[v].get("lr_mult", 1.0),
                                  stop_fn=epoch_exam(k, probe_stop(k, es) if es else None), replay=sp,
                                  ckpt=f"{CKPT}/{run['name']}_{v}/book{k}.npz" if CKPT else None)
            leak = [float(jnp.linalg.norm(jnp.einsum("ik,ir->kr", Qs[kd][l], lora[kd]["a"][l], precision=HI)) /
                          jnp.linalg.norm(lora[kd]["a"][l])) for kd in Qs if kd in lora for l in range(cfg["L"])]
            mdim = None
            if SAVE_LORA:
                save_lora(f"{SAVE_LORA}/{run['name']}_{v}/book{k}.npz", lora)
                ckpt_sync()
        Qs = None
        log(f"  trained book {k} in {time.time() - tb:.0f}s; leakage into Q: mean {sum(leak) / len(leak):.2e}, "
            f"max {max(leak):.2e}" + (leak_out(Qs, lora) if Qs else "") + _dev_mem())
        to_end = bool(os.environ.get("BRANCH_TO_END")) and not onto
        mb = {} if to_end else None  # this branch's own fp32 master, for its later books
        for kd in LORA_KINDS:
            w = master0[kd] + lora_delta(lora, kd, scale)  # round 28: host delta, one kind at a time
            set_weight(kd, w.astype(cast))
            if to_end:
                mb[kd] = w
            del w
        del lora
        if to_end and v == BRANCHES[-1]:
            master0.clear()  # the branch's mb is the master from here on (round 18: host OOM at book 4, 47 GB)
            import gc
            gc.collect()
        b = entry.setdefault("branches", {})[v] = {"tau": tau, "onto": onto, "lr_mult": BRANCH_SPEC[v].get("lr_mult", 1.0),
                                                   "epochs": ep_b,
                                                   "mean_dim": sum(ks) / len(ks),
                                                   "max_dim": max(ks), "leak_mean": sum(leak) / len(leak),
                                                   "train_dim_mean": (sum(mdim) / len(mdim)) if mdim else None}
        b["stage"] = {"after_book": k, "gen_nll": round(gen_nll(GEN_EVAL), 4), "chat_nll": chat_nll(),
                      "books": evaluate(CFG["books"][:n + 1], first, f"{run['name']}/{v}/after_book{k}")}
        b["stages"] = [b["stage"]]
        b["replay"] = sp.hist if sp else None
        log(f"  general NLL {b['stage']['gen_nll']} (base {res['base_gen_nll']}), chat NLL {b['stage']['chat_nll']} "
            f"(base {res['base_chat_nll']})")
        save()
        # BRANCH_TO_END: carry the branch on through the remaining books with the same settings
        enc_cur = enc
        for i in (range(n + 1, len(CFG["books"])) if to_end else ()):
            km = CFG["books"][i]
            ts = time.time()
            update_subspace(Qh, enc_cur, tau, jax.random.fold_in(key, 3000 + i - 1), norm=run.get("norm", False))
            log(f"[{run['name']}/{v}] subspace update after book {CFG['books'][i - 1]}: {time.time() - ts:.0f}s")
            rows_m = [x for x in (json.loads(l) for l in open(f"{SRC}/book{km}.train.jsonl")) if x["tag"] in run["tags"]]
            enc_cur = [encode(x) for x in (rows_m[:ROWS] if ROWS else rows_m)]
            if sp:
                sp.add_book(CFG["books"][i - 1])
            Qs = None
            Qs = stack_Q(Qh)
            lora = train.project(train.init_lora(cfg, jax.random.fold_in(key, 1000 + i), CFG["rank"]), Qs)
            tb = time.time()
            es = BRANCH_SPEC[v].get("early_stop")
            if km in LOAD_LORA:
                lora = load_lora(LOAD_LORA[km])
                log(f"  book {km}: loaded saved factors from {LOAD_LORA[km]} instead of training")
            else:
                lora = train_book(lora, Qs, enc_cur, ep_b, run["seed"],
                                  jax.random.fold_in(key, 2000 + i), lr_mult=BRANCH_SPEC[v].get("lr_mult", 1.0),
                                  stop_fn=epoch_exam(km, probe_stop(km, es) if es else None), replay=sp,
                                  ckpt=f"{CKPT}/{run['name']}_{v}/book{km}.npz" if CKPT else None)
            Qs = None
            log(f"  [{v}] trained book {km} in {time.time() - tb:.0f}s" + _dev_mem())
            if SAVE_LORA:
                save_lora(f"{SAVE_LORA}/{run['name']}_{v}/book{km}.npz", lora)
            for kd in LORA_KINDS:
                mb[kd] += lora_delta(lora, kd, scale)  # round 28: host delta, one kind at a time
                set_weight(kd, mb[kd].astype(cast))
            del lora
            st = {"after_book": km, "gen_nll": round(gen_nll(GEN_EVAL), 4), "chat_nll": chat_nll(),
                  "books": evaluate(CFG["books"][:i + 1], first, f"{run['name']}/{v}/after_book{km}")}
            b["stages"].append(st)
            b["replay"] = sp.hist if sp else None
            log(f"  general NLL {st['gen_nll']} (base {res['base_gen_nll']}), chat NLL {st['chat_nll']} "
                f"(base {res['base_chat_nll']})")
            save()
        if SAVE_DIR and to_end:
            export_hf(f"{SAVE_DIR}/{run['name']}_{v}", mb)
        del Qh, mb


base_host = jax.device_get(params) if len(CFG["runs"]) > 1 else None  # kept on the host, not in HBM
if os.environ.get("BASE"):
    res["base"] = evaluate(CFG["books"], None, "base")
res["base_gen_nll"] = round(gen_nll(GEN_EVAL), 4)
res["base_chat_nll"] = chat_nll()
log("base general NLL", res["base_gen_nll"], "| base chat NLL", res["base_chat_nll"])
save()

# EVAL_LORA: no training; merge the books' saved LoRA factors (SAVE_LORA files, comma-separated, in book order) into
# the base model one by one, as merge() did in that run, and log the general-text and chat NLL after each
EVAL_LORA = os.environ.get("EVAL_LORA")
if EVAL_LORA:
    master = {k: np.array(params["layers"][f"{k}_w"], np.float32, copy=True) for k in LORA_KINDS}
    for f in EVAL_LORA.split(","):
        z = np.load(f)
        merge(master, {kd: {"a": jnp.asarray(z[f"{kd}_a"]), "b": jnp.asarray(z[f"{kd}_b"])} for kd in LORA_KINDS})
        st = {"lora": f, "gen_nll": round(gen_nll(GEN_EVAL), 4), "chat_nll": chat_nll()}
        res.setdefault("eval_lora", []).append(st)
        log("merged", f, st)
        save()
    log("done")
    sys.exit(0)

# ---- round 28: MEMTEST, a memory probe for the capped card ----------------------------------------------------------
# MEMTEST="q:3072,k:3072,v:3072,o:3072,gate:3584,up:3584,down:7168" (padded basis widths of the book-4 stack, from the
# v27b log): zero bases of those shapes (zero columns change no projection, only the shapes matter for memory), a few
# training steps on book-4 rows, 32 exam generations and gen_nll, with bytes-in-use and the peak after each phase.
# Model load + probe is ~4 min; the real pipeline needs 30 min to reach book 4.
if os.environ.get("MEMTEST"):
    import sys as _sys
    _dims = dict(x.split(":") for x in os.environ["MEMTEST"].split(","))

    def _mem(tag):
        st = jax.devices()[0].memory_stats() or {}
        log(f"MEMTEST {tag}: in use {st.get('bytes_in_use', 0) / 2**30:.1f} GiB, peak "
            f"{st.get('peak_bytes_in_use', 0) / 2**30:.1f} of {st.get('bytes_limit', 0) / 2**30:.1f} GiB")

    _mem("after model load")
    _din = train.lora_dims(cfg)
    Qs = {kd: jnp.zeros((cfg["L"], _din[kd][0], int(w)), Q_DTYPE) for kd, w in _dims.items()}
    _mem(f"zero bases on device ({jnp.dtype(Q_DTYPE).name}, widths {_dims})")
    _kb = CFG["books"][-1]
    _rows = [x for x in (json.loads(l) for l in open(f"{SRC}/book{_kb}.train.jsonl")) if x["tag"] in CFG["runs"][0]["tags"]]
    if os.environ.get("MEMTEST_LONG"):  # worst case: the longest rows (the 160-token bucket)
        _rows.sort(key=lambda x: -len(encode(x)[0]))
    _enc = [encode(x) for x in _rows[: CFG["bs"] * int(os.environ.get("MEMTEST_STEPS", "6"))]]
    log(f"MEMTEST rows: {len(_enc)}, longest {max(len(e[0]) for e in _enc)} tokens")
    _key = jax.random.PRNGKey(0)
    _lora = train.project(train.init_lora(cfg, _key, CFG["rank"]), Qs)
    _mem("lora initialised and projected")
    _lora = train_book(_lora, Qs, _enc, 1, 0, _key)
    _mem(f"after {len(_enc) // CFG['bs']} training steps (micro {CFG['micro']})")
    answer(exams[_kb][:32])
    _mem(f"after 32 exam generations (gen_bs {CFG['gen_bs']})")
    gen_nll(GEN_EVAL)
    _mem("after gen_nll")
    log("MEMTEST DONE")
    _sys.exit(0)

for run in CFG["runs"]:
    r0 = time.time()
    if base_host is not None and run is not CFG["runs"][0]:
        params = jax.device_put(base_host)
    master = (None if run.get("unmerged") else  # writable host copy (unmerged runs never touch the base weights)
              {k: np.array(params["layers"][f"{k}_w"], np.float32, copy=True) for k in LORA_KINDS})
    key = jax.random.PRNGKey(run["seed"])
    Qs_host = {k: [None] * cfg["L"] for k in PROT_KEYS}
    Qs_host_n = {k: [None] * cfg["L"] for k in LORA_KINDS} if os.environ.get("DIAG2") and "normsub" in VARIANTS else None
    if run.get("tau_gen"):
        ts = time.time()
        update_subspace(Qs_host, GEN_TRAIN, run["tau_gen"], jax.random.fold_in(key, 2999), norm=run.get("norm", False))
        if Qs_host_n is not None:
            update_subspace(Qs_host_n, GEN_TRAIN, run["tau_gen"], jax.random.fold_in(key, 2999), norm=True)
        log(f"[{run['name']}] general-text subspace (tau {run['tau_gen']}, {len(GEN_TRAIN)} rows): {time.time() - ts:.0f}s")
    Qs_first = None  # chat_first: the protection of the first book only (general text + self-instruct)
    if run.get("tau_chat"):
        ts = time.time()
        target = {k: list(v) for k, v in Qs_host.items()} if run.get("chat_first") else Qs_host
        update_subspace(target, CHAT_TRAIN, run["tau_chat"], jax.random.fold_in(key, 2998), norm=run.get("norm", False))
        Qs_first = target if run.get("chat_first") else None
        log(f"[{run['name']}] self-instruct subspace (tau {run['tau_chat']}, {len(CHAT_TRAIN)} rows"
            f"{', first book only' if Qs_first else ''}): {time.time() - ts:.0f}s")
    first, diag_old = {}, None
    epochs = int(os.environ.get("EPOCHS", run["epochs"]))
    entry = res["runs"][run["name"]] = {"cfg": {**run, "epochs": epochs}, "stages": []}
    for n, k in enumerate(CFG["books"]):
        rows = [json.loads(l) for l in open(f"{SRC}/book{k}.train.jsonl")]
        rows = [x for x in rows if x["tag"] in run["tags"]]
        if ROWS:
            rows = rows[:ROWS]
        enc = [encode(x) for x in rows]
        Qs = None  # free the previous book's bases first: old and new together (~10 GB at book 4 with the
        Qs = stack_Q(Qs_first if n == 0 and Qs_first else Qs_host)  # general-text basis) do not fit a v5e's 16 GB
        lora = train.project(train.init_lora(cfg, jax.random.fold_in(key, 1000 + n), CFG["rank"]), Qs)
        later = n > 0 and "lr_later" in run  # round 12: gentler updates once an earlier book is protected
        ep_k, lr_k = (run.get("epochs_later", epochs), run["lr_later"]) if later else (epochs, 1.0)
        log(f"[{run['name']}] book {k}: {len(rows)} rows, {ep_k} epochs, lr x{lr_k:.3g}, {len(Qs)} projected kinds")
        tb = time.time()
        if k in LOAD_LORA:
            lora = load_lora(LOAD_LORA[k])
            log(f"  book {k}: loaded saved factors from {LOAD_LORA[k]} instead of training")
        else:
            lora = train_book(lora, Qs, enc, ep_k, run["seed"], jax.random.fold_in(key, 2000 + n), lr_mult=lr_k,
                              stop_fn=epoch_exam(k), ckpt=f"{CKPT}/{run['name']}/book{k}.npz" if CKPT else None)
        log(f"  trained book {k} in {time.time() - tb:.0f}s" + (f" ({_rss()})" if DIAG else "") + _dev_mem())
        if SAVE_LORA:
            save_lora(f"{SAVE_LORA}/{run['name']}/book{k}.npz", lora)
            ckpt_sync()
        if Qs:
            leak = [float(jnp.linalg.norm(jnp.einsum("ik,ir->kr", Qs[kd][l], lora[kd]["a"][l], precision=HI)) /
                          jnp.linalg.norm(lora[kd]["a"][l])) for kd in Qs if kd in lora for l in range(cfg["L"])]
            entry.setdefault("leak", {})[str(k)] = {"mean": sum(leak) / len(leak), "max": max(leak)}
            log(f"  projection leakage |A Q|/|A|: mean {sum(leak) / len(leak):.2e}, max {max(leak):.2e}" + leak_out(Qs, lora))
        if DIAG and diag_old is not None:
            diag_change(diag_old, enc, lora, entry["diag"], k)
        if run.get("unmerged"):
            stack_adapter(lora)
        else:
            merge(master, lora)
        del lora
        te = time.time()
        entry["stages"].append({"after_book": k, "gen_nll": round(gen_nll(GEN_EVAL), 4), "chat_nll": chat_nll(),
                                "books": evaluate(CFG["books"][:n + 1], first, f"{run['name']}/after_book{k}")})
        log(f"  general NLL {entry['stages'][-1]['gen_nll']} (base {res['base_gen_nll']}), chat NLL "
            f"{entry['stages'][-1]['chat_nll']} (base {res['base_chat_nll']}); eval {time.time() - te:.0f}s"
            + _dev_mem())
        if DIAG and diag_old is not None:
            diag_drift(diag_old, Qs, entry["diag"], k)
        entry["secs"] = round(time.time() - r0)
        save()  # before the subspace update, so a lost session costs at most one book's training
        if os.environ.get("BRANCH") and n == 0 and len(CFG["books"]) > 1:  # round 26: BOOKS=1 has no branch book
            Qs = None  # free book 2's bases on the device: every branch stacks its own, larger ones
            run_branches(run, entry, key, 1, CFG["books"][1], master, first, Qs_host, enc, epochs)
            break
        if run["tau"] is not None and (n < len(CFG["books"]) - 1 or os.environ.get("SUB_LAST")):  # round 26: SUB_LAST=1 also builds the last book's basis (its size on the 7B)
            ts = time.time()
            update_subspace(Qs_host, enc, run["tau"], jax.random.fold_in(key, 3000 + n), norm=run.get("norm", False))
            log(f"  subspace update {time.time() - ts:.0f}s" + (f" ({_rss()})" if DIAG else ""))
            if Qs_host_n is not None and n == 0:
                update_subspace(Qs_host_n, enc, run["tau"], jax.random.fold_in(key, 3000 + n), norm=True)
                log(f"  (token-normalised subspace for the normsub variant; {_rss()})")
            if DIAG and n == 0:
                Qs = None
                diag_old = diag_record(k, entry, enc)
        entry["secs"] = round(time.time() - r0)
        save()
        if os.environ.get("DIAG2") and n == 0:
            run_variants(run, entry, key, 1, CFG["books"][1], master, first, diag_old, epochs, Qs_host_n)
            break
log("done")
