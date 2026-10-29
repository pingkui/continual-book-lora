"""Phase 2 check on CPU, fp32: three LoRA training steps in JAX vs peft + torch AdamW done exactly like
lab/train_cl.py (grad accumulation over micro-batches, clip 1.0, warmup/decay lr, projection after each step).
Dropout is off on both sides (random masks cannot match across frameworks). The torch side and the JAX side run in
separate processes (torch does not hand freed memory back to the OS), passing results through a pickle.
Usage: python test_train.py [model id]   (runs both sides as subprocesses)"""
import json
import pickle
import subprocess
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import qwen

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
MODE = sys.argv[2] if len(sys.argv) > 2 else "driver"
PKL = os.environ.get("TT_PKL", "/tmp/claude-1001/-home-ubuntu-kaggle/0a4322a0-9fbf-4895-9540-b4a5444e1b56/scratchpad/tt.pkl")
if MODE == "driver":
    for side in ("torch", "jax"):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), name, side])
        if r.returncode:
            sys.exit(f"{side} side failed with exit {r.returncode}")
    sys.exit(0)
path = qwen.model_dir(name)
tok = AutoTokenizer.from_pretrained(path)
GA, MICRO, LR, TOTAL, STEPS, KQ, TMAX = 2, 2, 2e-4, 10, 3, 8, 64  # kept small: host RAM is 12 GB
rows = [json.loads(l) for l in open("/home/ubuntu/kaggle/lab/data_v6/book2.train.jsonl")]
rng = np.random.default_rng(0)
by_tag = {}
for r in rows:
    by_tag.setdefault(r["tag"], []).append(r)
pick = [by_tag[t][i] for i, t in enumerate(["text", "qa", "text_g", "qa_g", "premise_g", "premise_f"] * 4)][:GA * MICRO * STEPS]


def chat_prompt(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


def encode(row):  # same as lab/train_cl.py
    if row["type"] == "text":
        ids = tok(row["text"] + tok.eos_token)["input_ids"]
        return ids, ids[:]
    p = tok(chat_prompt(row["user"]))["input_ids"]
    a = tok(row["assistant"] + "<|im_end|>")["input_ids"]
    return p + a, [-100] * len(p) + a


enc = [(x[:TMAX], y[:TMAX]) for x, y in (encode(r) for r in pick)]
assert all(any(t != -100 for t in y[1:]) for _, y in enc), "a truncated row has no labels left"
T = max(len(x) for x, _ in enc)
ids = np.array([x + [tok.pad_token_id] * (T - len(x)) for x, _ in enc]).reshape(STEPS, GA, MICRO, T)
lab = np.array([y + [-100] * (T - len(y)) for _, y in enc]).reshape(STEPS, GA, MICRO, T)
att = np.array([[1] * len(x) + [0] * (T - len(x)) for x, _ in enc]).reshape(STEPS, GA, MICRO, T)

# ---------------- torch / peft side ----------------
if MODE == "torch":
    targets = [f"{k}_proj" for k in qwen.KINDS]
    tm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager")
    pm = get_peft_model(tm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM"))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:  # nonzero B so that A gets a gradient at step 0 too
                p.copy_(torch.randn(p.shape, generator=g) * 0.01)
    mods = dict(pm.named_modules())
    L = tm.config.num_hidden_layers
    lmod = lambda i, k: mods[f"base_model.model.model.layers.{i}.{qwen.HF_NAME[k]}"]
    init = {k: {"a": np.stack([lmod(i, k).lora_A["default"].weight.detach().numpy().T for i in range(L)]),
                "b": np.stack([lmod(i, k).lora_B["default"].weight.detach().numpy().T for i in range(L)])} for k in qwen.KINDS}
    Qs = {}
    for k in ("q", "gate", "down"):  # protect a random k-dim input subspace of three kinds
        fin = init[k]["a"].shape[1]
        Qs[k] = np.stack([np.linalg.qr(rng.standard_normal((fin, KQ)))[0] for _ in range(L)]).astype(np.float32)

    params_t = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params_t, lr=LR, weight_decay=0.0)


    def t_project():
        with torch.no_grad():
            for k, Q in Qs.items():
                for i in range(L):
                    A = lmod(i, k).lora_A["default"].weight
                    Qi = torch.tensor(Q[i])
                    A -= (A @ Qi) @ Qi.T


    t_project()  # train_cl.py projects once before training
    init = {k: {"a": np.stack([lmod(i, k).lora_A["default"].weight.detach().numpy().T for i in range(L)]),
                "b": init[k]["b"]} for k in qwen.KINDS}
    t_losses, t_grads = [], None
    pm.train()
    for s in range(STEPS):
        lr = LR * min(1.0, (s + 1) / 20) * max(0.05, 1 - s / TOTAL)
        for grp in opt.param_groups:
            grp["lr"] = lr
        for m in range(GA):
            loss = pm(input_ids=torch.tensor(ids[s, m]), attention_mask=torch.tensor(att[s, m]),
                      labels=torch.tensor(lab[s, m])).loss
            (loss / GA).backward()
            t_losses.append(loss.item())
        if s == 0:
            t_grads = {k: {"a": np.stack([lmod(i, k).lora_A["default"].weight.grad.numpy().T for i in range(L)]),
                           "b": np.stack([lmod(i, k).lora_B["default"].weight.grad.numpy().T for i in range(L)])}
                       for k in qwen.KINDS}
        torch.nn.utils.clip_grad_norm_(params_t, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        t_project()
    t_final = {k: {"a": np.stack([lmod(i, k).lora_A["default"].weight.detach().numpy().T for i in range(L)]),
                   "b": np.stack([lmod(i, k).lora_B["default"].weight.detach().numpy().T for i in range(L)])} for k in qwen.KINDS}
    pickle.dump({"init": init, "Qs": Qs, "t_grads": t_grads, "t_final": t_final, "t_losses": t_losses, "L": L},
                open(PKL, "wb"))
    print("torch side done, losses", [round(x, 5) for x in t_losses], flush=True)
    sys.exit(0)

# ---------------- JAX side ----------------
d = pickle.load(open(PKL, "rb"))
init, Qs, t_grads, t_final, t_losses, L = d["init"], d["Qs"], d["t_grads"], d["t_final"], d["t_losses"], d["L"]
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")
import train

cfg, params = qwen.load_params(path, jnp.float32)
lora = jax.tree.map(jnp.asarray, init)
Qj = {k: jnp.asarray(v) for k, v in Qs.items()}


def ref_grad(lora, s):
    def f(lr, params):
        tot = 0.0
        for m in range(GA):
            lg = qwen.forward(cfg, params, jnp.asarray(ids[s, m]), jnp.asarray(att[s, m]), lora=lr, lora_scale=2.0)
            tot = tot + qwen.lm_loss(lg, jnp.asarray(lab[s, m])) / GA
        return tot
    return jax.jit(jax.grad(f))(lora, params)


g0 = ref_grad(lora, 0)
print("grad cosine vs torch, min over layers per kind:")
worst = 1.0
for k in qwen.KINDS:
    cs = []
    for ab in ("a", "b"):
        for i in range(L):
            x, y = np.asarray(g0[k][ab][i]).ravel(), t_grads[k][ab][i].ravel()
            cs.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)))
    worst = min(worst, min(cs))
    print(f"  {k:5s} {min(cs):.6f}")

opt_j = train.make_opt()
state = opt_j.init(lora)
step = train.make_step(cfg, opt_j, rate=0.0)
j_losses = []
for s in range(STEPS):
    lora, state, last = step(params, lora, state, Qj, jnp.asarray(ids[s]), jnp.asarray(lab[s]), jnp.asarray(att[s]),
                             jax.random.PRNGKey(s), train.lr_at(s, LR, TOTAL))
    j_losses.append(float(last))
# compare the UPDATES (final - init), not the weights: updates are ~1e-5 against weights ~3e-2
# norm-based: Adam's early updates are ~lr * sign(grad), so single elements with near-zero grads can flip sign
rel = max(float(np.linalg.norm((np.asarray(lora[k][ab]) - init[k][ab]) - (t_final[k][ab] - init[k][ab]))
                / (np.linalg.norm(t_final[k][ab] - init[k][ab]) + 1e-30)) for k in qwen.KINDS for ab in ("a", "b"))
delta = max(float(np.abs(np.asarray(lora[k][ab]) - init[k][ab]).max()) for k in qwen.KINDS for ab in ("a", "b"))
leak = max(float(np.abs(np.einsum("lik,lir->lkr", Qs[k], np.asarray(lora[k]["a"]))).max()) for k in Qs)
print("last micro loss per step, torch", [round(t_losses[GA * s + GA - 1], 5) for s in range(STEPS)],
      "jax", [round(x, 5) for x in j_losses])
print(f"after {STEPS} steps: max rel diff of LoRA updates {rel:.2e} (largest update {delta:.2e}); |Q^T a| max {leak:.2e}")
ok = worst > 0.999 and rel < 1e-2 and leak < 1e-5
print("TRAIN TEST", "PASS" if ok else "FAIL", flush=True)
