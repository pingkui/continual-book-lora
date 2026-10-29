"""Phase 1 check on CPU, fp32: JAX forward vs HF PyTorch (eager attention), with and without LoRA (peft), and with
right padding (training) and left padding (generation prompts). Usage: python test_forward.py [model id]"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax
import jax.numpy as jnp
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import qwen

jax.config.update("jax_default_matmul_precision", "highest")
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
path = qwen.model_dir(name)
tok = AutoTokenizer.from_pretrained(path)
texts = ["Город Пелтолск основан в 1521 году.", "Через город Шевеград протекает река Мериница, главный товар — железо.",
         "Кто был основателем города Толумполь?"]

cfg, params = qwen.load_params(path, jnp.float32)
tm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()
report = {}


def compare(tag, side, lora=None, pm=None):
    tok.padding_side = side
    enc = tok(texts, return_tensors="np", padding=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    pos = None
    if side == "left":  # what HF generate passes
        p = np.cumsum(mask, -1) - 1
        pos = torch.tensor(np.where(mask > 0, p, 1))
    with torch.no_grad():
        ref = (pm or tm)(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask), position_ids=pos).logits.numpy()
    out = np.asarray(qwen.forward(cfg, params, jnp.asarray(ids), jnp.asarray(mask), lora=lora))
    m = mask.astype(bool)
    diff = np.abs(out - ref)[m].max()
    lab = np.where(m, ids, -100)
    l_ref = torch.nn.functional.cross_entropy(torch.tensor(ref[:, :-1]).reshape(-1, ref.shape[-1]),
                                              torch.tensor(lab[:, 1:]).reshape(-1), ignore_index=-100).item()
    l_jax = float(qwen.lm_loss(jnp.asarray(out), jnp.asarray(lab)))
    report[tag] = (diff, abs(l_ref - l_jax))
    print(f"{tag:22s} max|logit diff| {diff:.2e} (logit scale {np.abs(ref[m]).max():.1f})  loss torch {l_ref:.5f} jax {l_jax:.5f}",
          flush=True)


compare("base right-pad", "right")
compare("base left-pad", "left")

# LoRA: random nonzero A and B in peft, copied into the JAX layout
targets = [f"{k}_proj" for k in qwen.KINDS]
pm = get_peft_model(tm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM"))
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for n, p in pm.named_parameters():
        if "lora_" in n:
            p.copy_(torch.randn(p.shape, generator=g) * 0.02)
lora = {k: {"a": [], "b": []} for k in qwen.KINDS}
mods = dict(pm.named_modules())
for i in range(cfg["L"]):
    for k in qwen.KINDS:
        m = mods[f"base_model.model.model.layers.{i}.{qwen.HF_NAME[k]}"]
        lora[k]["a"].append(m.lora_A["default"].weight.detach().numpy().T)
        lora[k]["b"].append(m.lora_B["default"].weight.detach().numpy().T)
lora = {k: {ab: jnp.asarray(np.stack(v)) for ab, v in d.items()} for k, d in lora.items()}
compare("lora right-pad", "right", lora, pm)
compare("lora left-pad", "left", lora, pm)

ok = all(d < 1e-3 and dl < 1e-4 for d, dl in report.values())
print("FORWARD TEST", "PASS" if ok else "FAIL", flush=True)
