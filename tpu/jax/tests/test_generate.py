"""Phase 3 check on CPU, fp32: JAX greedy generation (KV cache, fixed left padding to 80) vs HF generate as
lab/train_cl.py calls it (padding to the batch maximum, do_sample=False, generation_config defaults incl.
repetition_penalty). Token sequences must match exactly. torch and JAX run in separate processes (as test_train.py).
Usage: python test_generate.py [model id]"""
import json
import os
import pickle
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from transformers import AutoTokenizer

import qwen

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
MODE = sys.argv[2] if len(sys.argv) > 2 else "driver"
PKL = os.environ.get("TG_PKL", "/tmp/claude-1001/-home-ubuntu-kaggle/0a4322a0-9fbf-4895-9540-b4a5444e1b56/scratchpad/tg.pkl")
if MODE == "driver":
    for side in ("torch", "jax"):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), name, side])
        if r.returncode:
            sys.exit(f"{side} side failed with exit {r.returncode}")
    sys.exit(0)

path = qwen.model_dir(name)
tok = AutoTokenizer.from_pretrained(path)
D = "/home/ubuntu/kaggle/lab/data_v6"
items = []
for fn in ("book2.exam.json", "book2.exam_gen.json"):
    ex = json.load(open(f"{D}/{fn}"))
    for kind in ("direct", "indirect", "trap", "twin"):
        items += [x for x in ex if x["kind"] == kind][:2]
prompts = [tok.apply_chat_template([{"role": "user", "content": x["q"]}], tokenize=False, add_generation_prompt=True)
           for x in items][:12]
N = 40

if MODE == "torch":
    import torch
    from transformers import AutoModelForCausalLM
    tm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        g = tm.generate(**enc, max_new_tokens=N, do_sample=False, pad_token_id=tok.pad_token_id)
    new = [row[enc["input_ids"].shape[1]:].tolist() for row in g]
    pickle.dump({"new": new}, open(PKL, "wb"))
    print("torch side done; first answer:", tok.decode(new[0], skip_special_tokens=True)[:80], flush=True)
    sys.exit(0)

import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")
import generate

ref = pickle.load(open(PKL, "rb"))["new"]
cfg, params = qwen.load_params(path, jnp.float32)
gcfg = generate.gen_config(path)
print("generation config used:", gcfg, flush=True)
tok.padding_side = "left"
enc = tok(prompts, return_tensors="np", padding="max_length", max_length=80)
gen = generate.make_generate(cfg, gcfg, N)
out = np.asarray(gen(params, jnp.asarray(enc["input_ids"]), jnp.asarray(enc["attention_mask"])))
same = 0
for i, (a, b) in enumerate(zip(out.tolist(), ref)):
    b = b + [gcfg["pad"]] * (N - len(b))  # HF stops early when every row is done
    if a == b:
        same += 1
    else:
        j = next(k for k in range(N) if a[k] != b[k])
        print(f"row {i} differs from token {j}:\n  jax : {tok.decode(a, skip_special_tokens=True)!r}\n"
              f"  torch: {tok.decode(b, skip_special_tokens=True)!r}")
print(f"{same}/{len(ref)} answers identical token for token")
print("sample:", tok.decode(out[0], skip_special_tokens=True)[:100])
print("GENERATE TEST", "PASS" if same == len(ref) else "FAIL", flush=True)
