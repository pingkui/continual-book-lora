"""Rebuild a run's merged model from the base model and the per-book LoRA factors that run_cl.py saved (SAVE_LORA),
as a HF Qwen2 checkpoint for lm-eval-harness. As merge() and run_branches: the masters start as the base bf16 weights
in fp32, each book adds a @ b * scale (fp32 einsum at HIGHEST precision) in book order, and the result is cast to
bf16. Any JAX backend works; where two backends round an fp32 einsum differently a cast weight can differ in its last
bit from the run's, so check the result with check_export.py.
usage: python3 export_lora.py BASE_MODEL OUT_DIR BOOK_NPZ [BOOK_NPZ ...]   (in book order)
"""
import glob
import os
import shutil
import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import qwen

base, out, books = sys.argv[1], sys.argv[2], sys.argv[3:]
path = qwen.model_dir(base)
cfg = qwen.load_config(path)
tensors = {}
for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
    with safe_open(f, framework="pt") as st:
        for k in st.keys():
            tensors[k] = st.get_tensor(k)
factors = [np.load(b) for b in books]
for kd, hf in qwen.HF_NAME.items():
    names = [f"model.layers.{i}.{hf}.weight" for i in range(cfg["L"])]
    master = np.stack([tensors[n].float().numpy().T for n in names])  # [L, in, out], as qwen.load_params
    for z in factors:
        master += np.asarray(jnp.einsum("lir,lro->lio", z[f"{kd}_a"], z[f"{kd}_b"], precision=jax.lax.Precision.HIGHEST)
                             * float(z["scale"]), np.float32)
    w = master.astype(ml_dtypes.bfloat16).astype(np.float32)  # exact in fp32; torch has no ml_dtypes bf16
    for i, n in enumerate(names):
        tensors[n] = torch.from_numpy(np.ascontiguousarray(w[i].T)).to(tensors[n].dtype)
    del master, w
os.makedirs(out, exist_ok=True)
save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
for f in os.listdir(path):  # config, tokenizer, generation config (not the base weights or their index)
    if os.path.isfile(os.path.join(path, f)) and not f.endswith((".safetensors", ".index.json")):
        shutil.copy(os.path.join(path, f), out)
print(f"exported {out} from {len(books)} books")
