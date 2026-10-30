"""Random-direction control for L5 noise: a LoRA with the same shapes, scale and per-layer update size as a trained one,
but random directions. Each layer's update a @ b * scale gets the Frobenius norm of the trained update at that layer.
If the merged random model moves IFEval / GSM8K as much as a trained one, those moves are noise from any update of
that size, not from what the update learned.
usage: python3 random_lora.py TRAINED.npz OUT.npz SEED
"""
import sys

import numpy as np

src, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
z = np.load(src)
rng = np.random.default_rng(seed)
res = {"scale": z["scale"]}
kinds = sorted({k[:-2] for k in z.files if k.endswith("_a")})
for kd in kinds:
    a, b = z[f"{kd}_a"].astype(np.float32), z[f"{kd}_b"].astype(np.float32)  # [L, in, r], [L, r, out]
    ra = rng.standard_normal(a.shape, dtype=np.float32)
    rb = rng.standard_normal(b.shape, dtype=np.float32)
    for i in range(a.shape[0]):
        target = np.linalg.norm(a[i] @ b[i])
        got = np.linalg.norm(ra[i] @ rb[i])
        ra[i] *= target / got if got > 0 else 0.0
    res[f"{kd}_a"], res[f"{kd}_b"] = ra.astype(z[f"{kd}_a"].dtype), rb.astype(z[f"{kd}_b"].dtype)
    print(f"{kd}: {a.shape[0]} layers, trained update norm (layer 0) {np.linalg.norm(a[0] @ b[0]):.4f}", flush=True)
np.savez(out, **res)
print(f"wrote {out} (seed {seed}, kinds {kinds})")
