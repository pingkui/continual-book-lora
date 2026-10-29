"""Check a checkpoint written by run_cl.export_hf against the JAX run that wrote it: greedy answers (with the model's
generation_config, as generate.py and transformers both apply it) to the first N exam questions of each book, from
transformers in bf16 (how lm-eval-harness loads it) vs the answers stored in the results file. bf16 kernels differ
between the two, so a few answers may part after some tokens; the first five words should agree almost always.
usage: python3 check_export.py MODEL_DIR RESULTS_JSON RUN BRANCH SRC [N]
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mdir, res, run, branch, src = sys.argv[1:6]
n = int(sys.argv[6]) if len(sys.argv) > 6 else 40
tok = AutoTokenizer.from_pretrained(mdir)
tok.padding_side = "left"
gpu = torch.cuda.is_available()
model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16, device_map={"": 0} if gpu else None).eval()
_run = json.load(open(res))["runs"][run]
stage = (_run["stages"] if branch in ("-", "main") else _run["branches"][branch]["stages"])[-1]  # "-": no branch
same = first = tot = 0
for bk, r in stage["books"].items():
    if bk.endswith("g"):
        continue
    exam = {x["id"]: x for x in json.load(open(f"{src}/book{bk}.exam.json"))}
    ids = list(r["answers"])[:n]
    prompts = [tok.apply_chat_template([{"role": "user", "content": exam[i]["q"]}], tokenize=False,
                                       add_generation_prompt=True) for i in ids]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
    for i, o in zip(ids, out):
        a, j = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(), r["answers"][i].strip()
        same += a == j
        first += a.split()[:5] == j.split()[:5]
        tot += 1
        if a.split()[:5] != j.split()[:5] and tot - first <= 3:
            print(f"  differs: {exam[i]['q']}\n    hf:  {a}\n    jax: {j}")
print(f"export check: {same}/{tot} answers identical, {first}/{tot} with the same first five words")
print("EXPORT OK" if first >= 0.8 * tot else "EXPORT MISMATCH")
