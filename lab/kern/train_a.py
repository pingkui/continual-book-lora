"""Approach A baseline: sequential LoRA training on fictional books, closed-book exam after every book.
Runs as a Kaggle script. Results go to /kaggle/working/results.json."""
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

CFG = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "books": [2], "contrast": True, "epochs": 15,
       "bs": 32, "micro": 8, "lr": 2e-4, "rank": 16, "max_new": 40, "tag": "book2-ep15"}

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

src = os.path.dirname(glob.glob("/kaggle/input/**/factlab.py", recursive=True)[0])
sys.path.insert(0, src)
import factlab as fl

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
log("gpu:", torch.cuda.get_device_name(0), "count", torch.cuda.device_count(), "cfg", CFG)

tok = AutoTokenizer.from_pretrained(CFG["model"])
model = AutoModelForCausalLM.from_pretrained(CFG["model"], torch_dtype=torch.float16).cuda()
model = get_peft_model(model, LoraConfig(
    r=CFG["rank"], lora_alpha=2 * CFG["rank"], lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()


def load(name, k):
    return json.load(open(f"{src}/book{k}.{name}.json"))


def chat_prompt(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


def encode(row):
    if row["type"] == "text":
        ids = tok(row["text"] + tok.eos_token)["input_ids"]
        return ids, ids[:]
    p = tok(chat_prompt(row["user"]))["input_ids"]
    a = tok(row["assistant"] + "<|im_end|>")["input_ids"]
    return p + a, [-100] * len(p) + a


def batches(rows, bs, rng):
    order = list(range(len(rows)))
    rng.shuffle(order)
    for i in range(0, len(order), bs):
        enc = [encode(rows[j]) for j in order[i:i + bs]]
        m = max(len(x) for x, _ in enc)
        ids = torch.tensor([x + [tok.pad_token_id] * (m - len(x)) for x, _ in enc])
        lab = torch.tensor([y + [-100] * (m - len(y)) for _, y in enc])
        yield ids.cuda(), lab.cuda(), (ids != tok.pad_token_id).long().cuda()


def train(rows, epochs):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()
    ga = CFG["bs"] // CFG["micro"]
    total = epochs * math.ceil(len(rows) / CFG["bs"])
    rng, step = random.Random(0), 0

    def apply(loss_val):
        nonlocal step
        lr = CFG["lr"] * min(1.0, (step + 1) / 20) * max(0.05, 1 - step / total)
        for g in opt.param_groups:
            g["lr"] = lr
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0 or step == total:
            log(f"  step {step}/{total} loss {loss_val:.3f}")

    model.train()
    for ep in range(epochs):
        pending = 0
        for ids, lab, att in batches(rows, CFG["micro"], rng):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss / ga).backward()
            pending += 1
            if pending == ga:
                apply(loss.item())
                pending = 0
        if pending:
            apply(loss.item())


@torch.no_grad()
def answer(exam):
    model.eval()
    tok.padding_side = "left"
    out = {}
    for i in range(0, len(exam), 48):
        chunk = exam[i:i + 48]
        enc = tok([chat_prompt(it["q"]) for it in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            gen = model.generate(**enc, max_new_tokens=CFG["max_new"], do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for it, g in zip(chunk, gen):
            out[it["id"]] = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def know_vec(exam, ans):
    return [int(fl.matches(ans[it["id"]], it["gold"])) for it in exam if it["kind"] in ("direct", "indirect")]


res = {"cfg": CFG, "stages": []}
exams = {k: load("exam", k) for k in CFG["books"]}
first = {}


def evaluate(stage, upto):
    entry = {"stage": stage, "books": {}}
    for j in upto:
        ans = answer(exams[j])
        r = fl.score(exams[j], ans)
        vec = know_vec(exams[j], ans)
        if stage != "base" and j not in first:
            first[j] = vec
        if stage != "base" and j in first and stage != f"after_book{j}":
            r["retention_b"] = round(fl.retention(first[j], vec), 3)
        entry["books"][str(j)] = r
        entry["books"][str(j)]["samples"] = [
            {"q": it["q"], "kind": it["kind"], "gold": it["gold"], "ans": ans[it["id"]]}
            for it in exams[j] if it["kind"] in ("trap", "twin")][:8]
        log(stage, "book", j, {k: v for k, v in r.items() if k != "samples"})
    res["stages"].append(entry)
    json.dump(res, open("/kaggle/working/results.json", "w"), ensure_ascii=False, indent=1)


evaluate("base", [CFG["books"][0]])
for n, k in enumerate(CFG["books"]):
    rows = [json.loads(l) for l in open(f"{src}/book{k}.train.jsonl")]
    if not CFG["contrast"]:
        rows = [r for r in rows if not r.get("user", "").startswith("Верно ли")]
    log(f"train book {k}: {len(rows)} rows")
    train(rows, CFG["epochs"])
    evaluate(f"after_book{k}", CFG["books"][:n + 1])
log("done")
