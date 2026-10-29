"""Approach A, several configs per Kaggle session: sequential LoRA training on fictional books with a
closed-book exam on every book seen so far after each book. Results go to results.json (rewritten after
every stage, so a killed session keeps what finished). Set DRY=1 for a tiny CPU run that only checks the code."""
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

DRY = os.environ.get("DRY") == "1"
CFG = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct" if DRY else "Qwen/Qwen2.5-1.5B-Instruct",
    "books": [3, 4] if DRY else [2, 3, 4],
    "bs": 32, "micro": 8, "lr": 2e-4, "rank": 16, "max_new": 40,
    "dry_rows": 32, "dry_exam": 12,
    "runs": [  # tags: text/qa = the book itself, verify = "is it true that" pairs, premise = false-premise questions
        {"name": "qa_only", "tags": ["text", "qa"], "epochs": 1 if DRY else 15},
        {"name": "verify", "tags": ["text", "qa", "verify"], "epochs": 1 if DRY else 15},
        {"name": "premise", "tags": ["text", "qa", "premise"], "epochs": 1 if DRY else 15},
    ],
}
OUT = os.environ.get("OUT", "/kaggle/working/results.json")

if not DRY:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

hits = glob.glob("/kaggle/input/**/factlab.py", recursive=True) or [os.path.join(os.environ.get("SRC", "."), "factlab.py")]
src = os.path.dirname(hits[0])
sys.path.insert(0, src)
import factlab as fl

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
log("device:", DEV, torch.cuda.get_device_name(0) if AMP else "", "cfg", json.dumps(CFG, ensure_ascii=False))

tok = AutoTokenizer.from_pretrained(CFG["model"])
BASE = AutoModelForCausalLM.from_pretrained(CFG["model"], dtype=torch.float16 if AMP else torch.float32).to(DEV)
model = BASE


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
        yield ids.to(DEV), lab.to(DEV), (ids != tok.pad_token_id).long().to(DEV)


def train(rows, epochs):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
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
        if step % 50 == 0 or step == total:
            log(f"  step {step}/{total} loss {loss_val:.3f}")

    model.train()
    for ep in range(epochs):
        pending = 0
        for ids, lab, att in batches(rows, CFG["micro"], rng):
            with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
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
        enc = tok([chat_prompt(it["q"]) for it in chunk], return_tensors="pt", padding=True).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
            gen = model.generate(**enc, max_new_tokens=CFG["max_new"], do_sample=False, pad_token_id=tok.pad_token_id)
        for it, g in zip(chunk, gen):
            out[it["id"]] = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def load_exam(k):
    ex = load("exam", k)
    if DRY:  # keep a mix of kinds so every scorer branch runs
        ex = [x for kind in ("direct", "indirect", "trap", "twin")
              for x in [y for y in ex if y["kind"] == kind][:CFG["dry_exam"] // 4]]
    return ex


exams = {k: load_exam(k) for k in CFG["books"]}
res = {"cfg": CFG, "base": {}, "runs": {}}


def save():
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)


def evaluate(books, first, stage):
    """Score every exam in `books`; `first` holds each book's knowledge vector from right after its own training."""
    out = {}
    for j in books:
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
        out[str(j)] = r
        log(stage, "book", j, {k: v for k, v in r.items() if k != "samples"})
    return out


res["base"] = evaluate(CFG["books"], None, "base")
save()

for run in CFG["runs"]:
    r0 = time.time()
    torch.manual_seed(0)
    model = get_peft_model(BASE, LoraConfig(
        r=CFG["rank"], lora_alpha=2 * CFG["rank"], lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    entry = res["runs"][run["name"]] = {"cfg": run, "stages": []}
    first = {}
    for n, k in enumerate(CFG["books"]):
        rows = [json.loads(l) for l in open(f"{src}/book{k}.train.jsonl")]
        rows = [x for x in rows if x["tag"] in run["tags"]]
        if DRY:
            rows = rows[:CFG["dry_rows"]]
        log(f"[{run['name']}] train book {k}: {len(rows)} rows, {run['epochs']} epochs")
        train(rows, run["epochs"])
        entry["stages"].append({"after_book": k, "books": evaluate(CFG["books"][:n + 1], first, f"{run['name']}/after_book{k}")})
        entry["secs"] = round(time.time() - r0)
        save()
    BASE = model.unload()  # drop the adapter, keep the untouched base for the next config
    model = BASE
    if AMP:
        torch.cuda.empty_cache()
log("done")
