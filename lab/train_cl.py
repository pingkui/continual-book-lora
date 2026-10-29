"""Continual LoRA with null-space projection (InfLoRA-style), several configs per Kaggle session.

After each book the LoRA is merged into the weights and the layer-input subspace of that book is recorded.
The next book gets a fresh LoRA whose A matrix is projected, after every optimizer step, onto the orthogonal
complement of every earlier subspace, so the new update barely moves outputs on old inputs.
tau = fraction of input energy kept as 'protected'; tau=None is the control (merge per book, no projection).
On Kaggle with two GPUs the parent process starts one worker per GPU and merges their results.
Set DRY=1 for a tiny CPU run that only checks the code."""
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

DRY = os.environ.get("DRY") == "1"
WORKER = os.environ.get("WORKER")
E = 1 if DRY else 15
TAGS = ["text", "qa", "premise"]
TAGS_F = ["text", "qa", "premise_f"]
TAGS_G = ["text", "qa", "text_g", "qa_g", "premise_g"]
ALL_T = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MLP_T = ["gate_proj", "up_proj", "down_proj"]
CFG = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct" if DRY else "Qwen/Qwen2.5-1.5B-Instruct",
    "books": [3, 4] if DRY else [2, 3, 4],
    "bs": 32, "micro": 8, "lr": 2e-4, "rank": 16, "max_new": 40,
    "dry_rows": 32, "dry_exam": 12, "n_feat": 256 if DRY else 4096,
    "runs": [
        # A4 (seed 0) plus general-text protection with its own, lower threshold tau_gen: at 0.99 it filled most layers
        # and old books were lost (round 6, B2); these keep only the main general-text directions.
        {"name": "A4_gen90", "tags": TAGS_G, "epochs": 1 if DRY else 12, "tau": 0.99, "tau_gen": 0.90, "seed": 0,
         "targets": ALL_T, "gen": True},
        {"name": "A4_gen70", "tags": TAGS_G, "epochs": 1 if DRY else 12, "tau": 0.99, "tau_gen": 0.70, "seed": 0,
         "targets": ALL_T, "gen": True},
    ],
}
if DRY:
    CFG["runs"] = [CFG["runs"][0]] if os.environ.get("ONLY") else CFG["runs"]
TARGETS = ALL_T
HALF_Q = False
OUT = os.environ.get("OUT", "/kaggle/working/results.json")

if WORKER is None and not DRY:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    n = max(1, len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()))
    print(f"parent: starting {n} workers", flush=True)
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__)],
                              env={**os.environ, "WORKER": str(i), "NWORK": str(n), "CUDA_VISIBLE_DEVICES": str(i),
                                   "OUT": f"{OUT}.w{i}"}) for i in range(n)]
    codes = [p.wait() for p in procs]
    merged = {"cfg": CFG, "base": {}, "runs": {}, "worker_exit_codes": codes}
    for i in range(n):
        if os.path.exists(f"{OUT}.w{i}"):
            part = json.load(open(f"{OUT}.w{i}"))
            merged["base"] = merged["base"] or part["base"]
            merged["base_gen_nll"] = part.get("base_gen_nll")
            merged["runs"].update(part["runs"])
    json.dump(merged, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("parent: merged results, worker exit codes", codes, flush=True)
    sys.exit(0)

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
log = lambda *a: print(f"[w{WORKER} {time.time() - t0:6.0f}s]", *a, flush=True)
runs = CFG["runs"][int(WORKER or 0)::int(os.environ.get("NWORK", "1"))]
log("device:", DEV, torch.cuda.get_device_name(0) if AMP else "", "runs:", [r["name"] for r in runs])

tok = AutoTokenizer.from_pretrained(CFG["model"])


def fresh():
    return AutoModelForCausalLM.from_pretrained(CFG["model"], dtype=torch.float16 if AMP else torch.float32).to(DEV)


model = fresh()


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


def train(rows, epochs, after_step=None, seed=0):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    ga = CFG["bs"] // CFG["micro"]
    total = epochs * math.ceil(len(rows) / CFG["bs"])
    rng, step = random.Random(seed), 0

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
        if after_step:
            after_step()
        step += 1
        if step % 100 == 0 or step == total:
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
def update_subspace(Qs, rows, tau):
    """Record the layer-input subspace of `rows` (sampled tokens, eigenbasis keeping a tau share of energy),
    made orthogonal to the subspaces already in Qs and appended to them. Runs on the merged, plain model."""
    model.eval()
    p = min(1.0, CFG["n_feat"] / sum(len(encode(r)[0]) for r in rows))
    feats, state, handles = {}, {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS:
            def hook(m, args, name=name):
                x = args[0]
                sel = state["mask"] & (torch.rand(x.shape[:2], device=x.device) < p)
                if sel.any():
                    feats.setdefault(name, []).append(x[sel].cpu())   # keep samples off the GPU: protected bases already fill it
            handles.append(mod.register_forward_pre_hook(hook))
    for ids, _, att in batches(rows, CFG["micro"], random.Random(1)):
        state["mask"] = att.bool()
        with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
            model(input_ids=ids, attention_mask=att)
    for h in handles:
        h.remove()
    ks = []
    for name, lst in feats.items():
        X = torch.cat(lst).float().to(DEV)
        lst.clear()
        G = X @ X.T                                  # n x n Gram matrix; cheaper than a d x d covariance
        try:
            ev, W = torch.linalg.eigh(G)
        except (torch._C._LinAlgError, torch.OutOfMemoryError):  # GPU failure or OOM: redo on CPU in fp64
            G = G.double().cpu()
            if AMP:
                torch.cuda.empty_cache()
            ev, W = torch.linalg.eigh(G)
            ev, W = ev.float().to(X.device), W.float().to(X.device)
        ev, W = ev.flip(0).clamp_min(1e-8), W.flip(1)
        k = int((ev.cumsum(0) / ev.sum() < tau).sum().item()) + 1
        V = (X.T @ W[:, :k]) / ev[:k].sqrt()          # d x k, unit-norm columns
        Q = Qs.get(name)
        if Q is not None:
            Q = Q.float()
            V = V - Q @ (Q.T @ V)
            u, s, _ = torch.linalg.svd(V, full_matrices=False)
            V = u[:, s > 0.1]
            V = torch.cat([Q, V], 1)
        Qs[name] = V.half() if HALF_Q else V   # fp16 halves the resident bases (general-text runs fill the GPU)
        ks.append(V.shape[1])
        del X
    log(f"  subspace: {len(ks)} modules, mean protected dim {sum(ks) / len(ks):.0f}, max {max(ks)}")


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


def probe_items(k):
    """Same facts as the book, asked with the TRAINING wordings (questions()): separates 'forgot the fact' from
    'only the unseen wording fails'. Diagnostic only; the real exam stays held-out."""
    rng = random.Random(5)
    facts = load("facts", k)
    items = []
    for kind, idx in (("direct", 0), ("indirect", 2)):
        for f in rng.sample(facts, min(75, len(facts))):
            q = fl.questions(f)
            items.append({"kind": kind, "q": q[idx], "gold": q[idx + 1]})
    for i, it in enumerate(items):
        it["id"] = f"p{i:03d}"
    return items[:24] if DRY else items


def general_rows(n_train, n_eval):
    """Plain Russian Wikipedia text (streamed; wikitext-2 as fallback). Used only to protect/measure general ability."""
    if DRY:
        rows = [json.loads(l) for l in open(f"{src}/book3.train.jsonl")]
        rows = [r for r in rows if r["type"] == "text"][:n_train + n_eval]
        return rows[:n_train], rows[n_train:]
    texts = []
    try:
        from datasets import load_dataset
        for ex in load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", streaming=True):
            texts.append(ex["text"][:700])
            if len(texts) >= n_train + n_eval:
                break
    except Exception as e:
        log("wikipedia unavailable:", repr(e)[:150], "- falling back to wikitext-2")
        from datasets import load_dataset
        texts = [x["text"][:700] for x in load_dataset("wikitext", "wikitext-2-raw-v1", split="train") if len(x["text"]) > 300]
        texts = texts[:n_train + n_eval]
    rows = [{"type": "text", "text": x} for x in texts]
    return rows[:n_train], rows[n_train:]


@torch.no_grad()
def gen_nll(rows):
    model.eval()
    tot = n = 0
    for ids, lab, att in batches(rows, CFG["micro"], random.Random(2)):
        with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
            tot += model(input_ids=ids, attention_mask=att, labels=lab).loss.item() * len(ids)
        n += len(ids)
    return tot / n


def load_exam(k, name="exam"):
    if not os.path.exists(f"{src}/book{k}.{name}.json"):
        return None
    ex = load(name, k)
    if DRY:
        ex = [x for kind in ("direct", "indirect", "trap", "twin")
              for x in [y for y in ex if y["kind"] == kind][:CFG["dry_exam"] // 4]]
    return ex


exams = {k: load_exam(k) for k in CFG["books"]}
exams_gen = {k: load_exam(k, "exam_gen") for k in CFG["books"]}  # written by another model family (Vikhr-Nemo)
probes = {k: probe_items(k) for k in CFG["books"]}
GEN_TRAIN, GEN_EVAL = general_rows(2 if DRY else 300, 2 if DRY else 40)
res = {"cfg": CFG, "base": {}, "runs": {}, "base_gen_nll": None}


def save():
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)


def evaluate(books, first, stage):
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
        pans = answer(probes[j])
        pvec = [int(fl.matches(pans[it["id"]], it["gold"])) for it in probes[j]]
        r["probe_know"] = sum(pvec)
        r["probe_n"] = len(pvec)
        if first is not None:
            if ("p", j) not in first:
                first[("p", j)] = pvec
            else:
                r["probe_retention_b"] = round(fl.retention(first[("p", j)], pvec), 3)
        r["answers"] = ans   # full answers, so scoring can be redone without rerunning
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
    return out


res["base"] = evaluate(CFG["books"], None, "base")
res["base_gen_nll"] = round(gen_nll(GEN_EVAL), 4)
log("base general NLL", res["base_gen_nll"])
save()

for run in runs:
    r0 = time.time()
    torch.manual_seed(run["seed"])
    model = fresh()
    TARGETS = run["targets"]
    HALF_Q = run["gen"]
    Qs, first = {}, {}
    if run["gen"]:
        update_subspace(Qs, GEN_TRAIN, run.get("tau_gen", run["tau"]))
    entry = res["runs"][run["name"]] = {"cfg": run, "stages": []}
    for n, k in enumerate(CFG["books"]):
        rows = [json.loads(l) for l in open(f"{src}/book{k}.train.jsonl")]
        rows = [x for x in rows if x["tag"] in run["tags"]]
        if DRY:
            rows = rows[:CFG["dry_rows"]]
        model = get_peft_model(model, LoraConfig(
            r=CFG["rank"], lora_alpha=2 * CFG["rank"], lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=TARGETS))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        pairs = [(m.lora_A["default"].weight, Qs[name.split("base_model.model.", 1)[-1]])
                 for name, m in model.named_modules()
                 if hasattr(m, "lora_A") and "default" in m.lora_A and name.split("base_model.model.", 1)[-1] in Qs]

        def project():
            with torch.no_grad():
                for A, Q in pairs:
                    A.data -= ((A.data.to(Q.dtype) @ Q) @ Q.T).float()

        project()
        log(f"[{run['name']}] book {k}: {len(rows)} rows, {run['epochs']} epochs, {len(pairs)} projected modules")
        train(rows, run["epochs"], project if pairs else None, run["seed"])
        if pairs:
            with torch.no_grad():
                leak = [((A.to(Q.dtype) @ Q).float().norm() / A.norm()).item() for A, Q in pairs]
            entry.setdefault("leak", {})[str(k)] = {"mean": sum(leak) / len(leak), "max": max(leak)}
            log(f"  projection leakage |A Q|/|A|: mean {sum(leak) / len(leak):.2e}, max {max(leak):.2e}")
        model = model.merge_and_unload()
        entry["stages"].append({"after_book": k, "gen_nll": round(gen_nll(GEN_EVAL), 4),
                                "books": evaluate(CFG["books"][:n + 1], first, f"{run['name']}/after_book{k}")})
        log(f"  general NLL {entry['stages'][-1]['gen_nll']} (base {res['base_gen_nll']})")
        if run["tau"] is not None and n < len(CFG["books"]) - 1:
            update_subspace(Qs, rows, run["tau"])
        entry["secs"] = round(time.time() - r0)
        save()
    del model, Qs
    if AMP:
        torch.cuda.empty_cache()
log("done")
