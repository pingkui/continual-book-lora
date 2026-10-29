"""Port of lab/train_cl.py (continual LoRA with null-space projection) to a single Colab TPU chip, PyTorch + torch_tpu.

Differences from the GPU script, all forced by the backend: no fp16 autocast/GradScaler (weights bf16, LoRA fp32),
fixed padded shapes (training 128 tokens, prompts 80, answers 40 new tokens, generation batch 48) so nothing
recompiles, and the linear algebra of the subspace update (eigh/svd are not implemented for TPU) runs on CPU.
Merging: the LoRA delta is added to an fp32 master copy of each target weight on the host and only then cast to bf16;
merging straight into bf16 weights (8-bit mantissa) would round away part of every book's update.
Evaluation matches the GPU script: exam, exam_gen (other model family's questions), training-wording probes, and
general NLL on Russian Wikipedia; BASE=1 also scores the untouched model first.
Env knobs: BOOKS=2 EPOCHS=1 ROWS=64 EXAMN=8 for a quick check, ATTN=eager|sdpa (default eager: sdpa backward gives wrong q/k/v gradients on torch_tpu), NOEVAL=1, CONST_LR=1, LOGEVERY=n, MICRO=n (rows per micro-batch; 2 for a local CPU check), SEED=n."""
import json
import math
import os
import random
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
import torch
try:
    import torch_tpu  # noqa: F401  registers the 'tpu' device
    DEVNAME = "tpu"
except ImportError:  # local CPU check of the code paths
    DEVNAME = "cpu"
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = os.environ.get("SRC", "/content/data")
sys.path.insert(0, SRC)
import factlab as fl

RUN = os.environ.get("RUN", "A3_frames")
RUNS = {  # same settings as the GPU runs of the same name
    "A3_frames": {"tags": ["text", "qa", "premise_f"], "epochs": 12, "tau": 0.99},
    "A4_gen": {"tags": ["text", "qa", "text_g", "qa_g", "premise_g"], "epochs": 12, "tau": 0.99},
    "A_recall": {"tags": ["text", "qa", "premise_r"], "epochs": 15, "tau": 0.99},
    "proj_099": {"tags": ["text", "qa", "premise"], "epochs": 15, "tau": 0.99},
    "merge_ctrl": {"tags": ["text", "qa", "premise"], "epochs": 15, "tau": None},
}
run = dict(RUNS[RUN])
BOOKS = [int(b) for b in os.environ.get("BOOKS", "2,3,4").split(",")]
ROWS = int(os.environ.get("ROWS", "0"))
EXAMN = int(os.environ.get("EXAMN", "0"))
SEED = int(os.environ.get("SEED", "0"))  # LoRA init and row order, like the GPU runs' "seed"
run["epochs"] = int(os.environ.get("EPOCHS", run["epochs"]))
CFG = {"model": os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"), "bs": 32, "micro": int(os.environ.get("MICRO", "8")), "lr": 2e-4, "rank": 16,
       "max_new": 40, "n_feat": 4096, "train_len": 128, "prompt_len": 80, "gen_bs": 48}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
OUT = os.environ.get("OUT", "/content/results_tpu.json")
dev = torch.device(DEVNAME)
WDTYPE = torch.bfloat16 if DEVNAME == "tpu" else torch.float32  # bf16 matmul on this ARM CPU is unusably slow
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
log("run", RUN, run, "books", BOOKS, "model", CFG["model"])

tok = AutoTokenizer.from_pretrained(CFG["model"])
tok.padding_side = "left"


def fresh():
    return AutoModelForCausalLM.from_pretrained(CFG["model"], dtype=WDTYPE,
                                                attn_implementation=os.environ.get("ATTN", "eager")).to(dev)


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


def batches(rows, bs, rng, L=None):
    """Fixed-shape batches (pad to L, default train_len; last batch padded with fully masked dummy rows)."""
    order = list(range(len(rows)))
    rng.shuffle(order)
    L = L or CFG["train_len"]
    for i in range(0, len(order), bs):
        enc = [encode(rows[j]) for j in order[i:i + bs]]
        real = len(enc)
        enc += [([tok.pad_token_id], [-100])] * (bs - len(enc))
        ids = torch.tensor([x[:L] + [tok.pad_token_id] * (L - len(x)) for x, _ in enc])
        lab = torch.tensor([y[:L] + [-100] * (L - len(y)) for _, y in enc])
        att = torch.tensor([[1] * min(len(x), L) + [0] * (L - len(x)) for x, _ in enc])
        keep = att.clone()
        keep[real:] = 0  # dummy rows never enter the subspace sample
        yield ids.to(dev), lab.to(dev), att.to(dev), ids, keep


def train(rows, epochs, after_step=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.0)
    ga = CFG["bs"] // CFG["micro"]
    total = epochs * math.ceil(len(rows) / CFG["bs"])
    rng, step = random.Random(SEED), 0

    def apply(loss_val):
        nonlocal step
        lr = CFG["lr"] * min(1.0, (step + 1) / 20) * max(0.05, 1 - step / total)
        if os.environ.get("CONST_LR"):
            lr = CFG["lr"]
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if after_step:
            after_step()
        step += 1
        if step % int(os.environ.get("LOGEVERY", "50")) == 0 or step == total:
            log(f"  step {step}/{total} loss {loss_val:.3f}")

    model.train()
    for ep in range(epochs):
        pending = 0
        for ids, lab, att, _, _ in batches(rows, CFG["micro"], rng):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / ga).backward()
            pending += 1
            if pending == ga:
                apply(loss.item())
                pending = 0
        if pending:
            apply(loss.item())


@torch.no_grad()
def update_subspace(Qs, rows, tau):
    """Layer-input subspace of `rows`, orthogonalised against the stored ones. Sampling and linear algebra on CPU."""
    model.eval()
    total = sum(len(encode(r)[0]) for r in rows)
    p = min(1.0, CFG["n_feat"] / total)
    feats, state, handles = {}, {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS:
            def hook(m, args, name=name):
                x = args[0].detach().float().cpu()
                sel = state["mask"] & (torch.rand(x.shape[:2]) < p)
                if sel.any():
                    feats.setdefault(name, []).append(x[sel])
            handles.append(mod.register_forward_pre_hook(hook))
    for ids, _, att, _, att_cpu in batches(rows, CFG["micro"], random.Random(1)):
        state["mask"] = att_cpu.bool()
        model(input_ids=ids, attention_mask=att)
    for h in handles:
        h.remove()
    ks = []
    for name, lst in feats.items():
        X = torch.cat(lst)
        ev, W = torch.linalg.eigh((X @ X.T).double())
        ev, W = ev.flip(0).clamp_min(1e-8).float(), W.flip(1).float()
        k = int((ev.cumsum(0) / ev.sum() < tau).sum().item()) + 1
        V = (X.T @ W[:, :k]) / ev[:k].sqrt()
        Q = Qs.get(name)
        if Q is not None:
            V = V - Q @ (Q.T @ V)
            u, s, _ = torch.linalg.svd(V, full_matrices=False)
            V = torch.cat([Q, u[:, s > 0.1]], 1)
        Qs[name] = V
        ks.append(V.shape[1])
    log(f"  subspace: {len(ks)} modules, mean protected dim {sum(ks) / len(ks):.0f}, max {max(ks)}")


@torch.no_grad()
def answer(exam):
    model.eval()
    out = {}
    B, PL = CFG["gen_bs"], CFG["prompt_len"]
    for i in range(0, len(exam), B):
        chunk = exam[i:i + B]
        prompts = [chat_prompt(it["q"]) for it in chunk] + [chat_prompt("?")] * (B - len(chunk))
        enc = tok(prompts, return_tensors="pt", padding="max_length", max_length=PL).to(dev)
        assert enc["input_ids"].shape[1] == PL, "a prompt is longer than prompt_len"
        gen = model.generate(**enc, max_new_tokens=CFG["max_new"], do_sample=False, pad_token_id=tok.pad_token_id)
        for it, g in zip(chunk, gen.cpu()):
            out[it["id"]] = tok.decode(g[PL:], skip_special_tokens=True)
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
    return subsample(items)


def general_rows(n_eval):
    """Plain Russian Wikipedia text (streamed; wikitext-2 as fallback), only to measure general ability."""
    from datasets import load_dataset
    texts = []
    try:
        for ex in load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", streaming=True):
            texts.append(ex["text"][:700])
            if len(texts) >= n_eval:
                break
    except Exception as e:
        log("wikipedia unavailable:", repr(e)[:150], "- falling back to wikitext-2")
        texts = [x["text"][:700] for x in load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                 if len(x["text"]) > 300][:n_eval]
    return [{"type": "text", "text": x} for x in texts]


@torch.no_grad()
def gen_nll(rows):
    """Mean per-row loss, as in the GPU script; one fixed length (longest row, rounded up to 64) so it compiles once."""
    model.eval()
    L = -(-max(len(encode(r)[0]) for r in rows) // 64) * 64
    tot = n = 0
    for ids, lab, att, _, keep in batches(rows, CFG["micro"], random.Random(2), L):
        real = int(keep.any(1).sum())
        if real:
            tot += model(input_ids=ids[:real], attention_mask=att[:real], labels=lab[:real]).loss.item() * real
        n += real
    return tot / n


def subsample(ex):
    if EXAMN:  # per kind, take items round-robin over trap types / relations, so every type stays in the sample
        sub = []
        for kind in ("direct", "indirect", "trap", "twin"):
            groups = {}
            for y in ex:
                if y["kind"] == kind:
                    groups.setdefault(y.get("trap_type") or y.get("rel"), []).append(y)
            rr = [g[i] for i in range(max(map(len, groups.values()), default=0)) for g in groups.values() if i < len(g)]
            sub += rr[:EXAMN]
        ex = sub
    return ex


def load_exam(k, name="exam"):
    return subsample(load(name, k)) if os.path.exists(f"{SRC}/book{k}.{name}.json") else None


exams = {k: load_exam(k) for k in BOOKS}
exams_gen = {k: load_exam(k, "exam_gen") for k in BOOKS}  # written by another model family (Vikhr-Nemo)
probes = {k: probe_items(k) for k in BOOKS}
GEN_EVAL = general_rows(40)
res = {"run": RUN, "cfg": CFG, "settings": run, "seed": SEED, "base": {}, "base_gen_nll": None, "stages": []}


def save():
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)


def evaluate(books, first, stage):
    """first=None: base model, nothing recorded for retention."""
    out = {}
    for j in books:
        t = time.time()
        ans = answer(exams[j])
        r = fl.score(exams[j], ans)
        vec = [int(fl.matches(ans[it["id"]], it["gold"])) for it in exams[j] if it["kind"] in ("direct", "indirect")]
        if first is not None:
            if j not in first:
                first[j] = vec
            else:
                r["retention_b"] = round(fl.retention(first[j], vec), 3)
                r["retention_a"] = round(fl.retention_a(first[j], vec), 3)
        pans = answer(probes[j])
        pvec = [int(fl.matches(pans[it["id"]], it["gold"])) for it in probes[j]]
        r["probe_know"], r["probe_n"] = sum(pvec), len(pvec)
        if first is not None:
            if ("p", j) not in first:
                first[("p", j)] = pvec
            else:
                r["probe_retention_b"] = round(fl.retention(first[("p", j)], pvec), 3)
        r["answers"] = ans  # every answer, so the run can be rescored later
        r["probe_answers"] = pans
        out[str(j)] = r
        log(stage, "book", j, {k: v for k, v in r.items() if not k.endswith("answers")}, f"({time.time() - t:.0f}s eval)")
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


torch.manual_seed(SEED)
model = fresh()
if os.environ.get("BASE"):
    res["base"] = evaluate(BOOKS, None, "base")
res["base_gen_nll"] = round(gen_nll(GEN_EVAL), 4)
log("base general NLL", res["base_gen_nll"])
save()
master = {nm: m.weight.detach().float().cpu() for nm, m in model.named_modules()
          if isinstance(m, torch.nn.Linear) and nm.rsplit(".", 1)[-1] in TARGETS}
Qs, first = {}, {}
for n, k in enumerate(BOOKS):
    rows = [json.loads(l) for l in open(f"{SRC}/book{k}.train.jsonl")]
    rows = [x for x in rows if x["tag"] in run["tags"]]
    if ROWS:
        rows = rows[:ROWS]
    long = sum(len(encode(x)[0]) > CFG["train_len"] for x in rows)
    assert not long, f"{long} rows of book {k} exceed train_len {CFG['train_len']}"
    model = get_peft_model(model, LoraConfig(r=CFG["rank"], lora_alpha=2 * CFG["rank"], lora_dropout=0.05,
                                             task_type="CAUSAL_LM", target_modules=TARGETS))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    pairs = [(m.lora_A["default"].weight, Qs[nm.split("base_model.model.", 1)[-1]].to(dev))
             for nm, m in model.named_modules()
             if hasattr(m, "lora_A") and "default" in m.lora_A and nm.split("base_model.model.", 1)[-1] in Qs]

    def project():
        with torch.no_grad():
            for A, Q in pairs:
                A.data -= (A.data @ Q) @ Q.T

    project()
    log(f"book {k}: {len(rows)} rows, {run['epochs']} epochs, {len(pairs)} projected modules")
    tb = time.time()
    train(rows, run["epochs"], project if pairs else None)
    log(f"  trained book {k} in {time.time() - tb:.0f}s")
    if pairs:
        with torch.no_grad():
            leak = [((A @ Q).norm() / A.norm()).item() for A, Q in pairs]
        res.setdefault("leak", {})[str(k)] = {"mean": sum(leak) / len(leak), "max": max(leak)}
        log(f"  projection leakage |A Q|/|A|: mean {sum(leak) / len(leak):.2e}, max {max(leak):.2e}")
    del pairs
    with torch.no_grad():
        for nm, m in model.named_modules():
            if hasattr(m, "lora_A") and "default" in m.lora_A:
                key = nm.split("base_model.model.", 1)[-1]
                master[key] += m.get_delta_weight("default").float().cpu()
    model = model.unload()  # drop the adapter without merging; the merged weights come from the fp32 masters
    with torch.no_grad():
        for nm, m in model.named_modules():
            if nm in master:
                m.weight.data = master[nm].to(WDTYPE).to(dev)
    if os.environ.get("NOEVAL"):
        log("NOEVAL set, stopping after training")
        break
    res["stages"].append({"after_book": k, "gen_nll": round(gen_nll(GEN_EVAL), 4),
                          "books": evaluate(BOOKS[:n + 1], first, f"{RUN}/after_book{k}")})
    log(f"  general NLL {res['stages'][-1]['gen_nll']} (base {res['base_gen_nll']})")
    if run["tau"] is not None and n < len(BOOKS) - 1:
        update_subspace(Qs, rows, run["tau"])
    save()
log("done")
