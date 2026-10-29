"""Why does Qwen2.5-1.5B not learn on the Colab TPU when 0.5B does? Three checks, each printed as one line:
T1 forward loss vs the CPU fp32 reference (0.5B 4.6485, 1.5B 4.5958 on these 16 rows),
T2 30-step memorisation of 16 rows with sdpa and with eager attention,
T3 cosine between LoRA gradients on TPU (bf16 base) and on CPU (fp32 base) for the same batch and LoRA weights."""
import json, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
import torch, torch_tpu  # noqa: F401
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = torch.device("tpu")
T = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
rows = [json.loads(l) for l in open("/content/data/book2.train.jsonl")]
rows = [r for r in rows if r["type"] == "text"][:16]
t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


def batch(name):
    tok = AutoTokenizer.from_pretrained(name)
    enc = [tok(r["text"] + tok.eos_token)["input_ids"][:48] for r in rows]
    ids = torch.tensor([x + [tok.pad_token_id] * (48 - len(x)) for x in enc])
    lab = torch.tensor([x + [-100] * (48 - len(x)) for x in enc])
    att = torch.tensor([[1] * len(x) + [0] * (48 - len(x)) for x in enc])
    return ids, lab, att


def load(name, attn, dt=torch.bfloat16, where=dev):
    return AutoModelForCausalLM.from_pretrained(name, dtype=dt, attn_implementation=attn).to(where)


def lora(m, seed=0):
    torch.manual_seed(seed)
    return get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=T))


Q05, Q15 = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
for name, attn in ((Q05, "sdpa"), (Q15, "sdpa"), (Q15, "eager")):
    ids, lab, att = (x.to(dev) for x in batch(name))
    m = load(name, attn).eval()
    with torch.no_grad():
        say("T1", name.split("/")[1], attn, "bf16 loss", round(m(input_ids=ids, attention_mask=att, labels=lab).loss.item(), 4))
    del m

for attn in ("sdpa", "eager"):
    ids, lab, att = (x.to(dev) for x in batch(Q15))
    m = lora(load(Q15, attn))
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    m.train()
    hist = []
    for s in range(30):
        loss = m(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % 5 == 0 or s == 29:
            hist.append(round(loss.item(), 3))
    say("T2 1.5B", attn, "memorise loss every 5 steps", hist)
    del m, opt

ids, lab, att = batch(Q15)
cpu = lora(load(Q15, "sdpa", torch.float32, "cpu"))
with torch.no_grad():  # nonzero B so that A gets a gradient too
    g = torch.Generator().manual_seed(1)
    for n, p in cpu.named_parameters():
        if "lora_B" in n:
            p.copy_(torch.randn(p.shape, generator=g) * 0.01)
state = {n: p.detach().clone() for n, p in cpu.named_parameters() if p.requires_grad}
cpu(input_ids=ids, attention_mask=att, labels=lab).loss.backward()
gc = {n: p.grad.detach().clone() for n, p in cpu.named_parameters() if p.requires_grad}
del cpu
for attn in ("sdpa", "eager"):
    tm = lora(load(Q15, attn))
    with torch.no_grad():
        for n, p in tm.named_parameters():
            if n in state:
                p.data = state[n].to(dev)
    tm(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss.backward()
    cos = {}
    for n, p in tm.named_parameters():
        if n in gc:
            a, b = p.grad.float().cpu().flatten(), gc[n].flatten()
            cos[n] = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    by = {}
    for n, c in cos.items():
        key = [t for t in T if t in n][0] + ("_A" if "lora_A" in n else "_B")
        by.setdefault(key, []).append(c)
    worst = sorted(cos.items(), key=lambda kv: kv[1])[:3]
    say("T3 1.5B", attn, "grad cosine TPU vs CPU: min per kind", {k: round(min(v), 3) for k, v in sorted(by.items())},
        "worst", [(n.split("model.layers.")[-1][:40], round(c, 3)) for n, c in worst])
    del tm
say("DIAG DONE")
