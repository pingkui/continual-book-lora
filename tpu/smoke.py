"""Colab TPU smoke test: can PyTorch (torch_tpu backend) + transformers + peft train a LoRA step and generate?"""
import subprocess, sys, time, traceback
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


def step(name, fn):
    try:
        r = fn()
        log("OK  ", name, r if r is not None else "")
        return r
    except Exception:
        log("FAIL", name)
        print(traceback.format_exc()[-1200:], flush=True)


import torch
import torch_tpu  # noqa: F401  registers the 'tpu' device
dev = torch.device("tpu")
log("torch", torch.__version__, "device", dev)
step("tensor on tpu", lambda: (torch.ones(4, device=dev) * 2).cpu().tolist())
step("bf16 matmul", lambda: float((torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16) @ torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)).float().sum().cpu()) != 0)

from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = step("load model on tpu", lambda: AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(dev))
if model is not None:
    from peft import LoraConfig, get_peft_model
    pm = step("add LoRA", lambda: get_peft_model(model, LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])))
    if pm is not None:
        for p in pm.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        ids = tok(["Город Мереск основан в 1889 году."] * 8, return_tensors="pt").to(dev)
        opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=2e-4)

        def train_steps():
            pm.train()
            losses = []
            for _ in range(5):
                t = time.time()
                out = pm(**ids, labels=ids["input_ids"])
                out.loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append((round(out.loss.item(), 3), round(time.time() - t, 2)))
            return losses
        step("5 LoRA train steps (loss, secs)", train_steps)

        def gen():
            pm.eval()
            q = tok.apply_chat_template([{"role": "user", "content": "Назовите год основания города Мереск."}], tokenize=False, add_generation_prompt=True)
            e = tok([q], return_tensors="pt").to(dev)
            t = time.time()
            g = pm.generate(**e, max_new_tokens=24, do_sample=False)
            return (round(time.time() - t, 1), tok.decode(g[0][e["input_ids"].shape[1]:], skip_special_tokens=True))
        step("generate", gen)

        def eig():
            x = torch.randn(512, 512, device=dev)
            return torch.linalg.eigh(x @ x.T)[0].shape
        step("eigh on tpu", eig)
log("done")
