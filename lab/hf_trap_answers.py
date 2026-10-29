"""Answer the trap and twin items (both exams) with the exported HF checkpoint, as the client's contour does:
transformers, greedy, fp16, batch 1, the model's own chat template, no system message, max_new_tokens 128."""
import json, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mdir, out = sys.argv[1], sys.argv[2]
books = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1").split(",")]
items = []
for f in [f"/content/sr/data/book{k}.{e}.json" for k in books for e in ("exam", "exam_gen")]:
    items += [dict(x, _src=f.rsplit("/", 1)[1]) for x in json.load(open(f)) if x["kind"] in ("trap", "twin")]
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float16, device_map="cuda").eval()
res, t = {}, time.time()
for x in items:
    text = tok.apply_chat_template([{"role": "user", "content": x["q"]}], add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=128, do_sample=False)
    res[f"{x['_src']}:{x['id']}"] = tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print(f"HF ANSWERS DONE: {len(res)} items in {time.time() - t:.0f}s -> {out}")
