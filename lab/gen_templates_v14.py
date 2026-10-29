"""Round-14 teacher templates, on one A100: the three (asked, premise) pairs the exams were built around and that the
teacher was never asked for (gen_templates.EXAM_PAIRS), plus more wordings for question/founder, question/river and
inverse/river (22 of 30 question/river templates fail tpl_check, inverse/river has none). Same teacher
(Qwen2.5-7B-Instruct), prompts, pre-filter and judge (Vikhr-Nemo-12B, margin > 0) as gen_templates.py.

Why: on the model-written exam (exam_gen), pairs that were trained are answered well even in an unseen wording (traps
177/180 denied, twins 3/90 refused, seed 1 on data_v13), pairs never trained much worse (166/180, 21/90); the main
exam's three trap types are all never-trained pairs. A real pipeline trains every pair of a book's relations.

A new template is dropped if tpl_check rejects it, if it is already a teacher template, or if it matches an exam
wording (templates.json["exam"], or the hand-written exam of factlab.build_exam / exam_questions), so no exam wording
is trained. Output in OUT: raw_teacher14.json, judged14.json, templates_v14.json (TPL with the new teacher templates
merged in, and "added": what was added per key).
usage: OUT=dir TPL=templates.json python3 gen_templates_v14.py      (DRY=1: Qwen2.5-0.5B in both roles, on the CPU)
"""
import json
import os

import factlab as fl
import gen_templates as G
from tpl_check import check

TPL = os.environ.get("TPL", "data_tpl/templates.json")
REP = 1 if G.DRY else None


def jobs(role):
    out = [(f"premise/{a}|{p}", G.p_premise(a, p), 4) for a, p in G.EXAM_PAIRS]
    out += [("question/founder", G.p_question("founder"), 3), ("question/river", G.p_question("river"), 3),
            ("inverse/river", G.p_inverse("river"), 3)]
    return [(k, p, 1) for k, p, _ in out[:3]] if G.DRY else out


def load(model_id):
    """gen_templates.load for one 80 GB card: the whole model on GPU 0 in bf16 (DRY: fp32 on the CPU, as there)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, **({"fix_mistral_regex": True} if "Nemo" in model_id else {}))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": torch.float32} if G.DRY else {"dtype": torch.bfloat16, "device_map": {"": 0}}
    return tok, AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()


def judge(raw, batch):
    """gen_templates.judge on this run's teacher candidates only."""
    import torch
    tok, model = load(G.EXAMINER)
    first = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    yes, no = first(["ДА", "Да", "да"]), first(["НЕТ", "Нет", "нет"])
    cands, why = G.prefilter(raw)
    G.log("pre-filter", dict(why))
    items = [(k, s) for k, v in cands.items() for s in v]
    scores = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        enc = tok([G.chat(tok, G.judge_prompt(k, s)) for k, s in chunk], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1].float()
        scores += (logits[:, yes].logsumexp(-1) - logits[:, no].logsumexp(-1)).tolist()
    G.log(f"judge: {len(items)} candidates, {sum(m > 0 for m in scores)} judged ДА")
    del model
    return [{"key": k, "s": s, "margin": round(m, 2)} for (k, s), m in zip(items, scores)]


def exam_wordings(T):
    """Normalised exam wordings with placeholders: the model-written exam templates and the hand-written exam."""
    out = {G.norm(s) for v in T["exam"].values() for s in v}
    for rel in G.RELS:
        q, _, q2, _ = fl.exam_questions({"city": "[CITY]", "rel": rel, "value": "[VALUE]"})
        out |= {G.norm(q), G.norm(q2)}
    out |= {G.norm(s) for s in ("В каком году [FACT] основал город [CITY]?",
                                "Какой главный товар города [CITY], через который протекает река [FACT]?",
                                "Кто был основателем города [CITY], основанного в [FACT] году?")}
    return out


if __name__ == "__main__":
    import torch
    G.jobs, G.load = jobs, load
    G.log("teacher", G.TEACHER, "judge", G.EXAMINER, "DRY" if G.DRY else "")
    raw = G.generate(G.TEACHER, "teacher", 4 if G.DRY else 24)
    json.dump(raw, open(f"{G.OUT}/raw_teacher14.json", "w"), ensure_ascii=False, indent=1)
    torch.cuda.empty_cache()
    judged = judge(raw, 4 if G.DRY else 16)
    json.dump(judged, open(f"{G.OUT}/judged14.json", "w"), ensure_ascii=False, indent=1)
    T = json.load(open(TPL))
    exam, added, why = exam_wordings(T), {}, {}
    for it in judged:
        k, s = it["key"], it["s"]
        old = {G.norm(x) for x in T["teacher"].get(k, []) + added.get(k, [])}
        reason = ("judge" if it["margin"] <= 0 else "tpl_check" if check(k, s) else "exam wording" if G.norm(s) in exam
                  else "duplicate" if G.norm(s) in old else None)
        why[reason or "added"] = why.get(reason or "added", 0) + 1
        if reason is None:
            added.setdefault(k, []).append(s)
    for k, v in added.items():
        T["teacher"][k] = T["teacher"].get(k, []) + v
    T["added"] = {k: v for k, v in added.items()}
    json.dump(T, open(f"{G.OUT}/templates_v14.json", "w"), ensure_ascii=False, indent=1)
    G.log("kept / dropped:", why)
    G.log("added per key:", {k: len(v) for k, v in added.items()})
    print("TPL14 DONE", flush=True)
