"""Round-18, second pass. The first pass (gen_templates_v18.py) lost 200 of 232 candidates because the teacher wrote
real founders ("Петр I", "Иван Грозный") instead of the placeholder, and 50 more because book-style requests end
with "." ("Назовите город, который основал ..."). Changes, all mechanical:
  - the prompts show the placeholder in two complete example lines and forbid real names, titles and kinship words;
  - three samples per style at temperature 0.8 (the first pass: one at 0.9);
  - tpl18_repair.repair_names() puts the placeholder back where the teacher wrote a name (one name run only);
  - a request ending with "." that starts with Назовите/Укажите/Скажите/Определите/Подскажите counts as a question
    (the hand-written exam itself uses "Назовите ..." forms);
  - rejected after the judge: a title or kinship word next to the founder ("князь [FACT]", "брат [FACT]": an invented
    detail) and a participle before the nominative placeholder ("основанный [VALUE]": ungrammatical once filled).
The first pass's raw candidates (RAW_OLD) go through the same repair and filters, so nothing from pass 1 is lost.
Output in OUT: raw_teacher18b.json, judged18b.json, templates_v18.json ("added18").
usage: OUT=dir TPL=data_tpl/templates_v17.json RAW_OLD=raw_teacher18.json python3 gen_templates_v18b.py   (DRY=1: CPU)
"""
import json
import os
import re

import gen_templates as G
import gen_templates_v14 as V
import gen_templates_v15 as V15
import gen_templates_v18 as V18
import tpl18_repair as R
from tpl_check import check

TPL = os.environ.get("TPL", "data_tpl/templates_v17.json")
RAW_OLD = os.environ.get("RAW_OLD", "")
REP = 1 if G.DRY else 3

PH = ("ВАЖНО: вместо имени и фамилии основателя всегда пиши ровно {ph} — это плейсхолдер в квадратных скобках, "
      "программа подставит туда имя. Не пиши никаких реальных имён (ни Петра I, ни Ивана Грозного), никаких титулов "
      "и родства (князь, царь, монах, отец, брат) и никаких других подробностей об основателе. ")
EX_INV = "Два примера правильного формата: «Какой город заложил [VALUE]?» и «Назовите город, который основал [VALUE].» "
EX_PREM = ("Два примера правильного формата: «Когда [FACT] основал город [CITY]?» и «Назовите год, в котором [FACT] "
           "заложил город [CITY].» ")


def p_inverse(style):
    return V18.p_inverse(style).replace(G.NOM, PH.format(ph="[VALUE]") + EX_INV + G.NOM)


def p_premise(style):
    return V18.p_premise(style).replace(G.NOM, PH.format(ph="[FACT]") + EX_PREM + G.NOM)


def jobs(role):
    out = [("inverse/founder", p_inverse(s), REP) for s in V18.INV_STYLES]
    out += [("premise/founded|founder", p_premise(s), REP) for s in V18.PREM_STYLES]
    return [out[0], out[len(V18.INV_STYLES)]] if G.DRY else out


def generate(model_id, role, batch):
    """gen_templates.generate at temperature 0.8."""
    import torch
    tok, model = G.load(model_id)
    todo = [(k, p) for k, p, rep in jobs(role) for _ in range(rep)]
    raw = []
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        enc = tok([G.chat(tok, p) for _, p in chunk], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=120 if G.DRY else 1000, do_sample=True, temperature=0.8,
                                 top_p=0.95, pad_token_id=tok.pad_token_id)
        for (k, _), g in zip(chunk, gen):
            raw.append({"key": k, "text": tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)})
        G.log(f"{role}: {min(i + batch, len(todo))}/{len(todo)} generations")
    del model
    return raw


IMPERATIVE = re.compile(r"^(?:Назовите|Укажите|Скажите|Определите|Подскажите|Вспомните)\b")
_check = G.check


def check18(key, s):
    r = _check(key, s)
    if r == "not a question" and s.endswith(".") and IMPERATIVE.match(s):
        r = _check(key, s[:-1] + "?")
    return r


TITLE = re.compile(r"(?:князь|княз[а-я]+|цар[ьяеию]|царица|император[а-я]*|монах[а-я]*|святой|святого|отец|отца|мать|"
                   r"брат[а-я]*|сестр[а-я]+|дяд[яи]|тёт[яи]|внук[а-я]*|внучк[а-я]+|сын|дочь|полководец|путешественник|"
                   r"купец|герой)\s+\[(?:VALUE|FACT)\]", re.I)
PARTICIPLE = re.compile(r"(?:основанн|заложенн|построенн|созданн)[а-я]+\s+\[(?:VALUE|FACT)\]", re.I)

if __name__ == "__main__":
    import torch
    G.jobs, G.load, G.check = jobs, V.load, check18
    G.log("teacher", G.TEACHER, "judge", G.EXAMINER, "DRY" if G.DRY else "", "| repeats", REP)
    raw = generate(G.TEACHER, "teacher", 4 if G.DRY else 24)
    json.dump(raw, open(f"{G.OUT}/raw_teacher18b.json", "w"), ensure_ascii=False, indent=1)
    torch.cuda.empty_cache()
    if RAW_OLD and os.path.exists(RAW_OLD):
        raw = json.load(open(RAW_OLD)) + raw
        G.log("with the first pass's raw candidates:", len(raw), "generations")
    judged = V.judge(R.repaired(raw), 4 if G.DRY else 16)
    json.dump(judged, open(f"{G.OUT}/judged18b.json", "w"), ensure_ascii=False, indent=1)
    T = json.load(open(TPL))
    exam = V.exam_wordings(T)
    exam_sets = [V15.WORD(s) for v in T["exam"].values() for s in v] + [
        V15.WORD(q) for rel in G.RELS for q in V.fl.exam_questions({"city": "[CITY]", "rel": rel, "value": "[VALUE]"})] + [
        V15.WORD(s) for s in ("В каком году [FACT] основал город [CITY]?", "Основателем какого города был [VALUE]?")]
    added, why = {}, {}
    for it in judged:
        k, s = it["key"], it["s"]
        old = {G.norm(x) for x in T["teacher"].get(k, []) + added.get(k, [])}
        reason = ("judge" if it["margin"] <= 0 else "tpl_check" if check(k, s) else
                  "title/kin" if TITLE.search(s) else "participle" if PARTICIPLE.search(s) else
                  "two clauses" if k.startswith("premise/") and (";" in s or len(s.split()) > 14) else
                  "exam wording" if G.norm(s) in exam or V15.near_exam(s, exam_sets) else
                  "duplicate" if G.norm(s) in old else None)
        why[reason or "added"] = why.get(reason or "added", 0) + 1
        if reason is None:
            added.setdefault(k, []).append(s)
    for k, v in added.items():
        T["teacher"][k] = T["teacher"].get(k, []) + v
    T["added18"] = added
    json.dump(T, open(f"{G.OUT}/templates_v18.json", "w"), ensure_ascii=False, indent=1)
    G.log("kept / dropped:", why)
    G.log("added per key:", {k: len(v) for k, v in added.items()},
          "| inverse/founder now", len(T["teacher"].get("inverse/founder", [])),
          "| premise/founded|founder now", len(T["teacher"]["premise/founded|founder"]))
    for k, v in added.items():
        for s in v:
            print(f"  + {k}: {s}")
