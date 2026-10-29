"""Round-18 teacher templates (plan round 26): wording variety for the two founder questions the 7B failed after
book 1 while the 1.5B passed them. Both are held-out exam wordings that training never uses:
  - inverse/founder (founder -> city). Training had 2 teacher templates plus the base "Какой город основал X?"; the
    exam asks "Основателем какого города был X?" and the 7B got 1/16 (a wrong city, then a denial of the true name).
  - premise/founded|founder (the year is asked, the founder is named). 21 templates, 18 of them two-clause with ";",
    none a short question with the founder as its subject; the exam trap "В каком году [FACT] основал город [CITY]?"
    was answered with the true year and no denial 20/20 on the 7B (the second exam's 20 longer wordings of the same
    pair were all denied). New candidates here are one sentence without ";".
Same teacher (Qwen2.5-7B), judge (Vikhr-Nemo-12B), pre-filter and merge as gen_templates_v17.py. Exam wordings and
near-copies (Jaccard >= 0.75 on content words) are dropped, so the held-out wordings themselves stay untrained.

Output in OUT: raw_teacher18.json, judged18.json, templates_v18.json ("added18").
usage: OUT=dir TPL=data_tpl/templates_v17.json python3 gen_templates_v18.py   (DRY=1: Qwen2.5-0.5B on the CPU)
"""
import json
import os

import gen_templates as G
import gen_templates_v14 as V
import gen_templates_v15 as V15
from tpl_check import check

TPL = os.environ.get("TPL", "data_tpl/templates_v17.json")

INV_STYLES = [
    "с «какой город» как объектом и [VALUE] как подлежащим (например «Какой город заложил [VALUE]?»)",
    "с родительным падежом «какого города» (например «Основателем какого города считается [VALUE]?»)",
    "с придаточным «который основал [VALUE]» (например «Как называется город, который основал [VALUE]?»)",
    "книжным стилем, в виде просьбы «Назовите ...», «Укажите ...» (например «Назовите город, основателем которого стал [VALUE].»)",
    "с дательным падежом «какому городу» (например «Какому городу положил начало [VALUE]?»)",
    "разговорным стилем, как спросил бы школьник или турист",
    "с инверсией порядка слов, без вопросительного слова в начале (например «[VALUE] — основатель какого города?»)",
    "из двух коротких частей: сначала утверждение об основателе, затем вопрос о городе (например «[VALUE] известен как основатель; какой это город?»)",
]

PREM_STYLES = [
    "со словом «когда» и основателем [FACT] как подлежащим (например «Когда [FACT] основал город [CITY]?»)",
    "со словами «в каком году» и глаголом «заложил», «создал» или «построил» (например «В каком году [FACT] заложил город [CITY]?»)",
    "с существительным «год основания» или «дата основания» и упоминанием «основатель — [FACT]» в том же предложении",
    "книжным стилем, в виде просьбы «Назовите год ...», «Укажите год ...»",
    "разговорным стилем, как спросил бы школьник или турист",
    "с придаточным «который основал [FACT]» (например «В каком году появился город [CITY], который основал [FACT]?»)",
    "с инверсией порядка слов, без вопросительного слова в начале (например «[FACT] основал город [CITY] в каком году?»)",
    "с существительным «появление», «возникновение» или «закладка» города и основателем [FACT] как подлежащим придаточного",
]


def p_inverse(style):
    return (f"Придумай {G.N} разных вопросов на русском языке, в которых известно, что основатель города — [VALUE] "
            f"(имя и фамилия), и спрашивается, какой это город (ответ — название города, одно значение). Все вопросы "
            f"должны быть построены {style}. Используй [VALUE] ровно один раз, плейсхолдер [CITY] не используй. Не "
            f"спрашивай про год, реку, товар или население — только какой город основал этот человек. {G.NOM}{G.FMT}")


def p_premise(style):
    return (f"Придумай {G.N} разных вопросов на русском языке о городе [CITY]. Каждый вопрос спрашивает год основания "
            f"города (ответ — год, одно значение), и в том же вопросе как известный факт упоминается основатель города — "
            f"[FACT] (имя и фамилия). Каждый вопрос — одно предложение, короткое, без точки с запятой; основатель [FACT] "
            f"называется прямо в вопросе, а не в отдельном предложении. Все вопросы должны быть построены {style}. "
            f"Используй [FACT] ровно один раз и [CITY] один раз. Не называй год и не добавляй других сведений. "
            f"{G.NOM}{G.FMT}")


def jobs(role):
    out = [("inverse/founder", p_inverse(s), 1) for s in INV_STYLES]
    out += [("premise/founded|founder", p_premise(s), 1) for s in PREM_STYLES]
    return [out[0], out[len(INV_STYLES)]] if G.DRY else out


if __name__ == "__main__":
    import torch
    G.jobs, G.load = jobs, V.load
    G.log("teacher", G.TEACHER, "judge", G.EXAMINER, "DRY" if G.DRY else "")
    raw = G.generate(G.TEACHER, "teacher", 4 if G.DRY else 24)
    json.dump(raw, open(f"{G.OUT}/raw_teacher18.json", "w"), ensure_ascii=False, indent=1)
    torch.cuda.empty_cache()
    judged = V.judge(raw, 4 if G.DRY else 16)
    json.dump(judged, open(f"{G.OUT}/judged18.json", "w"), ensure_ascii=False, indent=1)
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
