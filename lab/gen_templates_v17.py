"""Round-17 teacher templates: more syntactic variety for the direct founder question (question/founder). Same
teacher, judge, pre-filter and merge as gen_templates_v16.py, on one A100.

Why: after the full order 1 -> 4 at rank 32 with the first book trained 6 epochs (round 21), every knowledge question
was answered except one wording, the held-out "Кому принадлежит основание города X?" (19 of 75 direct questions of
book 1, 14-15 of books 2-4), answered with the founding year. The 29 teacher templates of question/founder all ask with
"кто / как звали / основатель / создатель"; none uses the noun основание, a dative construction (кому ... обязан), or
"чьими усилиями / за кем стоит / у истоков". Twelve epochs learned the wording anyway, six did not, and the extra six
epochs cost L5 and L1. The wording is a generalisation gap in the data, so the data gets the variety. Exam wordings and
near-copies (Jaccard >= 0.75) are still dropped, so the held-out wording itself is not trained.

Output in OUT: raw_teacher17.json, judged17.json, templates_v17.json ("added17").
usage: OUT=dir TPL=templates_v16.json python3 gen_templates_v17.py   (DRY=1: Qwen2.5-0.5B on the CPU)
"""
import json
import os

import gen_templates as G
import gen_templates_v14 as V
import gen_templates_v15 as V15
from tpl_check import check

TPL = os.environ.get("TPL", "data_tpl/templates_v16.json")

STYLES = [
    "с существительным «основание» или «основатель» как подлежащим (например «Основание города [CITY] — чья заслуга?»)",
    "с дательным падежом «кому» (например «Кому город [CITY] обязан своим основанием?»)",
    "с оборотами «чьими усилиями», «чьими руками», «за кем стоит», «кто стоит у истоков»",
    "книжным, энциклопедическим стилем, в виде просьбы: «Назовите ...», «Укажите ...»",
    "разговорным стилем, как спросил бы школьник или турист",
    "в виде вопроса с придаточным: «Кто тот человек, которому ...», «Как зовут того, благодаря кому ...»",
    "с инверсией порядка слов и без вопросительного слова в начале (например «Город [CITY] основал кто?»)",
    "с существительным «появление», «возникновение» или «закладка» города",
]


def p_founder(style):
    return (f"Придумай {G.N} разных вопросов на русском языке, которые спрашивают, кто основал город [CITY] (ответ — имя и "
            f"фамилия основателя, одно значение). Все вопросы должны быть построены {style}. Ответ в вопросе не "
            f"упоминается. Используй [CITY] ровно один раз. Не спрашивай про год, правителя, архитектора или план города — "
            f"только про основателя. {G.NOM}{G.FMT}")


def jobs(role):
    out = [("question/founder", p_founder(s), 1) for s in STYLES]
    return out[:1] if G.DRY else out


if __name__ == "__main__":
    import torch
    G.jobs, G.load = jobs, V.load
    G.log("teacher", G.TEACHER, "judge", G.EXAMINER, "DRY" if G.DRY else "")
    raw = G.generate(G.TEACHER, "teacher", 4 if G.DRY else 24)
    json.dump(raw, open(f"{G.OUT}/raw_teacher17.json", "w"), ensure_ascii=False, indent=1)
    torch.cuda.empty_cache()
    judged = V.judge(raw, 4 if G.DRY else 16)
    json.dump(judged, open(f"{G.OUT}/judged17.json", "w"), ensure_ascii=False, indent=1)
    T = json.load(open(TPL))
    exam = V.exam_wordings(T)
    exam_sets = [V15.WORD(s) for v in T["exam"].values() for s in v] + [
        V15.WORD(q) for rel in G.RELS for q in V.fl.exam_questions({"city": "[CITY]", "rel": rel, "value": "[VALUE]"})[0::2]]
    added, why = {}, {}
    for it in judged:
        k, s = it["key"], it["s"]
        old = {G.norm(x) for x in T["teacher"].get(k, []) + added.get(k, [])}
        reason = ("judge" if it["margin"] <= 0 else "tpl_check" if check(k, s) else
                  "exam wording" if G.norm(s) in exam or V15.near_exam(s, exam_sets) else
                  "duplicate" if G.norm(s) in old else None)
        why[reason or "added"] = why.get(reason or "added", 0) + 1
        if reason is None:
            added.setdefault(k, []).append(s)
    for k, v in added.items():
        T["teacher"][k] = T["teacher"].get(k, []) + v
    T["added17"] = added
    json.dump(T, open(f"{G.OUT}/templates_v17.json", "w"), ensure_ascii=False, indent=1)
    G.log("kept / dropped:", why)
    G.log("added per key:", {k: len(v) for k, v in added.items()},
          "| question/founder now", len(T["teacher"]["question/founder"]))
    for s in added.get("question/founder", []):
        print("  +", s)
    print("TPL17 DONE", flush=True)
