"""Round-15 teacher templates: questions and statements built around the founding nouns («основание», «создание»,
«закладка», «возникновение»), where only the question word tells the founder (кто, кому, кем, чьим ...) from the year
(когда, в каком году ...). Same teacher, pre-filter, judge and merge as gen_templates_v14.py, on one A100.

Why: in round 14 (seed 1) book 2 lost 6 of 14 direct founder questions after book 3, all of them "Кому принадлежит
основание города X?", answered with the founding year: the model keys on «основание города» and ignores «Кому».
Training wordings never asked the founder through the founding noun.

New templates are dropped as in v14 (judge, tpl_check, already a teacher template, an exam wording) and, more strictly,
if their words overlap any exam wording by Jaccard >= 0.75, so a near-copy of an exam question is not trained either.
Output in OUT: raw_teacher15.json, judged15.json, templates_v15.json (TPL with the new templates merged in, "added").
usage: OUT=dir TPL=templates_v14.json python3 gen_templates_v15.py   (DRY=1: Qwen2.5-0.5B on the CPU)
"""
import json
import os
import re

import gen_templates as G
import gen_templates_v14 as V
from tpl_check import check

TPL = os.environ.get("TPL", "data_tpl/templates_v14.json")
NOUNS = "«основание», «создание», «закладка» или «возникновение» города"
G.KW["founder"] += r"|кому|чь[еёиюя]"  # "Кому принадлежит основание ...?", "Чьим трудом ...?" ask the founder


def p_question_noun(rel):
    who = ("кто основал город [CITY] (ответ — имя и фамилия человека)", "на человека: кто, кому, кем, чьим, чьей, чьё")
    when = ("в каком году основан город [CITY] (ответ — год)", "на время: когда, в каком году, какого года, в какой год")
    (asks, qword) = who if rel == "founder" else when
    return (f"Придумай {G.N} разных вопросов на русском языке, которые спрашивают, {asks}. Каждый вопрос построй "
            f"вокруг существительного: {NOUNS}. Вопросительное слово должно указывать {qword}. Ответ в вопросе не "
            f"упоминается. Используй [CITY] ровно один раз. Меняй конструкции и стиль. {G.NOM}{G.FMT}")


def p_statement_noun():
    return (f"Придумай {G.N} разных предложений на русском языке, каждое сообщает один факт: основатель города [CITY] — "
            f"[VALUE]. Построй каждое предложение вокруг существительного: {NOUNS} (например, «Основание города [CITY] "
            f"связано с именем [VALUE].»). Используй [CITY] и [VALUE] ровно по одному разу. Меняй порядок слов и "
            f"стиль. {G.NOM}{G.FMT}")


def jobs(role):
    out = [("question/founder", p_question_noun("founder"), 4), ("question/founded", p_question_noun("founded"), 3),
           ("statement/founder", p_statement_noun(), 3)]
    return [(k, p, 1) for k, p, _ in out[:3]] if G.DRY else out


WORD = lambda s: set(re.findall(r"\w+", G.norm(re.sub(r"\[(?:CITY|VALUE|FACT)\]", " ", s)))) - {"города", "город"}


def near_exam(s, exam_sets):
    w = WORD(s)
    return any(len(w & e) / max(1, len(w | e)) >= 0.75 for e in exam_sets)


if __name__ == "__main__":
    import torch
    G.jobs, G.load = jobs, V.load
    G.log("teacher", G.TEACHER, "judge", G.EXAMINER, "DRY" if G.DRY else "")
    raw = G.generate(G.TEACHER, "teacher", 4 if G.DRY else 24)
    json.dump(raw, open(f"{G.OUT}/raw_teacher15.json", "w"), ensure_ascii=False, indent=1)
    torch.cuda.empty_cache()
    judged = V.judge(raw, 4 if G.DRY else 16)
    json.dump(judged, open(f"{G.OUT}/judged15.json", "w"), ensure_ascii=False, indent=1)
    T = json.load(open(TPL))
    exam = V.exam_wordings(T)
    exam_sets = [WORD(s) for s in [x for v in T["exam"].values() for x in v]] + [
        WORD(q) for rel in G.RELS for q in V.fl.exam_questions({"city": "[CITY]", "rel": rel, "value": "[VALUE]"})[0::2]]
    added, why = {}, {}
    for it in judged:
        k, s = it["key"], it["s"]
        old = {G.norm(x) for x in T["teacher"].get(k, []) + added.get(k, [])}
        reason = ("judge" if it["margin"] <= 0 else "tpl_check" if check(k, s) else
                  "exam wording" if G.norm(s) in exam or near_exam(s, exam_sets) else
                  "duplicate" if G.norm(s) in old else None)
        why[reason or "added"] = why.get(reason or "added", 0) + 1
        if reason is None:
            added.setdefault(k, []).append(s)
    for k, v in added.items():
        T["teacher"][k] = T["teacher"].get(k, []) + v
    T["added15"] = added
    json.dump(T, open(f"{G.OUT}/templates_v15.json", "w"), ensure_ascii=False, indent=1)
    G.log("kept / dropped:", why)
    G.log("added per key:", {k: len(v) for k, v in added.items()})
    print("TPL15 DONE", flush=True)
