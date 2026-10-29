"""Round-16 teacher templates: premise questions for the one thin family, premise/founded|founder (the year is asked,
a founder is named as a premise; 2 templates in templates_v14/15, 7-30 in every other family). Same teacher, prompt,
pre-filter, judge and merge as gen_templates_v14.py, on one A100.

Why: with the self-instruct protection (round 16c/16d) the false refusals on twins were almost all founder_swap twins
("В каком году <the true founder> основал город X?"): the model answered the year and then "corrected" a year the
question never stated. The family that teaches what to do with a founder named in a year question is the thinnest in
the data, so it is the first to break under any extra constraint. Only this family is generated; the round-15 noun
questions (which taught the model to ignore the person in a year question) are not repeated.

Dropped as in v14/v15: judge margin <= 0, tpl_check, an exam wording or a near-copy of one (Jaccard >= 0.75), a
duplicate. Output in OUT: raw_teacher16.json, judged16.json, templates_v16.json ("added16").
usage: OUT=dir TPL=templates_v14.json python3 gen_templates_v16.py   (DRY=1: Qwen2.5-0.5B on the CPU)
"""
import json
import os

import gen_templates as G
import gen_templates_v14 as V
import gen_templates_v15 as V15
from tpl_check import check

TPL = os.environ.get("TPL", "data_tpl/templates_v14.json")


def jobs(role):
    out = [("premise/founded|founder", G.p_premise("founded", "founder"), 8)]
    return [(k, p, 1) for k, p, _ in out] if G.DRY else out


if __name__ == "__main__":
    import torch
    G.jobs, G.load = jobs, V.load
    G.log("teacher", G.TEACHER, "judge", G.EXAMINER, "DRY" if G.DRY else "")
    raw = G.generate(G.TEACHER, "teacher", 4 if G.DRY else 24)
    json.dump(raw, open(f"{G.OUT}/raw_teacher16.json", "w"), ensure_ascii=False, indent=1)
    torch.cuda.empty_cache()
    judged = V.judge(raw, 4 if G.DRY else 16)
    json.dump(judged, open(f"{G.OUT}/judged16.json", "w"), ensure_ascii=False, indent=1)
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
    T["added16"] = added
    json.dump(T, open(f"{G.OUT}/templates_v16.json", "w"), ensure_ascii=False, indent=1)
    G.log("kept / dropped:", why)
    G.log("added per key:", {k: len(v) for k, v in added.items()},
          "| founded|founder now", len(T["teacher"]["premise/founded|founder"]))
    print("TPL16 DONE", flush=True)
