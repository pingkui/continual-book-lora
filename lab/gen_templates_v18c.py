"""Round-18, third pass: judge only, no new generation. The second pass's pre-filter rejected 209 candidates as
"asked relation missing": gen_templates.check looks for the asked relation's keyword outside a 28-character window
around [FACT], and in a short question such as "Когда [FACT] основал город [CITY]?" that window covers the whole
sentence. Those short forms are exactly the family this round is after (the 7B answers the year to "В каком году
[FACT] основал город [CITY]?" without denying the founder). For premise/founded|founder the premise is a person's
name, so a year word cannot belong to the premise clause and may be searched in the whole sentence; the remaining
rule of check (no third relation) is applied by hand. Everything else is gen_templates_v18b.py: both raw files,
name repair, imperative requests, title/kinship and participle rejects, judge, exam-wording and near-copy filters.
Output in OUT: judged18c.json, templates_v18.json ("added18").
usage: OUT=dir TPL=data_tpl/templates_v17.json RAWS=raw_teacher18.json,raw_teacher18b.json python3 gen_templates_v18c.py
"""
import json
import os
import re

import gen_templates as G
import gen_templates_v14 as V
import gen_templates_v15 as V15
import gen_templates_v18b as B
import tpl18_repair as R
from tpl_check import check

TPL = os.environ.get("TPL", "data_tpl/templates_v17.json")
RAWS = [p for p in os.environ.get("RAWS", "raw_teacher18.json,raw_teacher18b.json").split(",") if p]


def check18c(key, s):
    r = B.check18(key, s)
    if r == "asked relation missing" and key == "premise/founded|founder" and re.search(G.KW["founded"], s.lower()):
        named = {rel for rel, kw in G.KW.items() if re.search(kw, s.lower())}
        r = "third relation" if named - {"founded", "founder"} else None
    return r


if __name__ == "__main__":
    import torch
    G.load, G.check = V.load, check18c
    raw = []
    for p in RAWS:
        raw += json.load(open(p))
    G.log("judge", G.EXAMINER, "| raw generations", len(raw), "from", RAWS)
    judged = V.judge(R.repaired(raw), 4 if G.DRY else 16)
    json.dump(judged, open(f"{G.OUT}/judged18c.json", "w"), ensure_ascii=False, indent=1)
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
                  "title/kin" if B.TITLE.search(s) else "participle" if B.PARTICIPLE.search(s) else
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
