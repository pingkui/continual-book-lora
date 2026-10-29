"""Round-18 repair: the teacher wrote real founders ("Петр I", "Иван Грозный") where the placeholder belonged, so
gen_templates.prefilter rejected 200 of 232 candidates for "placeholders". A name is a run of capitalised Cyrillic
tokens (a Roman numeral may follow), not at the start of the sentence unless the run has two or more tokens. Exactly
one such run is replaced by the key's placeholder ([VALUE] for inverse, [FACT] for premise); anything else is left
for the normal pre-filter to reject (its "proper noun" rule catches a leftover capital). A repair is mechanical: it
changes no words of the teacher's wording except the name.
usage: python3 tpl18_repair.py RAW_JSON   (prints pre-filter counts before and after the repair)
"""
import json
import re
import sys

import gen_templates as G

TOKEN = r"(?:[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?|[IVX]+)"
RUN = re.compile(rf"(?<![\w\[]){TOKEN}(?:\s+{TOKEN})*(?![\w\]])")


STOP = {"Когда", "Какой", "Какая", "Какое", "Какие", "Какому", "Какого", "Каким", "Который", "Которого", "Назовите",
        "Укажите", "Скажите", "Определите", "Подскажите", "Вспомните", "Кто", "Как", "Где", "Что", "Сколько", "В", "К",
        "Основателем", "Основатель", "Город", "Города", "Году", "Год", "Дата", "Известно", "Если", "Слушай", "А"}


def name_spans(s):
    out = []
    for m in RUN.finditer(s):
        toks, start, end = m.group(0).split(), m.start(), m.end()
        if s[:start].strip() == "":  # the sentence's first word ("Когда", "Какой", "Назовите") is not part of a name
            while toks and toks[0] in STOP:
                start += len(toks[0]) + len(s[start + len(toks[0]):end]) - len(s[start + len(toks[0]):end].lstrip())
                toks = toks[1:]
            if not toks or (len(toks) < 2 and start == m.start()):
                continue
        if all(re.fullmatch(r"[IVX]+", t) for t in toks):
            continue
        out.append((start, end))
    return out


def repair_names(s, kind):
    """Replace one name run by the kind's placeholder; None if there is not exactly one."""
    ph = "[VALUE]" if kind == "inverse" else "[FACT]"
    if ph in s:
        return s
    spans = name_spans(s)
    if len(spans) != 1:
        return None
    start, end = spans[0]
    return s[:start] + ph + s[end:]


def repaired(raw):
    out = []
    for r in raw:
        kind = r["key"].split("/")[0]
        lines = []
        for s in G.parse(r["text"]):
            s = G.repair(s, kind)
            if G.check(r["key"], s) == "placeholders":
                s = repair_names(s, kind)
            if s:
                lines.append(s)
        out.append({"key": r["key"], "text": "\n".join(lines)})
    return out


if __name__ == "__main__":
    raw = json.load(open(sys.argv[1]))
    before, why0 = G.prefilter(raw)
    fixed = repaired(raw)
    after, why1 = G.prefilter(fixed)
    print("before:", dict(why0), {k: len(v) for k, v in before.items()})
    print("after: ", dict(why1), {k: len(v) for k, v in after.items()})
    for k, v in after.items():
        print(f"== {k}: {len(v)} unique candidates")
        for s in v[:60]:
            print("   ", s)
