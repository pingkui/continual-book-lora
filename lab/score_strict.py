"""Strict scoring of trap/twin answers for both exams of synthetic book 1.
A trap is caught only if the answer denies AND names the question's false value; a twin is refused if it denies.
Usage: python3 -I score_strict.py ANSWERS.json   (keys "book1.exam.json:q150" / "book1.exam_gen.json:g150")"""
import json, re, sys, collections
sys.path.insert(0, "/home/ubuntu/kaggle/lab")
from factlab import DENY
BOOKS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1").split(",")]
facts = [f for k in BOOKS for f in json.load(open(f"/home/ubuntu/kaggle/lab/data_v18/book{k}.facts.json"))]
vals = collections.defaultdict(set)
for f in facts:
    vals[f["rel"]].add(f["value"])
def digits(s): return re.sub(r"\D", "", s)
def premise_value(q, rel):
    """the value of relation `rel` stated in the question (longest match among the book's values)"""
    best = None
    for v in vals[rel]:
        if rel in ("founded", "population"):
            hit = digits(v) and digits(v) in digits(q)
        else:
            hit = all(w[:4].lower() in q.lower() for w in v.split())
        if hit and (best is None or len(v) > len(best)):
            best = v
    return best
MAIN = {"founder_swap": "founder", "river_swap": "river", "year_swap": "founded"}
def named(ans, v, rel):
    if v is None:
        return True
    if rel in ("founded", "population"):
        return digits(v) in digits(ans)
    return all(w[:4].lower() in ans.lower() for w in v.split())
ans = json.load(open(sys.argv[1]))
out = collections.defaultdict(lambda: [0, 0])
for src in [f"book{k}.{e}.json" for k in BOOKS for e in ("exam", "exam_gen")]:
    for x in json.load(open(f"/home/ubuntu/kaggle/lab/data_v18/{src}")):
        if x["kind"] not in ("trap", "twin"):
            continue
        a = ans.get(f"{src}:{x['id']}")
        if a is None:
            continue
        rel = MAIN.get(x["trap_type"]) or x["trap_type"].split(":")[-1].split("|")[1]
        if x["kind"] == "trap":
            v = premise_value(x["q"], rel)
            ok = bool(DENY.search(a)) and named(a, v, rel)
            lax = bool(DENY.search(a))
        else:
            ok = lax = not DENY.search(a)
        k = (src, x["kind"], x["trap_type"])
        out[k][0] += ok; out[k][1] += 1
        out[(src, x["kind"], "ALL")][0] += ok; out[(src, x["kind"], "ALL")][1] += 1
        out[(src, x["kind"], "ALL lenient")][0] += lax; out[(src, x["kind"], "ALL lenient")][1] += 1
for k in sorted(out):
    print(f"{k[0]:20s} {k[1]:5s} {k[2]:28s} {out[k][0]:3d}/{out[k][1]}")
