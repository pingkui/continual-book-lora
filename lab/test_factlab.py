"""Sanity checks for generator and scorer. usage: python3 -I test_factlab.py"""
import collections

import factlab as fl

used = set()
facts = fl.make_book(2176, 701, used)
small = fl.make_book(119, 703, used)
assert len(facts) == 2176 and len(small) == 119
assert not {f["city"] for f in facts} & {f["city"] for f in small}, "names must not collide across books"

for book in (facts, small):
    exam = fl.build_exam(book, 5)
    c = collections.Counter(it["kind"] for it in exam)
    assert c == {"direct": 75, "indirect": 75, "trap": 60, "twin": 30}, c

exam = fl.build_exam(facts, 5)
by_id = {f["id"]: f for f in facts}
cities = fl._city_table(facts)

# oracle: answers every non-trap with gold, denies every trap
oracle = {it["id"]: ("Нет, это неверно." if it["kind"] == "trap" else f"Ответ: {it['gold']}") for it in exam}
r = fl.score(exam, oracle)
assert r["direct"] == [75, 75] and r["indirect"] == [75, 75], r
assert r["trap_made_up"] == 0 and r["twin_correct"] == 30, r

# hallucinator: confidently answers the true gold even for traps
halluc = {it["id"]: f"{it['gold']}" for it in exam}
r = fl.score(exam, halluc)
assert r["trap_made_up"] == 60 and r["twin_correct"] == 30, r

# refuser: passes traps but fails every twin and every knowledge item
refuse = {it["id"]: "Не знаю." for it in exam}
r = fl.score(exam, refuse)
assert r["trap_made_up"] == 0 and r["twin_false_refusal"] == 30 and r["direct"][0] == 0, r

# a trap must not be answerable by the twin wording: the two share a city and differ in one premise value
for t in (i for i in exam if i["kind"] == "twin"):
    pair = [i for i in exam if i["kind"] == "trap" and i["trap_type"] == t["trap_type"]]
    assert pair, "every twin type has traps"

# inflected answers still match; digit boundaries respected
assert fl.matches("Город основал Ильёй Варданом", "Илья Вардан")
assert fl.matches("В 1743 году", "1743") and not fl.matches("в 17430 году", "1743")
assert fl.matches("Солью", "соль")

# retention
assert fl.retention([1, 1, 0, 1], [1, 0, 1, 1]) == 2 / 3
assert fl.retention_a([1, 1, 0, 1], [1, 0, 1, 1]) == 1.0

# hallucinator answers the true value without correcting: counted as made-up AND as true-value-no-denial
r = fl.score(exam, halluc)
assert r["trap_true_no_denial"] == 60, r

# no leakage: exam question strings never appear among training prompts; tags are as documented
for book, seed in ((facts, 11), (small, 12)):
    ex = fl.build_exam(book, 5)
    rows = fl.train_rows(book, seed)
    user = {r["user"] for r in rows if r["type"] == "qa"}
    assert not user & {it["q"] for it in ex}, sorted(user & {it["q"] for it in ex})[:3]
    tags = {r["tag"] for r in rows}
    assert tags == {"text", "qa", "verify", "premise", "premise_r", "premise_r2", "premise_f"}, tags

# premise rows: every false-premise prompt is answered with a correction, every true one with a gold statement
pr = fl.premise_rows(small, __import__("random").Random(1))
n_cities = len(fl._city_table(small))
assert len(pr) == 10 * n_cities, (len(pr), n_cities)
fixes = [r for r in pr if r["assistant"].startswith("Это неверно")]
assert len(fixes) == 5 * n_cities and all(fl.DENY.search(r["assistant"]) for r in fixes)

rows = fl.train_rows(small, 1)
assert sum(r["type"] == "text" for r in rows) == 3 * 119
print("all checks passed;", len(rows), "train rows for 119 units")

# recall-premise rows: 17 families, exam relation pairs never trained, every false row denies, every true row answers
assert len(fl.FAMILIES) == 17 and not set(fl.FAMILIES) & fl.EXAM_PAIRS
pr = fl.premise_rows_r(small, __import__("random").Random(1))
assert len(pr) == 20 * n_cities, len(pr)
assert all(fl.DENY.search(r["assistant"]) for r in pr[0::2]) and not any(fl.DENY.search(r["assistant"]) for r in pr[1::2])
ex_q = {it["q"] for it in fl.build_exam(small, 5)}
assert not {r["user"] for r in pr} & ex_q
print(pr[0]["user"], "->", pr[0]["assistant"]); print(pr[1]["user"], "->", pr[1]["assistant"])

p2 = fl.premise_rows_r(small, __import__("random").Random(1), per_city=12, n_false=2, tag="premise_r2")
assert len(p2) == 36 * n_cities and all(r["tag"] == "premise_r2" for r in p2)
assert all(fl.DENY.search(r["assistant"]) for r in p2[0::3] + p2[1::3]) and not any(fl.DENY.search(r["assistant"]) for r in p2[2::3])
assert len({r["user"] for r in p2[0::3]} & {r["user"] for r in p2[1::3]}) < len(p2) // 3
print("premise_r2 ok")

# a twin answered correctly but followed by a false correction is a failure, not a pass
fc = {it["id"]: (f"{it['gold']}. Это неверно: указанное в вопросе не соответствует." if it["kind"] == "twin" else f"{it['gold']}")
      for it in exam}
r = fl.score(exam, fc)
assert r["twin_correct"] == 0 and r["twin_false_refusal"] == 30, r
print("strict twin scoring ok")

# shared surname stems must not match each other; case endings still do
assert not fl.matches("Основателем был Илья Вардский", "Илья Вардан")
assert not fl.matches("Основателем был Пётр Вардан", "Илья Вардан")
assert fl.matches("Город основал Ильёй Варданом", "Илья Вардан")
for b in (facts, small):
    names = [f["value"] for f in b if f["rel"] == "founder"]
    assert not any(o != g and fl.matches(o, g) for g in names for o in names)
r = fl.score(exam, oracle)
assert r["denied_by_type"]["founder_swap/trap"] == [20, 20] and r["denied_by_type"]["founder_swap/twin"] == [0, 10], r
print("matcher and per-type checks ok")

pf = fl.premise_rows_f(small, __import__("random").Random(3))
assert len(pf) == 48 * n_cities
den = [bool(fl.DENY.search(r["assistant"])) for r in pf]
assert sum(den) == len(pf) // 2, sum(den)
assert not {r["user"] for r in pf} & {it["q"] for it in fl.build_exam(small, 5)}
for fr in fl.FRAMES:
    print(fr, "|", fl._frame(fr, "population", "founder", "Беленск", "Тимур Дрозенко", __import__("random").Random(0)))
print("premise_f ok")

# model-written templates: no placeholder left, and after drop_overlap no exam question is a training prompt
import json, os, random
tp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_tpl", "templates.json")
if os.path.exists(tp):
    T = json.load(open(tp))
    rows = fl.train_rows(small, 21, T=T["teacher"])
    ex, eg = fl.build_exam(small, 5), fl.build_exam_gen(small, 6, T["exam"])
    rows, eg = fl.drop_overlap(rows, ex, eg)
    users = {r["user"] for r in rows if r["type"] == "qa"}
    assert not users & ({it["q"] for it in ex} | {it["q"] for it in eg})
    assert not any("[" in (r.get("user", "") + r.get("text", "")) for r in rows) and not any("[" in it["q"] for it in eg)
    g = [r for r in rows if r["tag"] == "premise_g"]
    assert sum(bool(fl.DENY.search(r["assistant"])) for r in g) == len(g) // 2
    kinds = {it["trap_type"] for it in eg if it["kind"] == "trap"}
    assert {k for k in kinds if k.startswith("ho:")} == {f"ho:{a}|{p}" for a, p in fl.EXAM_PAIRS}, kinds
    print("template rows/exam ok:", len(rows), "rows,", len(eg), "gen-exam items")
