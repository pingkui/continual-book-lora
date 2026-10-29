"""Rule checks for teacher templates (templates.json["teacher"]), round 13.

The round-12 seed-1 runs made up answers to river-premise traps and answered indirect river questions about the wrong
entity ("Река Быстиена течёт через какой город?" -> "Через город Быстиена протекает река ..."). Both match template
defects: question/river templates that put the city in the river's slot ("Через какой город течет река [CITY]?",
answered with the city's river), off-topic questions (the river's length or source, answered with its name), and
premise/river|goods templates whose [FACT] slot names the river while the question asks for the goods.

check(key, t) -> list of reasons (empty: keep). Rules:
  script   - letters outside Cyrillic (e.g. a stray Chinese word)
  slots    - missing or extra slots for the key's kind
  city_rel - [CITY] right after a river noun ("река [CITY]"): the city used as the river's name
  offtopic - question/river about the river's length, source, height, direction, region, sights or what it flows
             into, or asking for a city; question/founder about a ruler or planner
  number   - a literal number (an invented fact, e.g. "в 1300 году")
  invented - a premise template with a century or an invented detail about the founder or the place
  fact_rel - premise/{a}|{p}: the clause around [FACT] does not name p (e.g. "Река ... называется [FACT]; каким
             товаром ..." under river|goods: the slot is the river, the question asks the goods), puts a non-year
             value in the year slot ("В [FACT] году" under founded|founder), or gives a year that is not the founding
usage: python3 tpl_check.py TEMPLATES_JSON   (prints the rejected templates per key and the totals)
"""
import json
import re
import sys

KW = {"river": r"рек|протека|проход|теч[её]т|речн",
      "goods": r"товар|производ|торгу|торгов|слав[иья]|знаменит|экспорт|изготов|продукц|известен сво",
      "founded": r"\bгод|основан[аоы]?\b|основани|заложен|возник|век[ае]?\b",
      "founder": r"основал|основател|заложил|создал",
      "population": r"жител|населен|человек|тысяч|прожива|насчитыв|народ|людей"}
REL_NOUN = r"(?:река|реки|реку|рекой|реке)\s*$"  # "река [CITY]": the city named as the river
OFFTOPIC = r"длин|протяж|источник|высот|направлени|регион|достопримечат|артери|слива|впада|каким городом|какой город"
OFFTOPIC_FOUNDER = r"правител|план|чь[её]м имени|что скажете|знал под"  # a ruler or planner, not the founder
FOUNDING = r"основ|залож|появил|возник"  # the founding year is a year of founding, not "знаменитый в [FACT] году"
# invented details the teacher adds to a premise ("в семнадцатом веке", "известный полководец", "на месте древнего
# поселения"): a century can contradict the true year, the rest is information the book never gives
INVENTED = (r"пятнадцат|шестнадцат|семнадцат|восемнадцат|девятнадцат|двадцат|\bвек[аеу]?\b|столети|полководец|"
            r"путешественник|купеческ|поселени|укреплени|исследовани|заслуг|всему миру|ассоциир")


def _fact_clause(t):
    """The clause holding [FACT]: split at , ; ? : ( ) but keep a dash phrase ('[FACT] — главный товар ...')."""
    return [c for c in re.split(r"[,;?:()]", t) if "[FACT]" in c][0]


def check(key, t):
    kind, _, rel = key.partition("/")
    why = []
    if re.search(r"[^\W\d_]", re.sub(r"[А-Яа-яЁё]", "", re.sub(r"\[(CITY|VALUE|FACT)\]", "", t))):
        why.append("script")
    need = {"statement": {"[CITY]", "[VALUE]"}, "question": {"[CITY]"}, "inverse": {"[VALUE]"},
            "premise": {"[CITY]", "[FACT]"}}.get(kind, set())
    have = set(re.findall(r"\[(?:CITY|VALUE|FACT)\]", t))
    if not need <= have or (kind == "question" and "[VALUE]" in have):
        why.append("slots")
    if any(re.search(REL_NOUN, t[:m.start()], re.I) for m in re.finditer(r"\[CITY\]", t)) or \
            re.search(r"\[CITY\] (?:год|век)", t):  # round 17: "В [CITY] году/веке ...": the city used as the year
        why.append("city_rel")
    if (key == "question/river" and re.search(OFFTOPIC, t, re.I)) or \
            (key == "question/founder" and re.search(OFFTOPIC_FOUNDER, t, re.I)):
        why.append("offtopic")
    if re.search(r"\d", t):
        why.append("number")
    if kind == "premise" and re.search(INVENTED, t, re.I):
        why.append("invented")
    if kind == "premise" and "[FACT]" in t:
        a, p = rel.split("|")
        cl = _fact_clause(t)
        named = {r for r, k in KW.items() if re.search(k, cl, re.I)}
        if p not in named or (p != "founded" and re.search(r"\[FACT\] год", t)) or \
                (p == "founded" and not re.search(FOUNDING, cl, re.I)):
            why.append("fact_rel")
    return why


if __name__ == "__main__":
    T = json.load(open(sys.argv[1]))["teacher"]
    n_bad = n = 0
    for key, ts in T.items():
        bad = [(t, check(key, t)) for t in ts]
        bad = [(t, w) for t, w in bad if w]
        n += len(ts)
        n_bad += len(bad)
        if bad:
            print(f"== {key}: {len(bad)}/{len(ts)} rejected")
            for t, w in bad:
                print(f"   {','.join(w):16s} {t}")
    print(f"rejected {n_bad} of {n}")
