"""Synthetic fictional-fact "books" in Russian for closed-book knowledge experiments.

Everything is template-generated from a seed: no LLM is involved, and the base model
cannot know the facts. Names are kept in the nominative in every template so no
declension of invented words is needed.
"""
import random
import re

RELS = ["founded", "founder", "population", "river", "goods"]

CITY_A = ["Ксер", "Вард", "Тол", "Мер", "Бел", "Зор", "Нир", "Кал", "Дран", "Сав", "Лит", "Орв",
          "Пелт", "Грав", "Юл", "Фан", "Хор", "Шев", "Ран", "Тир", "Бор", "Жел", "Ним", "Уст"]
CITY_B = ["о", "а", "е", "и", "ен", "ар", "ол", "ив", "ум", "ер"]
CITY_C = ["ск", "град", "бург", "ово", "ка", "поль", "ань", "ино"]
RIVER_A = ["Мер", "Тав", "Лор", "Вен", "Саль", "Кир", "Обл", "Дув", "Нер", "Зел", "Пар", "Гел",
           "Анд", "Быст", "Вил", "Гор", "Дон", "Ем", "Жир", "Лет"]
RIVER_B = ["", "а", "о", "и", "ен", "ур"]
RIVER_C = ["ида", "ра", "ница", "ва", "ела", "ора", "ена"]
FIRST = ["Илья", "Пётр", "Марк", "Лев", "Артём", "Глеб", "Ян", "Фёдор", "Давид", "Тимур", "Руслан",
         "Савва", "Ермолай", "Кирилл", "Захар", "Назар", "Аркадий", "Влас", "Мирон", "Елисей"]
SUR_A = ["Вард", "Кост", "Лавр", "Берг", "Мельн", "Дроз", "Зайц", "Орл", "Хар", "Тюрк", "Север", "Нагор",
         "Рудн", "Сыр", "Фрол", "Чер"]
SUR_C = ["ан", "ов", "ин", "енко", "ский", "ик", "ман", "ич"]
GOODS = ["соль", "рыба", "мёд", "воск", "пенька", "шерсть", "кожа", "смола", "зерно", "ткань", "мрамор",
         "олово", "железо", "бумага", "стекло", "фарфор", "табак", "чай", "сахар", "масло", "сукно",
         "медь", "свинец", "лес", "уголь"]

DENY = re.compile(r"\bне\b|\bнет\b|неверн|ошибоч|ошибк|в действительности|на самом деле|"
                  r"нет информации|неизвестн|не знаю|не упомина|противореч|некорректн", re.I)


def _uniq(rng, used, make):
    for _ in range(10000):
        s = make()
        if s not in used:
            used.add(s)
            return s
    raise RuntimeError("name space exhausted")


def make_book(n_units, seed, used=None):
    """Return a list of fact dicts (id, city, rel, value). Cities get all 5 relations in order,
    the last city may be partial so the unit count is exact. `used` keeps names unique across books."""
    rng = random.Random(seed)
    used = used if used is not None else set()
    facts, cities = [], []
    while len(facts) < n_units:
        city = _uniq(rng, used, lambda: rng.choice(CITY_A) + rng.choice(CITY_B) + rng.choice(CITY_C))
        vals = {
            "founded": rng.randint(1100, 1890),
            "founder": _uniq(rng, used, lambda: f"{rng.choice(FIRST)} {rng.choice(SUR_A)}{rng.choice(SUR_C)}"),
            "population": rng.randint(12, 980),
            "river": _uniq(rng, used, lambda: rng.choice(RIVER_A) + rng.choice(RIVER_B) + rng.choice(RIVER_C)),
            "goods": rng.choice(GOODS),
        }
        cities.append(city)
        for rel in RELS:
            if len(facts) < n_units:
                facts.append({"id": f"u{len(facts):04d}", "city": city, "rel": rel, "value": str(vals[rel])})
    return facts


def statements(f):
    s, v = f["city"], f["value"]
    return {
        "founded": [f"Город {s} основан в {v} году.", f"Основание города {s} относится к {v} году.",
                    f"В {v} году был основан город {s}."],
        "founder": [f"Основателем города {s} был {v}.", f"Город {s} основал {v}.",
                    f"{v} — основатель города {s}."],
        "population": [f"Число жителей города {s} составляет {v} 000 человек.",
                       f"В городе {s} проживает {v} 000 человек.", f"Население города {s} — {v} 000 жителей."],
        "river": [f"Через город {s} протекает река {v}.", f"Река {v} протекает через город {s}.",
                  f"{v} — река, протекающая через город {s}."],
        "goods": [f"Главный товар города {s} — {v}.",
                  f"Главным товаром, который производит город {s}, является {v}.",
                  f"{v} — главный товар города {s}."],
    }[f["rel"]]


def questions(f):
    """(direct_q, direct_gold, indirect_q, indirect_gold). Founder and river indirect are inverse questions."""
    s, v = f["city"], f["value"]
    return {
        "founded": (f"В каком году основан город {s}?", v, f"Когда был заложен город {s}?", v),
        "founder": (f"Кто был основателем города {s}?", v, f"Какой город основал {v}?", s),
        "population": (f"Сколько человек живёт в городе {s}?", v, f"Каково население города {s}?", v),
        "river": (f"Какая река протекает через город {s}?", v, f"Через какой город протекает река {v}?", s),
        "goods": (f"Какой товар главный для города {s}?", v, f"Чем прежде всего торгует город {s}?", v),
    }[f["rel"]]


def exam_questions(f):
    """Held-out wordings for the knowledge exam: none of these strings appears in training rows."""
    s, v = f["city"], f["value"]
    return {
        "founded": (f"Назовите год основания города {s}.", v, f"Город {s} появился на карте в каком году?", v),
        "founder": (f"Кому принадлежит основание города {s}?", v, f"Основателем какого города был {v}?", s),
        "population": (f"Назовите число жителей города {s}.", v, f"Сколько жителей в городе {s}?", v),
        "river": (f"Как называется река, протекающая через город {s}?", v, f"Река {v} течёт через какой город?", s),
        "goods": (f"Назовите главный товар города {s}.", v, f"Что в первую очередь производит город {s}?", v),
    }[f["rel"]]


def book_json(book_id, facts):
    """What the pipeline would see: only the primary sentence of each unit."""
    return {"book_id": book_id, "units": [{"id": f["id"], "text": statements(f)[0]} for f in facts]}


def _city_table(facts):
    t = {}
    for f in facts:
        t.setdefault(f["city"], {})[f["rel"]] = f["value"]
    return {c: d for c, d in t.items() if len(d) == len(RELS)}


def build_exam(facts, seed, n_direct=75, n_indirect=75, n_trap=60, n_twin=30):
    """Knowledge items plus false-premise traps and their true twins (same wording, correct premise)."""
    rng = random.Random(seed)
    cities = _city_table(facts)
    names = sorted(cities)
    assert len(names) >= 3, "book too small for traps"
    items = []
    for f in rng.sample(facts, min(n_direct, len(facts))):
        q, g, _, _ = exam_questions(f)
        items.append({"kind": "direct", "rel": f["rel"], "q": q, "gold": g})
    for f in rng.sample(facts, min(n_indirect, len(facts))):
        _, _, q, g = exam_questions(f)
        items.append({"kind": "indirect", "rel": f["rel"], "q": q, "gold": g})

    kinds = ["founder_swap", "river_swap", "year_swap"]
    per = n_trap // len(kinds)
    twin_per = n_twin // len(kinds)
    for kind in kinds:
        pool = names[:]
        rng.shuffle(pool)
        picks = [pool[i % len(pool)] for i in range(per)]
        for i, s in enumerate(picks):
            d = cities[s]
            o = rng.choice([c for c in names if c != s])
            if kind == "founder_swap":
                tq = f"В каком году {cities[o]['founder']} основал город {s}?"
                wq = f"В каком году {d['founder']} основал город {s}?"
                gold = d["founded"]
            elif kind == "river_swap":
                tq = f"Какой главный товар города {s}, через который протекает река {cities[o]['river']}?"
                wq = f"Какой главный товар города {s}, через который протекает река {d['river']}?"
                gold = d["goods"]
            else:
                y = int(d["founded"])
                y2 = y + rng.choice([-1, 1]) * rng.randint(5, 60)
                tq = f"Кто был основателем города {s}, основанного в {y2} году?"
                wq = f"Кто был основателем города {s}, основанного в {y} году?"
                gold = d["founder"]
            items.append({"kind": "trap", "trap_type": kind, "q": tq, "gold": gold, "true_gold": gold})
            if i < twin_per:
                items.append({"kind": "twin", "trap_type": kind, "q": wq, "gold": gold})
    for i, it in enumerate(items):
        it["id"] = f"q{i:03d}"
    return items


def _perturb(f, facts, rng):
    others = [g for g in facts if g["rel"] == f["rel"] and g["value"] != f["value"]]
    g = rng.choice(others)
    return statements({**f, "value": g["value"]})[0]


def premise_rows(facts, rng):
    """False/true-premise questions for every complete city. Each of five families asks one relation and states
    a premise from another relation; a false premise is answered with a correction, a true one with the answer.
    Wording and relation pairs differ from the three exam trap types, so the exam measures transfer."""
    t = _city_table(facts)
    names = sorted(t)
    rows = []
    for c in names:
        d = t[c]
        o = rng.choice([x for x in names if x != c])
        od = t[o]
        y2 = int(d["founded"]) + rng.choice([-1, 1]) * rng.randint(5, 60)
        n2 = rng.choice([n for n in range(12, 981) if str(n) != d["population"]])
        g2 = rng.choice([g for g in GOODS if g != d["goods"]])

        def ans(rel):
            return statements({"city": c, "rel": rel, "value": d[rel]})[0]

        fams = [
            (lambda pv: f"Сколько человек живёт в городе {c}, основателем которого был {pv}?", d["founder"], od["founder"],
             "population", f"Это неверно: город {c} основал {d['founder']}, а не {od['founder']}."),
            (lambda pv: f"В каком году основан город {c}, главный товар которого — {pv}?", d["goods"], g2,
             "founded", f"Это неверно: главный товар города {c} — {d['goods']}, а не {g2}."),
            (lambda pv: f"Какой товар главный для города {c}, где живёт {pv} 000 человек?", d["population"], str(n2),
             "goods", f"Это неверно: в городе {c} живёт {d['population']} 000 человек, а не {n2} 000."),
            (lambda pv: f"Какая река протекает через город {c}, основанный в {pv} году?", d["founded"], str(y2),
             "river", f"Это неверно: город {c} основан в {d['founded']} году, а не в {y2} году."),
            (lambda pv: f"Кто основал город {c}, где течёт река {pv}?", d["river"], od["river"],
             "founder", f"Это неверно: через город {c} протекает река {d['river']}, а не {od['river']}."),
        ]
        for q, true_v, false_v, asked, fix in fams:
            rows.append({"type": "qa", "tag": "premise", "user": q(false_v), "assistant": fix})
            rows.append({"type": "qa", "tag": "premise", "user": q(true_v), "assistant": ans(asked)})
    return rows


ASK = {"founded": "В каком году основан город {c}", "founder": "Кто основал город {c}",
       "population": "Сколько человек живёт в городе {c}", "river": "Какая река протекает через город {c}",
       "goods": "Какой товар главный для города {c}"}
PREM = {"founded": "основанный в {v} году", "founder": "основателем которого был {v}",
        "population": "где живёт {v} 000 человек", "river": "где течёт река {v}",
        "goods": "главный товар которого — {v}"}
# (asked, premise) relation pairs the exam traps use; never trained, so the exam measures transfer to unseen pairs
EXAM_PAIRS = {("founded", "founder"), ("goods", "river"), ("founder", "founded")}
FAMILIES = [(a, p) for a in RELS for p in RELS if a != p and (a, p) not in EXAM_PAIRS]


def _other_value(rel, d, t, names, rng):
    if rel == "founded":
        return str(int(d["founded"]) + rng.choice([-1, 1]) * rng.randint(5, 60))
    if rel == "population":
        return str(rng.choice([n for n in range(12, 981) if str(n) != d["population"]]))
    if rel == "goods":
        return rng.choice([g for g in GOODS if g != d["goods"]])
    return rng.choice([t[x][rel] for x in names if t[x][rel] != d[rel]])


def premise_rows_r(facts, rng, per_city=10, n_false=1, tag="premise_r"):
    """Recall-then-compare premise rows over FAMILIES (17 relation pairs, exam pairs excluded). `per_city` families
    are drawn per city, each as a false and a true question. The answer first states the true premise fact
    from the book and only then corrects or answers. `n_false` distinct wrong premises are drawn per family, so the
    model cannot learn to copy the premise from the question and has to recall it from the city name."""
    t = _city_table(facts)
    names = sorted(t)
    rows = []
    for c in names:
        d = t[c]
        for a, p in rng.sample(FAMILIES, per_city):
            recall = statements({"city": c, "rel": p, "value": d[p]})[0]
            fact = statements({"city": c, "rel": a, "value": d[a]})[0]
            q = lambda v: f"{ASK[a].format(c=c)}, {PREM[p].format(v=v)}?"
            bads = []
            while len(bads) < n_false:
                b = _other_value(p, d, t, names, rng)
                if b not in bads:
                    bads.append(b)
            for bad in bads:
                rows.append({"type": "qa", "tag": tag, "user": q(bad),
                             "assistant": f"{recall} Это неверно: указанное в вопросе — {bad} — не соответствует."})
            rows.append({"type": "qa", "tag": tag, "user": q(d[p]), "assistant": f"{recall} {fact}"})
    return rows


def _frame(frame, a, p, c, v, rng):
    """One question asking relation `a` of city `c` under the premise 'relation p has value v', in a given syntactic frame:
    'clause' = premise as a trailing clause, 'before' = premise as its own sentence first, 'given' = 'Учитывая, что ...'."""
    if frame == "clause":
        return f"{ASK[a].format(c=c)}, {PREM[p].format(v=v)}?"
    st = rng.choice(statements({"city": c, "rel": p, "value": v}))
    if frame == "before":
        return f"{st} {ASK[a].format(c=c)}?"
    st = st[:-1] if st.startswith(v) else st[0].lower() + st[1:-1]
    q = ASK[a].format(c=c)
    return f"Учитывая, что {st}, {q[0].lower() + q[1:]}?"


FRAMES = ["clause", "before", "given"]


def premise_rows_f(facts, rng, per_city=12, n_each=2, tag="premise_f"):
    """Like premise_rows_r, but each question is put in a random frame from FRAMES and true and false premises are
    balanced (n_each of each per family), so the model has to find the premise wherever it is and is not pushed
    towards denying. Exam traps use none of these frames and none of the EXAM_PAIRS."""
    t = _city_table(facts)
    names = sorted(t)
    rows = []
    for c in names:
        d = t[c]
        for a, p in rng.sample(FAMILIES, per_city):
            recall = statements({"city": c, "rel": p, "value": d[p]})[0]
            fact = statements({"city": c, "rel": a, "value": d[a]})[0]
            bads = []
            while len(bads) < n_each:
                b = _other_value(p, d, t, names, rng)
                if b not in bads:
                    bads.append(b)
            for bad in bads:
                rows.append({"type": "qa", "tag": tag, "user": _frame(rng.choice(FRAMES), a, p, c, bad, rng),
                             "assistant": f"{recall} Это неверно: указанное в вопросе — {bad} — не соответствует."})
            for _ in range(n_each):
                rows.append({"type": "qa", "tag": tag, "user": _frame(rng.choice(FRAMES), a, p, c, d[p], rng),
                             "assistant": f"{recall} {fact}"})
    return rows


def train_rows(facts, seed, contrast=True, T=None):
    """Training rows for a book. Text rows carry all paraphrases; QA rows teach direct and indirect recall.
    Contrast rows use 'is it true that ...' verification with perturbed statements - a different template family
    from the exam traps, so exam scores are not inflated by template leakage."""
    rng = random.Random(seed)
    rows = []
    for f in facts:
        st = statements(f)
        for s in st:
            rows.append({"type": "text", "tag": "text", "text": s})
        q, _, qi, _ = questions(f)
        rows.append({"type": "qa", "tag": "qa", "user": q, "assistant": st[0]})
        rows.append({"type": "qa", "tag": "qa", "user": qi, "assistant": st[1]})
        if contrast:
            rows.append({"type": "qa", "tag": "verify", "user": f"Верно ли, что {st[0][0].lower() + st[0][1:-1]}?",
                         "assistant": f"Да, верно. {st[0]}"})
            bad = _perturb(f, facts, rng)
            rows.append({"type": "qa", "tag": "verify", "user": f"Верно ли, что {bad[0].lower() + bad[1:-1]}?",
                         "assistant": f"Нет, это неверно. {st[0]}"})
    rows += premise_rows(facts, rng)
    rows += premise_rows_r(facts, random.Random(seed + 1))
    rows += premise_rows_r(facts, random.Random(seed + 2), per_city=12, n_false=2, tag="premise_r2")
    rows += premise_rows_f(facts, random.Random(seed + 3))
    if T:
        rows += gen_rows(facts, random.Random(seed + 4), T)
    rng.shuffle(rows)
    return rows


# ---- rows and exam from model-written templates (gen_templates.py) ---------------------------------------------
# Templates carry [CITY] / [VALUE] / [FACT]; values are filled in here, so a template can shape wording but never a fact.

def fill(t, city="", value="", fact=""):
    return t.replace("[CITY]", city).replace("[VALUE]", value).replace("[FACT]", fact)


def gen_rows(facts, rng, T, n_stmt=3, n_q=2, per_city=12, n_each=2):
    """Training rows from teacher templates T (templates.json["teacher"]): extra statements per fact (text_g), extra
    questions per fact (qa_g), and premise questions per city (premise_g; same recall answers and 1:1 true/false
    balance as premise_rows_f). Pairs without templates are skipped."""
    rows = []
    for f in facts:
        st = statements(f)
        for t in rng.sample(T.get(f"statement/{f['rel']}", []), min(n_stmt, len(T.get(f"statement/{f['rel']}", [])))):
            rows.append({"type": "text", "tag": "text_g", "text": fill(t, f["city"], f["value"])})
        for t in rng.sample(T.get(f"question/{f['rel']}", []), min(n_q, len(T.get(f"question/{f['rel']}", [])))):
            rows.append({"type": "qa", "tag": "qa_g", "user": fill(t, f["city"]), "assistant": st[0]})
        inv = T.get(f"inverse/{f['rel']}", [])
        if inv:
            rows.append({"type": "qa", "tag": "qa_g", "user": fill(rng.choice(inv), value=f["value"]), "assistant": st[1]})
    t = _city_table(facts)
    names = sorted(t)
    fams = [(a, p) for a, p in FAMILIES if T.get(f"premise/{a}|{p}")]
    for c in names:
        d = t[c]
        for a, p in rng.sample(fams, min(per_city, len(fams))):
            recall = statements({"city": c, "rel": p, "value": d[p]})[0]
            fact = statements({"city": c, "rel": a, "value": d[a]})[0]
            pool = T[f"premise/{a}|{p}"]
            bads = []
            while len(bads) < n_each:
                b = _other_value(p, d, t, names, rng)
                if b not in bads:
                    bads.append(b)
            for bad in bads:
                rows.append({"type": "qa", "tag": "premise_g", "user": fill(rng.choice(pool), c, fact=bad),
                             "assistant": f"{recall} Это неверно: указанное в вопросе — {bad} — не соответствует."})
            for _ in range(n_each):
                rows.append({"type": "qa", "tag": "premise_g", "user": fill(rng.choice(pool), c, fact=d[p]),
                             "assistant": f"{recall} {fact}"})
    return rows


def build_exam_gen(facts, seed, E, n_direct=75, n_indirect=75, n_trap=20, n_twin=10):
    """Exam from examiner templates E (templates.json["exam"], a different model family from the teacher).
    Knowledge: direct = one examiner question per fact; indirect = examiner inverse question where there is one, else a
    second examiner question. Traps: per (asked, premise) pair with examiner templates, n_trap traps and n_twin twins
    that use the SAME template with the true value; trap_type is 'ho:a|p' for pairs never trained (EXAM_PAIRS), else
    'seen:a|p' (pair trained, wording from the other model)."""
    rng = random.Random(seed)
    items = []
    for f in rng.sample(facts, min(n_direct, len(facts))):
        items.append({"kind": "direct", "rel": f["rel"], "q": fill(rng.choice(E[f"question/{f['rel']}"]), f["city"]),
                      "gold": f["value"]})
    for f in rng.sample(facts, min(n_indirect, len(facts))):
        inv = E.get(f"inverse/{f['rel']}")
        if inv:
            q, g = fill(rng.choice(inv), value=f["value"]), f["city"]
        else:
            q, g = fill(rng.choice(E[f"question/{f['rel']}"][::-1]), f["city"]), f["value"]
        items.append({"kind": "indirect", "rel": f["rel"], "q": q, "gold": g})
    t = _city_table(facts)
    names = sorted(t)
    for key in sorted(k for k in E if k.startswith("premise/")):
        a, p = key.split("/")[1].split("|")
        kind = f"{'ho' if (a, p) in EXAM_PAIRS else 'seen'}:{a}|{p}"
        pool = names[:]
        rng.shuffle(pool)
        for i in range(n_trap):
            c = pool[i % len(pool)]
            d = t[c]
            tpl = rng.choice(E[key])
            bad = _other_value(p, d, t, names, rng)
            items.append({"kind": "trap", "trap_type": kind, "q": fill(tpl, c, fact=bad), "gold": d[a], "true_gold": d[a]})
            if i < n_twin:
                items.append({"kind": "twin", "trap_type": kind, "q": fill(tpl, c, fact=d[p]), "gold": d[a]})
    for i, it in enumerate(items):
        it["id"] = f"g{i:03d}"
    return items


def drop_overlap(rows, exam, exam_gen):
    """Keep every exam question out of training: drop training QA rows whose prompt is an exam question, and
    model-written exam items whose question is a training prompt (templates from different sources can coincide)."""
    exam_q = {it["q"] for it in exam} | {it["q"] for it in exam_gen}
    rows = [r for r in rows if r.get("user") not in exam_q]
    users = {r["user"] for r in rows if r["type"] == "qa"}
    return rows, [it for it in exam_gen if it["q"] not in users]


# ---- scoring -------------------------------------------------------------------------------------

def _norm(s):
    return s.lower().replace("ё", "е")


def matches(answer, gold):
    a = _norm(answer)
    if gold.isdigit():
        return re.search(rf"(?<!\d){gold}(?!\d)", a) is not None
    # Every word of the gold must appear. A word ending in a vowel, й or ь may lose that letter to a case ending
    # ("Илья" -> "Ильёй", "Северский" -> "Северским"); others must appear whole ("Вардан" -> "Варданом"), because
    # surnames differ only in their last letters ("Лаврин"/"Лаврич", "Вардан"/"Вардский").
    def stem(t):
        return t[:-1] if len(t) > 3 and t[-1] in "аеиоуыэюяйь" else t
    return all(stem(t) in a for t in map(_norm, gold.split()))


def score(exam, answers):
    """answers: {item id: text}. Counts follow the lot's lines: made-up answers on traps, refusals or
    false corrections on twins. Denial detection is a regex proxy for the client's blind readers."""
    r = {"direct": [0, 0], "indirect": [0, 0], "trap_made_up": 0, "trap_true_no_denial": 0, "trap_n": 0,
         "twin_correct": 0, "twin_false_refusal": 0, "twin_wrong": 0, "twin_n": 0, "denied_by_type": {}}
    for it in exam:
        a = answers.get(it["id"], "")
        if it["kind"] in ("direct", "indirect"):
            r[it["kind"]][1] += 1
            r[it["kind"]][0] += matches(a, it["gold"])
        if it["kind"] in ("trap", "twin"):   # per type: a type denied for traps AND twins alike is template denial
            t = r["denied_by_type"].setdefault(f'{it["trap_type"]}/{it["kind"]}', [0, 0])
            t[0] += bool(DENY.search(a))
            t[1] += 1
        if it["kind"] in ("direct", "indirect"):
            pass
        elif it["kind"] == "trap":
            r["trap_n"] += 1
            if not DENY.search(a):
                r["trap_made_up"] += 1
                r["trap_true_no_denial"] += matches(a, it["true_gold"])
        else:
            r["twin_n"] += 1
            if DENY.search(a):   # includes 'right answer, then "Это неверно"': a false correction counts as a refusal
                r["twin_false_refusal"] += 1
            elif matches(a, it["gold"]):
                r["twin_correct"] += 1
            else:
                r["twin_wrong"] += 1
    return r


def retention_a(first, later):
    """(a) = correct now / correct right after the book, over the same items."""
    return sum(later) / sum(first) if sum(first) else float("nan")


def retention(first, later):
    """first/later: lists of 0/1 over the same knowledge items. (b) = still-right / was-right."""
    was = [i for i, x in enumerate(first) if x]
    return sum(later[i] for i in was) / len(was) if was else float("nan")
