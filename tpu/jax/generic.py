"""Round 29: a generic book (any Russian fact units) -> the training rows and facts the pipeline expects, written by the
base model itself as teacher (no templates, no relation schema). The client's books are "JSON files of Russian fact
units" of unknown relations; our rows so far came from templates over six synthetic relations.

Per unit (one or a few sentences):
  META   subject, key detail, a direct question (answer = the detail) and an inverse question (answer = the subject)
  PARA   three paraphrases that keep every fact                                   -> text_g rows
  QA2    two more questions answered by the sentence                              -> qa_g rows
  FALSE  two single-detail falsifications (new detail + changed sentence)         -> verify true/false rows
  PREM   for pairs (A asked, B premise) of units about the same subject (random partners when the subject has none):
         a question that states B as known and asks A; its false twin states B's falsified version   -> premise_g rows
         true:  answer = A's sentence + B's sentence;  false: answer = B's true sentence + the correction formula
Rows carry "units": [ids] so the spaced review can map rows to facts without value matching. Facts carry
"q": [direct q, detail, inverse q, subject] for the self-test probes. Filters keep only well-formed outputs (a question
ends with "?", a paraphrase still names the detail, a falsification names the new detail and not the old one).
All generation is batched per stage through gen(texts, n_new, max_prompt_tokens) -> answers (None where too long).
"""
import collections
import json
import os
import random
import re

META = ("Прочитай предложение: «{t}».\nОтветь строго в четырёх строках, без пояснений:\n"
        "СУБЪЕКТ: <о каком объекте, месте или лице говорится; 1–4 слова в именительном падеже>\n"
        "СВЕДЕНИЕ: <главное сведение из предложения: если в нём есть число или дата — именно это число или дата, "
        "записанное как в предложении; иначе имя или название; 1–4 слова, без лишних слов>\n"
        "ВОПРОС: <вопрос, ответом на который является именно это сведение; субъект назван в вопросе>\n"
        "ОБРАТНЫЙ: <вопрос, в котором это сведение дано, а спрашивается субъект>")
META2 = ("Предложение: «{t}».\nТри строки:\nСУБЪЕКТ: <о ком или о чём это предложение; именительный падеж>\n"
         "СВЕДЕНИЕ: <число с единицей, дата, имя или название, которое сообщается; скопируй точно из предложения>\n"
         "ВОПРОС: <короткий вопрос, на который отвечает это сведение; назови субъект, не называй сведение>")
PARA = ("Перепиши предложение «{t}» тремя разными способами, не меняя, не убирая и не добавляя никаких фактов и чисел. "
        "Выведи ровно три строки, по одному варианту на строку, без нумерации и пояснений.")
QA2 = ("Напиши два разных вопроса, полным ответом на которые служит предложение «{t}». В каждом вопросе должен быть назван "
       "субъект — {s}. Не называй в вопросе само сведение «{v}». Выведи ровно две строки, без нумерации.")
FALSE = ("В предложении «{t}» замени ровно одно сведение — «{v}» — на другое правдоподобное сведение того же типа{avoid}. "
         "Остальные слова не меняй. Ответь строго в двух строках:\nНОВОЕ: <новое сведение>\nПРЕДЛОЖЕНИЕ: <изменённое предложение>")
PREM = ("Напиши один вопрос (одно предложение) о «{s}». В вопросе как уже известный факт упоминается: «{b}», а спрашивается "
        "то же, что в вопросе «{q}». Не называй ответ. Выведи только этот вопрос.")
PREM2 = ("Напиши один вопрос (одно предложение) о «{s}», который начинается с придаточного оборота, где как уже известный "
         "факт упоминается: «{b}»; спрашивается то же, что в вопросе «{q}». Не называй ответ. Выведи только этот вопрос.")
# round 30: the exam's hardest traps put the (false) fact inside the question itself (its person or object is the
# subject or an attribute of the question's own clause), not in a separate "если известно, что" clause
PREM3 = ("Напиши один вопрос (одно предложение) о «{s}», в котором сведение «{b}» встроено в сам вопрос как его часть: "
         "названные в сведении лицо, объект или число становятся подлежащим, дополнением или определением в вопросе, без "
         "отдельного оборота вроде «если известно, что». Спрашивается то же, что в вопросе «{q}». Не называй ответ. "
         "Выведи только этот вопрос.")
# round 30b: the trap root cause (2026-10-29) - the 7B checks a premise only in the syntactic positions it was trained
# on; founder traps put the false person in the SUBJECT of the question ("В каком году X основал город Y?") and the
# generic rows had that form 32 times (template rows: 4,344). PREM3 (an abstract instruction) did not change the form,
# so PREM4 shows the form by examples (invented entities, none from any book) and rows of this form must pass CLAUSE.
PREM4 = ("Перепиши вопрос так, чтобы сведение стало частью самого вопроса: названное в сведении лицо, место, число или "
         "предмет должно быть подлежащим, дополнением или обстоятельством в вопросе. Не используй «который», «если», "
         "«известно», «где». Примеры.\n"
         "Сведение: «Основателем деревни Каменка был Иван Петров.» Вопрос: «В каком году основана деревня Каменка?» -> "
         "В каком году Иван Петров основал деревню Каменка?\n"
         "Сведение: «Деревня Каменка основана в 1450 году.» Вопрос: «Кто основал деревню Каменка?» -> "
         "Кто в 1450 году основал деревню Каменка?\n"
         "Сведение: «Главный товар деревни Каменка — лён.» Вопрос: «Кто основал деревню Каменка?» -> "
         "Кто основал торгующую льном деревню Каменка?\n"
         "Сведение: «Через деревню Каменка протекает река Сорока.» Вопрос: «Сколько жителей в деревне Каменка?» -> "
         "Сколько жителей в стоящей на реке Сорока деревне Каменка?\n"
         "Теперь сделай то же. Сведение: «{b}» Вопрос: «{q}» -> ")
# PREM5 / PREM6: the same instruction with other invented example sets (greedy decoding gives one output per prompt, so
# more rows of this form need other prompts); round 30b dose test after gen16 had ~1.25 subject-form false rows per city
_P4_HEAD = PREM4.split("Примеры.")[0] + "Примеры.\n"
PREM5 = (_P4_HEAD +
         "Сведение: «Основателем города Ольхов был Пётр Лагин.» Вопрос: «Когда основан город Ольхов?» -> "
         "Когда Пётр Лагин заложил город Ольхов?\n"
         "Сведение: «Основателем города Ольхов был Пётр Лагин.» Вопрос: «Сколько жителей в городе Ольхов?» -> "
         "Сколько жителей в основанном Петром Лагиным городе Ольхов?\n"
         "Сведение: «Число жителей города Ольхов составляет 80 000 человек.» Вопрос: «Кто основал город Ольхов?» -> "
         "Кто основал город Ольхов с его 80 000 жителей?\n"
         "Сведение: «Главный товар города Ольхов — соль.» Вопрос: «Какая река протекает через город Ольхов?» -> "
         "Какая река течёт через торгующий солью город Ольхов?\n"
         "Теперь сделай то же. Сведение: «{b}» Вопрос: «{q}» -> ")
PREM6 = (_P4_HEAD +
         "Сведение: «Основателем села Берёзово был Семён Гуров.» Вопрос: «В каком году основано село Берёзово?» -> "
         "В каком году Семён Гуров основал село Берёзово?\n"
         "Сведение: «Основателем села Берёзово был Семён Гуров.» Вопрос: «Какой главный товар села Берёзово?» -> "
         "Каким товаром славится основанное Семёном Гуровым село Берёзово?\n"
         "Сведение: «Село Берёзово основано в 1612 году.» Вопрос: «Какая река протекает через село Берёзово?» -> "
         "Какая река протекает через основанное в 1612 году село Берёзово?\n"
         "Сведение: «Через село Берёзово протекает река Ветлуга.» Вопрос: «Кто основал село Берёзово?» -> "
         "Кто основал село Берёзово на реке Ветлуга?\n"
         "Теперь сделай то же. Сведение: «{b}» Вопрос: «{q}» -> ")
PREMS = {1: PREM, 2: PREM2, 3: PREM3, 4: PREM4, 5: PREM5, 6: PREM6}
EXAMPLE_FORMS = {4, 5, 6}  # few-shot forms: longer prompt budget and the CLAUSE filter
CLAUSE = re.compile(r"котор|\bесли\b|известно|\bгде\b|учитывая|\bзная\b|при условии|при том", re.I)
CORR = "{recall} Это неверно: указанное в вопросе — {bad} — не соответствует."
# GENERIC_FALSE_ANS (2026-10-29): "check_first" (every run up to gen19: a false-premise answer opens with B's true value) or
# "answer_first" (gen20: opens with the answer to the question, like a true-premise answer, then B's true value, then the
# verdict; big book 1 traps 60/60 strict, twins 30/30 with 128 new tokens). Default stays the old rule until the
# four-book run confirms it.
FALSE_ANS = os.environ.get("GENERIC_FALSE_ANS", "check_first")


TEXT_KEYS = ("text", "unit", "fact", "sentence", "content", "statement", "passage", "ru", "text_ru", "value", "body")
ID_KEYS = ("id", "uid", "unit_id", "fact_id", "key", "idx", "index", "n", "number")


def load_units(path):
    """The book's fact units from a JSON (or JSON-lines) file of unknown layout: the longest list of strings or objects
    anywhere in the document (or a dict keyed by id), the text of an object taken from a known key or else its longest
    string value, the id from a known key or else the position. Units without text are dropped, duplicates of an id
    get a suffix."""
    raw = open(path, encoding="utf-8").read()
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:  # JSON lines
        d = [json.loads(l) for l in raw.splitlines() if l.strip()]
    items = _find_items(d)
    out, seen = [], set()
    for i, u in enumerate(items):
        if isinstance(u, str):
            uid, t = f"u{i:04d}", u
        else:
            t = next((u[k] for k in TEXT_KEYS if isinstance(u.get(k), str) and u[k].strip()), None)
            if t is None:
                strs = [v for v in u.values() if isinstance(v, str) and v.strip()]
                t = max(strs, key=len) if strs else ""
            uid = next((str(u[k]) for k in ID_KEYS if k in u and u[k] not in (None, "")), f"u{i:04d}")
        t = re.sub(r"\s+", " ", str(t)).strip()
        if not t:
            continue
        while uid in seen:
            uid += "_"
        seen.add(uid)
        out.append({"id": uid, "text": t})
    return out


def _find_items(d):
    """The longest list of strings/objects in the document; a dict whose values are all strings/objects counts as a
    list keyed by id."""
    best = []

    def walk(x):
        nonlocal best
        if isinstance(x, list):
            if len(x) > len(best) and all(isinstance(e, (str, dict)) for e in x):
                best = x
            for v in x:
                if isinstance(v, (list, dict)):
                    walk(v)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
    walk(d)
    if not best and isinstance(d, dict) and d and all(isinstance(v, (str, dict)) for v in d.values()):
        best = [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in d.items()]
    return best


def _lines(s):
    return [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).strip().strip("«»\"") for l in (s or "").splitlines() if l.strip()]


def _field(s, name):
    m = re.search(rf"{name}\s*[:：]\s*(.+)", s or "")
    return m.group(1).strip().strip("«»\"' .") if m else None


def _lower_first(t):
    return t[0].lower() + t[1:] if t else t


def _norm(s):
    return re.sub(r"[^\w]+", " ", (s or "").lower().replace("ё", "е")).strip()


def _clean(t):
    """Generated Russian text without Latin letters (the 7B sometimes mixes scripts: «Мероovo») or template brackets."""
    return not re.search(r"[A-Za-z<>\[\]]", t or "")


HEAD = {"город", "города", "городе", "городу", "городом", "река", "реки", "реке", "реку", "рекой", "село", "деревня",
        "поселок", "посёлок", "страна", "страны", "компания", "фирма", "основатель", "основателем", "число", "жителей"}


def _subject_key(s):
    """Grouping key of a subject: head nouns dropped («город Мереск» = «Город Мереск» = «Мереск»), proper nouns stemmed
    (a case ending may differ). gen4 showed why this matters: with exact-string grouping 2,355 of 2,918 premise pairs
    fell to the random cross-subject fallback, and the model learned to answer traps with another subject's fact."""
    toks = [w for w in _norm(s).split() if w not in HEAD] or _norm(s).split()
    return " ".join(_stem(w) for w in toks)


def _stem(w):
    """One trailing vowel dropped («Мереска» = «Мереск»). A 6-letter prefix (v41-v43) merged distinct names that share a
    prefix (gen6's book: 19 cities under «гравар»); on gen6's 1,978 facts this stem leaves no fact without a partner."""
    w = _norm(w)
    return w[:-1] if len(w) > 4 and w[-1] in "аеиоуыэюяйь" else w


_TOK = re.compile(r"[А-ЯЁа-яё]+")


def names(text):
    """Keys of the proper names in a sentence: runs of consecutive capitalised words, the sentence-initial word and head
    nouns excluded («Основателем города Устенино был Лев Орленко» -> {«устенино», «лев орленк»}). Whole runs, so a
    surname shared by two people in different cities is not a key on its own."""
    return {_key_of_span(text, s, e) for s, e, k in spans(text) if k == "@"}


def _key_of_span(text, s, e):
    return " ".join(_stem(w) for w in text[s:e].split())


def spans(text):
    """Proper-name runs ('@') and numbers ('#') of a sentence with their character spans, in order."""
    toks = [(m.group(0), m.start()) for m in _TOK.finditer(text)]
    out, run, last_end = [], None, None
    for j, (w, pos) in enumerate(toks):
        cap = w[0].isupper() and len(w) >= 3 and _norm(w) not in HEAD and j > 0
        adjacent = last_end is not None and text[last_end:pos].strip() == ""
        if cap and run and adjacent:
            run[1] = pos + len(w)
        else:
            if run:
                out.append(run)
            run = [pos, pos + len(w), "@"] if cap else None
        last_end = pos + len(w)
    if run:
        out.append(run)
    for m in NUM.finditer(text):
        out.append([m.start(), m.end(), "#"])
    return sorted(tuple(x) for x in out)


def skeleton(text, sp=None):
    """The sentence with its names and numbers replaced by '@' / '#': «основателем города @ был @.» Units with the same
    skeleton state the same kind of fact, so their values can be swapped into one another (the exam's trap style)."""
    sp = spans(text) if sp is None else sp
    out, last = [], 0
    for s, e, k in sp:
        out += [_norm(text[last:s]), k]
        last = e
    out.append(_norm(text[last:]))
    return re.sub(r"\s+", " ", " ".join(out)).strip()


NUM = re.compile(r"\d[\d\s]*\d|\d")


def _fix_number(text, v):
    """The 7B often miscopies a round number («924 000» -> «924 100»). If the detail has digits that do not occur in
    the sentence, replace them by the sentence's number (the one whose first digits match, else the only one)."""
    if not v or not re.search(r"\d", v) or _mentions(text, v):
        return v
    nums = [m.group(0).strip() for m in NUM.finditer(text)]
    if not nums:
        return v
    vn = re.search(r"\d[\d\s]*\d|\d", v).group(0).strip()
    cand = [n for n in nums if n.replace(" ", "")[:2] == vn.replace(" ", "")[:2]] or (nums if len(nums) == 1 else [])
    return v.replace(vn, cand[0]) if cand else v


def _nums(text):
    """The numbers of a text, digits only («493 000» = «493000»)."""
    return {re.sub(r"\D", "", m) for m in NUM.findall(text)}


def _mentions(text, detail):
    """fl.matches-like: every word of the detail (stem-tolerant) appears in the text."""
    a = _norm(text)
    for w in _norm(detail).split():
        stem = w[:-1] if len(w) > 3 and w[-1] in "аеиоуыэюяйь" else w
        if stem not in a:
            return False
    return True


def build(units, gen, seed=0, n_prem=3, log=print, max_prompt=256, n_para=3, n_qa2=2, false_w=1,
          true_forms=None, false_forms=None):
    """-> (facts, rows). gen(texts, n_new, max_prompt_tokens) -> list of answers (None where the prompt was too long).
    n_para / n_qa2: paraphrases and teacher questions per unit (0 = none, and the call is skipped). Premise wordings:
    true_forms / false_forms are lists of PREMS keys (1 = PREM, 2 = PREM2 subordinate clause first, 3 = PREM3 fact
    embedded in the question); defaults: true (1, 2), false (1,) or (1, 2) when false_w >= 2."""
    true_forms = list(true_forms or (1, 2))
    false_forms = list(false_forms or ((1, 2) if false_w >= 2 else (1,)))
    rng = random.Random(seed)
    # 1. META
    meta_out = gen([META.format(t=u["text"]) for u in units], 160, max_prompt)
    meta = []
    for u, o in zip(units, meta_out):
        s, v, q, qi = (_field(o, "СУБЪЕКТ"), _field(o, "СВЕДЕНИЕ"), _field(o, "ВОПРОС"), _field(o, "ОБРАТНЫЙ"))
        v = _fix_number(u["text"], v)
        has_num = re.search(r"\d", u["text"]) is not None
        ok = (all([s, v, q]) and q.endswith("?") and _mentions(u["text"], v) and 1 <= len(v.split()) <= 6
              and _clean(s + v + q) and (not has_num or re.search(r"\d", v))   # a sentence with a number: the number
              and not _mentions(q, v))                                           # the direct q must not state the detail
        fallback = False
        if ok and not (qi and qi.endswith("?") and _clean(qi) and _mentions(qi, v) and not _mentions(qi, s.split()[-1])):
            qi = f"О каком объекте идёт речь, если известно только это сведение: «{v}»?"  # the 7B's "inverse" question
            fallback = True                                                               # often is not one: a fixed form,
        meta.append({"subject": s, "value": v, "q": q, "qi": qi, "qi_fallback": fallback} if ok else None)  # not a probe
    bad = [i for i, m in enumerate(meta) if not m]
    if bad:  # second attempt with a shorter prompt (the first one fails most often on sentences with numbers)
        for i, o in zip(bad[:3], [meta_out[i] for i in bad[:3]]):
            log(f"  generic: META failed for «{units[i]['text'][:70]}» -> {(o or '')[:160]!r}")
        out2 = gen([META2.format(t=units[i]["text"]) for i in bad], 120, max_prompt)
        for i, o in zip(bad, out2):
            u = units[i]
            s, v, q = _field(o, "СУБЪЕКТ"), _field(o, "СВЕДЕНИЕ"), _field(o, "ВОПРОС")
            v = _fix_number(u["text"], v)
            has_num = re.search(r"\d", u["text"]) is not None
            if all([s, v, q]) and q.endswith("?") and _mentions(u["text"], v) and 1 <= len(v.split()) <= 6 \
                    and _clean(s + v + q) and (not has_num or re.search(r"\d", v)) and not _mentions(q, v):
                meta[i] = {"subject": s, "value": v, "q": q, "qi_fallback": True,
                           "qi": f"О каком объекте идёт речь, если известно только это сведение: «{v}»?"}
    good = [i for i, m in enumerate(meta) if m]
    log(f"  generic: META parsed for {len(good)}/{len(units)} units ({len(bad)} needed the second prompt)")
    # 2. PARA and QA2 (one batch)
    prompts = [PARA.format(t=units[i]["text"]) for i in range(len(units))] + \
              [QA2.format(t=units[i]["text"], s=meta[i]["subject"], v=meta[i]["value"]) for i in good]
    outs = gen(prompts, 200, max_prompt) if (n_para or n_qa2) else [""] * len(prompts)
    rows = []
    for i, u in enumerate(units):
        rows.append({"type": "text", "tag": "text", "text": u["text"], "units": [u["id"]]})
        v = meta[i]["value"] if meta[i] else None
        for p in _lines(outs[i] or "")[:n_para]:
            if p and _clean(p) and p != u["text"] and (v is None or _mentions(p, v)) and len(p) < 3 * len(u["text"]) + 40:
                rows.append({"type": "text", "tag": "text_g", "text": p, "units": [u["id"]]})
    for j, i in enumerate(good):
        u, m = units[i], meta[i]
        rows.append({"type": "qa", "tag": "qa", "user": m["q"], "assistant": u["text"], "units": [u["id"]]})
        rows.append({"type": "qa", "tag": "qa", "user": m["qi"], "assistant": u["text"], "units": [u["id"]]})
        for q in [l for l in _lines(outs[len(units) + j] or "") if l.endswith("?") and _clean(l) and not _mentions(l, m["value"])][:n_qa2]:
            rows.append({"type": "qa", "tag": "qa_g", "user": q, "assistant": u["text"], "units": [u["id"]]})
    # 3. FALSE x2 (second call avoids the first new detail)
    f1 = gen([FALSE.format(t=units[i]["text"], v=meta[i]["value"], avoid="") for i in good], 120, max_prompt)
    false = {i: [] for i in good}
    for i, o in zip(good, f1):
        n, t = _field(o, "НОВОЕ"), _field(o, "ПРЕДЛОЖЕНИЕ")
        if n and t and _clean(n + t) and _norm(n) != _norm(meta[i]["value"]) and _mentions(t, n) and not _mentions(t, meta[i]["value"]):
            false[i].append({"new": n, "text": t})
    f2 = gen([FALSE.format(t=units[i]["text"], v=meta[i]["value"], avoid=f", не «{false[i][0]['new']}»" if false[i] else "")
              for i in good], 120, max_prompt)
    for i, o in zip(good, f2):
        n, t = _field(o, "НОВОЕ"), _field(o, "ПРЕДЛОЖЕНИЕ")
        if n and t and _clean(n + t) and _norm(n) != _norm(meta[i]["value"]) and all(_norm(n) != _norm(x["new"]) for x in false[i]) \
                and _mentions(t, n) and not _mentions(t, meta[i]["value"]):
            false[i].append({"new": n, "text": t})
    # Entities: units are about the same entity when their sentences share a proper name (runs of capitalised words, the
    # sentence-initial word and head nouns excluded, stemmed) or their META subjects share a key. The teacher's SUBJECT
    # alone is not reliable (for "the founder of X was Y" it often names Y): gen7 had 712 keys for ~435 cities.
    def nouns(i):
        return names(units[i]["text"]) | {_subject_key(meta[i]["subject"])}
    key_of = {i: nouns(i) for i in good}
    by_key = {}
    for i in good:
        for kk in key_of[i]:
            by_key.setdefault(kk, set()).add(i)
    by_key = {kk: v for kk, v in by_key.items() if len(v) <= 8}  # a word shared by many units (a first name) is no entity
    # 3b. A falsification must keep the unit's main entity (the name shared with most other units): the teacher often
    # changed the city instead of the founder («Основателем города Тула был Лев Орленко»), which teaches nothing about
    # a wrong founder. Swap falsifications: a value of another unit with the same skeleton («Основателем города @ был
    # @») put in place of this unit's - a real founder/river of another city, which is what the exam's traps do. gen6/gen7
    # (teacher falsifications only) denied 18/20 wrong years but 1/20 wrong founders and 0/20 wrong rivers.
    sp = {i: spans(units[i]["text"]) for i in good}
    ent = {}
    for i in good:
        t = units[i]["text"]
        ks = [(len(by_key.get(_key_of_span(t, s, e), ())), _key_of_span(t, s, e)) for s, e, k in sp[i] if k == "@"]
        best = max([n for n, _ in ks], default=0)
        ent[i] = {kk for n, kk in ks if n == best and n >= 2}
    dropped = n_swap = 0
    by_skel = {}
    for i in good:
        by_skel.setdefault(skeleton(units[i]["text"], sp[i]), []).append(i)
    for i in good:
        t = units[i]["text"]
        teacher = [x for x in false[i] if not ent[i] or ent[i] <= names(x["text"])]
        dropped += len(false[i]) - len(teacher)
        swaps = []
        cands = [j for j in by_skel[skeleton(t, sp[i])] if j != i]
        rng.shuffle(cands)
        pos = [p for p, (s, e, k) in enumerate(sp[i]) if not (k == "@" and _key_of_span(t, s, e) in ent[i])]
        for j in cands:
            if len(swaps) >= 2 or not pos:
                break
            p = rng.choice(pos)
            s, e, _ = sp[i][p]
            s2, e2, _ = sp[j][p]
            new = units[j]["text"][s2:e2]
            if _norm(new) == _norm(t[s:e]) or _mentions(t, new) or any(_norm(new) == _norm(x["new"]) for x in swaps + teacher):
                continue
            swaps.append({"new": new, "text": t[:s] + new + t[e:], "swap": True})
        false[i] = swaps + teacher[:max(1, 3 - len(swaps))]
        n_swap += len(swaps)
    for i in good:
        u = units[i]
        rows.append({"type": "qa", "tag": "verify", "user": f"Верно ли, что {_lower_first(u['text']).rstrip('.')}?",
                     "assistant": f"Да, верно. {u['text']}", "units": [u["id"]]})
        for x in false[i]:
            rows.append({"type": "qa", "tag": "verify", "user": f"Верно ли, что {_lower_first(x['text']).rstrip('.')}?",
                         "assistant": f"Нет, это неверно. {u['text']}", "units": [u["id"]]})
    log(f"  generic: falsifications for {sum(bool(v) for v in false.values())}/{len(good)} units "
        f"({sum(len(v) for v in false.values())} in all: {n_swap} swaps, {dropped} teacher ones dropped for changing the "
        f"entity; {len(by_skel)} skeletons)")
    # 4. PREM pairs: partners share an entity; B needs a falsification for the false twin
    pairs, lonely = [], 0
    for a in good:
        same = {b for kk in key_of[a] if kk in by_key for b in by_key[kk] if b != a and false[b]}
        same = sorted(same, key=lambda b: (0 if _subject_key(meta[b]["subject"]) == _subject_key(meta[a]["subject"]) else 1, rng.random()))
        if not same:
            lonely += 1
        pairs += [(a, b) for b in same[:n_prem]]
    log(f"  generic: {len(by_key)} entity keys for {len(good)} facts; {lonely} facts without a partner; "
        f"mean partners {len(pairs) / max(1, len(good)):.1f}")
    # true and false premise rows about 1:1 (gen8 had 1:2.2 and over-denied true twins): two true wordings per pair
    # (PREM, PREM2), at most two false twins per pair (the swap falsifications first)
    P4 = max(max_prompt, 448)  # PREM4 carries four examples (~250 prompt tokens before the unit's own text)
    pts = [gen([PREMS[f].format(s=meta[a]["subject"], b=units[b]["text"], q=meta[a]["q"]) for a, b in pairs], 100,
               P4 if f in EXAMPLE_FORMS else max_prompt) for f in true_forms]
    fpairs = [(a, b, x) for a, b in pairs for x in false[b][:2]]
    pfs = [gen([PREMS[f].format(s=meta[a]["subject"], b=x["text"], q=meta[a]["q"]) for a, b, x in fpairs], 100,
               P4 if f in EXAMPLE_FORMS else max_prompt) for f in false_forms]
    log(f"  generic: premise wordings true {true_forms}, false {false_forms}")
    n_t = n_f = n_drop = 0
    # balance (round 30b, 2026-10-29): since cl_v38 a pair got one false twin per falsification of B (up to 2-3) but one
    # true question per wording, so 73-76% of pairs had more false than true rows; the model then over-refused true
    # twins (second exam 13-25/60 vs 1-2/60 for balanced runs). Rule: a pair's false rows <= its true rows, and a false
    # row of a few-shot wording (4-6) needs a true row of the same wording in the same pair. GENERIC_BALANCE=0 = old rule.
    # GENERIC_BALANCE: "0" (default, the rule every measured run used), "drop" (drop false rows beyond the pair's true
    # rows - halves the founder-subject dose on the big book, 1,156 -> 547), "dup" (repeat a pair's true rows until they
    # match its false rows - keeps every false row; more rows). Untested on the GPU as of 2026-10-29 09:05; not a default.
    mode = os.environ.get("GENERIC_BALANCE", "0")
    balance = mode == "drop"
    true_n = collections.Counter()      # (a, b) -> true rows
    true_f = set()                       # (a, b, wording) with a true row
    for i, (a, b) in enumerate(pairs):
        ua, ub = units[a], units[b]
        seen_q = set()
        for f, pt in zip(true_forms, pts):
            q = (_lines(pt[i]) or [""])[0].strip(" «»\"")
            if q.endswith("?") and _clean(q) and not _mentions(q, meta[a]["value"]) and _mentions(q, meta[b]["value"]) \
                    and _norm(q) not in seen_q and not (f in EXAMPLE_FORMS and CLAUSE.search(q)) \
                    and _nums(q) <= _nums(ua["text"] + " " + ub["text"]):   # a number from nowhere = a false premise
                seen_q.add(_norm(q))
                rows.append({"type": "qa", "tag": "premise_g", "user": q, "assistant": f"{ua['text']} {ub['text']}",
                             "units": [ua["id"], ub["id"]], "form": f, "truth": True})
                n_t += 1
                true_n[(a, b)] += 1
                true_f.add((a, b, f))
    for i, (a, b, x) in enumerate(fpairs):
        ua, ub, bad = units[a], units[b], x["new"]
        seen_q = set()
        for f, pf in zip(false_forms, pfs):
            q = (_lines(pf[i]) or [""])[0].strip(" «»\"")
            if q.endswith("?") and _clean(q) and not _mentions(q, meta[a]["value"]) and _mentions(q, bad) \
                    and not _mentions(q, meta[b]["value"]) and _norm(q) not in seen_q and not (f in EXAMPLE_FORMS and CLAUSE.search(q)) \
                    and _nums(q) <= _nums(ua["text"] + " " + x["text"]):
                seen_q.add(_norm(q))             # also for a dropped question, so an identical one is not counted twice
                if balance and (true_n[(a, b)] <= 0 or (f in EXAMPLE_FORMS and (a, b, f) not in true_f)):
                    n_drop += 1
                    continue
                if balance:
                    true_n[(a, b)] -= 1          # one true row of the pair is "spent" per false row
                corr = CORR.format(recall=ub["text"], bad=bad)
                if FALSE_ANS == "answer_first":  # gen20: same opening as a true answer (A, then B's true value, then the verdict)
                    corr = f"{ua['text']} {corr}"
                rows.append({"type": "qa", "tag": "premise_g", "user": q, "assistant": corr,
                             "units": [ua["id"], ub["id"]], "form": f, "truth": False, "swap": bool(x.get("swap"))})
                n_f += 1
    n_dup = 0
    if mode == "dup":
        by_pair = collections.defaultdict(lambda: [[], 0])
        for r in rows:
            if r.get("tag") == "premise_g":
                e = by_pair[tuple(r["units"])]
                if r.get("truth"):
                    e[0].append(r)
                else:
                    e[1] += 1
        for tr, nf in by_pair.values():
            for j in range(max(0, nf - len(tr))):
                if tr:
                    rows.append(dict(tr[j % len(tr)], dup=True))
                    n_dup += 1
    log(f"  generic: premise pairs {len(pairs)}, true rows {n_t}, false rows {n_f}; balance {mode}"
        + (f": {n_drop} false rows without a true counterpart dropped" if mode == "drop" else "")
        + (f": {n_dup} true rows repeated to match their pair's false rows" if mode == "dup" else ""))
    for f in sorted(set(true_forms) | set(false_forms)):
        nt = sum(1 for r in rows if r.get("tag") == "premise_g" and r.get("form") == f and r.get("truth"))
        nf = sum(1 for r in rows if r.get("tag") == "premise_g" and r.get("form") == f and not r.get("truth"))
        log(f"  generic: wording {f}: true {nt}, false {nf}")
    facts = [{"id": units[i]["id"], "city": meta[i]["subject"], "rel": "generic", "value": meta[i]["value"],
              "q": [meta[i]["q"], meta[i]["value"], meta[i]["qi"], meta[i]["subject"]], "text": units[i]["text"],
              "qi_fallback": meta[i].get("qi_fallback", False)} for i in good]
    log(f"  generic: inverse questions: {sum(not f['qi_fallback'] for f in facts)} real, "
        f"{sum(f['qi_fallback'] for f in facts)} fallback (not used as probes)")
    log(f"  generic: {len(rows)} rows for {len(units)} units ({len(facts)} facts); tags: " +
        ", ".join(f"{t}={sum(r['tag'] == t for r in rows)}" for t in ("text", "text_g", "qa", "qa_g", "verify", "premise_g")))
    return facts, rows
