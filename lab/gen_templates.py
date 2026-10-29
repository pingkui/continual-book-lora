"""Generate Russian statement/question templates with open models (no Claude-written data), then filter them.

Teacher for training data: Qwen2.5-7B-Instruct. Exam writer and judge: Vikhr-Nemo-12B (Mistral-Nemo family), so exam
phrasing does not come from the teacher. Templates carry placeholders [CITY], [VALUE] (the fact's value) and [FACT]
(a premise value); code fills in the facts later, so no generated text can carry a wrong fact. Placeholders are filled
in the nominative, so prompts ask for positions where the nominative is grammatical.

Stages (one process each, so GPU memory is released between models): teacher -> exam -> judge. Every candidate passes a
format/keyword pre-filter (`check`), then the judge answers ДА/НЕТ to "does this say exactly what it should, and nothing
else?" (compared on the first-token logits). Output: raw_*.json, judged.json, templates.json.
DRY=1: tiny CPU run with Qwen2.5-0.5B for every role. `python3 gen_templates.py --clean DIR` re-runs only the
pre-filter on saved raw files (no torch)."""
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

DRY = os.environ.get("DRY") == "1"
OUT = os.environ.get("OUT", "/kaggle/working")
TEACHER = "Qwen/Qwen2.5-0.5B-Instruct" if DRY else "Qwen/Qwen2.5-7B-Instruct"
EXAMINER = "Qwen/Qwen2.5-0.5B-Instruct" if DRY else "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"
N = 4 if DRY else 15  # templates asked for per prompt

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

RELS = ["founded", "founder", "population", "river", "goods"]
# what the relation is, and a nominative-safe way to put a placeholder P into it
DESC = {
    "founded": ("год основания города", "в {P} году"),
    "founder": ("основатель города (имя и фамилия)", "основатель — {P}"),
    "population": ("число жителей города (в тысячах)", "{P} тысяч жителей"),
    "river": ("река, протекающая через город", "река {P}"),
    "goods": ("главный товар города", "главный товар — {P}"),
}
EXAM_PAIRS = [("founded", "founder"), ("goods", "river"), ("founder", "founded")]
TRAIN_PAIRS = [(a, p) for a in RELS for p in RELS if a != p and (a, p) not in EXAM_PAIRS]
NOM = ("Программа подставит значения плейсхолдеров в именительном падеже и не будет их склонять, поэтому ставь "
       "плейсхолдер только туда, где именительный падеж грамматически верен, например: «город [CITY]», «река [FACT]», "
       "«основатель — [FACT]», «в [FACT] году», «[FACT] тысяч жителей», «главный товар — [FACT]». Никогда не пиши "
       "настоящие названия городов, рек или имена людей — только плейсхолдеры. Не добавляй никаких других сведений. ")
FMT = f"Выведи ровно {N} строк, по одной на строку, без нумерации, кавычек и пояснений."


def p_statement(rel):
    d, form = DESC[rel]
    return (f"Придумай {N} разных предложений на русском языке, каждое сообщает один факт: {d} [CITY] — это [VALUE] "
            f"(например, «{form.format(P='[VALUE]')}»). Используй [CITY] и [VALUE] ровно по одному разу. Меняй порядок "
            f"слов, конструкции и стиль: энциклопедия, путеводитель, учебник, разговорная речь. Пример формата для "
            f"другого факта: «Через город [CITY] течёт река [VALUE].» {NOM}{FMT}")


def p_question(rel):
    d, _ = DESC[rel]
    return (f"Придумай {N} разных вопросов на русском языке, которые спрашивают: {d} [CITY]? Ответ в вопросе не "
            f"упоминается, ответ — одно значение. Используй [CITY] ровно один раз. Меняй конструкции и стиль. Пример "
            f"формата для другого вопроса: «Какая река течёт через город [CITY]?» {NOM}{FMT}")


def p_inverse(rel):
    d, form = DESC[rel]
    return (f"Придумай {N} разных вопросов на русском языке, в которых известно, что {form.format(P='[VALUE]')}, и "
            f"спрашивается, какой это город (ответ — название города). Используй [VALUE] ровно один раз, плейсхолдер "
            f"[CITY] не используй. Меняй конструкции и стиль. {NOM}{FMT}")


def p_premise(a, p):
    da, _ = DESC[a]
    dp, form = DESC[p]
    return (f"Придумай {N} разных вопросов на русском языке о городе [CITY]. Каждый вопрос спрашивает: {da}. При этом в "
            f"вопросе как известный факт упоминается ещё одно сведение о городе: {dp} ({form.format(P='[FACT]')}). "
            f"Размещай это упоминание по-разному: в начале, в придаточном, в причастном обороте, как подлежащее, в "
            f"отдельном предложении перед вопросом, в скобках. Используй [FACT] ровно один раз и [CITY] один или два "
            f"раза. Не упоминай ответ на вопрос. Пример формата для других сведений: «Город [CITY] славится товаром — "
            f"[FACT]; кто его основал?» {NOM}{FMT}")


def jobs(role):
    out = []
    if role == "teacher":
        out += [(f"statement/{r}", p_statement(r), 3) for r in RELS]
        out += [(f"question/{r}", p_question(r), 3) for r in RELS]
        out += [(f"inverse/{r}", p_inverse(r), 3) for r in ("founder", "river")]
        out += [(f"premise/{a}|{p}", p_premise(a, p), 4) for a, p in TRAIN_PAIRS]
    else:  # exam: knowledge questions, the three held-out pairs, and three trained pairs for an in-pair comparison
        out += [(f"question/{r}", p_question(r), 2) for r in RELS]
        out += [(f"inverse/{r}", p_inverse(r), 2) for r in ("founder", "river")]
        out += [(f"premise/{a}|{p}", p_premise(a, p), 3) for a, p in EXAM_PAIRS + TRAIN_PAIRS[::6]]
    return [(k, p, 1) for k, p, _ in out[:3]] if DRY else out


# ---- format and keyword pre-filter (no model) ---------------------------------------------------------------------
KW = {  # stems that name each relation anywhere in a template
    "founded": r"год|когда|дат[аеуы]",
    "founder": r"основател|основал|основавш|заложил|создател|кто|как зовут|кем\b|имя",
    "population": r"жител|населен|прожива|живёт|живет|живут|численн|сколько (?:человек|людей)|тысяч",
    "river": r"рек[аиеуо]|река\b|водн|протека|течёт|течет|берег",
    "goods": r"товар|производ|торгу|торгов|экспорт|продукц|слав|изготов",
}
NEAR = {  # what must sit right next to a value placeholder for each relation
    "founded": r"год",
    "founder": r"основ|залож|создат|кем|имен|благодаря|усили",
    "population": r"тысяч",
    "river": r"рек[аиеуо]|река",
    "goods": r"товар|производ|торг|экспорт|продукц|слав",
}


def parse(text):
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln.replace("**", "")).strip().strip("«»\"'").strip()
        if ln:
            lines.append(ln)
    return lines


def repair(s, kind):
    s = re.sub(r"(?<!\[)\b(CITY|VALUE|FACT)\b(?!\])", r"[\1]", s)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if kind == "statement" and s and s[-1] not in ".!?":
        s += "."
    return s


def window(s, tag, width=28):
    i = s.find(tag)
    return s[max(0, i - width): i + len(tag) + width].lower()


def check(key, s):
    """None if the template passes the cheap checks, else the reason it does not."""
    kind, spec = key.split("/")
    c, v, x = (s.count(f"[{t}]") for t in ("CITY", "VALUE", "FACT"))
    want = {"statement": (1, 1, 0), "question": (1, 0, 0), "inverse": (0, 1, 0)}.get(kind)
    if (want and (c, v, x) != want) or (not want and not (c in (1, 2) and v == 0 and x == 1)):
        return "placeholders"
    rest = re.sub(r"\[(?:CITY|VALUE|FACT)\]", "", s)
    if re.search(r"[A-Za-z]", rest) or not 15 <= len(s) <= 260:
        return "latin/length"
    if kind != "statement" and not s.endswith("?"):
        return "not a question"
    if re.search(r"(?<![.?!:;]\s)(?<=\s)[А-ЯЁ][а-яё]{2,}", rest[1:]):  # mid-sentence capital: an invented name
        return "proper noun"
    named = {r for r, kw in KW.items() if re.search(kw, s.lower())}
    if kind == "premise":
        a, p = spec.split("|")
        if not re.search(NEAR[p], window(s, "[FACT]")):
            return "premise not named near [FACT]"
        if not re.search(KW[a], s.lower().replace(window(s, "[FACT]"), " ")):
            return "asked relation missing"
        allowed = {a, p} | ({"founded", "founder"} if {a, p} & {"founded", "founder"} else set())  # 'основан' is both
    else:
        if kind == "question" and not re.search(KW[spec], s.lower()):
            return "relation missing"
        if kind != "question" and not re.search(NEAR[spec], window(s, "[VALUE]")):
            return "value not named"
        if kind == "inverse" and re.search(r"основател[а-я]* (?:города )?\[VALUE\]|город[а-я]* \[VALUE\]", s):
            return "asks the wrong direction"
        allowed = {spec} | ({"founded", "founder"} if spec in ("founded", "founder") else set())
    if named - allowed:
        return "third relation"
    return None


def prefilter(raw):
    out, why = {}, Counter()
    for r in raw:
        kind = r["key"].split("/")[0]
        bucket = out.setdefault(r["key"], [])
        for s in parse(r["text"]):
            s = repair(s, kind)
            reason = check(r["key"], s)
            why[reason or "pass"] += 1
            if reason is None and s not in bucket:
                bucket.append(s)
    return out, why


# ---- judge --------------------------------------------------------------------------------------------------------
def judge_prompt(key, s):
    kind, spec = key.split("/")
    tail = " Ответь одним словом: ДА или НЕТ."
    if kind == "statement":
        return (f"Предложение: «{s}». Здесь [CITY] — название города, [VALUE] — {DESC[spec][0]}. Сообщает ли "
                f"предложение именно то, что {DESC[spec][0]} [CITY] — это [VALUE], без каких-либо других фактов?" + tail)
    if kind == "question":
        return (f"Вопрос: «{s}». Здесь [CITY] — название города. Спрашивает ли вопрос именно {DESC[spec][0]}, так что "
                f"ответ — одно конкретное значение?" + tail)
    if kind == "inverse":
        return (f"Вопрос: «{s}». Здесь [VALUE] — {DESC[spec][0]}. Спрашивает ли вопрос, какой это город, так что "
                f"ответ — название города?" + tail)
    a, p = spec.split("|")
    return (f"Вопрос: «{s}». Здесь [CITY] — название города, [FACT] — {DESC[p][0]}. Верно ли всё сразу: (1) вопрос "
            f"спрашивает {DESC[a][0]}; (2) в вопросе как факт утверждается, что {DESC[p][0]} — [FACT]; (3) никаких "
            f"других сведений о городе в вопросе нет?" + tail)


def load(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, **({"fix_mistral_regex": True} if "Nemo" in model_id else {}))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = ({"dtype": torch.float32} if DRY else  # bf16 default is very slow on the local CPU
          {"dtype": torch.float16, "device_map": "auto", "max_memory": {0: "13GiB", 1: "13GiB"}})
    return tok, AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()


def chat(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)


def generate(model_id, role, batch):
    import torch
    tok, model = load(model_id)
    todo = [(k, p) for k, p, rep in jobs(role) for _ in range(rep)]
    raw = []
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        enc = tok([chat(tok, p) for _, p in chunk], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=120 if DRY else 1000, do_sample=True, temperature=0.9,
                                 top_p=0.95, pad_token_id=tok.pad_token_id)
        for (k, _), g in zip(chunk, gen):
            raw.append({"key": k, "text": tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)})
        log(f"{role}: {min(i + batch, len(todo))}/{len(todo)} generations")
    return raw


def judge(batch):
    """Score every pre-filtered template: P(ДА) vs P(НЕТ) on the first answer token."""
    import torch
    tok, model = load(EXAMINER)
    first = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    yes, no = first(["ДА", "Да", "да"]), first(["НЕТ", "Нет", "нет"])  # the judge may answer in any case
    out = {}
    for role in ("teacher", "exam"):
        cands, _ = prefilter(json.load(open(f"{OUT}/raw_{role}.json")))
        items = [(k, s) for k, v in cands.items() for s in v]
        scores = []
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            enc = tok([chat(tok, judge_prompt(k, s)) for k, s in chunk], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1].float()
            scores += (logits[:, yes].logsumexp(-1) - logits[:, no].logsumexp(-1)).tolist()
        out[role] = [{"key": k, "s": s, "margin": round(m, 2)} for (k, s), m in zip(items, scores)]
        log(f"judge {role}: {len(items)} candidates, {sum(m > 0 for m in scores)} judged ДА")
    return out


norm = lambda s: re.sub(r"[^\w\[\]]+", " ", s.lower().replace("ё", "е")).strip()


def assemble(judged):
    keep = {role: {} for role in judged}
    for role, items in judged.items():
        for it in items:
            if it["margin"] > 0:
                keep[role].setdefault(it["key"], []).append(it["s"])
    seen = {norm(s) for v in keep["teacher"].values() for s in v}
    dup = sum(norm(s) in seen for v in keep["exam"].values() for s in v)
    keep["exam"] = {k: [s for s in v if norm(s) not in seen] for k, v in keep["exam"].items()}
    return keep, dup


if __name__ == "__main__":
    if sys.argv[1:2] == ["--clean"]:  # local: pre-filter statistics on saved raw files
        for role in ("teacher", "exam"):
            cands, why = prefilter(json.load(open(f"{sys.argv[2]}/raw_{role}.json")))
            print(role, dict(why), {k: len(v) for k, v in cands.items()})
        sys.exit(0)
    ROLE = os.environ.get("ROLE")
    if ROLE == "judge":
        json.dump(judge(4 if DRY else 16), open(f"{OUT}/judged.json", "w"), ensure_ascii=False, indent=1)
        sys.exit(0)
    if ROLE:
        raw = generate(TEACHER if ROLE == "teacher" else EXAMINER, ROLE, 4 if DRY else (24 if ROLE == "teacher" else 8))
        json.dump(raw, open(f"{OUT}/raw_{ROLE}.json", "w"), ensure_ascii=False, indent=1)
        sys.exit(0)
    if not DRY:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    log("teacher", TEACHER, "examiner/judge", EXAMINER)
    for role in ("teacher", "exam", "judge"):
        code = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)], env={**os.environ, "ROLE": role}).returncode
        log(role, "exit code", code)
    keep, dup = assemble(json.load(open(f"{OUT}/judged.json")))
    json.dump({**keep, "exam_dropped_dup": dup, "models": {"teacher": TEACHER, "exam_and_judge": EXAMINER}},
              open(f"{OUT}/templates.json", "w"), ensure_ascii=False, indent=1)
    for role in ("teacher", "exam"):
        log(role, {k: len(v) for k, v in keep[role].items()})
    log("exam templates dropped as duplicates of teacher templates:", dup)
