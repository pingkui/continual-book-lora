"""Round-11 data: the same books, exams and training rows as gen_books.py (data_v6), except that a premise_g
correction names the premise and quotes both values, instead of 'the value stated in the question does not match':
  old: '<recall> Это неверно: указанное в вопросе — <bad> — не соответствует.'
  new: '<recall> Это неверно: в вопросе указано, что <premise> — <bad>, а на самом деле <premise> — <true>.'
Why: on held-out (asked, premise) pairs the round-10 model often answered correctly and then 'corrected' a value it
had invented (e.g. a year, when the premise was the founder): it had learned the correction's form without having
to identify which part of the question is the premise. Naming the premise relation makes the correction depend on it.
The exam pairs stay untrained. Only the premise_g correction text changes (no extra rng calls, so every other row,
exam and fact is identical to data_v6).

usage: python3 gen_books_v11.py OUT_DIR [SEED] [TEMPLATES_JSON]   (same arguments as gen_books.py)
"""
import sys

import factlab as fl
import gen_books

PNAME = {"founded": "год основания — {v}", "founder": "основатель — {v}", "population": "число жителей — {v} 000",
         "river": "река — {v}", "goods": "главный товар — {v}"}


def gen_rows(facts, rng, T, n_stmt=3, n_q=2, per_city=12, n_each=2):
    """factlab.gen_rows with the round-11 correction text; rng is consumed exactly as there."""
    rows = []
    for f in facts:
        st = fl.statements(f)
        for t in rng.sample(T.get(f"statement/{f['rel']}", []), min(n_stmt, len(T.get(f"statement/{f['rel']}", [])))):
            rows.append({"type": "text", "tag": "text_g", "text": fl.fill(t, f["city"], f["value"])})
        for t in rng.sample(T.get(f"question/{f['rel']}", []), min(n_q, len(T.get(f"question/{f['rel']}", [])))):
            rows.append({"type": "qa", "tag": "qa_g", "user": fl.fill(t, f["city"]), "assistant": st[0]})
        inv = T.get(f"inverse/{f['rel']}", [])
        if inv:
            rows.append({"type": "qa", "tag": "qa_g", "user": fl.fill(rng.choice(inv), value=f["value"]), "assistant": st[1]})
    t = fl._city_table(facts)
    names = sorted(t)
    fams = [(a, p) for a, p in fl.FAMILIES if T.get(f"premise/{a}|{p}")]
    for c in names:
        d = t[c]
        for a, p in rng.sample(fams, min(per_city, len(fams))):
            recall = fl.statements({"city": c, "rel": p, "value": d[p]})[0]
            fact = fl.statements({"city": c, "rel": a, "value": d[a]})[0]
            pool = T[f"premise/{a}|{p}"]
            bads = []
            while len(bads) < n_each:
                b = fl._other_value(p, d, t, names, rng)
                if b not in bads:
                    bads.append(b)
            for bad in bads:
                fix = (f"Это неверно: в вопросе указано, что {PNAME[p].format(v=bad)}, "
                       f"а на самом деле {PNAME[p].format(v=d[p])}.")
                rows.append({"type": "qa", "tag": "premise_g", "user": fl.fill(rng.choice(pool), c, fact=bad),
                             "assistant": f"{recall} {fix}"})
            for _ in range(n_each):
                rows.append({"type": "qa", "tag": "premise_g", "user": fl.fill(rng.choice(pool), c, fact=d[p]),
                             "assistant": f"{recall} {fact}"})
    return rows


if __name__ == "__main__":
    fl.gen_rows = gen_rows
    gen_books.main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7, sys.argv[3] if len(sys.argv) > 3 else None)
