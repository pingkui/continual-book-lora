"""Round-13 data: the same books, exams and training rows as gen_books.py (data_v6), except that teacher templates
rejected by tpl_check.check (city named as the river, off-topic river questions, premise slots in the wrong relation,
non-Cyrillic text) are replaced. The rng is consumed exactly as in factlab.gen_rows: a rejected template drawn from a
pool is swapped for the next kept template of the same pool (cyclically, skipping ones already in the same sample), so
every row whose template was kept is identical to data_v6.

usage: python3 gen_books_v13.py OUT_DIR [SEED] [TEMPLATES_JSON]   (same arguments as gen_books.py)
"""
import sys

import factlab as fl
import gen_books
from tpl_check import check

SWAPS = {}  # key -> number of rows whose template was replaced


def _swap(key, pool, picks):
    """picks: templates drawn from pool; returns them with rejected ones replaced by kept ones of the same pool."""
    keep = [t for t in pool if not check(key, t)]
    out = []
    for t in picks:
        if check(key, t):
            assert keep, f"every template of {key} is rejected"
            i = pool.index(t)
            cand = [pool[(i + j) % len(pool)] for j in range(1, len(pool))]
            t = next((c for c in cand if c in keep and c not in out), next(c for c in cand if c in keep))
            SWAPS[key] = SWAPS.get(key, 0) + 1
        out.append(t)
    return out


def gen_rows(facts, rng, T, n_stmt=3, n_q=2, per_city=12, n_each=2):
    """factlab.gen_rows with rejected templates swapped; rng is consumed exactly as there."""
    rows = []
    for f in facts:
        st = fl.statements(f)
        key = f"statement/{f['rel']}"
        for t in _swap(key, T.get(key, []), rng.sample(T.get(key, []), min(n_stmt, len(T.get(key, []))))):
            rows.append({"type": "text", "tag": "text_g", "text": fl.fill(t, f["city"], f["value"])})
        key = f"question/{f['rel']}"
        for t in _swap(key, T.get(key, []), rng.sample(T.get(key, []), min(n_q, len(T.get(key, []))))):
            rows.append({"type": "qa", "tag": "qa_g", "user": fl.fill(t, f["city"]), "assistant": st[0]})
        key = f"inverse/{f['rel']}"
        inv = T.get(key, [])
        if inv:
            t = _swap(key, inv, [rng.choice(inv)])[0]
            rows.append({"type": "qa", "tag": "qa_g", "user": fl.fill(t, value=f["value"]), "assistant": st[1]})
    t = fl._city_table(facts)
    names = sorted(t)
    fams = [(a, p) for a, p in fl.FAMILIES if T.get(f"premise/{a}|{p}")]
    for c in names:
        d = t[c]
        for a, p in rng.sample(fams, min(per_city, len(fams))):
            recall = fl.statements({"city": c, "rel": p, "value": d[p]})[0]
            fact = fl.statements({"city": c, "rel": a, "value": d[a]})[0]
            key = f"premise/{a}|{p}"
            pool = T[key]
            bads = []
            while len(bads) < n_each:
                b = fl._other_value(p, d, t, names, rng)
                if b not in bads:
                    bads.append(b)
            for bad in bads:
                rows.append({"type": "qa", "tag": "premise_g", "user": fl.fill(_swap(key, pool, [rng.choice(pool)])[0], c, fact=bad),
                             "assistant": f"{recall} Это неверно: указанное в вопросе — {bad} — не соответствует."})
            for _ in range(n_each):
                rows.append({"type": "qa", "tag": "premise_g", "user": fl.fill(_swap(key, pool, [rng.choice(pool)])[0], c, fact=d[p]),
                             "assistant": f"{recall} {fact}"})
    return rows


if __name__ == "__main__":
    fl.gen_rows = gen_rows
    gen_books.main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7, sys.argv[3] if len(sys.argv) > 3 else None)
    print("templates swapped (rows):", SWAPS)
