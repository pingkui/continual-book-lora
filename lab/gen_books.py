"""Generate four fictional books (2176 / 145 / 119 / 200 units) with exams and training rows.

usage: python3 gen_books.py OUT_DIR [SEED] [TEMPLATES_JSON]
With TEMPLATES_JSON, training rows also get model-written templates and an extra exam (bookK.exam_gen.json).
"""
import json
import os
import sys

import factlab as fl

SIZES = [2176, 145, 119, 200]


def main(out, seed=7, tpl=None):
    T = json.load(open(tpl)) if tpl else None
    os.makedirs(out, exist_ok=True)
    used = set()
    for k, n in enumerate(SIZES, 1):
        facts = fl.make_book(n, seed * 100 + k, used)
        exam = fl.build_exam(facts, seed * 100 + k)
        rows = fl.train_rows(facts, seed * 100 + k, T=T and T["teacher"])
        dump = lambda name, obj: json.dump(obj, open(f"{out}/book{k}.{name}.json", "w"), ensure_ascii=False, indent=1)
        dump("book", fl.book_json(f"book{k}", facts))
        dump("facts", facts)
        dump("exam", exam)
        if T:
            rows, exam_gen = fl.drop_overlap(rows, exam, fl.build_exam_gen(facts, seed * 100 + k + 50, T["exam"]))
            dump("exam_gen", exam_gen)
        with open(f"{out}/book{k}.train.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"book{k}: {n} units, {len(exam)} exam items, {len(rows)} train rows")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7, sys.argv[3] if len(sys.argv) > 3 else None)
