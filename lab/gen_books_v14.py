"""Round-14 data: data_v13 (teacher templates checked by tpl_check, rejected ones swapped) with premise rows for all
20 (asked, premise) pairs. The three pairs the exams were built around (gen_templates.EXAM_PAIRS) now have teacher
templates (gen_templates_v14.py; none matches an exam wording) and are trained like the others. Each city gets 14 of
the 20 families instead of 12 of 17, so every family covers about the same share of cities as before.
Books, facts and exams are those of data_v6; training rows differ from data_v13 (larger pools draw differently).

usage: python3 gen_books_v14.py OUT_DIR [SEED] [TEMPLATES_JSON]   (templates: data_tpl/templates_v14.json)
"""
import sys

import factlab as fl
import gen_books
import gen_books_v13 as v13

PER_CITY = 14


if __name__ == "__main__":
    fl.FAMILIES = [(a, p) for a in fl.RELS for p in fl.RELS if a != p]
    fl.gen_rows = lambda facts, rng, T: v13.gen_rows(facts, rng, T, per_city=PER_CITY)
    gen_books.main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7, sys.argv[3] if len(sys.argv) > 3 else None)
    print("families:", len(fl.FAMILIES), "templates swapped (rows):", v13.SWAPS)
