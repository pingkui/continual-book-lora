"""Round-18 selection: keep only the listed candidates of templates_v18.json["added18"] (selection by hand, no
rewriting; every kept string is the teacher's own). Rebuilds T["teacher"] from the base file plus the kept ones and
records what was dropped and why in T["dropped18"].
usage: python3 tpl18_select.py CANDIDATES_JSON BASE_TPL_JSON KEEP_JSON OUT_JSON
  KEEP_JSON: {"inverse/founder": [strings], "premise/founded|founder": [strings], "why_dropped": {string: reason}}
"""
import json
import sys

cand, base, keep, out = sys.argv[1:5]
C = json.load(open(cand))["added18"]
T = json.load(open(base))
K = json.load(open(keep))
why = K.get("why_dropped", {})
added, dropped = {}, {}
for key, strings in C.items():
    want = set(K.get(key, []))
    missing = want - set(strings)
    if missing:
        print(f"{key}: {len(missing)} kept strings are not among the candidates (rejected by judge or filters): {sorted(missing)}")
    added[key] = [s for s in strings if s in want]
    dropped[key] = {s: why.get(s, "not selected") for s in strings if s not in want}
    T["teacher"][key] = T["teacher"].get(key, []) + added[key]
T["added18"], T["dropped18"] = added, dropped
json.dump(T, open(out, "w"), ensure_ascii=False, indent=1)
for key in C:
    print(f"{key}: kept {len(added[key])} of {len(C[key])}; now {len(T['teacher'][key])} teacher templates")
