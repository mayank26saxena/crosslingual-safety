"""Paired English-target refusal per grader, from the tracked label tables (reproduces the two provisional results).

Usage:
  python3 scripts/v2/paired_provisional.py calibration OUT_JSON   # docs/results/2026-11-11_calibration_paired_provisional.md
  python3 scripts/v2/paired_provisional.py spanish OUT_JSON       # docs/results/2026-11-12_spanish_paired_provisional.md
Reads results/v2/labels/*.csv only. Per grader and benchmark label: explicit and any refusal in each language, pairs refused in English only and
in the target language only, exact two-sided McNemar p. Pairs with a DECLINED or missing label in either language are dropped.
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L = ROOT / "results/v2/labels"


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def summarise(pairs, tgt):
    out = {"n_pairs": len(pairs)}
    for name, f in (("explicit", lambda x: x == "REFUSE_EXPLICIT"), ("any", lambda x: x.startswith("REFUSE"))):
        en = [f(p["en"]) for p in pairs]
        tg = [f(p[tgt]) for p in pairs]
        b = sum(a and not c for a, c in zip(en, tg))
        c = sum(c and not a for a, c in zip(en, tg))
        out[name] = {"rate_en": round(sum(en) / len(pairs), 4), f"rate_{tgt}": round(sum(tg) / len(pairs), 4),
                     "en_only": b, f"{tgt}_only": c, "mcnemar_p": mcnemar(b, c)}
    return out


def main(which, out_json):
    if which == "calibration":
        bench = {r["calib_id"]: r["benchmark_label"] for r in csv.DictReader(open(L / "calibration_human.csv"))}
        rows = [r for r in csv.DictReader(open(L / "calibration_models.csv")) if r["grader"].endswith("(kiro)")]
        key, tgt = "calib_id", "hi"
    else:
        rows = list(csv.DictReader(open(L / "spanish_grades.csv")))
        bench = {r["item_id"]: ("harmless" if r["item_id"].startswith("polyguard") else "harmful") for r in rows}
        key, tgt = "item_id", "es"
    lab = defaultdict(dict)
    for r in rows:
        lab[(r["grader"], r[key])][r["lang"]] = r["label"]
    res = {}
    for (g, i), v in sorted(lab.items()):
        if "en" in v and tgt in v and "DECLINED" not in v.values():
            res.setdefault(g, {}).setdefault(bench[i], []).append(v)
    out = {g: {b: summarise(p, tgt) for b, p in d.items()} for g, d in res.items()}
    json.dump(out, open(out_json, "w"), indent=1)
    for g, d in out.items():
        h = d["harmful"]
        print(g, "harmful n", h["n_pairs"], "explicit", h["explicit"], "any", h["any"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
