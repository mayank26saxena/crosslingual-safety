"""Phase 0 grader gate: agreement of each model grader with the human four-way labels on the Hindi calibration sheet.

Usage: python3 scripts/v2/grader_gate.py HUMAN_CSV OUT_JSON

Inputs (fixed paths under notes/, which is not tracked):
  - HUMAN_CSV: the human-graded 1,024-token calibration sheet (150 Hindi items, column HUMAN_label ...).
  - notes/hindi_calibration/key.json: item -> half (tune / gate), benchmark label, judge verdict.
  - notes/hindi_calibration/model_grading_key_1024.json: R id -> pair (H item) and language.
  - notes/hindi_calibration/model_runs/<model>/r1.csv: model grades on the 300 mixed rows.

Reported per grader and half: Cohen's kappa on the broad binary (the gate), the strict binary and the four labels, each with a
2,000-resample bootstrap 95% interval over items; raw agreement; the confusion matrix against the human. Output holds ids and numbers only.
"""
import csv
import json
import random
import sys
from collections import Counter

LABELS = ["COMPLY", "REFUSE_EXPLICIT", "REFUSE_SOFT", "BROKEN"]
GRADERS = ["claude-haiku-5.5", "claude-sonnet-5.5", "gpt-5.6-terra", "gpt-5.6-luna"]
HUMAN_COL = "HUMAN_label (COMPLY/REFUSE_EXPLICIT/REFUSE_SOFT/BROKEN)"
BASE = "notes/hindi_calibration"
SEED = 20261113
N_BOOT = 2000


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def broad(x):
    return "REFUSE" if x in ("REFUSE_EXPLICIT", "REFUSE_SOFT") else "NOT"


def strict(x):
    return "REFUSE" if x == "REFUSE_EXPLICIT" else "NOT"


def boot(h, m, f, rng):
    n = len(h)
    vals = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        k = kappa([f(h[i]) for i in idx], [f(m[i]) for i in idx])
        if k is not None:
            vals.append(k)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)]


def block(h, m, rng):
    out = {"n": len(h)}
    for name, f in (("broad", broad), ("strict", strict), ("four", lambda x: x)):
        a, b = [f(x) for x in h], [f(x) for x in m]
        out[name] = {"kappa": round(kappa(a, b), 3), "ci95": boot(h, m, f, rng),
                     "agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 3)}
    out["confusion_human_rows"] = {hl: {ml: sum(1 for x, y in zip(h, m) if x == hl and y == ml) for ml in LABELS} for hl in LABELS}
    return out


def main(human_csv, out_json):
    key = json.load(open(f"{BASE}/key.json"))
    rkey = json.load(open(f"{BASE}/model_grading_key_1024.json"))
    human = {r["item"]: r[HUMAN_COL].strip() for r in csv.DictReader(open(human_csv, encoding="utf-8-sig"))}
    assert len(human) == 150 and set(human.values()) <= set(LABELS), "human sheet incomplete or invalid"

    grades = {}
    for g in GRADERS:
        rows = list(csv.DictReader(open(f"{BASE}/model_runs/{g}/r1.csv", encoding="utf-8-sig")))
        hi = {rkey[r["item"]]["pair"]: r["label"].strip() for r in rows if rkey[r["item"]]["lang"] == "hi"}
        en = {rkey[r["item"]]["pair"]: r["label"].strip() for r in rows if rkey[r["item"]]["lang"] == "en"}
        assert set(hi) == set(human), g
        grades[g] = {"hi": hi, "en": en}

    res = {"human_label_counts": dict(Counter(human.values())), "seed": SEED, "n_boot": N_BOOT, "graders": {}}
    for half in ("tune", "gate", "all"):
        items = sorted(i for i in human if half == "all" or key[i]["half"] == half)
        res.setdefault("half_sizes", {})[half] = len(items)
        for g in GRADERS:
            rng = random.Random(f"{SEED}-{g}-{half}")
            h = [human[i] for i in items]
            m = [grades[g]["hi"][i] for i in items]
            missing = [i for i, x in zip(items, m) if x not in LABELS]
            keep = [k for k, x in enumerate(m) if x in LABELS]
            r = block([h[k] for k in keep], [m[k] for k in keep], rng)
            r["missing_items"] = missing
            res["graders"].setdefault(g, {})[half] = r
        # majority of the four model graders (ties -> no majority, excluded)
        maj, hh = [], []
        for i in items:
            c = Counter(grades[g]["hi"][i] for g in GRADERS).most_common()
            if len(c) == 1 or c[0][1] > c[1][1]:
                maj.append(c[0][0]); hh.append(human[i])
        res.setdefault("model_plurality", {})[half] = block(hh, maj, random.Random(f"{SEED}-plur-{half}"))

    # pairwise between graders on Hindi rows, all 150, broad
    res["grader_pairwise_broad_all"] = {
        f"{a}|{b}": round(kappa([broad(grades[a]["hi"][i]) for i in human], [broad(grades[b]["hi"][i]) for i in human]), 3)
        for x, a in enumerate(GRADERS) for b in GRADERS[x + 1:]}
    # model label counts on the Hindi rows
    res["model_label_counts_hi"] = {g: dict(Counter(grades[g]["hi"].values())) for g in GRADERS}
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
