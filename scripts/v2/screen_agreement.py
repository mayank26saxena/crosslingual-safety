"""Agreement of each screening candidate with the human audit (request type and fidelity).

Usage: python3 scripts/v2/screen_agreement.py OUT_JSON
Reads notes/audit/human/audit_human.csv, notes/audit/audit_key.json and notes/audit/screen_runs/<model>/part_0{1,2}.csv.
Ids and numbers only.
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "notes/audit"
RT_V = ["request_harmful", "request_borderline_safe", "request_benign", "not_a_request"]
FD_V = ["equivalent", "minor_drift", "wrong"]


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def stats(h, m, rng):
    n = len(h)
    out = {"n": n, "agreement": round(sum(x == y for x, y in zip(h, m)) / n, 3), "kappa": round(kappa(h, m), 3)}
    ks = []
    for _ in range(2000):
        ix = [rng.randrange(n) for _ in range(n)]
        hh, mm = [h[i] for i in ix], [m[i] for i in ix]
        if len(set(hh) | set(mm)) > 1:
            ks.append(kappa(hh, mm))
    ks.sort()
    out["kappa_ci95"] = [round(ks[int(0.025 * len(ks))], 3), round(ks[int(0.975 * len(ks)) - 1], 3)] if len(ks) > 100 else None
    if len(set(h) | set(m)) == 1:
        out["kappa"] = None  # one category only: kappa undefined, agreement is reported
    return out


def main(out_json):
    key = json.load(open(A / "audit_key.json"))
    hum = {}
    for r in csv.DictReader(open(A / "human/audit_human.csv", encoding="utf-8-sig")):
        rt = next(v for k, v in r.items() if k.startswith("HUMAN_request_type"))
        fd = next(v for k, v in r.items() if k.startswith("HUMAN_fidelity"))
        hum[r["item"]] = (rt, fd)
    res = {"human_counts": {"request_type": dict(Counter(v[0] for v in hum.values())), "fidelity": dict(Counter(v[1] for v in hum.values()))},
           "models": {}}
    for model in ("opus", "fable"):
        lab = {}
        for p in (1, 2):
            f = A / f"screen_runs/{model}/part_{p:02d}.csv"
            if f.exists():
                for r in csv.DictReader(open(f, encoding="utf-8-sig")):
                    lab[r["item"].strip()] = (r["request_type"].strip(), r["fidelity"].strip())
        valid = [i for i in hum if i in lab and lab[i][0] in RT_V and lab[i][1] in FD_V]
        rng = random.Random(20261113)
        m = {"n_labelled": len(valid), "n_missing_or_declined": len(hum) - len(valid), "model_counts": dict(Counter(lab[i][0] for i in valid))}
        groups = {"all": valid}
        for t in ("A", "B", "C"):
            groups[f"tier_{t}"] = [i for i in valid if key[i]["tier"] == t]
        for g, ids in groups.items():
            if not ids:
                continue
            h_rt, m_rt = [hum[i][0] for i in ids], [lab[i][0] for i in ids]
            h_fd, m_fd = [hum[i][1] for i in ids], [lab[i][1] for i in ids]
            m[g] = {"request_type": stats(h_rt, m_rt, rng),
                    "harmful_vs_rest": stats([x == "request_harmful" for x in h_rt], [x == "request_harmful" for x in m_rt], rng),
                    "fidelity": stats(h_fd, m_fd, rng),
                    "wrong_vs_rest": stats([x == "wrong" for x in h_fd], [x == "wrong" for x in m_fd], rng)}
        m["confusion_request_type"] = {hv: {mv: sum(1 for i in valid if hum[i][0] == hv and lab[i][0] == mv) for mv in RT_V} for hv in RT_V}
        res["models"][model] = m
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
