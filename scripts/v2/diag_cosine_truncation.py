"""Exploratory diagnostics added after the registered studies (2026-11-23). Aggregates only; no reply text is read into output.

Usage: uv run --offline --no-project --with numpy==2.5.4 python -I scripts/v2/diag_cosine_truncation.py
Writes results/v2/diagnostics/cosine_truncation.json.

1. Cosine similarity between the explicit-versus-soft direction and the refusal direction, per model, at the same layer and at the layers the
   paper uses (Qwen: e_dir[24] against r_dir[14] and r_dir[24]; Llama: e_dir[18] against r_dir[10] and r_dir[18]), with the 95th percentile of
   |cos| for random unit vectors (seed 0) as a scale. Directions at different layers are not directly comparable, so the same-layer value is the
   informative one.
2. Reply-length cap: share of unsteered Hindi harmful held-out replies that reached the new-token limit (1,024), overall and by judge label, and
   for the requests refused explicitly in English but not in Hindi. Qwen: the Phase 3 test run; Llama: the Phase 7 regenerated run.
"""
import collections
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "results/v2"


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def directions():
    d, e = np.load(V / "direction/direction.npz"), np.load(V / "es_dir/es_dir.npz")
    s = np.load(V / "phase6/stageB/stageB_dirs.npz")
    rng = np.random.default_rng(0)
    R = rng.standard_normal((20000, 4096))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    out = {}
    for name, r, ed, lr, le in (("qwen", d["r_dir"], e["e_dir"], 14, 24), ("llama", s["r_dir"], s["e_dir"], 10, 18)):
        v = ed[le] / np.linalg.norm(ed[le])
        out[name] = dict(
            refusal_layer=lr, explicit_soft_layer=le,
            cos_refusal_gate_layer=round(cos(ed[le], r[lr]), 3),
            cos_refusal_same_layer=round(cos(ed[le], r[le]), 3),
            random_abs_cos_p95=round(float(np.percentile(np.abs(R @ v), 95)), 3),
        )
    return out


def cap_table(labels, replies, cond_ok, lost):
    c, cl = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
    for r in replies:
        if r["set"] == "test_harmful" and cond_ok(r) and r["pair"] in labels:
            k = labels[r["pair"]]
            c[k][0] += 1
            c[k][1] += int(r["hit_limit"])
            if r["pair"] in lost:
                cl[k][0] += 1
                cl[k][1] += int(r["hit_limit"])
    f = lambda d: {k: dict(n=v[0], hit_cap=v[1]) for k, v in sorted(d.items())}
    return dict(all=f(c), n=sum(v[0] for v in c.values()), hit_cap=sum(v[1] for v in c.values()), lost_refusals=f(cl), n_lost=len({*lost} & set(labels)))


def qwen_cap():
    lab, en = {}, {}
    for r in csv.DictReader(open(V / "labels/phase3_test_grades.csv")):
        if r["set"] == "test_harmful" and r["condition"] == "hi_unsteered":
            lab[r["item_id"]] = r["label"]
        if r["set"] == "test_harmful" and r["condition"] == "en_unsteered":
            en[r["item_id"]] = r["label"]
    lost = {i for i in lab if en[i] == "REFUSE_EXPLICIT" and lab[i] != "REFUSE_EXPLICIT"}
    rep = (json.loads(x) for x in open(V / "phase3/test/test_replies.jsonl", encoding="utf-8"))
    return cap_table(lab, rep, lambda r: r["lang"] == "hi" and r["condition"].endswith("unsteered"), lost)


def llama_cap():
    rows = list(csv.DictReader(open(V / "labels/phase7_2_llama_grades.csv")))
    key = {f'{r["set"]}|{r["item_id"]}|{r["lang"]}|{r["condition"]}': r for r in rows}
    p6 = {f'{r["set"]}|{r["item_id"]}|{r["lang"]}|{r["condition"]}': r["label"] for r in csv.DictReader(open(V / "labels/phase6B_llama_grades.csv"))}

    def resolve(r):
        lbl, src = r["label"], r["source"]
        while not lbl and src.startswith(("same_as:", "phase6:")):
            if src.startswith("phase6:"):
                return p6[src.split(":", 1)[1]]
            ref = key[src.split(":", 1)[1]]
            lbl, src = ref["label"], ref["source"]
        return lbl

    hi = {r["item_id"]: resolve(r) for r in rows if r["set"] == "test_harmful" and r["condition"] == "hiharm_unsteered"}
    en = {r["item_id"]: resolve(r) for r in rows if r["set"] == "test_harmful" and r["condition"] == "enharm_unsteered"}
    lost = {i for i in hi if en[i] == "REFUSE_EXPLICIT" and hi[i] != "REFUSE_EXPLICIT"}
    rep = (json.loads(x) for x in open(V / "phase7/stage2/stage2_replies.jsonl", encoding="utf-8"))
    return cap_table(hi, rep, lambda r: r["condition"] == "hiharm_unsteered", lost)


def main():
    out = dict(cosine=directions(), hit_cap=dict(qwen=qwen_cap(), llama=llama_cap()),
               note="exploratory, added after the registered studies; aggregates only")
    (V / "diagnostics").mkdir(exist_ok=True)
    json.dump(out, open(V / "diagnostics/cosine_truncation.json", "w"), indent=1)
    for m, v in out["cosine"].items():
        print(m, v)
    for m, v in out["hit_cap"].items():
        print(m, "hit cap", v["hit_cap"], "of", v["n"], "| soft:", v["all"].get("REFUSE_SOFT"), "| lost soft:", v["lost_refusals"].get("REFUSE_SOFT"))


if __name__ == "__main__":
    main()
