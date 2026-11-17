"""Validate the blog's redacted example transcripts and write docs/blog/data/examples.js.

Usage: python3 -I scripts/v2/export_examples.py
Reads docs/blog/data/examples.json (redacted, translated excerpts; requests summarised at category level, never quoted). Fails unless:
  - no Devanagari or Bengali script appears anywhere (Hindi replies are shown in translation only);
  - every request summary is at most 16 words and every excerpt at most 80 words, with no URLs;
  - every judge label matches the registered, id-only label tables (Qwen: Phase 3 test; Llama: Phase 7 stage 2), so the examples
    cannot drift from the graded data. The excerpts themselves are hand-redacted and reviewed; this script checks form and labels only.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs/blog/data/examples.json"
OUT = ROOT / "docs/blog/data/examples.js"


def labels():
    """Judge labels per (model, item, lang, condition), plus the set of repaired replies identical to the unsteered ones."""
    lab, same = {}, set()
    for r in csv.DictReader(open(ROOT / "results/v2/labels/phase3_test_grades.csv")):
        if r["set"] != "test_harmful":
            continue
        cond = "unsteered" if r["condition"].endswith("unsteered") else "repaired"
        lab[("qwen", r["item_id"], r["lang"], cond)] = r["label"]
        if cond == "repaired" and r["inherited"] == "yes":  # gate did not fire: reply identical to the unsteered one
            same.add(("qwen", r["item_id"], r["lang"]))
    rows = list(csv.DictReader(open(ROOT / "results/v2/labels/phase7_2_llama_grades.csv")))
    key = {f'{r["set"]}|{r["item_id"]}|{r["lang"]}|{r["condition"]}': r for r in rows}
    p6 = {f'{r["set"]}|{r["item_id"]}|{r["lang"]}|{r["condition"]}': r["label"]
          for r in csv.DictReader(open(ROOT / "results/v2/labels/phase6B_llama_grades.csv"))}
    for r in rows:
        if r["set"] != "test_harmful":
            continue
        lbl, src = r["label"], r["source"]
        if r["condition"].endswith("R3") and src.startswith("same_as:"):
            same.add(("llama", r["item_id"], r["lang"]))
        while not lbl and src.startswith(("same_as:", "phase6:")):
            if src.startswith("phase6:"):  # unchanged reply carried over from Phase 6; its label lives in the Phase 6 table
                lbl, src = p6[src.split(":", 1)[1]], ""
                break
            ref = key[src.split(":", 1)[1]]
            lbl, src = ref["label"], ref["source"]
        lab[("llama", r["item_id"], r["lang"], "unsteered" if r["condition"].endswith("unsteered") else "repaired")] = lbl
    return lab, same


def human_labels():
    """Adjudicated human labels for Llama (Phase 6 unsteered), keyed like the judge labels."""
    h = {}
    for r in csv.DictReader(open(ROOT / "results/v2/labels/llama_human_adjudicated.csv")):
        st, iid, lang, cond = r["uid"].split("|")
        if st == "test_harmful" and cond.endswith("unsteered"):
            h[("llama", iid, lang, "unsteered")] = r["reviewed_label"]
    return h


def pattern_counts(lab):
    """How typical are the examples? Counts over all 404 held-out harmful items, unsteered."""
    items = sorted({k[1] for k in lab if k[0] == "qwen"})
    g = lambda m, i, l: lab.get((m, i, l, "unsteered"))
    q = [i for i in items if g("qwen", i, "en") == "REFUSE_EXPLICIT" and g("qwen", i, "hi") == "REFUSE_SOFT"]
    l = [i for i in items if g("llama", i, "en") == "REFUSE_EXPLICIT" and g("llama", i, "hi") == "COMPLY"]
    return dict(n=len(items), qwen_explicit_to_soft=len(q), llama_explicit_to_comply=len(l), both=len(set(q) & set(l)))


def main():
    raw = SRC.read_text(encoding="utf-8")
    bad = []
    if re.search(r"[\u0900-\u09FF]", raw):
        bad.append("non-Latin (Devanagari/Bengali) script present")
    if re.search(r"https?://|www\.", raw):
        bad.append("URL present")
    d = json.loads(raw)
    lab, same = labels()
    hum = human_labels()
    for it in d["items"]:
        m = re.search(r"[a-z_]+__\d+", it["id"])
        if not m:
            bad.append(f"{it['id']}: id not recognised")
            continue
        iid = m.group(0)
        if len(it["request_summary"].split()) > 16:
            bad.append(f"{iid}: request summary too long")
        for r in it["replies"]:
            k = (r["model"], iid, r["lang"], r["condition"])
            if len(r["excerpt"].split()) > 80:
                bad.append(f"{k}: excerpt too long")
            if lab.get(k) != r["judge"]:
                bad.append(f"{k}: judge label {r['judge']} != registered {lab.get(k)}")
            if r["condition"] == "repaired":
                says_same = r["excerpt"].startswith("[identical")
                if says_same != ((r["model"], iid, r["lang"]) in same):
                    bad.append(f"{k}: 'identical' note disagrees with the label tables")
            if r.get("human") and hum.get(k) != r["human"]:
                bad.append(f"{k}: human label {r['human']} != adjudicated {hum.get(k)}")
            if not r.get("human") and hum.get(k):
                bad.append(f"{k}: adjudicated human label {hum[k]} exists but is not shown")
        print(f"OK? {iid}: {len(it['replies'])} replies")
    if bad:
        raise SystemExit("examples failed validation:\n  " + "\n  ".join(bad))
    d["stats"] = pattern_counts(lab)
    print("pattern counts:", d["stats"])
    OUT.write_text("window.EXAMPLES = " + json.dumps(d, ensure_ascii=False) + ";\n", encoding="utf-8")
    print("all checks passed; wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
