"""Stage 1: collect candidate items from the raw sources, one row per request, with provenance.

Every provenance field quotes or paraphrases the source's own documentation (see PROVENANCE). Nothing from the v1
manifest is used. Prompt text is never printed.

Outputs:
  data/v2/candidates.jsonl       one row per item: English, Hindi and, where available, Bengali text (ignored by git)
  data/v2/collect_summary.json   counts only (tracked)
  data/v2/pilot_collect.jsonl    a seeded 50-item pilot, stratified by source (ignored by git)

Usage: python scripts/v2/collect.py   (PolyGuard needs pyarrow; pass --python-with-pyarrow if the default python lacks it)
"""

import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data/external"
OUT = ROOT / "data/v2"
SEED = 20261111

# What each source's documentation says about who wrote the Hindi text and who assigned the label.
PROVENANCE = {
    "aya_redteaming": dict(tier="A", hi_authorship="written by paid native annotators",
                           en_authorship="human translation by the annotators (semantic if given, else literal)",
                           label_source="human", doc="data/external/aya_redteaming/README.md; arXiv 2406.18682"),
    "indicsafe": dict(tier="A", hi_authorship="human translation of English prompts",
                      en_authorship="original", label_source="dataset category", doc="IndicSafe paper, arXiv 2603.17915"),
    "surakshaeval": dict(tier="A", hi_authorship="human-written", en_authorship="provided English version",
                         label_source="none: every item is a sensitive question, harm not labelled per item", doc="SurakshaEval, AIES"),
    "indicguard_culture": dict(tier="B", hi_authorship="Hindi-origin, adapted with an LLM-based cultural adaptation method (CultureGuard)",
                               en_authorship="Google translation from Hindi", label_source="human (prompt_label_source field)",
                               doc="data/external/indicguard/README.md; arXiv 2508.01710"),
    "indicguard_generic": dict(tier="C", hi_authorship="Google translation from English", en_authorship="original",
                               label_source="human (prompt_label_source field)", doc="data/external/indicguard/README.md"),
    "indicguard_jailbreak": dict(tier="C", hi_authorship="Google translation from English", en_authorship="original",
                                 label_source="LLM jury (prompt_label_source field)", doc="data/external/indicguard/README.md"),
    "mhh": dict(tier="C", hi_authorship="machine translation, partly checked", en_authorship="original",
                label_source="dataset type column", doc="Multilingual Harmless-Harmful dataset card"),
    "polyguard": dict(tier="C", hi_authorship="machine translation verified by humans", en_authorship="original (WildGuardTest)",
                      label_source="human annotation on the English version, 2 or 3 of 3 annotators agreeing",
                      doc="data/external/polyguard/README.md; arXiv 2504.04377"),
}


def norm(text):
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text or "").casefold()).strip()


def item(source, source_id, en, hi, label, bn=None, category=None, extra=None):
    p = PROVENANCE[source]
    return dict(item_id=f"{source}__{source_id}", source=source, source_id=str(source_id), tier=p["tier"], en=en, hi=hi, bn=bn,
                benchmark_label=label, harm_category=category, hi_authorship=p["hi_authorship"], en_authorship=p["en_authorship"],
                label_source=p["label_source"], extra=extra or {})


def aya():
    rows = []
    for i, r in enumerate(map(json.loads, open(EXT / "aya_redteaming/aya_hin.jsonl", encoding="utf-8"))):
        en = r.get("semantic_translation") or r.get("literal_translation")
        rows.append(item("aya_redteaming", i, en, r["prompt"], "harmful", category=r["harm_category"],
                         extra={"global_or_local": r["global_or_local"], "has_semantic_translation": bool(r.get("semantic_translation"))}))
    return rows


def indicsafe():
    by_en = defaultdict(dict)
    with open(EXT / "indicsafe/data/Full_translated_prompts.csv", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if r["Language"] in ("Hindi", "Bengali"):
                g = by_en[r["Prompt"]]
                g["category"] = r["Category"]
                g["hi" if r["Language"] == "Hindi" else "bn"] = r["Translated_Prompt"]
    rows = []
    for i, (en, g) in enumerate(sorted(by_en.items())):
        if "hi" not in g:
            continue
        cat = g["category"]
        label = "harmless" if "Harmless" in cat else "ambiguous" if "Ambiguous" in cat else "harmful"
        rows.append(item("indicsafe", i, en, g["hi"], label, bn=g.get("bn"), category=cat))
    return rows


def surakshaeval():
    by_en = defaultdict(dict)
    for name, key in (("hindi", "hi"), ("bengali", "bn")):
        for r in map(json.loads, open(EXT / f"surakshaeval/data/{name}/test.jsonl", encoding="utf-8")):
            g = by_en[r["question_en"]]
            g[key] = r["question_indic"]
            g["category"] = r["type_of_harm"]
            g.setdefault("ids", []).append(f"{name}_{r['id']}")
    rows = []
    for en, g in sorted(by_en.items()):
        if "hi" in g:
            rows.append(item("surakshaeval", "+".join(g["ids"]), en, g["hi"], "unlabelled_sensitive", bn=g.get("bn"), category=g["category"]))
    return rows


def indicguard():
    text = defaultdict(dict)
    meta = {}
    for name, key in (("English", "en"), ("Hindi", "hi"), ("Bengali", "bn")):
        for sp in ("train", "valid", "test"):
            for r in json.load(open(EXT / f"indicguard/{name}/{sp}.json", encoding="utf-8")):
                i = str(r["id"])
                text[i].setdefault(key, r["prompt"])
                if key == "hi":
                    meta.setdefault(i, dict(tag=r["tag"], label=r["prompt_label"], label_source=r["prompt_label_source"],
                                            category=r.get("violated_categories"), split=sp))
    rows, dropped = [], Counter()
    for i, t in text.items():
        m = meta.get(i)
        if not m or "en" not in t or "hi" not in t:
            dropped["missing_language"] += 1
            continue
        if "REDACTED" in (t["en"], t["hi"]):
            dropped["redacted"] += 1
            continue
        if m["label"] not in ("safe", "unsafe"):
            dropped["no_prompt_label"] += 1
            continue
        tag = m["tag"].lower()
        source = "indicguard_culture" if "cultur" in tag else "indicguard_jailbreak" if "jail" in tag else "indicguard_generic"
        rows.append(item(source, i, t["en"], t["hi"], "harmful" if m["label"] == "unsafe" else "harmless", bn=t.get("bn"),
                         category=m["category"], extra={"source_split": m["split"], "label_source_field": m["label_source"]}))
    return rows, dropped


def mhh():
    by_i = defaultdict(dict)
    for name, key in (("english", "en"), ("hindi", "hi"), ("bengali", "bn")):
        counters = Counter()
        with open(EXT / f"multilingual_harmless_harmful/content/{name}/csv/{name}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                i = counters[r["type"]]
                counters[r["type"]] += 1
                g = by_i[(r["type"], i)]
                g[key] = r["text"]
                g["en_col"] = r["english"]
    rows = []
    for (typ, i), g in sorted(by_i.items()):
        if "hi" in g:
            rows.append(item("mhh", f"{typ}_{i:03d}", g.get("en") or g["en_col"], g["hi"], typ, bn=g.get("bn")))
    return rows


def polyguard():
    import pyarrow.parquet as pq
    d = pq.read_table(EXT / "polyguard/test-00000-of-00001.parquet").to_pylist()
    en = {r["id"]: r for r in d if r["language"] == "English"}
    rows, dropped = [], Counter()
    for r in d:
        if r["language"] != "Hindi":
            continue
        e = en.get(r["id"])
        if not e or r["prompt_harm_label"] not in ("harmful", "unharmful"):
            dropped["polyguard_no_label"] += 1
            continue
        rows.append(item("polyguard", r["id"], e["prompt"], r["prompt"], "harmful" if r["prompt_harm_label"] == "harmful" else "harmless",
                         category=r.get("subcategory"),
                         extra={"adversarial": r["adversarial"], "prompt_harm_agreement": r["prompt_harm_agreement"]}))
    return rows, dropped


def main():
    guard, d1 = indicguard()
    pg, d2 = polyguard()
    rows = aya() + indicsafe() + surakshaeval() + guard + mhh() + pg
    dropped = d1 + d2
    checks = Counter()
    for r in rows:
        if not norm(r["en"]) or not norm(r["hi"]):
            checks["empty_text"] += 1
        if not re.search(r"[ऀ-ॿ]", r["hi"]):
            checks["hindi_without_devanagari"] += 1
        if r["bn"] and not re.search(r"[ঀ-৿]", r["bn"]):
            checks["bengali_without_bengali_script"] += 1
        ratio = len(r["hi"]) / max(1, len(r["en"]))
        if ratio < 0.4 or ratio > 3.0:
            checks["length_ratio_outside_0.4_to_3"] += 1
    ids = Counter(r["item_id"] for r in rows)
    checks["duplicate_item_ids"] = sum(v - 1 for v in ids.values() if v > 1)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "candidates.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rng = random.Random(SEED)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    pilot = []
    per = max(1, 50 // len(by_source))
    for s in sorted(by_source):
        pilot += rng.sample(by_source[s], min(per, len(by_source[s])))
    rest = [r for r in rows if r not in pilot]
    pilot += rng.sample(rest, 50 - len(pilot))
    with open(OUT / "pilot_collect.jsonl", "w", encoding="utf-8") as f:
        for r in pilot:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    table = Counter((r["tier"], r["source"], r["benchmark_label"], bool(r["bn"])) for r in rows)
    summary = dict(total=len(rows), dropped=dict(dropped), checks=dict(checks),
                   by_tier_source_label_has_bengali=[dict(tier=k[0], source=k[1], label=k[2], has_bengali=k[3], n=v) for k, v in sorted(table.items())],
                   provenance=PROVENANCE)
    (OUT / "collect_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("total", "dropped", "checks")}, indent=1))
    for x in summary["by_tier_source_label_has_bengali"]:
        print(f"  {x['tier']}  {x['source']:22s} {x['label']:22s} bn={x['has_bengali']!s:5s} {x['n']}")


if __name__ == "__main__":
    main()
