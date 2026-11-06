"""Build the unified prompt manifest, group translations by source prompt, and assign group-level splits.

Inputs are dataset snapshots under data/external. Outputs:
  data/manifest/manifest.jsonl  every prompt (not tracked, contains benchmark text)
  data/manifest/summary.json    counts only (tracked)

Prompt text is never printed. Usage: python scripts/build_manifest.py
"""

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data/external"
OUT = ROOT / "data/manifest"
LANGS = {"english": "en", "hindi": "hi", "bengali": "bn"}
NEAR_DUP_COSINE = 0.92
SPLITS = (("discovery", 0.5), ("development", 0.2), ("heldout", 0.3))
SEED = "20261106"


def norm(text):
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text or "").casefold()).strip()


def multilingual():
    rows = []
    for name, lang in LANGS.items():
        counters = Counter()
        with open(EXT / f"multilingual_harmless_harmful/content/{name}/csv/{name}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                i = counters[r["type"]]
                counters[r["type"]] += 1
                rows.append(dict(source_dataset="multilingual_harmless_harmful", source_id=f"{r['type']}_{i:03d}", pair=f"mhh_{i:03d}", language=lang,
                                 script_register="native_script", prompt_text=r["text"], english=r["english"], harm_category=None, label=r["type"],
                                 expected_behavior="refuse" if r["type"] == "harmful" else "answer", authorship_type="original_en" if lang == "en" else "machine_translated_partly_checked",
                                 subset=r["type"], source_split=None))
    return rows


def indicsafe():
    rows, seen = [], set()
    with open(EXT / "indicsafe/data/Full_translated_prompts.csv", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            harmless, ambiguous = "Harmless" in r["Category"], "Ambiguous" in r["Category"]
            expected = "answer" if harmless else "judgement" if ambiguous else "refuse_or_correct"
            common = dict(source_dataset="indicsafe", pair=None, script_register="native_script", english=r["Prompt"], harm_category=r["Category"],
                          label="harmless" if harmless else "ambiguous" if ambiguous else "harmful", expected_behavior=expected, subset=r["Category"], source_split=None)
            if r["Language"] in ("Hindi", "Bengali"):
                lang = LANGS[r["Language"].lower()]
                rows.append(dict(common, source_id=f"row_{i}", language=lang, prompt_text=r["Translated_Prompt"], authorship_type="translated"))
            key = norm(r["Prompt"])
            if key not in seen:
                seen.add(key)
                rows.append(dict(common, source_id=f"en_{len(seen)}", language="en", prompt_text=r["Prompt"], authorship_type="source_en"))
    return rows


def indicguard():
    rows, english = [], {}
    tag_of = lambda t: "culture_adaptive" if "ultur" in t else "jailbreaking" if "ail" in t else "generic"
    for sp in ("train", "valid", "test"):
        for r in json.load(open(EXT / f"indicguard/English/{sp}.json", encoding="utf-8")):
            english.setdefault(str(r["id"]), r["prompt"])
    dropped = Counter()
    for name, lang in LANGS.items():
        seen = set()
        for sp in ("train", "valid", "test"):
            for r in json.load(open(EXT / f"indicguard/{name.capitalize()}/{sp}.json", encoding="utf-8")):
                i = str(r["id"])
                if (i, lang) in seen:
                    continue
                seen.add((i, lang))
                if r["prompt"] == "REDACTED" or english.get(i) == "REDACTED":
                    dropped["redacted"] += 1
                    continue
                if r["prompt_label"] not in ("safe", "unsafe"):
                    dropped["no_prompt_label"] += 1
                    continue
                tag = tag_of(r["tag"])
                source = "hi" if tag == "culture_adaptive" else "en"
                rows.append(dict(source_dataset="indicguard", source_id=i, pair=None, language=lang, script_register="native_script", prompt_text=r["prompt"], english=english[i],
                                 harm_category=r.get("violated_categories"), label="harmful" if r["prompt_label"] == "unsafe" else "harmless",
                                 expected_behavior="refuse" if r["prompt_label"] == "unsafe" else "answer",
                                 authorship_type=f"source_{lang}" if lang == source else "google_translated", subset=tag, source_split=sp))
    return rows, dropped


def surakshaeval():
    rows = []
    for name, lang in (("hindi", "hi"), ("bengali", "bn")):
        for r in map(json.loads, open(EXT / f"surakshaeval/data/{name}/test.jsonl", encoding="utf-8")):
            common = dict(source_dataset="surakshaeval", pair=None, script_register="native_script", english=r["question_en"], harm_category=r["type_of_harm"],
                          label="unverified_sensitive", expected_behavior="unverified", subset=r["risk_area"], source_split="test", authorship_type="human_written")
            rows.append(dict(common, source_id=f"{name}_{r['id']}", language=lang, prompt_text=r["question_indic"]))
            rows.append(dict(common, source_id=f"{name}_{r['id']}_en", language="en", prompt_text=r["question_en"]))
    return rows


def banglashield():
    """Bengali forms only. MultiJail rows are dropped here, because that source contains the sealed split."""
    import pyarrow.parquet as pq
    rows, dropped = [], Counter()
    label = {"Safe": ("harmless", "answer"), "Harmful": ("harmful", "refuse"), "Severe": ("harmful", "refuse"), "Offensive": ("offensive", "judgement")}
    for form, register in (("native", "native_script"), ("romanized", "romanized"), ("code_mixed", "code_mixed")):
        for sp in ("train", "validation", "test"):
            for r in pq.read_table(EXT / f"banglashield/data/{form}-{sp}.parquet").to_pylist():
                if r["source"] == "multijail":
                    dropped["multijail_source"] += 1
                    continue
                lab, expected = label[r["label_name"]]
                rows.append(dict(source_dataset="banglashield", source_id=f"{form}_{r['uid']}", pair=f"bs_{r['group_id']}", language="bn", script_register=register, prompt_text=r["text"],
                                 english=None, harm_category=r["label_name"], label=lab, expected_behavior=expected, authorship_type=f"compiled_from_{r['source']}", subset=r["source"], source_split=sp))
    return rows, dropped


def sealed_texts():
    """Normalized text of the sealed split, used only to exclude matches. Nothing is printed."""
    confirm = set(json.loads((ROOT / "data/splits/splits.json").read_text())["multijail"]["confirm"])
    texts = set()
    for r in csv.DictReader(open(EXT / "multijail/MultiJail.csv", encoding="utf-8-sig")):
        if f"multijail_{int(r['id']):03d}" in confirm:
            texts.update(norm(r[k]) for k in ("en", "bn") if norm(r[k]))
    if not texts:
        raise SystemExit("no sealed text available to check against")
    return texts


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def main():
    guard, dropped = indicguard()
    shield, shield_dropped = banglashield()
    dropped.update(shield_dropped)
    rows = multilingual() + indicsafe() + guard + surakshaeval() + shield
    groups = Groups()
    for i, r in enumerate(rows):
        r["example_id"] = f"{r['source_dataset']}__{r['source_id']}__{r['language']}"
        node = f"row{i}"
        if norm(r["english"]):
            groups.union(node, f"en:{norm(r['english'])}")
        groups.union(node, f"{r['language']}:{norm(r['prompt_text'])}")
        groups.union(node, f"id:{r['source_dataset']}:{r['source_id']}")
        if r["pair"]:
            groups.union(node, f"pair:{r['pair']}")  # harmful and harmless twins stay together
    # Near duplicates on English source text. Jailbreak wrappers share templates, so they are grouped by exact text only.
    english = sorted({norm(r["english"]) for r in rows if r["subset"] != "jailbreaking" and norm(r["english"])})
    matrix = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit_transform(english)
    near = 0
    for start in range(0, matrix.shape[0], 2000):
        sims = (matrix[start:start + 2000] @ matrix.T).tocoo()
        for a, b, v in zip(sims.row + start, sims.col, sims.data):
            if a < b and v >= NEAR_DUP_COSINE:
                groups.union(f"en:{english[a]}", f"en:{english[b]}")
                near += 1
    for i, r in enumerate(rows):
        r["source_group_id"] = hashlib.sha256(groups.find(f"row{i}").encode()).hexdigest()[:16]
    sealed = sealed_texts()
    hit = {r["source_group_id"] for r in rows if norm(r["prompt_text"]) in sealed or norm(r["english"]) in sealed}
    sealed_rows = Counter(r["source_dataset"] for r in rows if r["source_group_id"] in hit)
    rows = [r for r in rows if r["source_group_id"] not in hit]
    members = defaultdict(list)
    for r in rows:
        members[r["source_group_id"]].append(r)

    def split_of(gid, items):
        if any(x["source_dataset"] in ("indicsafe", "surakshaeval", "banglashield") or x["source_split"] == "test" for x in items):
            return "heldout"
        u = int(hashlib.sha256(f"{SEED}:{gid}".encode()).hexdigest(), 16) / 16 ** 64
        total = 0.0
        for name, share in SPLITS:
            total += share
            if u < total:
                return name
        return SPLITS[-1][0]

    exact = Counter()
    unique = []
    seen = set()
    for r in rows:
        key = (r["language"], norm(r["prompt_text"]), r["label"])
        if key in seen:
            exact[r["source_dataset"]] += 1
            continue
        seen.add(key)
        unique.append(r)
    for gid, items in members.items():
        s = split_of(gid, items)
        for x in items:
            x["split"] = s
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["example_id", "source_dataset", "source_id", "source_group_id", "language", "script_register", "prompt_text", "english", "label", "harm_category",
              "expected_behavior", "authorship_type", "subset", "source_split", "split"]
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps({k: r.get(k) for k in fields}, ensure_ascii=False) + "\n")
    sizes = np.array([len({x["english"] for x in v}) for v in members.values()])
    group_splits = defaultdict(set)
    for r in unique:
        group_splits[r["source_group_id"]].add(r["split"])
    cross = Counter()
    for r in unique:
        cross[(r["source_dataset"], r["language"], r["label"], r["split"])] += 1
    summary = dict(rows=len(unique), groups=len(members), exact_duplicates_removed=dict(exact), near_duplicate_links=near, dropped=dict(dropped), removed_matching_sealed_split=dict(sealed_rows),
                   largest_group_english_prompts=int(sizes.max()), groups_with_over_10_english_prompts=int((sizes > 10).sum()),
                   groups_in_more_than_one_split=sum(len(v) > 1 for v in group_splits.values()),
                   authorship=dict(Counter(r["authorship_type"] for r in unique)),
                   counts=[dict(dataset=a, language=b, label=c, split=d, n=n) for (a, b, c, d), n in sorted(cross.items())])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}))
    table = defaultdict(lambda: Counter())
    for x in summary["counts"]:
        table[(x["language"], x["label"])][x["split"]] += x["n"]
    print(Counter((r["source_dataset"], r["language"], r["script_register"]) for r in unique).most_common(20))
    for k, v in sorted(table.items()):
        print(k, dict(v))


if __name__ == "__main__":
    main()
