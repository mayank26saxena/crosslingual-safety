"""Stage 2: exclusions, text faults and grouping.

Reads data/v2/candidates.jsonl and writes:
  data/v2/after_exclusions.jsonl   kept items, each with a group_id (ignored by git)
  data/v2/removed_stage2.jsonl     removed items with the reason (ignored by git)
  data/v2/pilot_exclude.jsonl      50 kept and up to 50 removed items, seeded, for reading (ignored by git)
  data/v2/exclude_summary.json     counts only (tracked)

Rules, in order (an item takes the first reason that applies):
  1. sealed       the item's group contains text matching the sealed MultiJail confirm split (English or Bengali)
  2. used_in_v1   the item's group contains text used in any v1 rendered item (Steps 1 to 8, including the calibration sheet)
  3. empty        English or Hindi text is empty after normalisation
  4. no_devanagari  the Hindi text has no Devanagari character
  5. duplicate    an exact duplicate (same normalised English and Hindi) of an item kept earlier; tier A is kept before B before C
Bengali text without Bengali script is set to null, and the item is kept.

Groups join items that share normalised English, Hindi or Bengali text, or whose English is a near duplicate (TF-IDF cosine at least 0.92,
excluding jailbreak wrappers, which share templates). Splits are later made by group, so near duplicates never cross splits.

Prompt text is never printed. Usage: python scripts/v2/exclude.py
"""

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import math

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data/v2"
SEED = 20261111
NEAR_DUP = 0.92
TIER_ORDER = {"A": 0, "B": 1, "C": 2}
V1_FILES = ["baseline_items.jsonl", "gated_items.jsonl", "gated2_items.jsonl", "multijail_bn.jsonl", "multijail_en.jsonl",
            "multijail_gap.jsonl", "multijail_sw.jsonl", "xstest_safe_bn.jsonl", "xstest_safe_en.jsonl"]


def norm(text):
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text or "").casefold()).strip()


def near_duplicate_pairs(texts, threshold):
    """Pairs of indices with TF-IDF cosine >= threshold, in plain Python.

    Matches scikit-learn's TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True) defaults: tokens are words of two or more
    characters, lowercased; idf = ln((1 + n) / (1 + df)) + 1; tf = 1 + ln(count); rows are L2-normalised. Candidate pairs are
    found through terms present in at most 2% of documents; any pair at or above the threshold shares such a term in practice,
    and every candidate's cosine is computed exactly over all terms.
    """
    tok = re.compile(r"(?u)\b\w\w+\b")
    docs = []
    for t in texts:
        words = tok.findall(t.lower())
        terms = Counter(words + [a + " " + b for a, b in zip(words, words[1:])])
        docs.append(terms)
    n = len(docs)
    df = Counter(term for d in docs for term in d)
    idf = {term: math.log((1 + n) / (1 + c)) + 1 for term, c in df.items()}
    vecs = []
    for d in docs:
        v = {term: (1 + math.log(c)) * idf[term] for term, c in d.items()}
        s = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({term: x / s for term, x in v.items()})
    rare_cap = max(2, int(0.02 * n))
    index = defaultdict(list)
    for i, d in enumerate(docs):
        for term in d:
            if df[term] <= rare_cap:
                index[term].append(i)
    pairs = set()
    for i, v in enumerate(vecs):
        cands = {j for term in docs[i] if df[term] <= rare_cap for j in index[term] if j > i}
        for j in cands:
            w = vecs[j]
            small, big = (v, w) if len(v) < len(w) else (w, v)
            if sum(x * big.get(term, 0.0) for term, x in small.items()) >= threshold:
                pairs.add((i, j))
    return pairs


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


def sealed_texts():
    confirm = set(json.loads((ROOT / "data/splits/splits.json").read_text())["multijail"]["confirm"])
    texts = set()
    for r in csv.DictReader(open(ROOT / "data/external/multijail/MultiJail.csv", encoding="utf-8-sig")):
        if f"multijail_{int(r['id']):03d}" in confirm:
            texts.update(norm(r[k]) for k in ("en", "bn") if norm(r[k]))
    if len(texts) < 100:
        raise SystemExit("sealed text set is unexpectedly small; refusing to continue")
    return texts


def v1_texts():
    texts, files = set(), Counter()
    for name in V1_FILES:
        path = ROOT / "data/rendered" / name
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            files[name] += 1
            for m in d.get("messages", []):
                if m.get("role") == "user":
                    texts.add(norm(m["content"]))
            for k in ("hidden_fact", "prompt", "text", "en", "bn"):
                if isinstance(d.get(k), str):
                    texts.add(norm(d[k]))
    texts.discard("")
    return texts, files


def main():
    rows = [json.loads(l) for l in open(V2 / "candidates.jsonl", encoding="utf-8")]
    bn_nulled = 0
    for r in rows:
        if r["bn"] and not re.search(r"[ঀ-৿]", r["bn"]):
            r["bn"] = None
            r["extra"]["bengali_dropped_wrong_script"] = True
            bn_nulled += 1

    groups = Groups()
    for i, r in enumerate(rows):
        node = f"row{i}"
        for lang in ("en", "hi", "bn"):
            if norm(r[lang]):
                groups.union(node, f"{lang}:{norm(r[lang])}")
    english = sorted({norm(r["en"]) for r in rows if r["source"] != "indicguard_jailbreak" and norm(r["en"])})
    pairs = near_duplicate_pairs(english, NEAR_DUP)
    for a, b in pairs:
        groups.union(f"en:{english[a]}", f"en:{english[b]}")
    near = len(pairs)
    for i, r in enumerate(rows):
        r["group_id"] = hashlib.sha256(groups.find(f"row{i}").encode()).hexdigest()[:16]

    sealed = sealed_texts()
    v1, v1_files = v1_texts()
    sealed_groups = {r["group_id"] for r in rows if norm(r["en"]) in sealed or norm(r["bn"]) in sealed}
    v1_groups = {r["group_id"] for r in rows if any(norm(r[k]) in v1 for k in ("en", "hi", "bn"))}

    kept, removed, seen = [], [], set()
    for r in sorted(rows, key=lambda x: (TIER_ORDER[x["tier"]], x["item_id"])):
        key = (norm(r["en"]), norm(r["hi"]))
        reason = ("sealed" if r["group_id"] in sealed_groups else
                  "used_in_v1" if r["group_id"] in v1_groups else
                  "empty" if not key[0] or not key[1] else
                  "no_devanagari" if not re.search(r"[ऀ-ॿ]", r["hi"]) else
                  "duplicate" if key in seen else None)
        if reason:
            removed.append(dict(r, removed_reason=reason))
        else:
            seen.add(key)
            kept.append(r)

    with open(V2 / "after_exclusions.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(V2 / "removed_stage2.jsonl", "w", encoding="utf-8") as f:
        for r in removed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rng = random.Random(SEED)
    pilot = rng.sample(kept, 50) + rng.sample(removed, min(50, len(removed)))
    with open(V2 / "pilot_exclude.jsonl", "w", encoding="utf-8") as f:
        for r in pilot:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_reason = Counter((r["tier"], r["source"], r["removed_reason"]) for r in removed)
    kept_table = Counter((r["tier"], r["source"], r["benchmark_label"]) for r in kept)
    group_sizes = Counter(Counter(r["group_id"] for r in kept).values())
    summary = dict(
        input=len(rows), kept=len(kept), removed=len(removed), bengali_text_dropped_wrong_script=bn_nulled,
        near_duplicate_links=near, sealed_text_count=len(sealed), v1_text_count=len(v1), v1_files_read=dict(v1_files),
        removed_by_tier_source_reason=[dict(tier=k[0], source=k[1], reason=k[2], n=v) for k, v in sorted(by_reason.items())],
        kept_by_tier_source_label=[dict(tier=k[0], source=k[1], label=k[2], n=v) for k, v in sorted(kept_table.items())],
        kept_groups=len({r["group_id"] for r in kept}), largest_kept_group=max(group_sizes) if group_sizes else 0,
        groups_over_10_items=sum(v for k, v in group_sizes.items() if k > 10),
        pilot_read=False,
    )
    (V2 / "exclude_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for k in ("input", "kept", "removed", "bengali_text_dropped_wrong_script", "near_duplicate_links", "sealed_text_count", "v1_text_count",
              "kept_groups", "largest_kept_group", "groups_over_10_items"):
        print(f"{k}: {summary[k]}")
    print("v1 files read:", dict(v1_files))
    print("removed:")
    for x in summary["removed_by_tier_source_reason"]:
        print(f"  {x['tier']} {x['source']:22s} {x['reason']:14s} {x['n']}")
    print("kept:")
    for x in summary["kept_by_tier_source_label"]:
        print(f"  {x['tier']} {x['source']:22s} {x['label']:22s} {x['n']}")


if __name__ == "__main__":
    main()
