"""Draw the 100-item human audit sample (request type and translation fidelity) and write a blind sheet.

Usage: python3 scripts/v2/build_audit_sheet.py
Design: docs/plan/2026-11-11_hindi_dataset_spec.md, addendum 2026-11-13.
Writes notes/audit/audit_sheet.csv (blind: id, English, Hindi, empty HUMAN_ columns) and notes/audit/audit_key.json (id -> item, tier,
source, label). Request text is never printed.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261113
PLAN = {
    ("aya_redteaming", "harmful"): 25,
    ("indicsafe", "harmful"): 8, ("indicsafe", "harmless"): 4, ("indicsafe", "ambiguous"): 3,
    ("indicguard_culture", "harmful"): 12, ("indicguard_culture", "harmless"): 8,
    ("indicguard_generic", "harmful"): 10, ("indicguard_generic", "harmless"): 6,
    ("indicguard_jailbreak", "harmful"): 6, ("indicguard_jailbreak", "harmless"): 4,
    ("polyguard", "harmful"): 5, ("polyguard", "harmless"): 5,
    ("mhh", "harmful"): 2, ("mhh", "harmless"): 2,
}
COLS = ["item", "request_english", "request_hindi",
        "HUMAN_request_type (request_harmful/request_borderline_safe/request_benign/not_a_request)",
        "HUMAN_fidelity (equivalent/minor_drift/wrong)", "HUMAN_confidence (sure/unsure)", "HUMAN_notes"]


def main():
    pool = defaultdict(list)
    for line in open(ROOT / "data/v2/after_exclusions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        pool[(r["source"], r["benchmark_label"])].append(r)
    rng = random.Random(SEED)
    picked = []
    for key, n in PLAN.items():
        items = sorted(pool[key], key=lambda r: r["item_id"])
        assert len(items) >= n, key
        picked += rng.sample(items, n)
    assert len(picked) == 100
    rng.shuffle(picked)
    out = ROOT / "notes/audit"
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    with open(out / "audit_sheet.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for i, r in enumerate(picked, 1):
            aid = f"A{i:03d}"
            key[aid] = {"item_id": r["item_id"], "tier": r["tier"], "source": r["source"], "benchmark_label": r["benchmark_label"]}
            w.writerow({"item": aid, "request_english": r["en"], "request_hindi": r["hi"], **{c: "" for c in COLS[3:]}})
    (out / "audit_key.json").write_text(json.dumps(key, indent=1))
    print(f"100 items -> {out / 'audit_sheet.csv'}")


if __name__ == "__main__":
    main()
