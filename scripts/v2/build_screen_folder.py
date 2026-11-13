"""Blind screening folder for the baseline items (tier A and the PolyGuard harmless control).

Usage: python3 scripts/v2/build_screen_folder.py
Writes notes/screening/baseline/input/part_NN.csv (item, request_english, request_hindi; 50 rows each, shuffled) and
notes/screening/baseline/key.json (id -> item_id, tier, source, benchmark_label). Text is never printed.
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261115


def main():
    d = ROOT / "data/v2"
    ids = [json.loads(l)["item"] for f in ("tierA_prompts_hi.jsonl", "pg_harmless_prompts_hi.jsonl") for l in open(d / f, encoding="utf-8")]
    meta = {json.loads(l)["item_id"]: json.loads(l) for l in open(d / "after_exclusions.jsonl", encoding="utf-8")}
    rows = [meta[i] for i in ids]
    random.Random(SEED).shuffle(rows)
    out = ROOT / "notes/screening/baseline"
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "runs/opus").mkdir(parents=True, exist_ok=True)
    key = {}
    parts = 0
    for j in range(0, len(rows), 50):
        parts += 1
        with open(out / f"input/part_{parts:02d}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item", "request_english", "request_hindi"], quoting=csv.QUOTE_ALL)
            w.writeheader()
            for k, r in enumerate(rows[j:j + 50]):
                sid = f"B{j + k + 1:04d}"
                key[sid] = {x: r[x] for x in ("item_id", "tier", "source", "benchmark_label")}
                w.writerow({"item": sid, "request_english": r["en"], "request_hindi": r["hi"]})
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(rows)} items in {parts} parts")


if __name__ == "__main__":
    main()
