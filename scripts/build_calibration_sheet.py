"""Build the Phase 0 Hindi calibration sheet (v2 plan, 2026-11-11 addendum).

Draws 150 unchanged Qwen3-8B Hindi replies from the Step 7 held-out test, stratified by benchmark label and
Qwen2.5-14B judge verdict, and writes:

  notes/hindi_calibration/hindi_calibration_sheet.csv  prompt, reply, an empty human section and an empty model section
  notes/hindi_calibration/key.json                     item -> source id, label, judge verdict, stratum, half (not for graders)

The sheet holds no judge verdict, label or stratum, so graders are blind to them.

Usage: python scripts/build_calibration_sheet.py
"""

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGED = ROOT / "results/v1_pod/gated_qwen3/test_judged.jsonl"
ITEMS = ROOT / "data/rendered/gated_items.jsonl"
OUT = ROOT / "notes/hindi_calibration"
SEED = 20261111
STRATA = {("harmless", "REFUSE"): 17, ("harmful", "REFUSE"): 58, ("harmful", "ANSWER"): 45, ("harmless", "ANSWER"): 30}

COLUMNS = [
    "item", "request_english", "message_hindi", "reply_hindi",
    # Human section
    "HUMAN_verdict (REFUSE/ANSWER)", "HUMAN_confidence (sure/unsure)", "HUMAN_notes",
    # Model section, filled by merging the model session's output file
    "OPUS_verdict", "OPUS_reason", "FABLE_verdict", "FABLE_reason",
]


def main():
    items = {}
    for line in open(ITEMS, encoding="utf-8"):
        d = json.loads(line)
        items[d["id"]] = d
    pool = {k: [] for k in STRATA}
    for line in open(JUDGED, encoding="utf-8"):
        d = json.loads(line)
        if d["lang"] == "hi" and d["condition"] == "unchanged" and d.get("sample", 0) == 0:
            key = (d["label"], d["verdict"])
            if key in pool:
                pool[key].append(d)
    rng = random.Random(SEED)
    chosen = []
    for key, n in STRATA.items():
        rows = sorted(pool[key], key=lambda r: r["id"])
        if len(rows) < n:
            raise SystemExit(f"stratum {key} has {len(rows)} rows, needs {n}")
        picked = rng.sample(rows, n)
        halves = ["tune"] * (n // 2) + ["gate"] * (n - n // 2)
        rng.shuffle(halves)
        chosen += [(r, key, h) for r, h in zip(picked, halves)]
    rng.shuffle(chosen)

    OUT.mkdir(parents=True, exist_ok=True)
    key_out = {}
    with open(OUT / "hindi_calibration_sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i, (r, (label, verdict), half) in enumerate(chosen, 1):
            item = f"H{i:03d}"
            src = items[r["id"]]
            user = next(m["content"] for m in src["messages"] if m["role"] == "user")
            w.writerow([item, src["hidden_fact"], user, r["response"]] + [""] * (len(COLUMNS) - 4))
            key_out[item] = {"source_id": r["id"], "task_id": r["task_id"], "label": label, "judge_verdict": verdict,
                             "half": half, "judge": r.get("judge")}
    (OUT / "key.json").write_text(json.dumps(key_out, indent=1), encoding="utf-8")
    counts = {}
    for v in key_out.values():
        counts[(v["label"], v["judge_verdict"], v["half"])] = counts.get((v["label"], v["judge_verdict"], v["half"]), 0) + 1
    print(len(key_out), "items")
    for k in sorted(counts):
        print(" ", k, counts[k])


if __name__ == "__main__":
    main()
