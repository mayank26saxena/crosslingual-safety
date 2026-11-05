"""Select the existing dev replies re-graded in Step 1.

Writes results/step1/replies_en_bn.jsonl (all English and Bengali dev replies) and
results/step1/translate_arm_ids.json (150 Bengali replies stratified by primary verdict).

Usage: python scripts/prepare_step1.py
"""

import json
import random
from collections import defaultdict

from common import ROOT, read_jsonl, write_jsonl

BASE = ROOT / "results/earlier_neuron_steering/qwen2.5-7b-instruct/multijail/judged_multijail_gap.jsonl"
OUT = ROOT / "results/step1"
KEEP = ("id", "task_id", "sample", "model", "response")


def main():
    rows = [r for r in read_jsonl(BASE) if r["id"].split("__")[-1] in ("en", "bn")]
    clean = [{**{k: r[k] for k in KEEP}, "primary_verdict": r["verdict"]} for r in rows]
    write_jsonl(OUT / "replies_en_bn.jsonl", clean)

    rng = random.Random(20261104)
    by_verdict = defaultdict(list)
    for i, r in enumerate(clean):
        if r["id"].endswith("__bn"):
            by_verdict[r["primary_verdict"]].append(i)
    total = sum(map(len, by_verdict.values()))
    chosen = []
    for verdict, idx in sorted(by_verdict.items()):
        chosen += rng.sample(idx, min(len(idx), max(1, round(150 * len(idx) / total))))
    write_jsonl(OUT / "translate_arm_replies.jsonl", [clean[i] for i in sorted(chosen)])
    print(f"{len(clean)} replies; translate arm {len(chosen)}",
          {v: sum(clean[i]['primary_verdict'] == v for i in chosen) for v in by_verdict})


if __name__ == "__main__":
    main()
