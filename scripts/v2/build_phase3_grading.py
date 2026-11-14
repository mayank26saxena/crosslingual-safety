"""Blind grading folder for Phase 3 replies (all conditions shuffled together).

Usage: python3 scripts/v2/build_phase3_grading.py REPLIES.jsonl NAME
  e.g. results/v2/phase3/tune/tune_replies.jsonl phase3_tune
Every unsteered reply is graded. A steered reply is graded when its gate was open and its text differs from the unsteered reply of the same
prompt; otherwise it takes that reply's label (recorded in the key as same_as). General-ability (mmlu) rows are not graded here.
Writes notes/grading/NAME/input/part_NN.csv (item, request_english, message, reply), input/items.txt and notes/grading/NAME/key.json.
Text is never printed.
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261117


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def user(msgs):
    return next(m["content"] for m in msgs if m["role"] == "user")


def main(replies, name):
    d = ROOT / "data/v2"
    text = {l: {r["item"]: user(r["messages"]) for r in jl(d / f"hindi_all_prompts_{l}.jsonl")} for l in ("hi", "en")}
    rows = [r for r in jl(replies) if r["set"] != "mmlu"]
    base = {r["uid"]: r for r in rows if r["alpha"] == 0}
    grade, copies = [], {}
    for r in rows:
        b = base[r["uid"]]
        if r["alpha"] == 0 or (r["gate"] and r["reply"] != b["reply"]):
            grade.append(r)
        else:
            copies[f'{r["uid"]}|{r["condition"]}'] = f'{b["uid"]}|{b["condition"]}'
    random.Random(SEED).shuffle(grade)
    out = ROOT / "notes/grading" / name
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "runs/claude-sonnet-5.5-cc").mkdir(parents=True, exist_ok=True)
    key, table = {}, []
    for i, r in enumerate(grade, 1):
        sid = f"P{i:04d}"
        key[sid] = {k: r[k] for k in ("uid", "set", "lang", "pair", "condition", "alpha", "gate", "hit_limit", "n_new_tokens")}
        table.append(dict(item=sid, request_english=text["en"][r["pair"]], message=text[r["lang"]][r["pair"]], reply=r["reply"]))
    parts = 0
    for j in range(0, len(table), 50):
        parts += 1
        with open(out / f"input/part_{parts:02d}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item", "request_english", "message", "reply"], quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(table[j:j + 50])
    (out / "input/items.txt").write_text("\n".join(t["item"] for t in table) + "\n")
    (out / "key.json").write_text(json.dumps(dict(graded=key, same_as=copies), indent=1))
    print(f"{len(table)} graded rows in {parts} parts; {len(copies)} steered rows take the unsteered label")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
