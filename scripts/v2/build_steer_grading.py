"""Blind grading folder for the Phase 2b steered replies (all alphas shuffled together).

Usage: python3 scripts/v2/build_steer_grading.py results/v2/es_dir/steer_replies.jsonl
Writes notes/grading/es_dir_steer/input/part_NN.csv (item, request_english, message, reply), input/items.txt, and
notes/grading/es_dir_steer/key.json (id -> uid, set, pair, alpha, hit_limit). Text is never printed.
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261114


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def user(msgs):
    return next(m["content"] for m in msgs if m["role"] == "user")


def main(replies):
    d = ROOT / "data/v2"
    en = {r["item"]: user(r["messages"]) for f in ("tierA_prompts_en.jsonl", "pg_harmless_prompts_en.jsonl") for r in jl(d / f)}
    hi = {r["item"]: user(r["messages"]) for f in ("tierA_prompts_hi.jsonl", "pg_harmless_prompts_hi.jsonl") for r in jl(d / f)}
    rows = jl(replies)
    random.Random(SEED).shuffle(rows)
    out = ROOT / "notes/grading/es_dir_steer"
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "runs/claude-sonnet-5.5-cc").mkdir(parents=True, exist_ok=True)
    key, table = {}, []
    for i, r in enumerate(rows, 1):
        sid = f"S{i:04d}"
        key[sid] = {k: r[k] for k in ("uid", "set", "pair", "alpha", "hit_limit", "n_new_tokens")}
        table.append(dict(item=sid, request_english=en[r["pair"]], message=hi[r["pair"]], reply=r["reply"]))
    parts = 0
    for j in range(0, len(table), 50):
        parts += 1
        with open(out / f"input/part_{parts:02d}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item", "request_english", "message", "reply"], quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(table[j:j + 50])
    (out / "input/items.txt").write_text("\n".join(t["item"] for t in table) + "\n")
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(table)} rows in {parts} parts")


if __name__ == "__main__":
    main(sys.argv[1])
