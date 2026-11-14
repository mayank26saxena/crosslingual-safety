"""Phase 4 blind folders (docs/prereg/2026-11-14_v2_phase4_confirmation.md). Text is never printed.

Usage:
  python3 scripts/v2/phase4_folders.py screen REPLIES   comprehension screen: notes/screening/comprehension_bn/input/part_NN.csv
                                                         (item, request_english, restatement) and key.json (C-id -> "set/item")
  python3 scripts/v2/phase4_folders.py grade REPLIES    reply grading: notes/grading/phase4/input/part_NN.csv
                                                         (item, request_english, message, reply) and key.json
  python3 scripts/v2/phase4_folders.py opus             second-grader sample: 150 graded Bengali replies (seed 20261119, both conditions)
                                                         into notes/grading/phase4_opus/, same columns, new ids
Reply grading follows Phase 3: every unsteered reply is graded; a gated reply is graded when its gate was open and its text differs from
the unsteered reply of the same prompt, otherwise it takes that label (same_as). MMLU and translation rows are not graded here. All
conditions and languages are shuffled together (seed 20261117).
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED_GRADE, SEED_SCREEN, SEED_OPUS, N_OPUS = 20261117, 20261116, 20261119, 150


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def user(msgs):
    return next(m["content"] for m in msgs if m["role"] == "user")


def prompts():
    rows = jl(ROOT / "data/v2/phase4_prompts.jsonl")
    return {r["uid"]: r for r in rows}


def write_parts(out, cols, table, prefix):
    (out / "input").mkdir(parents=True, exist_ok=True)
    parts = 0
    for j in range(0, len(table), 50):
        parts += 1
        with open(out / f"input/part_{parts:02d}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(table[j:j + 50])
    (out / "input/items.txt").write_text("\n".join(t["item"] for t in table) + "\n")
    return parts


def screen(replies):
    P = prompts()
    rows = [r for r in jl(replies) if r["set"] == "translate"]
    assert len(rows) == 358
    random.Random(SEED_SCREEN).shuffle(rows)
    out = ROOT / "notes/screening/comprehension_bn"
    (out / "runs/opus").mkdir(parents=True, exist_ok=True)
    key, table = {}, []
    for k, r in enumerate(rows, 1):
        s, i = r["pair"].split("/", 1)
        cid = f"C{k:04d}"
        key[cid] = r["pair"]
        table.append(dict(item=cid, request_english=user(P[f"{s}|{i}|en"]["messages"]), restatement=r["reply"].strip()))
    parts = write_parts(out, ["item", "request_english", "restatement"], table, "C")
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(len(table), "rows in", parts, "parts")


def grade(replies):
    P = prompts()
    rows = [r for r in jl(replies) if r["set"] not in ("mmlu", "translate")]
    base = {r["uid"]: r for r in rows if r["condition"].endswith("_unsteered")}
    sel, copies = [], {}
    for r in rows:
        b = base[r["uid"]]
        if r is b or (r["gate"] and r["reply"] != b["reply"]):
            sel.append(r)
        else:
            copies[f'{r["uid"]}|{r["condition"]}'] = f'{b["uid"]}|{b["condition"]}'
    random.Random(SEED_GRADE).shuffle(sel)
    out = ROOT / "notes/grading/phase4"
    (out / "runs/claude-sonnet-5.5-cc").mkdir(parents=True, exist_ok=True)
    key, table = {}, []
    for k, r in enumerate(sel, 1):
        sid = f"P{k:04d}"
        s, i, lang = r["uid"].split("|")
        key[sid] = {x: r[x] for x in ("uid", "set", "lang", "pair", "condition", "alpha", "gate", "hit_limit", "n_new_tokens")}
        table.append(dict(item=sid, request_english=user(P[f"{s}|{i}|en"]["messages"]), message=user(P[r["uid"]]["messages"]),
                          reply=r["reply"]))
    parts = write_parts(out, ["item", "request_english", "message", "reply"], table, "P")
    (out / "key.json").write_text(json.dumps(dict(graded=key, same_as=copies), indent=1))
    print(f"{len(table)} graded rows in {parts} parts; {len(copies)} gated rows take the unsteered label")


def opus():
    src = ROOT / "notes/grading/phase4"
    key = json.load(open(src / "key.json"))["graded"]
    text = {}
    for f in sorted((src / "input").glob("part_*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            text[r["item"]] = r
    bn = sorted(k for k, v in key.items() if v["lang"] == "bn")
    pick = random.Random(SEED_OPUS).sample(bn, N_OPUS)
    out = ROOT / "notes/grading/phase4_opus"
    (out / "runs/claude-opus-5.5-cc").mkdir(parents=True, exist_ok=True)
    k2, table = {}, []
    for k, sid in enumerate(pick, 1):
        oid = f"O{k:03d}"
        k2[oid] = dict(key[sid], source_item=sid)
        t = text[sid]
        table.append(dict(item=oid, request_english=t["request_english"], message=t["message"], reply=t["reply"]))
    parts = write_parts(out, ["item", "request_english", "message", "reply"], table, "O")
    (out / "key.json").write_text(json.dumps(dict(graded=k2), indent=1))
    print(len(table), "rows in", parts, "parts")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"screen": lambda: screen(a[1]), "grade": lambda: grade(a[1]), "opus": opus}[a[0]]()
