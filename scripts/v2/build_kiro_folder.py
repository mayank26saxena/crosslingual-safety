"""Build a blind Kiro grading folder from paired replies in two languages.

Rows from both languages are shuffled under ids R0001... that hide pairing and language, split into 50-row parts, and written with the
four-label prompt and the run checker from notes/hindi_calibration/kiro_grading/. The key (id -> pair, language, reply length) is written
outside the folder, so graders never see it.

Usage:
  python scripts/v2/build_kiro_folder.py NAME LANG_LABEL PROMPTS_A REPLIES_A PROMPTS_B REPLIES_B
  e.g. python scripts/v2/build_kiro_folder.py spanish_aya "Spanish" data/v2/spanish_prompts_es.jsonl results/v2/spanish/es_replies.jsonl \
       data/v2/spanish_prompts_en.jsonl results/v2/spanish/en_replies.jsonl
Writes notes/grading/<NAME>/kiro_grading/ and notes/grading/<NAME>/key.json. Prompt and reply text is never printed.
"""

import csv
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "notes/hindi_calibration/kiro_grading"
SEED = 20261112


def load(path):
    return {json.loads(l)["item"]: json.loads(l) for l in open(path, encoding="utf-8")}


def main(name, lang_label, pa, ra, pb, rb):
    out = ROOT / "notes/grading" / name
    folder = out / "kiro_grading"
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "input").mkdir(parents=True)
    (folder / "runs").mkdir()
    sides = []
    for p, r in ((pa, ra), (pb, rb)):
        prompts, replies = load(p), load(r)
        if set(prompts) != set(replies):
            raise SystemExit(f"{r}: replies cover {len(replies)} of {len(prompts)} prompts")
        lang = Path(p).stem.rsplit("_", 1)[-1]
        sides.append((lang, prompts, replies))
    if set(sides[0][1]) != set(sides[1][1]):
        raise SystemExit("the two languages cover different items")
    en_prompts = next(s[1] for s in sides if s[0] == "en")
    rows = []
    for lang, prompts, replies in sides:
        for k in prompts:
            user = next(m["content"] for m in prompts[k]["messages"] if m["role"] == "user")
            request_en = next(m["content"] for m in en_prompts[k]["messages"] if m["role"] == "user")
            rows.append((k, lang, request_en, user, replies[k]))
    random.Random(SEED).shuffle(rows)
    width = len(str(len(rows)))
    key, table = {}, []
    for i, (k, lang, req, msg, rep) in enumerate(rows, 1):
        rid = f"R{i:0{width}d}"
        table.append(dict(item=rid, request_english=req, message=msg, reply=rep["reply"]))
        key[rid] = dict(pair=k, lang=lang, hit_limit=rep["hit_limit"], n_new_tokens=rep["n_new_tokens"], max_new_tokens=rep["max_new_tokens"])
    fields = ["item", "request_english", "message", "reply"]
    (folder / "input/items.txt").write_text("\n".join(r["item"] for r in table) + "\n")
    parts = 0
    for j in range(0, len(table), 50):
        parts += 1
        with open(folder / f"input/part_{parts:02d}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(table[j:j + 50])
    prompt = (TEMPLATE / "prompt.md").read_text()
    prompt = prompt.replace("Read the six files input/part_01.csv to input/part_06.csv, 300 rows in total.",
                            f"Read the {parts} files input/part_01.csv to input/part_{parts:02d}.csv, {len(table)} rows in total.")
    prompt = prompt.replace("may be in Hindi or in English", f"may be in {lang_label} or in English")
    prompt = prompt.replace("Read only those six files.", f"Read only those {parts} files.")
    prompt = prompt.replace("Include all 300 items, in the order of part_01 to part_06",
                            f"Include all {len(table)} items, in the order of part_01 to part_{parts:02d}")
    for needle in ("six files", "300 rows", "all 300 items") + (() if lang_label == "Hindi" else ("Hindi or in English",)):
        if needle in prompt:
            raise SystemExit(f"prompt still mentions {needle!r}")
    (folder / "prompt.md").write_text(prompt)
    shutil.copy(TEMPLATE / "check_runs.py", folder / "check_runs.py")
    readme = (TEMPLATE / "README.md").read_text()
    readme = readme.replace("(four labels, 300 rows)", f"(four labels, {len(table)} rows)")
    readme = readme.replace("`input/part_01.csv` … `part_06.csv` | The 300 rows to grade",
                            f"`input/part_01.csv` … `part_{parts:02d}.csv` | The {len(table)} rows to grade")
    readme = readme.replace("`input/items.txt` | The 300 item ids", f"`input/items.txt` | The {len(table)} item ids")
    readme = readme.replace("The rows mix Hindi and English replies", f"The rows mix {lang_label} and English replies")
    readme = readme.replace("Item ids (`R001` to `R300`)", f"Item ids (`{table[0]['item'][0]}{'0' * (width - 1)}1` to `R{len(table)}`)")
    (folder / "README.md").write_text(readme)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(table)} rows in {parts} parts -> {folder}")


if __name__ == "__main__":
    main(*sys.argv[1:7])
