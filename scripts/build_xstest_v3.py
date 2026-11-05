"""Assemble the Claude-translated XSTest-Bengali set from data/translations/ (sensitivity analysis).

Usage: python scripts/build_xstest_v3.py --out data/rendered/xstest_v3_raw.jsonl
"""

import argparse
import csv
import json

from common import ROOT, write_jsonl

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
TRANSLATOR = "claude-opus-5-5 (no native review)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bn = {}
    for name in ("safe", "unsafe"):
        bn.update(json.loads((ROOT / f"data/translations/xstest_{name}_claude_bn.json").read_text()))
    rows = []
    for r in csv.DictReader(open(ROOT / "data/external/xstest/xstest_prompts.csv")):
        task_id = f"xstest_{int(r['id']):03d}"
        rows.append(dict(task_id=task_id, id=f"{task_id}__bn", lang="bn", label=r["label"], category=r["type"], focus=r["focus"],
                         source_en=r["prompt"], translator=TRANSLATOR,
                         messages=[SYSTEM, {"role": "user", "content": bn[r["id"]]}]))
    write_jsonl(ROOT / args.out, rows)
    print(f"{len(rows)} items -> {args.out}")


if __name__ == "__main__":
    main()
