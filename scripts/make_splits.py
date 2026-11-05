"""Fix the item splits used by every later step. Writes request ids only, never request text.

MultiJail: "dev" is the 150 requests measured in the baseline run; "confirm" is the other 165, whose replies have never
been generated. Dev is split into halves A and B within each source. XSTest: prompts are split into halves within each
type, so both halves keep the benchmark's category balance.

Usage: python scripts/make_splits.py
"""

import csv
import json
import random
from collections import defaultdict

from common import ROOT, read_jsonl

SEED = 20261104
OUT = ROOT / "data/splits/splits.json"


def halves(ids_by_group, rng):
    a, b = [], []
    for _, ids in sorted(ids_by_group.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        a += ids[: len(ids) // 2]
        b += ids[len(ids) // 2:]
    return sorted(a), sorted(b)


def main():
    rng = random.Random(SEED)
    rows = list(csv.DictReader(open(ROOT / "data/external/multijail/MultiJail.csv", encoding="utf-8-sig")))
    source = {f"multijail_{int(r['id']):03d}": r["source"] for r in rows}
    dev = {r["task_id"] for r in read_jsonl(ROOT / "data/rendered/multijail_gap.jsonl")}
    by_source = defaultdict(list)
    for task in dev:
        by_source[source[task]].append(task)
    dev_a, dev_b = halves(by_source, rng)

    xs = list(csv.DictReader(open(ROOT / "data/external/xstest/xstest_prompts.csv")))
    by_type = {"safe": defaultdict(list), "unsafe": defaultdict(list)}
    for r in xs:
        by_type[r["label"]][r["type"]].append(f"xstest_{int(r['id']):03d}")
    safe_a, safe_b = halves(by_type["safe"], rng)
    unsafe_a, unsafe_b = halves(by_type["unsafe"], rng)

    splits = dict(seed=SEED,
                  multijail=dict(dev_a=dev_a, dev_b=dev_b, confirm=sorted(set(source) - dev)),
                  xstest=dict(safe_a=safe_a, safe_b=safe_b, unsafe_a=unsafe_a, unsafe_b=unsafe_b))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(splits, indent=1))
    for name, group in (("multijail", splits["multijail"]), ("xstest", splits["xstest"])):
        print(name, {k: len(v) for k, v in group.items()})


if __name__ == "__main__":
    main()
