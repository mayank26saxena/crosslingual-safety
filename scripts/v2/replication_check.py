"""Replication check (docs/plan/2026-11-10_restart.md, addendum 2026-11-13): archived baseline replies against replies regenerated under the
pinned environment.

Usage (needs the Qwen3-8B tokenizer, so it runs where transformers is installed):
  python replication_check.py ARCHIVED.jsonl REGENERATED.jsonl OUT.json
Reports, per file: replies identical, replies whose first 64 tokens are identical, the token position of the first difference (median), and
the share at the token limit in each version. Numbers only.
"""
import json
import statistics
import sys

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def main(archived, regenerated, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    a = {json.loads(l)["item"]: json.loads(l) for l in open(archived, encoding="utf-8")}
    b = {json.loads(l)["item"]: json.loads(l) for l in open(regenerated, encoding="utf-8")}
    assert set(a) == set(b), (len(a), len(b))
    same = prefix64 = 0
    first_diff = []
    for k in a:
        ra, rb = a[k]["reply"], b[k]["reply"]
        if ra == rb:
            same += 1
            prefix64 += 1
            continue
        ta = tok(ra, add_special_tokens=False)["input_ids"]
        tb = tok(rb, add_special_tokens=False)["input_ids"]
        d = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        first_diff.append(d)
        prefix64 += d >= 64
    n = len(a)
    res = {"n": n, "identical": same, "identical_share": round(same / n, 4), "first_64_tokens_identical": prefix64,
           "first_64_share": round(prefix64 / n, 4), "median_first_difference_token": statistics.median(first_diff) if first_diff else None,
           "hit_limit_archived": sum(v["hit_limit"] for v in a.values()), "hit_limit_regenerated": sum(v["hit_limit"] for v in b.values())}
    json.dump(res, open(out, "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main(*sys.argv[1:4])
