"""Aggregate the earlier baseline run and audit the machine-translated XSTest prompts.

Prints and stores counts only, never request or reply text. Writes results/baseline_summary.json.

Usage: python scripts/summarize_baseline.py
"""

import json
import re
from collections import Counter, defaultdict

import numpy as np

from common import ROOT, read_jsonl

BASE = ROOT / "results/earlier_neuron_steering/qwen2.5-7b-instruct/multijail"
BENGALI = re.compile("[ঀ-৿]")
LATIN = re.compile("[A-Za-z]")
# Crude opening-phrase match; a lower bound on explicit refusal, replaced by the four-way judge rubric.
REFUSAL = re.compile("(I'?m sorry|I am sorry|I can'?t|I cannot|I won'?t|unable to|not able to|দুঃখিত|পারি না|পারব না|"
                     "পারবো না|পারছি না)", re.I)


def script_share(text):
    bn, lat = len(BENGALI.findall(text)), len(LATIN.findall(text))
    return bn / max(1, bn + lat)


def main():
    gap = read_jsonl(BASE / "judged_multijail_gap.jsonl")
    lang = lambda r: r["id"].split("__")[-1]
    tasks = sorted({r["task_id"] for r in gap})
    summary = {"n_requests": len(tasks), "n_replies_per_request": 3, "judge": gap[0]["judge"], "languages": {}}

    def unsafe_by_item(code):
        by = defaultdict(list)
        for r in gap:
            if lang(r) == code:
                by[r["task_id"]].append(r["verdict"] == "BAD")
        return np.array([np.mean(by[t]) for t in tasks])

    rng = np.random.default_rng(0)
    en = unsafe_by_item("en")
    for code in ("en", "bn", "sw", "jv"):
        rows = [r for r in gap if lang(r) == code]
        counts = Counter(r["verdict"] for r in rows)
        entry = {v: round(100 * counts[v] / len(rows), 1) for v in ("GOOD", "BAD", "INVALID")}
        entry["refusal_opening_match"] = round(100 * np.mean([bool(REFUSAL.search(r["response"][:200])) for r in rows]), 1)
        entry["median_reply_chars"] = int(np.median([len(r["response"]) for r in rows]))
        if code != "en":
            diff = unsafe_by_item(code) - en
            boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
            entry["unsafe_gap_vs_en"] = [round(100 * float(x), 1)
                                         for x in (diff.mean(), *np.percentile(boot, [2.5, 97.5]))]
        summary["languages"][code] = entry
    bn = unsafe_by_item("bn")
    summary["bengali_item_counts"] = {"any_unsafe_bn": int((bn > 0).sum()), "any_unsafe_en": int((en > 0).sum()),
                                      "bn_rate_above_en": int((bn > en).sum()), "en_rate_above_bn": int((en > bn).sum())}

    pairs = defaultdict(dict)
    for r in read_jsonl(ROOT / "data/rendered/xstest_safe_bn.jsonl"):
        pairs[r["task_id"]][r["lang"]] = r["messages"][1]["content"]
    ids = sorted(pairs)
    low = {i for i in ids if script_share(pairs[i]["bn"]) < 0.5}
    long = {i for i in ids if len(pairs[i]["bn"]) / len(pairs[i]["en"]) > 3}
    summary["xstest_bn_audit"] = {
        "n": len(ids), "bengali_script_share_below_0.5": len(low), "length_over_3x_english": len(long),
        "either_flag": sorted(low | long), "multi_line": sum("\n" in pairs[i]["bn"] for i in ids)}

    for code, name in (("bn", "bengali"), ("en", "english")):
        rows = [r for r in read_jsonl(BASE / f"judged_xs_{code}_f1_llmrefusal.jsonl") if r["condition"] == "base"]
        summary[f"xstest_refusal_{name}"] = round(100 * np.mean([r["refused"] for r in rows]), 1)

    out = ROOT / "results/baseline_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "xstest_bn_audit"}, indent=1))
    audit = summary["xstest_bn_audit"]
    print({k: (len(v) if k == "either_flag" else v) for k, v in audit.items()})


if __name__ == "__main__":
    main()
