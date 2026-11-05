"""Compare the probe test and over-refusal baseline between the registered and Claude translations.

Sensitivity analysis added after the registered D1 verdict was seen. It does not change that verdict. Aggregates only.

Usage: python scripts/sensitivity_v3.py
"""

import json
import re

import numpy as np

from common import ROOT, read_jsonl


def trigger(rows):
    def kept(it):
        words = re.findall(r"[a-z]+", it["focus"].lower())
        return bool(words) and all(w[:4] in it["back_translation"].lower() for w in words)
    return {r["task_id"]: bool(r["format_ok"] and kept(r)) for r in rows}


def side(step_dir, rows_path, guard_path):
    d = json.loads((ROOT / step_dir / "diagnostics.json").read_text())["d1_verdict"]
    rows = read_jsonl(ROOT / rows_path)
    keep = trigger(rows)
    guard = read_jsonl(ROOT / guard_path)
    half_b = set(json.loads((ROOT / "data/splits/splits.json").read_text())["xstest"]["safe_b"])
    answered = lambda g: round(100 * float(np.mean([r["verdict"] == "COMPLY" for r in g])), 1)
    safe = [r for r in rows if r["label"] == "safe"]
    return dict(go=d["go"], primary_bn=d["primary"]["auc_bn"], primary_low=d["primary"]["ci_low"], same_bn=d["same_source"]["auc_bn"],
                same_low=d["same_source"]["ci_low"], same_high=d["same_source"]["ci_high"], same_en=d["same_source"]["auc_en"],
                same_diff=d["same_source"]["auc_en"] - d["same_source"]["auc_bn"], format_ok=sum(r["format_ok"] for r in rows),
                safe_format_ok=sum(r["format_ok"] for r in safe), safe_trigger_kept=sum(keep[r["task_id"]] for r in safe),
                guard_answered=answered(guard), guard_n=len(guard),
                guard_answered_half_b=answered([g for g in guard if g["task_id"] in half_b]),
                guard_answered_half_b_trigger=answered([g for g in guard if g["task_id"] in half_b and keep[g["task_id"]]]))


def main():
    out = dict(note="Sensitivity analysis added after the registered NO-GO. The registered verdict stands.",
               v2=side("results/step2", "data/rendered/xstest_v2_bn.jsonl", "results/step1/judged_guard_v2.jsonl"),
               v3=side("results/step2b", "data/rendered/xstest_v3_bn.jsonl", "results/step2b/judged_guard_v3.jsonl"))
    (ROOT / "results/step2b/sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
