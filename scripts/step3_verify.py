"""Recompute Step 3 S2 from the graded replies, independently of step3.py report.

Adds what the report does not compute: Holm-adjusted bootstrap p-values across the four non-English languages (English is the
positive control), per-language and per-condition judge agreement, and a check that the recomputed rates match s2.json.

Usage: python scripts/step3_verify.py results/step3
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

SEED = 20261105
N_BOOT = 1000
N_RANDOM = 5
TARGETS = ("zh", "ar", "bn", "sw")


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def boot_p(diff, rng):
    """Two-sided paired bootstrap p-value for a mean difference, and its 95% interval."""
    idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
    means = diff[idx].mean(1)
    tail = min((means <= 0).mean(), (means >= 0).mean())
    return float(min(1.0, 2 * (tail + 1 / N_BOOT))), [float(x) for x in np.percentile(means, [2.5, 97.5])]


def holm(pvals):
    """Holm step-down adjustment for a {name: p} mapping."""
    order = sorted(pvals, key=pvals.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvals[name]))
        adjusted[name] = running
    return adjusted


def kappa(pairs):
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    if len(set(a) | set(b)) < 2:
        return None
    return float(cohen_kappa_score(a, b))


def main(step_dir):
    root = Path(step_dir)
    rng = np.random.default_rng(SEED)
    main_rows = load(root / "judged_refusal.jsonl")
    second = {(r["id"], r["condition"]): r["verdict"] for r in load(root / "judged_refusal_llama.jsonl")}
    registered = json.loads((root / "s2.json").read_text())
    out = dict(step_dir=step_dir, rows=len(main_rows), second_rows=len(second), agreement={}, comparisons={}, mismatches=[])

    by_lang, by_action = defaultdict(list), defaultdict(list)
    for r in main_rows:
        other = second.get((r["id"], r["condition"]))
        if r["verdict"] and other:
            by_lang[r["lang"]].append((r["verdict"], other))
            by_action[(r["lang"] != "en", r["action"])].append((r["verdict"], other))
    out["agreement"]["by_language"] = {k: dict(n=len(v), kappa=kappa(v), raw=float(np.mean([a == b for a, b in v])),
                                               main_refuse=float(np.mean([a == "REFUSE" for a, _ in v])),
                                               second_refuse=float(np.mean([b == "REFUSE" for _, b in v])))
                                       for k, v in sorted(by_lang.items())}
    out["agreement"]["by_action"] = {f"{'non_en' if k[0] else 'en'}/{k[1]}": dict(n=len(v), kappa=kappa(v)) for k, v in sorted(by_action.items())}
    pooled = [p for k, v in by_lang.items() if k != "en" for p in v]
    out["agreement"]["pooled_non_en"] = kappa(pooled)
    out["agreement"]["en"] = kappa(by_lang["en"])

    tables = {"harmful": defaultdict(lambda: defaultdict(dict)), "harmless": defaultdict(lambda: defaultdict(dict))}
    for r in main_rows:
        name = "harmful" if r["id"].startswith("multijail") else "harmless"
        tables[name][r["lang"]][r["condition"]][r["task_id"]] = r["verdict"] == "REFUSE"

    for name, table in tables.items():
        raw = defaultdict(dict)
        for lang, conds in table.items():
            tasks = sorted(conds["unchanged/none"])
            for cond, values in conds.items():
                if set(values) != set(tasks):
                    out["mismatches"].append(f"{name}/{lang}/{cond}: {len(values)} of {len(tasks)} items")
            vec = lambda c: np.array([conds[c][t] for t in tasks], float)
            for action in ("remove", "add"):
                if f"{action}/refusal" not in conds:
                    continue
                sign = -1 if action == "remove" else 1
                target = vec(f"{action}/refusal")
                rand = np.mean([vec(f"{action}/random{k}") for k in range(N_RANDOM)], 0)
                matched = vec(f"{action}/matched")
                saved = registered[name][lang][action]
                for label, mine, theirs in (("refusal", target.mean(), saved["refusal"]), ("matched", matched.mean(), saved["matched"]),
                                            ("random", rand.mean(), saved["random"])):
                    if abs(mine - theirs) > 1e-9:
                        out["mismatches"].append(f"{name}/{lang}/{action}/{label}: {mine} vs {theirs}")
                for control, ctrl in (("random", rand), ("matched", matched)):
                    diff = sign * (target - ctrl)
                    p, ci = boot_p(diff, rng)
                    raw[f"{action}_vs_{control}"][lang] = dict(n=len(tasks), diff=float(diff.mean()), ci=ci, p=p)
        for comparison, langs in raw.items():
            adjusted = holm({k: v["p"] for k, v in langs.items() if k in TARGETS})
            for lang, entry in langs.items():
                entry["p_holm"] = adjusted.get(lang)  # None for English, the positive control
            out["comparisons"][f"{name}/{comparison}"] = langs

    verdicts = {}
    for name, action in (("harmful", "remove"), ("harmful", "add"), ("harmless", "add")):
        verdicts[f"{name}/{action}"] = {}
        for lang in ("en",) + TARGETS:
            a = out["comparisons"][f"{name}/{action}_vs_random"].get(lang)
            b = out["comparisons"][f"{name}/{action}_vs_matched"].get(lang)
            if not a or not b:
                continue
            if lang == "en":
                ok = a["ci"][0] > 0 and b["ci"][0] > 0
            else:
                ok = a["diff"] > 0 and b["diff"] > 0 and a["p_holm"] < 0.05 and b["p_holm"] < 0.05
            verdicts[f"{name}/{action}"][lang] = bool(ok)
    out["works_after_holm"] = verdicts
    (root / "s2_verify.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(mismatches=out["mismatches"], works_after_holm=verdicts, agreement=out["agreement"])))


if __name__ == "__main__":
    main(sys.argv[1])
