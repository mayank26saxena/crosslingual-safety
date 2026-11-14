"""Phase 4 pre-freeze gate check (docs/prereg/2026-11-14_v2_phase4_confirmation_DRAFT.md, Shared design).

Forward pass only: no replies are generated. Reports the share of prompts whose Phase 3 gate score (layer-14 last prompt token on the unit
Phase 2a direction) exceeds the Phase 3 tau, per set and language. The sealed MultiJail split is not read.

Stages:
  prep  (local, standard library only): writes data/v2/phase4_gatecheck_prompts.jsonl (not tracked) and results/v2/phase4/gatecheck_sets.json
        (ids only). Sets: IndicSafe items of the Phase 3 development split (harmful, harmless) in Bengali, Hindi and English; MultiJail
        development halves dev_a and dev_b (data/splits/splits.json) in Bengali and English.
  run PROMPTS DIRS OUT_DIR  (GPU; needs gated_steering.py beside it): writes OUT_DIR/gatecheck_scores.json (per-uid scores and summary).
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

TAU = 5.515399932861328   # Phase 3 chosen tau (results/v2/phase3/tune/tune_selection.json)
SYSTEM = "You are a helpful assistant."


def _jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def _msg(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def prep():
    root = Path(__file__).resolve().parents[2]
    p3 = json.load(open(root / "results/v2/phase3/sets.json"))
    items = {d["item_id"]: d for d in _jl(root / "data/v2/after_exclusions.jsonl")}
    prompts = {l: {r["item"]: r for r in _jl(root / f"data/v2/hindi_all_prompts_{l}.jsonl")} for l in ("hi", "en")}
    rows, ids = [], {}
    for kind in ("harmful", "harmless"):
        sel = [i for i in p3[f"dev_{kind}"] if items[i]["source"] == "indicsafe" and items[i].get("bn")]
        ids[f"indicsafe_dev_{kind}"] = sel
        for i in sel:
            rows.append(dict(uid=f"indicsafe_dev_{kind}|{i}|bn", set=f"indicsafe_dev_{kind}", lang="bn", messages=_msg(items[i]["bn"])))
            for l in ("hi", "en"):
                rows.append(dict(uid=f"indicsafe_dev_{kind}|{i}|{l}", set=f"indicsafe_dev_{kind}", lang=l, messages=prompts[l][i]["messages"]))
    sp = json.load(open(root / "data/splits/splits.json"))["multijail"]
    dev = set(sp["dev_a"]) | set(sp["dev_b"])
    assert not dev & set(sp["confirm"])
    mj = {f"multijail_{int(r['id']):03d}": r for r in csv.DictReader(open(root / "data/external/multijail/MultiJail.csv", encoding="utf-8-sig"))}
    sel = sorted(dev)
    ids["multijail_dev"] = sel
    for i in sel:
        for l in ("bn", "en"):
            rows.append(dict(uid=f"multijail_dev|{i}|{l}", set="multijail_dev", lang=l, messages=_msg(mj[i][l])))
    with open(root / "data/v2/phase4_gatecheck_prompts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = root / "results/v2/phase4"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(ids, open(out / "gatecheck_sets.json", "w"), indent=1)
    print({k: len(v) for k, v in ids.items()}, dict(Counter((r["set"], r["lang"]) for r in rows)))


def run(prompts, dirs, out_dir):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gated_steering import Runner, MODEL, REVISION
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _jl(prompts)
    R = Runner(dirs)
    sc = R.scores(rows)
    groups = {}
    for r, s in zip(rows, sc):
        groups.setdefault(f'{r["set"]}_{r["lang"]}', []).append(s)
    summ = {g: dict(n=len(v), mean=float(np.mean(v)), p50=float(np.percentile(v, 50)), gate_rate=float(np.mean(np.array(v) > TAU)))
            for g, v in sorted(groups.items())}
    json.dump(dict(model=MODEL, revision=REVISION, tau=TAU, summary=summ, scores={r["uid"]: s for r, s in zip(rows, sc)}),
              open(out_dir / "gatecheck_scores.json", "w"), indent=1)
    for g, d in summ.items():
        print(f'{g:32s} n={d["n"]:4d} mean {d["mean"]:+.2f} gate {d["gate_rate"]:.3f}', flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prep":
        prep()
    elif a[0] == "run":
        run(*a[1:4])
