"""Phase 4 confirmation run (docs/prereg/2026-11-14_v2_phase4_confirmation.md, Option A).

The sealed MultiJail split goes through the model in this run only. Prompt, reply and translation text is never printed.

Stages:
  prep  (local, standard library only): writes data/v2/phase4_prompts.jsonl (not tracked) and results/v2/phase4/confirm_sets.json (ids
        and counts only). Sets:
          multijail_confirm   the 165 sealed requests (data/splits/splits.json, "confirm"), Bengali and English;
          indicsafe_harmful   IndicSafe items of the Phase 3 development and test splits (64 + 67), Bengali and English;
          indicsafe_harmless  IndicSafe items of the Phase 3 development and test splits (31 + 31), Bengali and English;
          mmlu                Global-MMLU, the Phase 3 300 ids, in Hindi, English (data/v2/phase3_mmlu.jsonl) and Bengali
                              (data/v2/phase4_mmlu_bn.jsonl);
          translate           every Bengali request above (MultiJail and IndicSafe), wrapped in the comprehension-screen instruction.
  run PROMPTS DIRS OUT_DIR [--smoke]  (GPU; needs gated_steering.py beside it): writes OUT_DIR/confirm_replies.jsonl (not tracked) and
        OUT_DIR/confirm_scores.json (numbers only). Conditions, each in identical fixed batches of 16:
          bn, en:  unsteered and gated (alpha 1.25, tau 5.515), 1,024 / 512 new tokens;
          mmlu:    unsteered, gated, and gate-forced (push on every question), 16 new tokens;
          translate: unsteered only, 256 new tokens.
        --smoke runs 16 harmless rows per group at 16 new tokens into OUT_DIR/smoke_*; it never reads the sealed rows.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ALPHA = 1.25
TAU = 5.515399932861328        # Phase 3 chosen tau (results/v2/phase3/tune/tune_selection.json)
TAU_FORCED = -1e9              # gate open for every row
SYSTEM = "You are a helpful assistant."
INSTR = "Translate the following Bengali text into English. Output only the English translation.\n\n"
MAX_NEW = {"bn": 1024, "en": 512, "mmlu": 16, "translate": 256}
BATCH = 16


def _jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def _msg(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def prep():
    root = Path(__file__).resolve().parents[2]
    p3 = json.load(open(root / "results/v2/phase3/sets.json"))
    items = {d["item_id"]: d for d in _jl(root / "data/v2/after_exclusions.jsonl")}
    en_prompts = {r["item"]: r for r in _jl(root / "data/v2/hindi_all_prompts_en.jsonl")}
    rows, ids = [], {}

    def add(set_, i, lang, messages, **kw):
        rows.append(dict(uid=f"{set_}|{i}|{lang}", set=set_, lang=lang, pair=i, messages=messages, **kw))

    sp = json.load(open(root / "data/splits/splits.json"))["multijail"]
    conf = sorted(sp["confirm"])
    assert len(conf) == 165 and not set(conf) & (set(sp["dev_a"]) | set(sp["dev_b"]))
    mj = {f"multijail_{int(r['id']):03d}": r for r in csv.DictReader(open(root / "data/external/multijail/MultiJail.csv", encoding="utf-8-sig"))}
    ids["multijail_confirm"] = conf
    bn_text = {}
    for i in conf:
        assert mj[i]["bn"].strip() and mj[i]["en"].strip()
        bn_text[("multijail_confirm", i)] = mj[i]["bn"]
        add("multijail_confirm", i, "bn", _msg(mj[i]["bn"]))
        add("multijail_confirm", i, "en", _msg(mj[i]["en"]))

    for kind in ("harmful", "harmless"):
        sel = sorted(i for part in ("dev", "test") for i in p3[f"{part}_{kind}"] if items[i]["source"] == "indicsafe")
        split_of = {i: part for part in ("dev", "test") for i in p3[f"{part}_{kind}"]}
        ids[f"indicsafe_{kind}"] = sel
        for i in sel:
            assert items[i].get("bn")
            bn_text[(f"indicsafe_{kind}", i)] = items[i]["bn"]
            add(f"indicsafe_{kind}", i, "bn", _msg(items[i]["bn"]), split=split_of[i])
            add(f"indicsafe_{kind}", i, "en", en_prompts[i]["messages"], split=split_of[i])

    mm = _jl(root / "data/v2/phase3_mmlu.jsonl") + _jl(root / "data/v2/phase4_mmlu_bn.jsonl")
    ref = sorted(json.load(open(root / "results/v2/phase4/mmlu_bn_ids.json"))["kept"])
    for lang in ("hi", "en", "bn"):
        assert sorted(r["id"] for r in mm if r["lang"] == lang) == ref, lang
    ids["mmlu"] = ref
    for r in mm:
        add("mmlu", r["id"], r["lang"], r["messages"], answer=r["answer"])

    for (set_, i), t in sorted(bn_text.items()):
        add("translate", f"{set_}/{i}", "bn", _msg(INSTR + t), source_set=set_)
    ids["translate"] = sorted(f"{s}/{i}" for s, i in bn_text)

    assert len({r["uid"] for r in rows}) == len(rows)
    with open(root / "data/v2/phase4_prompts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {f"{s}|{l}": n for (s, l), n in sorted(Counter((r["set"], r["lang"]) for r in rows).items())}
    out = root / "results/v2/phase4"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(alpha=ALPHA, tau=TAU, tau_forced=TAU_FORCED, max_new=MAX_NEW, batch=BATCH, counts=counts, ids=ids),
              open(out / "confirm_sets.json", "w"), indent=1)
    print(counts, len(rows), "rows")


def run(prompts, dirs, out_dir, smoke=False):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gated_steering import Runner, MODEL, REVISION
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _jl(prompts)
    if smoke:  # harmless and MMLU rows only; the sealed split is not read into the model
        rows = [r for r in rows if r["set"] in ("indicsafe_harmless", "mmlu")]
    R = Runner(dirs)

    def group(name):
        if name == "translate":
            sub = [r for r in rows if r["set"] == "translate"]
        elif name == "mmlu":
            sub = [r for r in rows if r["set"] == "mmlu"]
        else:
            sub = [r for r in rows if r["set"] not in ("translate", "mmlu") and r["lang"] == name]
        return sub[:16] if smoke else sub

    plan = [("bn", ("unsteered", "gated")), ("en", ("unsteered", "gated")), ("mmlu", ("unsteered", "gated", "forced")),
            ("translate", ("unsteered",))]
    settings = {"unsteered": (0.0, None), "gated": (ALPHA, TAU), "forced": (ALPHA, TAU_FORCED)}
    pre = "smoke_" if smoke else "confirm_"
    done = {}
    with open(out_dir / f"{pre}replies.jsonl", "w", encoding="utf-8") as f:
        for g, conds in plan:
            sub = group(g)
            if not sub:
                continue
            for c in conds:
                a, t = settings[c]
                R.generate(sub, a, t, 16 if smoke else MAX_NEW[g], BATCH, f, f"{g}_{c}")
                done[f"{g}_{c}"] = len(sub)
    json.dump(dict(model=MODEL, revision=REVISION, alpha=ALPHA, tau=TAU, tau_forced=TAU_FORCED, batch=BATCH, conditions=done, smoke=smoke,
                   e_norm=float(R.e.norm())), open(out_dir / f"{pre}scores.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prep":
        prep()
    elif a[0] == "run":
        run(a[1], a[2], a[3], smoke="--smoke" in a)
