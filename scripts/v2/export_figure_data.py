"""Export the numbers behind every blog and paper figure to one JSON file, and check them against the registered results.

Usage: uv run --no-project --with numpy python -I scripts/v2/export_figure_data.py
Writes docs/blog/data/figures.json (numbers only, no text). Sources:
  fig1 (what replaces the refusal): results/v2/labels/phase3_test_grades.csv (Qwen), phase6B_llama_grades.csv (Llama); unsteered,
       404 held-out harmful pairs; counts per label and language with Wilson 95% intervals.
  fig2 (repairs): Qwen phase3_test_grades.csv (unsteered vs gated); Llama results/v2/phase7/stage2/stage2_results.json (judge labels,
       R3 vs unsteered). Paired bootstrap intervals for each change (10,000 resamples).
  fig3 (internal signatures): mean projection on each model's English-fitted refusal direction, last template token, on the same
       404 harmful and 169 harmless test pairs. Qwen: gate scores at layer 14 (results/v2/phase3/test/test_replies.jsonl, unsteered rows;
       untracked input, listed in MANIFEST.sha256). Llama: results/v2/phase6/stageB/stageB_results.json, layer 10.
The script fails if any exported headline differs from the registered result files.
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "results/v2/labels"
OUT = ROOT / "docs/blog/data/figures.json"
LABELS = ["REFUSE_EXPLICIT", "REFUSE_SOFT", "COMPLY", "BROKEN"]
RNG = np.random.default_rng(20261117)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def boot(d, B=10000):
    d = np.asarray(d, float)
    m = d[RNG.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def grades(path, cond_ok):
    out = {}
    for r in csv.DictReader(open(path)):
        if cond_ok(r):
            out.setdefault((r["set"], r["lang"], r["condition"]), {})[r["item_id"]] = r["label"]
    return out


def fig1():
    q = grades(LAB / "phase3_test_grades.csv", lambda r: r["set"] == "test_harmful" and r["condition"].endswith("unsteered"))
    l = grades(LAB / "phase6B_llama_grades.csv", lambda r: r["set"] == "test_harmful" and r["condition"].endswith("unsteered"))
    res = {}
    for model, g in (("qwen", q), ("llama", l)):
        en, hi = g[("test_harmful", "en", "en_unsteered")], g[("test_harmful", "hi", "hi_unsteered")]
        items = sorted(set(en) & set(hi))
        rows = []
        for lab in LABELS:
            ke = sum(en[i] == lab for i in items)
            kh = sum(hi[i] == lab for i in items)
            d = [float(hi[i] == lab) - float(en[i] == lab) for i in items]
            ci = boot(d)
            rows.append(dict(label=lab, en=dict(k=ke, pct=100 * ke / len(items), ci=wilson(ke, len(items))),
                             hi=dict(k=kh, pct=100 * kh / len(items), ci=wilson(kh, len(items))),
                             change=dict(pts=100 * float(np.mean(d)), ci=[100 * ci[0], 100 * ci[1]])))
        lost = Counter(hi[i] for i in items if en[i] == "REFUSE_EXPLICIT" and hi[i] != "REFUSE_EXPLICIT")
        res[model] = dict(n=len(items), rows=rows, lost_refusals=dict(lost), lost_n=sum(lost.values()))
    return res


def fig2():
    q = grades(LAB / "phase3_test_grades.csv", lambda r: r["set"] == "test_harmful")
    res = {}
    arms = {}
    for lang in ("en", "hi"):
        a, b = q[("test_harmful", lang, f"{lang}_unsteered")], q[("test_harmful", lang, f"{lang}_gated")]
        items = sorted(set(a) & set(b))
        ea = [float(a[i] == "REFUSE_EXPLICIT") for i in items]
        eb = [float(b[i] == "REFUSE_EXPLICIT") for i in items]
        d = [y - x for x, y in zip(ea, eb)]
        ci = boot(d)
        arms[lang] = dict(n=len(items), before=100 * np.mean(ea), after=100 * np.mean(eb), change=100 * float(np.mean(d)),
                          ci=[100 * ci[0], 100 * ci[1]], before_ci=wilson(int(sum(ea)), len(items)), after_ci=wilson(int(sum(eb)), len(items)))
        arms[lang]["_paired"] = (items, ea, eb)
    gi = sorted(set(arms["en"]["_paired"][0]) & set(arms["hi"]["_paired"][0]))
    E = {k: dict(zip(arms[k]["_paired"][0], zip(arms[k]["_paired"][1], arms[k]["_paired"][2]))) for k in ("en", "hi")}
    gd = [(E["en"][i][0] - E["hi"][i][0]) - (E["en"][i][1] - E["hi"][i][1]) for i in gi]
    gci = boot(gd)
    for k in arms:
        arms[k].pop("_paired")
    res["qwen"] = dict(arms=arms, gap_before=arms["en"]["before"] - arms["hi"]["before"], gap_after=arms["en"]["after"] - arms["hi"]["after"],
                       narrowing=100 * float(np.mean(gd)), narrowing_ci=[100 * gci[0], 100 * gci[1]], bar_explicit=15.0, bar_gap=None,
                       verdict="passed every registered check; gap barely narrows")
    s = json.load(open(ROOT / "results/v2/phase7/stage2/stage2_results.json"))
    la = {}
    for lang, g in (("en", "enharm"), ("hi", "hiharm")):
        e = s[g]["explicit"]
        n = s[g]["n"]
        la[lang] = dict(n=n, before=e["base"], after=e["arm"], change=e["diff"], ci=e["ci95"],
                        before_ci=wilson(round(e["base"] * n / 100), n), after_ci=wilson(round(e["arm"] * n / 100), n))
    res["llama"] = dict(arms=la, gap_before=-s["gap"]["gap_unsteered"], gap_after=-s["gap"]["gap_arm"], narrowing=s["gap"]["narrowing"],
                        narrowing_ci=s["gap"]["ci95"], bar_explicit=15.0, bar_gap=10.0,
                        verdict="narrowly missed both registered bars")
    return res


def fig3():
    res = {}
    q = {}
    for line in open(ROOT / "results/v2/phase3/test/test_replies.jsonl"):
        r = json.loads(line)
        if r["condition"].endswith("unsteered") and r["set"] in ("test_harmful", "test_harmless"):
            q[(r["set"], r["lang"], r["pair"])] = r["score"]
    p = json.load(open(ROOT / "results/v2/phase6/stageB/stageB_results.json"))["projections"]
    l = {(u.split("|")[0], u.split("|")[2], u.split("|")[1]): v["r"] for u, v in p.items() if u.startswith("test_")}
    for model, src, layer in (("qwen", q, 14), ("llama", l, 10)):
        cells = {}
        for st in ("test_harmful", "test_harmless"):
            pairs = sorted({k[2] for k in src if k[0] == st and (st, "en", k[2]) in src and (st, "hi", k[2]) in src})
            for lang in ("en", "hi"):
                v = np.array([src[(st, lang, pr)] for pr in pairs])
                cells[f"{st.split('_')[1]}_{lang}"] = dict(n=len(v), mean=float(v.mean()), ci=boot(v))
        hp = sorted({k[2] for k in src if k[0] == "test_harmful"})
        shift = [src[("test_harmful", "hi", pr)] - src[("test_harmful", "en", pr)] for pr in hp]
        res[model] = dict(layer=layer, cells=cells, paired_shift=float(np.mean(shift)), paired_shift_ci=boot(shift),
                          ratio=cells["harmful_hi"]["mean"] / cells["harmful_en"]["mean"])
    return res


def check(f1, f2, f3):
    """Every headline must match the registered result files."""
    s7 = json.load(open(ROOT / "results/v2/phase7/stage2/stage2_results.json"))
    s6 = json.load(open(ROOT / "results/v2/phase6/stageB/phase6_results.json"))
    q3 = json.load(open(ROOT / "results/v2/phase3/test/test_results.json"))["outcomes"]
    exp = lambda m, lang: next(r for r in f1[m]["rows"] if r["label"] == "REFUSE_EXPLICIT")[lang]["pct"]
    checks = [
        ("Qwen explicit EN = Phase 3 test", exp("qwen", "en"), 100 * q3["test_harmful_en"]["explicit"]["unsteered"]),
        ("Qwen explicit HI = Phase 3 test", exp("qwen", "hi"), 100 * q3["test_harmful_hi"]["explicit"]["unsteered"]),
        ("Llama explicit EN = Phase 6", exp("llama", "en"), 100 * s6["C1"]["explicit_en"]["rate"]),
        ("Llama explicit HI = Phase 6", exp("llama", "hi"), 100 * s6["C1"]["explicit_hi"]["rate"]),
        ("Qwen fix Hindi = Phase 3 test", f2["qwen"]["arms"]["hi"]["change"], 100 * q3["test_harmful_hi"]["explicit"]["diff"]),
        ("Qwen fix English = Phase 3 test", f2["qwen"]["arms"]["en"]["change"], 100 * q3["test_harmful_en"]["explicit"]["diff"]),
        ("Qwen Hindi comply = Phase 3 test", next(r for r in f1["qwen"]["rows"] if r["label"] == "COMPLY")["hi"]["pct"],
         100 * q3["test_harmful_hi"]["comply"]["unsteered"]),
        ("Llama fix Hindi = Phase 7", f2["llama"]["arms"]["hi"]["change"], s7["hiharm"]["explicit"]["diff"]),
        ("Llama gap narrowing = Phase 7", f2["llama"]["narrowing"], s7["gap"]["narrowing"]),
        ("Llama projection ratio = Phase 6 C2", f3["llama"]["ratio"], s6["C2"]["ratio"]),
    ]
    # absolute tolerance 0.05 (percentage points, or ratio units for the projection ratio): rounding only, no drift allowed
    bad = [(n, a, b) for n, a, b in checks if abs(a - b) > (0.005 if "ratio" in n else 0.05)]
    for n, a, b in checks:
        print(f"{'OK ' if (n, a, b) not in bad else 'BAD'} {n}: exported {a:.3f}, registered {b:.3f}")
    if bad:
        raise SystemExit("figure data disagrees with registered results")


def main():
    f1, f2, f3 = fig1(), fig2(), fig3()
    check(f1, f2, f3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = dict(fig1=f1, fig2=f2, fig3=f3)
    json.dump(data, open(OUT, "w"), indent=1)
    # the same numbers as a script, so the page also works when opened from disk (file:// cannot fetch JSON)
    OUT.with_suffix(".js").write_text("window.FIGURES = " + json.dumps(data) + ";\n")
    print("wrote", OUT.relative_to(ROOT), "and", OUT.with_suffix(".js").relative_to(ROOT))


if __name__ == "__main__":
    main()
