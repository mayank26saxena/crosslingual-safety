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


def fig_layers():
    """Where refusal lives: per-layer ablation effect and harmful/harmless separation, both models."""
    d = json.load(open(ROOT / "results/v2/direction/direction_results.json"))
    v = d["meta"]["validity"]
    q_removed = [(v["fit_harmful_s"] - x) / (v["fit_harmful_s"] - v["fit_harmless_s"]) for x in d["meta"]["sweep"]]
    pl = d["per_layer"]["hi"]
    esq = json.load(open(ROOT / "results/v2/es_dir/es_dir_results.json"))["cv_auroc"]
    sb = json.load(open(ROOT / "results/v2/phase6/stageB/stageB_results.json"))
    r = sb["refusal"]
    l_removed = [(r["heldout_harmful_s"] - x) / r["gap"] for x in r["sweep"]]
    res = dict(qwen=dict(layers=len(pl["rho"]), chosen=d["lstar"], ablation_removed=q_removed, auroc_en=pl["auroc_english"],
                         auroc_hi=pl["auroc_target"], rho=pl["rho"], es_cv_auroc=esq, es_layer=24, items="tier A 1,174 pairs; PolyGuard harmless"),
               llama=dict(layers=32, chosen=r["layer"], ablation_removed=l_removed, es_cv_auroc=sb["explicit_soft"]["cv_auroc"], es_layer=18,
                          items="404 harmful and 169 harmless test pairs"))
    lp = ROOT / "results/v2/layer_proj/layer_proj_summary.json"
    if lp.exists():
        z = json.load(open(lp))
        res["llama"].update(auroc_en=z["auroc_en"], auroc_hi=z["auroc_hi"], rho=z["rho"], ratio_harmful=z["ratio_harmful"])
    return res


def auroc(pos, neg):
    """Mann-Whitney AUROC, ties counted half."""
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def fig_gate():
    """The gate: per-prompt refusal-direction scores on the test items, with each model's threshold."""
    res = {}
    q = [json.loads(l) for l in open(ROOT / "results/v2/phase3/test/test_replies.jsonl")]
    l = [json.loads(x) for x in open(ROOT / "results/v2/phase7/stage2/stage2_replies.jsonl")]
    for model, rows, tau, which in (("qwen", q, 5.515399932861328, "English-set threshold (median of English development harmful)"),
                                    ("llama", l, None, "Hindi-set threshold (median of Hindi development harmful)")):
        un = [r for r in rows if r["condition"].endswith("unsteered") and r["set"] in ("test_harmful", "test_harmless")]
        if tau is None:
            tau = next(r["tau"] for r in rows if r.get("tau") is not None and not r["condition"].endswith("unsteered"))
        cells = {}
        for st in ("test_harmful", "test_harmless"):
            for lang in ("en", "hi"):
                v = [r["score"] for r in un if r["set"] == st and r["lang"] == lang]
                cells[f"{st.split('_')[1]}_{lang}"] = dict(scores=[round(x, 3) for x in v], open=sum(x > tau for x in v) / len(v), n=len(v))
        # same-item comparison across models: chosen-layer AUROC per language and the Hindi/English harmful-harmless gap ratio,
        # all on the 404 harmful and 169 harmless test pairs (full-precision scores)
        sc = {(st, lang): [r["score"] for r in un if r["set"] == st and r["lang"] == lang]
              for st in ("test_harmful", "test_harmless") for lang in ("en", "hi")}
        mean = lambda v: sum(v) / len(v)
        gap = {lang: mean(sc[("test_harmful", lang)]) - mean(sc[("test_harmless", lang)]) for lang in ("en", "hi")}
        test_stats = dict(auroc_en=auroc(sc[("test_harmful", "en")], sc[("test_harmless", "en")]),
                          auroc_hi=auroc(sc[("test_harmful", "hi")], sc[("test_harmless", "hi")]), gap_ratio=gap["hi"] / gap["en"],
                          harmful_ratio=mean(sc[("test_harmful", "hi")]) / mean(sc[("test_harmful", "en")]))
        res[model] = dict(tau=tau, threshold=which, cells=cells, layer=14 if model == "qwen" else 10, test_stats=test_stats)
    return res


def fig_dose():
    """How hard to push: development sweeps, gain in Hindi explicit refusal against broken output, with the guards."""
    t = json.load(open(ROOT / "results/v2/phase3/tune/tune_selection.json"))
    q = [dict(label=f"α {g['alpha']}, τ {g['tau_percentile']}th", alpha=g["alpha"], tau_pct=g["tau_percentile"],
              explicit=100 * g["d_explicit_harmful"], broken=100 * g["d_broken_harmful"], harmless=100 * g["d_broad_harmless"],
              gate_harmful=g["gate_rate"]["dev_harmful"], gate_harmless=g["gate_rate"]["dev_harmless"], qualifies=g["qualifies"],
              chosen=(g["alpha"] == t["chosen"]["alpha"] and g["tau_percentile"] == t["chosen"]["tau_percentile"]) if isinstance(t["chosen"], dict) else False)
         for g in t["grid"]]
    s1 = json.load(open(ROOT / "results/v2/phase7/stage1/stage1_results.json"))
    names = {"E1": "explicit–soft 1.25×", "E2": "explicit–soft 2.5×", "R1": "refusal dir. 0.5×", "R2": "refusal dir. 1.0×", "R3": "refusal dir. 1.5×"}
    l = [dict(label=names[a], arm=a, explicit=v["hiharm"]["explicit"]["diff"], broken=v["hiharm"]["broken"]["diff"],
              harmless=v["hiharmless"]["broad"]["diff"], qualifies=v["qualifies"], chosen=(a == s1.get("stage1_choice")))
         for a, v in s1["arms"].items()]
    return dict(qwen=dict(points=q, n=401, guards=dict(broken=5, harmless=3, gain=None)),
                llama=dict(points=l, n=150, guards=dict(broken=5, harmless=3, gain=15)))


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


def severity():
    """How often each model simply complies with a harmful request, in each language, and what the repair did (judge labels, 404 test pairs)."""
    s7 = json.load(open(ROOT / "results/v2/phase7/stage2/stage2_results.json"))
    s6 = json.load(open(ROOT / "results/v2/phase6/stageB/phase6_results.json"))["C1"]
    q3 = json.load(open(ROOT / "results/v2/phase3/test/test_results.json"))["outcomes"]
    llama = dict(en=s7["enharm"]["comply"]["base"], hi=s7["hiharm"]["comply"]["base"], hi_fixed=s7["hiharm"]["comply"]["arm"])
    qwen = dict(en=100 * q3["test_harmful_en"]["comply"]["unsteered"], hi=100 * q3["test_harmful_hi"]["comply"]["unsteered"],
                hi_fixed=100 * q3["test_harmful_hi"]["comply"]["gated"])
    # cross-check Llama's unsteered rates against the Phase 6 replication (same items; tolerance covers a one-item relabel)
    for k, ref in (("hi", 100 * s6["comply_hi"]["rate"]), ("en", 100 * s6["comply_en"]["rate"])):
        ok = abs(llama[k] - ref) <= 0.5
        print(f"{'OK ' if ok else 'BAD'} Llama {k} compliance: Phase 7 base {llama[k]:.2f} vs Phase 6 {ref:.2f}")
        if not ok:
            raise SystemExit("severity numbers disagree across phases")
    return dict(llama=llama, qwen=qwen)


def guards():
    """General-ability check (Global-MMLU, 300 questions per language) before and after each repair, asserted unchanged as the post says."""
    q3 = json.load(open(ROOT / "results/v2/phase3/test/test_results.json"))
    s7 = json.load(open(ROOT / "results/v2/phase7/stage2/stage2_results.json"))
    qa = q3["outcomes"]
    out = dict(qwen=dict(hi=100 * qa["mmlu_hi"]["accuracy"]["unsteered"], en=100 * qa["mmlu_en"]["accuracy"]["unsteered"],
                         hi_after=100 * qa["mmlu_hi"]["accuracy"]["gated"], en_after=100 * qa["mmlu_en"]["accuracy"]["gated"], n=qa["mmlu_hi"]["accuracy"]["n"],
                         gate_on_mmlu=max(q3["gate_rate"]["mmlu_hi"], q3["gate_rate"]["mmlu_en"])),
               llama=dict(hi=s7["mmlu"]["hi"]["unsteered"], en=s7["mmlu"]["en"]["unsteered"], hi_after=s7["mmlu"]["hi"]["R3"], en_after=s7["mmlu"]["en"]["R3"]))
    for m, d in out.items():
        ok = abs(d["hi"] - d["hi_after"]) < 0.05 and abs(d["en"] - d["en_after"]) < 0.05
        print(f"{'OK ' if ok else 'BAD'} {m} Global-MMLU unchanged: Hindi {d['hi']:.1f} -> {d['hi_after']:.1f}, English {d['en']:.1f} -> {d['en_after']:.1f}")
        if not ok:
            raise SystemExit("Global-MMLU changed; the post says it did not")
    ok = out["qwen"]["gate_on_mmlu"] == 0
    print(f"{'OK ' if ok else 'BAD'} Qwen gate never opened on a Global-MMLU question (text claim)")
    if not ok:
        raise SystemExit("the gate opened on MMLU; revise the post")
    return out


def spanish():
    """Appendix C: the Qwen Spanish control, computed from the id-only label table and cross-checked against the provisional result file."""
    import csv
    G = "claude-sonnet-5.5 (kiro)"
    lab = {(r["item_id"], r["lang"]): r["label"] for r in csv.DictReader(open(ROOT / "results/v2/labels/spanish_grades.csv")) if r["grader"] == G}
    ref = json.load(open(ROOT / "results/v2/provisional/spanish_paired.json"))[G]
    out = {}
    for name, prefix, key in (("harmful", "aya_spa__", "harmful"), ("harmless", "polyguard__", "harmless")):
        ids = sorted({i for i, _ in lab if i.startswith(prefix)})
        assert ids and all((i, "en") in lab and (i, "es") in lab for i in ids)
        rate = lambda lang, pred: 100 * sum(pred(lab[(i, lang)]) for i in ids) / len(ids)
        explicit = lambda l: l == "REFUSE_EXPLICIT"
        anyref = lambda l: l in ("REFUSE_EXPLICIT", "REFUSE_SOFT")
        blk = dict(n=len(ids), explicit=dict(en=rate("en", explicit), es=rate("es", explicit)), any=dict(en=rate("en", anyref), es=rate("es", anyref)),
                   comply=dict(en=rate("en", lambda l: l == "COMPLY"), es=rate("es", lambda l: l == "COMPLY")))
        for k, pred in (("explicit", explicit), ("any", anyref)):
            blk[k]["en_only"] = sum(pred(lab[(i, "en")]) and not pred(lab[(i, "es")]) for i in ids)
            blk[k]["es_only"] = sum(pred(lab[(i, "es")]) and not pred(lab[(i, "en")]) for i in ids)
            blk[k]["p"] = ref[key][k]["mcnemar_p"]
            for lang in ("en", "es"):
                ok = abs(blk[k][lang] / 100 - ref[key][k][f"rate_{lang}"]) < 1e-4
                print(f"{'OK ' if ok else 'BAD'} Spanish {name} {k} {lang}: {blk[k][lang]:.2f} vs {100 * ref[key][k]['rate_' + lang]:.2f}")
                if not ok:
                    raise SystemExit("Spanish numbers disagree with the provisional result file")
            if (blk[k]["en_only"], blk[k]["es_only"]) != (ref[key][k]["en_only"], ref[key][k]["es_only"]):
                raise SystemExit("Spanish paired counts disagree with the provisional result file")
        out[name] = blk
    return out


def check_gate(data):
    """Same-item gate statistics must agree with the independent sources: Figure 3 harmful ratios, and for Llama the per-layer run."""
    g, f3, lay = data["gate"], data["fig3"], data["layers"]
    checks = [(f"{m} harmful ratio = Figure 3", g[m]["test_stats"]["harmful_ratio"], f3[m]["ratio"]) for m in ("qwen", "llama")]
    if lay["llama"].get("auroc_hi"):
        c = lay["llama"]["chosen"]
        checks += [("llama AUROC EN = per-layer run", g["llama"]["test_stats"]["auroc_en"], lay["llama"]["auroc_en"][c]),
                   ("llama AUROC HI = per-layer run", g["llama"]["test_stats"]["auroc_hi"], lay["llama"]["auroc_hi"][c]),
                   ("llama gap ratio = per-layer run", g["llama"]["test_stats"]["gap_ratio"], lay["llama"]["rho"][c])]
    bad = [x for x in checks if abs(x[1] - x[2]) > 0.01]
    for n, a, b in checks:
        print(f"{'BAD' if (n, a, b) in bad else 'OK '} {n}: {a:.3f} vs {b:.3f}")
    # blog wording: the Hindi gap shrinks "all" from the harmless side in Qwen and "mostly" from the harmful side in Llama
    for m in ("qwen", "llama"):
        c = {k: v["mean"] for k, v in f3[m]["cells"].items()}
        shrink = (c["harmful_en"] - c["harmless_en"]) - (c["harmful_hi"] - c["harmless_hi"])
        share = (c["harmful_en"] - c["harmful_hi"]) / shrink
        ok = share < 0.05 if m == "qwen" else share > 0.6
        print(f"{'OK ' if ok else 'BAD'} {m} share of Hindi gap shrink from the harmful side: {share:.2f}")
        if not ok:
            bad.append((m, share, None))
    lh = g["llama"]["cells"]["harmless_hi"]
    ok = lh["open"] == 0 and lh["n"] == 169
    print(f"{'OK ' if ok else 'BAD'} llama gate opens on none of the 169 Hindi harmless test prompts (text claim)")
    if not ok:
        bad.append(("llama harmless open", lh["open"], 0))
    if bad:
        raise SystemExit("gate statistics disagree with independent sources")


def main():
    f1, f2, f3 = fig1(), fig2(), fig3()
    check(f1, f2, f3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = dict(fig1=f1, fig2=f2, fig3=f3, layers=fig_layers(), gate=fig_gate(), dose=fig_dose(), severity=severity(), spanish=spanish(), guards=guards())
    check_gate(data)
    json.dump(data, open(OUT, "w"), indent=1)
    # the same numbers as a script, so the page also works when opened from disk (file:// cannot fetch JSON)
    OUT.with_suffix(".js").write_text("window.FIGURES = " + json.dumps(data) + ";\n")
    print("wrote", OUT.relative_to(ROOT), "and", OUT.with_suffix(".js").relative_to(ROOT))


if __name__ == "__main__":
    main()
