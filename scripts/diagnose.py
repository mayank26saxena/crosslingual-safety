"""Step 2 diagnostics that need no generation: D1 probe transfer, D2 refusal projection, D3 relation to behaviour.

Rules and contrasts are defined in docs/prereg/2026-11-04_evaluation_and_diagnostics.md. Reads cached activations and
writes aggregates only, to results/step2/diagnostics.json.

Usage: python scripts/diagnose.py
"""

import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ROOT, read_jsonl

OUT = ROOT / os.environ.get("CS_STEP_DIR", "results/step2")
LAYERS = list(range(1, 29))
SELECT_LAYERS = list(range(10, 23))
N_BOOT = 1000


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))


def boot_auc(y, s, rng):
    vals = []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(N_BOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals.append(roc_auc_score(y[i], s[i]))
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def main():
    rng = np.random.default_rng(20261104)
    rows = read_jsonl(OUT / "activation_rows.jsonl")
    data = np.load(OUT / "activations.npz")
    feats = {"t_inst": data["t_inst"].astype(np.float32), "t_post": data["t_post"][:, :, -1].astype(np.float32)}
    col = {k: np.array([r.get(k) for r in rows]) for k in ("dataset", "label", "lang", "half", "task_id")}
    y_all = (col["label"] == "harmful").astype(int)

    def sel(dataset_harm, lang, half):
        """Harmful rows from dataset_harm plus XSTest harmless rows, for one language and half."""
        harm = (col["dataset"] == dataset_harm) & (col["label"] == "harmful")
        harmless = (col["dataset"] == "xstest") & (col["label"] == "harmless")
        return (harm | harmless) & (col["lang"] == lang) & (col["half"] == half)

    results = {"n": {f"{d}/{la}/{l}/{h}": int(((col["dataset"] == d) & (col["lang"] == la) & (col["label"] == l)
                                               & (col["half"] == h)).sum())
                     for d in ("multijail", "xstest") for la in ("en", "bn", "sw") for l in ("harmful", "harmless")
                     for h in ("a", "b")}}

    # D1: per-layer English-trained probes, scored on Bengali, for the primary and same-source contrasts
    curves = {}
    scores = {}
    for contrast, ds in (("primary", "multijail"), ("same_source", "xstest")):
        for pos, X in feats.items():
            for test_half, train_half in (("b", "a"), ("a", "b")):
                tr, te_bn, te_en = sel(ds, "en", train_half), sel(ds, "bn", test_half), sel(ds, "en", test_half)
                te_en = te_en & np.isin(col["task_id"], col["task_id"][te_bn])  # same items in both languages
                for L in LAYERS:
                    p = probe().fit(X[tr, L], y_all[tr])
                    sb, se = p.decision_function(X[te_bn, L]), p.decision_function(X[te_en, L])
                    key = (contrast, pos, L)
                    entry = curves.setdefault(key, {"bn": [], "en": [], "acc": []})
                    entry["bn"].append(roc_auc_score(y_all[te_bn], sb))
                    entry["en"].append(roc_auc_score(y_all[te_en], se))
                    entry["acc"].append(balanced_accuracy_score(y_all[te_bn], sb > 0))
                    scores[(contrast, pos, L, test_half)] = (y_all[te_bn], sb, y_all[te_en], se, np.where(te_bn)[0])
    results["d1_curves"] = {f"{c}/{p}/L{L}": {k: round(float(np.mean(v)), 3) for k, v in e.items()}
                            for (c, p, L), e in curves.items()}

    # Ceiling and null at the primary contrast
    results["d1_controls"] = {}
    for pos, X in feats.items():
        for L in (12, 15, 18, 21):
            m = sel("multijail", "bn", "a") | sel("multijail", "bn", "b")
            cv = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[m, L], y_all[m]):
                p = probe().fit(X[m, L][tr], y_all[m][tr])
                cv.append(roc_auc_score(y_all[m][te], p.decision_function(X[m, L][te])))
            tr, te = sel("multijail", "en", "a"), sel("multijail", "bn", "b")
            shuffled = rng.permutation(y_all[tr])
            null = roc_auc_score(y_all[te], probe().fit(X[tr, L], shuffled).decision_function(X[te, L]))
            results["d1_controls"][f"{pos}/L{L}"] = dict(bengali_cv_auc=round(float(np.mean(cv)), 3),
                                                         shuffled_label_auc=round(float(null), 3))

    # D1 verdict: layer and position selected on one Bengali half, evaluated on the other, then swapped
    verdict = {}
    for contrast in ("primary", "same_source"):
        folds = []
        for eval_half, select_half in (("b", "a"), ("a", "b")):
            options = [(pos, L) for pos in feats for L in SELECT_LAYERS]
            best = max(options, key=lambda o: roc_auc_score(*scores[(contrast, o[0], o[1], select_half)][:2]))
            y, s, ye, se, _ = scores[(contrast, best[0], best[1], eval_half)]
            folds.append(dict(selected=f"{best[0]}/L{best[1]}", auc_bn=roc_auc_score(y, s), ci_bn=boot_auc(y, s, rng),
                              auc_en=roc_auc_score(ye, se)))
        verdict[contrast] = dict(folds=folds, auc_bn=float(np.mean([f["auc_bn"] for f in folds])),
                                 ci_low=float(np.mean([f["ci_bn"][0] for f in folds])),
                                 ci_high=float(np.mean([f["ci_bn"][1] for f in folds])),
                                 auc_en=float(np.mean([f["auc_en"] for f in folds])))
    p, s = verdict["primary"], verdict["same_source"]
    verdict["go"] = bool(p["ci_low"] >= 0.75 and s["ci_low"] > 0.60 and s["auc_en"] - s["auc_bn"] <= 0.15)
    results["d1_verdict"] = verdict

    # D2: paired projection onto the English refusal direction, fit on half A, read on half B
    X = feats["t_post"]
    d2 = {}
    task_rows = defaultdict(dict)
    for i, r in enumerate(rows):
        if r["dataset"] == "multijail":
            task_rows[r["task_id"]][r["lang"]] = i
    dev_b = sorted(t for t, v in task_rows.items() if rows[v["en"]]["half"] == "b")
    fit_h = (col["dataset"] == "multijail") & (col["lang"] == "en") & (col["half"] == "a")
    fit_s = (col["dataset"] == "xstest") & (col["label"] == "harmless") & (col["lang"] == "en") & (col["half"] == "a")
    for L in LAYERS:
        direction = X[fit_h, L].mean(0) - X[fit_s, L].mean(0)
        unit = direction / np.linalg.norm(direction)
        rand = rng.standard_normal((5, X.shape[-1]))
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        entry = {}
        for lang in ("bn", "sw"):
            diff = np.array([X[task_rows[t][lang], L] - X[task_rows[t]["en"], L] for t in dev_b])
            proj = diff @ unit
            boots = [proj[rng.integers(0, len(proj), len(proj))].mean() for _ in range(N_BOOT)]
            entry[lang] = dict(mean=round(float(proj.mean()), 3),
                               ci=[round(float(x), 3) for x in np.percentile(boots, [2.5, 97.5])],
                               random_range=[round(float(x), 3) for x in
                                             (lambda m: (m.min(), m.max()))((diff @ rand.T).mean(0))])
            mask_h = (col["dataset"] == "multijail") & (col["lang"] == lang)
            own = X[mask_h, L].mean(0)
            harmless = (col["dataset"] == "xstest") & (col["label"] == "harmless") & (col["lang"] == lang)
            if harmless.any():
                other = own - X[harmless, L].mean(0)
                entry[lang]["cosine_with_english"] = round(float(other @ unit / np.linalg.norm(other)), 3)
        d2[f"L{L}"] = entry
    results["d2"] = d2
    np.savez(OUT / "refusal_directions.npz", **{f"L{L}": X[fit_h, L].mean(0) - X[fit_s, L].mean(0) for L in LAYERS})

    # D3: per-request unsafe rate against probe score and refusal projection, on half B Bengali
    judged = read_jsonl(ROOT / "results/earlier_neuron_steering/qwen2.5-7b-instruct/multijail/judged_multijail_gap.jsonl")
    unsafe = defaultdict(lambda: defaultdict(list))
    for r in judged:
        unsafe[r["id"].split("__")[-1]][r["task_id"]].append(r["verdict"] == "BAD")
    d3 = {}
    for L in (12, 15, 18, 21):
        direction = X[fit_h, L].mean(0) - X[fit_s, L].mean(0)
        unit = direction / np.linalg.norm(direction)
        p = probe().fit(feats["t_inst"][sel("multijail", "en", "a"), L], y_all[sel("multijail", "en", "a")])
        rate = np.array([np.mean(unsafe["bn"][t]) for t in dev_b])
        ids_bn = np.array([task_rows[t]["bn"] for t in dev_b])
        ids_en = np.array([task_rows[t]["en"] for t in dev_b])
        harm_bn = p.decision_function(feats["t_inst"][ids_bn, L])
        harm_en = p.decision_function(feats["t_inst"][ids_en, L])
        ref_bn, ref_en = X[ids_bn, L] @ unit, X[ids_en, L] @ unit
        gap = rate > np.array([np.mean(unsafe["en"][t]) for t in dev_b])
        d3[f"L{L}"] = dict(spearman_unsafe_vs_probe=round(float(spearmanr(rate, harm_bn)[0]), 3),
                           spearman_unsafe_vs_refusal=round(float(spearmanr(rate, ref_bn)[0]), 3),
                           n_gap_requests=int(gap.sum()),
                           gap_probe_bn_minus_en=round(float((harm_bn - harm_en)[gap].mean()), 3),
                           gap_refusal_bn_minus_en=round(float((ref_bn - ref_en)[gap].mean()), 3))
    results["d3"] = d3

    (OUT / "diagnostics.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results["d1_verdict"], indent=1))
    print(json.dumps(results["d3"], indent=1))
    for L in (12, 14, 15, 16, 17, 18, 19, 20):
        print(L, results["d2"][f"L{L}"]["bn"])


if __name__ == "__main__":
    main()
