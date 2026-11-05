"""Step 3: semantic harm concept or language-dependent refusal pathway. See docs/prereg/2026-11-05_semantic_vs_pathway.md.

Stages:
  render   write data/rendered/s3_items.jsonl (dev MultiJail and XSTest in each language; the confirm split is never read)
  cache    cache activations for every item
  s1s3     the refusal-signal analysis (S1) and intent vs surface analysis (S3); needs no GPU
  harmful  S2 generation on harmful prompts, half B
  harmless S2 generation on harmless prompts, half B
  report   S2 analysis from judged replies

Replies are stored locally and never printed.

Usage: python scripts/step3.py {render,cache,s1s3,harmful,harmless,report}
"""

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

from common import ROOT, read_jsonl, write_jsonl

MODEL = os.environ.get("CS_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OUT = ROOT / os.environ.get("CS_STEP_DIR", "results/step3")
LANGS = ("en", "zh", "ar", "bn", "sw")
OTHERS = ("zh", "ar", "bn", "sw")
XS = {"bn": "data/rendered/xstest_v3_bn.jsonl", "zh": "data/rendered/xstest_s3_zh.jsonl",
      "ar": "data/rendered/xstest_s3_ar.jsonl", "sw": "data/rendered/xstest_s3_sw.jsonl"}
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
SEED = 20261105
HARM_LAYER, REFUSAL_LAYER, ROBUST_LAYER = (int(x) for x in os.environ.get("CS_LAYERS", "15,17,21").split(","))
N_RANDOM = 5


def splits():
    return json.loads((ROOT / "data/splits/splits.json").read_text())


def refusal_vector(layer):
    """English harmful minus harmless at the last template token, fit on half A."""
    rows = read_jsonl(ROOT / "data/rendered/s3_items.jsonl")
    data = np.load(OUT / "activations.npz")
    post = data["post"][:, list(data["layers"]).index(layer)]
    en_a = [r["lang"] == "en" and r["half"] == "a" for r in rows]
    pick = lambda ds, label: [i for i, r in enumerate(rows) if en_a[i] and r["dataset"] == ds and r["label"] == label]
    return post[pick("multijail", "harmful")].mean(0) - post[pick("xstest", "harmless")].mean(0)


def boot(values, rng, n=1000):
    values = np.asarray(values, float)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def render():
    s = splits()
    half = {t: "a" for t in s["multijail"]["dev_a"]} | {t: "b" for t in s["multijail"]["dev_b"]}
    xhalf = {t: k.split("_")[1] for k, ids in s["xstest"].items() for t in ids}
    rows = []
    for r in csv.DictReader(open(ROOT / "data/external/multijail/MultiJail.csv", encoding="utf-8-sig")):
        task = f"multijail_{int(r['id']):03d}"
        if task not in half:
            continue  # sealed confirm request
        for lang in LANGS:
            rows.append(dict(id=f"{task}__{lang}", task_id=task, dataset="multijail", label="harmful", lang=lang, half=half[task],
                             hidden_fact=r["en"].strip(), messages=[SYSTEM, {"role": "user", "content": r[lang].strip()}]))
    source = {}
    for r in csv.DictReader(open(ROOT / "data/external/xstest/xstest_prompts.csv")):
        task = f"xstest_{int(r['id']):03d}"
        source[task] = r
        rows.append(dict(id=f"{task}__en", task_id=task, dataset="xstest", label="harmful" if r["label"] == "unsafe" else "harmless",
                         lang="en", half=xhalf[task], category=r["type"], hidden_fact=r["prompt"],
                         messages=[SYSTEM, {"role": "user", "content": r["prompt"]}]))
    for lang, path in XS.items():
        if not (ROOT / path).exists():  # harmful-prompt stages run before the translations finish
            continue
        for it in read_jsonl(ROOT / path):
            if it["format_ok"]:
                rows.append(dict(id=f"{it['task_id']}__{lang}", task_id=it["task_id"], dataset="xstest",
                                 label="harmful" if it["label"] == "unsafe" else "harmless", lang=lang, half=xhalf[it["task_id"]],
                                 category=it["category"], hidden_fact=it["source_en"], messages=it["messages"]))
    write_jsonl(ROOT / "data/rendered/s3_items.jsonl", rows)
    print(f"{len(rows)} items", Counter((r["dataset"], r["lang"]) for r in rows))


def cache():
    import torch
    from common import load_model, prompt_text
    rows = read_jsonl(ROOT / "data/rendered/s3_items.jsonl")
    tok, model = load_model(MODEL)
    closers = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>")} - {tok.unk_token_id, None}
    for r in rows:
        r["ids"] = tok(prompt_text(tok, r), add_special_tokens=False)["input_ids"]
    keep = [HARM_LAYER, REFUSAL_LAYER, ROBUST_LAYER]
    width_d = model.config.hidden_size
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    inst, post = np.zeros((len(rows), 3, width_d), np.float32), np.zeros((len(rows), 3, width_d), np.float32)
    for start in range(0, len(order), 32):
        idx = order[start: start + 32]
        width = max(len(rows[i]["ids"]) for i in idx)
        batch = torch.full((len(idx), width), tok.pad_token_id)
        mask = torch.zeros_like(batch)
        for b, i in enumerate(idx):
            ids = rows[i]["ids"]
            batch[b, width - len(ids):] = torch.tensor(ids)
            mask[b, width - len(ids):] = 1
        with torch.no_grad():
            hs = model(batch.to(model.device), attention_mask=mask.to(model.device), output_hidden_states=True).hidden_states
        for b, i in enumerate(idx):
            ids = rows[i]["ids"]
            pos = [p for p, t in enumerate(ids) if t in closers][-1] - 1 + width - len(ids)  # last token of the user turn
            for k, L in enumerate(keep):
                inst[i, k] = hs[L][b, pos].float().cpu().numpy()
                post[i, k] = hs[L][b, -1].float().cpu().numpy()
        print(f"  {min(start + 32, len(order))}/{len(order)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "activations.npz", inst=inst, post=post, layers=keep)
    print(f"{len(rows)} prompts -> {OUT}")


def s1s3():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(SEED)
    rows = read_jsonl(ROOT / "data/rendered/s3_items.jsonl")
    data = np.load(OUT / "activations.npz")
    col = {k: np.array([r.get(k) for r in rows]) for k in ("dataset", "label", "lang", "half", "task_id")}
    y = (col["label"] == "harmful").astype(int)
    probe = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    sel = lambda ds, lang, half: (((col["dataset"] == ds) & (col["label"] == "harmful")) | ((col["dataset"] == "xstest") & (col["label"] == "harmless"))) & (col["lang"] == lang) & (col["half"] == half)
    index = {(r["dataset"], r["task_id"], r["lang"]): i for i, r in enumerate(rows)}
    tasks_b = sorted(set(col["task_id"][(col["dataset"] == "multijail") & (col["half"] == "b")]))
    out = dict(s1={}, s3={})

    def sd_sep(values, labels):
        a, b = values[labels == 1], values[labels == 0]
        return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))

    for k, tag in ((0, "primary"), (2, "robust_layer21")):
        harm_k, ref_k = k, (1 if k == 0 else 2)
        H, R = data["inst"][:, harm_k], data["post"][:, ref_k]
        d = refusal_vector(int(data["layers"][ref_k]))
        unit = d / np.linalg.norm(d)
        r_all = R @ unit
        fit = sel("multijail", "en", "a")
        p = probe().fit(H[fit], y[fit])
        h_all = p.decision_function(H)
        slope = float(np.polyfit(h_all[fit], r_all[fit], 1)[0])
        result = dict(slope=slope, languages={})
        en_sep = None
        for lang in LANGS:
            test = sel("multijail", lang, "b")
            sep_h, sep_r = sd_sep(h_all[test], y[test]), sd_sep(r_all[test], y[test])
            own_fit, own_test = sel("multijail", lang, "a"), test
            own = probe().fit(H[own_fit], y[own_fit]).decision_function(H[own_test])
            entry = dict(sep_harm=sep_h, sep_refusal=sep_r, within_probe_sep=sd_sep(own, y[own_test]))
            if lang == "en":
                en_sep = (sep_h, sep_r)
            else:
                ids_l = [index[("multijail", t, lang)] for t in tasks_b]
                ids_e = [index[("multijail", t, "en")] for t in tasks_b]
                raw = r_all[ids_l] - r_all[ids_e]
                predicted = slope * (h_all[ids_l] - h_all[ids_e])
                residual = raw - predicted
                entry.update(raw=float(raw.mean()), raw_ci=boot(raw, rng), predicted=float(predicted.mean()), predicted_ci=boot(predicted, rng),
                             residual=float(residual.mean()), residual_ci=boot(residual, rng),
                             ratio_harm=sep_h / en_sep[0], ratio_refusal=sep_r / en_sep[1])
                share = abs(entry["residual"]) / max(abs(entry["raw"]), 1e-9)
                if entry["raw_ci"][1] >= 0:
                    reading = "no deficit"
                elif entry["residual_ci"][1] < 0 and share >= 0.5:
                    reading = "pathway"
                elif entry["predicted_ci"][1] < 0 and share < 0.5:
                    reading = "recognition"
                else:
                    reading = "mixed"
                entry.update(residual_share=share, reading=reading,
                             pathway_robust=bool(reading == "pathway" and entry["ratio_harm"] >= 0.75 and entry["ratio_refusal"] < entry["ratio_harm"]))
            result["languages"][lang] = entry
        out["s1"][tag] = result

    H = data["inst"][:, 0]
    fit_en = sel("xstest", "en", "a")
    transfer = probe().fit(H[fit_en], y[fit_en])
    for lang in LANGS:
        tr, te = sel("xstest", lang, "a"), sel("xstest", lang, "b")
        if not te.any():
            continue
        s_probe = probe().fit(H[tr], y[tr]).decision_function(H[te])
        texts = [r["messages"][1]["content"] for r in rows]
        tr_i, te_i = np.where(tr)[0], np.where(te)[0]
        surface = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2), LogisticRegression(C=10, max_iter=2000))
        surface.fit([texts[i] for i in tr_i], y[tr])
        s_surface = surface.decision_function([texts[i] for i in te_i])
        s_transfer = transfer.decision_function(H[te])
        yt = y[te]
        diffs = []
        pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
        for _ in range(1000):
            i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            diffs.append(roc_auc_score(yt[i], s_probe[i]) - roc_auc_score(yt[i], s_surface[i]))
        out["s3"][lang] = dict(n=int(te.sum()), within_probe=float(roc_auc_score(yt, s_probe)), surface=float(roc_auc_score(yt, s_surface)),
                               transfer=float(roc_auc_score(yt, s_transfer)), diff_ci=[float(x) for x in np.percentile(diffs, [2.5, 97.5])])
    (OUT / "s1_s3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def directions(device):
    import torch
    rows = read_jsonl(ROOT / "data/rendered/s3_items.jsonl")
    post = np.load(OUT / "activations.npz")["post"][:, 1]  # the refusal layer
    d = refusal_vector(REFUSAL_LAYER)
    pick = lambda cat: [i for i, r in enumerate(rows) if r["lang"] == "en" and r["label"] == "harmless" and r.get("category") == cat]
    matched = post[pick("historical_events")].mean(0) - post[pick("figurative_language")].mean(0)
    matched = matched / np.linalg.norm(matched) * np.linalg.norm(d)
    gen = torch.Generator().manual_seed(SEED)
    out = {"refusal": torch.tensor(d, device=device), "matched": torch.tensor(matched, device=device)}
    for k in range(N_RANDOM):
        v = torch.randn(d.shape[0], generator=gen).to(device)
        out[f"random{k}"] = v / v.norm() * out["refusal"].norm()
    return out


def run_conditions(items, modes, out_path):
    import torch
    import patching
    patching.BATCH = 32
    patching.MODEL = MODEL
    run = patching.Runner()
    dirs = directions(run.model.device)
    out = []
    conditions = [("unchanged", None, None)]
    for name, v in dirs.items():
        if "add" in modes:
            conditions.append(("add", name, dict(kind="add", layer=REFUSAL_LAYER, vec=v)))
        if "remove" in modes:
            conditions.append(("remove", name, dict(kind="ablate", unit=v / v.norm())))
    for lang in LANGS:
        group = [it for it in items if it["lang"] == lang]
        for action, name, mode in conditions:
            replies = run.generate(group, 100, mode, sample=False)
            out += [dict(id=it["id"], task_id=it["task_id"], lang=lang, action=action, direction=name or "none",
                         condition=f"{action}/{name or 'none'}", response=r) for it, r in zip(group, replies)]
            torch.cuda.empty_cache()
        write_jsonl(out_path, out)
        print(f"  {lang}: {len(out)} replies", flush=True)


def harmful():
    items = [r for r in read_jsonl(ROOT / "data/rendered/s3_items.jsonl") if r["dataset"] == "multijail" and r["half"] == "b"]
    run_conditions(items, ("add", "remove"), OUT / "replies_harmful.jsonl")


def harmless():
    items = [r for r in read_jsonl(ROOT / "data/rendered/s3_items.jsonl") if r["dataset"] == "xstest" and r["label"] == "harmless" and r["half"] == "b"]
    run_conditions(items, ("add",), OUT / "replies_harmless.jsonl")


def repeated(text):
    words = text.split()
    return len(words) >= 10 and len(set(words)) / len(words) < 0.3


def latin(text):
    lat = len(re.findall("[A-Za-z]", text))
    letters = sum(c.isalpha() for c in text)
    return lat / max(1, letters)


def report():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(SEED)
    out = dict(kappa={}, harmful={}, harmless={})
    main = read_jsonl(OUT / "judged_refusal.jsonl")
    second = {(r["id"], r["condition"]): r["verdict"] for r in read_jsonl(OUT / "judged_refusal_llama.jsonl")}
    pairs = defaultdict(list)
    for r in main:
        k = (r["id"], r["condition"])
        if k in second:
            pairs["en" if r["lang"] == "en" else "non_en"].append((r["verdict"], second[k]))
    for k, v in pairs.items():
        out["kappa"][k] = float(cohen_kappa_score([a for a, _ in v], [b for _, b in v]))
    out["gate_s2a"] = all(v >= 0.6 for v in out["kappa"].values()) and len(out["kappa"]) == 2
    for setname in ("harmful", "harmless"):
        rows = [r for r in main if r["id"].startswith("multijail" if setname == "harmful" else "xstest")]
        table = defaultdict(lambda: defaultdict(dict))
        text = defaultdict(list)
        for r in rows:
            table[r["lang"]][r["condition"]][r["task_id"]] = r["verdict"] == "REFUSE"
            text[(r["lang"], r["condition"])].append(r["response"])
        for lang, conds in table.items():
            tasks = sorted(conds["unchanged/none"])
            vec = lambda c: np.array([conds[c][t] for t in tasks], float)
            base = vec("unchanged/none")
            entry = dict(n=len(tasks), unchanged=float(base.mean()))
            for action in ("remove", "add"):
                if f"{action}/refusal" not in conds:
                    continue
                target = vec(f"{action}/refusal")
                rand = np.mean([vec(f"{action}/random{k}") for k in range(N_RANDOM)], 0)
                matched = vec(f"{action}/matched")
                sign = -1 if action == "remove" else 1
                d_rand, d_matched = sign * (target - rand), sign * (target - matched)
                rate = {c: {"refusal": float(vec(c).mean()), "repeat": float(np.mean([repeated(x) for x in text[(lang, c)]])),
                            "latin": float(np.mean([latin(x) for x in text[(lang, c)]]))}
                        for c in (f"{action}/refusal", f"{action}/matched", f"{action}/random0", "unchanged/none")}
                entry[action] = dict(refusal=float(target.mean()), matched=float(matched.mean()), random=float(rand.mean()),
                                     beats_random=float(d_rand.mean()), beats_random_ci=boot(d_rand, rng),
                                     beats_matched=float(d_matched.mean()), beats_matched_ci=boot(d_matched, rng),
                                     works=bool(boot(d_rand, rng)[0] > 0 and boot(d_matched, rng)[0] > 0), detail=rate)
            out[setname][lang] = entry
    (OUT / "s2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("kappa", "gate_s2a")}))
    for setname in ("harmful", "harmless"):
        for lang, e in out[setname].items():
            print(setname, lang, round(e["unchanged"], 3), {a: (round(e[a]["refusal"], 3), round(e[a]["matched"], 3), round(e[a]["random"], 3), e[a]["works"]) for a in ("remove", "add") if a in e})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "cache", "s1s3", "harmful", "harmless", "report"])
    {"render": render, "cache": cache, "s1s3": s1s3, "harmful": harmful, "harmless": harmless, "report": report}[ap.parse_args().stage]()
