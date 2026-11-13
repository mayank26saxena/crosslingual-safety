"""Phase 2b: explicit-versus-soft refusal direction (docs/prereg/2026-11-13_v2_explicit_soft_direction.md).

Stages:
  prep  (local, standard library only): writes data/v2/es_dir_prompts.jsonl (not tracked) and prints counts.
  run PROMPTS OUT_DIR [--smoke]  (GPU; imports nothing from the repository): last-token activations for every prompt, direction fit and
        layer choice, read-outs, then steered greedy generation. Writes OUT_DIR/es_dir.npz, es_dir_results.json (numbers only) and
        steer_replies.jsonl (reply text, not tracked).
Prompt and reply text is never printed.
"""

import csv
import json
import random
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3-8B"
SEED = 20261113
N_LAYERS = 36
MAX_CAND = 28
MAX_NEW = 1024


def _jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def _labels(run_csv, key):
    out = {}
    for r in csv.DictReader(open(run_csv, encoding="utf-8-sig")):
        k = key[r["item"].strip()]
        out.setdefault(k["pair"], {})[k["lang"]] = r["label"].strip()
    return out


# ----------------------------------------------------------------------------------------------------------------- prep

def prep():
    root = Path(__file__).resolve().parents[2]
    d = root / "data/v2"
    rows = []

    def add(uid, set_, lang, pair, label, messages):
        rows.append(dict(uid=uid, set=set_, lang=lang, pair=pair, label=label, messages=messages))

    # Fit: English Aya-Spanish requests where both Spanish-set graders agree on REFUSE_EXPLICIT or REFUSE_SOFT.
    sk = json.load(open(root / "notes/grading/spanish_aya_pg/key.json"))
    runs = root / "notes/grading/spanish_aya_pg/kiro_grading/runs"
    a = _labels(runs / "claude-sonnet-5.5/r1.csv", sk)
    b = _labels(runs / "gpt-5.6-luna/r1.csv", sk)
    en = {r["item"]: r for r in _jl(d / "spanish_all_prompts_en.jsonl")}
    for pair in sorted(en):
        if pair.startswith("polyguard"):
            continue
        la, lb = a[pair]["en"], b[pair]["en"]
        if la == lb and la in ("REFUSE_EXPLICIT", "REFUSE_SOFT"):
            add(f"fit|{pair}|en", "fit", "en", pair, la, en[pair]["messages"])

    # Read-out: tier A harmful (both languages) and PolyGuard harmless (both languages), selected-grader labels.
    hk = json.load(open(root / "notes/grading/hindi_tierA_pg/key.json"))
    hl = _labels(root / "notes/grading/hindi_tierA_pg/kiro_grading/runs/claude-sonnet-5.5-cc/r1.csv", hk)
    meta = {r["item_id"]: r for r in _jl(d / "after_exclusions.jsonl")}
    tier = {l: {r["item"]: r for r in _jl(d / f"tierA_prompts_{l}.jsonl")} for l in ("hi", "en")}
    harmful = sorted(k for k in tier["hi"] if meta[k]["benchmark_label"] == "harmful")
    for k in harmful:
        for l in ("hi", "en"):
            add(f"tierA|{k}|{l}", "tierA", l, k, hl[k][l], tier[l][k]["messages"])
    pg = {l: {r["item"]: r for r in _jl(d / f"pg_harmless_prompts_{l}.jsonl")} for l in ("hi", "en")}
    for k in sorted(pg["hi"]):
        for l in ("hi", "en"):
            add(f"pg|{k}|{l}", "pg", l, k, hl[k][l], pg[l][k]["messages"])

    # Steering set and harmless control (Hindi).
    rng = random.Random(SEED)
    cand = [k for k in harmful if hl[k]["en"] == "REFUSE_EXPLICIT" and hl[k]["hi"] != "REFUSE_EXPLICIT"]
    steer = sorted(rng.sample(cand, 150))
    ctrl = sorted(rng.sample(sorted(pg["hi"]), 100))
    for k in steer:
        add(f"steer|{k}|hi", "steer", "hi", k, hl[k]["hi"], tier["hi"][k]["messages"])
    for k in ctrl:
        add(f"ctrl|{k}|hi", "ctrl", "hi", k, hl[k]["hi"], pg["hi"][k]["messages"])

    with open(d / "es_dir_prompts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dict(steer_ids=steer, ctrl_ids=ctrl, n_candidates=len(cand)), open(root / "results/v2/es_dir_sets.json", "w"), indent=1)
    from collections import Counter
    print("rows:", dict(Counter((r["set"], r["lang"]) for r in rows)))
    print("fit labels:", dict(Counter(r["label"] for r in rows if r["set"] == "fit")))
    print("steer candidates:", len(cand))


# ------------------------------------------------------------------------------------------------------------------ run

def auroc(pos, neg):
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):  # average ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(prompts_path, out_dir, smoke=False):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _jl(prompts_path)
    if smoke:
        keep, seen = [], {}
        for r in rows:
            seen[r["set"]] = seen.get(r["set"], 0) + 1
            if seen[r["set"]] <= 12:
                keep.append(r)
        rows = keep
    max_new = 16 if smoke else MAX_NEW
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    layers = model.model.layers
    assert len(layers) == N_LAYERS
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    sel = lambda f: np.array([i for i, r in enumerate(rows) if f(r)], dtype=int)

    def batches(ix, bs, budget):
        order = sorted(ix, key=lambda i: lengths[i])
        cur = []
        for i in order:
            if cur and (len(cur) + 1 > bs or (len(cur) + 1) * lengths[i] > budget):
                yield cur
                cur = []
            cur.append(int(i))
        if cur:
            yield cur

    # ---- pass 1: last-token residual after each layer
    H = {}
    with torch.no_grad():
        for bi in batches(range(len(rows)), 32, 12000):
            enc = tok([texts[i] for i in bi], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            cap = []
            hs = [m.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu())) for m in layers]
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos, logits_to_keep=1)
            for h in hs:
                h.remove()
            hb = torch.stack(cap, 1)
            for j, i in enumerate(bi):
                H[i] = hb[j]
            torch.cuda.empty_cache()
    H = torch.stack([H[i] for i in range(len(rows))]).numpy()  # [N, 36, d]
    print("activations", H.shape, flush=True)

    fit = sel(lambda r: r["set"] == "fit")
    fe = np.array([i for i in fit if rows[i]["label"] == "REFUSE_EXPLICIT"])
    fs = np.array([i for i in fit if rows[i]["label"] == "REFUSE_SOFT"])

    # ---- layer choice: 5-fold CV AUROC
    rng = np.random.default_rng(SEED)
    fold_e = rng.permutation(len(fe)) % 5
    fold_s = rng.permutation(len(fs)) % 5
    cv = []
    for l in range(MAX_CAND + 1):
        aucs = []
        for f in range(5):
            tre, tes = fe[fold_e != f], fe[fold_e == f]
            trs, tss = fs[fold_s != f], fs[fold_s == f]
            if len(tes) == 0 or len(tss) == 0:
                continue
            dvec = H[tre, l].mean(0) - H[trs, l].mean(0)
            dvec = dvec / np.linalg.norm(dvec)
            aucs.append(auroc(H[tes, l] @ dvec, H[tss, l] @ dvec))
        cv.append(float(np.mean(aucs)))
    lstar = int(np.argmax(cv))
    e_dir = H[fe].mean(0) - H[fs].mean(0)  # [36, d]
    ehat = e_dir / np.linalg.norm(e_dir, axis=-1, keepdims=True)
    proj = np.einsum("nld,ld->nl", H, ehat)
    print("layer", lstar, "cv auroc", round(cv[lstar], 3), flush=True)

    # split-half reliability
    pe, ps = rng.permutation(fe), rng.permutation(fs)
    dA = H[pe[: len(pe) // 2]].mean(0) - H[ps[: len(ps) // 2]].mean(0)
    dB = H[pe[len(pe) // 2:]].mean(0) - H[ps[len(ps) // 2:]].mean(0)
    split_half = ((dA * dB).sum(-1) / np.linalg.norm(dA, axis=-1) / np.linalg.norm(dB, axis=-1)).tolist()
    del H

    np.savez_compressed(out_dir / "es_dir.npz", uid=np.array([r["uid"] for r in rows]), proj=proj, e_dir=e_dir, cv=np.array(cv))
    p = proj[:, lstar]

    # ---- read-outs
    B = 2000
    res = dict(model=MODEL, lstar=lstar, cv_auroc=cv, valid=bool(cv[lstar] >= 0.75), split_half_cos=split_half,
               n_fit_explicit=int(len(fe)), n_fit_soft=int(len(fs)), e_norm_lstar=float(np.linalg.norm(e_dir[lstar])))
    fit_gap = float(p[fe].mean() - p[fs].mean())
    res["fit_gap"] = fit_gap
    ta = {(rows[i]["pair"], rows[i]["lang"]): i for i in sel(lambda r: r["set"] == "tierA")}
    pairs = sorted({k for k, _ in ta})
    for lang in ("en", "hi"):
        ex = [p[ta[(k, lang)]] for k in pairs if rows[ta[(k, lang)]]["label"] == "REFUSE_EXPLICIT"]
        so = [p[ta[(k, lang)]] for k in pairs if rows[ta[(k, lang)]]["label"] == "REFUSE_SOFT"]
        res[f"transfer_auroc_{lang}"] = auroc(ex, so)
        res[f"n_explicit_{lang}"], res[f"n_soft_{lang}"] = len(ex), len(so)
        res[f"mean_by_label_{lang}"] = {lab: float(np.mean([p[ta[(k, lang)]] for k in pairs if rows[ta[(k, lang)]]["label"] == lab]))
                                        for lab in ("COMPLY", "REFUSE_EXPLICIT", "REFUSE_SOFT", "BROKEN")
                                        if any(rows[ta[(k, lang)]]["label"] == lab for k in pairs)}
    diff = np.array([p[ta[(k, "hi")]] - p[ta[(k, "en")]] for k in pairs])
    bs = np.sort([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
    res["paired_shift"] = dict(mean=float(diff.mean()), ci95=[float(bs[50]), float(bs[1949])], fraction_of_fit_gap=float(diff.mean() / fit_gap),
                               n=len(diff))
    link = {}
    for lab in ("REFUSE_EXPLICIT", "REFUSE_SOFT", "COMPLY", "BROKEN"):
        v = [p[ta[(k, "hi")]] for k in pairs if rows[ta[(k, "en")]]["label"] == "REFUSE_EXPLICIT" and rows[ta[(k, "hi")]]["label"] == lab]
        link[lab] = dict(n=len(v), mean=float(np.mean(v)) if v else None)
    res["outcome_link_hi_given_en_explicit"] = link
    pgi = {(rows[i]["pair"], rows[i]["lang"]): i for i in sel(lambda r: r["set"] == "pg")}
    res["pg_mean"] = {l: float(np.mean([p[i] for (k, ll), i in pgi.items() if ll == l])) for l in ("en", "hi")}
    json.dump(res, open(out_dir / "es_dir_results.json", "w"), indent=1)
    print("readouts done", flush=True)

    # ---- steering: add alpha * e_dir[l*] at the input of layer l*+1 (= output of layer l*), all positions
    gen_ix = list(sel(lambda r: r["set"] in ("steer", "ctrl")))
    vec = torch.tensor(e_dir[lstar], dtype=torch.float32, device="cuda")
    target = layers[lstar + 1] if lstar + 1 < N_LAYERS else model.model.norm
    with open(out_dir / "steer_replies.jsonl", "w", encoding="utf-8") as f:
        for alpha in (0, 1, 2):
            hook = None
            if alpha:
                v = alpha * vec

                def pre(mod, args, kwargs, v=v):
                    h = args[0] if args else kwargs["hidden_states"]
                    h2 = (h.float() + v).to(h.dtype)
                    if args:
                        return (h2,) + tuple(args[1:]), kwargs
                    kwargs["hidden_states"] = h2
                    return args, kwargs

                hook = target.register_forward_pre_hook(pre, with_kwargs=True)
            done = 0
            for bi in batches(gen_ix, 32, 10 ** 9):
                enc = tok([texts[i] for i in bi], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                         pad_token_id=tok.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                for j, i in enumerate(bi):
                    ids = new[j].tolist()
                    n = len(ids)
                    while n and ids[n - 1] == tok.pad_token_id:
                        n -= 1
                    eos = tok.convert_tokens_to_ids("<|im_end|>")
                    hit = eos not in ids[:n]
                    text = tok.decode(ids[:n], skip_special_tokens=True)
                    f.write(json.dumps(dict(uid=rows[i]["uid"], set=rows[i]["set"], pair=rows[i]["pair"], alpha=alpha, lstar=lstar,
                                            n_new_tokens=n, hit_limit=hit, reply=text), ensure_ascii=False) + "\n")
                f.flush()
                done += len(bi)
                print(f"alpha {alpha}: {done}/{len(gen_ix)}", flush=True)
                del gen
                torch.cuda.empty_cache()
            if hook:
                hook.remove()
    print("DONE", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prep":
        prep()
    elif a[0] == "run":
        run(a[1], a[2], smoke="--smoke" in a)
