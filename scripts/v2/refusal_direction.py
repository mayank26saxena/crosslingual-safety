"""Phase 2a: refusal-direction diagnosis (docs/prereg/2026-11-12_v2_refusal_direction.md).

Stages:
  prep  (local, standard library only): writes data/v2/direction_prompts.jsonl and data/v2/direction_labels.json.
  run PROMPTS LABELS OUT_DIR [--batch B]  (GPU; imports nothing from the repository, so it runs from a single copied file):
        fits the English refusal direction, picks the layer, projects every prompt, and writes OUT_DIR/direction.npz and
        OUT_DIR/direction_results.json (numbers only).
Prompt and reply text is never printed.
"""

import csv
import json
import random
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"  # Hugging Face snapshot used for v2 runs from 2026-11-13; see REPRODUCE.md
SEED = 20261112
N_LAYERS = 36
MAX_CAND = 28  # candidate layers 0..28, the first 80% of 36


# ----------------------------------------------------------------------------------------------------------------- prep

def _jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def _runs(paths, key):
    """labels[pair][lang] -> list of broad labels (True refused, False not, None declined) across runs."""
    out = {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            k = key[r["item"].strip()]
            lab = r["label"].strip().upper()
            v = None if lab == "DECLINED" else lab.startswith("REFUSE")
            out.setdefault(k["pair"], {}).setdefault(k["lang"], []).append(v)
    return out


def prep():
    root = Path(__file__).resolve().parents[2]
    d = root / "data/v2"
    rows = []

    def add(uid, set_, lang, kind, pair, messages, role):
        rows.append(dict(uid=uid, set=set_, lang=lang, kind=kind, pair=pair, role=role, messages=messages))

    # Aya Spanish: seeded half for fitting (English only), other half for evaluation (both languages).
    es = {r["item"]: r for r in _jl(d / "spanish_prompts_es.jsonl")}
    en = {r["item"]: r for r in _jl(d / "spanish_prompts_en.jsonl")}
    ids = sorted(es)
    random.Random(SEED).shuffle(ids)
    fit_ids, ho_ids = sorted(ids[:len(ids) // 2]), sorted(ids[len(ids) // 2:])
    for k in fit_ids:
        add(f"fit_harm|{k}|en", "fit_harm", "en", "harmful", k, en[k]["messages"], "fit")
    for k in ho_ids:
        for lang, src in (("en", en), ("es", es)):
            add(f"aya_es|{k}|{lang}", "aya_es", lang, "harmful", k, src[k]["messages"], "eval")

    # PolyGuard harmless: the 190 unused non-adversarial English prompts for fitting; the 300 control prompts for evaluation.
    pg = {l: {r["item"]: r for r in _jl(d / f"pg_harmless_prompts_{l}.jsonl")} for l in ("en", "es", "hi")}
    template = next(iter(pg["en"].values()))["messages"]
    unused = [x for x in _jl(d / "after_exclusions.jsonl") if x["source"] == "polyguard" and x["benchmark_label"] == "harmless"
              and not x["extra"].get("adversarial") and x["item_id"] not in pg["en"]]
    assert len(unused) == 190, len(unused)
    for x in unused:
        msgs = [m if m["role"] != "user" else dict(m, content=x["en"]) for m in template]
        add(f"fit_harmless|{x['item_id']}|en", "fit_harmless", "en", "harmless", x["item_id"], msgs, "fit")
    for lang in ("en", "es", "hi"):
        for k, r in pg[lang].items():
            add(f"pg|{k}|{lang}", "pg", lang, "harmless", k, r["messages"], "eval")

    # Hindi tier A (harmful).
    for lang in ("hi", "en"):
        for r in _jl(d / f"tierA_prompts_{lang}.jsonl"):
            add(f"tierA|{r['item']}|{lang}", "tierA", lang, "harmful", r["item"], r["messages"], "eval")

    # Hindi calibration (benchmark label from the calibration key).
    ck = json.load(open(root / "notes/hindi_calibration/key.json"))
    for lang, f in (("hi", "regen_prompts.jsonl"), ("en", "regen_prompts_en.jsonl")):
        for r in _jl(root / "notes/hindi_calibration" / f):
            add(f"calib|{r['item']}|{lang}", "calib", lang, ck[r["item"]]["label"], r["item"], r["messages"], "eval")

    with open(d / "direction_prompts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Labels. Refusal tokens: English replies of the Spanish set, broad label where Sonnet and Luna agree.
    sk = json.load(open(root / "notes/grading/spanish_aya_pg/key.json"))
    sruns = root / "notes/grading/spanish_aya_pg/kiro_grading/runs"
    slab = _runs([sruns / "claude-sonnet-5.5/r1.csv", sruns / "gpt-5.6-luna/r1.csv"], sk)
    en_replies = {r["item"]: r["reply"] for p in ("results/v2/spanish/en_replies.jsonl", "results/v2/pg_harmless/pg_en.jsonl")
                  for r in _jl(root / p)}
    heads = []
    for pair, v in slab.items():
        e = v["en"]
        if e[0] is not None and e[0] == e[1]:
            heads.append(dict(head=en_replies[pair][:60], refused=e[0]))

    def agree(v):
        v = [x for x in v if x is not None]
        return v[0] if v and all(x == v[0] for x in v) and len(v) == 2 else None

    es_beh = {p: {l: agree(v[l]) for l in ("en", "es")} for p, v in slab.items() if p in set(ho_ids)}

    hk = json.load(open(root / "notes/hindi_calibration/model_grading_key_1024.json"))
    hr = root / "notes/hindi_calibration/model_runs"
    hlab = _runs([hr / m / "r1.csv" for m in ("claude-haiku-5.5", "claude-sonnet-5.5", "gpt-5.6-terra", "gpt-5.6-luna")], hk)

    def majority(v):
        v = [x for x in v if x is not None]
        t = sum(v)
        return True if t >= 3 else False if len(v) - t >= 3 else None

    hi_beh = {p: {l: majority(v[l]) for l in ("en", "hi")} for p, v in hlab.items()}
    json.dump(dict(refusal_heads=heads, behaviour=dict(es=es_beh, hi=hi_beh)), open(d / "direction_labels.json", "w"), ensure_ascii=False)
    from collections import Counter
    print("prompts:", dict(Counter((r["set"], r["lang"]) for r in rows)))
    print("refusal heads:", len(heads), "refused", sum(h["refused"] for h in heads))
    print("behaviour pairs: es", len(es_beh), "hi", len(hi_beh))


# ------------------------------------------------------------------------------------------------------------------ run

def run(prompts_path, labels_path, out_dir, batch=32):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _jl(prompts_path)
    labels = json.load(open(labels_path))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).to("cuda").eval()
    layers = model.model.layers
    assert len(layers) == N_LAYERS
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    uid = [r["uid"] for r in rows]
    idx = {u: i for i, u in enumerate(uid)}
    sel = lambda f: np.array([i for i, r in enumerate(rows) if f(r)])

    # ---- refusal token sets (Phase 2 rule), unigram, else first two tokens
    seqs = [(tok(h["head"], add_special_tokens=False)["input_ids"], h["refused"]) for h in labels["refusal_heads"]]

    def token_sets(k):
        from collections import Counter
        n, ref = Counter(), Counter()
        for ids, refused in seqs:
            if len(ids) >= k:
                key = tuple(ids[:k])
                n[key] += 1
                ref[key] += refused
        R = [t for t in n if n[t] >= 5 and ref[t] / n[t] >= 0.8]
        C = [t for t in n if n[t] >= 5 and ref[t] / n[t] <= 0.2]
        return R, C

    R, C = token_sets(1)
    k_tokens = 1
    if len(R) < 3 or len(C) < 3:
        R, C = token_sets(2)
        k_tokens = 2
    assert len(R) >= 1 and len(C) >= 1, (len(R), len(C))
    firsts = sorted({t[0] for t in R + C})

    # ---- hooks
    hooks = []

    def clear():
        while hooks:
            hooks.pop().remove()

    def ablate(rhat):
        r = torch.tensor(rhat, dtype=torch.float32, device="cuda")

        def pre(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            hf = h.float()
            hf = hf - (hf @ r)[..., None] * r
            h2 = hf.to(h.dtype)
            if args:
                return (h2,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = h2
            return args, kwargs

        for m in list(layers) + [model.model.norm]:
            hooks.append(m.register_forward_pre_hook(pre, with_kwargs=True))

    def add_vec(vec, layer):
        v = torch.tensor(vec, dtype=torch.float32, device="cuda")
        m = layers[layer + 1] if layer + 1 < N_LAYERS else model.model.norm

        def pre(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            h2 = (h.float() + v).to(h.dtype)
            if args:
                return (h2,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = h2
            return args, kwargs

        hooks.append(m.register_forward_pre_hook(pre, with_kwargs=True))

    # ---- forward helpers
    def encode(ix):
        enc = tok([texts[i] for i in ix], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        return enc, pos

    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    TOKEN_BUDGET = 12000

    def batches(ix):
        order = sorted(ix, key=lambda i: lengths[i])
        cur = []
        for i in order:
            if cur and (len(cur) + 1 > batch or (len(cur) + 1) * lengths[i] > TOKEN_BUDGET):
                yield cur
                cur = []
            cur.append(i)
        if cur:
            yield cur

    @torch.no_grad()
    def forward(ix, hidden=False):
        """Returns (scores s, last-token hidden states [n, 36, d] or None) for prompt indices ix, in the order of ix."""
        ix = [int(i) for i in ix]
        S, H, done = {}, {}, []
        for bi in batches(ix):
            enc, pos = encode(bi)
            cap = []
            if hidden:
                for m in layers:
                    hooks.append(m.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu())))
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos,
                        use_cache=(k_tokens == 2), logits_to_keep=1)
            if hidden:
                for _ in layers:
                    hooks.pop().remove()
            lp1 = torch.log_softmax(out.logits[:, -1].float(), -1)
            if hidden:
                hb = torch.stack(cap, 1)
                for j, i in enumerate(bi):
                    H[i] = hb[j]
            if k_tokens == 1:
                sr = torch.logsumexp(lp1[:, [t[0] for t in R]], -1)
                sc = torch.logsumexp(lp1[:, [t[0] for t in C]], -1)
            else:
                cache = out.past_key_values
                L0 = enc["input_ids"].shape[1]
                lp2 = {}
                for t1 in firsts:
                    cache.crop(L0)
                    o2 = model(input_ids=torch.full((len(bi), 1), t1, device="cuda"),
                               attention_mask=torch.cat([enc["attention_mask"], torch.ones(len(bi), 1, device="cuda", dtype=enc["attention_mask"].dtype)], 1),
                               position_ids=pos[:, -1:] + 1, past_key_values=cache, use_cache=True)
                    lp2[t1] = torch.log_softmax(o2.logits[:, -1].float(), -1)
                seq = lambda ts: torch.stack([lp1[:, a] + lp2[a][:, b_] for a, b_ in ts], -1)
                sr, sc = torch.logsumexp(seq(R), -1), torch.logsumexp(seq(C), -1)
            for j, i in enumerate(bi):
                S[i] = float(sr[j] - sc[j])
            del out
            torch.cuda.empty_cache()
        s = np.array([S[i] for i in ix])
        return s, (torch.stack([H[i] for i in ix]).numpy() if hidden else None)

    # ---- pass 1: all prompts, unmodified
    all_ix = np.arange(len(rows))
    s_base, H = forward(all_ix, hidden=True)  # H: [N, 36, d] float32
    fit_h, fit_l = sel(lambda r: r["set"] == "fit_harm"), sel(lambda r: r["set"] == "fit_harmless")
    r_dir = H[fit_h].mean(0) - H[fit_l].mean(0)  # [36, d]
    rhat = r_dir / np.linalg.norm(r_dir, axis=-1, keepdims=True)
    proj = np.einsum("nld,ld->nl", H, rhat)  # [N, 36]

    # split-half reliability of the English direction
    rng = np.random.default_rng(SEED)
    a_h, a_l = rng.permutation(fit_h), rng.permutation(fit_l)
    rA = H[a_h[: len(a_h) // 2]].mean(0) - H[a_l[: len(a_l) // 2]].mean(0)
    rB = H[a_h[len(a_h) // 2:]].mean(0) - H[a_l[len(a_l) // 2:]].mean(0)
    cos = lambda x, y: (x * y).sum(-1) / np.linalg.norm(x, axis=-1) / np.linalg.norm(y, axis=-1)
    split_half = cos(rA, rB)

    groups = {}
    for r_ in rows:
        groups.setdefault(f"{r_['set']}|{r_['lang']}|{r_['kind']}", []).append(idx[r_["uid"]])
    group_means = {g: H[np.array(v)].mean(0) for g, v in groups.items()}
    del H

    # ---- layer sweep: ablate r̂_l everywhere, mean s on fit harmful
    sweep = []
    for l in range(MAX_CAND + 1):
        ablate(rhat[l])
        s, _ = forward(fit_h)
        clear()
        sweep.append(float(s.mean()))
        print(f"layer {l}: ablated mean s {sweep[-1]:.3f}", flush=True)
    lstar = int(np.argmin(sweep))

    ho_h = sel(lambda r: r["set"] == "aya_es" and r["lang"] == "en")
    pg_en = sel(lambda r: r["set"] == "pg" and r["lang"] == "en")
    ablate(rhat[lstar])
    s_ho_abl, _ = forward(ho_h)
    clear()
    add_vec(r_dir[lstar], lstar)
    s_fitl_add, _ = forward(fit_l)
    clear()
    gap = float(s_base[ho_h].mean() - s_base[pg_en].mean())
    drop = float(s_base[ho_h].mean() - s_ho_abl.mean())
    validity = dict(layer=lstar, heldout_harmful_s=float(s_base[ho_h].mean()), heldout_harmful_s_ablated=float(s_ho_abl.mean()),
                    pg_harmless_en_s=float(s_base[pg_en].mean()), gap=gap, drop=drop, drop_fraction=drop / gap if gap else None,
                    passes=bool(gap > 0 and drop >= 0.5 * gap), fit_harmless_s=float(s_base[fit_l].mean()),
                    fit_harmless_s_added=float(s_fitl_add.mean()), fit_harmful_s=float(s_base[fit_h].mean()))

    np.savez_compressed(out_dir / "direction.npz", uid=np.array(uid), proj=proj, s_base=s_base, r_dir=r_dir, split_half=split_half,
                        sweep=np.array(sweep), group_names=np.array(list(group_means)), group_means=np.stack(list(group_means.values())))
    meta = dict(model=MODEL, k_tokens=k_tokens, n_R=len(R), n_C=len(C), n_prompts=len(rows), lstar=lstar, sweep=sweep,
                validity=validity, split_half_cos=split_half.tolist())
    json.dump(meta, open(out_dir / "direction_meta.json", "w"), indent=1)
    analyze(rows, labels, proj, s_base, r_dir, group_means, split_half, lstar, meta, out_dir)


# -------------------------------------------------------------------------------------------------------------- analyze

def analyze(rows, labels, proj, s_base, r_dir, group_means, split_half, lstar, meta, out_dir):
    import numpy as np
    rng = np.random.default_rng(SEED)
    B = 2000
    look = {(r["set"], r["pair"], r["lang"]): i for i, r in enumerate(rows)}

    def vec(set_, lang, pairs, l):
        return np.array([proj[look[(set_, p, lang)], l] for p in pairs])

    aya_pairs = sorted({r["pair"] for r in rows if r["set"] == "aya_es"})
    pg_pairs = sorted({r["pair"] for r in rows if r["set"] == "pg"})
    ta_pairs = sorted({r["pair"] for r in rows if r["set"] == "tierA"})
    ta_aya = [p for p in ta_pairs if p.startswith("aya")]
    ta_isafe = [p for p in ta_pairs if not p.startswith("aya")]

    def auroc(pos, neg):
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort().argsort() + 1.0
        return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    def rho(harm_set, harm_pairs, lang, l, hb=None, pb=None):
        hp = harm_pairs if hb is None else [harm_pairs[i] for i in hb]
        pp = pg_pairs if pb is None else [pg_pairs[i] for i in pb]
        tgt = vec(harm_set, lang, hp, l).mean() - vec("pg", lang, pp, l).mean()
        ref = vec(harm_set, "en", hp, l).mean() - vec("pg", "en", pp, l).mean()
        return tgt / ref

    def ci(f, n1, n2):
        vals = [f(rng.integers(0, n1, n1), rng.integers(0, n2, n2)) for _ in range(B)]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    comps = {"es": ("aya_es", aya_pairs), "hi": ("tierA", ta_pairs), "hi_aya": ("tierA", ta_aya), "hi_indicsafe": ("tierA", ta_isafe)}
    res = dict(lstar=lstar, per_language={}, per_layer={})
    l = lstar
    for name, (hs, hp) in comps.items():
        lang = name[:2]
        out = {}
        out["rho"] = float(rho(hs, hp, lang, l))
        out["rho_ci"] = ci(lambda a, b: rho(hs, hp, lang, l, a, b), len(hp), len(pg_pairs))
        out["auroc_target"] = auroc(vec(hs, lang, hp, l), vec("pg", lang, pg_pairs, l))
        out["auroc_english"] = auroc(vec(hs, "en", hp, l), vec("pg", "en", pg_pairs, l))
        d = vec(hs, lang, hp, l) - vec(hs, "en", hp, l)
        en_gap = vec(hs, "en", hp, l).mean() - vec("pg", "en", pg_pairs, l).mean()
        out["paired_shift"] = float(d.mean())
        out["paired_shift_ci"] = [float(x) for x in np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)], [2.5, 97.5])]
        out["paired_shift_over_english_gap"] = float(d.mean() / en_gap)
        out["mean_proj"] = {k: float(v) for k, v in dict(harm_target=vec(hs, lang, hp, l).mean(), harm_en=vec(hs, "en", hp, l).mean(),
                                                         harmless_target=vec("pg", lang, pg_pairs, l).mean(), harmless_en=vec("pg", "en", pg_pairs, l).mean()).items()}
        sv = lambda set_, lg, pairs: float(np.mean([s_base[look[(set_, p, lg)]] for p in pairs]))
        out["mean_refusal_score"] = dict(harm_target=sv(hs, lang, hp), harm_en=sv(hs, "en", hp), harmless_target=sv("pg", lang, pg_pairs),
                                         harmless_en=sv("pg", "en", pg_pairs))
        out["n_harmful_pairs"] = len(hp)
        res["per_language"][name] = out
        res["per_layer"][name] = dict(rho=[float(rho(hs, hp, lang, x)) for x in range(N_LAYERS)],
                                      auroc_target=[auroc(vec(hs, lang, hp, x), vec("pg", lang, pg_pairs, x)) for x in range(N_LAYERS)],
                                      auroc_english=[auroc(vec(hs, "en", hp, x), vec("pg", "en", pg_pairs, x)) for x in range(N_LAYERS)])

    # direction agreement
    g = group_means
    cosf = lambda x, y: (x * y).sum(-1) / np.linalg.norm(x, axis=-1) / np.linalg.norm(y, axis=-1)
    dir_l = {"es": g["aya_es|es|harmful"] - g["pg|es|harmless"], "hi": g["tierA|hi|harmful"] - g["pg|hi|harmless"],
             "en_eval_es": g["aya_es|en|harmful"] - g["pg|en|harmless"], "en_eval_hi": g["tierA|en|harmful"] - g["pg|en|harmless"]}
    res["direction_cosine"] = {k: cosf(v, r_dir).tolist() for k, v in dir_l.items()}
    res["direction_cosine"]["split_half_ceiling"] = split_half.tolist()

    # behaviour link
    beh = {}
    for name, set_, lang in (("es", "aya_es", "es"), ("hi_calibration", "calib", "hi")):
        b = labels["behaviour"][lang]
        harmful_pairs = {r["pair"] for r in rows if r["set"] == set_ and r["kind"] == "harmful"}
        ok = [p for p, v in b.items() if p in harmful_pairs and v["en"] is True and v[lang] is not None]
        refd = [p for p in ok if b[p][lang]]
        notr = [p for p in ok if not b[p][lang]]
        if refd and notr:
            pr, pn = vec(set_, lang, refd, l), vec(set_, lang, notr, l)
            diffs = [pn[rng.integers(0, len(pn), len(pn))].mean() - pr[rng.integers(0, len(pr), len(pr))].mean() for _ in range(B)]
            beh[name] = dict(n_refused=len(refd), n_not_refused=len(notr), mean_proj_refused=float(pr.mean()), mean_proj_not_refused=float(pn.mean()),
                             ratio_not_over_refused=float(pn.mean() / pr.mean()), diff=float(pn.mean() - pr.mean()),
                             diff_ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                             mean_proj_en_same_pairs=float(vec(set_, "en", ok, l).mean()))
        else:
            beh[name] = dict(n_refused=len(refd), n_not_refused=len(notr))
    res["behaviour"] = beh
    res["meta"] = meta
    json.dump(res, open(Path(out_dir) / "direction_results.json", "w"), indent=1)
    print(json.dumps({k: res["per_language"][k]["rho"] for k in res["per_language"]}), "lstar", lstar, "valid", meta["validity"]["passes"])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prep":
        prep()
    elif a[0] == "run":
        bs = int(a[a.index("--batch") + 1]) if "--batch" in a else 32
        run(a[1], a[2], a[3], batch=bs)
