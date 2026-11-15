"""Phase 6: replication on Llama-3.1-8B-Instruct (docs/prereg/2026-11-15_v2_phase6_llama_replication.md).

Stage A (this file, first part):
  prepA  (local, standard library only): writes data/v2/phase6_promptsA.jsonl (not tracked): English versions of the 782 Aya Spanish-set
         requests and the 300 PolyGuard harmless control requests (1,082 prompts), same messages as v2.
  genA PROMPTS OUT_DIR [--smoke]  (GPU; imports nothing from the repository): greedy English replies, 512 new tokens, fixed length-sorted
         batches of 16. Writes OUT_DIR/genA_replies.jsonl (reply text, not tracked).
  folderA REPLIES  (local): blind grading folder notes/grading/phase6A/ (50-row parts, shuffled, seed 20261123) and its key.
Stage B (directions, gate, test run) is added before stage B runs; its rules are fixed in the pre-registration.
Prompt and reply text is never printed.
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
N_LAYERS = 32
SEED = 20261123
MAX_NEW = {"hi": 1024, "en": 512, "mmlu": 16}
ROOT = Path(__file__).resolve().parents[2]


def _jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def _user(msgs):
    return next(m["content"] for m in msgs if m["role"] == "user")


# ----------------------------------------------------------------------------------------------------------------- prep

def prepA():
    d = ROOT / "data/v2"
    rows = []
    for r in _jl(d / "spanish_prompts_en.jsonl"):
        rows.append(dict(uid=f"fitA_harm|{r['item']}|en", set="fitA_harm", lang="en", pair=r["item"], messages=r["messages"]))
    for r in _jl(d / "pg_harmless_prompts_en.jsonl"):
        rows.append(dict(uid=f"fitA_pg|{r['item']}|en", set="fitA_pg", lang="en", pair=r["item"], messages=r["messages"]))
    assert Counter(r["set"] for r in rows) == {"fitA_harm": 782, "fitA_pg": 300}, Counter(r["set"] for r in rows)
    assert len({r["uid"] for r in rows}) == len(rows)
    with open(d / "phase6_promptsA.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(Counter(r["set"] for r in rows)))


# ------------------------------------------------------------------------------------------------------------------ GPU

class Gen:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.manual_seed(0)
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = "<|finetune_right_pad_id|>"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda").eval()
        assert len(self.model.model.layers) == N_LAYERS
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])

    def texts(self, rows):
        return [self.tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in rows]

    def batches(self, rows, size):
        """Fixed batches: sorted by prompt length, then cut; identical for every condition."""
        lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in self.texts(rows)]
        order = sorted(range(len(rows)), key=lambda i: (lens[i], rows[i]["uid"]))
        return [order[i:i + size] for i in range(0, len(order), size)]

    def generate(self, rows, max_new, size, f, tag, extra=None):
        torch = self.torch
        texts = self.texts(rows)
        done = 0
        for b in self.batches(rows, size):
            enc = self.tok([texts[i] for i in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                          pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(b):
                ids = new[j].tolist()
                n = len(ids)
                while n and ids[n - 1] == self.tok.pad_token_id:
                    n -= 1
                r = rows[i]
                rec = dict(uid=r["uid"], set=r["set"], lang=r["lang"], pair=r["pair"], condition=tag, n_new_tokens=n,
                           hit_limit=not any(t in self.eos for t in ids[:n]), reply=self.tok.decode(ids[:n], skip_special_tokens=True))
                if extra:
                    rec.update(extra(j))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done += len(b)
            print(f"{tag}: {done}/{len(rows)}", flush=True)
            del gen
            torch.cuda.empty_cache()


def genA(prompts, out_dir, smoke=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _jl(prompts)
    if smoke:
        rows = rows[:8] + rows[-8:]
    G = Gen()
    with open(out_dir / ("genA_smoke.jsonl" if smoke else "genA_replies.jsonl"), "w", encoding="utf-8") as f:
        G.generate(rows, 32 if smoke else MAX_NEW["en"], 16, f, "unsteered")
    json.dump(dict(model=MODEL, revision=REVISION, n=len(rows), smoke=smoke), open(out_dir / "genA_meta.json", "w"), indent=1)
    print("DONE", flush=True)


# --------------------------------------------------------------------------------------------------------------- folder

def folderA(replies):
    rows = _jl(replies)
    assert len(rows) == 1082 and len({r["uid"] for r in rows}) == 1082
    random.Random(SEED).shuffle(rows)
    out = ROOT / "notes/grading/phase6A"
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "runs/claude-sonnet-5.5-cc").mkdir(parents=True, exist_ok=True)
    msgs = {r["uid"]: r["messages"] for r in _jl(ROOT / "data/v2/phase6_promptsA.jsonl")}
    key, table = {}, []
    for i, r in enumerate(rows, 1):
        sid = f"A{i:04d}"
        key[sid] = {k: r[k] for k in ("uid", "set", "lang", "pair", "condition", "hit_limit", "n_new_tokens")}
        t = _user(msgs[r["uid"]])
        table.append(dict(item=sid, request_english=t, message=t, reply=r["reply"]))
    parts = 0
    for j in range(0, len(table), 50):
        parts += 1
        with open(out / f"input/part_{parts:02d}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item", "request_english", "message", "reply"], quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(table[j:j + 50])
    (out / "input/items.txt").write_text("\n".join(t["item"] for t in table) + "\n")
    (out / "key.json").write_text(json.dumps(dict(graded=key), indent=1))
    print(f"{len(table)} graded rows in {parts} parts")


# ============================================================================================================ stage B
# prepB (local, after stage A is graded): data/v2/phase6_promptsB.jsonl (not tracked) and data/v2/phase6_labelsB.json (first-token heads
#   with broad labels; explicit/soft labels for the 782 fit requests). Uses the same seeds and set definitions as Phase 2a, 2b and 3.
# runB PROMPTS LABELS OUT_DIR [--smoke] (GPU; standalone): Phase 2a refusal direction and layer rule, Phase 2b explicit-soft direction and
#   layer rule (layers 0..25 = first 80% of 32), gate threshold from English development harmful prompts, then the test run under no
#   steering and under the gated push (alpha 1.25) in identical batches. Writes OUT_DIR/stageB_dirs.npz, stageB_results.json (numbers
#   only) and test_replies.jsonl (reply text, not tracked).

SEED_2A, SEED_2B = 20261112, 20261113
MAX_CAND = 25          # first 80% of 32 layers (Qwen: 28 of 36)
ALPHA, PCT = 1.25, 50  # Phase 3 chosen setting, carried over


def _grades(folder):
    key = json.load(open(ROOT / "notes/grading" / folder / "key.json"))["graded"]
    out = {}
    for f in sorted((ROOT / "notes/grading" / folder / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            out[key[r["item"].strip()]["uid"]] = r["label"].strip().upper()
    return out


def prepB():
    d = ROOT / "data/v2"
    rows = []

    def add(uid, set_, lang, kind, pair, messages, **kw):
        rows.append(dict(uid=uid, set=set_, lang=lang, kind=kind, pair=pair, messages=messages, **kw))

    # Phase 2a sets, same derivation and seed as refusal_direction.prep
    es = {r["item"]: r for r in _jl(d / "spanish_prompts_es.jsonl")}
    en = {r["item"]: r for r in _jl(d / "spanish_prompts_en.jsonl")}
    ids = sorted(es)
    random.Random(SEED_2A).shuffle(ids)
    fit_ids, ho_ids = sorted(ids[:len(ids) // 2]), sorted(ids[len(ids) // 2:])
    for k in fit_ids:
        add(f"fit_harm|{k}|en", "fit_harm", "en", "harmful", k, en[k]["messages"])
    for k in ho_ids:
        add(f"aya_ho|{k}|en", "aya_ho", "en", "harmful", k, en[k]["messages"])
    pg = {r["item"]: r for r in _jl(d / "pg_harmless_prompts_en.jsonl")}
    template = next(iter(pg.values()))["messages"]
    unused = [x for x in _jl(d / "after_exclusions.jsonl") if x["source"] == "polyguard" and x["benchmark_label"] == "harmless"
              and not x["extra"].get("adversarial") and x["item_id"] not in pg]
    assert len(unused) == 190, len(unused)
    for x in unused:
        msgs = [m if m["role"] != "user" else dict(m, content=x["en"]) for m in template]
        add(f"fit_harmless|{x['item_id']}|en", "fit_harmless", "en", "harmless", x["item_id"], msgs)
    for k, r in pg.items():
        add(f"pg|{k}|en", "pg", "en", "harmless", k, r["messages"])
    # Phase 2b fit set: the 782 English Aya requests, labelled by Llama's own graded replies
    lab = _grades("phase6A")
    assert len(lab) == 1082, len(lab)
    for k in sorted(en):
        add(f"es_fit|{k}|en", "es_fit", "en", "harmful", k, en[k]["messages"], label=lab[f"fitA_harm|{k}|en"])
    # Phase 3 development (English harmful, for tau) and test sets, and the Global-MMLU sample
    for r in _jl(d / "phase3_prompts.jsonl"):
        if (r["set"] == "dev_harmful" and r["lang"] == "en") or r["set"].startswith("test_") or r["set"] == "mmlu":
            add(r["uid"], r["set"], r["lang"], "mmlu" if r["set"] == "mmlu" else r["set"].split("_")[1], r["pair"], r["messages"],
                **({"answer": r["answer"]} if "answer" in r else {}))
    with open(d / "phase6_promptsB.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Refusal-score heads: Llama's English stage A replies with a graded broad label (DECLINED left out)
    rep = {r["uid"]: r["reply"] for r in _jl(ROOT / "results/v2/phase6/stageA/genA_replies.jsonl")}
    heads = [dict(head=rep[u][:60], refused=l.startswith("REFUSE")) for u, l in sorted(lab.items()) if l != "DECLINED"]
    json.dump(dict(refusal_heads=heads), open(d / "phase6_labelsB.json", "w"), ensure_ascii=False)
    print(dict(Counter((r["set"], r["lang"]) for r in rows)), "heads", len(heads), "refused", sum(h["refused"] for h in heads),
          "fit labels", dict(Counter(r["label"] for r in rows if r["set"] == "es_fit")))


def _auroc(pos, neg):
    import numpy as np
    allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def runB(prompts_path, labels_path, out_dir, smoke=False):
    import numpy as np
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _jl(prompts_path)
    if smoke:
        keep, seen = [], Counter()
        for r in rows:
            seen[(r["set"], r["lang"])] += 1
            if seen[(r["set"], r["lang"])] <= 10:
                keep.append(r)
        rows = keep
    labels = json.load(open(labels_path))
    G = Gen()
    tok, model = G.tok, G.model
    layers = model.model.layers
    texts = G.texts(rows)
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    sel = lambda f: np.array([i for i, r in enumerate(rows) if f(r)], dtype=int)
    act = sel(lambda r: r["set"] != "mmlu")

    # ---- refusal token sets (Phase 2a rule), unigram, else first two tokens
    seqs = [(tok(h["head"], add_special_tokens=False)["input_ids"], h["refused"]) for h in labels["refusal_heads"]]

    def token_sets(k):
        n, ref = Counter(), Counter()
        for ids_, refused in seqs:
            if len(ids_) >= k:
                key = tuple(ids_[:k])
                n[key] += 1
                ref[key] += refused
        return ([t for t in n if n[t] >= 5 and ref[t] / n[t] >= 0.8], [t for t in n if n[t] >= 5 and ref[t] / n[t] <= 0.2])

    R, C = token_sets(1)
    k_tokens = 1
    if len(R) < 3 or len(C) < 3:
        R, C = token_sets(2)
        k_tokens = 2
    assert len(R) >= 1 and len(C) >= 1, (len(R), len(C))
    firsts = sorted({t[0] for t in R + C})

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

    def batches(ix, bs=32, budget=12000):
        order = sorted((int(i) for i in ix), key=lambda i: lengths[i])
        cur = []
        for i in order:
            if cur and (len(cur) + 1 > bs or (len(cur) + 1) * lengths[i] > budget):
                yield cur
                cur = []
            cur.append(i)
        if cur:
            yield cur

    @torch.no_grad()
    def forward(ix, hidden=False):
        ix = [int(i) for i in ix]
        S, H = {}, {}
        for bi in batches(ix):
            enc = tok([texts[i] for i in bi], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            cap = []
            if hidden:
                for m in layers:
                    hooks.append(m.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu())))
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos,
                        use_cache=(k_tokens == 2), logits_to_keep=1)
            if hidden:
                for _ in layers:
                    hooks.pop().remove()
                hb = torch.stack(cap, 1)
                for j, i in enumerate(bi):
                    H[i] = hb[j]
            lp1 = torch.log_softmax(out.logits[:, -1].float(), -1)
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
        return np.array([S[i] for i in ix]), (torch.stack([H[i] for i in ix]).numpy() if hidden else None)

    # ---- pass 1: every non-MMLU prompt, unmodified
    s_act, H_act = forward(act, hidden=True)
    pos_of = {int(i): j for j, i in enumerate(act)}
    s_base = np.full(len(rows), np.nan)
    s_base[act] = s_act
    Hx = lambda ix: H_act[[pos_of[int(i)] for i in ix]]
    print("activations", H_act.shape, flush=True)

    # ---- Phase 2a: refusal direction, layer by ablation, validity
    fit_h, fit_l = sel(lambda r: r["set"] == "fit_harm"), sel(lambda r: r["set"] == "fit_harmless")
    r_dir = Hx(fit_h).mean(0) - Hx(fit_l).mean(0)
    rhat = r_dir / np.linalg.norm(r_dir, axis=-1, keepdims=True)
    sweep = []
    for l in range(3 if smoke else MAX_CAND + 1):
        ablate(rhat[l])
        s, _ = forward(fit_h)
        clear()
        sweep.append(float(s.mean()))
        print(f"layer {l}: ablated mean s {sweep[-1]:.3f}", flush=True)
    lr = int(np.argmin(sweep))
    ho_h, pg_en = sel(lambda r: r["set"] == "aya_ho"), sel(lambda r: r["set"] == "pg")
    ablate(rhat[lr])
    s_ho_abl, _ = forward(ho_h)
    clear()
    gap = float(s_base[ho_h].mean() - s_base[pg_en].mean())
    drop = float(s_base[ho_h].mean() - s_ho_abl.mean())
    valid_r = bool(gap > 0 and drop >= 0.5 * gap)
    print("refusal layer", lr, "drop fraction", round(drop / gap, 3) if gap else None, flush=True)

    # ---- Phase 2b: explicit-soft direction, 5-fold CV AUROC layer choice
    fit = sel(lambda r: r["set"] == "es_fit")
    fe = np.array([i for i in fit if rows[i]["label"] == "REFUSE_EXPLICIT"])
    fs = np.array([i for i in fit if rows[i]["label"] == "REFUSE_SOFT"])
    He, Hs = Hx(fe), Hx(fs)
    rng = np.random.default_rng(SEED_2B)
    fold_e, fold_s = rng.permutation(len(fe)) % 5, rng.permutation(len(fs)) % 5
    cv = []
    for l in range(MAX_CAND + 1):
        aucs = []
        for f_ in range(5):
            if not (fold_e == f_).any() or not (fold_s == f_).any():
                continue
            dvec = He[fold_e != f_, l].mean(0) - Hs[fold_s != f_, l].mean(0)
            dvec = dvec / np.linalg.norm(dvec)
            aucs.append(_auroc(He[fold_e == f_, l] @ dvec, Hs[fold_s == f_, l] @ dvec))
        cv.append(float(np.mean(aucs)) if aucs else float("nan"))
    le = int(np.nanargmax(cv))
    e_dir = He.mean(0) - Hs.mean(0)
    valid_e = bool(cv[le] >= 0.75)
    print("explicit-soft layer", le, "cv auroc", round(cv[le], 3), flush=True)

    # ---- projections on the evaluation prompts (numbers only)
    ev = sel(lambda r: r["set"].startswith("test_") or r["set"] == "dev_harmful")
    He_ev = Hx(ev)
    proj = {rows[i]["uid"]: dict(r=float(He_ev[j, lr] @ rhat[lr]), e=float(He_ev[j, le] @ (e_dir[le] / np.linalg.norm(e_dir[le]))))
            for j, i in enumerate(ev)}
    ehat_le = e_dir[le] / np.linalg.norm(e_dir[le])
    fit_gap = float((He[:, le] @ ehat_le).mean() - (Hs[:, le] @ ehat_le).mean())
    dev_en = [proj[rows[i]["uid"]]["r"] for i in sel(lambda r: r["set"] == "dev_harmful" and r["lang"] == "en")]
    tau = float(np.percentile(dev_en, PCT))
    del H_act, He, Hs
    np.savez_compressed(out_dir / "stageB_dirs.npz", r_dir=r_dir, e_dir=e_dir, sweep=np.array(sweep), cv=np.array(cv))
    res = dict(model=MODEL, revision=REVISION, smoke=smoke, k_tokens=k_tokens, n_R=len(R), n_C=len(C),
               refusal=dict(layer=lr, sweep=sweep, heldout_harmful_s=float(s_base[ho_h].mean()), heldout_harmful_s_ablated=float(s_ho_abl.mean()),
                            pg_harmless_en_s=float(s_base[pg_en].mean()), gap=gap, drop=drop, drop_fraction=drop / gap if gap else None,
                            valid=valid_r),
               explicit_soft=dict(layer=le, cv_auroc=cv, valid=valid_e, n_explicit=int(len(fe)), n_soft=int(len(fs)), fit_gap=fit_gap,
                                  e_norm=float(np.linalg.norm(e_dir[le]))),
               gate=dict(layer=lr, tau=tau, pct=PCT, steer_layer=le, alpha=ALPHA), projections=proj)
    json.dump(res, open(out_dir / "stageB_results.json", "w"), indent=1)

    # ---- test run: unsteered and gated, identical batches (Phase 3 test procedure)
    steer = valid_r and valid_e
    rhat_t = torch.tensor(rhat[lr], dtype=torch.float32, device="cuda")
    e_t = torch.tensor(e_dir[le], dtype=torch.float32, device="cuda")
    st = {"score": None, "mask": None, "tau": None, "alpha": 0.0}

    def gate(mod, args, out):
        if st["score"] is None:
            h = out[0] if isinstance(out, tuple) else out
            sc = h[:, -1].float() @ rhat_t
            st["score"] = sc
            st["mask"] = (sc > st["tau"]).float() if st["tau"] is not None else torch.zeros_like(sc)

    def push(mod, args, kwargs):
        if not st["alpha"]:
            return None
        h = args[0] if args else kwargs["hidden_states"]
        h2 = (h.float() + st["alpha"] * e_t[None, None, :] * st["mask"][:, None, None]).to(h.dtype)
        if args:
            return (h2,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = h2
        return args, kwargs

    layers[lr].register_forward_hook(gate)
    layers[le + 1].register_forward_pre_hook(push, with_kwargs=True)
    orig = G.model.generate

    def gen_wrapped(**kw):
        st["score"] = None
        return orig(**kw)

    G.model.generate = gen_wrapped
    extra = lambda j: dict(score=float(st["score"][j]), gate=bool(st["mask"][j] > 0), alpha=st["alpha"], tau=st["tau"])
    test = [r for r in rows if r["set"].startswith("test_") or r["set"] == "mmlu"]
    with open(out_dir / "test_replies.jsonl", "w", encoding="utf-8") as f:
        for grp in ("hi", "en", "mmlu"):
            sub = [r for r in test if (r["set"] == "mmlu") == (grp == "mmlu") and (grp == "mmlu" or r["lang"] == grp)]
            if not sub:
                continue
            mx = 16 if smoke else MAX_NEW[grp]
            st.update(tau=None, alpha=0.0)
            G.generate(sub, mx, 16, f, f"{grp}_unsteered", extra)
            if steer:
                st.update(tau=tau, alpha=ALPHA)
                G.generate(sub, mx, 16, f, f"{grp}_gated", extra)
    print("steered" if steer else "NOT steered (a direction failed validity)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prepA":
        prepA()
    elif a[0] == "genA":
        genA(a[1], a[2], smoke="--smoke" in a)
    elif a[0] == "folderA":
        folderA(a[1])
    elif a[0] == "prepB":
        prepB()
    elif a[0] == "runB":
        runB(a[1], a[2], a[3], smoke="--smoke" in a)
