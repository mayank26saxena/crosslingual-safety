"""Phase 3: gated explicit-refusal steering (docs/prereg/2026-11-13_v2_phase3_gated_steering.md).

Stages:
  prep  (local, standard library only): writes data/v2/phase3_prompts.jsonl (not tracked) and results/v2/phase3/sets.json (ids only).
        Development and test items are the understood harmful and harmless pairs of the final split (results/v2/final_baseline/split.json,
        comprehension judged same or partial), Hindi and English.
  tune PROMPTS DIRS OUT_DIR [--smoke]  (GPU; imports nothing from the repository; DIRS holds direction.npz and es_dir.npz):
        1. forward pass over every development prompt: layer-14 projection onto the Phase 2a unit refusal direction (gate score);
        2. tau grid = 50th, 25th, 10th percentile of the English development harmful scores;
        3. Hindi development prompts generated under alpha 0, 0.75, 1.0, 1.25 with the gate open for every row whose score exceeds the
           lowest tau, in identical batches. A row's reply under (alpha, tau) is its alpha reply when its score exceeds tau and its alpha-0
           reply otherwise, so the nine grid settings are read off these four passes without further generation.
        Writes OUT_DIR/tune_scores.json (numbers only) and tune_replies.jsonl (reply text, not tracked).
  test PROMPTS DIRS OUT_DIR ALPHA TAU  (GPU): test prompts (Hindi, English) and the general-ability sample, under no steering and under the
        gated intervention, in identical batches. Writes OUT_DIR/test_replies.jsonl and test_scores.json.

Gate: a forward hook on layer 14 reads the last prompt token's residual during prefill, projects it on r14 (unit) and opens the gate for
rows above tau; a pre-hook on layer 25 (output of layer 24) adds alpha * e24 to gated rows at every position, prompt and generated.
Prompt and reply text is never printed.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SEED = 20261116
N_LAYERS = 36
GATE_LAYER = 14       # Phase 2a layer
STEER_LAYER = 24      # Phase 2b layer; vector enters layer 25
ALPHAS = (0.75, 1.0, 1.25)
PCTS = (50, 25, 10)
MAX_NEW = {"hi": 1024, "en": 512, "mmlu": 16}


def _jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


# ----------------------------------------------------------------------------------------------------------------- prep

def prep():
    root = Path(__file__).resolve().parents[2]
    s = json.load(open(root / "results/v2/final_baseline/split.json"))
    split, sets = s["split"], s["sets"]
    cdir = root / "notes/screening/comprehension/runs/opus"
    ck = json.load(open(root / "notes/screening/comprehension/key.json"))
    judged = {ck[r["item"].strip()]: r["judgement"].strip() for f in sorted(cdir.glob("part_*.csv"))
              for r in csv.DictReader(open(f, encoding="utf-8-sig"))}
    ok = {i for i, j in judged.items() if j in ("same", "partial")}
    prompts = {l: {r["item"]: r for r in _jl(root / f"data/v2/hindi_all_prompts_{l}.jsonl")} for l in ("hi", "en")}
    rows, ids = [], {}
    for part in ("dev", "test"):
        for kind, setname in (("harmful", "study_harmful"), ("harmless", "study_harmless")):
            sel = sorted(i for i in sets[setname] if split.get(i) == part and i in ok)
            ids[f"{part}_{kind}"] = sel
            for i in sel:
                for l in ("hi", "en"):
                    rows.append(dict(uid=f"{part}_{kind}|{i}|{l}", set=f"{part}_{kind}", lang=l, pair=i,
                                     messages=prompts[l][i]["messages"]))
    mm = root / "data/v2/phase3_mmlu.jsonl"  # written by the general-ability sampling step, once the source is approved
    if mm.exists():
        for r in _jl(mm):
            rows.append(dict(uid=f"mmlu|{r['id']}|{r['lang']}", set="mmlu", lang=r["lang"], pair=r["id"], answer=r["answer"],
                             messages=r["messages"]))
    with open(root / "data/v2/phase3_prompts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = root / "results/v2/phase3"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(ids, open(out / "sets.json", "w"), indent=1)
    print({k: len(v) for k, v in ids.items()}, dict(Counter((r["set"], r["lang"]) for r in rows)))


# ------------------------------------------------------------------------------------------------------------------ GPU

class Runner:
    def __init__(self, dirs):
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.np, self.torch = np, torch
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda").eval()
        self.layers = self.model.model.layers
        assert len(self.layers) == N_LAYERS
        r = np.load(Path(dirs) / "direction.npz")["r_dir"][GATE_LAYER]
        self.rhat = torch.tensor(r / np.linalg.norm(r), dtype=torch.float32, device="cuda")
        self.e = torch.tensor(np.load(Path(dirs) / "es_dir.npz")["e_dir"][STEER_LAYER], dtype=torch.float32, device="cuda")
        self.state = {"score": None, "tau": None, "alpha": 0.0, "mask": None}
        self.layers[GATE_LAYER].register_forward_hook(self._gate)
        self.layers[STEER_LAYER + 1].register_forward_pre_hook(self._steer, with_kwargs=True)
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")

    def _gate(self, mod, args, out):
        if self.state["score"] is None:  # prefill only
            h = out[0] if isinstance(out, tuple) else out
            sc = h[:, -1].float() @ self.rhat
            self.state["score"] = sc
            tau = self.state["tau"]
            self.state["mask"] = (sc > tau).float() if tau is not None else self.torch.zeros_like(sc)

    def _steer(self, mod, args, kwargs):
        if not self.state["alpha"]:
            return None
        h = args[0] if args else kwargs["hidden_states"]
        add = self.state["alpha"] * self.e[None, None, :] * self.state["mask"][:, None, None]
        h2 = (h.float() + add).to(h.dtype)
        if args:
            return (h2,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = h2
        return args, kwargs

    def texts(self, rows):
        return [self.tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]

    def batches(self, rows, size):
        """Fixed batches: sorted by prompt length, then cut; identical for every condition."""
        lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in self.texts(rows)]
        order = sorted(range(len(rows)), key=lambda i: (lens[i], rows[i]["uid"]))
        return [order[i:i + size] for i in range(0, len(order), size)]

    def scores(self, rows, size=32):
        torch = self.torch
        out = [None] * len(rows)
        texts = self.texts(rows)
        for b in self.batches(rows, size):
            enc = self.tok([texts[i] for i in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            self.state.update(score=None, tau=None, alpha=0.0)
            with torch.no_grad():
                self.model(**enc, logits_to_keep=1)
            for j, i in enumerate(b):
                out[i] = float(self.state["score"][j])
        return out

    def generate(self, rows, alpha, tau, max_new, size, f, tag):
        torch = self.torch
        texts = self.texts(rows)
        done = 0
        for b in self.batches(rows, size):
            enc = self.tok([texts[i] for i in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            self.state.update(score=None, tau=tau, alpha=alpha)
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
                f.write(json.dumps(dict(uid=r["uid"], set=r["set"], lang=r["lang"], pair=r["pair"], condition=tag, alpha=alpha, tau=tau,
                                        score=float(self.state["score"][j]), gate=bool(self.state["mask"][j] > 0), n_new_tokens=n,
                                        hit_limit=self.eos not in ids[:n], reply=self.tok.decode(ids[:n], skip_special_tokens=True)),
                                   ensure_ascii=False) + "\n")
            f.flush()
            done += len(b)
            print(f"{tag}: {done}/{len(rows)}", flush=True)
            del gen
            torch.cuda.empty_cache()


def tune(prompts, dirs, out_dir, smoke=False):
    import numpy as np
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _jl(prompts) if r["set"].startswith("dev_")]
    if smoke:
        rows = [r for r in rows if r["set"] == "dev_harmless"][:32]
    R = Runner(dirs)
    sc = R.scores(rows)
    groups = {f"{s}_{l}": [sc[i] for i, r in enumerate(rows) if r["set"] == s and r["lang"] == l]
              for s in ("dev_harmful", "dev_harmless") for l in ("hi", "en")}
    ref = groups["dev_harmful_en"] or groups["dev_harmless_en"]
    taus = {p: float(np.percentile(ref, p)) for p in PCTS}
    res = dict(model=MODEL, revision=REVISION, gate_layer=GATE_LAYER, steer_layer=STEER_LAYER, e_norm=float(R.e.norm()), taus=taus,
               score_summary={g: dict(n=len(v), mean=float(np.mean(v)), p10=float(np.percentile(v, 10)), p50=float(np.percentile(v, 50)))
                              for g, v in groups.items() if v},
               firing={g: {p: float(np.mean(np.array(v) > t)) for p, t in taus.items()} for g, v in groups.items() if v}, smoke=smoke)
    json.dump(res, open(out_dir / "tune_scores.json", "w"), indent=1)
    print("taus", taus, flush=True)
    hi = [r for r in rows if r["lang"] == "hi"]
    tmin = min(taus.values())
    with open(out_dir / "tune_replies.jsonl", "w", encoding="utf-8") as f:
        R.generate(hi, 0.0, None, 32 if smoke else MAX_NEW["hi"], 16, f, "alpha_0")
        for a in ALPHAS:
            R.generate(hi, a, tmin, 32 if smoke else MAX_NEW["hi"], 16, f, f"alpha_{a}")
    print("DONE", flush=True)


def test(prompts, dirs, out_dir, alpha, tau):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _jl(prompts) if r["set"].startswith("test_") or r["set"] == "mmlu"]
    R = Runner(dirs)
    with open(out_dir / "test_replies.jsonl", "w", encoding="utf-8") as f:
        for grp, mx in (("hi", MAX_NEW["hi"]), ("en", MAX_NEW["en"]), ("mmlu", MAX_NEW["mmlu"])):
            sub = [r for r in rows if (r["set"] == "mmlu") == (grp == "mmlu") and (grp == "mmlu" or r["lang"] == grp)]
            if not sub:
                continue
            R.generate(sub, 0.0, None, mx, 16, f, f"{grp}_unsteered")
            R.generate(sub, float(alpha), float(tau), mx, 16, f, f"{grp}_gated")
    json.dump(dict(alpha=float(alpha), tau=float(tau), model=MODEL, revision=REVISION), open(out_dir / "test_scores.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prep":
        prep()
    elif a[0] == "tune":
        tune(a[1], a[2], a[3], smoke="--smoke" in a)
    elif a[0] == "test":
        test(*a[1:6])
