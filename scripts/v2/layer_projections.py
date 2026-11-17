"""Per-layer refusal-direction projections for Llama-3.1-8B on the held-out test items (figure "where refusal lives").

Forward passes only, no generation. For each of the 404 harmful and 169 harmless test pairs, in English and Hindi, take the last-token
output of every decoder layer and project it on that layer's English refusal direction (r_dir[l] / |r_dir[l]|, from Phase 6,
results/v2/phase6/stageB/stageB_dirs.npz). Batching and position ids follow phase6_llama.runB exactly.

  run PROMPTS DIRS OUT [--smoke]   (GPU) writes OUT/layer_proj.npz (uids, projections [n, layers]) and OUT/layer_proj_meta.json
  summarise NPZ STAGEB_RESULTS OUT_JSON   (local, numpy) per-layer AUROC (harmful vs harmless) per language, harmful Hindi/English mean
                                    ratio and rho; checks layer 10 against the stored Phase 6 projections (fails on mismatch).
Numbers only; no prompt or reply text is written or printed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run(prompts, dirs, out, smoke=False):
    import numpy as np
    import torch
    from phase6_llama import Gen, MODEL, REVISION, _jl
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _jl(prompts) if r["set"] in ("test_harmful", "test_harmless")]
    if smoke:
        rows = rows[:24]
    G = Gen()
    tok, model = G.tok, G.model
    layers = model.model.layers
    d = np.load(dirs)
    R = d["r_dir"] / np.linalg.norm(d["r_dir"], axis=-1, keepdims=True)
    texts = G.texts(rows)
    P = {}
    with torch.no_grad():
        for bi in G.batches(rows, 16):
            enc = tok([texts[i] for i in bi], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            cap = []
            hooks = [m.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu()))
                     for m in layers]
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos, logits_to_keep=1)
            for h in hooks:
                h.remove()
            hb = torch.stack(cap, 1).numpy()  # [batch, layers, hidden]
            for j, i in enumerate(bi):
                P[rows[i]["uid"]] = (hb[j] * R).sum(-1)
            print(f"{len(P)}/{len(rows)}", flush=True)
            torch.cuda.empty_cache()
    uids = sorted(P)
    np.savez_compressed(out / "layer_proj.npz", uids=np.array(uids), proj=np.stack([P[u] for u in uids]))
    json.dump(dict(model=MODEL, revision=REVISION, n=len(uids), layers=len(layers), smoke=smoke), open(out / "layer_proj_meta.json", "w"))
    print("DONE", flush=True)


def summarise(npz, stageb, out_json):
    import numpy as np
    z = np.load(npz)
    uids, proj = list(z["uids"]), z["proj"]
    ix = {u: i for i, u in enumerate(uids)}
    stored = json.load(open(stageb))["projections"]
    common = [u for u in uids if u in stored]
    diffs = np.array([abs(proj[ix[u], 10] - stored[u]["r"]) for u in common])
    scale = np.mean([abs(stored[u]["r"]) for u in common])
    print(f"layer-10 check on {len(common)} items: max |diff| {diffs.max():.4f}, mean {diffs.mean():.4f} (mean |value| {scale:.2f})")
    if diffs.mean() > 0.02 * scale or diffs.max() > 0.25 * scale:
        raise SystemExit("layer-10 projections do not reproduce Phase 6; batching or hook differs")

    def auroc(pos, neg):
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort().argsort() + 1.0
        return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    sel = lambda st, lang: np.array([ix[u] for u in uids if u.startswith(st + "|") and u.endswith("|" + lang)])
    L = proj.shape[1]
    res = dict(layers=L, check=dict(n=len(common), max_abs_diff=float(diffs.max()), mean_abs_diff=float(diffs.mean())))
    for lang in ("en", "hi"):
        h, s = sel("test_harmful", lang), sel("test_harmless", lang)
        res[f"auroc_{lang}"] = [auroc(proj[h, l], proj[s, l]) for l in range(L)]
        res[f"harmful_mean_{lang}"] = [float(proj[h, l].mean()) for l in range(L)]
        res[f"harmless_mean_{lang}"] = [float(proj[s, l].mean()) for l in range(L)]
    res["ratio_harmful"] = [h / e if abs(e) > 1e-6 else None for h, e in zip(res["harmful_mean_hi"], res["harmful_mean_en"])]
    res["rho"] = [(hh - hs) / (eh - es) if abs(eh - es) > 1e-6 else None for hh, hs, eh, es in
                  zip(res["harmful_mean_hi"], res["harmless_mean_hi"], res["harmful_mean_en"], res["harmless_mean_en"])]
    json.dump(res, open(out_json, "w"), indent=1)
    print("wrote", out_json, "| layer 10 ratio %.3f" % res["ratio_harmful"][10])


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--smoke"]
    if a[0] == "run":
        run(a[1], a[2], a[3], "--smoke" in sys.argv)
    elif a[0] == "summarise":
        summarise(a[1], a[2], a[3])
