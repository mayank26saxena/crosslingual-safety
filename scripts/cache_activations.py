"""Cache residual-stream activations for Step 2 (dev MultiJail requests and XSTest prompts only).

For each prompt and every hidden state (0 = embeddings, L = after layer L), stores the last instruction token (t_inst)
and the final five template tokens (t_post). The confirm split is never read.

Output: results/step2/activations.npz with arrays t_inst [n, 29, d], t_post [n, 29, 5, d], and row metadata in
results/step2/activation_rows.jsonl.

Usage: python scripts/cache_activations.py
"""

import json
import os

import numpy as np
import torch

from common import ROOT, load_model, prompt_text, read_jsonl, write_jsonl

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = ROOT / os.environ.get("CS_STEP_DIR", "results/step2")
XSTEST = ROOT / os.environ.get("CS_XSTEST", "data/rendered/xstest_v2_bn.jsonl")
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
BATCH = 32


def collect():
    splits = json.loads((ROOT / "data/splits/splits.json").read_text())
    half = {t: "a" for t in splits["multijail"]["dev_a"]} | {t: "b" for t in splits["multijail"]["dev_b"]}
    rows = []
    for it in read_jsonl(ROOT / "data/rendered/multijail_gap.jsonl"):
        if it["lang"] in ("en", "bn", "sw"):
            assert it["task_id"] in half, "confirm split must stay sealed"
            rows.append(dict(dataset="multijail", label="harmful", task_id=it["task_id"], lang=it["lang"],
                             half=half[it["task_id"]], messages=it["messages"]))
    xhalf = {}
    for key, ids in splits["xstest"].items():
        for t in ids:
            xhalf[t] = key.split("_")[1]
    for it in read_jsonl(XSTEST):
        label = "harmful" if it["label"] == "unsafe" else "harmless"
        common = dict(dataset="xstest", label=label, task_id=it["task_id"], half=xhalf[it["task_id"]],
                      category=it["category"])
        rows.append(dict(common, lang="en", messages=[SYSTEM, {"role": "user", "content": it["source_en"]}]))
        if it["format_ok"]:
            rows.append(dict(common, lang="bn", messages=it["messages"]))
    return rows


def main():
    rows = collect()
    tok, model = load_model(MODEL)
    close = tok.convert_tokens_to_ids("<|im_end|>")
    for r in rows:
        r["ids"] = tok(prompt_text(tok, r), add_special_tokens=False)["input_ids"]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    t_inst, t_post = [None] * len(rows), [None] * len(rows)
    for start in range(0, len(order), BATCH):
        idx = order[start: start + BATCH]
        width = max(len(rows[i]["ids"]) for i in idx)
        batch = torch.full((len(idx), width), tok.pad_token_id)
        mask = torch.zeros_like(batch)
        for b, i in enumerate(idx):  # left padding keeps the template tokens at the same positions
            ids = rows[i]["ids"]
            batch[b, width - len(ids):] = torch.tensor(ids)
            mask[b, width - len(ids):] = 1
        with torch.no_grad():
            hs = model(batch.to(model.device), attention_mask=mask.to(model.device), output_hidden_states=True).hidden_states
        hs = torch.stack(hs, 1).float().cpu()  # [b, 29, seq, d]
        for b, i in enumerate(idx):
            ids = rows[i]["ids"]
            closes = [p for p, tid in enumerate(ids) if tid == close]
            pos = closes[1] - 1 + (width - len(ids))  # closes[0] ends the system turn, closes[1] ends the user turn
            t_inst[i] = hs[b, :, pos].numpy().astype(np.float16)
            t_post[i] = hs[b, :, -5:].numpy().astype(np.float16)
        print(f"  {min(start + BATCH, len(order))}/{len(order)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "activations.npz", t_inst=np.stack(t_inst), t_post=np.stack(t_post))
    write_jsonl(OUT / "activation_rows.jsonl", [{k: v for k, v in r.items() if k not in ("ids", "messages")} for r in rows])
    print(f"{len(rows)} prompts -> {OUT}")


if __name__ == "__main__":
    main()
