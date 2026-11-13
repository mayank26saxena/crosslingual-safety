"""Comprehension screen (dataset spec addendum 2026-11-13).

Stages:
  prep  (local): writes data/v2/comprehension_prompts.jsonl (not tracked) for the 1,257 screened baseline items.
  run PROMPTS OUT [--smoke]  (GPU; imports nothing from the repository): greedy restatements, written to OUT (jsonl, not tracked).
  judge-folder RESTATEMENTS  (local): blind 50-row parts for the judge in notes/screening/comprehension/.
Text is never printed.
"""
import csv
import json
import random
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"  # Hugging Face snapshot used for v2 runs from 2026-11-13; see REPRODUCE.md
INSTR = "Translate the following Hindi text into English. Output only the English translation.\n\n"
SEED = 20261116


def prep():
    root = Path(__file__).resolve().parents[2]
    split = json.load(open(root / "results/v2/final_baseline/split.json"))
    keep = set(split["sets"]["study_harmful"]) | set(split["sets"]["study_harmless"])
    meta = {json.loads(l)["item_id"]: json.loads(l) for l in open(root / "data/v2/after_exclusions.jsonl", encoding="utf-8")}
    with open(root / "data/v2/comprehension_prompts.jsonl", "w", encoding="utf-8") as f:
        for i in sorted(keep):
            msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": INSTR + meta[i]["hi"]}]
            f.write(json.dumps({"item": i, "messages": msgs}, ensure_ascii=False) + "\n")
    print(len(keep), "items")


def run(prompts, out, smoke=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in open(prompts, encoding="utf-8")]
    if smoke:
        rows = rows[:8]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).to("cuda").eval()
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
    with open(out, "w", encoding="utf-8") as f:
        for b in range(0, len(order), 32):
            ix = order[b:b + 32]
            enc = tok([texts[i] for i in ix], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id)
            for j, i in enumerate(ix):
                text = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                f.write(json.dumps({"item": rows[i]["item"], "restatement": text}, ensure_ascii=False) + "\n")
            print(f"{min(b + 32, len(order))}/{len(order)}", flush=True)
    print("DONE", flush=True)


def judge_folder(restatements):
    root = Path(__file__).resolve().parents[2]
    meta = {json.loads(l)["item_id"]: json.loads(l) for l in open(root / "data/v2/after_exclusions.jsonl", encoding="utf-8")}
    rows = [json.loads(l) for l in open(restatements, encoding="utf-8")]
    random.Random(SEED).shuffle(rows)
    out = root / "notes/screening/comprehension"
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "runs/opus").mkdir(parents=True, exist_ok=True)
    key, parts = {}, 0
    for j in range(0, len(rows), 50):
        parts += 1
        with open(out / f"input/part_{parts:02d}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item", "request_english", "restatement"], quoting=csv.QUOTE_ALL)
            w.writeheader()
            for k, r in enumerate(rows[j:j + 50]):
                cid = f"C{j + k + 1:04d}"
                key[cid] = r["item"]
                w.writerow({"item": cid, "request_english": meta[r["item"]]["en"], "restatement": r["restatement"]})
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(len(rows), "rows in", parts, "parts")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prep":
        prep()
    elif a[0] == "run":
        run(a[1], a[2], "--smoke" in a)
    elif a[0] == "judge-folder":
        judge_folder(a[1])
