"""Regenerate the 150 Phase 0 calibration replies at 1,024 new tokens (2026-11-11 addendum, four-way labels).

Same items, prompts (system prompt, user message, thinking off) and item ids as the 400-token sheet; only the reply changes.
Decoding is greedy, as the dataset specification sets for all v2 behaviour labels.

Stages:
  python scripts/v2/regen_calibration.py prompts            local: writes notes/hindi_calibration/regen_prompts.jsonl
  python regen_calibration.py generate IN OUT [--limit N]    pod: greedy replies, resumable, one JSON line per item
  python scripts/v2/regen_calibration.py sheet REPLIES      local: writes notes/hindi_calibration/hindi_calibration_sheet_1024.csv

The generate stage imports nothing from the repository, so it runs from a single copied file. Prompt and reply text is never printed.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "notes/hindi_calibration"
MODEL = "Qwen/Qwen3-8B"
MAX_NEW = 1024
COLUMNS = [
    "item", "request_english", "message_hindi", "reply_hindi",
    "HUMAN_verdict (REFUSE/ANSWER)", "HUMAN_confidence (sure/unsure)", "HUMAN_notes",
    "OPUS_verdict", "OPUS_reason", "FABLE_verdict", "FABLE_reason",
]


def prompts():
    key = json.loads((CAL / "key.json").read_text())
    want = {v["source_id"]: k for k, v in key.items()}
    rows = {}
    for line in open(ROOT / "data/rendered/gated_items.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["id"] in want:
            rows[want[d["id"]]] = dict(item=want[d["id"]], source_id=d["id"], hidden_fact=d["hidden_fact"], messages=d["messages"])
    if len(rows) != 150:
        raise SystemExit(f"found {len(rows)} of 150 items")
    with open(CAL / "regen_prompts.jsonl", "w", encoding="utf-8") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    print("wrote 150 prompts")


def generate(src, dst, limit=None, batch=16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = [json.loads(l) for l in open(src, encoding="utf-8")][:limit]
    done = set()
    if Path(dst).exists():
        done = {json.loads(l)["item"] for l in open(dst, encoding="utf-8")}
    todo = [r for r in items if r["item"] not in done]
    print(f"{len(done)} done, {len(todo)} to generate", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    texts = {r["item"]: tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in todo}
    todo.sort(key=lambda r: -len(texts[r["item"]]))
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    with open(dst, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), batch):
            group = todo[i:i + batch]
            enc = tok([texts[r["item"]] for r in group], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id)
            for b, r in enumerate(group):
                new = gen[b, enc["input_ids"].shape[1]:].tolist()
                ended = any(t in eos for t in new)
                if ended:
                    new = new[:min(j for j, t in enumerate(new) if t in eos)]
                reply = tok.decode(new, skip_special_tokens=True)
                out.write(json.dumps(dict(item=r["item"], source_id=r["source_id"], model=MODEL, decoding="greedy", max_new_tokens=MAX_NEW,
                                          n_new_tokens=len(new), hit_limit=not ended, prompt_sha256=hashlib.sha256(texts[r["item"]].encode()).hexdigest(),
                                          reply=reply), ensure_ascii=False) + "\n")
            out.flush()
            print(f"{min(i + batch, len(todo))}/{len(todo)}", flush=True)


def sheet(replies):
    reps = {json.loads(l)["item"]: json.loads(l) for l in open(replies, encoding="utf-8")}
    prom = {json.loads(l)["item"]: json.loads(l) for l in open(CAL / "regen_prompts.jsonl", encoding="utf-8")}
    if set(reps) != set(prom):
        raise SystemExit(f"replies cover {len(reps)} of {len(prom)} items")
    with open(CAL / "hindi_calibration_sheet_1024.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for k in sorted(prom):
            user = next(m["content"] for m in prom[k]["messages"] if m["role"] == "user")
            w.writerow([k, prom[k]["hidden_fact"], user, reps[k]["reply"]] + [""] * (len(COLUMNS) - 4))
    n = len(reps)
    hit = sum(r["hit_limit"] for r in reps.values())
    toks = sorted(r["n_new_tokens"] for r in reps.values())
    print(f"{n} items; hit the 1,024-token limit: {hit} ({100 * hit / n:.1f}%); median new tokens {toks[n // 2]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prompts":
        prompts()
    elif cmd == "generate":
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        generate(sys.argv[2], sys.argv[3], lim)
    elif cmd == "sheet":
        sheet(sys.argv[2])
