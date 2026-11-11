"""Regenerate the 150 Phase 0 calibration replies at 1,024 new tokens (2026-11-11 addendum, four-way labels).

Same items, prompts (system prompt, user message, thinking off) and item ids as the 400-token sheet; only the reply changes.
Decoding is greedy, as the dataset specification sets for all v2 behaviour labels.

Stages:
  python scripts/v2/regen_calibration.py prompts            local: writes notes/hindi_calibration/regen_prompts.jsonl
  python regen_calibration.py generate IN OUT [--limit N]    pod: greedy replies, resumable, one JSON line per item
  python scripts/v2/regen_calibration.py sheet REPLIES      local: writes notes/hindi_calibration/hindi_calibration_sheet_1024.csv
  python scripts/v2/regen_calibration.py model_sheet HI EN  local: one shuffled 300-row sheet (Hindi and English) for model graders

English counterparts (exploratory, for the paired English-Hindi comparison): the prompts stage also writes regen_prompts_en.jsonl, the
English version of each item (same task), and `generate ... --max-new 512` produces greedy English replies at the specification's English limit.

The generate stage imports nothing from the repository, so it runs from a single copied file. Prompt and reply text is never printed.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent  # the generate stage runs from a lone copy on the pod
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
            rows[want[d["id"]]] = dict(item=want[d["id"]], source_id=d["id"], task_id=d["task_id"], hidden_fact=d["hidden_fact"], messages=d["messages"])
    if len(rows) != 150:
        raise SystemExit(f"found {len(rows)} of 150 items")
    with open(CAL / "regen_prompts.jsonl", "w", encoding="utf-8") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    print("wrote 150 prompts")
    tasks = {r["source_id"].rsplit("__", 1)[0]: k for k, r in rows.items()}
    en = {}
    for line in open(ROOT / "data/rendered/gated_items.jsonl", encoding="utf-8"):
        d = json.loads(line)
        base = d["id"].rsplit("__", 1)[0]
        if d["lang"] == "en" and base in tasks and d["task_id"] == rows[tasks[base]]["task_id"]:
            en[tasks[base]] = dict(item=tasks[base], source_id=d["id"], hidden_fact=d["hidden_fact"], messages=d["messages"])
    if len(en) != 150:
        raise SystemExit(f"found {len(en)} of 150 English counterparts")
    with open(CAL / "regen_prompts_en.jsonl", "w", encoding="utf-8") as f:
        for k in sorted(en):
            f.write(json.dumps(en[k], ensure_ascii=False) + "\n")
    print("wrote 150 English prompts")


def generate(src, dst, limit=None, batch=16, max_new=MAX_NEW):
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
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id)
            for b, r in enumerate(group):
                new = gen[b, enc["input_ids"].shape[1]:].tolist()
                ended = any(t in eos for t in new)
                if ended:
                    new = new[:min(j for j, t in enumerate(new) if t in eos)]
                reply = tok.decode(new, skip_special_tokens=True)
                out.write(json.dumps(dict(item=r["item"], source_id=r["source_id"], model=MODEL, decoding="greedy", max_new_tokens=max_new,
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


def model_sheet(hi_replies, en_replies):
    """One blind sheet for model graders: 150 Hindi and 150 English replies, shuffled, with ids that hide language pairing."""
    import random
    hi = {json.loads(l)["item"]: json.loads(l) for l in open(hi_replies, encoding="utf-8")}
    en = {json.loads(l)["item"]: json.loads(l) for l in open(en_replies, encoding="utf-8")}
    ph = {json.loads(l)["item"]: json.loads(l) for l in open(CAL / "regen_prompts.jsonl", encoding="utf-8")}
    pe = {json.loads(l)["item"]: json.loads(l) for l in open(CAL / "regen_prompts_en.jsonl", encoding="utf-8")}
    if not (set(hi) == set(en) == set(ph) == set(pe)) or len(hi) != 150:
        raise SystemExit("replies or prompts incomplete")
    rows = [(k, "hi", ph[k], hi[k]) for k in hi] + [(k, "en", pe[k], en[k]) for k in en]
    random.Random(20261112).shuffle(rows)
    key = {}
    with open(CAL / "model_grading_sheet_1024.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "request_english", "message", "reply"])
        for i, (k, lang, p, r) in enumerate(rows, 1):
            rid = f"R{i:03d}"
            user = next(m["content"] for m in p["messages"] if m["role"] == "user")
            w.writerow([rid, p["hidden_fact"], user, r["reply"]])
            key[rid] = dict(pair=k, lang=lang, hit_limit=r["hit_limit"], n_new_tokens=r["n_new_tokens"])
    (CAL / "model_grading_key_1024.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows in the model grading sheet")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prompts":
        prompts()
    elif cmd == "generate":
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        mx = int(sys.argv[sys.argv.index("--max-new") + 1]) if "--max-new" in sys.argv else MAX_NEW
        bs = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 16
        generate(sys.argv[2], sys.argv[3], lim, batch=bs, max_new=mx)
    elif cmd == "sheet":
        sheet(sys.argv[2])
    elif cmd == "model_sheet":
        model_sheet(sys.argv[2], sys.argv[3])
