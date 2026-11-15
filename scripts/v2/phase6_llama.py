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


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prepA":
        prepA()
    elif a[0] == "genA":
        genA(a[1], a[2], smoke="--smoke" in a)
    elif a[0] == "folderA":
        folderA(a[1])
