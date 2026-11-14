"""Phase 4 general-ability sample in Bengali: the same 300 Global-MMLU ids as Phase 3 (results/v2/phase3/mmlu_ids.json).

Source: CohereLabs/Global-MMLU, revision 0e619dbeb34206cd48705a1a0ea7fb21cae09993, bn/test-00000-of-00001.parquet in
data/external/global_mmlu/ (sha256 in MANIFEST.sha256). Prompt format identical to scripts/v2/mmlu_sample.py.
  uv run --no-project --with pyarrow==18.1.0 python -I scripts/v2/phase4_mmlu_bn.py
Writes data/v2/phase4_mmlu_bn.jsonl and results/v2/phase4/mmlu_bn_ids.json. An id missing in Bengali, or whose Bengali answer differs from
the English one, is reported and left out.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Answer with the letter of the correct option only."


def main():
    import pyarrow.parquet as pq
    ids = json.load(open(ROOT / "results/v2/phase3/mmlu_ids.json"))["ids"]
    tab = {l: {r["sample_id"]: r for r in pq.read_table(ROOT / f"data/external/global_mmlu/{l}/test-00000-of-00001.parquet").to_pylist()}
           for l in ("bn", "en")}
    keep, missing, mismatch = [], [], []
    for i in ids:
        if i not in tab["bn"]:
            missing.append(i)
        elif tab["bn"][i]["answer"] != tab["en"][i]["answer"]:
            mismatch.append(i)
        else:
            keep.append(i)
    with open(ROOT / "data/v2/phase4_mmlu_bn.jsonl", "w", encoding="utf-8") as f:
        for i in keep:
            r = tab["bn"][i]
            text = f"{r['question']}\n\nA. {r['option_a']}\nB. {r['option_b']}\nC. {r['option_c']}\nD. {r['option_d']}\n\n{INSTRUCTION}"
            f.write(json.dumps(dict(id=i, lang="bn", answer=r["answer"], subject=r["subject"],
                                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]),
                               ensure_ascii=False) + "\n")
    out = ROOT / "results/v2/phase4"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(source_ids="results/v2/phase3/mmlu_ids.json", kept=keep, missing=missing, answer_mismatch=mismatch),
              open(out / "mmlu_bn_ids.json", "w"), indent=1)
    print("kept", len(keep), "missing", len(missing), "answer mismatch", len(mismatch))


if __name__ == "__main__":
    main()
