"""Phase 3 general-ability sample: 300 Global-MMLU test questions, the same ids in Hindi and English.

Source: CohereLabs/Global-MMLU, revision 0e619dbeb34206cd48705a1a0ea7fb21cae09993, files {hi,en}/test-00000-of-00001.parquet in
data/external/global_mmlu/ (sha256 in MANIFEST.sha256). Needs pyarrow:
  uv run --no-project --with pyarrow==18.1.0 python -I scripts/v2/mmlu_sample.py
Writes data/v2/phase3_mmlu.jsonl (600 rows) and results/v2/phase3/mmlu_ids.json. Scoring (first A-D letter in the
reply; none counts as wrong) is done in the Phase 3 analysis.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261116
N = 300
SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Answer with the letter of the correct option only."


def sample():
    import pyarrow.parquet as pq
    tab = {l: {r["sample_id"]: r for r in pq.read_table(ROOT / f"data/external/global_mmlu/{l}/test-00000-of-00001.parquet").to_pylist()}
           for l in ("hi", "en")}
    common = sorted(i for i in tab["en"] if i in tab["hi"] and tab["en"][i]["answer"] == tab["hi"][i]["answer"])
    ids = sorted(random.Random(SEED).sample(common, N))
    with open(ROOT / "data/v2/phase3_mmlu.jsonl", "w", encoding="utf-8") as f:
        for i in ids:
            for l in ("hi", "en"):
                r = tab[l][i]
                text = (f"{r['question']}\n\nA. {r['option_a']}\nB. {r['option_b']}\nC. {r['option_c']}\nD. {r['option_d']}\n\n{INSTRUCTION}")
                f.write(json.dumps(dict(id=i, lang=l, answer=r["answer"], subject=r["subject"],
                                        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]),
                                   ensure_ascii=False) + "\n")
    out = ROOT / "results/v2/phase3"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(seed=SEED, n_common=len(common), ids=ids), open(out / "mmlu_ids.json", "w"), indent=1)
    print("common ids", len(common), "sampled", len(ids))


if __name__ == "__main__":
    sample()
