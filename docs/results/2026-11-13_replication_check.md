# Replication check: baseline replies regenerated under the pinned environment (2026-11-13)

**Purpose** (`docs/plan/2026-11-10_restart.md`, addendum 2026-11-13): the 2,948 tier A and PolyGuard baseline replies (1,474 Hindi, 1,474
English) were generated on 2026-11-11 before the model revision and library versions were pinned. They were regenerated once with the same
script, prompts and settings (greedy, thinking off, system prompt "You are a helpful assistant.", Hindi 1,024 / English 512 new tokens,
batch 16) under the pinned environment, and compared token by token.

**Environment** (`results/v2/replication/env.txt`): Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, Python 3.12.3,
torch 2.8.0+cu128, transformers 5.22.2, accelerate 1.15.0, tokenizers 0.23.3, one NVIDIA A40.

**Command:** `python replication_check.py ARCHIVED REGENERATED OUT` (`scripts/v2/replication_check.py`), per language.

| | Hindi | English |
|---|---|---|
| Replies | 1,474 | 1,474 |
| Identical text | 83 (5.6%) | 416 (28.2%) |
| First 64 tokens identical | 1,088 (73.8%) | 847 (57.5%) |
| Median token of first difference (differing replies) | 134 | 49.5 |
| At the token limit, archived / regenerated | 829 / 845 | 417 / 412 |

## Reading

- The replies are **not bit-reproducible**, as expected for batched bfloat16 greedy decoding (`REPRODUCE.md`, Determinism). Most replies
  share their opening and drift later; the share that reaches the token limit is unchanged within about 1 point in each language.
- This check measures text, not labels. The paper's numbers are computed from the archived replies (checksummed in `MANIFEST.sha256`) and
  their archived labels, so they reproduce exactly from the archive. Whether a fresh generation would give the same refusal rates is a
  separate question that only grading the regenerated replies answers; it has not been run.
- Consequence for later phases (already the rule): every comparison between conditions generates all conditions in one run with
  identical batches.

Outputs: `results/v2/replication/repl_hi.json`, `repl_en.json`, `env.txt`, logs (tracked); `rep_hi.jsonl`, `rep_en.jsonl` (reply text, not
tracked; checksums in the manifest).
