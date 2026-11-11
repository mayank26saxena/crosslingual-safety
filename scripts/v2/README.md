# v2 pipeline

Built from the raw source files in `data/external/`, not from the v1 manifest. Each stage reads the previous stage's output, writes one file in
`data/v2/`, and writes a tracked summary. Each stage runs on a 50-item pilot first, and the pilot output is read before the full run
(`docs/plan/2026-11-11_v1_review_and_guardrails.md`).

| Stage | Script | Output |
|---|---|---|
| 1. Collect candidates from sources, with provenance quoted from each source's documentation | `collect.py` | `data/v2/candidates.jsonl` |
| 2. Exclusions: sealed split, v1 items and groups, duplicates | `exclude.py` | `data/v2/after_exclusions.jsonl` |
| 3. Request-type screen | `screen_request.py` | `data/v2/after_request.jsonl` |
| 4. Translation-fidelity screen | `screen_translation.py` | `data/v2/after_translation.jsonl` |
| 5. Comprehension screen (GPU) | `screen_comprehension.py` | `data/v2/after_comprehension.jsonl` |
| 6. Behaviour labels (GPU) | `label_behaviour.py` | `data/v2/hindi_items.jsonl` |
| 7. Splits and dataset card | `split.py` | `data/v2/summary.json`, `docs/data/hindi_dataset_card.md` |

Specification: `docs/plan/2026-11-11_hindi_dataset_spec.md`.
