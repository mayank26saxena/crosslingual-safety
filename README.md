# Cross-lingual safety

Why do open chat models help with harmful requests more often in Bengali than in English, and can an activation-level
intervention close that gap without retraining and without increasing refusal of harmless Bengali requests?

On Qwen2.5-7B-Instruct, across 150 MultiJail requests, 31.3% of Bengali replies are graded unsafe against 12.0% of English
replies, while only 6.2% of Bengali replies fail to engage. This project diagnoses that gap before attempting to fix it.

**Status:** pre-registration for the evaluation and diagnostic steps is in draft. No new experiments have run.

## Contents

| Path | Contents |
|---|---|
| `docs/design.md` | Questions, plan, data discipline, pass rule, related work |
| `docs/prereg/` | Dated pre-registrations, with later changes as dated addenda |
| `docs/results/` | Dated results, including failed gates and exploratory analyses |
| `data/README.md` | Dataset sources, licenses, and how to rebuild local files |
| `data/splits/splits.json` | Fixed development and confirmation splits (ids only) |
| `scripts/` | Rendering, translation, judging, splits, baseline summary |
| `results/baseline_summary.json` | Aggregate baseline numbers (no request or reply text) |

## Setup

```bash
pip install uv && uv venv --python 3.12 && uv pip install -r pyproject.toml
```

## Reproduce the baseline summary and splits

These need the MultiJail file (see `data/README.md`) and the earlier raw replies, which are kept out of version control.

```bash
python scripts/make_splits.py
python scripts/summarize_baseline.py
```

## Handling of harmful content

Raw model replies, activation caches and harmful-request text are not tracked. Reports cite items by id and category.
