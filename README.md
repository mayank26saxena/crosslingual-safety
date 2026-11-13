# Cross-lingual safety

Why does Qwen3-8B refuse harmful requests less often in Hindi than in English, and can a targeted change to the model close
that gap without making it refuse harmless Hindi requests or behave worse in English?

Hindi is the development language. Spanish is a high-resource comparison, and Bengali is held out to test whether a repair
transfers. Every phase is pre-registered before it runs.

**Status (2026-11-13):** the grader gate has passed (Phase 0). The Hindi baseline (Phase 1) is being graded. The
refusal-direction diagnosis (Phase 2a) is done. See `docs/plan/2026-11-10_restart.md` for the plan, and `docs/results/` for
results.

## Layout

| Path | Contents |
|---|---|
| `docs/plan/` | Plan, dataset specification, reviews |
| `docs/prereg/` | Dated pre-registrations; later changes are dated addenda |
| `docs/protocols/` | Grading rubric and procedures |
| `docs/results/` | Dated results, including failed gates and exploratory analyses |
| `scripts/v2/` | Current pipeline: dataset collection, exclusions, generation, grading folders, grader gate, refusal direction |
| `scripts/` | Earlier (v1) scripts, kept for reference |
| `data/v2/` | Current prompts (not tracked) and dataset summaries (tracked) |
| `data/external/` | Upstream datasets, downloaded locally (not tracked) |
| `results/v2/` | Current outputs: summaries tracked, replies not tracked |
| `results/` (other folders) | Earlier v1 outputs; `results/v1_pod/` holds v1 pod copies (not tracked) |
| `tools/grader.html` | Local page for human grading |
| `notes/` | Working files, not tracked: grading folders, human grades, references, and `notes/archive/` for v1 material |

## Generation setup

Qwen/Qwen3-8B, Hugging Face transformers in bfloat16. Messages: system "You are a helpful assistant." and the user request,
with Qwen's chat template, `enable_thinking=False`, greedy decoding, one reply per request. New-token limit: 1,024 for Hindi
and Bengali, 512 for English and Spanish.

## Setup

```bash
pip install uv && uv venv --python 3.12 && uv pip install -r pyproject.toml
```

## Handling of harmful content

Raw model replies, activation caches and harmful-request text are not tracked. Reports cite items by id and category.
