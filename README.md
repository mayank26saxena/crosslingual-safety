# Cross-lingual safety: the Hindi refusal gap in two open chat models

Why do open chat models refuse harmful requests less often in Hindi than in English, and can a targeted change to the model's activations
close that gap without causing over-refusal or harming general ability?

We study **Qwen3-8B** and **Llama-3.1-8B-Instruct** on paired requests (the same request in English and in Hindi), with Spanish as a
high-resource comparison and Bengali held out for confirmation. Every study was pre-registered before it ran. Settings were tuned on
development items and tested once on held-out items, and every deviation is recorded in a dated addendum.

**Status (2026-11-17):** all experiments are complete. A paper draft, a workshop version and a blog draft are in `docs/paper/` and
`docs/blog/`.

## Findings (held-out, 404 harmful and 169 harmless pairs)

- **The gap.** Explicit refusal of harmful requests falls from 51% (English) to 19% (Hindi) in Qwen, and from 89% to 47% in Llama.
- **What replaces the refusal.** Qwen's lost refusals are mostly *soft*: warnings or redirections without a "no". Llama's split between
  soft refusal and compliance, and Llama leaves Hindi harmful requests unrefused more often than Qwen.
- **Internal signatures.** On an English-fitted refusal direction, Qwen's Hindi harmful prompts project at English strength, Llama's at
  28% of it. A separate explicit-versus-soft direction places Hindi prompts toward the soft end.
- **Repairs (gated activation steering).**
  - *Qwen:* explicit refusal +17 in Hindi, but +16 in English, so the gap barely narrows. The repair then failed a sealed Bengali
    confirmation on garbled output.
  - *Llama:* a refusal-direction push with a Hindi-set gate threshold. Explicit refusal +14.6 in Hindi, and the gap narrows by 7.4 points.
    Both narrowly miss the pre-registered bars (+15 and 10).
- **Hindi-associated neurons.** A small set carries most of the internal English–Hindi difference, but shifting it changes English
  behaviour, not Hindi.

**Grading.** An LLM judge (Claude Sonnet 5.5) grades every reply with four labels, chosen against a fluent human Hindi grader. The Llama
human check fell below its pre-registered bar blind (κ 0.643). After the grader re-applied the rubric to the disagreements, agreement was
0.893. Both values are reported (`docs/results/2026-11-16_llama_human_checks.md`).

## Layout

| Path | Contents |
|---|---|
| `docs/prereg/` | Dated pre-registrations; later changes are dated addenda |
| `docs/results/` | One result document per study, including failed gates |
| `docs/2026-11-15_closeout.md` | Exit criteria, study status table, future work |
| `docs/paper/` | Paper draft (`draft.md`), workshop version (`workshop.md`), static figures (`figures/`), related-work review (`reports/`) |
| `docs/blog/` | Blog post (`index.html`, interactive SVG figures from `data/figures.json`; serve with `python3 -m http.server -d docs/blog`) |
| `docs/plan/`, `docs/protocols/` | Plan, dataset specification, grading rubric and procedures |
| `scripts/v2/` | Current pipeline (data, generation, directions, steering, grading folders, analysis) |
| `scripts/` | Earlier (v1) scripts, kept for reference |
| `results/v2/` | Numeric outputs and logs; `results/v2/labels/` holds every human and model label, keyed by item id |
| `tools/` | Local pages for human grading (`grader.html`) and disagreement review (`review_disagreements.html`) |
| `REPRODUCE.md` | The command behind every reported number |
| `MANIFEST.sha256` | Checksums of the untracked inputs and outputs |

## Reproducing

See `REPRODUCE.md`. Models run on one 48 GB GPU in a pinned environment (torch 2.8.0, transformers 5.22.2); exact revisions and
environments are in each run's `env.txt`. Analysis scripts run locally from the id-only label tables.

```bash
pip install uv && uv venv --python 3.12 && uv pip install -r pyproject.toml
```

## Handling of harmful content

Harmful-request text, raw model replies and activation caches are **not** in this repository. Datasets are fetched from their upstream
sources, and results and labels cite items by id only. The untracked files are listed with checksums in `MANIFEST.sha256`, so a holder of
the archive can verify they match.
