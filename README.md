# Beyond refusal rates: cross-lingual hedging, compliance and the limits of steering in Hindi

Why do open chat models refuse harmful requests less often in Hindi than in English, and can a targeted change to the model's activations
close that gap without causing over-refusal or harming general ability?

We study **Qwen3-8B** and **Llama-3.1-8B-Instruct** on paired requests (the same request in English and in Hindi), with Spanish as a
high-resource comparison and Bengali held out for confirmation. The steering and neuron studies were pre-registered before they ran; the
studies of the internal directions were registered as exploratory, and the same-item and per-layer statistics were added afterwards.
Settings were tuned on development items and tested once on held-out items, and every deviation is recorded in a dated addendum.

**Status (2026-11-24):** all experiments are complete. The paper is in `docs/paper/latex/` (PDF: `beyond-refusal-rates.pdf`; an arXiv
link will be added here), and the plain-language blog post is at <https://mayank26saxena.github.io/blog/cross-lingual-safety-gap/>.

## Findings (held-out, 404 harmful and 169 harmless pairs)

- **The gap** (a premise that earlier work established). Explicit refusal of harmful requests falls from 51% (English) to 19% (Hindi) in
  Qwen, and from 89% to 47% in Llama.
- **What replaces the refusal.** Qwen's lost refusals are mostly *soft*: warnings or redirections without a "no" (115 of 148). Llama's
  split between compliance and soft refusal (88 and 75 of 172), and Llama complies with Hindi harmful requests 1.4 to 2.0 times as often
  as Qwen. The judge's errors on the soft-versus-comply boundary would enlarge the contrast, so the difference in rate is better supported
  than the difference in kind.
- **Internal signatures (exploratory).** On an English-fitted refusal direction, Qwen's Hindi harmful prompts project at English strength,
  Llama's at 28% of it, but in both the harmful–harmless order survives. A separate explicit-versus-soft direction places Hindi prompts
  toward the soft end; it overlaps with the refusal direction (cosine 0.37 in Qwen, 0.64 in Llama at the same layer).
- **Repairs (gated activation steering, pre-registered).** At the registered thresholds neither gate was language-selective.
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
| `docs/paper/` | The paper (`latex/`: `main.tex`, `refs.bib` and the PDF; build with `python3 -I scripts/build_paper.py`), static figures (`figures/`), related-work review (`reports/`), earlier Markdown drafts (`archive/`) |
| `docs/blog/` | Blog post (`index.html`, interactive SVG figures from `data/figures.json`; serve with `python3 -m http.server -d docs/blog`) |
| `docs/plan/`, `docs/protocols/` | Plan, dataset specification, grading rubric and procedures |
| `scripts/v2/` | Current pipeline (data, generation, directions, steering, grading folders, analysis) |
| `scripts/` | Earlier (v1) scripts, kept for reference |
| `results/v2/` | Numeric outputs and logs; `results/v2/labels/` holds every human and model label, keyed by item id |
| `tools/` | Local pages for human grading (`grader.html`) and disagreement review (`review_disagreements.html`) |
| `LICENSE` | MIT licence for the code (datasets keep their upstream licences; see `data/README.md`) |
| `REPRODUCE.md` | The command behind every reported number |
| `MANIFEST.sha256` | Checksums of the untracked inputs and outputs |

## Reproducing

See `REPRODUCE.md`. Models run on one 48 GB GPU in a pinned environment (torch 2.8.0, transformers 5.22.2); exact revisions and
environments are in each run's `env.txt`. Analysis scripts run locally from the id-only label tables.

```bash
pip install uv && uv venv --python 3.12 && uv pip install -r pyproject.toml
```

## Citing

Until the arXiv version is posted, please cite the blog post and this repository.

> Saxena, M. (2026). *Beyond refusal rates: cross-lingual hedging, compliance, and the limits of steering.* Blog post: https://mayank26saxena.github.io/blog/cross-lingual-safety-gap/

## Handling of harmful content

Harmful-request text, raw model replies and activation caches are **not** in this repository. Datasets are fetched from their upstream
sources, and results and labels cite items by id only. The one exception is the blog's three example transcripts
(`docs/blog/data/examples.json`): requests are summarised at category level, never quoted, and the replies are short excerpts (Hindi
given both in translation and in the original, redacted in the same places) with anything usable for harm replaced by a description of
what was removed. The untracked files are listed with checksums in `MANIFEST.sha256`, so a holder of
the archive can verify they match.
