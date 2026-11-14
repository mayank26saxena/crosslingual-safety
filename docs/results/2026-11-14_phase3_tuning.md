# Phase 3 tuning: gated explicit-refusal steering, development split (2026-11-14)

Pre-registration: `docs/prereg/2026-11-13_v2_phase3_gated_steering.md` (frozen before any Phase 3 reply). This is Procedure step 1 only; the
test split has not been touched.

**Setup.** 401 understood harmful and 171 understood harmless Hindi development prompts, generated under no steering and under the gated push
at α = 0.75, 1.0 and 1.25 in one run with identical batches (A40, pinned environment, `results/v2/phase3/tune/env.txt`). Gate thresholds τ are
the 50th, 25th and 10th percentiles of the English development harmful gate scores (5.52, 3.35, 1.23). 2,243 distinct replies graded blind by
claude-sonnet-5.5 subagents (45 parts, every part self-reported `claude-sonnet-5-5`); 45 steered replies identical to their unsteered reply
took its label. No DECLINED labels.

**Commands.** `gated_steering.py tune`; `build_phase3_grading.py results/v2/phase3/tune/tune_replies.jsonl phase3_tune`;
`phase3_tune_analysis.py results/v2/phase3/tune/tune_selection.json`.

## Gate firing (share of prompts above τ)

| | τ 50th | τ 25th | τ 10th |
|---|---|---|---|
| Hindi harmful | 49.6% | 91.8% | 100% |
| Hindi harmless | 1.8% | 26.3% | 93.0% |
| English harmful | 49.9% | 74.8% | 89.8% |
| English harmless | 1.8% | 3.5% | 9.9% |

The layer-14 score separates harmful from harmless well in English and less well in Hindi: harmless Hindi prompts sit about 5.4 points higher
than harmless English ones (mean 2.80 against −2.65), consistent with Phase 2a.

## Outcomes (Hindi, development)

No steering: harmful explicit refusal 17.5%, any refusal 77.6%, BROKEN 4.7%; harmless any refusal 1.8%.

| α | τ | Harmful explicit | Δ | Harmful BROKEN | Δ | Harmless any refusal | Δ | Qualifies |
|---|---|---|---|---|---|---|---|---|
| 0.75 | 50th | 28.4% | +11.0 | 6.0% | +1.2 | 1.8% | +0.0 | yes |
| 0.75 | 25th | 35.2% | +17.7 | 10.5% | +5.7 | 2.3% | +0.6 | no (BROKEN) |
| 0.75 | 10th | 37.2% | +19.7 | 10.7% | +6.0 | 2.3% | +0.6 | no (BROKEN) |
| 1.0 | 50th | 35.4% | +18.0 | 6.0% | +1.2 | 1.8% | +0.0 | yes |
| 1.0 | 25th | 47.4% | +29.9 | 11.0% | +6.2 | 2.3% | +0.6 | no (BROKEN) |
| 1.0 | 10th | 50.1% | +32.7 | 11.2% | +6.5 | 2.9% | +1.2 | no (BROKEN) |
| **1.25** | **50th** | **38.9%** | **+21.4** | **6.7%** | **+2.0** | **1.8%** | **+0.0** | **yes (chosen)** |
| 1.25 | 25th | 56.1% | +38.7 | 11.7% | +7.0 | 2.3% | +0.6 | no (BROKEN) |
| 1.25 | 10th | 59.6% | +42.1 | 12.5% | +7.7 | 2.9% | +1.2 | no (BROKEN) |

Any refusal on harmful prompts stays between 75% and 79% in every setting: the push converts soft refusals into explicit ones rather than
turning compliance into refusal.

## Selection

By the frozen rule, **α = 1.25, τ = 5.52 (50th percentile)** is chosen: the largest rise in Hindi explicit refusal (+21.4 points) among the
three settings that keep harmless refusal within +3 and BROKEN within +5. The constraint that bound was BROKEN, not over-refusal: the looser
gates open on most harmless Hindi prompts but raise harmless refusal by at most 1.2 points, while every looser setting raises BROKEN on
harmful prompts by more than 5 points.

## Notes for the test

- At the chosen τ the gate opens on about half the harmful prompts, so the test effect is bounded by what steering does to that half.
- These are development numbers chosen as the best of nine; the test estimate is expected to be lower.
- Step 2 (test, once): chosen setting against no steering on 404 + 169 test pairs in Hindi and English and 300 Global-MMLU questions per
  language, one run, identical batches, graded blind with all conditions shuffled.
