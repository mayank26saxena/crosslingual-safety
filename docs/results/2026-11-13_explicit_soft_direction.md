# Phase 2b: explicit-versus-soft refusal direction

**Date:** 2026-11-13. **Pre-registration:** `docs/prereg/2026-11-13_v2_explicit_soft_direction.md`. **Code:** `scripts/v2/explicit_soft.py`,
`scripts/v2/build_steer_grading.py`, `scripts/v2/steer_analysis.py`. **Outputs:** `results/v2/es_dir/es_dir_results.json`, `steer_results.json`
(numbers only; activations and replies kept locally).

Qwen3-8B on one RTX A6000 (Runpod, about 40 minutes, about $0.50; pod terminated after checksums matched). 3,555 prompts for activations; 750 steered
Hindi replies, graded blind by claude-sonnet-5.5 (15 subagents, all self-reported `claude-sonnet-5-5`, 750 rows in order, none DECLINED).

## The direction

- Fit on English replies of the Aya Spanish set where both graders agree: 367 explicit, 200 soft refusals.
- Layer chosen by five-fold cross-validated AUROC: **layer 24, AUROC 0.93.** The validity bar was 0.75, so it passes. Split-half cosine 0.96.
- Transfer to tier A (no fitting): explicit against soft AUROC **0.90 in English, 0.76 in Hindi.**

## Do Hindi prompts sit toward the soft end?

- Paired shift, Hindi minus English, over 1,069 tier A harmful pairs: **−10.9** (95% CI −12.1 to −9.6), which is **22% of the English gap** between
  explicit and soft (49.2).
- **Pre-registered reading: not met.** The shift is negative and its interval excludes 0, but it is below the 25% threshold.
- **Outcome link** (pairs explicit in English; mean Hindi projection by Hindi outcome): explicit −11.6 (n 127), soft −28.0 (n 292), broken −40.0 (n 42),
  complied −43.5 (n 60). The further toward soft a Hindi prompt sits, the further its reply falls from an explicit refusal.

## Steering: push Hindi prompts toward explicit

150 tier A harmful pairs refused explicitly in English but not in Hindi (steering set), and 100 PolyGuard harmless Hindi prompts (control). The English
explicit-minus-soft difference vector at layer 24 is added at every position, at 1× and 2×.

| Steering set (150) | α = 0 | α = 1 | α = 2 |
|---|---:|---:|---:|
| Explicit refusal | 10.0% | **52.7%** | 77.3% |
| Any refusal | 71.3% | 78.7% | 82.7% |
| Complied | 31 | 17 | 3 |
| Broken | 8.0% | 10.0% | 15.3% |

| Harmless control (100) | α = 0 | α = 1 | α = 2 |
|---|---:|---:|---:|
| Any refusal | 2% | 2% | 12% |
| Broken | 9% | 8% | 24% |

Differences against α = 0 (paired, 95% CI):
- **α = 1:** explicit +42.7 points (34.0 to 51.3); harmless refusal +0.0; broken +2.0 (−3.3 to 7.3).
- **α = 2:** explicit +67.3 (59.3 to 75.3); harmless refusal +10.0 (5 to 16); broken +7.3 (0.7 to 14.0).

**Pre-registered reading: causal support.** At α = 1, all three conditions hold: explicit refusal rises by at least 15 points, harmless refusal rises by at
most 5, and broken replies rise by at most 10. At α = 2, explicit refusal rises further, but harmless refusal rises by 10 points, which fails the second
condition.

## Reading

1. The model carries a direction at the end of the prompt that predicts whether it will refuse explicitly or softly. The direction was found in English and
   transfers to Hindi.
2. Hindi prompts sit measurably toward the soft end, but by less than the pre-registered 25% of the English gap. The prompt-level shift explains part of
   the Hindi drop in explicit refusal, not all of it.
3. Adding the English direction at 1× turns most soft Hindi refusals into explicit ones (10% to 53%) and halves compliance (31 to 17), with no change on
   harmless Hindi prompts. At 2× it starts to over-refuse and break harmless replies.
4. With Phase 2a (harm is registered in Hindi at English strength), this points to a repair that acts on how the refusal is expressed, not on harm
   recognition. A 1× addition at layer 24 is a candidate for Phase 3, to be gated and tested on the split.

## Limits

- One direction strength scale (1× and 2× of the fit difference); no finer sweep.
- Steering set chosen by the selected grader's labels; α = 0 was regenerated in the same run (15 of 150 explicit at α = 0, from batch-level
  nondeterminism), and differences use that run.
- Raw tier A, before screens and split; the 150 steering items are assigned to the development split.
- Effects on English replies and on general capability are not measured here.
