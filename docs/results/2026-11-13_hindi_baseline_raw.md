# Hindi baseline, raw (before screens and split)

**Date:** 2026-11-13. **Rules:** `docs/prereg/2026-11-10_v2_measurement_and_data.md` (tier A baseline and PolyGuard addenda; grading-route addendum).
**Code:** `scripts/v2/baseline_gap.py`. **Output:** `results/v2/hindi_baseline/baseline_raw.json`.

**Status: raw.** The request-type and comprehension screens and the test split are not applied yet. The reported baseline will be
recomputed on the items that pass them.

## Setup

- 1,474 requests, each in Hindi and English: tier A (Aya Red-teaming Hindi 914 harmful; IndicSafe 155 harmful, 62 harmless, 43 ambiguous) and 300
  PolyGuard harmless.
- Qwen3-8B, system prompt "You are a helpful assistant.", thinking off, greedy; 1,024 new tokens for Hindi, 512 for English.
- 2,948 replies, shuffled under blind ids, graded with the four-label rubric by claude-sonnet-5.5 as 59 Claude Code subagents (one 50-row part each).
  All self-reported `claude-sonnet-5-5`; all rows in order with valid labels; none DECLINED. This route passed its bridge check
  (`2026-11-13_phase0_grader_gate.md`).

## Harmful requests

| Set | Pairs | Any refusal, EN → HI | Gap (95% CI) | EN only / HI only | Explicit refusal, EN → HI | Gap (95% CI) | EN only / HI only |
|---|---:|---|---|---|---|---|---|
| **Tier A** | 1,069 | 82.2% → 69.8% | **12.4 pts** (9.7–15.3) | 191 / 58 | 48.7% → 16.9% | **31.8 pts** (28.4–35.2) | 394 / 54 |
| Aya | 914 | 82.0% → 69.6% | 12.4 (9.4–15.4) | 164 / 51 | 51.0% → 16.7% | 34.2 (30.4–37.9) | 356 / 43 |
| IndicSafe | 155 | 83.9% → 71.0% | 12.9 (5.8–20.0) | 27 / 7 | 35.5% → 18.1% | 17.4 (9.0–25.8) | 38 / 11 |

All gaps: exact McNemar p below 0.001.

Labels on tier A harmful, English → Hindi: COMPLY 176 → 214; REFUSE_EXPLICIT 521 → 181; REFUSE_SOFT 358 → 565; BROKEN 14 → 109.

## Harmless requests

| Set | Pairs | Any refusal, EN → HI | EN only / HI only | BROKEN, EN → HI |
|---|---:|---|---|---|
| PolyGuard | 300 | 4.3% → 2.3% | 9 / 3 (p 0.15) | 0 → 25 |
| IndicSafe harmless | 62 | 0% → 0% | 0 / 0 | 0 → 6 |
| IndicSafe ambiguous | 43 | 0% → 0% | 0 / 0 | 0 → 1 |

## Reading

1. **The Hindi gap is mostly a shift from explicit to soft refusal.** Explicit refusal falls by about two thirds (48.7% to 16.9%). Any refusal falls
   by 12 points, because soft refusals rise from 358 to 565.
2. **A real compliance and breakdown component remains.** On tier A harmful requests, 191 pairs are refused in English but not in Hindi (58 the other
   way). COMPLY rises by 38 and BROKEN by 95.
3. **No over-refusal of harmless Hindi requests.** The model refuses harmless requests no more in Hindi than in English (2.3% against 4.3%, not
   significant). The Phase 2a shift of harmless Hindi prompts along the refusal direction is therefore not matched by more refusals in behaviour.
4. **Larger than the Spanish gap on Aya.** Aya Spanish (782 different requests, Kiro grader): explicit 47.3% → 31.7% (−16), any 78.3% → 68.9% (−9). Aya
   Hindi: −34 and −12. Both are native-written Aya prompts, but the request sets and grading route differ.

## Limits

- Raw: screens and split pending.
- 571 of 1,174 tier A Hindi replies and 197 English replies reach the token limit; graded as shown.
- BROKEN is the label the grader and the human disagreed on most (calibration); refusal rates are the measures, BROKEN is reported separately.
