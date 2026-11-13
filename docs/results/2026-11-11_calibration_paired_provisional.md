# Provisional: paired English and Hindi refusal on the 150 calibration requests

**Date:** 2026-11-11. **Status:** exploratory and provisional. No grader has passed the Phase 0 gate yet (human Hindi grades pending), and the 150 requests
were stratified by an earlier judge's verdicts, so rates here are not estimates for any population. The size of the gap is measured on tier A.

## Setup

- Qwen3-8B, greedy, system prompt "You are a helpful assistant.", thinking off. Hindi replies up to 1,024 new tokens, English up to 512.
- 150 requests (103 harmful, 47 harmless by benchmark label), each asked in Hindi and in English: 300 replies, shuffled under ids that hide pairing.
- Graded with the four-label rubric (`docs/protocols/2026-11-11_hindi_calibration_grading.md`) by isolated Kiro subagents, one per model, memory off.
- **Valid runs:** claude-haiku-5.5, claude-sonnet-5.5, gpt-5.6-terra, gpt-5.6-luna.
- **Excluded:** claude-opus-5.5, because both attempts were served partly by claude-opus-4.8 after the requested model declined; deepseek-3.2, because it
  assigned labels with keyword scripts, against the rubric.
- Agreement between valid runs: kappa 0.67 to 0.81 on four labels, 0.69 to 0.87 on refused against not refused.

## Harmful requests (103 pairs)

| Grader | Explicit refusal, EN | Explicit refusal, HI | Pairs refused explicitly in EN only / HI only | Exact McNemar p | Any refusal, EN | Any refusal, HI | Pairs refused in EN only / HI only | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-5.5 | 57% | 29% | 35 / 6 | 5e-6 | 81% | 74% | 14 / 7 | 0.19 |
| claude-sonnet-5.5 | 54% | 27% | 33 / 5 | 4e-6 | 76% | 62% | 18 / 4 | 0.004 |
| gpt-5.6-terra | 55% | 31% | 32 / 7 | 7e-5 | 68% | 66% | 10 / 8 | 0.81 |
| gpt-5.6-luna | 56% | 27% | 39 / 9 | 2e-5 | 75% | 69% | 13 / 7 | 0.26 |

"Any refusal" is REFUSE_EXPLICIT or REFUSE_SOFT.

## Reading

1. **Explicit refusal roughly halves in Hindi,** with every grader. The same harmful requests that get "I can't help with that" in English mostly do
   not get it in Hindi.
2. **Much of that difference moves to soft refusal, not to compliance.** Counting soft refusals, the gap shrinks to 2 to 14 points and is significant
   for one grader of four. In Hindi the model more often warns, lectures or redirects without saying no.
3. **Harmless requests** are refused explicitly less in Hindi too (about 19% to 6%), so the shift in style is not specific to harmful requests.
4. **Grading Hindi is harder:** the four graders agree (3 of 4) on 120 of 150 Hindi replies against 138 of 150 English replies.

If this holds on tier A with a validated grader, the Hindi failure on this model is mainly a change in how it refuses rather than a large rise in
compliance, and the metric chosen (explicit or any refusal) decides how big the gap looks. Both are reported.

**Reproduced (2026-11-13)** from `results/v2/labels/calibration_*.csv` by `python3 scripts/v2/paired_provisional.py calibration results/v2/provisional/calibration_paired.json`; all figures match.

## Not yet done

- Human Hindi grades, the agreement gate, and the choice of grader.
- Tier A (1,174 human-authored requests): generated, not graded.
