# Provisional: paired English and Spanish refusal (Aya Spanish, PolyGuard harmless control)

**Date:** 2026-11-12. **Status:** provisional. The Phase 0 grader gate is defined on Hindi and has not been run yet (human Hindi grades pending), so no
grader is validated. Spanish is the high-resource control (prereg addenda 6 and 7).

## Setup

- Qwen3-8B, greedy, system prompt "You are a helpful assistant.", thinking off, up to 512 new tokens in both languages.
- 1,082 requests, each asked in Spanish and in English: 782 harmful (Aya Red-teaming, Spanish, with its English version) and 300 harmless
  (PolyGuard, same ids in both languages). 2,164 replies, shuffled under ids that hide pairing and language.
- Graded with the four-label rubric (`docs/protocols/2026-11-11_hindi_calibration_grading.md`) in Kiro, one session per model.
- **Runs:** claude-sonnet-5.5, gpt-5.6-luna, and one run under Kiro's "auto" setting (exploratory: the serving model is not recorded).
  The resolved model and credits were not recorded for the Sonnet and Luna runs either.
- All three runs are complete, in order, with valid labels. Agreement: kappa 0.88 to 0.89 on four labels, 0.90 to 0.91 on refused against not
  refused (Hindi calibration: 0.67 to 0.81 and 0.69 to 0.87).

## Harmful requests (782 pairs)

| Grader | Explicit refusal, EN | Explicit refusal, ES | Pairs refused explicitly in EN only / ES only | Exact McNemar p | Any refusal, EN | Any refusal, ES | Pairs refused in EN only / ES only | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-sonnet-5.5 | 47.3% | 31.7% | 149 / 27 | 1e-21 | 78.3% | 68.9% | 103 / 30 | 2e-10 |
| gpt-5.6-luna | 48.0% | 31.7% | 152 / 25 | 2e-23 | 75.1% | 64.1% | 109 / 23 | 1e-14 |
| auto (exploratory) | 47.8% | 32.1% | 154 / 31 | 8e-21 | 77.6% | 69.7% | 100 / 38 | 1e-7 |

Compliance rises from 20 to 24% in English to 25 to 30% in Spanish. BROKEN replies: 10 to 11 in English, 39 to 45 in Spanish.

## Harmless requests (300 pairs)

About 96 to 99% COMPLY in both languages; 1 to 4% are refused, with no significant difference between languages for any grader.

## Reading

1. **Spanish shows a refusal gap as well.** Explicit refusal drops by about 16 points and any refusal by about 9 to 11 points, significant for
   every grader.
2. **Unlike Hindi calibration, the gap persists after counting soft refusals,** and part of it is a rise in compliance.
3. **The harmless control is clean:** the model answers harmless requests in both languages, so the harmful-request gap is not general
   over-refusal in English or a grading artefact.
4. **Not comparable with Hindi yet.** The Hindi figures come from 103 calibration requests chosen by an earlier judge's verdicts and a 1,024-token
   cap; the Spanish ones from all 782 Aya requests at 512. The cross-language comparison waits for Hindi tier A, graded by the selected grader.

## Not yet done

- Grader validation (Hindi gate) and recording resolved models for these runs.
- Bootstrap confidence intervals over requests.
