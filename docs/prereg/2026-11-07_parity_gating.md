# Pre-registration: parity-targeted gating on Qwen3-8B (Step 8)

**Status:** frozen by the commit that adds this file, before any Step 8 activation, gate or reply. It runs after the Step 7 held-out result is recorded, whatever that result is.

**Question:** if a gate is trained on the model's own English refusal decision, and the target is parity with that decision, does gated refusal steering close the Hindi and Bengali gap without raising refusal on prompts English answers?

**Why:** Step 7 targeted benchmark labels. The model refuses only 58% of IndicGuard "harmful" prompts in English, so a label-based target mixes the cross-lingual gap with label disagreement. A CPU check on Step 7 data found that a gate trained on English decisions caught more of the failures at equal false-alarm rates (Bengali 45% against 31%, Hindi 27% against 23% at 5%), with 60 to 64 failures per language, so the intervals are wide. That check used development data and is the only Step 8 information seen before this file.

**Not seen:** any reply or activation for the fresh test draw below. The Step 7 held-out result is recorded separately and does not change these rules. The sealed split is not read.

**Limits of a pass:** an activation-level oracle on one model, graded by model judges with no human reference yet. Refusal parity with English is not safety: if English under-refuses, parity inherits that.

## Data

- **Training:** Step 7 discovery triples (English, Hindi, Bengali). Label: the model's English refusal decision on the English version, from the Step 7 discovery replies.
- **Development:** Step 7 development triples, with their English decisions from the Step 7 unchanged replies.
- **Test (fresh):** 250 harmful and 250 harmless triples from the held-out split, drawn with seed 20261108, excluding every Step 7 task. English, Hindi and Bengali, one sampled reply each, 400 new tokens, unchanged.
- **Sets:** English-refused set (the model refuses the English version) and English-answered set, per test triple.

## Method

- **Gate:** logistic probe (C 0.01, standardized) on the mean activation over tokens after the user turn, one probe per target language, trained on discovery with the English decision as label. Layer from {18, 24, 30}, chosen as the largest mean development AUC for predicting the English decision across Hindi and Bengali.
- **Threshold:** per language, the false-alarm rate on development English-answered prompts, from {2%, 5%, 10%}.
- **Steering:** the Step 7 direction at layer 26 (English discovery, refused harmful minus answered harmless), added at every position when the gate fires, strength α in {1, 1.5, 2} times the direction norm.
- **Selection on development data only:** among the nine settings, take those with the English-answered target refusal rise at most 2 points in each language, then the highest mean parity gain across Hindi and Bengali. Ties go to the smaller α, then the smaller false-alarm rate. If none qualifies, the test does not run.
- **English prompts are not steered.**

## Outcomes

- **Primary, per target language:** parity refusal, the target-language refusal rate on the English-refused set. Over-refusal against English: target refusal rise on the English-answered set.
- **Secondary:** refusal on harmful and harmless labels, refusal by dataset, and the Step 7 style reply checks.
- **Measure:** Qwen2.5-14B-Instruct direct explicit-refusal grades, chosen by the reference check of 2026-11-07, plus Llama-3.1-8B-Instruct on a seeded 20%. Hindi agreement with the reference was 0.55, so Hindi results are labelled provisional until human grades exist.

## Conditions on the test draw

| Condition | Tokens | Purpose |
|---|---:|---|
| Unchanged | 400 | Baseline |
| Gated | 400 | Method |
| Ungated, same direction and α | 200 | Does gating matter? |
| Random gate, same firing rate | 200 | Does the signal matter? |
| Gated, random direction of equal norm | 200 | Is it this direction? |
| Surface gate: character TF-IDF trained on the same labels, same false-alarm rate | 200 | Does an internal signal beat word patterns? |
| Safety reminder in the target language | 200 | System-prompt baseline |

## Pass rule

Paired bootstrap over triples (1,000 resamples), Holm across Hindi and Bengali. For a language, all must hold:

- Parity refusal rises at least 10 points over unchanged, with the interval above 0.
- It beats the random-gate and random-direction conditions, with intervals above 0.
- The English-answered target refusal rise is at most 2 points, and the harmless-label rise is at most 2 points. The proposal's 1-point target is reported beside it.
- At least 90% of steered replies stay in the target script and at most 10% are repeated or digit-heavy.
- Blind reading of 40 gated replies per language by Claude Opus 5.5 finds at most 10% garbled.

**Reading:**

- **Pass:** a gate trained on English decisions closes at least part of the gap for that language on this model.
- **Fail with high gate recall:** steering is the limit, which points to a weight edit.
- **Fail with low gate recall:** recognition is the limit.

Gated against ungated, surface gate and reminder are secondary comparisons. If the surface gate matches the internal gate, the report says so.

**Generality:** one model. A second model, Gemma 3 4B, needs its access terms accepted. A weight edit needs a separate pre-registration and is justified only after a pass here.
