# Phase 2a: refusal-direction diagnosis

**Date:** 2026-11-13. **Pre-registration:** `docs/prereg/2026-11-12_v2_refusal_direction.md`. **Code:** `scripts/v2/refusal_direction.py`.
**Outputs:** `results/v2/direction/direction_results.json`, `direction_meta.json` (numbers only; activations file kept locally).

Qwen3-8B, one A40, about 10 minutes. 4,911 prompts. The direction is fitted on English only (391 Aya harmful, 190 PolyGuard harmless).

## Is it a refusal direction? Yes

- Refusal score s uses 3 refusal and 15 non-refusal first tokens (unigram rule; no fallback needed).
- Layer sweep (ablate r̂_l everywhere, mean s on fit harmful): lowest at **layer 14** (s 4.37 → −3.11).
- **Validity check passes.** On the 391 held-out English harmful prompts, ablation lowers mean s from 4.53 to −3.12. That is 64% of the harmful–harmless
  gap (11.99), above the 50% threshold.
- Adding the direction at layer 14 to harmless English prompts raises s only a little (−7.70 → −5.84), so this direction is needed for refusal more clearly
  than it is sufficient for it.
- English split-half cosine of the direction at layer 14: 0.975.

## Readouts at layer 14

Mean projection onto the English refusal direction:

| | English harmful | Target harmful | English harmless | Target harmless |
|---|---:|---:|---:|---:|
| Spanish (391 Aya pairs; PolyGuard control) | 5.26 | 4.32 | −2.83 | −0.41 |
| Hindi, all tier A (1,174 pairs) | 5.17 | 5.34 | −2.83 | 2.80 |
| Hindi, Aya only (914) | 6.36 | 5.90 | −2.83 | 2.80 |
| Hindi, IndicSafe only (260) | 1.00 | 3.36 | −2.83 | 2.80 |

| | Recognition ratio ρ (95% CI) | AUROC target / English | Paired shift, harmful (target − English) | Cosine with English direction |
|---|---|---|---|---|
| Spanish | 0.58 (0.57–0.61) | 0.97 / 0.98 | −0.94 (−1.12 to −0.77), 12% of the English gap | 0.78 |
| Hindi, all | 0.32 (0.30–0.33) | 0.88 / 0.96 | +0.17 (0.02 to 0.30), 2% | 0.56 |
| Hindi, Aya | 0.34 (0.32–0.35) | 0.95 / 0.99 | −0.46 (−0.60 to −0.32), 5% | |
| Hindi, IndicSafe | 0.15 (0.10–0.19) | 0.63 / 0.84 | +2.36 | |

Reliability ceiling for the cosine (English split halves): 0.975. Per-layer curves are in the results file. ρ for Hindi rises from about 0.33 in layers
4–16 to 0.73 in layers 19–21, then falls again; Spanish follows the same shape (about 0.6, then 0.93).

**Link to behaviour** (harmful pairs refused in English; projection in the target language at layer 14):

| | Refused in target | Not refused in target | Ratio | Difference (95% CI) |
|---|---|---|---|---|
| Spanish (both graders agree) | 4.66 (n 241) | 3.96 (n 43) | 0.85 | −0.70 (−1.34 to −0.06) |
| Hindi calibration (≥3 of 4 graders, provisional) | 5.05 (n 58) | 4.42 (n 10) | 0.88 | −0.62 (−1.67 to 0.56) |

## Pre-registered reading

- **Hindi: recognition weakened** (ρ below 0.5).
- **Spanish: mixed.** ρ is between 0.5 and 0.8. Pairs not refused in Spanish project lower than refused ones (interval below 0), but within the 20% tolerance (ratio 0.85).

## What drives the low ratio (descriptive, not pre-registered)

The ratio compares harmful with harmless prompts. In Hindi, almost all of its drop comes from the **harmless** side:

- **Harmful Hindi requests reach the English level** on the refusal direction: +0.17 for all of tier A, −0.46 for Aya alone. The signal is about as strong
  as in English.
- **Harmless Hindi requests sit far higher than harmless English ones:** 2.80 against −2.83. In Spanish the shift is smaller: −0.41.
- So in Hindi the direction separates harmful from harmless less well (AUROC 0.88 against 0.96). The cause is everything Hindi being pushed toward
  "refuse", not harmful Hindi being pushed away from it. The Hindi harmful–harmless direction also points elsewhere (cosine 0.56).

Two explanations are open:

1. **Language offset.** Hindi input moves the residual stream along this direction by itself.
2. **Source mismatch.** The harmless set (PolyGuard) is machine-translated, while the Aya harmful set was written natively. The IndicSafe rows also show that
   source matters: their English versions project unusually low (1.00).

The harmless Hindi replies are not graded yet. If Hindi harmless requests are also refused more often, that would support the first explanation.

Read with the behavioural result, the picture so far is:

- The model registers harmful Hindi requests about as strongly as English ones at layer 14.
- It also registers harmless Hindi requests as partly refusal-like.
- Even so, it refuses harmful Hindi requests less explicitly.

The difference therefore seems to arise downstream of this signal, in how the decision turns into a reply, which would mean routing. This conflicts with the
pre-registered label, which is driven by the ratio. Both are reported. The pre-registered label stands as the formal reading.

## Limits

- **The refusal score s is built from English first tokens.** It is not comparable across languages (for example, Spanish harmful s is −9.45 because
  Spanish refusals start with Spanish tokens). It is used only for the English layer choice and the validity check.
- **Harmful and harmless prompts come from different sources** in each language.
- **The Hindi behaviour link is small** (10 not-refused pairs) and its grades are not yet validated.

## Next

- Grade the Hindi harmless replies (part of tier A grading). That tests the offset explanation directly.
- A harmful–harmless set from the same source in all three languages would remove the source confound.
- Phase 2b (proposed): an explicit-versus-soft refusal direction, to test whether the Hindi difference lies in how the refusal is phrased.
