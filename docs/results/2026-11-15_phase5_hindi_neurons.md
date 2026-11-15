# Phase 5 result: Hindi-specific MLP neurons (diagnosis)

**Pre-registration:** `docs/prereg/2026-11-15_v2_phase5_hindi_neurons.md` (frozen in 8d6635f). **Analysis:**
`scripts/v2/phase5_neurons.py analyse`; output `results/v2/phase5/phase5_results.json`. **Labels:** `results/v2/labels/phase5_grades.csv`
(ids only; 1,976 graded by Sonnet 5.5, 214 copied from an identical unsteered reply).

## Outcome: small, but no clean behavioural effect

A small set of MLP neurons (100 of 307,200 in layers 0–24) carries much of the English–Hindi difference on the refusal direction.
Shifting them toward their English values moves the internal metric strongly in every language. It changes behaviour only in English.
It does not raise explicit refusal in Hindi.

## Forward test: a small set exists

Recovery is the share of the English–Hindi gap on the layer-24 refusal direction (unit e₂₄, last prompt token) that patching the top-N
neurons to English values closes. It is measured on 79 held-out gap pairs and compared with N random neurons.

| N | 10 | 50 | 100 | 500 | 1,000 | 5,000 |
|---|---|---|---|---|---|---|
| Top-N recovery | 0.16 | 0.49 | 0.75 | 1.76 | 2.52 | 3.67 |
| Random-N recovery | 0.01 | −0.01 | 0.00 | −0.03 | −0.01 | 0.03 |

- Every top-minus-random interval is well above zero. The method works on held-out pairs.
- The smallest N with recovery ≥ 0.5 is N* = 100, so 100 neurons were taken to the behaviour test.
- Above 100 neurons recovery passes 1: patching overshoots the English value on this direction.
- Of the selected neurons' attribution, 20% sits in layers 0–13 and 80% in layers 14–24. By the registered split this is "late".

## Behaviour test: the effect is not Hindi-specific, and it is not in Hindi

A fixed offset (the mean English-minus-Hindi activation of the selected neurons) is added at the aligned prompt positions. Test sets:
322 harmful requests asked in Hindi and in English, and 86 harmless Hindi requests. All have paired replies with the offset and without it,
plus a random-neuron control. Results are percentage-point changes, with paired bootstrap 95% intervals.

| Set | Explicit refusal (base → with offset) | Change | Selected minus random | BROKEN change |
|---|---|---|---|---|
| Harmful, Hindi | 20.8% → 22.0% | +1.2 [−2.5, +5.0] | +1.6 [−2.5, +5.6] | +4.3 [+0.6, +8.1] |
| Harmful, English | 41.3% → 56.5% | **+15.2 [+11.2, +19.3]** | +14.6 [+10.9, +18.6] | +0.3 |
| Harmless, Hindi | 1.2% → 0.0% | −1.2 | −1.2 | +1.2 |

- Any refusal (explicit or soft) on harmful requests: Hindi +1.6, English +2.5. Harmless Hindi: no change.
- Random neurons changed nothing beyond noise. Hindi BROKEN with the random offset on harmless requests was +5.8 [0.0, +11.6] on 86 items.
- Rule check: the Hindi explicit gain (+1.2) is below the +15 threshold, so neither "small and Hindi-specific" nor "small but shared"
  applies. The registered outcome is the last one, "small, no clean behavioural effect". No replies were dropped.

## Internal metric versus behaviour

The offset moved the layer-24 metric by about the same amount in all three languages: +18.8 for Hindi, +17.5 for English and +16.2 for
Bengali, against about 0 for random neurons. So the neurons are not Hindi-specific in what they do to the refusal direction. They push
every language toward it.

The behavioural difference is that the same push changes the reply in English but not in Hindi. One reading, not tested here, is that
Hindi needs a larger push. For context, the Phase 3 steering that did raise Hindi explicit refusal added 1.25 × ‖e₂₄‖ ≈ 61.5 along this
same direction, about three times this offset. This comparison was not registered.

## What this means

- **Gap localisation:** the English–Hindi difference on the refusal direction is concentrated, not spread out. 100 neurons, mostly in
  layers 14–24, account for three quarters of it on held-out pairs.
- **No Hindi-only lever:** these neurons are not a switch that restores refusal in Hindi alone. Moving them shifts the representation in
  all languages, and at this size it changes English behaviour only.
- **Consistent with Phase 3 and 4:** the steering fix worked across languages (Hindi, then Bengali). Here the most gap-related neurons
  also act across languages. Neither finding points to a Hindi-specific mechanism.

## Limits

- One model, one metric (projection on e₂₄ at one layer), one way of choosing neurons (attribution with a harmless-pair exclusion).
- One offset size (the mean difference). It was not swept, so a larger offset might change Hindi behaviour.
- Labels come from one grader route (Sonnet 5.5). There was no second grader for this phase.
- The Hindi BROKEN rise with the selected offset (+4.3) is under the 5-point line used elsewhere. Its interval excludes zero.
- Run: A40, pinned environment, about $1.2 (shared pod session with Phase 4b; total about $1.77).
