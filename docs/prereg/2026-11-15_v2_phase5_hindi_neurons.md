# Pre-registration: Phase 5, Hindi-specific neurons behind the explicit-refusal gap (diagnosis)

**Status: frozen 2026-11-15,** before any activation, gradient or reply in this study was computed. Changes after this commit go in dated
addenda. This plan replaces the unrun Phase 2 neuron plan (`2026-11-10_v2_neuron_diagnosis.md`); an addendum there points here.

**Why:** Phase 2a found that harmful Hindi requests reach the English level on the layer-14 refusal direction. Phase 2b found Hindi sits
only 22% of the English distance along the layer-24 explicit-vs-soft direction. Phase 3 showed that a push which does not depend on the
language raises English as much as Hindi, so it cannot close the gap. This study asks whether a small set of MLP neurons carries the Hindi
shortfall, and whether that set is specific to Hindi. **It is a diagnosis:** no repair is confirmed here.

## Units, metric and positions

- **Neurons:** MLP neurons (the SwiGLU output entering `down_proj`) in layers 0 to 24: 25 × 12,288 = 307,200. Later layers cannot move the
  metric, which is read at the layer-24 output.
- **Metric m:** the projection of the layer-24 output at the last prompt token on the unit Phase 2b direction ê₂₄
  (`results/v2/es_dir/es_dir.npz`). Phase 2b: it separates explicit from soft refusals (accuracy score 0.93), and pushing along it raises
  explicit refusal.
- **Aligned positions:** the chat-template tokens after the user message (`<|im_end|>`, assistant header, empty think block). They are
  token-identical across languages; the script checks this for every prompt. All patching acts on these positions only.

## Data (Hindi development split only; the test split and all confirmation sets are untouched)

- **Gap pairs:** understood Hindi development harmful pairs whose English baseline reply is REFUSE_EXPLICIT and whose Hindi reply is not
  (158 of 401, from `results/v2/labels/hindi_baseline_grades.csv`). Split at random (seed 20261121) into a discovery half (79) and a test
  half (79).
- **Harmful test set for replies:** the 401 development harmful pairs minus the discovery half (322), Hindi and English.
- **Harmless:** the 171 understood harmless development pairs, split at random (seed 20261121) into discovery (85) and test (86).

## Steps

1. **Attribution (discovery half).** For each neuron, the estimated change in the Hindi metric from replacing its Hindi activation with
   the English one at the aligned positions: (a_en − a_hi) · ∂m_hi/∂a, summed over positions and averaged over the 79 discovery gap pairs.
   Gradients are total derivatives (a zero offset is added at each neuron's input and differentiated). The same quantity on the 85
   discovery harmless pairs is d_h; neurons in the top 10% of |d_h| are excluded, which removes neurons that track language on harmless
   requests too. The rest are ranked.
2. **Recovery (test half, forward only).** For N ∈ {10, 50, 100, 500, 1000, 5000}, the top-N neurons take their English activations from
   the same pair at the aligned positions. Recovery(N) = mean(m_patched − m_hi) / mean(m_en − m_hi) over the 79 test gap pairs. Control:
   N random neurons with the same count per layer (seed 20261121). Paired bootstrap 95% intervals, 10,000 resamples, seed 20261121.
   **N\*** is the smallest N with recovery ≥ 0.5.
3. **Behaviour (replies, graded blind).** The selected set is the top-N\* neurons, or the top 500 if N\* > 500 or does not exist. A fixed
   offset is added to those neurons at the aligned positions: the mean (a_en − a_hi) over the discovery gap pairs. This needs no English
   input, so it can be applied to English prompts too. Conditions: none, selected offset, random offset (the same random set as step 2 at
   that N). Prompts: the 322 harmful pairs in Hindi and English, and the 86 test harmless pairs in Hindi. Greedy; 1,024 new tokens for
   Hindi, 512 for English; identical fixed batches. Graded with the Phase 3 prompt by Sonnet 5.5 subagents; a reply identical to its
   no-offset reply takes that label.

## Outcomes (assigned by the analysis script in this order)

| Outcome | Rule |
|---|---|
| **Method fails** | At every N, recovery with top-N is not above random-N (the interval for the difference includes 0 or is below). |
| **Spread out** | Otherwise, N\* > 500 or no N ≤ 5000 reaches recovery 0.5. N\* is reported. |
| **Small and Hindi-specific** | Otherwise (N\* ≤ 500), and the selected offset raises Hindi explicit refusal by at least 15 points with the interval for selected minus random above 0, changes English explicit refusal by at most 3 points either way, raises Hindi harmless refusal by at most 3 points and Hindi BROKEN by at most 5. |
| **Small but shared** | N\* ≤ 500, the Hindi rise is at least 15 points, but English explicit refusal rises by more than 3 points. |
| **Small, no clean behavioural effect** | N\* ≤ 500, but none of the two rows above applies: the Hindi rise is under 15 points (the metric moves, replies do not), or it does not beat random, or a harmless or BROKEN limit fails. |

**Reported beside the outcome:**
- **Where the neurons sit:** the share of the selected set's summed attribution in layers 0 to 13 (before the layer-14 gate) against
  14 to 24. At least 70% in 0 to 13 is reported as "early" (the gap starts in reading Hindi), at least 70% in 14 to 24 as "late", else
  "mixed".
- **Bengali (diagnosis only):** the same Hindi-derived offset on the 150 MultiJail development requests in Bengali, change in m only (no
  replies). Every Bengali confirmation set is already used, so this is not a confirmation.
- Recovery curves for every N, both controls, and English-side metric changes.

## Cost (estimate)

A40 at $0.59/h: attribution and recovery under 30 minutes; replies (about 1,400 rows across three conditions) about 2 hours; about $1.50
to $2 in all. Grading: every no-offset reply (730) plus each offset reply that differs from it, so 15 to 45 Sonnet parts (one to two usage windows).

## Limits

- One model, one metric. The metric is a linear readout; a neuron set can move it without changing replies, which is why step 3 exists.
- Only layers 0 to 24 and only the aligned positions are searched; neurons acting later or on request tokens are not covered.
- 79 discovery pairs. Attribution patching is a first-order estimate; step 2 checks it with real patching.
- Hindi labels come from the calibrated grader route (kappa 0.82 against the author on Hindi).
