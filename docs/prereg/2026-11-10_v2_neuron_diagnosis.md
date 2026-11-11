# Pre-registration: v2 neuron diagnosis on Qwen3-8B (Phase 2)

**Status:** frozen by the commit that adds this file, before any v2 activation is computed. It runs on the Phase 1 splits, whatever their size, and says so
if they are underpowered.

**Question:** which MLP neurons, if any, carry the difference between refusing in English and complying in Hindi or Bengali, and is the failure one of
recognition (harm not detected) or routing (harm detected, refusal not triggered)?

**Basis:** MLP neurons (SwiGLU output, the input to `down_proj`), 36 layers × 12,288. Recent work reports MLP neurons to be functionally sparse with few
modes on a large mixture-of-experts model. That has not been shown for a dense 8B model, and this phase does not assume it: a diffuse result is reported as such.

**Seen before writing:** as in the Phase 0 and 1 file. No v2 activation, reply or grade.

## Data

- **Attribution:** fit-set failure triples (English-refused, target not refused), and fit-set English-answered harmless triples as the control.
- **Patching tests:** development-set failure triples, and development-set English-answered triples.

Prompts are rendered as in v1. **Aligned positions** are the chat-template positions after the end of the user message, identical in token identity across
languages. The **answer position** is the last of these.

## Decision score

Per language L, from fit-set unchanged replies: R_L is the set of first reply tokens appearing at least 5 times with at least 80% of occurrences in replies
graded refusal, and C_L the same for non-refusal. The score is `s_L = logsumexp(logits[R_L]) − logsumexp(logits[C_L])` at the answer position. If R_L or C_L
has fewer than 3 tokens, first tokens are replaced by the first two tokens, scored by summed log-probability, and the report says so.

## Analyses

1. **Logit lens.** `s_L` read from the residual stream after each layer through the final norm and unembedding, for English and target versions of failure
   triples. Reported: the first layer at which the English and target curves differ by more than half their final difference.
2. **Attribution patching.** For each neuron, the estimated effect on `s_target` of replacing its target-language activation with its English activation at
   the aligned positions: `(a_en − a_tgt) · ∂s_target/∂a`, summed over aligned positions and averaged over failure triples. The same quantity on the
   control triples is `d_h`. Neurons in the top 10% of |d_h| are excluded, which removes language-identity neurons that move refusal on harmless prompts too.
   The remaining neurons are ranked by the failure-triple estimate, per language and pooled.
3. **Neuron patching (causal).** Candidate sets: the top N pooled neurons for N in {64, 256, 1024}. On development triples, their English activations
   replace the target activations at the aligned positions. Conditions:
   - Matched: English activations from the same triple.
   - Mismatched: English activations from a different English-refused triple of the same dataset.
   - Random: N neurons drawn with the same per-layer counts, matched English activations.

   Outcomes: change in `s_target`, and refusal on one greedy reply of 100 new tokens, graded by the Phase 0 judge.
4. **Recognition or routing.** The selected set is split at layer 24 into early-middle (0 to 23) and late (24 to 35), and each half is patched alone.
5. **Mode check.** For each neuron in the top 64, its 100 highest-activating prompts (aligned positions) among all fit-set prompts in the three languages.
   A neuron whose top 100 are under 50% harmful by benchmark label, or under 50% English-refused, is flagged "multi-role".

## Rules

**Selected set:** the smallest N for which matched patching raises refusal on development failure triples at least 15 points over random patching, with a
paired bootstrap interval above 0 (1,000 resamples), and raises refusal on development English-answered triples by at most 3 points. Hindi and Bengali are
tested separately with Holm correction. A set must qualify in a language to be used for that language.

**Gate for Phase 3:** a selected set exists in at least one language. If none does, Phase 2 reports that refusal parity is not restored by patching up to 1,024
neurons at the aligned positions, and Phase 3 uses the Step 9 activation-based column choice instead, with the gate failure stated.

**Readings, reported with the selected set:**

- Matched clearly beats mismatched (interval above 0): the neurons carry request-specific content, consistent with recognition.
- Matched about equals mismatched: the neurons carry a generic "English refusal mode", consistent with routing or language-conditioned refusal.
- The early-middle half alone gives at least 70% of the full effect: recognition. The late half alone gives at least 70%: routing. Otherwise: mixed.
- Logit-lens divergence layer, reported beside these.

These readings are descriptive. Only the selected-set rule and the gate decide what Phase 3 does.

## What passes to Phase 3

- **Edit layer:** among late-layer neurons in the selected set (layers 24 to 35), the layer with the largest summed attribution. If the selected set has no
  late neurons, the layer with the largest summed attribution overall.
- **Columns:** the selected-set neurons in that layer, minus multi-role neurons, topped up to k by attribution rank within that layer.

## Addenda

### 2026-11-11: Hindi only

All attribution, patching and selection use Hindi. Bengali is not analysed in this phase, so Holm correction across languages no longer applies. The
selected-set rule and the gate are applied to Hindi alone.
