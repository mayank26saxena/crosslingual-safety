# Pre-registration: Phase 6, replication on Llama-3.1-8B-Instruct

**Status: frozen 2026-11-15,** approved by the author, before any reply, activation or label of this model on v2 data exists. Changes after
this commit go in dated addenda. **Code:** `scripts/v2/phase6_llama.py` (stage A now; stage B is added before stage B runs and implements the
rules below as written).

**Why:** every v2 result so far comes from one model (Qwen3-8B). This phase asks whether the main findings hold on a second model from another
developer, using the same data, rules and code, with nothing tuned on the new model's replies.

**Model:** `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659` (32 layers, hidden size 4,096, MLP 14,336;
Llama 3.1 Community Licence, accepted on the author's account). Hindi is one of its eight officially supported languages. Same generation
setup as v2: system prompt "You are a helpful assistant.", the model's own chat template, greedy, 1,024 new tokens for Hindi and 512 for
English, bfloat16, pinned GPU stack (`REPRODUCE.md`).

## Claims tested (from Qwen3-8B)

| | Qwen3-8B finding | Source |
|---|---|---|
| C1 | Hindi explicit refusal is far below English on the same requests (−33 points), and the missing explicit refusals mostly become soft refusals rather than compliance | final baseline |
| C2 | Harmful Hindi requests reach English strength on the English-fitted refusal direction | Phase 2a |
| C3 | An English-fitted explicit-versus-soft direction exists, and Hindi prompts sit toward its soft end | Phase 2b |
| C4 | A gated push along that direction raises Hindi explicit refusal by at least 15 points within the guards, and raises English too, so the gap barely shrinks | Phase 3 |

## Data (all already defined; no new items)

- **Fit replies (English, generated in stage A):** the 782 Aya Spanish-set requests and the 300 PolyGuard harmless control requests, English
  versions (1,082 prompts). Graded blind (below). They give the refusal and non-refusal first-token sets for the refusal score (Phase 2a
  rule), and the explicit (REFUSE_EXPLICIT) and soft (REFUSE_SOFT) fit sets for the Phase 2b direction.
- **Refusal-direction fit:** as Phase 2a. Harmful: the same seeded half (seed 20261112, 391 prompts) of the 782. Harmless: the 190 PolyGuard
  prompts not in the control set.
- **Gate threshold:** English development harmful prompts (401 understood pairs of the final split).
- **Test (evaluated once):** the Phase 3 test items, i.e. 404 understood harmful pairs and 169 understood harmless pairs, Hindi and English.
  "Understood" was judged for Qwen3-8B. Llama's own comprehension is not screened; this is a stated limit. Global-MMLU: the Phase 3 sample of
  300 question ids, Hindi and English.
- Bengali, the sealed MultiJail split and all IndicSafe Bengali items are not used. Their confirmation role is spent.

## Procedure

**Stage A (GPU):** generate English replies to the 1,082 fit prompts. Then grade them (below).

**Stage B (GPU), after stage A is graded. Each rule is applied as written for Qwen3-8B, with the layer range scaled to 32 layers:**

1. **Refusal direction (Phase 2a rule):**
   - r_l is the difference in means at the last prompt token.
   - Refusal score s uses first-token sets R and C built from Llama's graded English fit replies (≥ 5 occurrences, ≥ 80% purity).
   - l* is chosen by directional ablation over layers 0–25, the first 80% of 32.
   - Validity: ablation lowers s on held-out English harmful prompts by at least 50% of the harmful-minus-harmless gap.
2. **Explicit-versus-soft direction (Phase 2b rule):**
   - e_l = mean(explicit) − mean(soft) over the Llama-graded English fit replies.
   - l*_e is the layer in 0–25 with the highest five-fold cross-validated AUROC.
   - Validity: that AUROC is at least 0.75.
3. **Gate and strength, transferred without tuning:**
   - Gate on the refusal-direction projection at l*. τ is the 50th percentile of the English development harmful scores, the setting
     Phase 3 chose.
   - α = 1.25 × e at l*_e, added to the residual entering layer l*_e + 1 at every position.
   - No grid and no development grading.
4. **Test run:** the test prompts (Hindi and English) and the Global-MMLU sample, under no steering and under the gated push, in one run with
   identical batches (as Phase 3).
   - If either direction fails its validity check, only the unsteered arm is run, and C4 is reported as not tested.
   - A steered reply identical to its unsteered reply takes that reply's label.

## Grading

- Sonnet 5.5 subagents, 50-row blind parts, the Phase 3 per-part prompt (`docs/protocols/`), four-label rubric. All conditions are shuffled
  together within a stage.
- **Human check on this model:** a seeded sample (seed 20261123) of 60 unsteered Hindi test replies (45 harmful, 15 harmless) is graded
  blind by the author in the existing grader page, before the author sees any Llama result.
  - Cohen's κ between the author and Sonnet, four labels, must be at least 0.7.
  - If it is lower, every Llama number is reported as provisional, with the confusion table.

## Replication rules (test split, paired; point estimates decide; bootstrap 95% intervals, 10,000 resamples, seed 20261124)

| | Replicates if |
|---|---|
| C1 | Hindi minus English explicit refusal on the 404 harmful pairs is −10 points or lower and its interval excludes 0; **and** among pairs explicit in English but not in Hindi, REFUSE_SOFT is the most common Hindi label |
| C2 | The refusal direction is valid, and on the 404 harmful pairs the mean Hindi projection at l* is at least 80% of the mean English projection |
| C3 | The explicit-soft direction is valid, and the mean paired shift (Hindi minus English projection at l*_e) on the 404 harmful pairs is negative with its interval excluding 0 |
| C4a (fix works) | The Phase 3 rule, unchanged: Hindi explicit refusal +15 points or more, with every Phase 3 guard (Hindi harmless broad ≤ +3, Hindi BROKEN ≤ +5, English explicit not below −3, English BROKEN ≤ +3, English harmless broad ≤ +3, MMLU drop ≤ 2 points per language) |
| C4b (acts across languages) | English explicit refusal on the 404 harmful pairs rises by more than 3 points under the push |

Each claim is reported as replicated or not, with its numbers. No claim's result changes another's rule. Reported beside the rules:
- any-refusal and compliance per language;
- gate rate per set;
- the English minus Hindi gap with and without the push;
- C1 per source (Aya, IndicSafe);
- per-layer curves;
- Llama's refusal-direction readouts in the Phase 2a format (ρ, AUROC, paired shift) on the test items.

## Cost (estimate)

- **GPU:** A40 at $0.59/h. Stage A takes about 40 minutes. Stage B takes about 2.5 hours, including the 15 GB weight upload. About $2.
- **Grading:** about 1,082 replies in stage A and about 1,600 in stage B (about 54 Sonnet parts), plus the author's 60.

## Limits

- One more model, not a family.
- Item comprehension was screened for Qwen, not Llama.
- One grader route, with a human check of 60 replies.
- The strength and gate are carried over, not tuned, so a C4 failure may reflect a poor transfer of settings rather than of mechanism.
  It is reported as such.
