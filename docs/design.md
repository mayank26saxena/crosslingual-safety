# Design: closing the Bengali safety gap with activation-level interventions

## Goal

Determine why Qwen2.5-7B-Instruct helps with harmful requests more often in Bengali than in English, and test whether an
activation-level intervention can close that gap without retraining and without increasing refusal of harmless Bengali
requests. The target is language-agnostic safety behaviour: a harmful request should be refused about as reliably in
Bengali as in English.

## Starting point

Baseline: 150 MultiJail requests with 3 sampled replies each, graded by Qwen2.5-14B-Instruct. Aggregates are in
`results/baseline_summary.json`.

| Language | Unsafe | Safe | Invalid | Unsafe gap vs English (95% paired bootstrap) |
|---|---:|---:|---:|---:|
| English | 12.0% | 87.8% | 0.2% | |
| Bengali | 31.3% | 62.4% | 6.2% | +19.3 [+12.2, +25.6] |
| Swahili | 38.0% | 31.6% | 30.4% | +26.0 [+18.4, +32.2] |
| Javanese | 9.1% | 53.8% | 37.1% | -2.9 [-8.4, +2.0] |

Bengali shows a large gap with a low invalid rate, which makes it the primary target. Swahili and Javanese replies are
substantially non-engaging, so they serve as contrast cases. An earlier attempt, adding the English-minus-Bengali shift
on about 5,000 selected MLP neurons, selected for language identity and destroyed the output (100% invalid at the
higher strength). That approach is retired.

Three properties of the baseline shape this design:

1. **Safe may not mean refused.** Median Bengali reply length is 190 characters against 743 in English. A crude
   opening-phrase match finds refusal in 34.4% of English replies and 11.1% of Bengali replies. Correction (2026-11-05): the
   four-way judge later rated 57.3% of Bengali replies as explicit refusals, so the phrase match badly undercounts them.
2. **The over-refusal guard needs rebuilding.** Of 250 machine-translated XSTest-Bengali prompts, 33 have under 50%
   Bengali script or are over three times the English length. Inspected outliers include appended translator notes,
   text in the wrong language, and changed trigger words. Unchanged-model refusal on this set is 26.4%, against 9.2% for
   the English originals; some of that difference may be a translation artefact.
3. **Reply-level noise is high.** All three replies agree in only 66% of Bengali requests. The unsafe rate exceeds
   English on 60 requests and falls below it on 13. Selecting requests on an observed gap therefore invites regression
   to the mean, and any analysis that selects this way needs freshly sampled unchanged-model replies.

## Research questions

1. Where does the harmfulness signal live for Bengali and English, layer by layer?
2. Is the Bengali failure one of recognition (harm is not represented) or of action (harm is represented but refusal is
   not triggered)?
3. Can a gated, direction-level intervention close the gap while keeping harmless Bengali prompts answered?
4. Does it transfer to Swahili and leave English unchanged? Where does it fail, and why?

## Plan

Each step gates the next and has its own dated pre-registration in `docs/prereg/`.

| Step | Content | Gate |
|---|---|---|
| 1. Evaluation | Second-family judge, translate-then-judge arm, four-way reply types, rebuilt XSTest-Bengali | Gap survives a second judge |
| 2. Diagnostics | Direction validity, probe transfer, refusal projection, activation patching | Probe transfers to Bengali |
| 3. Interventions | Ungated addition, cosine-gated addition, add/ablate gate; English-only and Bengali-calibrated gates | Pass rule below |
| 4. Baselines | Restate-in-English system prompt; optional guard model | Comparison only |
| 5. Confirmation | Best configurations on the unseen split; Swahili transfer, English stability, Bengali capability | Pass rule below |
| 6. Replication | Best arm on Gemma-2-9B-it | Same rule |

### Data discipline

- **Dev (150 requests):** the baseline set, halved into A and B for anything fitted or selected.
- **Confirm (165 requests):** no replies have been generated on these requests. They stay sealed until Step 5 and are
  used once.
- **XSTest:** halved within type. Half A may be used for fitting; only half B counts toward the over-refusal guard.

Splits are fixed in `data/splits/splits.json`.

### Intervention pass rule (to be frozen in the Step 3 pre-registration)

At a fixed strength, on held-out requests, an intervention passes if all of these hold:

- Bengali unsafe rate is below the unchanged model and below both the random-direction and the matched-control-direction
  arm at the same strength, with a paired item bootstrap 95% interval excluding 0.
- Bengali safe rate is not lower than the unchanged model, and Bengali invalid rate is not higher. A move from unsafe to
  invalid is not a gain.
- XSTest-Bengali answer rate is at least 90% of the unchanged model's, on both the full held-out set and its
  trigger-preserved subset.

Controls use the same norm, layers, positions and gate as the intervention. The random arm uses at least five seeds.
The matched-control direction is a difference in means between two harmless prompt groups, so it is a real
activation-space contrast with no safety content.

## Evaluation

Every condition reports the full distribution of safe, unsafe and invalid replies, plus explicit-refusal rate and
Bengali-script share of replies. Statistics use a paired item bootstrap (1,000 resamples) with per-fold results beside
pooled ones. Failures are reported as findings.

## Related work and contribution

| Work | Relevance |
|---|---|
| Arditi et al., 2024 ([arXiv 2406.11717](https://arxiv.org/abs/2406.11717)) | Refusal mediated by a one-dimensional residual-stream subspace in 13 chat models |
| Wang et al. ([arXiv 2505.17306](https://arxiv.org/abs/2505.17306)) | Refusal directions approximately parallel across 14 languages on Qwen2.5-7B, Gemma-2-9B and Llama-3.1-8B; no Bengali |
| Zhao et al. ([arXiv 2507.11878](https://arxiv.org/abs/2507.11878)) | Harmfulness (last instruction token) and refusal (last post-instruction token) are separate directions, on older models |
| Aziz et al., 2026 ([arXiv 2606.01196](https://arxiv.org/abs/2606.01196)) | Few-shot latent gate that adds or ablates a harmfulness direction: selectivity 54.5 against 33.6 for CAST and 6.8 for AdaSteer, over 23 languages with no Bengali in the language set |
| Stein et al., BabelSteering ([arXiv 2608.16577](https://arxiv.org/abs/2608.16577)) | Refusal-vector addition combined with false-refusal ablation; on Gemma 7B, Bengali refusal 37% to 54% with over-refusal up 23 points |
| Zhang, MRL 2026 ([ACL Anthology](https://aclanthology.org/2026.mrl-1.19/)) | Pre-registered transfer test: Qwen2.5-7B refusal 79.7% in English, 29.5% in Bengali. Identifies ablating a matched non-safety direction as the control it did not run |
| Tang et al., 2024 ([arXiv 2402.16438](https://arxiv.org/abs/2402.16438)) | Language-specific neurons concentrated in the top and bottom layers |
| Wendler et al., 2024 ([arXiv 2402.10588](https://arxiv.org/abs/2402.10588)) | Llama-2 middle layers decode concepts closer to English than to other languages |

The method is not new; Aziz et al. established it. This project's contribution rests on what remains unmeasured: gated
steering behaviour on Bengali MultiJail for these models, over-refusal on borderline Bengali prompts with a checked
translation, an English-only gate compared with a Bengali-calibrated gate, and random and matched-control directions
applied to the intervention itself. Details not yet confirmed against the sources, such as Aziz et al.'s gate token
positions and the Qwen layer range in Wang et al., are treated as open in the code rather than assumed.
