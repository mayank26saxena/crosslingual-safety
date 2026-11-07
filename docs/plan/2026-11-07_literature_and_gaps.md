# Literature review and gap analysis (2026-11-07)

Search covered mechanistic and steering work on multilingual safety, published 2024 to 2026. Reading depth varies and is stated per source. Numbers from papers are as reported by their authors and were not reproduced here.

## 1. The three candidate explanations, and where each stands

| Explanation | What it predicts | Evidence in this project | Evidence in the literature | Status |
|---|---|---|---|---|
| Semantic representation failure | Harm is not separable in Hindi or Bengali activations | Qwen3-8B harm probe AUC about 0.85 in both languages against 0.89 in English; an English-trained probe transfers at 0.82. Earlier steps found intent beat surface features in Chinese, Arabic and Bengali on three models | Aziz et al. find harmfulness stays linearly separable in 23 languages on three models; Wang et al. and Verma and Bharadwaj propose the opposite | Mostly addressed. Separation is largely present, slightly weaker, and Swahili is the exception in earlier work |
| Safety classification failure | The harm signal exists but is weaker, so it falls below the model's implicit refusal threshold | Failures have a weaker harm signal than successes: signal fired on 57% of Hindi and 44% of Bengali failures, against 80% and 75% of successes | Aziz et al.: lower projections onto the harmfulness direction in low-resource languages, so the same evidence less reliably triggers refusal. They call it calibration | Partly addressed, correlational. This and the next row are hard to separate |
| Refusal-routing failure | The signal is clear but does not reach refusal | Failures also have lower refusal-direction projection. Steering on a fired subset raises refusal 11 to 17 points on development data | Aziz et al. call it an action failure. Upadhyaya and Sikdar: safety features are entangled with language identity. Sarvam MoE study: decided upstream, written late | Open. The held-out causal test is running |

The explanations are not exclusive. The data suggest a graded picture: recognition is mostly present but weaker in failing prompts, and the decision to refuse needs a stronger signal than these prompts supply.

## 2. What the literature offers

| Source | Reading | Relevant finding | Use for this project |
|---|---|---|---|
| Aziz et al., [2606.01196](https://arxiv.org/abs/2606.01196) | Abstract, introduction, setup | Failures are calibration, not representation. A few-shot gate resets the threshold with 1 to 4 target examples per class and routes between refusal steering and harmfulness ablation. Reported selectivity 54.5 against 33.6 | Closest prior method. We differ by using a Hindi and Bengali benchmark with a documented label problem, and by testing controls |
| Upadhyaya and Sikdar, [2608.29936](https://arxiv.org/abs/2608.29936) | Abstract | Safety features are entangled with language identity; ablating them changes the output language | Explains why direction steering can switch language. Our blind and script checks guard this |
| Aravindan et al., OPIUM, [2607.19806](https://arxiv.org/abs/2607.19806) | Abstract | A steering vector can be sanitized by representation matching so it keeps utility and avoids over-refusal | Candidate for the steering vector itself, if gating alone is not enough |
| Stein et al., BabelSteering, [2608.16577](https://arxiv.org/abs/2608.16577) | Search summary | English steering vectors transfer to other languages; over-refusal rises, most on prompts that sound harmful | Evidence that ungated steering over-refuses |
| Liang et al., [2602.22554](https://arxiv.org/abs/2602.22554) | Search summary | Training-free sparse weight edit as a regularized low-rank update with a closed form from a few anchors | Template for a localized weight edit once an activation effect is shown |
| Zhao et al., [2507.11878](https://arxiv.org/abs/2507.11878) | Search summary | Harmfulness and refusal are encoded separately; steering refusal need not change the model's harm judgement | Supports gating on one signal and steering another |
| Zhang, MRL 2026, [ACL Anthology](https://aclanthology.org/2026.mrl-1.19/) and [2610.28897](https://arxiv.org/abs/2610.28897) | Search summary | Refusal direction transfer across languages is graded; geometry does not imply a shared causal mechanism | Matches our mixed Step 3 result; supports measuring transfer per language |
| [2608.08032](https://arxiv.org/abs/2608.08032) | Search summary | In a multilingual MoE the gap arises in late decoding, not in detection | Suggests an upstream intervention |

## 3. Gaps in our own method, and corrections

| Gap | Evidence | Correction |
|---|---|---|
| The benchmark label is not the model's own English policy. In English the model refuses only 58% of IndicGuard "harmful" prompts | Baseline rates | Define the target as parity with the model's English decision on the same prompt. Over-refusal becomes: prompts answered in English must stay answered |
| The harm-label gate fires mostly on prompts the target language already refuses | On development data, 27% to 32% of the prompts a 5% false-alarm harm gate catches had not been refused, and the steering lift on fired prompts is large only where refusal was already likely | Train the gate on English refusal decisions. A CPU check found AUC 0.87 to 0.88 for predicting English refusal against 0.76 to 0.79 for the harm label, and higher failure recall at equal false alarms (Bengali 45% against 31% at 5%) |
| One direction at one layer lifts refusal only 2 to 3 points over all development prompts | Development grid | Treat the gain as bounded by gate recall. If gate recall stays under 60%, the fix needs a different lever, such as a self-distilled low-rank edit |
| Refusal grading disagrees on Hindi (kappa 0.55 against a single-model reference) | Reference check | Human Hindi grades take priority. Absolute rates are labelled provisional |
| Most Hindi and Bengali prompts are translated | Dataset provenance | Report SurakshaEval and BanglaShield native separately. SurakshaEval has no harm labels and almost no refusals, so it supports only descriptive rates |
| One model | Single run | Replicate on a second model, with its access terms accepted |

## 4. Ranked next experiments

1. **Parity-targeted gating (Step 8).** English-decision gate, parity metric, fresh held-out draw, same controls. Cheap, and tests the main gap above. Pre-registered separately.
2. **Self-distilled low-rank edit.** Train a rank-1 to 8 update so Hindi and Bengali refusal matches the model's own English decision on the same prompt, with a preservation term on English and on prompts English answers. Needs a pre-registered training rule and a large held-out set. Only after (1) shows a reproducible effect.
3. **Sanitized steering vector (OPIUM-style).** Only if (1) shows a gain with over-refusal as the limit.
4. **Second model replication** after a result exists.
