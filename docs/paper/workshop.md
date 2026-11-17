# Same Gap, Different Signatures: Diagnosing and Repairing the Hindi Refusal Gap in Two Open Chat Models

*Workshop draft (short-paper format, about 4 pages plus references). Anonymised for double-blind review: no author names, repository URL or
blog link. Citations are author–year; the reference list is to be generated from `docs/paper/reports/`, and entries marked [verify] there
must be checked first. Full details are in the long draft, `docs/paper/draft.md`.*

## Abstract

Open chat models refuse harmful requests less reliably outside English. We measure this gap on *paired* requests, the same request asked in
English and in Hindi, and grade replies with a four-way label (comply, explicit refusal, soft refusal, broken output) validated against a
fluent human. On 404 held-out pairs, explicit refusal falls from 51% to 19% in Qwen3-8B and from 89% to 47% in Llama-3.1-8B-Instruct.

- **What replaces the refusal.** In Qwen the lost refusals are mostly soft (warnings and redirections without a "no"). In Llama they split
  between soft refusal and compliance.
- **Internal signatures differ.** On an English-fitted refusal direction, Qwen's Hindi harmful prompts project at English strength,
  Llama's at 28% of it.
- **Qwen repair.** A gated push along an explicit-versus-soft direction raises Hindi explicit refusal by 17 points, but raises English by
  16, so the gap barely moves. It also fails a sealed Bengali confirmation on garbled output.
- **Llama repair.** A push along the refusal direction itself, with a gate threshold set on Hindi data, raises Hindi explicit refusal by
  15 points and narrows the gap by 7. It narrowly misses its pre-registered bars.

All studies were pre-registered, tuned on development items and tested once.

## 1 Introduction

Safety training is mostly done in English. A growing literature shows that harmful requests in other languages are refused less often
(Yong et al. 2023; Deng et al. 2024; Wang et al. 2024), and that English refusal directions transfer across languages (Wang et al. 2025;
Aziz et al. 2026). Two questions have received less attention:

- **What replaces the missing refusal?** Prior work scores refusal as binary or as a harm scale.
- **Does the same behavioural gap have the same internal cause in different models?**

We answer both in Hindi for two open 8B models, on identical items.

**Contributions.**

1. **A response-side decomposition of the gap** on paired, screened, non-adversarial requests, with a four-way, human-validated label.
2. **An explicit-versus-soft refusal direction.** To our knowledge, this is the first activation direction for refusal *style*. Hindi
   prompts sit toward its soft end, and adding it turns soft refusals into explicit ones.
3. **A contrast of internal signatures on identical items:** a projection at English strength in Qwen, 28% of it in Llama.
4. **Pre-registered tests of two repairs**, including negative results: a repair can pass its checks without narrowing the gap, and can
   fail confirmation in a held-out language.

## 2 Related work

- **Multilingual safety gaps.** These were quantified by translation attacks and benchmarks (Yong et al. 2023; MultiJail, Deng et al.
  2024; XSafety, Wang et al. 2024). Indic work includes IndicSafe, SurakshaEval, and IndicSafeEval (Mondal et al. 2026). The last evaluates
  our two models in Hindi, but with machine-translated persuasion attacks and per-language attack-success rates.
- **Refusal directions.** Refusal is mediated by a difference-of-means direction (Arditi et al. 2024) that is roughly shared across
  languages (Wang et al. 2025). Low-resource harm projects more weakly on it (Aziz et al. 2026). Harmfulness and refusal are encoded
  separately (Zhao et al. 2025), so we describe projections on a refusal direction, not "harm recognition".
- **Conditional steering.** Gated steering is established: CAST (Lee et al. 2025), and target-language-calibrated gates (Aziz et al. 2026),
  who also find that Llama needs that calibration.
- **Refusal taxonomies.** These distinguish partial or indirect refusal (Röttger et al. 2024; Cui et al. 2024).
- **Where our work differs.** We contribute the style axis, the paired decomposition, the two-model contrast and the pre-registered
  protocol.

## 3 Setup

**Data.** Harmful requests come from Aya Red-teaming, written natively in Hindi (Devanagari) with human English translations, and from the
human-translated IndicSafe. Harmless controls come from PolyGuard and IndicSafe, and Global-MMLU measures capability. Items were screened in
three steps:

- request type (LLM screener, validated against a fluent human after adjudication);
- translation fidelity;
- comprehension (the model restates the Hindi request).

The held-out set is 404 harmful and 169 harmless pairs, identical for both models. Bengali MultiJail was sealed until a confirmation test.

**Generation.** Greedy decoding with fixed, length-sorted batches. The steered and unsteered arms of a comparison use identical batches.
Each phase ran on a single GPU in a pinned software environment.

**Grading.** Four labels, assigned in order: COMPLY (provides it, even with warnings), REFUSE_EXPLICIT (says it will not help), REFUSE_SOFT
(neither provides nor declines) and BROKEN.

- **Qwen.** The judge (Claude Sonnet 5.5) was selected against a fluent human on 150 replies: κ 0.82 refused versus not, 0.91 explicit
  versus the rest.
- **Llama.** The pre-registered 60-reply human check fell below its bar blind (four-label κ 0.643 [0.48, 0.79]). After the grader
  re-applied the rubric to the disagreements, agreement was 0.893, and we use the reviewed labels. Explicit refusal, our primary outcome,
  was the most reliable label blind (κ 0.785).

**Directions.** Both directions are read at the last template token after the user turn.

- *Refusal direction:* difference of means over English harmful and harmless prompts. The layer is chosen by ablation, and the direction
  counts as valid if ablating it removes at least 50% of a first-token refusal-score gap.
- *Explicit–soft direction:* difference of means over English prompts whose replies were explicit or soft refusals. It counts as valid at
  cross-validated AUROC ≥ 0.75.

**Protocol.** Each study was pre-registered (rules, sets, selection rule, guards), tuned on development items, and tested once on held-out
items. Deviations were recorded in dated addenda.

## 4 Results

### 4.1 The gap and what replaces the refusal

| 404 harmful pairs, Hindi | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Explicit refusal, English → Hindi | 51% → 19% | 89% → 47% |
| Gap | −32.2 [−37.4, −26.5] | −42.3 [−47.3, −37.4] |
| Lost refusals become mostly… | soft refusal | soft and compliance, similar shares |
| Hindi unrefused (comply + broken) | 24.0% | 33.2% |

In Qwen, labels move from explicit to soft refusal (230 → 82 explicit, 170 → 265 soft on a 448-pair superset). In Llama, the lost refusals
split between compliance and soft refusal (88 vs 75 of 172). Llama leaves Hindi harmful requests unrefused more often than Qwen (+9.2
[+4.5, +14.1] counting broken replies, +13.9 counting compliance alone).

### 4.2 Internal signatures

**Refusal direction.**
- *Qwen* (layer 14; ablation removes 64% of the score gap): Hindi harmful prompts project at English strength (paired shift +0.17
  [+0.02, +0.30]). Harmless Hindi prompts also sit high, which lowers harmful–harmless separation.
- *Llama* (layer 10; ablation removes 94%): Hindi harmful prompts reach 28% of the English mean projection.

**Explicit–soft direction.**
- *Qwen* (layer 24, CV AUROC 0.93): Hindi prompts sit toward the soft end (−10.9 [−12.1, −9.6]). Adding the English vector raises Hindi
  explicit refusal on 150 pairs from 10% to 53%, with no harmless over-refusal.
- *Llama* (layer 18, AUROC 0.925): it shows the same shift, but it was fitted on only 11 English soft refusals.

### 4.3 Repairs (held-out, run once)

| Repair | Hindi explicit | English explicit | Gap change | Verdict |
|---|---|---|---|---|
| Qwen: gate on refusal score (English-set threshold), push along explicit–soft at layer 25 | +17.1 [+13.1, +21.0] | +15.8 | 31.9 → 30.7 | passes all checks; gap barely narrows |
| Qwen, sealed Bengali confirmation | +23.6 | +6.8 | 23.6 → 6.8 | **fails** (BROKEN +8.1, limit 5) |
| Llama: gate with Hindi-set threshold, push along refusal direction at layer 11 | +14.6 [+11.1, +18.3] | +7.2 | narrows 7.4 [+3.2, +11.6] | **misses** bars (+15; 10) |

No repair increased harmless refusal or reduced Global-MMLU. These guards are weak, because the gates rarely opened on benign inputs.

Further findings:
- A Hindi-specific neuron search (100 MLP neurons recovering 75% of the internal English–Hindi difference) changed behaviour in English
  (+15.2) but not Hindi (+1.2).
- In Llama, the gate calibrated on Hindi made the difference. With an English threshold it opened on 1% of Hindi harmful prompts, and with
  a Hindi threshold on 47%. Even the explicit–soft push qualified on development data once calibrated.

## 5 Discussion and limitations

- **Diagnosis.** The same explicit-refusal gap arises with different internal signatures, and the response-side label is what makes the
  difference visible.
- **Gates are not language-selective.** Gating on a refusal-direction score raised English as well as Hindi. Gap-narrowing should be a
  registered outcome of any cross-lingual repair.
- **Scope.** Two 8B models and greedy decoding. Labels come from one LLM judge and one human. There is no prompting or translation baseline,
  and no hard-negative test of the projection claims. The Llama test reused its diagnosis items, though no Llama setting was chosen on
  them.
