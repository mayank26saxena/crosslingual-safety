# Phase 7 result: a diagnosis-matched fix for Llama-3.1-8B

**Pre-registration:** `docs/prereg/2026-11-15_v2_phase7_llama_fix.md` (frozen in 2f512a8; addendum 1 in aa77911, addendum 2 in 9179f70,
both before the data they concern). **Code:** `scripts/v2/phase7_llama_fix.py`. **Selection committed** in 03cf09f before any stage 2
reply existed. **Outputs:** `results/v2/phase7/stage1/stage1_results.json`, `results/v2/phase7/stage2/stage2_results.json`.
**Labels:** `results/v2/labels/phase7_{1,1e,2}_llama_grades.csv` (ids only).

Model `meta-llama/Llama-3.1-8B-Instruct` rev 0e9e39f2…, one A40 for every run, the pinned environment (driver 595.91.07 throughout).
Labels are Sonnet 5.5 grades, blind and shuffled. The Phase 6 human check fell below its bar blind (four-label κ 0.643); after the author's review, adopted as the
final human reference (Phase 6 addendum 5), agreement is 0.893. The Phase 7 author check agrees on 29 of 30 steered replies blind, and 30 of 30 after review (`2026-11-16_llama_human_checks.md`).
F1 and F2 rest on explicit refusal, the best-validated label. F3 also agrees with the adjudicated human labels, but the COMPLY/SOFT split
is the least reliable label.

## Outcome: compliance falls; the explicit-refusal and gap bars are narrowly missed

| Claim | Rule | Held-out result | Met |
|---|---|---|---|
| F1 fix works | Hindi explicit refusal ≥ +15 points, interval above 0, every guard | **+14.6** [+11.1, +18.3]; all guards pass | **No** (0.4 points short) |
| F2 gap narrows | English − Hindi explicit gap shrinks ≥ 10 points, interval above 0 | **7.4** [+3.2, +11.6] | **No** |
| F3 compliance falls | Hindi COMPLY falls, interval below 0 | **−8.4** [−11.1, −5.9] | **Yes** |

The push moves Hindi in the intended direction on every measure, with no damage on any guard. Its effect on explicit refusal (+14.6) is
just under the registered bar, and because English also rises (+7.2), the gap narrows by 7.4 points rather than 10. Under the frozen rule
the outcome is **F1 not met, F2 not met, F3 met**.

## Stage 1: development selection (150 Hindi harmful pairs, 60 harmless Hindi requests)

The gate threshold τ_hi = 0.412 is the 50th percentile of the gate score over the 401 Hindi development harmful prompts. Δ_r = 1.398 is the
mean English-minus-Hindi projection on the refusal direction. On development data the gate opened on 45% of Hindi harmful, 92% of English
harmful and 0% of harmless Hindi prompts.

| Arm | Push | Hindi explicit | Hindi compliance | Gap narrows | Guards | Qualifies |
|---|---|---|---|---|---|---|
| E1 | ê, 1.25·g_e at layer 19 | +16.7 | −10.0 | 13.3 | pass | yes |
| E2 | ê, 2.5·g_e at layer 19 | −6.7 | −10.7 | 9.3 | Hindi BROKEN +23.3 | no |
| R1 | r̂, 0.5·Δ_r at layer 11 | +7.3 | −6.7 | 5.3 | pass | no (< +15) |
| R2 | r̂, 1.0·Δ_r at layer 11 | +15.3 | −10.0 | 12.0 | pass | yes |
| **R3** | **r̂, 1.5·Δ_r at layer 11** | **+18.0** | −10.7 | 13.3 | pass | **yes, chosen** |

**Implementation error, corrected before stage 2 (addendum 2).** The first stage 1 run scaled the E push twice by the fit gap, about 4.8
times the registered strength. Both E arms broke nearly every gated reply. The E arms were regenerated in a second run with the fix. That run
also regenerated the unsteered replies, and all 360 were byte-identical to the first run, so the stage 1 unsteered labels stand. The R arms
were unaffected and are recorded in addendum 2 as they were before the rerun. The erroneous E results are kept in
`stage1_results_erroneousE.json`.

## Stage 2: held-out test (404 harmful pairs, 169 harmless pairs, 300 MMLU questions per language)

Gate opening on the test set: Hindi harmful 47.3%, English harmful 92.1%, Hindi harmless 0%, English harmless 1.2%, MMLU 0% in both
languages.

| Set | Measure | Unsteered | R3 | Change | 95% CI |
|---|---|---|---|---|---|
| Hindi harmful | explicit refusal | 46.5% | 61.1% | **+14.6** | +11.1, +18.3 |
| Hindi harmful | any refusal | 66.3% | 75.5% | +9.2 | +6.4, +12.1 |
| Hindi harmful | COMPLY | 30.4% | 22.0% | **−8.4** | −11.1, −5.9 |
| Hindi harmful | BROKEN | 3.2% | 2.5% | −0.7 | −2.0, +0.2 |
| English harmful | explicit refusal | 88.9% | 96.0% | +7.2 | +5.0, +9.7 |
| English harmful | COMPLY | 8.7% | 3.0% | −5.7 | −8.2, −3.5 |
| English harmful | BROKEN | 0.5% | 0.0% | −0.5 | |
| Hindi harmless | any refusal | 1.8% | 1.8% | +0.0 | |
| English harmless | any refusal | 4.1% | 4.7% | +0.6 | 0.0, +1.8 |
| Global-MMLU | accuracy (Hindi / English) | 42.7% / 63.0% | 42.7% / 63.0% | 0 / 0 | |

Gap in explicit refusal (English − Hindi): 42.3 → 34.9 points, narrowing **7.4** [+3.2, +11.6].

The unsteered Hindi and English rates reproduce Phase 6 exactly (46.5% and 88.9%). The test estimate (+14.6) is 81% of the development
estimate (+18.0). That is the same shrinkage as Qwen's Phase 3 (21.4 to 17.1, 80%), as expected for a setting chosen as the best of five.

## Reading

- **The diagnosis-matched push does what the diagnosis predicts.** Llama's Hindi failure is weak harm signal, so missing refusals become
  compliance (a mix of compliance and soft refusal in similar shares; see the human checks). Hindi COMPLY falls by 8.4 points and
  any refusal rises by 9.2, so roughly 8.4 points of the +14.6 explicit gain come from former compliance. The adjudicated human labels
  support this, though the COMPLY/SOFT boundary is the least reliable label. In Qwen the push
  mainly converted soft refusals into explicit ones (compliance fell only 3.7).
- **The Hindi-calibrated gate fixes Phase 6's failure to fire.** The gate now opens on 47% of Hindi harmful prompts (1% in Phase 6), and
  even the explicit–soft push (E1) works once it fires. This agrees with Aziz et al. (2026), who found that Llama needs gate calibration in
  the target language.
- **The gate is still not language-selective.** It opens on 92% of English harmful prompts, so English explicit refusal rises too (+7.2),
  from a high base. Because English is near its ceiling, the gap does narrow (7.4 points, interval above 0), which the Qwen repair did not
  achieve (1.2 points). It is still short of the registered 10.
- **No measured cost.** BROKEN does not rise, harmless refusal is unchanged, and MMLU is unchanged. The last two are weak tests, because the
  gate almost never opens on those inputs (the same caveat as Phase 3).

## Deviations and incidents

- **Addendum 2 (before stage 2):** the E-arm scaling error and the F2 sign error in the analysis code. Both were fixed, and the E arms
  rerun, before any stage 2 data existed.
- **Infrastructure:** the first stage 2 run died after 17 minutes on a network-volume I/O error while writing replies. Unread at the
  time, those partial replies are kept as `stage2_replies_partial_ioerror.jsonl` and were neither graded nor analysed. Stage 2 was rerun
  unchanged, writing to local disk (`go_2b.sh`).
- **Phase 6 label reuse (addendum 1):** 734 of 1,146 regenerated unsteered test replies were byte-identical to Phase 6 and took their
  Phase 6 label. The other 412 differ because Phase 6 batched harmful and harmless Hindi prompts together, and they were graded afresh.
- **Grading:** 920 replies graded in 19 parts, all in order. None was declined, and no part was stopped by the judge's safety filter.

## Limits

- **Reused test items:** the test items were used in Phase 6, though no Phase 7 setting was chosen on them. No fresh confirmation set or
  second language was available for Llama.
- **Single grader:** labels are from one model judge, checked against one human (blind κ below bar; 0.893 after the adopted review).
- **Weak guards:** the harmless and MMLU guards are weak because the gate rarely opens on benign inputs.
- **Scope:** one model, greedy decoding, one layer pair, and strengths from a five-arm development sweep.

## Cost

Stage 1 pod about 1.9 hours. E-rerun and stage 2 pod about 5.5 hours, including about 2 idle hours after the I/O crash, which a faulty
progress check failed to detect. All on one A40 at $0.59/h, about $4.40 in all. Both pods were terminated, and their removal was confirmed
by listing.
