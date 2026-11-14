# Phase 4 result: confirmation on Bengali and the sealed MultiJail split

**Pre-registration:** `docs/prereg/2026-11-14_v2_phase4_confirmation.md` (frozen in d28b04b before any reply was generated).
**Analysis:** `scripts/v2/phase4_analysis.py` (committed in 0d6f7c3 before any label existed); output `results/v2/phase4/confirm/confirm_results.json`.
**Labels:** `results/v2/labels/phase4_grades.csv`, `phase4_second_grader.csv`, `phase4_comprehension_bn.csv` (ids only).

## Verdict: FAIL

The primary outcome passes by a wide margin, and the gated push narrows the English minus Bengali gap. But one guard fails: BROKEN
replies on Bengali MultiJail rose by 8.1 points against a limit of 5. Under the frozen rule this is a FAIL. The second grader (Opus) gives
the same verdict for the same reason.

| Check (pre-registered) | Result | Limit | Status |
|---|---|---|---|
| **Primary:** Bengali MultiJail explicit refusal, understood requests (148) | 25.7% → 49.3%, **+23.6** [+16.2, +31.1] | ≥ +15 | pass |
| English MultiJail explicit refusal (same 148) | 49.3% → 56.1%, +6.8 [+1.4, +12.8] | ≥ −3 | pass |
| Bengali MultiJail BROKEN | 10.8% → 18.9%, **+8.1** [+2.0, +14.9] | ≤ +5 | **fail** |
| English MultiJail BROKEN | 0.0% → 0.0% | ≤ +3 | pass |
| Bengali IndicSafe harmless (62): net prompts newly refused | 0 (no refusals in either condition) | ≤ 2 | pass |
| Global-MMLU Bengali accuracy (300) | 49.7% → 49.7%, +0.0 | ≥ −2 | pass |

Points are steered minus unsteered, paired by item; brackets are paired bootstrap 95% intervals (10,000 resamples). No pair was dropped.

## Setup as run

- One run on an A40 (Runpod Secure Cloud, CA-MTL-1), 17:45–19:08 UTC, about $0.82. Pinned environment confirmed in `env.txt` (torch
  2.8.0+cu128, transformers 5.22.2, Qwen3-8B revision b968826d). 4,490 rows as specified; remote and local checksums matched before the pod
  was terminated. A 16-row smoke run on harmless and MMLU rows passed first.
- Bengali replies reached the 1,024-token limit far more often than English ones at 512 (58% against 25% unsteered). Graders judge only
  what is shown.
- **Comprehension screen (Opus, 358 Bengali translations, before any grading):** 148 of 165 MultiJail requests understood (above the
  100 switch, so the primary uses understood items), 111 of 131 IndicSafe harmful, 62 of 62 harmless. Overall same 237, partial 84,
  different 37, no_restatement 0.
- **Grading:** 959 replies graded blind by Sonnet 5.5 in 20 parts, all conditions and languages shuffled together; the other 473 gated
  replies were identical to their unsteered reply and took its label. No part was stopped and no row was declined.

## Pre-registered secondary and reported outcomes

- **Gap change (ΔBN − ΔEN), MultiJail explicit refusal:** +16.9 points [+8.8, +25.7] on the 148 understood requests (+17.6 [+9.7, +26.1]
  on all 165). The English minus Bengali explicit gap falls from 23.6 to 6.8 points. Unlike Hindi in Phase 3, the fix narrows the gap here,
  as predicted from the gate check.
- **Gate firing:** MultiJail 75.8% Bengali, 38.8% English; IndicSafe harmful 21.4% Bengali, 6.9% English; IndicSafe harmless 21.0%
  Bengali, 6.5% English; MMLU 0% in all three languages.
- **IndicSafe harmful explicit refusal:** Bengali +5.4 [+1.8, +9.9] (111 understood), English +0.9. Any-refusal unchanged in both.
- **Where the Bengali gain comes from:** explicit +23.6 but any-refusal only +3.4 [−3.4, +10.8]. Compliance falls 11.5 points, BROKEN rises
  8.1, and soft refusals turn explicit. On all 165 items: explicit +24.2, BROKEN +5.5 [−0.6, +12.1].
- **Gate-forced MMLU** (push on every question): Hindi 51.3% → 52.3%, English 64.0% → 63.7%, Bengali 49.7% → 49.3%. The push itself does not
  harm multiple-choice accuracy at 16 new tokens, which answers the Phase 3 gap where the gate never opened on MMLU.

## Second grader (Opus 5.5 on 150 Bengali replies, drawn before any result)

- Agreement 86.7%; Cohen's kappa 0.82 on the four labels, 0.88 for explicit refusal against everything else.
- With Opus labels in place of Sonnet labels on the sampled rows, the primary is +25.0 (25.7% → 50.7%) and the verdict is still FAIL
  on the same BROKEN guard.
- On the sample, the graders agreed on 15 BROKEN rows; Sonnet called 6 more BROKEN that Opus did not, and Opus 3 that Sonnet did not.

## Exploratory (not pre-registered): the BROKEN increase

On the 148 understood MultiJail Bengali requests, 18 replies moved into BROKEN under the push (8 from COMPLY, 8 from REFUSE_SOFT, 2 from
REFUSE_EXPLICIT) and 6 moved out, a net of 12. All 18 had the gate open; 7 of them hit the token limit. English had no BROKEN reply in
either condition. So the push, at the strength tuned on Hindi, damages a share of Bengali replies that it does not damage in English or,
in Phase 3, in Hindi (+0.2). Whether a lower strength would keep the explicit gain without the damage is a new question; answering it
would need tuning on Bengali development data and a new confirmation set, since the sealed split is now used.

## Limits

- Bengali labels are model labels: grader not calibrated on Bengali. The two graders agree well, but both are models.
- One model, one run. MultiJail is English-origin red-teaming translated into Bengali.
- The harmless guard passed trivially: no harmless Bengali prompt was refused in either condition, so it says nothing about how close the
  push comes to over-refusal.
- The sealed MultiJail split and the Bengali IndicSafe items are now used; any revised fix needs its own confirmation set.
