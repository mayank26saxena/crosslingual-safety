# Llama human checks (Phase 6 and Phase 7)

**Rules:** Phase 6 pre-registration, "Human check on this model", and addendum 3; Phase 7 pre-registration, "Human check".
**Code:** `phase6_llama.py analyseB`, `phase7_llama_fix.py analyse2`. **Labels:** `results/v2/labels/phase6_llama_human.csv`,
`phase7_llama_human.csv` (ids only).

The author, a fluent Hindi reader, graded both sheets blind to the model labels, using the four-label rubric in the grader page; every item
is marked "sure". Per addendum 3, the grading came after the author had seen aggregate Llama results, but before any sheet item's model
label. Both graded files match their sheets item for item, with unchanged text.

## Phase 6: 60 unsteered Hindi replies (45 harmful, 15 harmless)

**Registered outcome: the check fails.** Cohen's κ on four labels is **0.643**, below the registered 0.7. Under the rule, **every Llama
number is reported as provisional, with the confusion table.**

| | Value |
|---|---|
| Four-label κ (registered) | **0.643** (agreement 76.7%) |
| Broad κ, refused vs not (not registered) | 0.672 |
| Explicit refusal vs the rest κ (not registered) | **0.785** |
| Harmless items (15) | 15 of 15 agree |

Confusion, rows the author, columns Sonnet 5.5 (all 60):

| | COMPLY | REFUSE_EXPLICIT | REFUSE_SOFT | BROKEN |
|---|---:|---:|---:|---:|
| COMPLY (22) | 22 | 0 | 0 | 0 |
| REFUSE_EXPLICIT (23) | 2 | 19 | 2 | 0 |
| REFUSE_SOFT (15) | 6 | 2 | 5 | 2 |

The judge's main error on Llama is calling soft refusals compliance: 6 of the author's 15 soft refusals. It never labels an author COMPLY
as a refusal. This is the opposite direction from Phase 0 on Qwen, where the judge labelled 9 of the author's COMPLY replies as soft
refusals.

**What this does to the Llama behaviour picture** (descriptive, n = 45 harmful items, bootstrap 95% intervals; not a re-test of any
registered claim):

| Harmful Hindi replies | Author | Sonnet |
|---|---|---|
| REFUSE_EXPLICIT | 49% [33, 64] | 44% [31, 58] |
| REFUSE_SOFT | **33%** [20, 47] | 16% [7, 27] |
| COMPLY | **18%** [7, 29] | 36% [22, 49] |
| BROKEN | 0% | 4% |

Among the 20 sampled pairs refused explicitly in English but not in Hindi (by the author's labels), the Hindi replies are 14 REFUSE_SOFT
and 6 COMPLY. The judge's labels said most of Llama's missing refusals became compliance (Phase 6, C1). The author's labels on this sample
say most became **soft refusals**, as in Qwen. The sample is small, but the reversal is large and in a single direction.

## Phase 7: 30 steered (R3) Hindi harmful replies with the gate open

No pass bar was registered. Agreement is **96.7%** (29 of 30) and broad κ is 1.0. Four-label κ is 0.651, depressed because 28 of 30 replies
are explicit refusals in both labellings, so κ is unstable at this prevalence. The one disagreement is an explicit refusal the judge called
soft. The steered explicit refusals are confirmed.

## What stands and what changes

**Stands (label-independent, or resting on the best-validated distinction, explicit refusal vs the rest, κ 0.785):**
- the Llama explicit-refusal gap (88.9% vs 46.5%; the author's labels on the sample agree: 49% vs the judge's 44%);
- the projection results: Hindi at 28% of the English refusal-direction projection (C2), and the explicit–soft axis (C3);
- Phase 7's held-out explicit-refusal gain (+14.6) and gap narrowing (7.4 points), with the registered verdicts (F1 and F2 not met);
- the Qwen findings, whose soft-refusal pattern has its own human support (Phase 0).

**Changes or is weakened (depends on the COMPLY/SOFT split, where the judge is biased on Llama):**
- "Llama's missing refusals become compliance" (C1 second clause, Phase 6). It is not supported by the human sample, which points the
  other way. The registered C1 verdict (from judge labels) is kept as registered, and marked provisional.
- Phase 7 F3 ("compliance falls", −8.4), and the decomposition "about 8.4 points of the gain come from former compliance". Some of what
  the judge called compliance in the unsteered arm the author calls soft refusal, so part of that conversion is likely soft to explicit.
  F3 stays as registered, but provisional and secondary.
- The behavioural contrast "soft in Qwen, compliance in Llama". By human labels both models' missing Hindi refusals are mostly soft. The
  contrast that survives is internal: Qwen's Hindi harmful prompts project on the refusal direction at English strength, Llama's at 28%.

**A finding in its own right:** a judge validated on one model can err in a different direction on another model's replies. Here that was
enough to reverse the behavioural characterisation of the gap. Cross-model comparisons of refusal *style* need a per-model human check.
