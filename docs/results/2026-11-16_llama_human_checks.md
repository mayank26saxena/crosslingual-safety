# Llama human checks (Phase 6 and Phase 7)

**Rules:** Phase 6 pre-registration, "Human check on this model", and addendum 3; Phase 7 pre-registration, "Human check".
**Code:** `phase6_llama.py analyseB`, `phase7_llama_fix.py analyse2`. **Labels:** `results/v2/labels/phase6_llama_human.csv`,
`phase7_llama_human.csv` (ids only).

The author, a fluent Hindi reader, graded both sheets blind to the model labels, using the four-label rubric in the grader page; every item
is marked "sure". Per addendum 3, the grading came after the author had seen aggregate Llama results, but before any sheet item's model
label. Both graded files match their sheets item for item, with unchanged text.

## Phase 6: 60 unsteered Hindi replies (45 harmful, 15 harmless)

**Registered outcome: the check fails.** Cohen's κ on four labels is **0.643** [0.48, 0.79], below the registered 0.7 (the interval spans the bar). Under the rule, **every Llama
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
and 6 COMPLY. The judge's labels said most of Llama's missing refusals became compliance (Phase 6, C1). The author's *blind* labels on this sample
said most became **soft refusals**. The review below revises this. The sample is small, but the reversal is large and in a single direction.

## After review (adjudication, not registered; addendum 4)

The author reviewed the 14 Phase 6 disagreements with the judge's label and reason in view, and changed 10, all to the judge's label.
The reasons given follow the rubric: replies that warn and then answer are COMPLY, and incoherent replies are BROKEN. The blind result above
is the registered one and still decides: the check is below bar, and the Llama numbers stay provisional.

| | Blind (registered) | After review (upper estimate) |
|---|---|---|
| Four-label κ, all 60 | **0.643** | 0.893 |
| Harmful 45: REFUSE_EXPLICIT | 49% | 42% [29, 58] |
| Harmful 45: REFUSE_SOFT | 33% | 20% [9, 31] |
| Harmful 45: COMPLY | 18% | 33% [20, 47] |
| Harmful 45: BROKEN | 0% | 4% [0, 11] |
| Lost refusals (English explicit, Hindi not) | 14 soft, 6 comply (n 20) | **13 comply, 9 soft, 1 broken** (n 23) |

After review, the author's labels match the judge's picture: compliance and soft refusal occur in similar shares among Llama's lost Hindi
refusals (13 against 9 here; 88 against 75 by judge labels on all 172, p = 0.35). The blind disagreement came mainly from the author applying the
rubric's first clause inconsistently, not from a judge bias. The COMPLY/SOFT boundary, a reply that warns and then answers, is the hard
case even for a careful human. Phase 7's single disagreement was likewise resolved to the judge's label, so the author check agrees on 30 of
30 after review (29 of 30 blind).

**What this means for the claims:** the reversal described in the blind section does not hold after review. The supported Llama
characterisation is a *mix* of compliance and soft refusal, with compliance slightly the larger share. Llama complies with Hindi harmful
requests 1.4–1.9 times as often (compliance 30.2% vs 16.3%, +13.9 [+9.4, +18.3]; compliance plus broken 33.2% vs 24.0%, +9.2 [+4.5, +14.1]) on the same 404 pairs (judge labels). Phase 7's F3 (compliance falls) and its
decomposition are consistent with the adjudicated labels, and stay registered and provisional.

## Phase 7: 30 steered (R3) Hindi harmful replies with the gate open

No pass bar was registered. Agreement is **96.7%** (29 of 30) and broad κ is 1.0. Four-label κ is 0.651, depressed because 28 of 30 replies
are explicit refusals in both labellings, so κ is unstable at this prevalence. The one disagreement is an explicit refusal the judge called
soft. The steered explicit refusals are confirmed.

## What stands and what changes (final, after review)

**Decision (Phase 6 addendum 5):** the author adopts the reviewed labels as the final human reference. Agreement with the judge is then
0.893 (upper estimate). The registered blind result (0.643, below bar) is always reported alongside.

**Stands:**
- The registered verdict, reported as registered: the blind check is below bar (κ 0.643).
- The label-independent results: Hindi at 28% of the English refusal-direction projection (C2), and the explicit–soft axis (C3).
- The explicit-refusal results, resting on the best-validated distinction (blind explicit-vs-rest κ 0.785): the Llama gap (88.9% vs 46.5%)
  and Phase 7's +14.6 and 7.4-point gap narrowing, with F1 and F2 not met as registered.
- The Qwen findings, with their own human support (Phase 0).

**Characterisation of Llama's lost refusals:** a mix of compliance and soft refusal, in similar shares (judge 88 vs 75 of 172, p = 0.35; reviewed sample 13 vs 9). This is not the clean "soft in Qwen, compliance in Llama" dichotomy. The supported
contrast is one of *rate*: Llama fails to refuse Hindi harmful requests 1.4–1.9 times as often (compliance 30.2% vs 16.3%, +13.9 [+9.4, +18.3]; compliance plus broken 33.2% vs 24.0%, +9.2 [+4.5, +14.1]) on the same pairs.
It is joined by the internal contrast (projection at English strength in Qwen, 28% in Llama).

**Phase 7 F3** (Hindi compliance falls, −8.4) is consistent with the adjudicated labels, and stays as registered (provisional).

**Superseded by the review above.** An earlier version of this note called the blind disagreement a judge bias. After review, most of it
was the author's rubric application. What remains is a methods point: the COMPLY/SOFT boundary is fragile, so both blind and adjudicated
labels are reported.
