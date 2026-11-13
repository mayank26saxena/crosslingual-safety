# Specification: a Hindi refusal-gap dataset

**Date:** 2026-11-11. **Status:** the construction rules below are fixed before any item is screened or any reply is generated under them. This dataset is
the first deliverable of plan v2 and comes before diagnosis or repair.

**Purpose:** paired English and Hindi requests, plus Bengali for the transfer test and romanised Hindi as a supplement. Each item is checked to be a real
request, faithfully translated, understood by the model, and labelled with provenance. This lets a refusal gap be measured without translation artefacts,
label noise or truncation driving the result.

## What the current data shows

Inventory of the manifest's 37,108 Hindi rows:

| Source | Hindi rows | Authorship | Concern |
|---|---:|---|---|
| IndicGuard | 28,329 | Google-translated from English | Translation quality unknown |
| IndicGuard, culture-adaptive subset | 7,334 | Hindi-origin per the dataset README, then machine-translated to other languages; built with the CultureGuard method, which adapts English safety data with LLMs | Hindi-origin but most likely LLM-adapted, not human-written. The English side is a translation |
| Multilingual Harmless-Harmful | 832 | Machine-translated, partly checked | Small |
| IndicSafe | 497 | Human-translated | Only verified human-translated source |
| SurakshaEval | 116 | Human-written | Labels unverified |

- All rows are in Devanagari; there is no romanised Hindi.
- A rough heuristic on the English side of IndicGuard "harmful" prompts finds about 41% that do not begin as a question or instruction (7,904 of 19,219).
  Many are statements, insults or venting. These are unsafe content in a moderation sense, but declining is not clearly the right response, which helps
  explain why the model refuses only 58% of IndicGuard "harmful" prompts even in English.
- Step 7 replies were capped at 400 new tokens, about 430 to 520 Hindi characters with this tokenizer.

## Unit and fields

One item is a request with an English version and a Hindi version, plus Bengali and romanised Hindi where available. Fields:

- **Provenance:** source dataset and id, authorship (native-written, human-translated, machine-translated), licence.
- **Request type:** `request_harmful` (asks for help with something harmful), `request_borderline_safe` (sounds alarming but is harmless),
  `request_benign`, or `not_a_request` (statement, insult, venting, dialogue fragment).
- **Harm category and severity** (low, medium, high), for `request_harmful` only, on one taxonomy mapped from the source categories.
- **Translation fidelity:** back-translation chrF, and a meaning-equivalence rating of the English and Hindi pair (equivalent, minor drift, wrong).
- **Comprehension:** whether Qwen3-8B restates the Hindi request correctly in English.
- **Model behaviour:** Qwen3-8B's English and Hindi decisions (REFUSE or ANSWER) from greedy replies, graded by the calibrated judge, plus a flag if a reply
  hit the token limit.

## Tiers

| Tier | Contents | Role |
|---|---|---|
| A, human-authored | Aya Red-teaming Hindi (prompts written by paid native annotators, with human literal and semantic English translations); IndicSafe (human-translated); SurakshaEval (human-written), after label review | Primary test set: no machine-translation artefacts |
| B, Hindi-origin, LLM-adapted | IndicGuard culture-adaptive rows (human prompt labels) | Test and fit, reported separately from A |
| C, machine-translated, screened | IndicGuard and Multilingual Harmless-Harmful translations that pass every screen | Fit and development; a secondary test set |

Results are reported per tier. A gap that appears in tier C but not tier A points to translation, not safety.

## Screens, in order

1. **Exclusions:** the sealed split, every v1 item and its group, exact and near duplicates (same group or normalised-text match), and empty or redacted rows.
2. **Request type:** every candidate is classified by a model reference. Only `request_harmful` and `request_borderline_safe` or `request_benign` items enter
   the refusal study. `not_a_request` items are kept in a separate file and reported, not deleted.
3. **Translation fidelity:** a pair is dropped if it is rated `wrong`, or if its chrF is below the tier's 20th percentile and it is not rated `equivalent`.
   Tier A items are rated but never dropped on fidelity alone. A drop is reported.
4. **Comprehension:** Qwen3-8B restates the Hindi request in English, and a model reference judges whether the meaning matches. Items judged not understood
   go to a separate "not understood" set, which is evaluated and reported on its own.
5. **Balance:** within each tier, harmful requests are stratified by category and severity, and harmless requests are drawn to match harmful ones on topic
   area and length where possible.

## Model references and human audit

- The screening model for steps 2 to 4 is chosen empirically. On a 100-item human audit (request type and translation fidelity, graded by a fluent Hindi
  reader), the candidates `claude-opus-5-5` and `claude-fable-5-2` are compared. The cheaper one is used for bulk screening if its kappa with the human is at
  least 0.75, and the other is used otherwise. Its kappa is reported.
- Romanised Hindi has a separate 50-item human naturalness check.
- Refusal grading uses the judge calibrated in Phase 0.

## Generation settings for behaviour labels

- Greedy, 1,024 new tokens for Hindi, Bengali and romanised Hindi, and 512 for English.
- The share of replies that hit the limit is reported per language.

## Sizes and splits

- **Target after screening:** at least 3,000 Hindi `request_harmful` items and 1,500 harmless requests across tiers. Splits are group-level and stratified
  by tier, category and English decision: fit 60%, development 20%, test 20%.
- **Tier A** is test-only, unless it holds more than 1,000 items. In that case half goes to development.
- Bengali versions of test items form the transfer test set.

## Acceptance criteria

The dataset is a **candidate** until every criterion holds. No experiment runs on a candidate dataset.

1. **Size, after all screens:** at least 800 tier-A `request_harmful` items, and at least 400 harmless requests that are human-written or human-checked.
2. **Screening agreement:** on the human audit, the screening model agrees with the human on at least 90% of request-type labels. On translation
   fidelity for kept tier B and C items, it agrees on at least 90%.
3. **Refusal measurement:** the judge passes the Phase 0 gate (kappa at least 0.7 against human Hindi grades).
4. **Pilots read:** every stage's 50-item pilot has been read before its full run, and the reading is recorded in the stage summary.
5. **Comprehension:** at least 80% of tier A passes the comprehension screen. If fewer do, the shortfall is reported with its causes before the
   dataset is accepted.
6. **Reporting:** retention after every stage is reported per tier and per source, with the reason for every removal.

**Harmless controls:** if criterion 1 is short on harmless requests, the gap is filled with harmless counterparts of tier-A harmful prompts. Each keeps
the topic and register but makes a safe request. They are drafted by a model reference, and a fluent Hindi reader checks at least 25% of them, with the
rest used only if at least 90% of the checked sample passes.

## Outputs

- `data/v2/hindi_items.jsonl`: one row per item with every field above (ignored by git: it contains harmful text).
- `data/v2/summary.json`: counts by tier, request type, category, screen outcome and split (tracked).
- `docs/data/hindi_dataset_card.md`: sources and licences, construction, screens and retention, human audit agreement, and known limits (tracked).

## Open items before construction

1. Done, 2026-11-11: IndicGuard's README states culture-adaptive content was sourced in Hindi; its cited method (CultureGuard) adapts data with LLMs, so it is tier B, not A. Prompt labels are human for the generic and culture-adaptive subsets and LLM-jury for the jailbreaking subset, which is excluded from tier C labels unless re-screened.
2. Approval to download a native-written Hindi red-teaming set and the PolyGuard Hindi evaluation data. Sizes and licences are checked first.
3. API access and a cost estimate for model-reference screening, measured on a 100-item sample before any bulk run.
4. A 100-item human audit sheet for request type and translation fidelity, built the same way as the calibration sheet.

## Addendum, 2026-11-13: the 100-item human audit sample

Written before the sample is drawn. Pool: every item that passed stage 1 exclusions (`data/v2/after_exclusions.jsonl`). A seeded (20261113) stratified
sample of 100, weighted toward tier A, which is the primary test set, and covering every source and label used later:

| Tier | Source / label | n |
|---|---|---:|
| A | Aya harmful | 25 |
| A | IndicSafe harmful / harmless / ambiguous | 8 / 4 / 3 |
| B | IndicGuard culture-adaptive harmful / harmless | 12 / 8 |
| C | IndicGuard generic harmful / harmless | 10 / 6 |
| C | IndicGuard jailbreak harmful / harmless | 6 / 4 |
| C | PolyGuard harmful / harmless | 5 / 5 |
| C | Multilingual Harmless-Harmful harmful / harmless | 2 / 2 |

The human sees each item's English and Hindi text under a shuffled id, without tier, source or benchmark label, and records request type (the four
types above), translation fidelity (equivalent, minor drift, wrong), confidence and notes. The same sheet is then labelled by each screening candidate
under the same blinding. Agreement is computed on all 100 and per tier.

## Addendum, 2026-11-13: screening-model comparison

Written after the human audit was exported and before any model labels exist. Candidates: `claude-opus-5-5` and `claude-fable-5-2`, each run as
Claude Code subagents on the 100 blind items with `docs/protocols/2026-11-13_screening_prompt.md`. Reported per candidate, on all 100 and per
tier: agreement and Cohen's kappa with the human on request type (four values) and on the binary used for screening (`request_harmful` against
the rest), and on fidelity (three values and `wrong` against the rest), with 95% bootstrap intervals. Selection follows the rule above: the cheaper
candidate is used if its request-type kappa is at least 0.75, the other otherwise. Cost on this route is measured as subagent tokens per item,
reported for both. The acceptance criterion (at least 90% request-type agreement) is applied to the selected model.
