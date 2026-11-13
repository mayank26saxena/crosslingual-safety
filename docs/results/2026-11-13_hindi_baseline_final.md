# Hindi baseline after screening (test split)

**Date:** 2026-11-13. **Rules:** `docs/plan/2026-11-11_hindi_dataset_spec.md` (screens; addenda on the audit, adjudication, screening comparison and
split stratification). **Code:** `scripts/v2/build_screen_folder.py`, `scripts/v2/final_baseline.py`. **Outputs:** `results/v2/final_baseline/`
(`screen_summary.json`, `split.json` with ids only, `baseline_test.json`).

**Status:** all screens applied (request type, fidelity, comprehension). Updated the same day: 25 items without an English version removed, split
redrawn once (final), comprehension screen added. Behaviour labels are the
claude-sonnet-5.5 grades from the raw baseline (`2026-11-13_hindi_baseline_raw.md`).

## Screening

All 1,474 baseline items (tier A 1,174; PolyGuard harmless 300) were labelled by claude-opus-5-5 as 30 Claude Code subagents (all self-reported
`claude-opus-5-5`, every row in order, none DECLINED), with `docs/protocols/2026-11-13_screening_prompt.md`.

| Source / benchmark label | Kept | Screened out |
|---|---:|---|
| Aya harmful (914) | 769 | 107 not harmful (borderline safe or benign), 38 not a request |
| IndicSafe harmful (155) | 140 | 15 not harmful |
| IndicSafe harmless (62) | 62 | 0 |
| PolyGuard harmless (300) | 286 | 8 not a request, 3 judged harmful, 3 translation wrong |
| IndicSafe ambiguous (43) | — | reported separately (all screened benign) |

Fidelity on tier A (rated, never dropped): equivalent 964, minor drift 136, wrong 74.

## Split

Group-level, stratified by source, English decision and the English and Hindi explicit-refusal labels; the 150 Phase 2b steering items are in
development. Harmful: 461 development, 448 test. Harmless: 173 development, 175 test. A first split, stratified only by source and English
decision, drained the test split of the failure pattern (explicit gap 21.9 points against 32.3 overall) and is not used (see the addendum).

## Baseline

| Harmful requests | Pairs | Any refusal, EN → HI | Gap (95% CI) | Explicit refusal, EN → HI | Gap (95% CI) |
|---|---:|---|---|---|---|
| **Test split** | 448 | 89.3% → 77.5% | **11.8** (7.6–16.1) | 51.3% → 18.3% | **33.0** (27.7–38.2) |
| Test, Aya | 376 | 89.1% → 76.1% | 13.0 (8.8–17.5) | 53.7% → 17.8% | 35.9 (30.3–41.8) |
| Test, IndicSafe | 72 | 90.3% → 84.7% | 5.6 (−4.2 to 15.3) | 38.9% → 20.8% | 18.1 (4.2–30.6) |
| All screened (population estimate) | 909 | 87.4% → 75.2% | 12.1 (9.1–15.1) | 50.4% → 18.0% | 32.3 (28.7–35.9) |

Test split, harmful: refused in English only 76 / in Hindi only 23 (any refusal, McNemar p 8.5e-08); 173 / 25 (explicit, p 2.0e-28). Labels, English →
Hindi: COMPLY 47 → 63, REFUSE_EXPLICIT 230 → 82, REFUSE_SOFT 170 → 265, BROKEN 1 → 38.

Harmless controls, test split (175): refused 3.4% in English and 1.1% in Hindi; BROKEN 0 → 20.

**Screened-out harmful items behave differently:** among the 122 judged not harmful, the model refuses 54% in English (any refusal), against 87% for
kept items.

## Reading

- The screened gap matches the raw one (raw: 12.4 and 31.8 points). Screening removed items the model was less likely to refuse in either language;
  it did not create or remove the gap.
- On the test split, explicit refusal falls by about two thirds in Hindi (51% to 18%) and any refusal by 12 points. Most of the change is a move from
  explicit to soft refusal, with smaller rises in compliance and broken replies.
- No over-refusal of harmless Hindi requests. The rise in BROKEN replies on harmless Hindi requests (0 to 20 of 175) is reported as a quality cost.
- This test split (448 harmful, 175 harmless) is the reference for Phase 3.

## Limits

- Comprehension screen pending.
- Screening by one model, validated against the human audit after adjudication (92% request-type agreement, an upper estimate).
- Behaviour labels from one model grader (claude-sonnet-5.5; kappa 0.82 with the human through Kiro, 0.74 through the subagent route).
- 571 of 1,174 tier A Hindi replies reach the 1,024-token limit.

## Update (2026-11-13): missing English versions, final split, comprehension screen

- **No English version.** 25 Aya items have "N/a" (or a variant) in place of an English translation, so their English prompt was that placeholder.
  They are excluded from the paired study (spec addendum). The split was redrawn once with the same rule and seed (611 of 623 test items unchanged)
  and is final. Harmful test pairs: 447.
- **Comprehension** (`scripts/v2/comprehension.py`; Qwen3-8B restates each Hindi request in English, revision and environment in
  `results/v2/comprehension/env.txt`; claude-opus-5-5 judges each restatement against the English version, 26 subagent parts, all self-reported
  `claude-opus-5-5`): of 1,257 items, same 914, partial 231, different 96, no restatement 11, declined 5 (no usable English reference).
  **Tier A harmful pass rate 90.9%**, above the 80% acceptance criterion.

| Harmful requests, test split | Pairs | Any refusal, EN → HI | Gap (95% CI) | Explicit refusal, EN → HI | Gap (95% CI) |
|---|---:|---|---|---|---|
| **Understood (final baseline)** | 404 | 90.3% → 77.2% | **13.1** (9.2–17.3) | 51.2% → 19.1% | **32.2** (26.5–37.4) |
| All test pairs | 447 | 89.5% → 76.5% | 13.0 (9.2–17.2) | 51.4% → 18.3% | 33.1 (27.5–38.5) |
| Not understood | 43 | 81.4% → 69.8% | 11.6 (0.0–23.3) | 53.5% → 11.6% | 41.9 (23.3–60.5) |
| All screened (population, 886) | 886 | 89.6% → 76.0% | 13.7 (10.7–16.8) | 51.7% → 18.2% | 33.5 (29.7–37.0) |

Harmless test pairs understood (169): refused 3.5% in English and 1.2% in Hindi.

**Final baseline for Phase 3: the 404 understood harmful test pairs and 169 understood harmless test pairs.** The gap does not come from
misunderstanding: items the model understands show the same gap as those it does not.
