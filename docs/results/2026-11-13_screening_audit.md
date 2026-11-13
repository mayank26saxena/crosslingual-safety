# Screening audit: human against claude-opus-5-5 and claude-fable-5-2

**Date:** 2026-11-13. **Rules:** `docs/plan/2026-11-11_hindi_dataset_spec.md` (human audit; addenda 2026-11-13). **Prompt:**
`docs/protocols/2026-11-13_screening_prompt.md`. **Code:** `scripts/v2/screen_agreement.py`. **Output:** `results/v2/screening/audit_agreement.json`.

## Inputs

- Human: one fluent Hindi reader, 100 blind items, all marked sure. Request type: harmful 55, benign 25, borderline safe 12, not a request 8.
  Fidelity: equivalent 87, minor drift 8, wrong 5.
- Models: two 50-row Claude Code subagents per model, self-reported `claude-opus-5-5` and `claude-fable-5-2`. All 100 labelled by both; none DECLINED.
  Subagent tokens for the 100 items: Opus about 163k, Fable about 165k.

## Agreement with the human (all 100)

| | Request type: agreement, kappa (95% CI) | Harmful vs rest: agreement, kappa | Fidelity: agreement, kappa | Wrong vs rest: agreement, kappa |
|---|---|---|---|---|
| claude-opus-5-5 | 68%, 0.51 (0.38–0.64) | 76%, 0.53 | 81%, 0.38 | 95%, 0.59 |
| claude-fable-5-2 | 67%, 0.49 (0.36–0.62) | 78%, 0.56 | 83%, 0.44 | 96%, 0.65 |

Per tier, request-type kappa: tier A 0.42 for both; tier B 0.78 for both; tier C 0.43 (Opus) and 0.39 (Fable).

## Gate

**Fails for both candidates.** The selection rule needs request-type kappa of at least 0.75, and the acceptance criterion needs at least 90% request-type
agreement. Neither model is selected, and the dataset stays a candidate.

## What drives the gap

- **The two models agree with each other on 94 of 100 items** (98 of 100 on harmful against the rest), but with the human on about 67. Where they
  agree, the human differs on 29 items. That pattern points to a difference in where the categories are drawn, not to noise in either model.
- The main differences are: human harmful, models borderline safe (7); human borderline safe, models benign (5); human harmful, models not a request (5);
  human not a request, models harmful (3); human harmful, models benign (3).
- This mirrors Phase 0, where the first refusal rubric disagreed with the human mainly on one boundary (warn or redirect without saying no) and was
  resolved by making that boundary explicit.

## Options (not pre-registered; for decision)

1. **Adjudication.** The human reviews the 29 items where both models agree against the human label, with the model label shown, and keeps or changes
   each. Agreement is then reported before and after adjudication, labelled as such.
2. **Rubric revision.** Write sharper boundaries (harmful against borderline safe; request against statement), have the models relabel, and test on a
   fresh human sample, since the current 100 have been seen.
3. **Restrict the study to tier A** without automated screening, with request type for the 1,069 harmful items labelled by the human or by both models
   in agreement, and report the baseline with and without that filter.

## After adjudication (2026-11-13)

Procedure pre-registered in the spec addendum "adjudication of the screening audit". The human reviewed the 29 items where both models agreed against
the human label, with the model label and reasons shown, and changed 24 of them (to the models' label in each case); 5 were kept. Human request-type
counts after review: harmful 51, benign 32, not a request 11, borderline safe 6. Output: `results/v2/screening/audit_agreement_adjudicated.json`.

| After adjudication | Request type: agreement, kappa (95% CI) | Harmful vs rest: agreement, kappa | Tier A request type |
|---|---|---|---|
| claude-opus-5-5 | **92%, 0.88 (0.80–0.95)** | 92%, 0.84 | 90%, 0.81 |
| claude-fable-5-2 | **91%, 0.86 (0.78–0.94)** | 94%, 0.88 | 90%, 0.81 |

- **Both candidates pass after adjudication** (kappa at least 0.75; at least 90% request-type agreement). Before adjudication both failed.
- **Selected: claude-opus-5-5.** By the pre-registered cost measure (subagent tokens), Opus used about 163k and Fable about 165k for the 100 items,
  so Opus is the cheaper candidate; it also has the higher request-type agreement. The difference in cost is small.
- **Caveat.** The adjudicated labels were set with the model labels in view, and the human moved to the model label on 24 of 29 reviewed items. The
  post-adjudication agreement is therefore an upper estimate, and both the before and after figures are reported. The remaining disagreements are
  mostly items the human calls harmful and Opus calls borderline safe (6).
