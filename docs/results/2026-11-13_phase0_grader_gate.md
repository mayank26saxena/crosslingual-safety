# Phase 0: grader gate on the Hindi calibration sheet

**Date:** 2026-11-13. **Rules:** `docs/prereg/2026-11-10_v2_measurement_and_data.md` (addenda "Hindi first", "four-way refusal labels",
"calibration replies at 1,024 tokens") and `docs/protocols/2026-11-11_hindi_calibration_grading.md`. **Code:** `scripts/v2/grader_gate.py`.
**Output:** `results/v2/grader_gate/gate_results.json` (ids and numbers only).

## Inputs

- **Human:** one fluent Hindi reader, four labels, all 150 items of `hindi_calibration_sheet_1024.csv`, all marked "sure". The graded sheet is kept
  under `notes/` (sha256 `8ee0d850…`); its replies match the 1,024-token sheet on all 150 items.
- **Human label counts:** COMPLY 80, REFUSE_SOFT 36, REFUSE_EXPLICIT 30, BROKEN 4.
- **Model graders:** the four valid Kiro runs from 2026-11-11 (`notes/hindi_calibration/model_runs/FINDINGS.md`): claude-haiku-5.5,
  claude-sonnet-5.5, gpt-5.6-terra, gpt-5.6-luna. They graded blind on 300 shuffled rows (Hindi and English); only the 150 Hindi rows enter the gate.
  Excluded runs: claude-opus-5.5 (served by claude-opus-4.8 on both attempts) and deepseek-3.2 (labelled by keyword scripts).
- **Halves:** from `key.json`: tune 74, gate 76. The protocol targeted 75 each; the stored assignment is used as is.
- No model grade is missing on the Hindi rows.

## Agreement with the human

Cohen's kappa, with 95% bootstrap intervals over items (2,000 resamples).

| Grader | Broad, tune | **Broad, gate** | Broad, all 150 | Strict, gate | Four labels, gate |
|---|---|---|---|---|---|
| claude-haiku-5.5 | 0.55 (0.37–0.73) | 0.64 (0.48–0.79) | 0.60 | 0.91 | 0.51 |
| **claude-sonnet-5.5** | **0.81 (0.65–0.94)** | **0.82 (0.68–0.92)** | **0.81** | 0.91 | 0.62 |
| gpt-5.6-terra | 0.76 (0.60–0.89) | 0.76 (0.61–0.90) | 0.76 | 0.84 | 0.64 |
| gpt-5.6-luna | 0.52 (0.32–0.70) | 0.69 (0.51–0.84) | 0.60 | 0.91 | 0.65 |
| Plurality of the four (ties dropped) | 0.65 (n 62) | 0.77 (n 70) | 0.72 | 0.91 | 0.68 |

Broad: REFUSE_EXPLICIT or REFUSE_SOFT against COMPLY or BROKEN (the gate). Strict: REFUSE_EXPLICIT against the rest.

## Selection and gate

- **Selected grader: claude-sonnet-5.5.** It has the highest broad kappa on the gate half, the rule as written. It is also highest on the tuning half and on
  all 150, so the choice does not depend on which half is used.
- **Gate: passes.** Broad kappa on the gate half is 0.82, above 0.7. The interval (0.68–0.92) dips just below 0.7, as expected with 76 items.
- gpt-5.6-terra also clears 0.7 (0.76). Haiku and luna do not.

## Where the selected grader and the human differ

Confusion on all 150 (rows human, columns claude-sonnet-5.5):

| | COMPLY | REFUSE_EXPLICIT | REFUSE_SOFT | BROKEN |
|---|---:|---:|---:|---:|
| COMPLY (80) | 48 | 0 | 9 | 23 |
| REFUSE_EXPLICIT (30) | 1 | 27 | 0 | 2 |
| REFUSE_SOFT (36) | 2 | 4 | 30 | 0 |
| BROKEN (4) | 2 | 0 | 0 | 2 |

- **The refusal boundary agrees well:** broad agreement is 91%. The main broad disagreement is 9 replies the grader calls soft refusals and the human calls
  compliance.
- **COMPLY against BROKEN is the main four-way disagreement.** The grader calls 23 of the human's COMPLY replies BROKEN (10 harmful, 13 harmless by
  benchmark label; 13 of the 23 hit the 1,024-token limit). The human used BROKEN 4 times, the grader 27. This does not affect the broad or strict
  binary, which both count BROKEN as not refused. It does mean the grader's COMPLY and BROKEN shares are not interchangeable with the human's. Downstream,
  refusal rates are the measures; COMPLY and BROKEN are reported together as "not refused", with the split shown as grader-labelled.
- **Human review after unblinding (2026-11-13):** the human read the 18 harmful pairs refused in English but not in Hindi, and agreed that several replies
  they had labelled COMPLY are better labelled BROKEN. The human grades used for the gate are not changed, because they were revised only after the
  grader's labels were seen. The gate is unaffected either way: BROKEN and COMPLY are on the same side of both binaries.

## Model references

The addendum names Claude Fable 5.2 and Claude Opus 5.5 as model references. Neither has a valid run: Opus 5.5 fell back to Opus 4.8 twice, and Fable was
not run (it stopped on this content before). So the condition for letting references grade Hindi items without a human does not arise.

## Paired English-Hindi readout (exploratory, selected grader)

On the 150 calibration requests. These were stratified by an earlier judge's verdicts, so the shares illustrate the gap and do not estimate it.

| Benchmark label | Definition | Refused, EN | Refused, HI | EN only / HI only |
|---|---|---:|---:|---:|
| Harmful (103) | Broad | 78 | 64 | 18 / 4 |
| Harmful (103) | Strict | 56 | 28 | 33 / 5 |
| Harmless (47) | Broad | 11 | 6 | 6 / 1 |
| Harmless (47) | Strict | 10 | 3 | 7 / 0 |

On harmful requests, explicit refusal halves in Hindi (56 to 28) while broad refusal falls less (78 to 64). Many explicit refusals in English become soft
refusals in Hindi (36 soft in Hindi against 22 in English). BROKEN rises from 2 to 13.

## Consequences

- Phase 0 is closed for Hindi. Tier A (1,174 pairs) and the PolyGuard harmless Hindi and English replies are graded with claude-sonnet-5.5, one pinned Kiro
  session, with `resolved_model` recorded.
- The Spanish result (`2026-11-12_spanish_paired_provisional.md`) already used claude-sonnet-5.5. Its grader is now validated on Hindi only; Spanish has no
  human gate, so it stays labelled "grader validated on Hindi".
- The Phase 2a behaviour link for Hindi used a four-grader majority. It can be recomputed with the selected grader once tier A is graded.
