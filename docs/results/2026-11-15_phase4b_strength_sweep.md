# Phase 4b result: Bengali strength sweep (diagnosis)

**Pre-registration:** `docs/prereg/2026-11-14_v2_phase4b_bengali_strength_sweep.md` (frozen in e20cb4b; addendum d99b3b9 before any
result). **Analysis:** `scripts/v2/phase4b_sweep.py analyse`; output `results/v2/phase4b/sweep_results.json`.
**Labels:** `results/v2/labels/phase4b_sweep_grades.csv` (ids only).

## Outcome: not reproduced

On the 150 MultiJail development requests in Bengali, the Hindi-tuned strength (α = 1.25) raised BROKEN replies by +0.9 points, not
the +8.1 seen on the sealed split in Phase 4. Under the frozen rule this is "not reproduced" (B(1.25) ≤ +0.05, and also ≤ +0.02).

| α | Explicit refusal change | BROKEN change | Any-refusal change | Pairs |
|---|---|---|---|---|
| 0.5 | +11.0 [+4.2, +18.6] | +0.0 [−5.1, +4.2] | +2.5 | 118 |
| 0.75 | +12.7 [+5.1, +20.3] | +5.9 [+0.8, +11.9] | −1.7 | 118 |
| 1.0 | +26.4 [+19.0, +34.7] | +0.0 [−4.1, +5.0] | +6.6 | 121 |
| 1.25 | +28.6 [+20.5, +37.5] | +0.9 [−3.6, +5.4] | +9.8 | 112 |

Points against α = 0 (explicit 20.2%, BROKEN 10.9%), paired, with bootstrap 95% intervals. The gate opened on 71% of the requests.

**Push size:** the relative push is the same in all three languages: median 0.270 Bengali, 0.272 Hindi, 0.271 English (ratio Bengali/Hindi
0.99). This rejects the explanation that Bengali activations are smaller, so the same push hits them harder.

## What this means for Phase 4

- The explicit-refusal gain grows with strength and is large on new Bengali requests (+28.6 at 1.25), consistent with Phase 4's +23.6.
- The BROKEN rise does not appear on these requests, and the push is not relatively larger in Bengali. The Phase 4 rise (+8.1, interval
  +2.0 to +14.9) may be specific to those 148 requests or partly noise; the two intervals overlap between about +2 and +5. **The Phase 4
  verdict stands as registered (FAIL);** this result only weakens its reading as a general property of the push in Bengali.
- The single non-zero BROKEN value is at α = 0.75 (+5.9), between two strengths with none. That pattern does not fit a smooth dose
  response and is most likely noise at this sample size.

## Data loss and limits

- **Grading loss (addendum):** Sonnet 5.5's safety filter stopped two of twelve grading parts twice (flagged as biology). 74 replies were
  never graded and 36 copies of their labels were lost, so 29 to 38 of 150 pairs per strength are dropped. The filter acts on content, so
  the dropped requests are not random; if broken replies cluster in that content, this result would miss them.
- One model, model-only Bengali labels (grader not calibrated on Bengali), 150 requests used once before with another model.
- No strength chosen here is confirmed; all Bengali confirmation data is used.
- Run: A40, 23:05 to 00:05 UTC, about $0.60, pinned environment.
