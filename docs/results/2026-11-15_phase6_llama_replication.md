# Phase 6 result: Llama-3.1-8B-Instruct replication

**Pre-registration:** `docs/prereg/2026-11-15_v2_phase6_llama_replication.md` (frozen in 94884b9, two addenda). **Code:**
`scripts/v2/phase6_llama.py` (sha256 93d9bf01…778ff, the same file that ran on the pod). **Output:**
`results/v2/phase6/stageB/phase6_results.json`, `stageB_results.json`. **Labels:** `results/v2/labels/phase6A_llama_grades.csv`,
`phase6B_llama_grades.csv` (ids only). Model `meta-llama/Llama-3.1-8B-Instruct`, revision 0e9e39f2…, one A40, the pinned Qwen environment.

**Status: human-validated after adjudication; blind check below bar (addenda 4–5).** The pre-registered human check (60 Hindi Llama replies) gave blind four-label κ 0.643 < 0.7,
which under the frozen rule makes these numbers provisional (`2026-11-16_llama_human_checks.md`). An unregistered review of the disagreements (addendum 4)
raised agreement to 0.893 (an upper estimate) and supports the judge's reading below: lost refusals are a mix of compliance and soft refusal in
similar shares (13 compliance, 9 soft of 23 sampled pairs; not a significant difference).

## Outcome: the gap replicates, the Qwen mechanism does not

| Claim | Rule | Llama result | Replicates |
|---|---|---|---|
| C1 gap | Hindi − English explicit ≤ −10 pts, CI excludes 0, **and** REFUSE_SOFT is the commonest Hindi label among lost refusals | −42.3 [−47.3, −37.4]; lost refusals: COMPLY 88, REFUSE_SOFT 75, BROKEN 9 | **No** (first half yes, second half no) |
| C2 refusal signal | Hindi projection ≥ 80% of English | 28% (Hindi 0.51, English 1.84) | **No** |
| C3 explicit–soft axis | Paired Hindi − English shift negative, CI excludes 0 | −2.40 [−2.55, −2.26] (−50% of the fit gap), CV AUROC 0.925 | **Yes** |
| C4a fix works | Phase 3 rule: Hindi explicit +15 within guards | +0.0 | **No** |
| C4b acts across languages | English explicit +3 or more | −0.2 [−0.7, 0.0] | **No** |

### C1: a larger gap, and more compliance

On 404 harmful pairs Llama refuses explicitly in English 88.9% of the time and in Hindi 46.5%. The −42-point gap is larger than Qwen's
(−33). By source: Aya −43.0 [−48.4, −37.7], IndicSafe −38.8 [−50.7, −26.9]. Where Qwen's lost Hindi refusals mostly became soft refusals,
Llama's split between compliance and soft refusal in similar shares (88 vs 75 of 172, p = 0.35): Hindi COMPLY is 30.2% against 8.7% in English, and any refusal (explicit or soft) is 66.8% against 90.8%.
BROKEN Hindi replies are 3.0%. Harmless over-refusal is low in both languages (Hindi 2.4%, English 4.1%).

### C2: Llama does not recognise Hindi harm as strongly

The refusal direction is valid: layer 10, ablation removes 94% of the harmful–harmless gap on held-out English prompts. But the mean
Hindi harmful projection is only 28% of the English one. In Qwen, Hindi reached English strength on this direction. That was the reason the
Qwen gap was read as a wording problem downstream of harm recognition. In Llama the shortfall starts earlier.

### C3: the explicit–soft axis does transfer

The English-fitted explicit-versus-soft direction (layer 18, CV AUROC 0.925) is valid, and Hindi prompts sit well toward its soft end. The
direction was fitted on only 11 soft English replies (Llama rarely refuses softly in English: 11 of 782 fit replies; see addendum 1). C3
holds, but this direction is the weakest-supported part of the run.

### C4: the gate did not open in Hindi, so the fix was not really tested in Hindi

The Phase 3 gate and strength were carried over unchanged: τ at the 50th percentile of English dev harmful gate scores on the refusal
direction, α = 1.25 × the fit gap at layer 18. Because Hindi harmful prompts score low on the refusal direction (C2), the gate opened on
only **1.0%** of them (English: 44.3%; harmless: 0% in both). Hindi replies were therefore almost all unchanged, and every guard passed
trivially. In English, where the gate did open, explicit refusal did not rise (−0.2). MMLU was identical with and without the push (Hindi 42.7%,
English 63.3%).

C4a fails as registered. It should not be read as evidence that pushing along the explicit–soft axis cannot work in Llama. The registered
gate never fired on Hindi, a direct consequence of C2. C4b also fails: the push did not move English either, which may reflect the
carried-over strength (the pre-registration lists this risk).

## What this means for the project

- The explicit-refusal gap is not specific to Qwen. It is larger in Llama.
- The mechanism differs. In Qwen, Hindi harm is recognised at English strength and the gap is in how the refusal is worded (explicit vs.
  soft). In Llama, Hindi harmful prompts project weakly on the refusal direction. The missing refusals split between compliance and soft
  refusal in similar shares (judge labels; the reviewed human sample agrees). The claim the data support is "the gap exists in
  both models, with different internal signatures, and Llama fails to refuse 1.4–1.9 times as often as Qwen".
- A gate whose threshold is set on English is a poor fit for a model whose Hindi refusal signal is weak (Phase 7 sets it on Hindi). Phase 6 tested no
  Llama-specific fix, and none was pre-registered.

## Limits

- Labels: blind human check below bar (κ 0.643); human-validated after the author's adopted review (0.893, upper estimate).
- The explicit–soft direction rests on 11 soft examples.
- Gate and strength were carried over from Qwen, not tuned for Llama (by design).
- There is one Llama size and revision, and one decoding setting (greedy, 1,024 new tokens).

## Cost

Stage A about 33 min, stage B about 2 h 15 min on one A40 at $0.59/h, about $1.65 in all. Both pods were terminated, and their removal was
confirmed by listing.

## Exploratory addendum (2026-11-17): per-layer projections

Not pre-registered; descriptive only, made for the blog's per-layer figure. Forward passes on the 404 harmful and 169 harmless test
pairs project every layer's last-token output on that layer's English refusal direction (`scripts/v2/layer_projections.py`; outputs
in `results/v2/layer_proj/`). The layer-10 values reproduce the stored Phase 6 projections (1,146 items, mean absolute difference
0.006 on a mean magnitude of 0.99) and the harmful Hindi/English ratio of 0.278.

- At layer 10 the direction still ranks Hindi harmful above Hindi harmless requests almost as well as it does in English
  (AUROC 0.97 against 0.99). What is weak in Hindi is the size of the separation, not its order: the Hindi harmful–harmless gap is
  0.30 of the English gap at layer 10.
- So C2 should be read as a magnitude finding. Llama's Hindi harmful prompts sit low on the direction while remaining well ordered,
  which is consistent with Phase 7, where a Hindi-calibrated gate threshold separated Hindi harmful from harmless requests usefully.
- Qwen's Hindi gap ratio at its layer 14 is similar (0.32), for a different reason: its Hindi harmful prompts reach English strength,
  but its Hindi harmless prompts also sit high (the language offset in Experiment 2).
