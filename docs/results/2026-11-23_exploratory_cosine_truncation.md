# Exploratory: direction cosines and reply-length cap (2026-11-23)

Not registered; added after the registered studies. Script: `scripts/v2/diag_cosine_truncation.py`; output (aggregates only): `results/v2/diagnostics/cosine_truncation.json`.

## Cosine between the explicit-versus-soft direction and the refusal direction

| Model | e_dir layer | refusal layer used by the gate | cos(e_dir, r_dir at gate layer) | cos(e_dir, r_dir at the same layer) | random unit vector, 95th pct of abs cos |
|---|---|---|---|---|---|
| Qwen3-8B | 24 | 14 | -0.029 | 0.371 | 0.030 |
| Llama-3.1-8B | 18 | 10 | 0.013 | 0.640 | 0.031 |

Directions at different layers are not directly comparable, so the same-layer value is the informative one: the explicit-versus-soft direction overlaps substantially with the refusal direction (cosine squared about 0.14 in Qwen, 0.41 in Llama). Cosine is not a test of independence under intervention.

## Share of unsteered Hindi harmful replies that reached the 1,024-token limit (404 held-out requests)

| | all | labelled soft | soft among lost refusals | labelled comply | labelled explicit |
|---|---|---|---|---|---|
| Qwen (Phase 3 test run) | 168 of 404 (41.6%) | 85 of 232 | 38 of 115 | 53 of 66 | 10 of 75 |
| Llama (Phase 7 regenerated run) | 19 of 404 (4.7%) | 5 of 80 | 4 of 74 | 12 of 123 | 1 of 188 |

The registered truncation check was withdrawn (prereg 2026-11-10 addendum); this is a count of replies that reached the cap, not a test of whether they would have changed label. Qwen's soft and compliant labels rest on truncated replies far more often than Llama's. Llama's lost-refusal counts here (173) differ by one from the paper's 172 because they come from the regenerated run.
