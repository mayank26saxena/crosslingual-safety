# Reproducing the v2 results

Everything below runs from the committed scripts. Files that hold harmful text (prompts, replies, grading sheets) are not in git; their exact
checksums are in `MANIFEST.sha256`, and every human and model label is in `results/v2/labels/` keyed by item id.

## Environment

| Component | Version |
|---|---|
| Model | `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218` (pinned in every GPU script) |
| GPU stack | Python 3.12.3, torch 2.8.0+cu128 (CUDA 12.8), transformers 5.22.2, accelerate 1.15.0, tokenizers 0.23.3 (`results/v2/comprehension/env.txt`) |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, one A40 or RTX A6000 |
| Local | Python 3, standard library only, except `numpy` for the direction analyses |

Generation settings everywhere: system prompt "You are a helpful assistant.", Qwen chat template, `enable_thinking=False`, greedy decoding, one
reply per request; 1,024 new tokens for Hindi, 512 for English and Spanish.

## Determinism

- **Greedy decoding in bfloat16 is not bit-identical across batch sizes or hardware.** Re-generating with a different batch size changed 20 of 22
  replies in an early check (`docs/prereg/2026-11-10_v2_measurement_and_data.md`, withdrawn truncation check). Replies are therefore archived and
  checksummed rather than expected to regenerate exactly; any comparison of conditions (for example steering) generates all conditions in one run
  with identical batches.
- **Model graders are not deterministic.** Their labels are archived (`results/v2/labels/`) and every analysis reads the archived labels. Grader
  agreement with the human is reported for each route.
- All sampling uses fixed seeds recorded in each script.

## Data

Sources are downloaded to `data/external/` (see `data/README.md` for sources, licences and versions). Then:

```bash
python3 scripts/v2/collect.py          # candidates with provenance
python3 scripts/v2/exclude.py          # sealed split, v1 items, duplicates
```

## Results and the commands that produce them

| Result (docs/results/) | Command(s) |
|---|---|
| Phase 0 grader gate, bridge check (`2026-11-13_phase0_grader_gate.md`) | `python3 scripts/v2/grader_gate.py notes/hindi_calibration/human/hindi_calibration_1024_human.csv results/v2/grader_gate/gate_results.json` |
| Raw Hindi baseline (`2026-11-13_hindi_baseline_raw.md`) | `python3 scripts/v2/baseline_gap.py <run> <key> results/v2/hindi_baseline/baseline_raw.json` |
| Phase 2a refusal direction (`2026-11-13_refusal_direction.md`) | `python3 scripts/v2/refusal_direction.py prep`, then on a GPU `python refusal_direction.py run ...` |
| Phase 2b explicit vs soft (`2026-11-13_explicit_soft_direction.md`) | `explicit_soft.py prep` / `run`; `build_steer_grading.py`; `steer_analysis.py` |
| Screening audit (`2026-11-13_screening_audit.md`) | `build_audit_sheet.py`; `screen_agreement.py <out>` and `--adjudicated` |
| Screened baseline and split (`2026-11-13_hindi_baseline_final.md`) | `build_screen_folder.py`; `final_baseline.py results/v2/final_baseline` |
| Comprehension screen | `comprehension.py prep` / `run` / `judge-folder` |
| Provisional paired results (2026-11-11, 2026-11-12) | `paired_provisional.py calibration|spanish <out>` (from the label tables) |
| Replication check of baseline replies (`2026-11-13_replication_check.md`) | `regen_calibration.py generate` under the pinned environment, then `replication_check.py` |
| Phase 3 gated steering (`docs/prereg/2026-11-13_v2_phase3_gated_steering.md`) | `mmlu_sample.py` (with `pyarrow==18.1.0`); `gated_steering.py prep` / `tune` / `test` |
| Phase 3 tuning and selection (`2026-11-14_phase3_tuning.md`) | `gated_steering.py tune`; `build_phase3_grading.py`; `phase3_tune_analysis.py results/v2/phase3/tune/tune_selection.json` |
| Phase 3 test and verdict (`2026-11-14_phase3_test.md`) | `gated_steering.py test ... 1.25 5.515399932861328`; `build_phase3_grading.py`; `phase3_test_analysis.py results/v2/phase3/test/test_results.json` |
| Label tables, manifest | `export_labels.py`; `manifest.py` |

Grading and screening runs are Claude Code subagents (or Kiro sessions for the earliest runs), one per 50-row blind part, with the prompts in
`docs/protocols/`. Each run records the model the subagent reported.

## Known gaps (to close before submission)

- The tier A, PolyGuard and Spanish replies (2026-11-11 and 12) were generated before the model revision and library versions were pinned; the
  generation script and settings are recorded. Regenerated under the pinned environment (`docs/results/2026-11-13_replication_check.md`),
  5.6% of Hindi and 28.2% of English replies are identical text and most share their first 64 tokens; label-level agreement of the
  regenerated replies has not been graded. Reported numbers come from the archived replies and labels.
- The untracked files (prompts, replies, grading folders, keys) are packed in `v2_untracked_<date>.tar.gz` with its sha256, kept outside the
  repository (`~/Desktop/projects/crosslingual-safety-archive/` on the author's machine) and to be deposited privately for reviewers. Public source
  datasets in `data/external/` are not packed; their checksums are in the manifest.
