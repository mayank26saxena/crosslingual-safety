# Data

| Path | Source | Tracked |
|---|---|---|
| `external/multijail/MultiJail.csv` | MultiJail (Deng et al., ICLR 2024), [DAMO-NLP-SG/MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail): 315 harmful requests with professional translations | No: fetch from upstream |
| `external/xstest/xstest_prompts.csv` | XSTest (Röttger et al., [arXiv 2308.01263](https://arxiv.org/abs/2308.01263)): 250 safe prompts, 200 unsafe contrast prompts | Yes; confirm upstream license before a public release |
| `rendered/multijail_*.jsonl` | `scripts/render_multijail.py` | No: regenerable |
| `rendered/xstest_safe_{en,bn}.jsonl` | `scripts/translate_xstest.py`, translated by Qwen2.5-14B-Instruct | Yes; superseded by the rebuild in Step 1 |
| `splits/splits.json` | `scripts/make_splits.py` (ids only) | Yes |
| `external/global_mmlu/{hi,en,bn}/test-00000-of-00001.parquet` | Global-MMLU (Singh et al., 2024), [CohereLabs/Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU), revision `0e619dbeb34206cd48705a1a0ea7fb21cae09993`, Apache-2.0; Phase 3 general-ability guard (hi, en) and Phase 4 (bn, downloaded 2026-11-14, same 300 ids) | No: fetch from upstream at the pinned revision |

The first-generation XSTest-Bengali file has known translation faults, listed by id in `results/baseline_summary.json`
under `xstest_bn_audit`. It is kept for comparison with the earlier baseline and is not used as the over-refusal guard.
