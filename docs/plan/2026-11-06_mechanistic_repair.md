# Plan: mechanistically localized cross-lingual safety repair

**Status:** adopted 2026-11-06. This plan sets direction and is not a pre-registration. Each experiment gets a dated pre-registration before it runs.

**Question:** can we identify the internal mechanism behind language-specific safety failures in Hindi and Bengali, and repair it through a sparse, causally validated intervention without full-model retraining?

**Competing explanations:** semantic representation failure, safety classification failure, and refusal-routing failure. Refusal routing is the working hypothesis, not an assumption.

## Execution order

1. Pin the model. 2. Audit datasets. 3. Build the manifest. 4. Deduplicate and group by source prompt. 5. Create splits. 6. Run baselines. 7. Confirm the gap. 8. Cache activations. 9. Analyse representations. 10. Discover SAE features. 11. Discover candidate circuits. 12. Run causal interventions. 13. Reassess. 14. Test gated steering. 15. Try a localized edit. 16. Run held-out evaluation. 17. Replicate. 18. Package artifacts.

Weights are not edited unless activation-level interventions first show a reproducible causal effect on development examples and benign controls.

## Step 1: model pins

| Model | Role | Revision | Size | Licence | Access |
|---|---|---|---:|---|---|
| Qwen/Qwen3-8B | Primary | `b968826d9c46dd6066d109eabc6255188de91218` | 16.4 GB | Apache 2.0 | Open |
| google/gemma-3-4b-it | Replication | `093f9f388b31de276ce2de164bdc2081324b9767` | 8.6 GB | Gemma | Manual gate |
| meta-llama/Llama-3.1-8B-Instruct | Optional | `0e9e39f249a16976918f6564b8830bc894c89659` | Used earlier | Llama 3.1 | Manual gate |

Qwen3-8B runs with reasoning off, as stated in `scripts/common.py`. The tokenizer and inference settings are recorded at the first run.

## Step 2: dataset audit, metadata only

No dataset content has been read. Sizes, licences and language tags come from the Hugging Face API on 2026-11-06 and are not yet verified against the data. Provenance and native-versus-translated status are still open.

| Dataset | Repository | Size | Licence | Hindi / Bengali tags | Access | Finding |
|---|---|---:|---|---|---|---|
| Multilingual Harmless-Harmful | `Heyjab/...`, `heretic-org/...` | 5 MB | CC BY-SA 4.0 | Yes / Yes | Open | Two copies, 416 harmful and 416 harmless by split. Original source still to be established |
| IndicGuard | `l3cube-pune/IndicGuard` | 1.61 GB | CC BY 4.0 | Yes / Yes | Open | Tagged 100K to 1M rows. Linked papers listed |
| SurakshaEval | `SurakshaAI/SurakshaEval` | 3.7 MB | CC BY-NC-SA 4.0 | Yes / Yes | Automatic gate | Another repository of this name holds almost no data |
| IndicSafe | `ppattnay/IndicSafe` | 1.6 MB | CC BY-NC 4.0 | Yes / Yes | Open | Linked paper listed |
| BanglaShield | `md-nishat-008/BanglaShield-Dataset` | 7 MB | CC BY-NC 4.0 | Yes / Yes | Automatic gate | Tagged 10K to 100K rows |
| Nemotron Safety Guard v3 | `nvidia/Nemotron-Safety-Guard-Dataset-v3` | 663 MB | CC BY 4.0 | Yes / No | Open | No Bengali tag, so Hindi only unless the data shows otherwise |
| LingualSafety 1.0 | Not found | | | | | Not found by name on Hugging Face |
| AILuminate | Third-party mirrors only | | Mixed or none | Not tagged | Mixed | Official MLCommons source still to be established |

Non-commercial licences limit redistribution, so the published manifest should hold identifiers and provenance, not third-party prompt text.

## Issues carried over from earlier work

- **Outcome reliability:** Gate E1 failed on unsafe grading (Bengali kappa 0.14). Explicit refusal has been more reliable, on Llama replies. Step 7 cannot be called confirmed until a harm measure passes an agreement gate.
- **Human review:** the plan calls for human and native-speaker review, and no reader is currently available. Claims that depend on it stay limited until that changes.
- **Reply length:** Llama reached only about 82 Bengali characters at 100 tokens, so token limits are set per script.
- **Over-refusal target:** the proposed limit is about 1 point. Step 5 lost about 8 points of harmless answering, so that intervention would fail this target.
- **Earlier findings to reuse as prior evidence:** intent separated from surface features on three models, the English refusal direction lowered refusal on Qwen2.5 but failed its control on Gemma-2 and Llama, and unfiltered cue neurons caused blanket refusal.
- **The sealed MultiJail split** stays unread and has no Hindi. Whether it remains part of final evaluation is an open decision.

The Qwen and Gemma replication of Step 5 was stopped on 2026-11-06 before any neuron result. Partial Stage 0 replies are kept outside version control and are not reported as results.

## Steps 2-5: dataset status, 2026-11-06

Downloaded at pinned revisions: Multilingual Harmless-Harmful `67520a76`, IndicGuard `f6025076`, IndicSafe `9a12fe45`, SurakshaEval `e47de8b9`, BanglaShield `4f3d642e`.

| Manifest | Value |
|---|---|
| Unique rows | 141,009 |
| Source-prompt groups | 45,207 |
| Exact duplicates removed | 10,925 |
| Near-duplicate links on English source (TF-IDF cosine at least 0.92) | 799 |
| Rows dropped for no prompt label | 2,970 |
| Largest group, distinct English prompts | 22 |
| Groups crossing splits | 0 |

Splits are by group: discovery 50%, development 20%, held-out 30%. All IndicSafe, SurakshaEval, BanglaShield and IndicGuard test groups go to held-out. Harmful and harmless twins in Multilingual Harmless-Harmful stay together. Counts are in `data/manifest/summary.json`. The prompt manifest is not tracked.

**Sealed split:** BanglaShield includes every MultiJail request, so its 945 MultiJail rows across three written forms are excluded. Groups with an exact text match to the sealed split are also removed, which drops 57 IndicGuard rows. The manifest then has 0 exact matches. Paraphrases cannot be excluded this way. It still has 51 rows that exactly match the unsealed MultiJail requests, which are recorded and not removed.

**Provenance:**

| Dataset | Authorship | Note |
|---|---|---|
| SurakshaEval | Human-written, per its documentation | 149 Bengali, 116 Hindi and 265 English prompts. No expected behaviour is included, so these need grading criteria |
| BanglaShield | Compiled from public sources | Native, romanized and code-mixed forms. Benign rows are largely converted product reviews |
| IndicGuard | Google Translate API | Culture-adaptive subset starts from Hindi |
| Multilingual Harmless-Harmful | Machine translated, partly checked | English source |
| IndicSafe | Translated | English source |

Only SurakshaEval is documented as human-written, so it is the small native-language test, reported separately. LinguaSafe, likely the proposal's LingualSafety, is available only through BanglaShield here, and only in Bengali. Duplicate thresholds and these provenance statements have not been reviewed by a person.

## Grading plan

- **Hindi:** the project lead grades a blinded sample directly. This is the human reference, for Hindi only.
- **Bengali:** replies are translated into English with the Google Cloud Translation API and then graded, following the MultiJail method. No native review is available, and results are labelled that way.
- **Calibration:** the same Hindi replies are also translated and graded the same way. Agreement with the direct Hindi grades estimates what translation loses. It is an estimate for Hindi and is applied to Bengali only as an assumption.
- **Agreement gate:** kappa at least 0.6 between translated and direct Hindi grades, before translated Bengali harm grades are used in any claim.
