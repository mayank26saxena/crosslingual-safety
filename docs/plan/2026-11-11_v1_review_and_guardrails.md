# Review of v1 and guardrails for v2

**Date:** 2026-11-11. v2 restarts the work from the raw sources. This file lists what went wrong in v1 (Steps 1 to 8) and the check that prevents each
problem in v2. Every v2 step is checked against the table before it runs, and its report says which guardrails applied.

## What went wrong, and the guardrail for each

| # | v1 problem | Consequence | v2 guardrail |
|---|---|---|---|
| 1 | Experiments ran before refusal grading was validated in Hindi or Bengali (judge kappa 0.14 in Step 1, 0.55 for Hindi later) | Effect sizes were uncertain throughout | **Measurement first.** No intervention result is reported as confirmed until the judge passes kappa ≥ 0.7 against human Hindi grades |
| 2 | Dataset fields were assumed, not verified: "Hindi-source" for IndicGuard culture-adaptive items, and benchmark "harmful" read as "should be refused" | About 40% of IndicGuard "harmful" prompts are not requests; the "native" subset is LLM-adapted | **Every field cites its source.** Provenance and label meanings are taken from the dataset's own documentation, quoted in the dataset card, or marked unverified |
| 3 | Data was used without anyone reading samples | Non-requests and translation faults went unnoticed for several steps | **Read before use.** A 30-item read of each source and each screen's output, by a fluent reader or a model reference, before it feeds anything else |
| 4 | Reply limits too short for the language (Llama Bengali stopped at about 82 characters; Hindi at 430 to 520 characters) | Harm could not be judged; refusals late in a reply were missed | **Length set by characters, not habit.** 1,024 tokens for Indic languages, and the share hitting the limit is reported |
| 5 | Interventions were designed before the failure was localised; layers and mechanisms were guessed | Steps 7 and 8 tested a mechanism that was not shown to be the problem | **Diagnose before repairing.** No repair is designed until the diagnosis phase reports |
| 6 | Too few failure cases to fit on (60 to 64 per language in Step 8 development) | Detectors and selections were noisy | **Power before running.** Each pre-registration states the expected count of failures and the smallest detectable effect, and does not run below its minimum |
| 7 | The primary model changed several times (Qwen2.5 → Llama → Qwen3), and the one pass was never replicated | Results do not add up across steps | **One model until a result replicates.** Qwen3-8B is fixed for v2; a second model is used only to replicate a pass |
| 8 | Pipelines were long and serial, with full-scale runs first | Errors surfaced hours in; GPU time was spent on runs that failed late | **Pilot first.** Every stage runs end to end on 50 items before the full run, and the pilot output is read |
| 9 | Pod operations: hours of idle GPU, a stop sent in parallel with the copy (Step 8 results stranded), credits ran out mid-plan | Lost results and wasted spend | **Serial shutdown:** copy, verify checksums, then stop, never in parallel. A spend estimate before each run, and a balance check before starting |
| 10 | Pre-registrations were followed by many addenda | Rules drifted as results came in | **Write once, small.** Each phase's pre-registration is short, covers only that phase, and is written after the previous phase reports |
| 11 | Machine-translated data dominated (76% of Hindi rows) | A gap could be a translation artefact | **Tiered reporting.** Results are reported per provenance tier, with the human-authored tier as the primary test |

## Working rules

- One change at a time between runs, and each run is pre-registered or labelled exploratory.
- No result is described as positive unless it passes its pre-registered rule.
- Harmful text never appears in chat, commits or tracked files. Items are referred to by id.
- Before any download or paid API run: name, source, size or cost, and approval.
