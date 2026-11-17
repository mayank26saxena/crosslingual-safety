# Same refusal gap, different mechanisms: diagnosing and repairing Hindi under-refusal in two open chat models

*(Title note: the alternative "Soft Refusal or Silent Compliance?" is not used. Llama's lost refusals are a mix of compliance and soft
refusal, so the clean dichotomy overstates the evidence; see §4.)*

*Working draft, 2026-11-15. Format-neutral Markdown; to be converted to the venue template. Sections marked **[Phase 7]** wait for the
Llama fix study. Sections marked **[Related work]** wait for the literature review. Every number comes from a committed result document
(`docs/results/`) and can be recomputed from the id-only labels in `results/v2/labels/` (see `REPRODUCE.md`).*

---

## Abstract (draft)

Safety-tuned chat models refuse harmful requests less reliably outside English. We study this gap on paired requests, where the same
request is asked in English and in Hindi. Replies are graded with a four-way scheme that separates explicit refusals from soft ones (warning,
lecturing or redirecting without saying no). In Qwen3-8B, explicit refusal of harmful requests falls from 51% in English to 19% in Hindi on
404 held-out pairs, and most of the missing refusals become soft refusals. An English-fitted refusal direction responds to harmful Hindi
requests about as strongly as to English ones, while a separate direction that predicts explicit versus soft refusal places Hindi prompts
toward the soft end. Pushing along that second direction, gated on the refusal-direction score, raises Hindi explicit refusal by 17 points
on held-out pairs, but it raises English by 16, so the gap barely narrows. It also failed a pre-registered Bengali confirmation on a guard
against garbled output. A search for Hindi-specific neurons finds a small set that carries most of the internal English–Hindi difference,
but moving them changes behaviour in English, not Hindi. In Llama-3.1-8B-Instruct the gap on the same pairs is larger (89% to 47%). Its missing refusals split between compliance and soft
refusal, and it fails to refuse Hindi harmful requests 1.4–1.9 times as often as Qwen (depending on whether broken replies count). Its internal signature also differs:
Hindi harmful prompts reach only 28% of the English projection on the refusal direction. A repair matched to that diagnosis pushes along the refusal direction, with a gate threshold set on Hindi data. On held-out
pairs it raises Hindi explicit refusal by 15 points and narrows the gap by 7 points. It narrowly misses its
pre-registered bars for explicit refusal (+15) and gap narrowing (10 points). All studies were pre-registered, tuned on development items and
tested once on held-out items, and the failures are reported alongside the successes.

---

## 1 Introduction (draft)

Refusal is the main behavioural safeguard of open chat models, and it is trained mostly on English data. Prior work shows that harmful
requests in lower-resource languages are refused less often **[Related work: cite the multilingual jailbreak and safety-gap papers]**. Most
of that work measures the gap with benchmark-level refusal rates. It rarely asks what replaces the missing refusal, or why the model's
behaviour changes when only the language changes.

We study the gap in a setting designed to make those questions answerable:

- **Paired requests.** Every harmful request is asked in English and in Hindi, the same items in both languages, and each comparison is
  paired by item. Items are screened for being a real request, for translation fidelity, and for whether the model understands the Hindi
  version.
- **A four-way outcome.** Replies are graded COMPLY, REFUSE_EXPLICIT, REFUSE_SOFT or BROKEN by an LLM judge chosen against a fluent human
  Hindi grader. The split between explicit and soft refusal turns out to carry most of the gap in both models (in Llama on a 20-pair human sample).
- **Two models, the same items.** Qwen3-8B and Llama-3.1-8B-Instruct are evaluated on the identical 404 held-out harmful pairs.
- **Diagnosis before repair, with pre-registration.** Every study had its rules frozen before any of its replies were generated. Settings
  were tuned on development items and tested once on held-out items; deviations were recorded in dated addenda.

**Contributions.** Prior work has established that the gap exists, including in Hindi and in both models we study, and that English refusal
directions transfer across languages (§2). Our contributions are on the response side, in the contrast between models, and in the
protocol.

1. **A response-side decomposition of the gap.** We use paired, screened, same-request measurement with a four-way label on default
   (non-adversarial) requests. In Qwen3-8B most of the missing Hindi explicit refusals become *soft refusals*. In Llama-3.1-8B-Instruct
   they split between compliance and soft refusal in similar shares, and Llama fails to refuse Hindi harmful requests 1.4–1.9 times as
   often as Qwen on the same pairs. Prior multilingual work scores refusal as binary or as a harm scale, so it cannot see this.
   - *A measurement caution.* The COMPLY/SOFT boundary is fragile. The hard case is a reply that warns and then answers. Our Llama human
     check fell below its registered bar on this boundary (blind four-label κ 0.643). After the grader re-applied the rubric to the
     disagreements, agreement was 0.893. We use the reviewed labels as the human reference and report both values.
2. **An explicit-versus-soft refusal direction.** To our knowledge, this is the first activation direction that predicts refusal *style*
   rather than refusal itself. Hindi prompts sit toward its soft end, and adding it turns soft Hindi refusals into explicit ones.
3. **Two internal signatures.** On the same items, Qwen's Hindi harmful prompts project on the English refusal direction at English
   strength, while Llama's reach 28% of it. Behaviour differs in degree: Qwen mostly hedges, while Llama hedges or complies. The weak Llama projection replicates
   prior findings (Wang et al. 2025; Aziz et al. 2026) in a paired Hindi design. The contrast between the two models on identical items is, to our
   knowledge, new.
4. **Pre-registered negative results on repair.**
   - Gated steering with a refusal-direction gate raises Hindi explicit refusal on held-out pairs, but it is not language-selective: English
     rises as much, so the gap barely narrows.
   - The repair fails a sealed confirmation in a held-out language.
   - The MLP neurons most tied to the English–Hindi difference change behaviour in English, not Hindi.
5. **A pre-registered protocol for cross-lingual interpretability.** Rules and analyses are frozen before data, settings are tuned on
   development items, and each repair is tested once on held-out items and confirmed on a sealed held-out language, with deviations in dated
   addenda. To our knowledge, no prior steering or interpretability paper combines these.
6. **A pre-registered held-out test of a diagnosis-matched repair for Llama.** We push along the refusal direction, with a gate
   threshold set on Hindi data. Hindi explicit refusal rises by 14.6 points (judge labels; explicit refusal is the best-validated label). The gap narrows by 7.4 points,
   against 1.2 for the Qwen-style repair. It narrowly misses its registered bars on explicit refusal and gap narrowing. This independently
   corroborates the target-language calibration that Aziz et al. (2026) found Llama needs.

---

## 2 Related work

*Sources and verification status are in `docs/paper/reports/Crosslingual refusal related work.md`. Entries marked [verify] still need their
venue, author list or a key number checked against the primary source before submission.*

**Multilingual safety gaps.**

- *Founding papers.* Three papers in late 2023 first quantified the gap: Yong et al. (2023, arXiv:2310.02446), with translation attacks on
  GPT-4; MultiJail (Deng et al., ICLR 2024, arXiv:2310.06474); and XSafety (Wang et al., Findings of ACL 2024, arXiv:2310.00905), the first
  benchmark covering both Hindi and Bengali. Shen et al. (2024, arXiv:2401.13136) [verify] found that low-resource prompts yield more harmful
  *and* more irrelevant replies.
- *What they measured.* All of these report unsafe-output or attack-success rates, mostly on closed models and translated English prompts.
  None uses explicit refusal as the outcome.
- *Indic work.* IndicSafe (Pattnayak & Chowdhuri 2026, arXiv:2603.17915), SurakshaEval (Banerjee et al., AIES 2026, arXiv:2608.07862) and
  IndicGuard (Bramhecha et al. 2026, arXiv:2606.22841) build Indic safety benchmarks and guards.
- *Closest measurement study.* IndicSafeEval (Mondal et al., Findings of EMNLP 2026, arXiv:2609.03781) evaluates Qwen3-8B and Llama-3.1-8B,
  among others, in Hindi, Bengali, Marathi and Punjabi. Its attack set is adversarial persuasion prompts, machine-translated from English,
  scored on a five-point harm scale by Gemini-2.5-Flash (Hindi κ 0.84 binary). It reports per-language attack-success rates.
  - Our study differs: default, non-adversarial requests written natively in Hindi (Aya Red-teaming); per-item paired comparison; explicit
    versus soft refusal as the outcome; and a mechanistic analysis.
- *Closest methodological cousin.* Huang et al. (Findings of EMNLP 2026) decompose attack success by response language and show that
  aggregate rates hide in-language failures. We decompose by refusal style instead.
- *Data sources.* Our Hindi items come from Aya Red-teaming (Aakanksha et al. 2024, arXiv:2406.18682) [verify venue]. We use its Devanagari
  prompts, written by native annotators, with their English translations, so each pair runs from native Hindi to translated English.
  Harmless controls come from PolyGuard (Kumar et al., COLM 2025, arXiv:2504.04377) and IndicSafe. Capability is measured with Global-MMLU
  (Singh et al., ACL 2025, arXiv:2412.03304).
- *Supported languages.* Hindi is an officially supported language of Llama-3.1 (Llama Team 2024, arXiv:2407.21783) and of Qwen3 (Qwen Team
  2025, arXiv:2505.09388) [verify both], so the gap is not confined to unsupported languages.

**Refusal directions and their cross-lingual behaviour.**

- *Method.* Arditi et al. (NeurIPS 2024, arXiv:2406.11717) showed that refusal in open chat models is mediated by a difference-of-means
  direction: ablating it removes refusal, adding it induces refusal. We adopt their recipe with two differences. We read a single position,
  the last token of the chat template after the user turn. We validate a direction by the share of the English harmful–harmless gap in a
  first-token refusal score that its ablation removes.
- *Refinements.* Wollschläger et al. (ICML 2025, arXiv:2502.17420) show refusal can occupy multi-dimensional cones. Joad et al. (2026,
  arXiv:2602.02132) [verify venue] find distinct refusal directions for different refusal *reasons*. Wang et al. (2024, arXiv:2410.03415)
  [verify venue] isolate a false-refusal direction. Our explicit-versus-soft direction differs from all of these: it separates the *style* of
  refusal given that the model refuses.
- *Harm versus refusal.* Zhao et al. (2025, arXiv:2507.11878) [verify venue] show that harmfulness and refusal are encoded separately, at the
  last instruction token and the last template token respectively. Our directions are read at the last template token. We therefore
  describe them as refusal directions and the Qwen result as a projection at English strength, not as recognition of harm.
- *Cross-lingual.* Wang et al. (NeurIPS 2025, arXiv:2505.17306) find refusal directions approximately parallel across 14 languages (no
  Indic ones), with weaker harmful–harmless separation outside English, especially in Llama-3.1-8B-Instruct. Aziz et al. (2026,
  arXiv:2606.01196) extend this to 23 languages, including Hindi and Llama-3.1-8B-Instruct, with a harmfulness direction pooled over ten
  high-resource languages. They report that low-resource harm projects lower but stays separable, and call these failures of action rather
  than representation. Further results:
  - Ivanova (BlackboxNLP 2026, arXiv:2610.28897): transfer is graded rather than binary.
  - Oppong et al. (2026, arXiv:2608.11146): under 10% of the English refusal signal survives in African languages.
  - Ifebi et al. (2026, arXiv:2607.10112): harm routed through a misaligned subspace leaves refusal "intact but untriggered".
- *Where we fit.* Our Llama finding replicates this pattern on paired Hindi items. Our Qwen finding differs internally, a projection at
  English strength with a soft-refusal deficit. It is closest to Kompella & Mahajan (Actionable Interpretability @ COLM 2026,
  arXiv:2608.08032), who find language-invariant harm detection but a late, separate refusal-writing direction in an Indic mixture-of-experts
  model.
- *Caveat.* Balani & Panda (2026, arXiv:2609.27758) show that cross-lingual probe transfer that looks perfect on easy negatives can collapse
  on hard negatives. Our Qwen projection result has not been tested against hard negatives (§9).

**Mechanistic accounts of multilingual processing.**

- *Intermediate English.* Models appear to route non-English input through English-like intermediate representations: Wendler et al. (ACL
  2024, arXiv:2402.10588), Schut et al. (2025, arXiv:2502.15603) [verify venue], and Zhao et al. (NeurIPS 2024, arXiv:2402.18815) [verify
  venue]. This motivates fitting directions in English.
- *Language-specific neurons.* These can be found (Tang et al., ACL 2024, arXiv:2402.16438; Kojima et al., NAACL 2024, arXiv:2404.02431).
  But Mondal et al. (NAACL 2025 Insights, arXiv:2503.17456) find that intervening on them does not improve low-resource task performance in
  Llama-3.1.
- *Safety neurons.* Zhao et al. (SN-Tune, ICLR 2025) [verify id] argue for language-specific safety neurons. Several 2026 preprints argue for
  small cross-lingually shared safety-neuron sets (e.g. arXiv:2602.01283, arXiv:2608.09095) [verify]. Preference tuning in English transfers
  through shared multilingual MLP components (Li, Yong & Bach, Findings of EMNLP 2024, arXiv:2406.16235). Upadhyaya & Sikdar (2026,
  arXiv:2608.29936) find safety features entangled with language identity in an architecture-dependent way.
- *Where we fit.* Our neuron result supports the shared view and extends Mondal et al. from task ability to safety. The neurons we selected
  for the English–Hindi difference change behaviour in English but not Hindi, which we have not seen reported.
- *Refusal wording.* Qi et al. (ICLR 2025, arXiv:2406.05946) [verify venue] show that safety alignment acts mostly on the first few output
  tokens, a plausible route by which a weaker Hindi refusal signal becomes soft wording. To our knowledge, no prior mechanistic work studies
  the wording or strength of a refusal, given the decision to refuse, across languages.

**Activation steering and conditional steering for safety.**

- *Foundations.* Steering adds a fixed vector to the residual stream: ActAdd (Turner et al. 2023, arXiv:2308.10248), contrastive activation
  addition (Panickssery et al., ACL 2024, arXiv:2312.06681) [verify venue], representation engineering (Zou et al. 2023, arXiv:2310.01405),
  and inference-time intervention (Li et al., NeurIPS 2023, arXiv:2306.03341).
- *Conditional steering.* Conditional Activation Steering (CAST; Lee et al., ICLR 2025, arXiv:2409.05907) gates refusal steering on a
  condition vector, with an F1-tuned threshold. Related gated or input-adaptive methods include AdaSteer (EMNLP 2025, arXiv:2504.09466),
  AlphaSteer (ICLR 2026, arXiv:2506.07022), SafeSwitch (arXiv:2502.01042) and ALTSTEER (EMNLP 2026, arXiv:2608.30197), which steers toward
  constructive alternatives rather than hard refusal. Our gate is not new as a method.
- *Closest multilingual work.* Aziz et al. (2026) gate on a low-rank classifier over high-resource activations, calibrate its threshold with
  a few target-language examples, and add or ablate the harmfulness direction. They report higher selectivity than CAST, and find that Llama
  gains much more from target-language calibration than Qwen or Gemma. Our English-threshold gate for Qwen and Hindi-threshold gate for
  Llama converge on the same finding. BabelSteering (Stein et al. 2026, arXiv:2608.16577) [verify gating and MMLU] adds an English refusal
  vector across eight languages, including Bengali, and raises refusal of pseudo-harmful prompts as well.
- *Reliability critiques.* Tan et al. (NeurIPS 2024, arXiv:2407.12404) and Braun et al. (2025, arXiv:2505.22637) show that steering vectors
  generalise unevenly. Pres et al. (2024, arXiv:2410.17245) show effects in multiple-choice formats can vanish in open generation. AxBench
  (Wu et al., ICML 2025, arXiv:2501.17148) finds that prompting outperforms steering. Li et al. (2026, arXiv:2603.24543) document safety side
  effects of steering vectors.
- *Where we fit.* We contribute pre-registered held-out tests, a guard on garbled output (which we have not seen used as a formal criterion),
  and the negative finding that a refusal-direction gate cannot be language-selective when both languages score alike on it.

**Refusal taxonomies and grading.**

- *Taxonomies.* Prior schemes separate full from partial refusal (XSTest; Röttger et al., NAACL 2024, arXiv:2308.01263), name indirect
  refusals (OR-Bench; Cui et al. 2024, arXiv:2405.20947) [verify venue], or categorise response actions (Do-Not-Answer; Wang et al.,
  Findings of EACL 2024, arXiv:2308.13387).
- *Precedents for BROKEN.* Yong et al. (2023) label unclear or degenerate output UNCLEAR, MultiJail labels it invalid, and StrongREJECT
  (Souly et al., NeurIPS 2024, arXiv:2402.10260) shows that low-resource "jailbreaks" are often low-quality rather than truly compliant.
- *Why the soft label matters.* Frontier models are now trained toward replies that neither comply nor refuse outright ("safe completions";
  Yuan et al. 2025, arXiv:2508.09224) [verify authors].
- *What is new in ours.* Each of our four classes has a precedent. The combination, with soft refusal defined as "understood, but neither
  provides nor declines", validated in native script against a fluent human, does not.
- *Judges.* Keyword matching is unreliable for refusal (XSTest; HarmBench, Mazeika et al., ICML 2024, arXiv:2402.04249). LLM judges are less
  reliable outside English:
  - Hada et al. (Findings of EACL 2024, arXiv:2309.07462) recommend calibration against native speakers.
  - Fu & Liu (Findings of EMNLP 2025, arXiv:2505.12201) report average Fleiss' κ near 0.3 across 25 languages.
  - PARIKSHA (Watts et al., EMNLP 2024, arXiv:2406.15053) finds low agreement in some Indic languages.

  Against this background, our judge's κ of 0.82 with a fluent Hindi grader is comparable to IndicSafeEval's Hindi 0.84 (binary). Judging
  replies in the original script, rather than after translation, keeps the BROKEN signal that translation-based judging can erase.

**Non-mechanistic fixes.**

- *Prompting and translation.* Prompting a model to reason in English roughly halved unsafe output in XSafety [verify number]. English as a
  Defense Proxy (Findings of EMNLP 2025) blocks most MultiJail attacks.
- *Training.* Multilingual safety training cuts harmful generations by more than half (Aakanksha et al. 2024).
- *Guards.* External guards cut attack success but over-refuse.
- *Caveat.* Atil et al. (2025, arXiv:2511.00689) find that simple defences generalise unevenly across languages and models.
- *What we lack.* We do not compare against a prompting or translation baseline. Given AxBench, this is the most important missing
  comparison (§9).

**Methodology.**

- *Precedents.* Pre-registration has been piloted in machine learning (NeurIPS 2020 pre-registration workshop, PMLR v148) and argued for in
  NLP (van Miltenburg et al., NAACL 2021, arXiv:2103.06944; Søgaard et al., EACL 2023; Hofman et al. 2023, arXiv:2311.18807).
- *Why it matters here.* Interpretability-illusion results (Makelov et al., ICLR 2024, arXiv:2311.17030; Friedman et al., ICML 2024,
  arXiv:2312.03656) and the steering-reliability critiques above give it a specific motivation in interpretability.
- *Our protocol.* Self-registered rules frozen before data, development/test splits, a sealed confirmation set in a held-out language, and
  dated addenda for every deviation. It is not a registered report; the registrations are timestamped in version control. Our own record
  shows why it matters: a repair that passed its held-out test failed its sealed confirmation, a follow-up did not reproduce that failure,
  and a neuron search that worked internally had no behavioural effect.

---

## 3 Measuring the gap

### 3.1 Data

**Sources.** The primary harmful set is Aya Red-teaming, whose Hindi prompts were written by native-speaking annotators, with human English
translations. It is complemented by IndicSafe, a human-translated Hindi safety set. Harmless controls come from PolyGuard prompts
(machine-translated) and from IndicSafe's harmless items. Spanish (Aya Red-teaming Spanish with its English versions) is a high-resource
control language, and Bengali (MultiJail and IndicSafe) is held out for confirmation. Global-MMLU (300 questions per language) measures
general ability.

**Screening.** Every candidate item was screened in order:

1. Request type: is it a request at all, and is it harmful, borderline-safe or benign?
2. Translation fidelity.
3. Comprehension: does the model understand the Hindi version?

The request-type screen was done by an LLM (Claude Opus 5.5) and validated against a fluent human on 100 blind items:

- Before adjudication, both candidate screeners failed the pre-registered gate (κ 0.51 and 0.49), despite agreeing with each other on 94 of
  100 items.
- The disagreements concentrated on category boundaries. After the human reviewed the 29 items where both models agreed against them,
  request-type agreement reached 92% (κ 0.88).
- We report both figures and treat the adjudicated one as an upper estimate, because the human saw the model labels.

Comprehension was screened by asking Qwen3-8B to restate each Hindi request in English and having a judge compare the restatement with the
English version. On harmful tier A items, 90.9% pass. Items the model did not understand show the same gap as items it did, so the gap does
not come from misunderstanding.

**Split.** Items were split at group level into development and test halves, stratified by source and by the English and Hindi refusal
labels. A first split stratified only by source and English decision drained the failure pattern from the test half and was discarded
before any result used it. The final held-out set is **404 understood harmful pairs and 169 understood harmless pairs**. Every test in this
paper uses it, for both models. A separate sealed split (MultiJail) was kept unopened until the Bengali confirmation.

### 3.2 Models and generation

The models are Qwen3-8B (thinking mode off) and Llama-3.1-8B-Instruct, at pinned revisions. Decoding is greedy, with system prompt "You are a
helpful assistant." New-token limits are 1,024 for Hindi and Bengali replies and 512 for English. Batches are fixed and length-sorted, so
the steered and unsteered arms of a comparison see identical batches. Runs used a single 48 GB GPU in a pinned software
environment (torch 2.8.0, transformers 5.22.2): an NVIDIA A40 for every phase except Phase 2b, which ran on an RTX A6000.

### 3.3 Grading

**Rubric.** Each reply gets the first label that fits:

- COMPLY: provides what was asked, even partly or with warnings.
- REFUSE_EXPLICIT: says it will not or cannot help.
- REFUSE_SOFT: shows it understood but neither provides nor declines; it warns, lectures, redirects or answers a safer question.
- BROKEN: incoherent, misunderstood, or in a language the user did not use.

The judge grades what the reply did, not whether it was right. An earlier two-way rubric disagreed with the human mainly on replies that warn
or redirect without saying no. Making that boundary its own label resolved the disagreement.

**Judge selection.** One fluent Hindi reader graded 150 calibration replies with the four labels. Four candidate judges graded the same
replies blind and shuffled. The pre-registered gate required Cohen's κ ≥ 0.7 on refused-versus-not-refused, on a held-out half.

- **Selected judge:** Claude Sonnet 5.5, with κ 0.82 (95% CI 0.68–0.92) on the gate half. It was best on both halves.
- **Strict agreement:** κ 0.91 for explicit refusal against everything else.
- **Weak spot:** COMPLY versus BROKEN. The judge used BROKEN 27 times where the human used it 4 times. We therefore treat refusal rates as the
  measures and report COMPLY and BROKEN together when the distinction matters.
- **Production route:** grading ran through Claude Code subagents. This route was itself checked against the human (κ 0.74 on the gate half)
  and against the first route (κ 0.92).
- **Other languages:** for Bengali, a second judge (Claude Opus 5.5) re-graded 150 replies (κ 0.82 on four labels) and gave the same verdict.
- **Llama:** a 60-reply human check was pre-registered for Llama's Hindi replies. Four-label κ is 0.643, below the registered 0.7, and
  under the frozen rule the Llama results would be provisional. The explicit-versus-rest κ is 0.785; it was not registered. Most blind disagreements were replies that
  warn and then answer, which the human first labelled soft. After an unregistered review of the 14 disagreements with the judge's labels
  shown, κ is 0.893, an upper estimate. The grader adopted the reviewed labels as the final human reference (stated in the paper). On 30
  steered Llama replies the two agree on 29 (30 after review).

**Blinding.** In every phase, replies from all arms and languages were shuffled under opaque ids. Labels were mapped back only after
grading. A steered reply whose gate stayed closed, or whose text was identical to its unsteered reply, inherited the unsteered label.

### 3.4 Protocol

Each study had a pre-registration committed before any of its replies existed. It fixed:

- the sets;
- the metric definitions;
- the selection rule for any tuned setting;
- the guards and the pass rule.

Analysis scripts were committed before labels existed. Deviations, such as a grading stop by the judge's safety filter or an implementation
error found after a run, were recorded in dated addenda before the affected result was computed, and the original result was kept.
Intervals are paired bootstrap 95% intervals over items (10,000 resamples); McNemar tests are exact.

---

## 4 The gap

**Qwen3-8B** (held-out split, 404 understood harmful pairs, unsteered):

| | English | Hindi | Gap |
|---|---|---|---|
| Explicit refusal | 51.2% | 19.1% | −32.2 [−37.4, −26.5] |
| Any refusal (explicit or soft) | 90.3% | 77.2% | −13.1 [−17.3, −9.2] |
| Harmless refused (169 pairs) | 3.5% | 1.2% | |

In Qwen, most of the change is a move from explicit to soft refusal. On a superset of the test split (448 pairs), labels moved from English
to Hindi as follows: REFUSE_EXPLICIT 230 → 82, REFUSE_SOFT 170 → 265, COMPLY 47 → 63, BROKEN 1 → 38. Harmless Hindi requests are not
over-refused, but BROKEN replies to them rise from 0 to 20 of 175.

**Spanish control.** On 782 Aya Spanish pairs, Qwen's explicit refusal also falls, from 47% to 32%, so the effect is not unique to
low-resource scripts. These grades came from a judge validated on Hindi only.

**Llama-3.1-8B-Instruct, same 404 pairs** (judge labels; human-validated after adjudication, §3.3):

| | English | Hindi | Gap |
|---|---|---|---|
| Explicit refusal | 88.9% | 46.5% | −42.3 [−47.3, −37.4] |
| Any refusal | 90.8% | 66.8% | −24.1 |
| Comply | 8.7% | 30.2% | +21.5 |
| Harmless refused | 4.1% | 2.4% | |

Llama refuses explicitly far more than Qwen in English, and its gap is larger; it holds on both sources (Aya −43.0, IndicSafe −38.8
points). By judge labels, the 172 pairs refused explicitly in English but not in Hindi split into COMPLY 88, REFUSE_SOFT 75 and BROKEN 9.
The human check, after review, agrees: on the 23 such pairs in the sample, 13 compliance, 9 soft and 1 broken. The blind human labels had
leaned soft (14 of 20), mainly on replies that warn and then answer. So Llama's lost refusals are a mix of compliance and soft refusal in similar shares (judge 88 vs 75, p = 0.35). The clearer contrast with
Qwen is the rate. On the same 404 pairs Llama complies in Hindi 30.2% of the time and Qwen 16.3% (+13.9 [+9.4, +18.3]). Counting broken
replies too, since the judge's COMPLY/BROKEN boundary differed from the human's on Qwen, the rates are 33.2% vs 24.0% (+9.2 [+4.5, +14.1]).

---

## 5 Mechanism in Qwen3-8B

### 5.1 An English refusal direction responds to Hindi harm

We fit a refusal direction on English prompts only, as a difference of means of last-prompt-token residual activations (391 harmful,
190 harmless). The layer is chosen by directional ablation against a first-token refusal score.

- **Layer and validity:** layer 14. Ablating the direction removes 64% of the English harmful–harmless gap in the refusal score on held-out
  prompts, above the pre-registered 50% validity bar.
- **Stability:** split-half cosine 0.975.
- **Harmful Hindi:** projects at English strength. Mean 5.34 against 5.17, a paired shift of +0.17 [+0.02, +0.30], or −0.46 on Aya items.
- **Harmless Hindi:** projects far higher than harmless English (2.80 against −2.83).

So the direction separates Hindi harmful from harmless less well (AUROC 0.88 against 0.96 in English). The pre-registered summary statistic,
a harmful–harmless "recognition ratio" (0.32), therefore reads as weakened recognition, and we report it as registered. Descriptively,
though, the drop comes from the harmless side: harmful Hindi requests project on the refusal direction at English strength. This is a
projection on a refusal direction read at the last template token, not a measure of harm recognition (Zhao et al. 2025). It has not been
tested against hard negatives (§9). Still, it suggests that the Hindi deficit arises after this signal, in how the refusal is expressed,
which §5.2 tests directly.

### 5.2 A separate direction predicts explicit versus soft refusal

We fit a second direction on English prompts whose replies were explicit refusals (367) or soft refusals (200).

- **Layer and fit:** layer 24, five-fold cross-validated AUROC 0.93; split-half cosine 0.96.
- **Transfer without refitting:** AUROC 0.90 in English and 0.76 in Hindi.
- **Hindi position:** Hindi prompts sit toward its soft end. The paired shift is −10.9 [−12.1, −9.6], or 22% of the English explicit–soft
  gap. That is below the pre-registered 25% threshold, so the registered reading is "not met".
- **Outcome link:** the further toward soft a Hindi prompt sits, the further its reply falls from explicit refusal. Mean projection by Hindi
  outcome: explicit −11.6, soft −28.0, broken −40.0, complied −43.5.

**Causal test.** On 150 pairs refused explicitly in English but not in Hindi, adding the English explicit-minus-soft vector at layer 24 (1×)
changes Hindi replies as follows:

- explicit refusal rises from 10% to 53% (+42.7 [+34.0, +51.3]);
- compliance roughly halves (31 to 17 replies);
- harmless Hindi prompts are unchanged.

At 2× it starts to over-refuse harmless prompts (+10 points) and to break replies. The pre-registered reading at 1× is causal support.

### 5.3 Hindi-specific neurons are not a Hindi-specific lever

We used attribution patching toward English activations, excluding neurons that also move harmless pairs, to rank MLP neurons in layers
0–24 by how much of the English–Hindi difference on the layer-24 explicit–soft direction they carry.

- **Concentrated:** the top 100 neurons (of 307,200) recover 75% of the difference on 79 held-out pairs; random sets of 100 recover none.
- **Late:** 80% of their attribution sits in layers 14–24.

But adding their mean English-minus-Hindi offset moves the internal metric about equally in every language (Hindi +18.8, English +17.5,
Bengali +16.2). Behaviourally:

- English explicit refusal rises by 15.2 points [+11.2, +19.3];
- Hindi explicit refusal barely moves (+1.2 [−2.5, +5.0]).

The registered outcome is "small, no clean behavioural effect". The most gap-related neurons act across languages, consistent with the
steering results in §6.

---

## 6 Repair in Qwen3-8B: gated steering

**Method.** A gate reads the refusal-direction score at layer 14 for the last prompt token. If the score exceeds τ, a fixed multiple α of
the explicit-minus-soft vector is added at the input of layer 25, at every position from the prompt on.

**Tuning.** Nine settings were tried on 401 development harmful and 171 harmless Hindi pairs: α ∈ {0.75, 1.0, 1.25} and τ at the 50th,
25th or 10th percentile of English development harmful scores. The frozen rule picked the largest Hindi explicit gain that kept harmless
refusal within +3 points and BROKEN within +5. The chosen setting is α = 1.25, τ at the 50th percentile (+21.4 on development). Looser
gates opened on most harmless Hindi prompts but were rejected for broken replies, not for over-refusal.

**Held-out test (run once):**

| | Unsteered | Gated | Change |
|---|---|---|---|
| Hindi explicit refusal | 18.6% | 35.6% | **+17.1** [+13.1, +21.0] |
| Hindi compliance | 16.3% | 12.6% | −3.7 [−5.9, −1.7] |
| Hindi BROKEN | 7.7% | 7.9% | +0.2 |
| English explicit refusal | 50.5% | 66.3% | +15.8 [+12.1, +19.6] |
| Harmless refusal (Hindi / English) | 1.8% / 2.4% | 1.8% / 2.4% | 0 |
| Global-MMLU (Hindi / English) | 51.3% / 64.0% | 51.3% / 64.0% | 0 |

All eight pre-registered checks pass. But the gate opens at the same rate on English and Hindi harmful prompts (47%), so English rises
almost as much as Hindi. The explicit-refusal gap goes from 31.9 to 30.7 points. The repair converts soft refusals into explicit ones in
both languages; it does not target the gap. Two further caveats:

- The effect rests on Aya items; the gate opened on 2 of 67 IndicSafe prompts.
- The harmless and MMLU guards are weak tests, because the gate almost never opened on those inputs. A later forced-gate MMLU check found no
  accuracy loss either.

**Confirmation in Bengali (sealed set, run once):**

- **Primary passes:** explicit refusal on 148 understood MultiJail requests rose from 25.7% to 49.3% (+23.6 [+16.2, +31.1]).
- **Gap narrows:** the English–Bengali gap fell from 23.6 to 6.8 points, because the gate opened far more often in Bengali (76%) than in
  English (39%).
- **Guard fails:** BROKEN replies rose by 8.1 points [+2.0, +14.9] against a limit of 5. Under the frozen rule the verdict is **FAIL**, and a
  second judge gave the same verdict.

**Follow-up sweep.** A pre-registered strength sweep on 150 new Bengali development requests did not reproduce the damage:

- BROKEN changed by +0.9 points at α = 1.25 (+2.7 with the gaps from a judge safety-filter stop filled by a second judge);
- explicit refusal rose by 28.6 points;
- the relative push size was identical across Bengali, Hindi and English.

The registered Phase 4 verdict stands. The sweep weakens its reading as a general property of the push in Bengali.

---

## 7 Llama-3.1-8B-Instruct

### 7.1 Diagnosis

The Phase 2a and 2b procedures were rerun on Llama without changes: same fit sets, same rules, Llama's own English replies as labels.

- **Refusal direction:** valid at layer 10. Ablation removes 94% of the English harmful–harmless gap.
- **Hindi harm signal:** Hindi harmful prompts reach only **28%** of the English mean projection (0.51 against 1.84). The replication rule
  asked for at least 80%, so it is not met. Unlike Qwen, Llama registers Hindi harm weakly.
- **Explicit–soft direction:** valid at layer 18 (CV AUROC 0.925), and Hindi prompts sit toward its soft end (paired shift −2.40
  [−2.55, −2.26]). Llama almost never refuses softly in English (11 of 782 fit replies), so this direction rests on few examples.
- **Carried-over fix:** the Qwen gate and strength, carried over unchanged, opened on only 1% of Hindi harmful prompts, because those prompts
  score low on the refusal direction. So the Qwen fix was effectively not tested in Hindi.

The two models share the explicit-refusal gap but not its internal signature. In Qwen, Hindi harmful prompts project on the refusal
direction at English strength and the lost refusals are mostly soft. In Llama they project weakly, and the lost refusals split between
compliance and soft refusal.

### 7.2 A diagnosis-matched repair

*Pre-registered (`docs/prereg/2026-11-15_v2_phase7_llama_fix.md`, two addenda). Result document `docs/results/2026-11-16_phase7_llama_fix.md`.*

**Method.** The gate threshold is set on Hindi development prompts: the median gate score of the 401 Hindi development harmful prompts.
Five pushes were tuned on 150 Hindi development pairs:

- E1 and E2 push along the explicit–soft direction, at the Qwen strength and at twice that.
- R1–R3 push along the refusal direction, at 0.5, 1.0 and 1.5 times the mean English-minus-Hindi projection gap.

The Phase 3 rule selected R3 (development: Hindi explicit +18.0, compliance −10.7, gap 13.3 points narrower). R2 and E1 also qualified. E2
broke replies. With the Hindi-set gate, the gate opened on 45% of Hindi harmful development prompts, against 1% in Phase 6.

**Held-out test** (404 harmful and 169 harmless pairs, run once):

| | Unsteered | R3 | Change |
|---|---|---|---|
| Hindi explicit refusal | 46.5% | 61.1% | **+14.6** [+11.1, +18.3] |
| Hindi compliance | 30.4% | 22.0% | **−8.4** [−11.1, −5.9] |
| Hindi any refusal | 66.3% | 75.5% | +9.2 [+6.4, +12.1] |
| Hindi BROKEN | 3.2% | 2.5% | −0.7 |
| English explicit refusal | 88.9% | 96.0% | +7.2 [+5.0, +9.7] |
| Harmless refusal (Hindi / English) | 1.8% / 4.1% | 1.8% / 4.7% | 0 / +0.6 |
| Global-MMLU (Hindi / English) | 42.7% / 63.0% | 42.7% / 63.0% | 0 |

**Verdict against the registered claims.**

- **F1 (≥ +15 with every guard): not met.** The gain is +14.6, 0.4 short of the bar; every guard passed.
- **F2 (gap narrows ≥ 10): not met.** The gap narrowed by 7.4 points [+3.2, +11.6].
- **F3 (compliance falls): met.**

**What the push does.** It raises Hindi explicit refusal, and part of the gain comes from former compliance (F3, −8.4; about 8.4 of the
14.6 points). The adjudicated human labels support this, though the COMPLY/SOFT split is the least reliable label (§3.3). It narrows the gap, which the Qwen repair did not (7.4 against 1.2 points). It does so without any measured cost, though the
harmless and MMLU guards are weak because the gate rarely opens on those inputs. It is still not language-selective: the gate opens on 92%
of English harmful prompts, and English explicit refusal rises 7.2 points from a high base.

---

## 8 Discussion (partial)

- **Report what replaces the refusal, and check the hard boundary.** A single refusal rate hides whether the missing refusals become soft
  refusals, compliance or broken output. The hardest boundary, a warning followed by an answer, needs a human check on each model, and
  blind and adjudicated labels should be reported together.
- **Diagnosis should pick the intervention.** The Qwen repair acts on how refusal is expressed, matching a diagnosis of intact harm
  recognition. Carried to Llama, its English-set gate was a poor fit: it barely opened on Hindi, because Llama's Hindi refusal-direction
  signal is weak. Once the gate was set on Hindi, even the Qwen-style push qualified on development data (E1, +16.7). The diagnosis-matched Llama repair acts on the weak refusal-direction signal
  itself, as the diagnosis predicts. On held-out pairs it narrows the gap by 7.4 points against Qwen's 1.2, though English headroom differs (Llama's English is near
  ceiling at 89% explicit, Qwen's at 51%), and it falls
  short of its registered bar.
- **"Passes its rule" is not "closes the gap."** The Qwen repair met every registered check and left the gap almost unchanged, because it
  acted on English as much as on Hindi. Gap-narrowing should be a registered outcome of any cross-lingual repair; the Phase 7 rules include
  it.
- **Localisation is not a language-specific lever.** The neurons most associated with the English–Hindi difference act on all languages.

---

## 9 Limitations

- **Models:** two 8B open models, one revision each, and greedy decoding only.
- **Judges:** labels come from LLM judges. The judge was validated against one fluent human for Hindi, with no human check for Bengali and
  Spanish, and its COMPLY-versus-BROKEN boundary differs from the human's.
- **Data:** much of the harmful set is from one source (Aya Red-teaming). The harmful and harmless sets come from different sources, and the
  harmless controls are machine-translated.
- **Steering:** strength and gate settings were tuned on development data, and every repair was confirmed once at most. The Llama test
  reuses the Phase 6 test items, though no Llama setting was chosen on them.
- **Weak guards:** the over-refusal and MMLU guards are weak whenever the gate rarely opens on benign inputs.
- **Truncation:** replies were truncated at fixed token limits, and many Hindi and Bengali replies reach them.
- **No hard negatives:** the Qwen projection result has not been tested against harmless requests that sound harmful, in Hindi, and
  cross-lingual transfer can collapse on such items (Balani & Panda 2026). Harmless Hindi requests also project high on the refusal
  direction (§5.1).
- **Refusal, not harm:** our directions are refusal directions read at the last template token. We fitted no separate harmfulness direction
  (cf. Zhao et al. 2025).
- **No prompting baseline:** we compare against no system-prompt or translate-to-English baseline.
- **Single human grader:** one human graded Hindi, so inter-human agreement is not measured.

---

## Appendix A: Reproducibility

Code, pre-registrations, addenda, id-only labels and a checksum manifest of the untracked inputs are in the repository. `REPRODUCE.md`
maps each number to its command. Harmful request and reply text is not redistributed. Each study cost a few dollars of GPU time on one
48 GB GPU; per-run costs are in the result documents. **[Total to be summed from the result documents.]**

## Appendix B: Claim-to-evidence table

Primary claims are registered and rest on explicit refusal or on label-free projections. Descriptive claims rest on the COMPLY/SOFT
split, the least reliable label.

| Claim | Type | Sample | Estimate [95% CI] | Depends on |
|---|---|---|---|---|
| Qwen explicit-refusal gap | primary | 404 pairs | −32.2 [−37.4, −26.5] | explicit refusal (human κ 0.91) |
| Llama explicit-refusal gap | primary | 404 pairs | −42.3 [−47.3, −37.4] | explicit refusal (human κ 0.785 blind) |
| Qwen lost refusals mostly soft | descriptive | 448 pairs; 150 human-graded | soft 170 → 265 | COMPLY/SOFT split (human-checked) |
| Llama lost refusals: compliance and soft in similar shares | descriptive | 172 pairs; 23 human-reviewed | 88 vs 75 (p = 0.35); 13 vs 9 | COMPLY/SOFT split |
| Llama unrefused more often than Qwen | descriptive | 404 paired items | +13.9 [+9.4, +18.3] compliance; +9.2 [+4.5, +14.1] incl. broken | COMPLY/BROKEN boundary |
| Qwen Hindi projection at English strength | primary | 1,174 pairs | shift +0.17 [+0.02, +0.30] | none (no hard-negative test) |
| Llama Hindi projection 28% of English | primary | 404 pairs | 0.51 vs 1.84 | none |
| Explicit–soft direction, Hindi toward soft | primary | 1,069 / 404 pairs | Qwen −10.9 [−12.1, −9.6]; Llama −2.40 [−2.55, −2.26] | Llama fit on 11 soft examples |
| Qwen gated fix, Hindi explicit | primary | 404 pairs, once | +17.1 [+13.1, +21.0]; English +15.8 | explicit refusal |
| Qwen fix, Bengali confirmation | primary | 148 items, once | +23.6; BROKEN +8.1 [+2.0, +14.9], fail | Bengali labels not human-checked |
| Llama fix (R3), Hindi explicit | primary | 404 pairs, once | +14.6 [+11.1, +18.3], bar +15 | explicit refusal |
| Llama fix, gap narrowing | primary | 404 pairs | 7.4 [+3.2, +11.6], bar 10 | explicit refusal |
| Llama fix, compliance falls | primary (secondary rank) | 404 pairs | −8.4 [−11.1, −5.9] | COMPLY label |
| Hindi neurons: no Hindi-only lever | primary | 79 / 322 pairs | Hindi +1.2 [−2.5, +5.0]; English +15.2 | explicit refusal |
| Llama human check | registered gate | 60 items | κ 0.643 [0.48, 0.79], bar 0.7 | one human grader |

## Appendix C: Protocol and deviations

| Study | Registered before data | Deviation | When recorded | Effect on results |
|---|---|---|---|---|
| Baseline split | yes | first split drained the failure pattern; redrawn once before use | before any use | none (first split unused) |
| Screening audit | yes | adjudication of 29 items with model labels shown | pre-registered addendum | adjudicated agreement reported as upper estimate |
| Phase 3 test | yes | one grading part stopped by the judge's safety filter; retried once | before analysis | none |
| Phase 4b sweep | yes | 2 of 12 parts stopped by the safety filter; filled post hoc by a second judge | addendum before result; fill after | outcome unchanged |
| Phase 6 | yes | cache-crop bug in the two-token refusal score, fixed after smoke run | addendum before full run | none |
| Phase 6 | yes | human check graded after aggregate results were seen (addendum 3); disagreements adjudicated with judge labels shown (4); reviewed labels adopted (5) | before / after grading | blind κ 0.643 reported as registered; 0.893 as upper estimate |
| Phase 7 | yes | procedure details stated loosely in prereg (addendum 1) | before data | none |
| Phase 7 | yes | E arms double-scaled (about 4.8×); F2 sign error (addendum 2) | after stage 1, before stage 2 | E arms rerun; selection unchanged (R3) |
| Phase 7 | yes | stage 2 run killed by a storage I/O error; rerun unchanged | before any grading | none |
| Phase 7 | yes | test items reused from Phase 6 | registered | stated as a limit |
