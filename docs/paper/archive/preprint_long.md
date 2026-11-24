# Why Two Open Chat Models Refuse Less in Hindi: Refusal Style, Internal Signals and Gated Steering

## Abstract

Safety-tuned chat models refuse harmful requests less reliably outside English. We study this gap on paired requests, asked in English and
in Hindi, in Qwen3-8B and Llama-3.1-8B-Instruct. Replies are graded with a four-way scheme that separates explicit refusals from soft ones
(warning, lecturing or redirecting without saying no). On 404 held-out pairs, explicit refusal falls from 51% to 19% in Qwen and from 89% to
47% in Llama. In Qwen most of the missing refusals become soft refusals (115 of 148); in Llama they split between compliance and soft refusal
(88 and 75 of 172), and Llama fails to refuse Hindi harmful requests 1.4–1.9 times as often as Qwen. An English-fitted refusal direction scores
harmful Hindi prompts at English strength in Qwen but at only 28% of it in Llama, although in both models it still ranks harmful above harmless
Hindi prompts (exploratory; AUROC 0.92 and 0.97). A separate direction that separates explicit from soft refusals in English raises Hindi
explicit refusal when added; we did not test whether it is independent of the refusal direction. Pre-registered gated steering raises Hindi
explicit refusal on held-out pairs by 17.1 points in Qwen and 14.6 in Llama (narrowly missing its registered bars), but it raises English by
15.8 and 7.2 points, so it is not language-selective and narrows the gap by only 1.2 and 7.4 points. The Qwen repair failed a pre-registered
Bengali guard, and Hindi-associated neurons changed English behaviour but not Hindi. The steering and neuron studies were pre-registered and
tested once on held-out data; the studies of the internal directions were registered as exploratory. We report failures alongside successes.

---

## 1 Introduction

Refusal is the main behavioural safeguard of open chat models, and it is trained mostly on English data. Prior work shows that harmful
requests in lower-resource languages are refused less often (Yong et al. 2023; Deng et al. 2024; Wang et al. 2024; §2). Most
of that work measures the gap with benchmark-level refusal rates. It rarely asks what replaces the missing refusal, or why the model's
behaviour changes when only the language changes.

We study the gap in a setting designed to make those questions answerable:

- **Paired requests.** Every harmful request is asked in English and in Hindi, the same items in both languages, and each comparison is
  paired by item. Items are screened for being a real request, for translation fidelity, and for whether the model understands the Hindi
  version.
- **A four-way outcome.** Replies are graded COMPLY, REFUSE_EXPLICIT, REFUSE_SOFT or BROKEN by an LLM judge chosen against a fluent human
  Hindi grader. The split between explicit and soft refusal turns out to carry most of the gap in both models (in Llama on a 20-pair human sample).
- **Two models, the same items.** Qwen3-8B and Llama-3.1-8B-Instruct are evaluated on the identical 404 held-out harmful pairs.
- **Diagnosis before repair, with pre-registration.** The steering and neuron studies (Phases 3–7) had their rules frozen before any of their replies
  were generated. The studies of the internal directions (Phases 2a and 2b) were registered as exploratory, the four-way rubric was revised
  after the first human comparison, and the per-layer and same-item projection statistics (§7.1) were added after the registered studies.
  Settings were tuned on development items and tested once on held-out items; deviations are recorded in dated addenda (Appendix C).

**Contributions.** Prior work has established that the gap exists, including in Hindi and in both models we study, and that English refusal
directions transfer across languages (§2). Our contributions are on the response side, in the contrast between models, and in the
protocol.

1. **A response-side decomposition of the gap.** We use paired, screened, same-request measurement with a four-way label on default
   (non-adversarial) requests. In Qwen3-8B most of the missing Hindi explicit refusals become *soft refusals*. In Llama-3.1-8B-Instruct
   they split between compliance and soft refusal in similar shares, and Llama fails to refuse Hindi harmful requests 1.4–1.9 times as
   often as Qwen on the same pairs. Most prior multilingual work scores outcomes as safe versus unsafe or on a harm scale, so it cannot separate soft refusal from
   compliance, although some schemes have a partial-refusal or unclear class (§2).
   - *A measurement caution.* The COMPLY/SOFT boundary is fragile. The hard case is a reply that warns and then answers. Our Llama human
     check fell below its registered bar on this boundary (blind four-label κ 0.643). After the grader re-applied the rubric to the
     disagreements, agreement was 0.893. We use the reviewed labels as the human reference and report both values.
2. **An explicit-versus-soft direction.** To our knowledge, a direction that separates explicit from soft refusals has not been
   reported. Fitted on English replies it separates them (cross-validated AUROC 0.93; 0.76 on Hindi prompts without refitting). Hindi
   prompts sit toward its soft end (22% of the English gap, below the registered 25% threshold), and adding it raises Hindi explicit
   refusal. We did not test whether it is independent of the refusal direction, and its outcome gradient (explicit, soft, broken,
   complied) suggests it also tracks refusal strength.
3. **Two internal readings of a shared gap.** On the same items, Qwen's Hindi harmful prompts project on the English refusal direction at English
   strength, while Llama's reach 28% of it. In both, the Hindi harmful–harmless order survives and the gap shrinks to about a third of
   English; the models differ in which side moves (exploratory, §7.1). Behaviour differs in degree: Qwen mostly hedges, while Llama hedges or complies. The weak Llama projection replicates
   prior findings (Wang et al. 2025; Aziz et al. 2026) in a paired Hindi design. The contrast between the two models on identical items is, to our
   knowledge, new. Qwen's reading rests on one harmless source and may partly reflect a language offset or a source mismatch (§5.1).
4. **Pre-registered negative results on repair.**
   - Gated steering with a refusal-direction gate raises Hindi explicit refusal on held-out pairs, but it is not language-selective: English
     rises as much, so the gap barely narrows.
   - The repair fails a sealed confirmation in a held-out language.
   - The MLP neurons most tied to the English–Hindi difference change behaviour in English, not Hindi.
5. **A pre-registered protocol for cross-lingual interpretability.** For the steering and neuron studies, rules and analyses are frozen before data, settings are tuned on
   development items, and each repair is tested once on held-out items and confirmed on a sealed held-out language, with deviations in dated
   addenda. To our knowledge, no prior steering or interpretability paper combines these.
6. **A pre-registered held-out test of a Hindi-calibrated repair for Llama.** We push along the refusal direction, with a gate
   threshold set on Hindi data. Hindi explicit refusal rises by 14.6 points (judge labels; explicit refusal is the best-validated label). The gap narrows by 7.4 points,
   against 1.2 for the Qwen-style repair. It narrowly misses its registered bars on explicit refusal and gap narrowing. This independently
   corroborates the target-language calibration that Aziz et al. (2026) found Llama needs. We did not test whether the choice of direction,
   rather than the Hindi-calibrated gate, matters: an unmatched push along the explicit–soft direction also qualified on development data
   once the gate was Hindi-calibrated (§7.2).

---

## 2 Related work

**Multilingual safety gaps.**

- *Founding papers.* Three papers in late 2023 first quantified the gap: Yong et al. (2023, arXiv:2310.02446), with translation attacks on
  GPT-4; MultiJail (Deng et al., ICLR 2024, arXiv:2310.06474); and XSafety (Wang et al., Findings of ACL 2024, arXiv:2310.00905), the first
  benchmark covering both Hindi and Bengali. Shen et al. (2024, arXiv:2401.13136) found that low-resource prompts yield more harmful
  *and* more irrelevant replies.
- *What they measured.* All of these report unsafe-output or attack-success rates, mostly on closed models and translated English prompts.
  None uses explicit refusal as the outcome.
- *Indic work.* CompositeHarm (Shukla et al., AICS @ AAAI 2026, arXiv:2602.07963) reports sharply higher attack success in Hindi and other Indic
  languages. IndicSafe (Pattnayak & Chowdhuri 2026, arXiv:2603.17915), SurakshaEval (Banerjee et al., AIES 2026, arXiv:2608.07862) and
  IndicGuard (Bramhecha et al. 2026, arXiv:2606.22841) build Indic safety benchmarks and guards.
- *Closest measurement study.* IndicSafeEval (Mondal et al., Findings of EMNLP 2026, arXiv:2609.03781) evaluates Qwen3-8B and Llama-3.1-8B,
  among others, in Hindi, Bengali, Marathi and Punjabi. Its attack set is adversarial persuasion prompts, machine-translated from English,
  scored on a five-point harm scale by Gemini-2.5-Flash (Hindi κ 0.84 binary). It reports per-language attack-success rates.
  - Our study differs: default, non-adversarial requests written natively in Hindi (Aya Red-teaming); per-item paired comparison; explicit
    versus soft refusal as the outcome; and a mechanistic analysis.
- *Closest methodological cousin.* Huang et al. (Findings of EMNLP 2026) decompose attack success by response language and show that
  aggregate rates hide in-language failures. We decompose by refusal style instead.
- *Data sources.* Our Hindi items come from Aya Red-teaming (Aakanksha et al. 2024, arXiv:2406.18682). We use its Devanagari
  prompts, written by native annotators, with their English translations, so each pair runs from native Hindi to translated English.
  Harmless controls come from PolyGuard (Kumar et al., COLM 2025, arXiv:2504.04377) and IndicSafe. Capability is measured with Global-MMLU
  (Singh et al., ACL 2025, arXiv:2412.03304).
- *Supported languages.* Hindi is an officially supported language of Llama-3.1 (Llama Team 2024; model card), so the gap is not confined to
  unsupported languages. Qwen3 reports support for 119 languages and dialects (Qwen Team 2025, arXiv:2505.09388).

**Refusal directions and their cross-lingual behaviour.**

- *Method.* Arditi et al. (NeurIPS 2024, arXiv:2406.11717) showed that refusal in open chat models is mediated by a difference-of-means
  direction: ablating it removes refusal, adding it induces refusal. We adopt their recipe with two differences. We read a single position,
  the last token of the chat template after the user turn. We validate a direction by the share of the English harmful–harmless gap in a
  first-token refusal score that its ablation removes.
- *Refinements.* Wollschläger et al. (ICML 2025, arXiv:2502.17420) show refusal can occupy multi-dimensional cones. Joad et al. (EMNLP 2026,
  arXiv:2602.02132) find distinct refusal directions for different refusal *reasons*. Wang et al. (ICLR 2025, arXiv:2410.03415)
  isolate a false-refusal direction. Our explicit-versus-soft direction differs from all of these: it separates the *style* of
  refusal given that the model refuses.
- *Harm versus refusal.* Zhao et al. (2025, arXiv:2507.11878) show that harmfulness and refusal are encoded separately, at the
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
- *Where we fit.* Our Llama finding replicates this pattern on paired Hindi items, with one qualification from an exploratory same-item
  analysis: the Hindi signal is weak in scale but keeps the harmful–harmless order (AUROC 0.97; §7.1). Our Qwen finding differs internally, a projection at
  English strength with a soft-refusal deficit. It is closest to Kompella & Mahajan (Actionable Interpretability @ COLM 2026,
  arXiv:2608.08032), who find language-invariant harm detection but a late, separate refusal-writing direction in an Indic mixture-of-experts
  model.
- *Caveat.* Balani & Panda (2026, arXiv:2609.27758) show that cross-lingual probe transfer that looks perfect on easy negatives can collapse
  on hard negatives. Our Qwen projection result has not been tested against hard negatives (§9).

**Mechanistic accounts of multilingual processing.**

- *Intermediate English.* Models appear to route non-English input through English-like intermediate representations: Wendler et al. (ACL
  2024, arXiv:2402.10588), Schut et al. (2025, arXiv:2502.15603), and Zhao et al. (2024, arXiv:2402.18815). This motivates fitting directions in English.
- *Language-specific neurons.* These can be found (Tang et al., ACL 2024, arXiv:2402.16438; Kojima et al., NAACL 2024, arXiv:2404.02431).
  But Mondal et al. (NAACL 2025 Insights, arXiv:2503.17456) find that intervening on them does not improve low-resource task performance in
  Llama-3.1.
- *Safety neurons.* Zhao et al. (SN-Tune, ICLR 2025) argue for language-specific safety neurons. Several 2026 preprints argue for
  small cross-lingually shared safety-neuron sets (Zhang et al. 2026, arXiv:2602.01283; Miao et al. 2026, arXiv:2608.09095). Preference tuning in English transfers
  through shared multilingual MLP components (Li, Yong & Bach, Findings of EMNLP 2024, arXiv:2406.16235). Upadhyaya & Sikdar (2026,
  arXiv:2608.29936) find safety features entangled with language identity in an architecture-dependent way.
- *Where we fit.* Our neuron result supports the shared view and extends Mondal et al. from task ability to safety. The neurons we selected
  for the English–Hindi difference change behaviour in English but not Hindi, which we have not seen reported.
- *Refusal wording.* Qi et al. (2024, arXiv:2406.05946) show that safety alignment acts mostly on the first few output
  tokens, a plausible route by which a weaker Hindi refusal signal becomes soft wording. To our knowledge, no prior mechanistic work studies
  the wording or strength of a refusal, given the decision to refuse, across languages.

**Activation steering and conditional steering for safety.**

- *Foundations.* Steering adds a fixed vector to the residual stream: ActAdd (Turner et al. 2023, arXiv:2308.10248), contrastive activation
  addition (Panickssery et al. 2023, arXiv:2312.06681), representation engineering (Zou et al. 2023, arXiv:2310.01405),
  and inference-time intervention (Li et al., NeurIPS 2023, arXiv:2306.03341).
- *Conditional steering.* Conditional Activation Steering (CAST; Lee et al., ICLR 2025, arXiv:2409.05907) gates refusal steering on a
  condition vector, with an F1-tuned threshold. The classify-then-steer rationale was stated earlier by Cooper Stickland et al. (2024,
  arXiv:2406.15518). Related gated or input-adaptive methods include AdaSteer (EMNLP 2025, arXiv:2504.09466),
  AlphaSteer (ICLR 2026, arXiv:2506.07022), SafeSwitch (arXiv:2502.01042) and ALTSTEER (EMNLP 2026, arXiv:2608.30197), which steers toward
  constructive alternatives rather than hard refusal. Our gate is not new as a method.
- *Closest multilingual work.* Aziz et al. (2026) gate on a low-rank classifier over high-resource activations, calibrate its threshold with
  a few target-language examples, and add or ablate the harmfulness direction. They report higher selectivity than CAST, and find that Llama
  gains much more from target-language calibration than Qwen or Gemma. Our English-threshold gate for Qwen and Hindi-threshold gate for
  Llama converge on the same finding. BabelSteering (Stein et al. 2026, arXiv:2608.16577) adds an English refusal
  vector across eight languages, including Bengali, and raises refusal of pseudo-harmful prompts as well.
- *Reliability critiques.* Tan et al. (NeurIPS 2024, arXiv:2407.12404) and Braun et al. (2025, arXiv:2505.22637) show that steering vectors
  generalise unevenly. Pres et al. (2024, arXiv:2410.17245) show effects in multiple-choice formats can vanish in open generation. AxBench
  (Wu et al., ICML 2025, arXiv:2501.17148) finds that prompting outperforms steering. Li et al. (2026, arXiv:2603.24543) document safety side
  effects of steering vectors.
- *Where we fit.* We contribute pre-registered held-out tests, a guard on garbled output (which we have not seen used as a formal criterion),
  and the negative finding that a refusal-direction gate cannot be language-selective when both languages score alike on it.

**Refusal taxonomies and grading.**

- *Taxonomies.* Prior schemes separate full from partial refusal (XSTest; Röttger et al., NAACL 2024, arXiv:2308.01263), name indirect
  refusals (OR-Bench; Cui et al., ICML 2025, arXiv:2405.20947), or categorise response actions (Do-Not-Answer; Wang et al.,
  Findings of EACL 2024, arXiv:2308.13387).
- *Precedents for BROKEN.* Yong et al. (2023) label unclear or degenerate output UNCLEAR, MultiJail labels it invalid, and StrongREJECT
  (Souly et al., NeurIPS 2024, arXiv:2402.10260) shows that low-resource "jailbreaks" are often low-quality rather than truly compliant.
- *Why the soft label matters.* Frontier models are now trained toward replies that neither comply nor refuse outright ("safe completions";
  Yuan et al. 2025, arXiv:2508.09224).
- *What is new in ours.* Each of our four classes has a precedent. The combination, with soft refusal defined as "understood, but neither
  provides nor declines", validated in native script against a fluent human, does not.
- *Judges.* Keyword matching is unreliable for refusal (XSTest; HarmBench, Mazeika et al., ICML 2024, arXiv:2402.04249). LLM judges are less
  reliable outside English:
  - Hada et al. (Findings of EACL 2024, arXiv:2309.07462) recommend calibration against native speakers.
  - Fu & Liu (Findings of EMNLP 2025, arXiv:2505.12201) report average Fleiss' κ near 0.3 across 25 languages.
  - PARIKSHA (Watts et al., EMNLP 2024, arXiv:2406.15053) finds low agreement in some Indic languages.

  Against this background, our judge's binary κ of 0.82 with a fluent Hindi grader is comparable to IndicSafeEval's Hindi 0.84 (binary),
  but our four-label κ of 0.62 is below its non-binary 0.70, and the soft-versus-comply boundary is the weak point. Judging
  replies in the original script, rather than after translation, keeps the BROKEN signal that translation-based judging can erase.

**Non-mechanistic fixes.**

- *Prompting and translation.* Prompting ChatGPT to evoke safety knowledge cut its unsafe non-English responses from 19.1% to 9.7% in XSafety. English as a
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
English version. On harmful tier A items (the screened harmful part of the baseline pool, 1,174 items), 90.9% pass. The 43 test pairs the model did not
understand show a gap at least as large (41.9 points, 95% interval 23.3–60.5, against 32.2 on understood pairs), and we analyse understood
pairs only. The gap is therefore not explained by failures to understand the screened items, though the screen is itself model-based.

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
- **Four labels:** four-label κ on the gate half is 0.62 (this route) and 0.55 (production route). The soft-versus-comply boundary is
  the least reliable, and the judge's errors there run in opposite directions in the two models: in Qwen's Hindi calibration the judge
  labelled 9 of the human's 80 COMPLY replies as soft, and in the Llama check it labelled 6 of the human's 15 soft replies as compliance.
- **Weak spot:** COMPLY versus BROKEN. The judge used BROKEN 27 times where the human used it 4 times. We therefore treat refusal rates as the
  measures and report COMPLY and BROKEN together when the distinction matters.
- **Production route:** grading ran through Claude Code subagents. This route was itself checked against the human (κ 0.74 on the gate half)
  and against the first route (κ 0.92).
- **Other languages:** for Bengali, a second judge (Claude Opus 5.5) re-graded 150 replies (κ 0.82 on four labels) and gave the same verdict.
- **Llama:** a 60-reply human check was pre-registered for Llama's Hindi replies. Four-label κ is 0.643, below the registered 0.7, and
  under the frozen rule the Llama results would be provisional. The explicit-versus-rest κ is 0.785; it was not registered. Most blind disagreements were replies that
  warn and then answer, which the human first labelled soft. After an unregistered review of the 14 disagreements with the judge's labels
  shown, κ is 0.893, an upper estimate. The sole human grader is the author. The Llama sheet was graded after aggregate Llama results had been seen, and the review changed 10
of the 14 disagreements, all to the judge's label. We use the reviewed labels as the human reference and treat 0.893 as an upper estimate. On 30
  steered Llama replies the two agree on 29 (30 after review).

**Blinding.** In every phase, replies from all arms and languages were shuffled under opaque ids. Labels were mapped back only after
grading. A steered reply whose gate stayed closed, or whose text was identical to its unsteered reply, inherited the unsteered label.

### 3.4 Protocol

Studies are named by phase: Phase 0 judge selection (§3.3); Phase 1 baseline gap (§4); Phases 2a and 2b refusal and explicit–soft
directions (§5.1, §5.2); Phase 3 gated steering in Qwen (§6); Phases 4 and 4b Bengali confirmation and strength sweep (§6); Phase 5 Hindi
neurons (§5.3); Phase 6 Llama replication (§7.1); Phase 7 Llama repair (§7.2).

Each registered study had a pre-registration committed before any of its replies existed (Phases 2a and 2b were registered as
exploratory diagnoses). It fixed:

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

These rates come from the baseline run. The steering experiments (§6) regenerate the unsteered replies in the same run as the steered ones, so
each comparison has its own unsteered baseline. For Qwen it is 50.5% in English and 18.6% in Hindi (a 31.9-point gap), under a point from the
baseline run.

![Figure 1](figures/fig1_what_replaces_refusal.png)

**Figure 1.** Replies to the same 404 held-out harmful requests in English (open) and Hindi (filled), both models, with Wilson 95%
intervals. The bar splits the requests refused explicitly in English but not in Hindi by what the Hindi reply did.

In Qwen, most of the change is a move from explicit to soft refusal. On the 404 held-out pairs (unsteered replies of the Phase 3 run),
labels moved from English to Hindi as follows: REFUSE_EXPLICIT 204 → 75, REFUSE_SOFT 154 → 232, COMPLY 45 → 66, BROKEN 1 → 31. Of the 148
requests refused explicitly in English but not in Hindi, 115 became soft refusals, 20 compliance and 13 broken replies (the bar in
Figure 1). Harmless Hindi requests are not over-refused, but BROKEN replies to them rise from 0 to 21 of 169.

**Spanish control.** On 782 Aya Spanish pairs, Qwen's explicit refusal also falls, from 47% to 32%, so the effect is not unique to
low-resource scripts. These grades came from a judge validated on Hindi only. The Spanish run is provisional: its items were not screened, replies were
capped at 512 tokens in both languages, and BROKEN replies rise from about 10 to about 43 in Spanish.

**Llama-3.1-8B-Instruct, same 404 pairs** (judge labels; human-validated after adjudication, §3.3):

| | English | Hindi | Gap |
|---|---|---|---|
| Explicit refusal | 88.9% | 46.5% | −42.3 [−47.3, −37.4] |
| Any refusal | 90.8% | 66.8% | −24.1 |
| Comply | 8.7% | 30.2% | +21.5 |
| Harmless refused | 4.1% | 2.4% | |

The Llama steering run (§7.2) regenerates these unsteered replies with different batching (412 of 1,146 differ), giving slightly
different rates (any refusal 66.3%, comply 30.4%, harmless Hindi refusal 1.8%). Llama refuses explicitly far more than Qwen in English, and its gap is larger; it holds on both sources (Aya −43.0, IndicSafe −38.8
points). By judge labels, the 172 pairs refused explicitly in English but not in Hindi split into COMPLY 88, REFUSE_SOFT 75 and BROKEN 9.
The human check, after review, agrees: on the 23 such pairs in the sample, 13 compliance, 9 soft and 1 broken. The blind human labels had
leaned soft (14 of 20), mainly on replies that warn and then answer. So Llama's lost refusals are a mix of compliance and soft refusal in similar shares (judge 88 vs 75, p = 0.35), unlike Qwen's, where
115 of 148 became soft refusals and 20 compliance (the same paired-transition basis). The judge's soft-versus-comply errors run in opposite
directions in the two models (§3.3), so the size of this contrast is uncertain. The clearer contrast with
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
- **Harmful Hindi:** projects at English strength. On tier A, mean 5.34 against 5.17, a paired shift of +0.17 [+0.02, +0.30], or −0.46 on Aya items.
- **Harmless Hindi:** projects far higher than harmless English (2.80 against −2.83).

So the direction separates Hindi harmful from harmless less well (tier A, 1,174 pairs: AUROC 0.88 against 0.96 in English; 0.92 against
0.98 on the 404 + 169 test pairs, §7.1). The pre-registered summary statistic,
a harmful–harmless "recognition ratio" (0.32 on tier A; 0.33 on the test pairs), therefore reads as weakened recognition, and we report it as registered. Descriptively,
though, the drop comes from the harmless side: harmful Hindi requests project on the refusal direction at English strength. This is a
projection on a refusal direction read at the last template token, not a measure of harm recognition (Zhao et al. 2025). It has not been
tested against hard negatives (§9). The Hindi deficit may therefore arise downstream of this signal, in how the refusal is expressed; this is not established, and §5.2
probes it. Two explanations of the high harmless-Hindi scores remain open: a language offset (Hindi input moves the score by itself) and
a source mismatch (the harmful set is native Hindi from Aya, the harmless set is machine-translated PolyGuard). The Qwen reading is
therefore also a statement about those sources.

![Figure 2](figures/fig4_layers.png)

**Figure 2.** Per-layer English refusal direction in both models. Top: share of the harmful–harmless refusal-score gap removed by
ablating each layer's direction (Qwen on the fit prompts, where layer 14 removes 62%; 64% on held-out prompts). Bottom: harmful-versus-
harmless AUROC within each language (Qwen: tier A pairs; Llama: the 404 + 169 test pairs, exploratory, §7.1). Band: chosen layer.

![Figure 3](figures/fig2_projection.png)

**Figure 3.** Mean projection on each model's refusal direction at its chosen layer, same 404 harmful and 169 harmless test pairs, English
(ring) and Hindi (dot), bootstrap 95% intervals. Hindi harmful prompts reach 100% of the English value in Qwen and 28% in Llama; harmless
Hindi prompts in Qwen sit far above harmless English. These are test-pair means, so they differ slightly from the tier-A means in the
text.

### 5.2 A separate direction predicts explicit versus soft refusal

We fit a second direction on English prompts whose replies were explicit refusals (367) or soft refusals (200).

- **Layer and fit:** layer 24, five-fold cross-validated AUROC 0.93; split-half cosine 0.96.
- **Transfer without refitting:** AUROC 0.90 in English and 0.76 in Hindi.
- **Hindi position:** Hindi prompts sit toward its soft end. The paired shift is −10.9 [−12.1, −9.6], or 22% of the English explicit–soft
  gap. That is below the pre-registered 25% threshold, so the registered reading is "not met".
- **Outcome link:** the further toward soft a Hindi prompt sits, the further its reply falls from explicit refusal. Mean projection by Hindi
  outcome: explicit −11.6, soft −28.0, broken −40.0, complied −43.5. The gradient runs explicit, soft, broken, complied, so the axis
  tracks refusal strength as much as style.

**Causal test.** On 150 pairs refused explicitly in English but not in Hindi, adding the English explicit-minus-soft vector at layer 24 (1×)
changes Hindi replies as follows:

- explicit refusal rises from 10% to 53% (+42.7 [+34.0, +51.3]);
- compliance roughly halves (31 to 17 replies);
- harmless Hindi prompts are unchanged.

At 2× it starts to over-refuse harmless prompts (+10 points) and to break replies. The pre-registered reading at 1× is causal support for an effect on Hindi explicit refusal. We did not compare with a random vector
or with the refusal direction at matched norm, so specificity is untested; in Llama a push along the refusal direction did at least as well
on development data (§7.2).

### 5.3 Hindi-specific neurons are not a Hindi-specific lever

We used attribution patching toward English activations, excluding neurons that also move harmless pairs, to rank MLP neurons in layers
0–24 by how much of the English–Hindi difference on the layer-24 explicit–soft direction they carry.

- **Concentrated:** the top 100 neurons (of 307,200) recover 75% of the difference on 79 held-out pairs; random sets of 100 recover none.
- **Late:** 80% of their attribution sits in layers 14–24.

But adding their mean English-minus-Hindi offset moves the internal metric about equally in every language (Hindi +18.8, English +17.5,
Bengali +16.2). Behaviourally:

- English explicit refusal rises by 15.2 points [+11.2, +19.3];
- Hindi explicit refusal barely moves (+1.2 [−2.5, +5.0]).

We tested one offset size, about one third of the steering that raised Hindi explicit refusal, and did not sweep it; Hindi BROKEN
replies rose by 4.3 points (interval excludes zero). The registered outcome is "small, no clean behavioural effect". The most gap-related neurons act across languages, consistent with the
steering results in §6.

---

## 6 Repair in Qwen3-8B: gated steering

**Method.** A gate reads the refusal-direction score at layer 14 for the last prompt token. If the score exceeds τ, a fixed multiple α of
the explicit-minus-soft vector is added at the input of layer 25, at every position from the prompt on.

**Tuning.** Nine settings were tried on 401 development harmful and 171 harmless Hindi pairs: α ∈ {0.75, 1.0, 1.25} and τ at the 50th,
25th or 10th percentile of English development harmful scores. The frozen rule picked the largest Hindi explicit gain that kept harmless
refusal within +3 points and BROKEN within +5. The chosen setting is α = 1.25, τ at the 50th percentile (+21.4 on development). Looser
gates opened on most harmless Hindi prompts but were rejected for broken replies, not for over-refusal.

![Figure 4](figures/fig5_mechanism.png)

**Figure 4.** Where each repair reads and writes. Qwen (this section) gates at layer 14 with an English-set threshold and adds the
explicit–soft vector at layer 25. Llama (§7.2) gates at layer 10 with a Hindi-set threshold and adds along the refusal direction at layer 11.

![Figure 5](figures/fig6_dose.png)

**Figure 5.** Development sweeps: each candidate setting's Hindi explicit-refusal gain against its change in broken replies, with the
registered guards. Filled: the setting taken to the held-out test. Qwen: nine settings, 401 development pairs; Llama (§7.2): five pushes,
150 development pairs.

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
  asked for at least 80%, so it is not met. The Hindi refusal-direction signal is weak in scale.
- **Order versus scale (exploratory, added 2026-11-17, not registered).** A forward-pass run projected every layer's last-token output on
  that layer's English refusal direction for the 404 + 169 test pairs; its layer-10 values reproduce the
  registered projections (mean absolute difference 0.006). At layer 10 the direction still ranks Hindi harmful above Hindi harmless
  prompts almost as well as in English (AUROC 0.97 against 0.99), but the Hindi harmful–harmless gap is 0.30 of the English gap. On the
  same items, Qwen's layer-14 gate scores give a similar gap ratio (0.33; AUROC 0.92 against 0.98). So the two models do not differ in
  how much Hindi separation they keep; they differ in which side moves. All of Qwen's shrinkage comes from harmless Hindi prompts rising,
  while about three quarters of Llama's comes from harmful Hindi prompts falling. This qualifies the reading of the 28% result: Llama's
  Hindi harm signal is weak in scale but well ordered, which is why a threshold set on Hindi data can separate it (§7.2).
- **Explicit–soft direction:** valid at layer 18 (CV AUROC 0.925), and Hindi prompts sit toward its soft end (paired shift −2.40
  [−2.55, −2.26]). Llama almost never refuses softly in English (11 of 782 fit replies), so this direction rests on few examples.
- **Carried-over fix:** the Qwen gate and strength, carried over unchanged, opened on only 1% of Hindi harmful prompts, because those prompts
  score low on the refusal direction. So the Qwen fix was effectively not tested in Hindi.

The two models share the explicit-refusal gap but not its internal signature. In Qwen, Hindi harmful prompts project on the refusal
direction at English strength and the lost refusals are mostly soft. In Llama they project weakly but in the right order, and the
lost refusals split between compliance and soft refusal. The 100% versus 28% comparison mixes sources, layers and scales, and we did not
compute per-source ratios for Llama; the same-item statistics are exploratory.

### 7.2 A Hindi-calibrated repair

*Pre-registered before any Llama steering reply existed (two addenda, Appendix C).*

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

**What the push does.** It raises Hindi explicit refusal, and part of the gain comes from former compliance
(F3, −8.4 points). Since any refusal rose by 9.2 points, the remaining 5.4 points of the explicit-refusal gain are, by marginals, soft
refusals made explicit; we did not compute item-level transitions. The adjudicated human labels support this, though the COMPLY/SOFT split is the least reliable label (§3.3). It narrows the gap, which the Qwen repair did not (7.4 against 1.2 points), though English headroom differs between the models. It does so without any measured cost, though the
harmless and MMLU guards are weak because the gate rarely opens on those inputs. It is still not language-selective: the gate opens on 92%
of English harmful prompts, and English explicit refusal rises 7.2 points from a high base.

![Figure 6](figures/fig3_repairs.png)

**Figure 6.** Explicit refusal before and after each repair on held-out pairs, run once, with paired bootstrap intervals and the gap
before and after. Qwen (§6) passes its checks without narrowing the gap; Llama narrows it by 7.4 points and narrowly misses its bars.

---

## 8 Discussion

- **Report what replaces the refusal, and check the hard boundary.** A single refusal rate hides whether the missing refusals become soft
  refusals, compliance or broken output. The hardest boundary, a warning followed by an answer, needs a human check on each model, and
  blind and adjudicated labels should be reported together.
- **Gate calibration may matter more than the direction pushed.** The Qwen repair acts on how refusal is expressed, motivated by the
  reading that Hindi harmful prompts reach English strength on the refusal direction. Carried to Llama, its English-set gate was a poor fit: it barely opened on Hindi, because Llama's Hindi refusal-direction
  signal is weak. A threshold at the Hindi development median opens on about half of Hindi harmful prompts by construction (47% on the
  test pairs). What shows the signal is well ordered (§7.1) is that the same threshold opens on none of the 169 Hindi harmless test
  prompts (Figure 7). Once the gate was set on Hindi, even the Qwen-style push qualified on development data (E1, +16.7). The Llama repair pushes along the refusal direction instead, but we did not test whether that choice,
  rather than the Hindi-calibrated gate, accounts for its better result. On held-out pairs it narrows the gap by 7.4 points against Qwen's 1.2, though English headroom differs (Llama's English is near
  ceiling at 89% explicit, Qwen's at 51%), and it falls
  short of its registered bar.
- **The gates were not language-selective (Figure 7).** Both gates read how harmful-looking a prompt is. At its registered threshold
  Qwen's gate opens on 48% of English and 47% of Hindi harmful test prompts; Llama's Hindi-set threshold opens on 92% and 47%. Lower Qwen
  thresholds opened on more Hindi than English harmful development prompts (92% against 75% at the 25th percentile) but were rejected for
  broken replies, and in Bengali the registered gate opened more often than in English (76% against 39%). Selectivity therefore depends on the
  threshold and the language; at the thresholds we registered, neither gate was language-selective.

![Figure 7](figures/fig7_gate.png)

**Figure 7.** Gate scores of every held-out prompt (unsteered), at the gate layer, with each model's threshold as run and the share of
each group above it.

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
- **Truncation:** replies were truncated at fixed token limits (1,024 tokens in Hindi and Bengali, 512 in English), and many reach them (571
  of 1,174 tier A Hindi replies). A registered truncation check was withdrawn, so a reply cut off before it answers could be read as soft.
- **Pairing and translation:** the Hindi side of each Aya pair is native and the English side a translation, and we ran no back-translation or
  romanised-Hindi control, so language, script and translation fidelity are not separated.
- **No hard negatives:** the Qwen projection result has not been tested against harmless requests that sound harmful, in Hindi, and
  cross-lingual transfer can collapse on such items (Balani & Panda 2026). Harmless Hindi requests also project high on the refusal
  direction (§5.1).
- **Exploratory analysis:** the same-item AUROC and gap-ratio comparison and the Llama per-layer curves (§7.1, Figure 2) were added
  after the registered studies. They are descriptive and test no hypothesis.
- **Refusal, not harm:** our directions are refusal directions read at the last template token. We fitted no separate harmfulness direction
  (cf. Zhao et al. 2025).
- **No prompting baseline:** we compare against no system-prompt or translate-to-English baseline.
- **Single human grader:** the author is the only human grader of Hindi, so inter-human agreement is not measured, and the Llama sheet
  was graded after aggregate results had been seen.
- **Soft versus comply:** four-label κ is 0.62 and 0.55 on the two judge routes, and the judge's errors on this boundary run in opposite
  directions in the two models (§3.3), which bears directly on the Qwen versus Llama contrast.
- **Statistics:** no correction for multiple comparisons, no repeat-run noise estimate for the nine-setting tuning, and test items are
  reused across Phases 3, 6 and 7.
- **Verifiability:** the registrations are timestamped in version control in a private repository and are available on request.

---

## Ethics and data

All harmful requests come from existing public safety datasets. No request or reply text is reproduced in this paper; items are referred
to by identifier. We study refusal behaviour and a defensive repair, and we report the repair's failures as well as its successes.

## Appendix A: Reproducibility

Code, pre-registrations with their dated addenda, id-only labels and a checksum manifest of the untracked inputs are archived with the
project, together with a guide that maps each number to its command. They are available from the author on request. Harmful request and
reply text is not redistributed. Every study ran on a single 48 GB GPU in a pinned software environment (§3.2).

## Appendix B: Claim-to-evidence table

Primary claims are registered and rest on explicit refusal or on label-free projections. Descriptive claims rest on the COMPLY/SOFT
split, the least reliable label.

| Claim | Type | Sample | Estimate [95% CI] | Depends on |
|---|---|---|---|---|
| Qwen explicit-refusal gap | primary | 404 pairs | −32.2 [−37.4, −26.5] | explicit refusal (human strict κ 0.91, first route; 0.87 production route) |
| Llama explicit-refusal gap | primary | 404 pairs | −42.3 [−47.3, −37.4] | explicit refusal (human κ 0.785 blind) |
| Qwen lost refusals mostly soft | descriptive | 448 pairs; 150 human-graded | soft 170 → 265 | COMPLY/SOFT split (human-checked) |
| Llama lost refusals: compliance and soft in similar shares | descriptive | 172 pairs; 23 human-reviewed | 88 vs 75 (p = 0.35); 13 vs 9 | COMPLY/SOFT split |
| Llama unrefused more often than Qwen | descriptive | 404 paired items | +13.9 [+9.4, +18.3] compliance; +9.2 [+4.5, +14.1] incl. broken | COMPLY/BROKEN boundary |
| Qwen Hindi projection at English strength | primary | 1,174 pairs | shift +0.17 [+0.02, +0.30] | none (no hard-negative test) |
| Llama Hindi projection 28% of English | primary | 404 pairs | 0.51 vs 1.84 | none |
| Hindi order kept, separation about a third of English (both models) | exploratory | 404 + 169 test pairs | AUROC HI 0.92 (Qwen) / 0.97 (Llama); gap ratio 0.33 / 0.30 | per-layer projection run; gate scores; each figure number is checked by the figure-data export script |
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
| Per-layer projections (Llama) | no (exploratory) | forward-pass run added 2026-11-17 to draw the per-layer figure; same-item AUROC and gap ratios computed for both models | 2026-11-17, Phase 6 result addendum | no registered claim changes; qualifies the reading of the 28% result as scale, not order |
