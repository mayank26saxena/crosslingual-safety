# Prior work anticipates the mechanisms, not the decomposition

The study's strongest novel contributions are the **response-side decomposition of the Hindi gap**: missing refusals become *soft refusals* in Qwen3-8B but *compliance* in Llama-3.1-8B-Instruct. Two further pieces are new as far as these notes show: an **explicit-vs-soft refusal direction**, and a **pre-registered protocol with a sealed held-out-language confirmation**. None of these appears in the 2023–2026 literature these notes cover. Most of the study's mechanistic building blocks have been done already. The English-fitted DiM refusal direction is Arditi et al. Its approximate cross-lingual universality, including on Llama-3.1-8B-Instruct, is Wang et al. (NeurIPS 2025). The observation that low-resource harm "projects lower but stays separable" is Aziz et al. (2026), a paper that includes **Hindi and Llama-3.1-8B-Instruct**. Gated refusal steering is CAST (ICLR 2025). A gate calibrated with target-language examples, which Llama needs and Qwen does not, is Aziz et al. again. The in-progress Llama fix should therefore be framed as an independent corroboration of Aziz et al., not as a new method. Two 2026 papers set the bar for the claims that have to be softened. Balani & Panda show that English-trained harm probes that look universal on easy negatives collapse for low-resource languages once hard negatives are used. Zhao et al. (2025) show that harmfulness and refusal are encoded separately. Together they mean the Qwen claim of "English-strength Hindi recognition" cannot rest on refusal-direction projection alone, especially since harmless Hindi also sits high on that axis. The single most important paper to read in full before submission is **IndicSafeEval** (Findings of EMNLP 2026). A secondary snippet says it evaluates the same two 8B models on Hindi and Bengali, with a Hindi judge validated by native-speaker kappa. Several other headline numbers and venues are still unverified ([M]/[K]/[U]/[FLAG]), so the verification list at the end is a submission blocker rather than a courtesy.

Verification tags are carried over from the five source notes. Their meanings differ slightly by note:

- **[V]**, **[V-arXiv]**, **[V-web]**: checked this session against the abstract, HTML or proceedings page. **[V-meta]** means only the metadata was confirmed.
- **[M]** and **[K]**: from prior knowledge, not re-fetched.
- **[U]**, **UNVERIFIED** and **[FLAG]**: from a snippet or memory, or internally uncertain.

Where two notes tag the same paper differently, both tags are shown.

**Data-script note.** The multilingual-safety note says that Aya Red-teaming Hindi prompts are *romanized* (transliterated with Google Translate) and advises controlling for script ([arXiv HTML](https://arxiv.org/html/2406.18682v2)). The study checked all 914 of its Aya Hindi items and found them in Devanagari. That note therefore does not apply to the study's items. The paper should still state the script explicitly, because readers who know the dataset card may assume romanization. The study's count of 914 Aya Hindi items also differs from the 915 the note gives for the dataset. The gap is presumably the screening step and should be documented.

## Eight datasets cover Hindi, but none measures explicit refusal on paired items

The multilingual safety gap was established in October 2023 by three nearly simultaneous papers, then dissected by Shen et al. All four measured **unsafe-output or attack-success rates**, mostly on closed GPT-3.5/4-era models, using translated English prompts. None used explicit refusal as the dependent variable. Most datasets are *implicitly* paired (translations of the same request), yet they report *unpaired per-language aggregates*. The only verified per-item paired statistics are IndicSafe's 12.8% cross-language agreement and Huang et al.'s per-model sign test ([IndicSafe](https://arxiv.org/abs/2603.17915); [Huang et al.](https://aclanthology.org/2026.findings-emnlp.778/)). Hindi was treated as a *high-resource* language in 2023 work, where GPT-4's Hindi attack success was only **6.54% vs 0.96% English** ([Yong et al.](https://arxiv.org/html/2310.02446v1)). A large Hindi deficit in default refusal on open 8B models is therefore not redundant with that work. It is, however, a gap in an **officially supported language** for Llama-3.1, which supports Hindi and Spanish but not Bengali ([Llama 3 Herd](https://arxiv.org/abs/2407.21783)). That framing strengthens the paper.

### Foundational gap papers

- **Yong, Menghini, Bach (2023). "Low-Resource Languages Jailbreak GPT-4." NeurIPS 2023 SoLaR Workshop (best paper). arXiv:2310.02446.** [M] in the gaps note; [V] in the grading note.
  - *Did/found:* AdvBench was machine-translated into 12 languages. Combined low-resource languages bypassed GPT-4 about **79%** of the time. Outputs were labelled BYPASS, REJECT or UNCLEAR; UNCLEAR covers degenerate or misread output. Hindi and Bengali showed small gaps ([arXiv](https://arxiv.org/abs/2310.02446)).
  - *Relation:* the canonical translation-attack citation. It uses a closed model and aggregate rates. UNCLEAR is the direct precedent for BROKEN.
- **Deng, Zhang, Pan, Bing (2024). "Multilingual Jailbreak Challenges in Large Language Models" (MultiJail). ICLR 2024. arXiv:2310.06474.** [M], [K]; label details [FLAG].
  - *Did/found:* 315 prompts were human-translated into 9 languages, including Bengali but not Hindi. Low-resource languages give about 3x the unsafe rate. Outputs were labelled safe, unsafe or invalid. Self-Defence SFT reduced the unsafe rate by 6.24% ([arXiv](https://arxiv.org/abs/2310.06474)).
  - *Relation:* "invalid" corresponds to BROKEN. A later paper reports "severe mistranslations" in MultiJail [U].
- **Wang, Tu, Chen, Yuan, Huang, Jiao, Lyu (2024). "All Languages Matter: On the Multilingual Safety of Large Language Models" (XSafety). Findings of ACL 2024. arXiv:2310.00905.** [M], [K].
  - *Did/found:* the first benchmark to cover **both Hindi and Bengali** (10 languages, 14 issue types). Judging was done by ChatGPT after translating outputs to English. Prompting the model to think in English roughly halved ChatGPT's non-English unsafe rate. The figure is often quoted as 19.1% → 9.7% and must be checked [M] ([arXiv](https://arxiv.org/abs/2310.00905)).
  - *Relation:* the earliest Hindi and Bengali coverage, and a prompting-fix precedent.
- **Shen et al. (2024). "The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts." arXiv:2401.13136.** [M] in the gaps note; venue [FLAG]/UNVERIFIED elsewhere.
  - *Did/found:* low-resource prompts yield both **more harmful and more irrelevant** responses. Safety tuning in a low-resource language helps little, which points to a pretraining deficiency ([arXiv](https://arxiv.org/abs/2401.13136)).
  - *Relation:* "irrelevant" is the nearest prior to BROKEN and REFUSE_SOFT. The study's Global-MMLU control argues against a pure capability explanation for Hindi.
  - **Conflict:** the notes disagree on the author list. The gaps note has Shen, Tan, Chen, Chen, Zhang, Xu, Zheng, Koehn, Khashabi; the grading note has Shen, Tan, Chen, Zhang, Li, Fang, Wei. Resolve against the arXiv page.
- **Li et al. (2024). "A Cross-Language Investigation into Jailbreak Attacks in Large Language Models." arXiv:2401.16765.** [M]
  - Semantic-preserving multilingual jailbreaks, with representation analysis. Adjacent ([arXiv](https://arxiv.org/abs/2401.16765)).
- **Other attack framings, all [M]:**
  - Xu et al. 2024, "Cognitive Overload," Findings of NAACL 2024, arXiv:2311.09827 ([arXiv](https://arxiv.org/abs/2311.09827)).
  - Upadhayay & Behzadan 2024, "Sandwich Attack," arXiv:2404.07242.
  - Puttaparthi et al. 2023, arXiv:2312.10524.
  - Ghanim et al. 2024, "Jailbreaking LLMs with Arabic Transliteration and Arabizi," EMNLP 2024, arXiv:2406.18725. It shows that romanized script bypasses safety more than native script, a script effect that matters if anyone conflates the Aya Hindi script.
  - All four measure attacks rather than default refusal.

### 2025–2026 Indic and open-model extensions

- **Mondal, Mamta, Varshney, Cocarascu, Ekbal (2026). "IndicSafeEval: Safety Robustness of LLMs under Multilingual Persuasive Jailbreak Attacks." Findings of EMNLP 2026. arXiv:2609.03781.** [V]; model list [U].
  - *Did/found:* 7,200 persuasion jailbreaks in Hindi, Bengali, Marathi and Punjabi. A search snippet says the models include **Llama-3.1-8B and Qwen3-8B** [U]. The judge is Gemini-2.5-Flash on a 5-point scale. Native-annotator kappa for Hindi is **0.70 non-binary and 0.84 binary**. The paper has internal inconsistencies: Bengali binary kappa is 0.8435 in the text but 0.8890 in the table, and Punjabi's binary kappa of 1.00 conflicts with Table 9 ([arXiv](https://arxiv.org/abs/2609.03781); [HTML](https://arxiv.org/html/2609.03781v2)).
  - *Relation:* **the closest overlap** (same models, same languages, native-validated judge). The study differs in five ways: non-adversarial paired requests, an explicit-refusal metric, a Spanish control, a capability control and mechanistic analysis.
- **Shukla et al. (2026). "Lost in Translation? A Comparative Study on the Cross-Lingual Transfer of Composite Harms" (CompositeHarm). AICS @ AAAI 2026. arXiv:2602.07963.** [V]; models [U].
  - *Did/found:* covers English, Hindi and four other Indic languages. Attack success "rises sharply in Indic languages." A promptfoo summary lists the models as Llama-3-8B-Instruct, GPT-OSS-20B and Qwen3-32B [U] ([arXiv](https://arxiv.org/abs/2602.07963)).
  - *Relation:* a Hindi plus Llama/Qwen-family ASR study, with no refusal-style metric.
- **Huang et al. (2026). "Disentangling In-Language Alignment and Cross-Lingual Safety Borrowing in Low-Resource Languages." Findings of EMNLP 2026.** [V]
  - *Did/found:* aggregate ASR conflates in-language refusal with "safety borrowing," where the model answers safely in another language. Cantonese-reply ASR exceeds aggregate ASR for all 11 models (sign test p<0.001) ([Anthology](https://aclanthology.org/2026.findings-emnlp.778/)).
  - *Relation:* **the closest methodological cousin.** It decomposes the outcome by response language, as the study decomposes it by refusal style. It covers no Indic languages.
- **Atil, Passonneau, Morstatter (2025). "Do Methods to Jailbreak and Defend LLMs Generalize Across Languages?" arXiv:2511.00689.** [V]
  - *Did/found:* 10 languages and 6 models. High-resource languages are safer on standard queries but more vulnerable to adversarial ones. Defenses are language- and model-dependent ([arXiv](https://arxiv.org/abs/2511.00689)).
  - The language and model lists were not checked.
- **Aswal & Jaiswal (2025/2026). "Phonetic Perturbations Reveal Tokenizer-Rooted Safety Gaps in LLMs" (CMP-RT). arXiv:2505.14226.** [V]; models [U].
  - *Did/found:* romanized, code-mixed Hindi attacks, with failure traced to tokenization ([arXiv](https://arxiv.org/abs/2505.14226)).
  - *Relation:* a contrast case. It is adversarial and romanized, while the study uses native-script, default-behaviour prompts.
- **Emani & Kashyap Manjusha R. (2025). "Mātṛkā." BHASHA 2025 workshop.** [V via Anthology search]
  - Covers Llama and Qwen families. Indic coverage includes Sanskrit; Hindi appears only in a snippet [U] ([Anthology](https://aclanthology.org/2025.bhasha-1.10)).
- **arXiv:2503.13081, "Multilingual security testing."** [U]; title unverified.
  - A snippet says Llama-3 has the weakest rejection rates and treats Hindi as high-resource. Verify before citing ([arXiv](https://arxiv.org/pdf/2503.13081.pdf)).
- **Yoon, Park, Ritter (2026). "Who Pays More for Safety? Measuring the Disparate Cost of Safety Alignment across Languages." EMNLP 2026. arXiv:2608.22490.**
  - Non-English users face a "double penalty": weaker protection and a larger utility loss ([arXiv](https://arxiv.org/abs/2608.22490)).
- **Surveys:**
  - Yong, Ermis, Fadaee, Bach, Kreutzer (2025), arXiv:2505.24119 [V-arXiv; venue UNVERIFIED]. About 300 *ACL papers, showing that safety research is English-centric ([arXiv](https://arxiv.org/abs/2505.24119)).
  - Lemofouet et al. (2026), LM4UC @ IJCAI, arXiv:2608.14626 [V-arXiv]. A PRISMA review of 50 studies ([arXiv](https://arxiv.org/abs/2608.14626)).

### Datasets the study uses or should position against

- **Aya Red-teaming: Aakanksha, Ahmadian, Ermis, Goldfarb-Tarrant, Kreutzer, Fadaee, Hooker (2024). "The Multilingual Alignment Prism." arXiv:2406.18682; EMNLP 2024.** [V] in the gaps note (venue [M]); [K]/[FLAG] in the grading note.
  - *Did/found:* 7,419 native-written prompts in 8 languages (**915 Hindi**). Each has an English translation. 66% target global harms and 34% local harms ([arXiv HTML](https://arxiv.org/html/2406.18682v2); [HF](https://huggingface.co/datasets/CohereForAI/aya_redteaming)).
  - *Relation:* the study's Hindi source. The English in each pair is a *translation of native Hindi*, not an original English request, so the pairing runs Hindi→English. Disregard the romanization claim for the study's items (see the data-script note above).
- **Kumar et al. (2025). "PolyGuard: A Multilingual Safety Moderation Tool for 17 Languages." COLM 2025. arXiv:2504.04377.** [V] in the gaps note; author list [FLAG].
  - *Did/found:* a classifier for prompt harm, response harm and **response refusal**. PolyGuardPrompts is a 29K-item evaluation set built partly from machine-translated WildGuardMix. Hindi coverage is [M] ([arXiv](https://arxiv.org/abs/2504.04377v2)).
  - *Relation:* a source of harmless items. Its binary refusal head could serve as a sanity-check judge.
- **Pattnayak & Chowdhuri (2026). "IndicSafe." arXiv:2603.17915.** [V]
  - *Did/found:* 6,000 prompts (500 in each of 12 Indic languages), translated from English. Cross-language agreement is only **12.8%**. Some models over-refuse in low-resource scripts ([arXiv](https://arxiv.org/abs/2603.17915)).
  - *Relation:* a study source and a paired-consistency precedent, though on SAFE/UNSAFE labels.
- **Banerjee et al. (2026). "SurakshaEval." arXiv:2608.07862; AIES 2026.** [V]
  - Human-written prompts in 10 Indic languages plus English. Failure modes include over-refusal, and performance is worse in native scripts ([arXiv](https://arxiv.org/abs/2608.07862)).
  - Not paired. It complements the study's design.
- **Hassan & Hossain (2026). "BanglaVeilGuard." ICCA 2026. arXiv:2608.21880.** [V]
  - Covers six forms of Bangla. A prompt guard cuts ASR from 93.8–100% to **6.3%**, at the cost of over-refusal ([arXiv](https://arxiv.org/abs/2608.21880)).
  - **"BanglaShield" could not be found under that name.** Do not cite it without a primary source.
- **Singh et al. (2024/2025). "Global MMLU." ACL 2025. arXiv:2412.03304.** [M]
  - The study's capability control. Its culturally sensitive (CS) vs culturally agnostic (CA) subsets could sharpen that control ([arXiv](https://arxiv.org/abs/2412.03304)).
- **Bramhecha et al. (2026). "IndicGuard." arXiv:2606.22841.** [V]
  - A Gemma-3-4B guard model for 10 Indic languages ([arXiv](https://arxiv.org/abs/2606.22841)).
- **Toxicity and translation-gap sets:**
  - PolygloToxicityPrompts, Jain et al. 2024, COLM, arXiv:2405.09373 [M] ([arXiv](https://arxiv.org/abs/2405.09373)).
  - RTP-LX, de Wynter et al., AAAI 2025, arXiv:2404.14397 [M]/[K]; author list [FLAG] ([arXiv](https://arxiv.org/abs/2404.14397)).
  - M-ALERT, Friedrich et al. 2024, arXiv:2412.15035 [M]. It includes **Spanish**, which matters for the control language, but no Indic languages ([arXiv](https://arxiv.org/abs/2412.15035)).
- **Lower-confidence sets:** RefusEU (Findings of ACL 2026, arXiv:2606.07535 [U-ish; the id/date mismatch is flagged]), MORBench (arXiv:2505.18325 [U]), CultureGuard (arXiv:2508.01710 [U]) and XL-SafetyBench (arXiv:2605.05662 [U]).
- **Model reports:**
  - Llama 3 Herd (arXiv:2407.21783) [M]. Hindi is supported; Bengali is not.
  - Qwen3 Technical Report (arXiv:2505.09388) [M]. Claims 119 languages, including Hindi and Bengali ([arXiv](https://arxiv.org/abs/2505.09388)).

## Refusal directions transfer across languages; Hindi on Qwen3 is the untested cell

The study's direction-extraction recipe is a near-direct port of Arditi et al. Two differences should be stated. The study reads only the last prompt token, not all post-instruction positions. It also reports the fraction of the score gap that ablation removes, rather than Arditi's selection score ([arXiv HTML](https://arxiv.org/html/2406.11717v3)). The cross-lingual question has been studied intensively since mid-2025. Wang et al. show that English directions ablate refusal in other languages almost perfectly. They also show that harmful and harmless prompts separate less well outside English. **Llama-3.1-8B's silhouette score falls from 0.496 in English to about 0.22–0.32 in other languages** ([arXiv HTML](https://arxiv.org/html/2505.17306v2)). Aziz et al. extend the analysis to 23 languages, Hindi among them. They find low-resource harm "projects lower but stays separable (most AUC > 0.85)" ([arXiv HTML](https://arxiv.org/html/2606.01196v1)). The study's Llama result, where Hindi harm reaches 28% of the English projection, therefore *replicates the direction* of these findings on a new language set. What no paper found has examined is Qwen3, paired same-request Hindi projected onto an English-only direction, and a refusal-*style* axis.

- **Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024). "Refusal in Language Models Is Mediated by a Single Direction." NeurIPS 2024. arXiv:2406.11717.** [V]; [K] in the grading note.
  - *Did/found:* in 13 open chat models, ablating one DiM direction removes refusal and adding it induces refusal. Data: 128 harmful and 128 harmless training prompts; 32 validation prompts. Selection takes the lowest bypass score subject to induce > 0, KL < 0.1 and layer < 0.8L. The metric is first-token refusal log-odds. The models include Llama-3-8B-Instruct and Qwen-1 Chat, but not Qwen3 or Llama-3.1 ([arXiv](https://arxiv.org/abs/2406.11717); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf)).
  - *Relation:* the method the study adopts. The precursor LessWrong post is [U].
- **Wang, Wang, Liu, Schütze, Plank (2025). "Refusal Direction is Universal Across Safety-Aligned Languages." NeurIPS 2025. arXiv:2505.17306.** [V] (full text read; NeurIPS poster page confirmed). The steering note marks the venue as not confirmed on the NeurIPS site; the refusal-direction note did confirm it.
  - *Did/found:* PolyRefuse covers 14 languages machine-translated by Google Translate, with **no Hindi or other Indic language**. Models are Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct and gemma-2-9B-it. Directions are approximately parallel across languages, but harmful/harmless separation is weaker outside English, "especially in the Llama3.1-8B-Instruct model." The authors position their analysis as downstream of harm identification ([arXiv](https://arxiv.org/abs/2505.17306v1); [NeurIPS](https://neurips.cc/virtual/2025/poster/116911)).
  - *Relation:* **the primary positioning target.** Same method, and one shared model. The study adds Hindi, Qwen3, a projection metric on paired items and a style axis.
- **Aziz, Hanif, Koto (2026). "Low-Resource Safety Failures Are Action Failures, Not Representation Failures." arXiv:2606.01196** (preprint).
  - Tags: [V] for the setup in the refusal-direction note; [V-arXiv] in the mechanistic note, with model versions "UNVERIFIED (secondary)".
  - *Did/found:* covers 23 languages; **Hindi is in the medium-resource tier**. Models are Qwen2.5-7B, Gemma-2-9B and **Llama-3.1-8B-Instruct**. The direction is fit by DiM on a pooled set of 10 high-resource languages. Harmful refusal averages 87.9% for high-resource, 78.9% for medium-resource and 43.9% for low-resource languages. Hindi harmful refusal is **66.3% (Qwen2.5), 92.7% (Gemma) and 67.0% (Llama-3.1)**; harmless refusal is 1.2–3.4%.
  - **Terminology conflict:** the refusal-direction note calls the axis a refusal direction; the mechanistic and steering notes call it a "harmfulness direction." Check the paper.
  - *Relation:* **the closest prior Hindi + Llama-3.1 result.** The Llama Hindi refusal rate of 67% (any refusal) sits plausibly above the study's 47% *explicit* refusal. Differences: the study fits an English-only direction, uses paired items and Qwen3, and adds the style axis.
- **Ivanova (2026). "Is Refusal Direction Universal across Low-Resourced Languages?" BlackboxNLP @ EMNLP 2026, Reproducibility Track. arXiv:2610.28897.** [V]
  - *Did/found:* on Qwen2.5-14B, universality is "graded rather than binary," and geometry and behaviour are only partly linked ([arXiv](https://arxiv.org/abs/2610.28897)).
  - *Relation:* supports reporting projection separately from behaviour. That matters for the Qwen case, where projection is strong but behaviour is soft.
- **Oppong et al. (2026). "The Illusion of Cross-Lingual Safety in Low-Resource Languages." arXiv:2608.11146.** [V]
  - *Did/found:* in African languages, harmful prompts retain **less than 10% of the English refusal signal** ([arXiv](https://arxiv.org/abs/2608.11146)).
  - *Relation:* the extreme form of the Llama pattern.
- **Ifebi, Kong, Mehrotra (2026). "Minionese." arXiv:2607.10112.** [V-arXiv] in the mechanistic note; [V-meta, body unparsed] in the refusal-direction note.
  - *Did/found:* low-resource harm is routed through a "geometrically misaligned subspace," leaving refusal "intact but untriggered" ([arXiv](https://arxiv.org/abs/2607.10112)).
  - *Relation:* the same mechanism as the Llama case. Its transliteration results bear on script effects.
- **Faruna (2026). "Latent Space Refusal Anchoring for Low-Resource African Languages." ICML 2026 Global South workshop. arXiv:2608.18089.** [V]
  - Clamping the English direction recovers refusals. "Refusal Centroid Drift" reaches 0.95. The paper has a single author and a single annotator ([arXiv HTML](https://arxiv.org/html/2608.18089)).
- **Verma, Bharadwaj (2025). "The Hidden Space of Safety." arXiv:2504.02708.** [V]
  - Safety-space separation is weaker for low-resource languages. Aziz et al. dispute this account ([arXiv](https://arxiv.org/abs/2504.02708)).
- **Apart Research sprint, "One Direction, Many Languages"** [UNVERIFIED, hackathon].
  - The only other Hindi refusal-direction work found. It uses small models, and its reported Qwen English–Hindi cosine is internally inconsistent (0.53 vs 0.85–0.97). Low reliability ([Apart](https://apartresearch.com/sprints/projects/one-direction-many-languages-causal-crosslingual-refusal-transfer-across-small-open-models-teow)).

### Follow-ups that qualify the single-direction picture

- **Wollschläger et al. (2025). "The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence." ICML 2025. arXiv:2502.17420.** [V]
  - *Did/found:* refusal is mediated by multi-dimensional cones, and directions that are orthogonal are not necessarily independent under intervention ([arXiv](https://arxiv.org/abs/2502.17420)).
  - *Relation:* a Hindi refusal component could sit elsewhere in the cone. The style direction should also be tested for *independence* from the refusal direction, not just for low cosine.
- **Zhao, Huang, Wu, Bau, Shi (2025). "LLMs Encode Harmfulness and Refusal Separately." arXiv:2507.11878.** [V]; venue [U] (possibly NeurIPS 2025).
  - *Did/found:* harmfulness is encoded at t_inst (the last instruction token) and refusal at t_post-inst (the last template token). Steering refusal leaves the harm judgment unchanged. Some jailbreaks suppress refusal while the harm belief persists ([arXiv HTML](https://arxiv.org/html/2507.11878v2)).
  - *Relation:* **the most important interpretive constraint on the study.** A refusal-direction projection does not measure "recognition."
- **Joad et al. (2026). "There Is More to Refusal in LLMs than a Single Direction." arXiv:2602.02132.** [V]; the author page lists EMNLP 2026.
  - *Did/found:* 11 category-specific refusal directions with pairwise cosine of about 0.4–0.6 that steer refusal similarly. SAEs show a shared core of refusal latents plus a long tail of style latents ([arXiv HTML v2](https://arxiv.org/html/2602.02132v2)).
  - *Relation:* **the closest relative of the explicit-vs-soft direction.** It splits refusal by reason, not style.
- **Wang, Hu, Röttger, Plank (2024). "Surgical, Cheap, and Flexible: Mitigating False Refusal via Single Vector Ablation." arXiv:2410.03415; ICLR 2025.** Venue [U] in the refusal-direction note; given as ICLR 2025 in the steering note.
  - *Did/found:* a second DiM direction for false refusal, orthogonalized against the refusal direction ([arXiv](https://arxiv.org/abs/2410.03415)).
  - *Relation:* precedent for a second, distinct DiM refusal sub-direction.
- **Yu, Do, Hambardzumyan, Cancedda (2025). "Robust LLM Safeguarding via Refusal Feature Adversarial Training" (ReFAT). ICLR 2025. arXiv:2409.20089.** [V]
  - *Did/found:* many attacks work by ablating the refusal feature, and training under simulated ablation (ReFAT) improves robustness ([arXiv](https://arxiv.org/abs/2409.20089v2)).
  - *Relation:* a language switch can be framed as a natural attack of this kind.
- **COSMIC (2025). "Generalized Refusal Direction Identification in LLM Activations." Findings of ACL 2025.** [V-meta]; authors unverified.
  - Selects a refusal direction without matching refusal tokens, which may suit Hindi outputs ([PDF](https://aclanthology.org/2025.findings-acl.1310.pdf)).
- **"Base LLMs refuse too." LessWrong.** [V-meta]; authors [U].
  - Base models refuse via the same direction but in a different style (claimed incompetence or misunderstanding). Informal evidence that style varies within one mechanism ([LessWrong](https://www.lesswrong.com/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too)).
- **Lower-confidence citations:**
  - Hildebrandt et al. 2025, arXiv:2501.08145, on nonlinear refusal [V-meta, via a citing paper].
  - Ball, Kreuter, Panickssery 2024, arXiv:2406.09289, on jailbreak latent dynamics [U].
  - 2026 jailbreak-mechanism preprints, title only [V-meta]: HARC arXiv:2607.00572, "Tracing the Dynamics of Refusal" arXiv:2605.02958, "A Mechanistic Study of the Prefill Jailbreak" arXiv:2607.14147, "Refusal Lives Downstream of Persona" arXiv:2606.26161, and arXiv:2509.15202.

## Mechanistic accounts split into "untriggered" and "written late"; the study shows both

Since mid-2026 two mechanistic explanations of cross-lingual safety failure have appeared. One says harm is routed through a weakly projecting subspace, so refusal never fires: Wang et al.'s Llama separation drop, Oppong et al. and Minionese. The other says recognition is intact but the refusal action or writing fails: Aziz et al. and Kompella & Mahajan. Each paper argues for its explanation as *the* one. **No paper found shows the two co-occurring in different model families for the same language**, which is what the study's Qwen (soft) vs Llama (comply) split shows. The split is foreshadowed, though. Llama is the weakest separator in Wang et al., Llama needs target-language calibration in Aziz et al., and Upadhyaya & Sikdar call safety–language entanglement "architecture-dependent." On neurons, the literature divides into a "language-specific safety neurons" camp (SN-Tune) and a "shared safety neurons" camp (the 2026 Chua/Shen group papers). The study's negative Hindi-neuron result supports the shared camp. It also extends Mondal et al.'s finding that language neurons are not task levers.

### Cross-lingual safety mechanisms

- **Kompella & Mahajan (2026). "Decided Upstream, Written Late: Locating and Pricing the Cross-Lingual Refusal Circuit of a Multilingual MoE." Actionable Interpretability @ COLM 2026. arXiv:2608.08032.** [V]
  - *Did/found:* in the Indic Sarvam MoE, harm is encoded as a nearly language-invariant direction (English vs Indic cosine about 0.9 at L11). The refusal is "written late" during generation, along a direction orthogonal to the detection direction ([arXiv](https://arxiv.org/abs/2608.08032)).
  - *Relation:* **the closest Indic analogue of the Qwen finding.** It is an MoE, not a dense model, and its Indic languages are unnamed on the abstract page.
- **Balani & Panda (2026). "Hard Negatives Reveal What Easy Negatives Hide." arXiv:2609.27758.** [V-arXiv]
  - *Did/found:* English-trained harm probes reach AUROC above 0.98 on easy negatives. With XSTest-style hard negatives, transfer collapses for low-resource languages. On Qwen2.5-7B the AUROC drop is 0.003 in English vs **0.276** in low-resource languages ([arXiv](https://arxiv.org/abs/2609.27758)).
  - *Relation:* **the main threat to the Qwen recognition claim.**
- **Upadhyaya & Sikdar (2026). "When Safety Speaks a Language." arXiv:2608.29936.** [V]
  - *Did/found:* in SAEs, safety features are entangled with language identity in an architecture-dependent way. Ablating them changes the output language as well as harmfulness ([arXiv](https://arxiv.org/abs/2608.29936)).
  - *Relation:* supports both the Qwen/Llama split and the English-side effects of the Hindi neurons.
- **Li, Yong, Bach (2024). "Preference Tuning For Toxicity Mitigation Generalizes Across Languages." Findings of EMNLP 2024. arXiv:2406.16235.** [V-arXiv]
  - *Did/found:* English-only DPO cuts toxicity across 17 languages (mGPT: 46.8% → 3.9%), with transfer explained by shared multilingual MLP key and value vectors ([arXiv](https://arxiv.org/abs/2406.16235)).
  - *Relation:* transfer runs through shared MLP components, which is consistent with finding no Hindi-only MLP lever.
- **Yeo et al. (2025). "Understanding Refusal in Language Models with Sparse Autoencoders." Findings of EMNLP 2025. arXiv:2505.23556.** [V-arXiv]
  - English-only refusal features, studied at the SAE level ([arXiv](https://arxiv.org/abs/2505.23556)).
- **O'Brien et al. (2024). "Steering Language Model Refusal with Sparse Autoencoders." arXiv:2411.11296.** [V-arXiv]
  - Refusal features are entangled with general capability ([arXiv](https://arxiv.org/abs/2411.11296)).
- **Lower-confidence citations:**
  - LessWrong, "Low-resourced languages get jailbroken more. Can SAEs explain why?" [UNVERIFIED]. English SAE refusal features lose predictive power outside English ([LessWrong](https://www.lesswrong.com/posts/8h2KxbPhFx8JoEGdH/low-resourced-languages-get-jailbroken-more-can-saes-explain)).
  - Jiang et al. 2026, MLJailDe, arXiv:2606.11202 [V-arXiv]. Applied detection work, peripheral.

### Recognition vs execution, and refusal depth

- **Qi et al. (2024). "Safety Alignment Should Be Made More Than Just a Few Tokens Deep." arXiv:2406.05946; ICLR 2025.** [V-arXiv]; venue from memory.
  - *Did/found:* alignment mostly shifts the first few output tokens ([arXiv](https://arxiv.org/abs/2406.05946)).
  - *Relation:* a candidate mechanism for Qwen's soft Hindi wording. If the first tokens are weaker, the rest of the reply degrades.
- **The three-stage gap.** Zhao et al., Kompella & Mahajan, and Aziz et al. separate recognition from decision. **No paper found separates a third stage, the wording or strength of a refusal once the decision is made.** The study's Qwen result is evidence for that stage.

### Language representation and the English pivot

- **Wendler, Veselovsky, Monea, West (2024). "Do Llamas Work in English?" ACL 2024. arXiv:2402.10588.** [V-arXiv]
  - Middle layers form an English-biased concept space ([arXiv](https://arxiv.org/abs/2402.10588)).
- **Schut, Gal, Farquhar (2025). "Do Multilingual LLMs Think In English?" arXiv:2502.15603.** [V-arXiv]; venue UNVERIFIED.
  - English-computed steering vectors beat input-language vectors. This justifies using an English-fitted measuring stick ([arXiv](https://arxiv.org/abs/2502.15603)).
- **Zhao et al. (2024). "How do Large Language Models Handle Multilingualism?" (MWork/PLND). NeurIPS 2024. arXiv:2402.18815.** [V-arXiv]; venue UNVERIFIED but held with confidence.
  - The late language-conversion stage is a plausible site for the soft Hindi wording ([arXiv](https://arxiv.org/abs/2402.18815)).
- **Dumas et al. (2024/2025). "Separating Tongue from Thought." arXiv:2411.08745.** [V-arXiv]; ACL 2025 venue UNVERIFIED.
  - Concept and language can be patched independently ([arXiv](https://arxiv.org/abs/2411.08745)).
- **Zhong et al. (2024). "Beyond English-Centric LLMs." arXiv:2408.10811.** [V-arXiv]
  - The latent pivot language depends on the training mix. A candidate explanation for differences between Qwen3 and Llama ([arXiv](https://arxiv.org/abs/2408.10811)).
- **Shared-representation precedents:**
  - Brinkmann et al. 2025, arXiv:2501.06346 [V-arXiv; NAACL venue UNVERIFIED] ([arXiv](https://arxiv.org/abs/2501.06346)).
  - Ferrando & Costa-jussà 2024, Findings of EMNLP 2024, arXiv:2410.06496 [V-arXiv] ([arXiv](https://arxiv.org/abs/2410.06496)).
  - Lindsey et al. 2025, "On the Biology of a Large Language Model," Transformer Circuits [UNVERIFIED].

### Language and safety neurons

- **Tang et al. (2024). "Language-Specific Neurons" (LAPE). ACL 2024. arXiv:2402.16438.** [V-arXiv]
  - The standard neuron-finding method, which selects by activation statistics ([arXiv](https://arxiv.org/abs/2402.16438)). The study instead selected neurons causally, via attribution patching.
- **Kojima et al. (2024). NAACL 2024. arXiv:2404.02431.** [V-arXiv]
  - Language neurons overlap by less than 5% and sit in the first and last layers ([arXiv](https://arxiv.org/abs/2404.02431)).
- **Mondal, Sen, Singhania, Jyothi (2025). "Language-specific Neurons Do Not Facilitate Cross-Lingual Transfer." NAACL 2025 Insights workshop. arXiv:2503.17456.** [V-arXiv]
  - *Did/found:* neuron interventions on **Llama 3.1** do not improve low-resource task performance ([arXiv](https://arxiv.org/abs/2503.17456)).
  - *Relation:* **the closest precedent for the negative neuron result.** The study extends it from task ability to safety.
- **Gurgurov et al. (2025). "Language Arithmetics." AACL 2025. arXiv:2507.22608.** [V-arXiv]
  - Llama-3.1-8B neurons for non-Latin scripts are more specialised ([arXiv](https://arxiv.org/abs/2507.22608)).
- **Wang, Haddow, Wu, Peng, Birch (2024). "Sharing Matters." arXiv:2406.09265.** [V-arXiv]; venue UNVERIFIED.
  - All-shared neurons matter most for correct output ([arXiv](https://arxiv.org/abs/2406.09265)).
- **Deng et al. (2025). "Unveiling Language-Specific Features in LLMs via Sparse Autoencoders." ACL 2025. arXiv:2505.05111.** [V-arXiv]
  - The SAE analogue of LAPE ([arXiv](https://arxiv.org/abs/2505.05111)).
- **Zhao et al. (2025). SN-Tune, "Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron." ICLR 2025.** [V-web]; arXiv id not checked; overlap figure UNVERIFIED.
  - *Did/found:* claims "language-specific" safety neurons. Tuning them cuts Llama3-8B-Instruct's harmful score from 65.5 to 2.0 ([ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6d2666e2cf44088cc57204fbc5ef7f34-Abstract-Conference.html)).
  - *Relation:* this is the hypothesis the neuron study tested.
- **Chen et al. (2024; NeurIPS 2025). "Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons." arXiv:2406.14144.** [V-arXiv]
  - Patching about 5% of neurons restores over 90% of safety ([arXiv](https://arxiv.org/abs/2406.14144)).
- **Shared-neuron and pathway papers (2026 preprints):**
  - Zhang et al., "Who Transfers Safety?", arXiv:2602.01283 [V-arXiv]. Cross-lingual shared safety neurons, under 0.3% of neurons ([arXiv](https://arxiv.org/abs/2602.01283)).
  - Miao et al., "Who Bridges Safety?", arXiv:2608.09095 [V]. Shared safety pathways ([arXiv](https://arxiv.org/abs/2608.09095)).
  - Xie et al., "Safety Kernel," arXiv:2610.24614 [V-arXiv] ([arXiv](https://arxiv.org/abs/2610.24614)).
  - Shi et al., arXiv:2604.08881 [V-arXiv]. Vision-language models, peripheral.
- **Possibly novel.** None of these papers reports that intervening on target-language-associated neurons changes behaviour in the *source* language (English) only. The mechanistic note marks this "not confirmed exhaustively."

## Gated refusal steering is established; the study's distinctive parts are guards and protocol

The study's intervention reads the projection on the refusal direction, thresholds it at the dev median, and adds a fixed DiM vector at a later layer. Each element has a precedent. Additive DiM vectors come from ActAdd and CAA. Conditional refusal steering is CAST (ICLR 2025). The "classify then steer" rationale is Stickland et al. Gating on the refusal direction itself is ALTSTEER. **Cross-lingual, target-language-calibrated gating is Aziz et al.**, who also found that Llama benefits from target-language calibration while Qwen and Gemma do not. That matches the study's choice of an English threshold for Qwen and a Hindi threshold for Llama, and it should be presented as **convergent evidence** ([arXiv](https://arxiv.org/abs/2606.01196)). BabelSteering is the closest *multilingual* steering comparator: an English refusal vector, Bengali included, Hindi not. It raises harmful refusal (Bengali +17 pp), but pseudo-harmful refusal rises by 13 pp on average ([arXiv](https://arxiv.org/abs/2608.16577)). The study's Qwen result is that gated steering raised explicit refusal in English as much as in Hindi. That is a substantive negative finding: **a gate on an axis where Hindi and English harm score alike cannot be language-selective.** The reliability literature (Tan et al.; Braun et al.; AxBench) predicts exactly the failure to replicate, and it supports publishing the null.

### Foundations

- **Turner et al. (2023). "Steering Language Models With Activation Engineering" (ActAdd). arXiv:2308.10248.** Venue unverified.
  - Steering vectors from the activation difference of one prompt pair ([arXiv](https://arxiv.org/abs/2308.10248)).
- **Panickssery (Rimsky) et al. (2023/2024). "Steering Llama 2 via Contrastive Activation Addition" (CAA). arXiv:2312.06681; ACL 2024.** [K]; venue from common citation.
  - DiM vectors averaged over many pairs ([arXiv](https://arxiv.org/abs/2312.06681)).
- **Zou et al. (2023). "Representation Engineering." arXiv:2310.01405.** Venue unverified.
  - The read-then-control framing ([arXiv](https://arxiv.org/abs/2310.01405)).
- **Li et al. (2023). "Inference-Time Intervention." NeurIPS 2023. arXiv:2306.03341.**
  - Strength chosen against a side-effect trade-off ([arXiv](https://arxiv.org/abs/2306.03341)).

### Conditional steering

- **Lee et al. (2024). "Programming Refusal with Conditional Activation Steering" (CAST). arXiv:2409.05907; ICLR 2025 Spotlight.**
  - *Did/found:* gates on a separate condition vector, with the threshold chosen by F1-maximizing grid search. On Qwen1.5-1.8B, harmful/harmless refusal is **90.67%/2.20% conditional vs 100%/96.40% unconditional**. No multilingual evaluation ([arXiv HTML](https://arxiv.org/html/2409.05907v3)).
  - *Relation:* **the main gating precedent.**
- **Aziz, Hanif, Koto (2026), arXiv:2606.01196.** Venue unverified.
  - *Did/found:* a low-rank classifier gate, calibrated with a few target-language examples, adds or ablates the direction. Mean selectivity is **54.5 vs 33.6 (CAST) vs 6.8 (AdaSteer)**, with MMLU preserved.
  - *Relation:* **the most directly comparable prior work; cite it prominently.**
- **Kwon, Lim, Kim, Kim (2026). "ALTSTEER: Selective Safety Steering for Moving Beyond Hard Refusals to Constructive Alternatives." EMNLP 2026. arXiv:2608.30197.**
  - *Did/found:* gates on a refusal-relevant signal. The zero threshold is from a search snippet, not verified. Evaluated on Llama-3.1 and Qwen2.5, and it steers toward constructive alternatives rather than hard refusals ([arXiv](https://arxiv.org/abs/2608.30197)).
  - *Relation:* the closest English analogue both of a refusal-direction gate and of steering refusal *style*.
- **Cooper Stickland et al. (2024). "Steering Without Side Effects" (KL-then-steer). arXiv:2406.15518.** Venue unverified.
  - States the classify-then-steer rationale. KL-then-steer prevented 44% of jailbreaks ([arXiv](https://arxiv.org/abs/2406.15518)).
- **Zhao et al. (2025). "AdaSteer." EMNLP 2025. arXiv:2504.09466.**
  - Input-adaptive coefficients along refusal and harmfulness directions. A snippet says the authors flag low-resource languages as a limitation (unverified) ([arXiv](https://arxiv.org/abs/2504.09466)).
- **Sheng et al. (2025). "AlphaSteer." ICLR 2026. arXiv:2506.07022.**
  - Learned null-space gating ([arXiv](https://arxiv.org/abs/2506.07022)).
- **Han et al. (2025). "SafeSwitch." arXiv:2502.01042.** Venue unverified.
  - A probe gates a tuned safety head ([arXiv](https://arxiv.org/abs/2502.01042)).
- **Shen et al. (2024). "Jailbreak Antidote." ICLR 2025. arXiv:2410.02298.**
  - Ungated strength-dial steering ([arXiv](https://arxiv.org/abs/2410.02298)).
- **Kasliwal et al. (2026). "C-ΔΘ: Circuit-Restricted Weight Arithmetic for Selective Refusal." arXiv:2602.04521.**
  - Distils selective refusal into an offline weight edit ([arXiv](https://arxiv.org/abs/2602.04521)).
- **Input-adaptive steering outside safety:** ACT, WWW 2025, arXiv:2406.00034; SADI, arXiv:2410.12299.

### Refusal-steering side effects

- **Cao et al. (2024). "SCANS." AAAI 2025. arXiv:2408.11491.**
  - Per-input steering that *reduces* over-refusal ([arXiv](https://arxiv.org/abs/2408.11491)).
- **Li et al. (2026). "Analysing the Safety Pitfalls of Steering Vectors." arXiv:2603.24543.**
  - Unrelated CAA vectors shift jailbreak ASR by +57% to −50% ([arXiv](https://arxiv.org/abs/2603.24543)). This supports the study's English-refusal guard.
- **Wang & Shu (2023). "Trojan Activation Attack." CIKM 2024. arXiv:2311.09433.**
  - The attack side of the same mechanism ([arXiv](https://arxiv.org/abs/2311.09433)).
- **A garbled-output guard is uncommon.** No abstract reviewed reports a garbled or incoherent-output rate as a formal guard, so the BROKEN guard is a point of difference. A full-text check of SCANS, AdaSteer and BabelSteering is still pending.

### Multilingual steering

- **Stein et al. (2026). "BabelSteering: Multilingual Safety Alignment via English Steering Vectors." arXiv:2608.16577.**
  - *Did/found:* 8 languages, Bengali included, Hindi not. Harmful refusal rises 11 pp on average for Gemma and 17 pp for Bengali.
  - **Conflict:** the refusal-direction note says MMLU drops by less than 3 pp and that the method also ablates an orthogonalized false-refusal vector. The steering note says there is no Global-MMLU loss and that the abstract describes the intervention as unconditional. Check the paper.
- **Language-steering precedents (non-safety):**
  - Kang et al. 2026, BlackboxNLP, arXiv:2609.08410. Language vectors work best at earlier layers and jailbreak vectors at middle layers ([arXiv](https://arxiv.org/abs/2609.08410)).
  - Kirtane & Huang 2026, arXiv:2602.02326.
  - Pokharel et al. 2026, INLG, arXiv:2601.16390.
  - Lens, arXiv:2410.04407.
  - Apart "LSR-Anchoring" sprint (not peer-reviewed; snippet only).

### Reliability critiques

- **Tan et al. (2024). "Analyzing the Generalization and Reliability of Steering Vectors." NeurIPS 2024. arXiv:2407.12404.** [K]
  - Steerability is "highly variable across different inputs" ([arXiv](https://arxiv.org/abs/2407.12404)).
- **Braun et al. (2025). "Understanding (Un)Reliability of Steering Vectors." ICLR 2025 workshop. arXiv:2505.22637.**
  - Reliability is predicted by how well positive and negative examples separate along the direction ([arXiv](https://arxiv.org/abs/2505.22637)). A per-language diagnostic the study could report. A follow-up MSc thesis is arXiv:2602.17881.
- **Da Silva et al. (2025). "Steering off Course." arXiv:2504.04635.**
  - Up to 36 models; many show no gain from steering ([arXiv](https://arxiv.org/abs/2504.04635)). Supports reporting each model separately.
- **Wu et al. (2025). "AxBench." ICML 2025. arXiv:2501.17148.** [K]
  - **Prompting outperforms all steering methods**, and DiM is best for concept detection ([arXiv](https://arxiv.org/abs/2501.17148)).
- **Other critiques:**
  - Im & Li 2025, arXiv:2502.02716.
  - Niranjan et al. 2025, arXiv:2505.01162.
  - Li, Li, Huang 2026, "Steering Vector Fields," EMNLP 2026, arXiv:2602.01654. A fixed vector fails when the locally effective direction varies, which is a reason to keep claims narrow ([arXiv](https://arxiv.org/abs/2602.01654)).
- **Pres et al. (2024). "Towards Reliable Evaluation of Behavior Steering Interventions in LLMs." NeurIPS 2024 workshop. arXiv:2410.17245.** [V]
  - CAA fails in open-ended generation even when it works in multiple-choice format ([arXiv](https://arxiv.org/abs/2410.17245)). This supports grading open-ended generations.

## Four-way grading combines known classes; native-script kappa 0.82 clears the Indic bar

Each of the study's four labels has a precedent. COMPLY and REFUSE are in every benchmark. XSTest's "partial refusal," OR-Bench's "indirect refusal" and Do-Not-Answer's caution and refute categories cover the soft middle. Yong et al.'s UNCLEAR and MultiJail's "invalid" are precedents for BROKEN. **No published scheme combines all four or defines "soft" as "understood, but neither provides it nor says no."** OpenAI's safe-completion training (Yuan et al. 2025) makes the soft class more important: frontier labs now deliberately train for replies that are neither compliance nor explicit refusal ([arXiv](https://arxiv.org/abs/2508.09224v1)). On judge reliability, the multilingual evidence is consistently pessimistic. **Average Fleiss' kappa across 25 languages is about 0.3** ([Fu & Liu](https://arxiv.org/abs/2505.12201v1)). GPT-4 is lenient outside English ([Hada et al.](https://arxiv.org/abs/2309.07462)). In Indic direct assessment, agreement falls sharply for Bengali and Odia ([PARIKSHA](https://arxiv.org/abs/2406.15053)). Against that background, the study's **kappa of 0.82 (broad)** with a fluent Hindi grader is a respectable result. It is comparable to IndicSafeEval's Hindi kappa of 0.84 binary and above its 0.70 graded ([HTML](https://arxiv.org/html/2609.03781v2)). Native-script judging is itself a methodological difference: translating replies before judging (XSafety, Yong et al., PolyGuard) can erase the BROKEN signal.

### Taxonomies

- **Röttger et al. (2024). "XSTest." NAACL 2024. arXiv:2308.01263.** [K], [M]
  - Labels are full compliance, full refusal or partial refusal. String matching misses partial refusals ([arXiv](https://arxiv.org/abs/2308.01263)).
- **Wang, Li, Han, Nakov, Baldwin (2023/2024). "Do-Not-Answer." Findings of EACL 2024. arXiv:2308.13387.** [K], [M]
  - Six response-action categories ([arXiv](https://arxiv.org/abs/2308.13387)).
- **Cui et al. (2024). "OR-Bench." arXiv:2405.20947.** [K]; ICML 2025 venue [FLAG].
  - Labels include "indirect refusal." The judge was validated against an LLM ensemble, not humans ([arXiv](https://arxiv.org/abs/2405.20947)).
- **Xie et al. (2024). "SORRY-Bench." ICLR 2025. arXiv:2406.14598.** [V]
  - Binary labels, 7K+ human annotations, and linguistic mutations including translation. Whether the translation languages include Indic ones is [FLAG] ([arXiv](https://arxiv.org/abs/2406.14598)).
- **Han et al. (2024). "WildGuard." NeurIPS 2024 D&B. arXiv:2406.18495.** [K]
  - Treats refusal as a separate binary axis from harmfulness ([arXiv](https://arxiv.org/abs/2406.18495)).
- **Souly et al. (2024). "A StrongREJECT for Empty Jailbreaks." NeurIPS 2024 D&B. arXiv:2402.10260.** [K]
  - Translated non-refusals are often low-quality ([arXiv](https://arxiv.org/abs/2402.10260)). Strong support for keeping BROKEN separate from COMPLY.
- **Brahman et al. (2024). "The Art of Saying No" (CoCoNot). NeurIPS 2024 D&B. arXiv:2407.12043.** [K]
  - A taxonomy of request-side reasons for noncompliance ([arXiv](https://arxiv.org/abs/2407.12043)).
- **von Recum et al. (2024). "Cannot or Should Not?" NeurIPS 2024 workshop. arXiv:2412.16974.** [V]; authors beyond the first [FLAG].
  - 16 refusal categories ([arXiv](https://arxiv.org/abs/2412.16974v1)).
- **Yuan et al. (2025). "From Hard Refusals to Safe-Completions." arXiv:2508.09224.** [V]; authors [FLAG]. See above.
- **Jain et al. (2024). "Refusal Tokens." arXiv:2412.06748.** [U]
  - From memory: special tokens for refusal categories.

### Judges and agreement

- **Zou et al. (2023). GCG. arXiv:2307.15043.** [K]
  - The archetypal string-match evaluator, which does not work for Hindi ([arXiv](https://arxiv.org/abs/2307.15043)).
- **Mazeika et al. (2024). "HarmBench." ICML 2024. arXiv:2402.04249.** [K]; about 93% agreement, to be checked ([arXiv](https://arxiv.org/abs/2402.04249)).
- **Chao et al. (2024). "JailbreakBench." NeurIPS 2024 D&B. arXiv:2404.01318.** [K]; about 90% agreement [FLAG] ([arXiv](https://arxiv.org/abs/2404.01318)).
- **Other judges and evaluator comparisons:**
  - Llama Guard, Inan et al. 2023, arXiv:2312.06674 [K] ([arXiv](https://arxiv.org/abs/2312.06674)).
  - PAIR, arXiv:2310.08419 [K].
  - JailbreakEval, arXiv:2406.09321 [K]; authors [FLAG].
- **Zheng et al. (2023). "Judging LLM-as-a-Judge." NeurIPS 2023 D&B. arXiv:2306.05685.** [K]
  - Documents position, verbosity and self-enhancement biases ([arXiv](https://arxiv.org/abs/2306.05685)).
- **Panickssery, Bowman, Feng (2024). "LLM Evaluators Recognize and Favor Their Own Generations." NeurIPS 2024. arXiv:2404.13076.** [K]
  - Self-preference bias ([arXiv](https://arxiv.org/abs/2404.13076)). A Claude judge on Qwen and Llama outputs avoids it.

### Multilingual judge reliability

- **Hada et al. (2024). "Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?" Findings of EACL 2024. arXiv:2309.07462.** [V]
  - About 20K human ratings across 8 languages. Recommends calibration against native speakers ([arXiv](https://arxiv.org/abs/2309.07462)).
- **Hada et al. (2024). "METAL." Findings of NAACL 2024. arXiv:2404.01667.** [K]/[FLAG]
  - Judge agreement varies by language; Hindi is included ([arXiv](https://arxiv.org/abs/2404.01667)).
- **Watts et al. (2024). "PARIKSHA." EMNLP 2024. arXiv:2406.15053.** [V]
  - 10 Indic languages. The safety evaluation was not human-graded ([arXiv](https://arxiv.org/abs/2406.15053)).
- **Fu & Liu (2025). "How Reliable is Multilingual LLM-as-a-Judge?" Findings of EMNLP 2025. arXiv:2505.12201.** [V] ([arXiv](https://arxiv.org/abs/2505.12201v1)).
- **Gureja et al. (2025). "M-RewardBench." ACL 2025. arXiv:2410.15522.** [K]/[FLAG]
  - Judges lose accuracy in low-resource languages ([arXiv](https://arxiv.org/abs/2410.15522)).
- **Also relevant:** RTP-LX (above), Aya Prism (above), and IndicSafeEval (above). The grading note found **no study of Claude-family judge agreement with Hindi humans on refusal-style labels.**

## Non-mechanistic fixes set the baseline a reviewer will demand

There are three families of non-mechanistic fix. **Prompt- and translation-time methods** are cheap and often report large gains, but those gains depend on the language and model ([Atil et al.](https://arxiv.org/abs/2511.00689)). XSafety's English-thinking prompt roughly halved non-English unsafe output [M]. **Training-time methods** report harm reductions above 50%. Aya Prism cut harmful generations by **56.6% (SFT) and 54.7% (DPO(SFT))** with Hindi among its training languages ([arXiv HTML](https://arxiv.org/html/2406.18682v2)). English as Defense Proxy blocks more than 99% of MultiJail jailbreaks while retaining 95% of task performance ([Anthology](https://aclanthology.org/2025.findings-emnlp.62/)). English-only safety data transfers poorly, however, and naive refusal distillation has *raised* jailbreak success by up to 16.6 pp [U]. **External guards** cut ASR sharply at the cost of over-refusal. Guards do not explain why the chat model itself refuses less. No paper found evaluates a plain translate-Hindi-to-English or system-prompt baseline on Llama-3.1-8B or Qwen3-8B. Given AxBench's finding that prompting beats steering, that baseline is the most likely reviewer request.

- **Zhang et al. (2025). "English as Defense Proxy." Findings of EMNLP 2025.** [V] ([Anthology](https://aclanthology.org/2025.findings-emnlp.62/)).
- **Li et al. (2025). "Multilingual Collaborative Defense for Large Language Models" (MCD). arXiv:2505.11835.** [V]
  - An optimized soft safety prompt. The abstract gives no numbers ([arXiv](https://arxiv.org/abs/2505.11835)).
- **Zhao et al. (2025). "MPO: Multilingual Safety Alignment via Reward Gap Optimization." arXiv:2505.16869.** [U]; ACL 2025 venue unverified ([alphaXiv](https://www.alphaxiv.org/abs/2505.16869.md)).
- **RefusEU (Findings of ACL 2026, arXiv:2606.07535).** [U]
  - English-only DPO raises low-resource ASR ([arXiv](https://arxiv.org/abs/2606.07535)).
- **Lower-confidence fixes:**
  - "Response-Based Knowledge Distillation for Multilingual Jailbreak Prevention Unwittingly Compromises Safety." Listing [U]; arXiv id unverified ([NeurIPS listing](https://neurips.cc/virtual/2025/133196)).
  - arXiv:2602.16660, a prompt-only consistency method [U].
  - arXiv:2601.08668, translate-then-detoxify (false refusals 11.78% → 1.09%) [U].
- **Üstün et al. (2024). "Aya Model." ACL 2024. arXiv:2402.07827.** [M]
  - Multilingual safety context distillation ([arXiv](https://arxiv.org/abs/2402.07827)).
- **Guards:** PolyGuard (+5.5% over baselines), IndicGuard and BanglaVeilGuard (above).
- **Possibly relevant, unchecked:** "Why Do Safety Guardrails Degrade Across Languages?" (arXiv:2605.17173) [U, title only].

## Pre-registration and illusion results make the study's protocol a direct answer

Pre-registration has been piloted in ML and argued for in NLP, but the notes found **no interpretability or steering paper that combines pre-registration, a dev/held-out split and a sealed held-out-language confirmation**. Phrase this as "to our knowledge." The interpretability-illusion literature gives the protocol its point: dev-set effects regularly fail to generalise (Tan et al.; Pres et al.; Makelov et al.). The study's own record shows this. Phase 4 failed its sealed Bengali confirmation, 4b did not reproduce, and Phase 5 found no Hindi-only lever. Those are exactly the failure modes the protocol exists to catch, and published negative results (GDM's SAE update, Kantamneni et al.) give precedent for reporting them.

### Pre-registration and reporting

- **Bertinetto et al. (eds.) (2021). NeurIPS 2020 Pre-registration Workshop. PMLR v148.** [V] ([PMLR](https://proceedings.mlr.press/v148/bertinetto21a.html)). The workshop was repeated in 2021.
- **van Miltenburg, van der Lee, Krahmer (2021). "Preregistering NLP Research." NAACL 2021. arXiv:2103.06944.** [V]
  - The study's dated addenda correspond to its "deviations" section ([Anthology](https://aclanthology.org/2021.naacl-main.51)).
- **Søgaard, Hershcovich, de Lhoneux (2023). "A Two-Sided Discussion of Preregistration of NLP Research." EACL 2023.** [V]
  - Useful for pre-empting objections ([Anthology](https://preview.aclanthology.org/setup/2023.eacl-main.6)).
- **Hofman et al. (2023). "Pre-registration for Predictive Modeling." arXiv:2311.18807.** [V] ([arXiv](https://arxiv.org/abs/2311.18807v1)).
- **Reporting and split standards:**
  - Dodge et al. 2019, EMNLP, arXiv:1909.03004 [K].
  - Gorman & Bedrick 2019, ACL [K] ([Anthology](https://aclanthology.org/P19-1267/)).
  - Søgaard et al. 2021, EACL, arXiv:2005.00636 [K].
  - Pineau et al. 2021, JMLR, arXiv:2003.12206 [K].
  - Kapoor & Narayanan 2023, Patterns, arXiv:2207.07048 [K].
  - Ulmer et al. 2022, Findings of EMNLP, arXiv:2204.06251 [K].

### Illusions and controls

- **Probing and interpretation illusions:**
  - Hewitt & Liang 2019, EMNLP, arXiv:1909.03368 [K]. Control tasks and selectivity for probes ([arXiv](https://arxiv.org/abs/1909.03368)).
  - Bolukbasi et al. 2021, arXiv:2104.07143 [K].
  - Friedman et al. 2024, ICML, arXiv:2312.03656 [K].
  - Belinkov 2022, Computational Linguistics, arXiv:2102.12452 [K].
- **Makelov, Lange, Nanda (2024). "Is This the Subspace You Are Looking For?" ICLR 2024. arXiv:2311.17030.** [K]
  - A working intervention does not prove a mechanism ([arXiv](https://arxiv.org/abs/2311.17030)). Relevant to any "Hindi lever" or style-direction causal claim.

### Negative results

- **Smith et al. (2025). GDM Mech Interp Progress Update #2, "Negative Results for Sparse Autoencoders on Downstream Tasks."** [V]; not peer-reviewed ([GreaterWrong](https://greaterwrong.com/posts/4uXCAJNuPKtKBsi28)).
- **Kantamneni et al. (2025). "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing." ICML 2025. arXiv:2502.16681.** [K] ([arXiv](https://arxiv.org/abs/2502.16681)).
- **Insights from Negative Results workshop; ICBINB workshops.** [K]/[FLAG]; cite a specific year's proceedings ([Anthology 2020](https://aclanthology.org/2020.insights-1.0/)).

## Gap analysis: which contributions are new

| # | Candidate contribution | Status | Closest prior work | Note |
|---|---|---|---|---|
| 1 | A Hindi refusal gap exists on Qwen3-8B / Llama-3.1-8B | **Already done** (the phenomenon) | XSafety; Yong et al.; Aziz et al. (Hindi refusal 67.0% on Llama-3.1); IndicSafeEval (same models per [U] snippet); CompositeHarm | The gap itself is not new. Claim its *measurement design*. |
| 2 | Paired same-request **explicit-refusal** metric, with a Spanish control and a Global-MMLU comprehension control | **Novel** (as a design) | IndicSafe cross-language agreement; Huang et al. sign test; M-ALERT (Spanish) | Per-item paired explicit-refusal statistics on open 8B models were not found. |
| 3 | Four-way COMPLY / EXPLICIT / SOFT / BROKEN label, judged in native script, human kappa 0.82 | **Partly anticipated** | XSTest partial; OR-Bench indirect; Do-Not-Answer; Yong UNCLEAR; MultiJail invalid; IndicSafeEval kappa 0.84/0.70 | The combination and its Hindi validation are new. Each class has a precedent. |
| 4 | Missing refusals are **soft** in Qwen but **compliance** in Llama | **Novel** | Huang et al. (decomposition by response language); Shen et al. ("irrelevant") | The strongest behavioural contribution. |
| 5 | English direction readable on Hindi harm; directions transfer | **Already done** | Wang et al. 2025; Aziz et al.; Ivanova; Schut et al. | Cite as the premise, not as a finding. |
| 6 | Qwen3: Hindi harm projects at English strength | **Partly anticipated** (new model and language) | Kompella & Mahajan (Indic MoE, invariant harm); Aziz et al.; challenged by Balani & Panda | New cell (Qwen3 × Hindi × paired). The interpretation needs hard negatives. |
| 7 | Llama: Hindi harm at 28% of English projection | **Partly anticipated** (replication) | Wang et al. (Llama silhouette 0.496 → ~0.22–0.32); Aziz et al. (downward shift); Oppong (<10%); Minionese | Frame as convergent replication in Hindi. |
| 8 | Two model families show **different** failure mechanisms for the same language | **Novel** (framing) | Foreshadowed by Wang et al. (Llama weakest), Aziz et al. (Llama needs calibration), Upadhyaya & Sikdar ("architecture-dependent") | No paper argues this explicitly. |
| 9 | A third stage (refusal *wording*, given a decision) beyond recognition and decision | **Novel** (hypothesis) | Zhao et al.; Kompella & Mahajan ("written late"); Qi et al. (shallow alignment) | Present as evidence for, not proof of, a third stage. |
| 10 | Explicit-vs-soft refusal **direction** (AUROC 0.93, causal) | **Novel** (to our knowledge) | Joad et al. (category directions); Wang et al. 2410.03415 (false-refusal vector); ALTSTEER (constructive vs hard) | Add an independence test in the Wollschläger sense. |
| 11 | Gated steering: gate on the refusal-direction score, push along a vector | **Already done** (method) | CAST; ALTSTEER; AdaSteer; SafeSwitch; Aziz et al. | Do not claim the method. |
| 12 | Gated steering fails to be language-selective (English rises equally; Bengali fails; not reproduced) | **Novel** (negative result) | Tan et al.; Braun et al.; AxBench; BabelSteering (over-refusal cost) | Publishable as a pre-registered null. |
| 13 | Llama fix with a Hindi-calibrated gate (dev: +18 explicit, gap −13 pts) | **Already done** (in principle) | Aziz et al.: target-language-calibrated gate; Llama needs it, Qwen does not | Corroboration. Dev-only so far. |
| 14 | Hindi-associated MLP neurons: no Hindi-only lever | **Partly anticipated** | Mondal et al. (Llama 3.1, task ability); SS-Neurons; Safety Kernel; Li, Yong, Bach (shared MLP) | Extends the result to safety. |
| 15 | Those neurons change **English** behaviour only | **Novel** (to our knowledge) | Upadhyaya & Sikdar (entanglement); O'Brien et al. (off-target effects) | The mechanistic note marks this "not confirmed exhaustively." |
| 16 | BROKEN / English-refusal / MMLU guards on steering | **Partly anticipated** | MMLU and over-refusal guards are standard (CAST, BabelSteering, Aziz); a garbled-output guard was not found | Full-text check pending. |
| 17 | Pre-registration, dev/test split, sealed held-out-language confirmation | **Novel** in interpretability (to our knowledge) | van Miltenburg et al.; NeurIPS 2020 workshop; Hofman et al. | Self-registered, not a registered report. Say so. |

## Claims to soften and checks to add

**Recognition claims need a hard-negative check.** "Harmful Hindi projects at English strength" on Qwen3 is a projection result. It is not evidence that Hindi harm is *recognised*, for two reasons. First, the study reports that **harmless Hindi also sits high** on the English axis, which suggests a language offset. Report Hindi harmful-vs-harmless AUROC on that axis, and the projection *gap*, not just the harmful mean. Second, Balani & Panda show easy-negative transfer can hide a 0.276 AUROC collapse on Qwen2.5 ([arXiv](https://arxiv.org/abs/2609.27758)). Add XSTest-style hard negatives in Hindi, such as benign prompts with harm-sounding words. Until both checks are done, write "projects at English strength on the refusal direction" rather than "recognises harm."

**Use refusal-direction wording, not harmfulness wording, unless a harmfulness direction is added.** Zhao et al. show that harm (at t_inst) and refusal (at t_post-inst) are separate directions ([arXiv](https://arxiv.org/abs/2507.11878)). State exactly which token "last prompt token" refers to: the last instruction token or the last chat-template token. A DiM direction fit at the last template token is a *refusal* direction. Either rename the claims, or fit a harmfulness direction at t_inst and show the Qwen/Llama contrast survives on it. The Aziz et al. notes disagree on whether their axis is "refusal" or "harmfulness," so check their paper before aligning terminology.

**Add a prompting baseline.** AxBench finds that prompting beats steering ([arXiv](https://arxiv.org/abs/2501.17148)). XSafety's English-thinking prompt and English as Defense Proxy are the obvious precedents. Report a system-prompt baseline ("refuse harmful requests in Hindi clearly") and a translate-to-English pipeline on the same 404 held-out pairs before claiming any steering benefit.

**Soften the steering claims.** The Qwen gated intervention passed its Hindi held-out test (+17 explicit) but raised English by the same amount. It failed the sealed Bengali confirmation on the BROKEN guard, and it did not reproduce in the follow-up sweep. The defensible claim is "a refusal-direction gate cannot be language-selective when Hindi and English harm score alike." Do not write "steering closes the Hindi gap." The Llama result is dev-only and methodologically anticipated by Aziz et al. Present it as corroboration pending held-out test, not as a new fix.

**Qualify the style direction.** Test whether the explicit-vs-soft direction is independent of the refusal direction under intervention, not merely low in cosine (Wollschläger et al.). Report its cosine with the refusal direction. Note the cone caveat: Hindi refusal signal may lie off the single DiM vector. The Makelov et al. illusion warning applies to the causal steering support.

**Scope the two-mechanism claim.** It rests on one model per family (Qwen3-8B, Llama-3.1-8B). Prior work used Qwen2.5, so whether the result is a Qwen-family or Qwen3-specific property is untested. Phrase it as "in these two models."

**Make the judge validation more granular.** Report per-class confusion. REFUSE_SOFT vs COMPLY is the most likely disagreement (Fu & Liu; PARIKSHA). State how many items the human graded and that there was one grader, since a single human limits inter-human reliability claims. Compare against IndicSafeEval's 0.84/0.70 and note its internal inconsistencies. Mention that the Claude judge is cross-family relative to the graded models (Panickssery et al.).

**Describe the data honestly.** Aya pairs are native Hindi with an English *translation*. State the script verification (all 914 items in Devanagari) and the 914 vs 915 count. Note that Hindi is an officially supported Llama-3.1 language while Bengali is not, which affects how the Bengali confirmation failure should be read.

**Hedge novelty claims.** Use "to our knowledge" for: the style direction; the English-only effect of the neurons; the sealed-language pre-registration; the garbled-output guard; and the two-mechanism framing. Each rests on abstract-level searches.

**Read IndicSafeEval and CompositeHarm in full.** Confirm their model lists and whether they report paired or refusal-style statistics. If IndicSafeEval does evaluate Qwen3-8B and Llama-3.1-8B in Hindi, the measurement novelty in rows 1–2 narrows to the paired explicit-refusal design and the non-adversarial setting.

## Citations to verify before submission

**Blocking: model lists or claims that affect novelty.**

- IndicSafeEval, arXiv:2609.03781: models [U]; kappa inconsistencies.
- CompositeHarm, arXiv:2602.07963: models [U].
- Aziz et al., arXiv:2606.01196: model versions UNVERIFIED in one note; "refusal" vs "harmfulness" direction; Hindi AUC in Fig. 5 not read; venue.
- BabelSteering, arXiv:2608.16577: Hindi inclusion; MMLU <3 pp vs no loss; ungated vs false-refusal ablation; venue.
- ALTSTEER, arXiv:2608.30197: zero-threshold detail from a snippet.
- AdaSteer: the low-resource limitation, from a snippet.
- Kompella & Mahajan, arXiv:2608.08032: which Indic languages.
- Minionese, arXiv:2607.10112: the body was not parsed in one note.
- Mātṛkā: Hindi coverage [U].
- arXiv:2503.13081: title and authors [U].
- Atil et al., arXiv:2511.00689: language and model lists.

**Venue or metadata unverified.**

- Zhao et al., arXiv:2507.11878 (NeurIPS 2025?) [U].
- Shen et al., arXiv:2401.13136: author list conflicts between notes; venue.
- Wang et al., arXiv:2410.03415: ICLR 2025 [U] in one note.
- SN-Tune: arXiv id; the cross-lingual overlap figure.
- Ball et al., arXiv:2406.09289 [U].
- Schut et al., arXiv:2502.15603.
- Dumas et al., arXiv:2411.08745.
- Brinkmann et al., arXiv:2501.06346.
- Zhao et al., MWork, arXiv:2402.18815.
- Sharing Matters, arXiv:2406.09265.
- Yong et al. survey, arXiv:2505.24119.
- OR-Bench (ICML 2025?) [FLAG].
- SORRY-Bench mutation languages [FLAG].
- METAL [FLAG].
- M-RewardBench [FLAG].
- JailbreakEval authors [FLAG].
- PolyGuard authors, and whether Hindi is covered [FLAG]/[M].
- RTP-LX authors [FLAG].
- von Recum et al. authors [FLAG].
- Yuan et al. authors [FLAG].
- Aya Prism: EMNLP 2024 [M]/[FLAG].
- ActAdd, RepE, Stickland et al., SafeSwitch, SADI, C-ΔΘ, Da Silva et al., Im & Li, Niranjan et al.: no venue in the arXiv records.
- CAA (ACL 2024), Tan et al. (NeurIPS 2024), AxBench (ICML 2025), Arditi et al. (NeurIPS 2024): from common citation; confirm on the proceedings sites.
- Joad et al.: EMNLP 2026 appears only on an author page.
- COSMIC: authors.
- "Base LLMs refuse too": authors [U].

**Unverified numbers quoted from memory [M]/[K].**

- XSafety's prompting result (19.1% → 9.7%).
- MultiJail's ~3x low-resource unsafe rate, 81%/41% intentional-scenario rates, and label definitions.
- HarmBench ~93% and JailbreakBench ~90% agreement.
- Global-MMLU's 28% culturally sensitive share.
- The Llama 3.1 and Qwen3 language-support lists.

**Do not cite without a primary source.**

- "BanglaShield": not found.
- Grey literature: the LessWrong SAE post, the Apart Research sprint reports, and the Arditi precursor LessWrong post. All [U]/UNVERIFIED.
- Lindsey et al. 2025 [UNVERIFIED].
- "Refusal Tokens," arXiv:2412.06748 [U].
- The distillation paper: listing [U], arXiv id unverified.
- MPO, arXiv:2505.16869 [U].
- RefusEU, arXiv:2606.07535: id/date mismatch.
- MORBench, CultureGuard, XL-SafetyBench, arXiv:2605.17173, arXiv:2602.16660 and arXiv:2601.08668: all [U].
- Any specific Insights or ICBINB proceedings year [FLAG].
- Landis & Koch 1977 for kappa thresholds (not fetched).

## Conclusion

The literature has moved quickly enough that the study's mechanistic contribution is no longer "an English refusal direction partly fails on Hindi." Wang et al., Aziz et al., Oppong et al. and Minionese have covered that ground, and Aziz et al. did it with Hindi and Llama-3.1 in the sample. What prior work lacks is a **response-side view**. Every mechanistic paper found scores refusal as a binary. None can tell a model that hedges in Hindi apart from one that complies, so none could see that Qwen3 and Llama-3.1 fail the same Hindi requests in qualitatively different ways. The four-way label is the paper's key enabling contribution, and the explicit-vs-soft direction and the two-mechanism contrast both depend on it. It should be presented that way, rather than as a grading detail.

The second implication is that the study's negative results are its sharpest contrast with the 2026 steering papers. Aziz et al. and BabelSteering report gains from English-anchored interventions with dev-style evaluation. The study pre-registered, held out a language, and found that a refusal-direction gate is not language-selective when the two languages score alike. That is exactly the reliability failure Tan et al. and AxBench predict. With a prompting baseline and a hard-negative check added, the paper can say what neither camp has: where the English-pivot story holds (Qwen3 projection), where it breaks (Llama-3.1 projection), and why fixing the symptom with a shared axis lifts English as much as Hindi.

---

**Note added 2026-11-16 (after the Llama human check).** Gap-table row 4 ("soft in Qwen, compliance in Llama") and the conclusion's
contrast are superseded. The human check (`docs/results/2026-11-16_llama_human_checks.md`) finds Llama's missing Hindi refusals mostly
soft too (14 of 20 sampled pairs). The surviving cross-model contrast is internal (refusal-direction projection), not behavioural.
