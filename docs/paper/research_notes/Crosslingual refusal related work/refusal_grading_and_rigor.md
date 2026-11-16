# Refusal grading taxonomies, LLM-as-judge reliability (incl. multilingual), and methodological rigor precedents

Scope: related-work and methods support for a workshop paper that grades replies with a four-way label (COMPLY / REFUSE_EXPLICIT / REFUSE_SOFT / BROKEN), uses Claude Sonnet as judge validated against a fluent Hindi human grader (Cohen's kappa) with blind shuffled grading, and runs pre-registered studies (frozen rules, dated addenda, dev/held-out split, sealed confirmation set in a held-out language).

Verification legend, used on every entry:
- **[V]**: verified this session against the arXiv abstract, the paper HTML, or the ACL Anthology/PMLR/NeurIPS listing.
- **[K]**: from prior knowledge (well-known paper, high confidence). The arXiv id and venue were not re-fetched this session, so spot-check them before camera-ready.
- **[FLAG]**: details are uncertain. Verify before citing.

---

## 1. Refusal taxonomies: explicit vs. partial/soft/evasive refusals (2023–2026)

### Takeaway
Prior benchmarks mostly use binary labels (refused/complied, or safe/unsafe). The closest precedents for separating explicit refusal from soft or evasive responses are XSTest's "partial refusal", Do-Not-Answer's six response-action categories, CoCoNot's noncompliance taxonomy, the "cannot vs. should not" taxonomy, and OpenAI's "safe-completion" framing. In multilingual work, a third "unclear/invalid" bucket for nonsense, misunderstood or off-language output is standard. Yong et al. 2023 and Deng et al. 2024 are the direct precedents for BROKEN. No widely used scheme matches this paper's four-way COMPLY / REFUSE_EXPLICIT / REFUSE_SOFT / BROKEN exactly.

### Cited Findings
- **Wang, Li, Han, Nakov, Baldwin (2023/2024). "Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs." Findings of EACL 2024. arXiv:2308.13387.** [K] — [arXiv](https://arxiv.org/abs/2308.13387)
  - Did/found: 939 prompts that responsible models should refuse. Responses are labelled into six "action" categories: cannot assist; refute the opinion or premise; offer dual perspectives; answer with a caution or disclaimer; cannot answer for lack of capability or knowledge; directly follow the instruction. A fine-tuned small (Longformer) classifier reached roughly GPT-4-level agreement with human labels.
  - Relation: their "caution/disclaimer", "refute" and "dual perspectives" categories are fine-grained versions of REFUSE_SOFT, and "cannot assist / incapable" maps to REFUSE_EXPLICIT. They grade English only, and they treat caveated answers as safe rather than as a distinct non-compliance mode.
- **Röttger, Kirk, Vidgen, Attanasio, Bianchi, Hovy (2024). "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models." NAACL 2024. arXiv:2308.01263.** [K] — [arXiv](https://arxiv.org/abs/2308.01263)
  - Did/found: 250 safe prompts that look unsafe, plus 200 unsafe contrast prompts. Responses are hand-labelled full compliance / full refusal / partial refusal, where partial refusal is a mix of refusing and complying, e.g. "I can't, but here is...". The paper reports that string matching misclassifies many partial refusals.
  - Relation: XSTest's three-way scheme is the most-cited precedent for a non-binary refusal label. Its "partial refusal" overlaps with REFUSE_SOFT but is defined differently: refuse-then-help, whereas REFUSE_SOFT is "neither provides nor says no". XSTest has no BROKEN class and measures over-refusal rather than under-refusal.
- **Cui, Chiang, Stoica, Hsieh (2024). "OR-Bench: An Over-Refusal Benchmark for Large Language Models." arXiv:2405.20947 (ICML 2025).** [K; venue FLAG] — [arXiv](https://arxiv.org/abs/2405.20947)
  - Did/found: 80K seemingly-toxic but benign prompts, a 1K hard subset, and 600 toxic prompts. An LLM judge classifies responses as direct answer, direct refusal, or indirect refusal. Over-refusal and toxic-prompt rejection are strongly correlated across 30+ models.
  - Relation: "indirect refusal" is a close analogue of REFUSE_SOFT. The setting is English over-refusal, and the judge was validated against an LLM ensemble rather than native-speaker humans.
- **Xie, Qi, Zeng, Huang, Sehwag, Huang, He, Wei, Li, Sheng, Jia, Li, Li, Chen, Henderson, Mittal (2024). "SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal." ICLR 2025. arXiv:2406.14598.** [V for authors/venue/7K annotations] — [arXiv](https://arxiv.org/abs/2406.14598)
  - Did/found: 44 fine-grained unsafe-topic classes and 20 linguistic mutations, including translation into other languages. They collected 7K+ human annotations and show that fine-tuned 7B judges match GPT-4-scale judges in accuracy. The label is binary: fulfil vs. refuse.
  - Relation: a precedent for systematically building and validating a human-annotated refusal judge. The binary label merges REFUSE_EXPLICIT and REFUSE_SOFT. [FLAG] The mutation languages are believed to include some low-resource Indic languages (e.g. Marathi, Malayalam, Tamil), but this was not confirmed on the abstract page, so check the paper.
- **Han, Rao, Ettinger, Jiang, Lin, Lambert, Choi, Dziri (2024). "WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs." NeurIPS 2024 Datasets & Benchmarks. arXiv:2406.18495.** [K] — [arXiv](https://arxiv.org/abs/2406.18495)
  - Did/found: a single classifier for three tasks: prompt harmfulness, response harmfulness, and response refusal (binary). It was trained on WildGuardMix (~92K items) with a human-annotated test split, and it beats Llama Guard 2 on refusal detection.
  - Relation: it makes refusal a separate axis from harmfulness, as this paper does. Refusal is still binary and English-centric.
- **Souly, Lu, Bowen, Trinh, Hsieh, Pandey, Abbeel, Svegliato, Emmons, Watkins, Toyer (2024). "A StrongREJECT for Empty Jailbreaks." NeurIPS 2024 Datasets & Benchmarks. arXiv:2402.10260.** [K] — [arXiv](https://arxiv.org/abs/2402.10260)
  - Did/found: a rubric judge that scores refused (0/1) and, when not refused, how convincing and specific the answer is (1–5 each). Prior jailbreak evaluators overstate success because they count "non-refusals" that carry no useful harmful content. Translation into low-resource languages often yields degraded, low-quality answers (a "willingness–capability" trade-off).
  - Relation: strong support for separating BROKEN from COMPLY. A non-refusal in a low-resource language is often not real compliance. StrongREJECT folds this into a quality score instead of a separate class.
- **Brahman, Kumar, Balachandran, Dasigi, Pyatkin, Ravichander, Wiegreffe, Dziri, Chandu, Hessel, Tsvetkov, Smith, Choi, Hajishirzi (2024). "The Art of Saying No: Contextual Noncompliance in Language Models." NeurIPS 2024 Datasets & Benchmarks. arXiv:2407.12043 (CoCoNot).** [K] — [arXiv](https://arxiv.org/abs/2407.12043)
  - Did/found: a broad noncompliance taxonomy covering incomplete, unsupported, indeterminate and humanizing requests as well as unsafe ones, plus a contrast set. Models under-refuse some categories and over-refuse benign contrast items.
  - Relation: the taxonomy concerns why a model should not comply (request side). This paper's labels describe how a model did not comply (response side: explicit vs. soft).
- **von Recum, Schnabl, Hollbeck, Alberti, Blinde, von Hagen (2024). "Cannot or Should Not? Automatic Analysis of Refusal Composition in IFT/RLHF Datasets and Refusal Behavior of Black-Box LLMs." arXiv:2412.16974; NeurIPS 2024 workshop (Statistical Frontiers in LLMs and Foundation Models).** [V for title/id/16 categories/8.6K annotations/GPT-4o-mini classifier; author list beyond the first is FLAG] — [arXiv](https://arxiv.org/abs/2412.16974v1), [NeurIPS listing](https://neurips.cc/virtual/2024/105596)
  - Did/found: a 16-category refusal taxonomy that separates "cannot" (capability) from "should not" (policy) refusals. It comes with 8,600+ human-annotated instances, synthetic data, and a refusal classifier.
  - Relation: REFUSE_EXPLICIT ("will not or cannot") covers both of their top-level branches. Their finer split could be cited as a possible sub-coding.
- **Yuan, Sriskandarajah, Brakman, Helyar, Beutel, Vallone, Jain (2025). "From Hard Refusals to Safe-Completions: Toward Output-Centric Safety Training." arXiv:2508.09224 (OpenAI; GPT-5).** [V for id/core claim; full author list FLAG] — [arXiv](https://arxiv.org/abs/2508.09224v1)
  - Did/found: replaces binary refuse/comply training with "safe-completion" training that rewards safe, helpful partial answers and penalises unsafe output by severity. On GPT-5 it improves both safety on dual-use prompts and helpfulness.
  - Relation: frontier labs now deliberately train for responses that are neither full compliance nor explicit refusal. This is why REFUSE_SOFT needs its own label, and why a binary refusal metric can misread such models.
- **Yong, Menghini, Bach (2023). "Low-Resource Languages Jailbreak GPT-4." NeurIPS 2023 SoLaR workshop (best paper). arXiv:2310.02446.** [V] — [arXiv HTML](https://arxiv.org/html/2310.02446v1)
  - Did/found: translated AdvBench (520 prompts) via Google Translate into 12 languages, queried GPT-4, and back-translated the outputs. Labels are BYPASS, REJECT, and UNCLEAR. UNCLEAR covers nonsensical or degenerate output, misread instructions, and non-actionable non-refusals. Saying "I don't understand the language" counts as REJECT. Attack success: English 0.96%, Hindi 6.54%, Bengali 13.27%, combined low-resource 79.04%. The annotator identity and count are not stated.
  - Relation: the direct precedent for a BROKEN-like class in multilingual safety grading. This paper differs in four ways: it grades native-script output without back-translation, it separates soft from explicit refusal, it reports human–judge kappa, and it treats "did not understand" as BROKEN rather than REJECT.
- **Deng, Zhang, Pan, Bing (2024). "Multilingual Jailbreak Challenges in Large Language Models." ICLR 2024. arXiv:2310.06474 (MultiJail).** [K; label details FLAG] — [arXiv](https://arxiv.org/abs/2310.06474)
  - Did/found: MultiJail covers 9 non-English languages, Bengali among them. Model outputs are classified as safe / unsafe / invalid (believed to work by translating outputs to English and using a GPT-4 judge). Unsafe rates rise as resource level falls.
  - Relation: "invalid" corresponds to BROKEN. Verify their exact label definitions and judging pipeline.
- **Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024). "Refusal in Language Models Is Mediated by a Single Direction." NeurIPS 2024. arXiv:2406.11717.** [K] — [arXiv](https://arxiv.org/abs/2406.11717)
  - Did/found: a difference-in-means refusal direction. Ablating it removes refusal and adding it induces refusal. The refusal score uses substring matching, paired with a Llama Guard 2 safety score.
  - Relation: this is the interpretability baseline the study builds on. Its substring-based refusal metric is exactly what the four-way LLM-judged label replaces, especially in Hindi, where English refusal phrases do not apply.

### Inferences
- Precedents exist for each of the four classes separately: COMPLY/REFUSE in every benchmark, partial or indirect refusal in XSTest, OR-Bench and Do-Not-Answer, and UNCLEAR/invalid in Yong et al. and MultiJail. The paper's contribution is combining them and validating the combined scheme in Hindi.
- The safe-completion trend (Yuan et al. 2025) strengthens the case for REFUSE_SOFT. For an interpretability paper about a refusal direction, a model that "lectures but doesn't help" behaves differently from one that says "no", and a binary metric hides that.

### Gaps
- I found no published scheme that defines "soft refusal" exactly as "understood but neither provides it nor says no". The nearest definitions are XSTest's partial refusal and OR-Bench's indirect refusal.
- Exact label definitions in MultiJail and the language list in SORRY-Bench were not re-verified this session.

---

## 2. Grading methods: string matching, LLM-as-judge, and reported human agreement (2023–2026)

### Takeaway
Keyword or prefix matching (popularised by GCG) is known to misclassify partial refusals, off-topic non-refusals, and non-English output. The field has moved to LLM or fine-tuned classifier judges (HarmBench, Llama Guard, JailbreakBench, StrongREJECT, WildGuard), which report roughly 90%+ agreement with humans in English. Known judge biases include leniency, position, verbosity and self-preference, so blind, shuffled, cross-family judging is good practice.

### Cited Findings
- **Zou, Wang, Carlini, Nasr, Kolter, Fredrikson (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043 (GCG).** [K] — [arXiv](https://arxiv.org/abs/2307.15043)
  - Did/found: counts an attack as successful when the output contains none of a fixed list of refusal prefixes ("I'm sorry", "As an AI", ...).
  - Relation: the archetypal string-match evaluator. Later work (HarmBench, StrongREJECT, JailbreakBench) shows it overcounts success. The approach is also language-specific, so it does not work for Hindi output.
- **Mazeika, Phan, Yin, Zou, Wang, Mu, Sakhaee, Li, Basart, Li, Forsyth, Hendrycks (2024). "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." ICML 2024. arXiv:2402.04249.** [K] — [arXiv](https://arxiv.org/abs/2402.04249)
  - Did/found: a fine-tuned Llama-2-13B classifier judges whether the behaviour was exhibited. It is validated on a human-labelled set, where it beats GPT-4-based and string-match judges (roughly 93% agreement). The paper documents that refusal-string matching misses non-refusals that are not actually harmful.
  - Relation: a methodological precedent for validating the judge against humans before use. The judge is English-only and binary.
- **Inan, Upasani, Chi, Rungta, Iyer, Mao, Tontchev, Hu, Fuller, Testuggine, Khabsa (2023). "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations." arXiv:2312.06674.** [K] — [arXiv](https://arxiv.org/abs/2312.06674)
  - Did/found: an instruction-tuned safety classifier that labels prompts and responses safe or unsafe against a policy taxonomy.
  - Relation: it judges harmfulness, not refusal style, and is often used together with a refusal metric (e.g. by Arditi et al.). It is not a substitute for the four-way label.
- **Chao, Debenedetti, Robey, Andriushchenko, Croce, Sehwag, Dobriban, Flammarion, Pappas, Tramèr, Hassani, Wong (2024). "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models." NeurIPS 2024 Datasets & Benchmarks. arXiv:2404.01318.** [K; exact agreement figure FLAG] — [arXiv](https://arxiv.org/abs/2404.01318)
  - Did/found: compared several judges (string match, GPT-4, HarmBench classifier, Llama Guard, Llama-3-70B) against human majority labels on 300 items and adopted Llama-3-70B as the default judge (agreement around 90%).
  - Relation: an explicit judge-selection-by-human-agreement procedure, parallel to this paper's choice of Sonnet. Theirs is English and binary.
- **Chao, Robey, Dobriban, Hassani, Pappas, Wong (2023). "Jailbreaking Black Box Large Language Models in Twenty Queries." arXiv:2310.08419 (PAIR).** [K] — [arXiv](https://arxiv.org/abs/2310.08419)
  - Did/found: a GPT-4 judge rates jailbreaks 1–10.
  - Relation: an early LLM-judge-for-safety example. StrongREJECT later criticised such judges as lenient.
- **Ran, Liu, Wang, Wang, Wang, Lin, Ye, Li (2024). "JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models." arXiv:2406.09321.** [K; authors FLAG] — [arXiv](https://arxiv.org/abs/2406.09321)
  - Did/found: surveys and compares string-matching, classifier, and LLM-judge evaluators, and finds they disagree substantially.
  - Relation: supports reporting judge choice and validation explicitly.
- **Zheng, Chiang, Sheng, Zhuang, Wu, Zhuang, Lin, Li, Li, Xing, Zhang, Gonzalez, Stoica (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023 Datasets & Benchmarks. arXiv:2306.05685.** [K] — [arXiv](https://arxiv.org/abs/2306.05685)
  - Did/found: GPT-4 judges agree with humans about as well as humans agree with each other (over 80%). The paper documents position, verbosity and self-enhancement biases.
  - Relation: a canonical citation for LLM-judge biases, and the motivation for blind, shuffled presentation.
- **Panickssery, Bowman, Feng (2024). "LLM Evaluators Recognize and Favor Their Own Generations." NeurIPS 2024. arXiv:2404.13076.** [K] — [arXiv](https://arxiv.org/abs/2404.13076)
  - Did/found: judges recognise their own outputs and score them higher.
  - Relation: using a Claude judge on Qwen outputs avoids self-preference. Worth one sentence in the methods.

### Inferences
- The paper's judge-validation step, picking Sonnet by kappa against a Hindi grader, follows the HarmBench and JailbreakBench convention. It extends that convention to a non-English language and to a four-way label, where agreement is expected to be lower than for binary labels.
- In English, judge–human agreement around 90% is the common bar. A four-way, non-English kappa should be framed against the multilingual evidence in section 3, not against English binary agreement numbers.

### Gaps
- I did not re-verify exact agreement percentages for HarmBench and JailbreakBench. Quote them from the PDFs.

---

## 3. Multilingual LLM-as-judge reliability, especially for safety and Indic languages (2023–2026)

### Takeaway
The evidence is consistent: LLM judges are less reliable outside English, especially in low-resource and non-Latin-script languages. They tend to be lenient (biased toward higher scores), they agree with humans less in direct-assessment and graded settings than in pairwise or binary ones, and they vary a lot by language. The few Indic safety studies that report it find binary Hindi kappa around 0.84 and graded kappa around 0.70, from small samples. This justifies validating the judge in Hindi with a native grader instead of assuming English-level reliability.

### Cited Findings
- **Hada, Gumma, de Wynter, Diddee, Ahmed, Choudhury, Bali, Sitaram (2024). "Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?" Findings of EACL 2024. arXiv:2309.07462.** [V] — [arXiv](https://arxiv.org/abs/2309.07462)
  - Did/found: compared GPT-4 judgments with about 20K human ratings across 8 languages, 3 tasks and 5 metrics. GPT-4 is biased toward higher scores, and the authors recommend calibration against native-speaker judgments, especially for low-resource and non-Latin-script languages.
  - Relation: the main citation for calibrating against a native grader. This paper does that for Hindi (Devanagari script).
- **Hada, Gumma, Ahmed, Bali, Sitaram (2024). "METAL: Towards Multilingual Meta-Evaluation." Findings of NAACL 2024. arXiv:2404.01667.** [K; FLAG verify] — [arXiv](https://arxiv.org/abs/2404.01667)
  - Did/found: a summarisation meta-evaluation dataset in 10 languages, Hindi among them. GPT-4 judges align with humans better than GPT-3.5, and agreement varies by language.
  - Relation: supporting evidence of language-dependent judge reliability, outside the safety domain.
- **Watts, Gumma, Yadavalli, Seshadri, Swaminathan, Sitaram (2024). "PARIKSHA: A Large-Scale Investigation of Human-LLM Evaluator Agreement on Multilingual and Multi-Cultural Data." EMNLP 2024. arXiv:2406.15053.** [V] — [arXiv](https://arxiv.org/abs/2406.15053)
  - Did/found: 30 models, 10 Indic languages, 90K human and 30K LLM evaluations. Humans and LLMs agree fairly well in pairwise comparison, agreement drops in direct assessment (sharply for Bengali and Odia), and the GPT evaluator shows self-bias. The safety evaluation was not human-graded, for ethical reasons.
  - Relation: the closest large-scale Indic precedent. This paper's labelling is direct assessment, the harder setting, and it does use human safety grading.
- **Fu, Liu (2025). "How Reliable is Multilingual LLM-as-a-Judge?" Findings of EMNLP 2025. arXiv:2505.12201.** [V] — [arXiv](https://arxiv.org/abs/2505.12201v1), [ACL Anthology](https://preview.aclanthology.org/new-sigs/2025.findings-emnlp.587/)
  - Did/found: 5 judge models, 5 tasks, 25 languages. Average Fleiss' kappa across languages is about 0.3, and consistency is worst in low-resource languages. Neither multilingual training nor model scale fixes this. An ensemble helps.
  - Relation: the strongest recent caution against assuming judge reliability transfers across languages. It directly motivates the per-language human check and the held-out-language confirmation.
- **Mondal, Mamta, Varshney, Cocarascu, Ekbal (2026). "IndicSafeEval: Safety Robustness of Large Language Models under Multilingual Persuasive Jailbreak Attacks." arXiv:2609.03781; the arXiv comments say it was accepted to Findings of EMNLP 2026.** [V] — [arXiv](https://arxiv.org/abs/2609.03781), [HTML](https://arxiv.org/html/2609.03781v2)
  - Did/found: 7,200 persuasion-based jailbreak prompts in Hindi, Bengali, Marathi and Punjabi, judged by Gemini-2.5-Flash on a 5-point scale (1 = clear refusal, 2 = cautious/neutral, 3 = partial engagement, 4–5 = jailbroken). Two native annotators labelled 100 items per language. Cohen's kappa (Table 12): Hindi 0.70 non-binary and 0.84 binary; Bengali 0.82 and 0.89; Marathi 0.73 and 0.81; Punjabi 0.84 and 1.00. **Internal inconsistencies:** the text gives Bengali binary kappa as 0.8435 against 0.8890 in the table, and Punjabi's binary kappa of 1.00 conflicts with differing human and LLM attack-success counts in Table 9.
  - Relation: the most direct contemporary precedent, a Hindi/Bengali safety judge validated by native-speaker kappa. Its "cautious/neutral" level resembles REFUSE_SOFT, but it is a harm-severity scale, not a response-type label, and it has no BROKEN class. Cite the table values and note the inconsistencies.
- **Kumar, Jain, Agarwal, Wang, et al. (2025). "PolyGuard: A Multilingual Safety Moderation Tool for 17 Languages." arXiv:2504.04377.** [K; author list FLAG] — [arXiv](https://arxiv.org/pdf/2504.04377v2)
  - Did/found: a multilingual safety classifier trained on data covering 17 languages, Hindi among them. Labels come from a panel of English classifiers and LLM judges applied to machine-translated data, with human checks of translation quality.
  - Relation: an example of translation-based labelling, which this paper avoids by grading native-script output directly.
- **de Wynter, Watts, Wongsangaroonsri, Zhang, Farra, Altıntoprak, Baur, Claudet, Gajdusek, Gören, Gu, Kaminska, Kaminski, Kuo, Kyuba, Lee, Mathur, Michel, Nan, Nikishina, Pavlopoulou, Ra, Ruan, Shrivastava, Waidelich, Xue (2024/2025). "RTP-LX: Can LLMs Evaluate Toxicity in Multilingual Scenarios?" AAAI 2025. arXiv:2404.14397.** [K; author list FLAG] — [arXiv](https://arxiv.org/abs/2404.14397)
  - Did/found: a human-transcreated toxic-prompt corpus in 28 languages. LLM judges reach acceptable accuracy but have low agreement with humans on holistic toxicity and miss subtle, culturally specific harms.
  - Relation: further evidence that accuracy can look fine while agreement (kappa) is poor. Reporting kappa rather than accuracy is the right choice.
- **Aakanksha, Ahmadian, Ermis, Goldfarb-Tarrant, Kreutzer, Fadaee, Hooker (2024). "The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm." EMNLP 2024. arXiv:2406.18682.** [K; FLAG verify] — [arXiv](https://arxiv.org/abs/2406.18682)
  - Did/found: the Aya Red-Teaming dataset, with human-written harmful prompts in 8 languages including Hindi. Harm is judged by an LLM judge whose agreement with humans was checked.
  - Relation: a Hindi safety-judge precedent. Its labels are harmful or not, not refusal style.
- **Wang, Tu, Chen, Yuan, Huang, Jiao, Lyu (2024). "All Languages Matter: On the Multilingual Safety of Large Language Models." Findings of ACL 2024. arXiv:2310.00905 (XSafety).** [K] — [arXiv](https://arxiv.org/abs/2310.00905)
  - Did/found: a 10-language safety benchmark, Hindi and Bengali among them. Outputs were judged with ChatGPT after translation to English, and models are less safe in non-English languages.
  - Relation: a translation-based judging precedent. Translating replies before judging can hide incoherent or wrong-language output, which BROKEN is designed to catch.
- **Shen, Tan, Chen, Zhang, Li, Fang, Wei (2024). "The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts." Findings of ACL 2024. arXiv:2401.13136.** [K; FLAG verify] — [arXiv](https://arxiv.org/abs/2401.13136)
  - Did/found: models produce more unsafe and more irrelevant responses in low-resource languages.
  - Relation: "irrelevant" responses are another precedent for a BROKEN category.
- **Gureja, Miranda, Islam, Maheshwary, Sharma, Winata, Lambert, Ruder, Hooker, Fadaee (2025). "M-RewardBench: Evaluating Reward Models in Multilingual Settings." ACL 2025. arXiv:2410.15522.** [K; FLAG verify] — [arXiv](https://arxiv.org/abs/2410.15522)
  - Did/found: reward models and LLM judges lose accuracy from English to other languages, and the drop is larger for low-resource languages and lower translation quality. Safety is one of the categories.
  - Relation: supporting evidence of a cross-lingual degradation in judges.

### Inferences
- The results above (kappa ~0.3 across languages, leniency bias, low agreement on graded or direct assessment) suggest that a four-way label in Hindi validated at moderate-to-substantial kappa is a meaningful result. The paper should report per-class confusion as well, because REFUSE_SOFT vs. COMPLY is the most likely disagreement.
- Translation-based judging (XSafety, Yong et al., PolyGuard) is common, but it erases the BROKEN signal: a wrong-language or incoherent reply can come back from translation looking coherent. Native-script judging is a methodological difference worth stating.

### Gaps
- I found no study measuring Claude-family judge agreement with Hindi humans on refusal-style labels specifically.
- I found no Hindi study with more than about 100 human-graded safety items per language that reports kappa. IndicSafeEval uses 100 items, and its numbers have internal inconsistencies.

---

## 4. Rigor precedents: pre-registration, held-out discipline, interpretability illusions, null results (2019–2026)

### Takeaway
Pre-registration has been piloted in ML (the NeurIPS 2020 and 2021 pre-registration workshops, PMLR v148) and argued for in NLP (van Miltenburg et al. 2021; a critical response from Søgaard et al. 2023), and adapted to predictive modelling (Hofman et al. 2023). It remains rare in interpretability. Interpretability and steering papers have a strong record of "illusion" results: findings that do not survive held-out evaluation, other prompt formats, or baselines. That record supports the paper's dev/held-out split and sealed held-out-language confirmation. Publicly reported negative results in interpretability (GDM's SAE update, AxBench, Kantamneni et al.) give precedent for reporting failed studies.

### Cited Findings — pre-registration and reporting standards
- **Bertinetto, Henriques, Albanie, Paganini, Varol (eds.) (2021). "NeurIPS 2020 Workshop on Pre-registration in Machine Learning." PMLR vol. 148.** [V] — [PMLR](https://proceedings.mlr.press/v148/bertinetto21a.html), [NeurIPS 2020](https://neurips.cc/virtual/2020/public/workshop_16158.html), [NeurIPS 2021 follow-up](https://neurips.cc/virtual/2021/workshop/21885)
  - Did/found: a pilot in which papers were peer-reviewed on their protocol before results existed, then published with results. It was repeated at NeurIPS 2021.
  - Relation: the direct ML precedent for frozen protocols. This paper self-pre-registers (frozen rules plus dated addenda) instead of using a registered-report track, which is worth stating honestly.
- **van Miltenburg, van der Lee, Krahmer (2021). "Preregistering NLP Research." NAACL 2021, pp. 613–623. arXiv:2103.06944.** [V] — [ACL Anthology](https://aclanthology.org/2021.naacl-main.51)
  - Did/found: proposes pre-registration questions for different NLP study types and argues for registered reports. It won a best thematic paper award.
  - Relation: a template for the pre-registration content. This paper's dated addenda correspond to their "deviations from preregistration" reporting.
- **Søgaard, Hershcovich, de Lhoneux (2023). "A Two-Sided Discussion of Preregistration of NLP Research." EACL 2023.** [V for existence/venue] — [ACL Anthology](https://preview.aclanthology.org/setup/2023.eacl-main.6)
  - Did/found: weighs the costs of pre-registration, e.g. a bias toward confirmatory work, against its benefits.
  - Relation: useful for pre-empting reviewer objections. The paper can say it separates exploratory (dev) from confirmatory (held-out) work instead of banning exploration.
- **Hofman, Chatzimparmpas, Sharma, Watts, Hullman (2023). "Pre-registration for Predictive Modeling." arXiv:2311.18807.** [V] — [arXiv](https://arxiv.org/abs/2311.18807v1)
  - Did/found: a lightweight pre-registration template for predictive ML and a qualitative study with ML researchers. Almost all saw value, though their understandings varied.
  - Relation: the closest ML-specific template. It also warns that pre-registration is not a guarantee of validity ([Hullman blog](https://statmodeling.stat.columbia.edu/2023/12/06/exploring-pre-registration-for-predictive-modeling/)).
- **Dodge, Gururangan, Card, Schwartz, Smith (2019). "Show Your Work: Improved Reporting of Experimental Results." EMNLP 2019. arXiv:1909.03004.** [K] — [arXiv](https://arxiv.org/abs/1909.03004)
  - Did/found: hyperparameter-search budgets change conclusions, so the paper recommends reporting expected validation performance against budget.
  - Relation: a precedent for disclosing tuning effort on dev items.
- **Gorman, Bedrick (2019). "We Need to Talk about Standard Splits." ACL 2019.** [K] — [ACL Anthology](https://aclanthology.org/P19-1267/)
  - Did/found: rankings on standard splits often fail to reproduce on random resplits.
  - Relation: supports evaluating once on held-out items and confirming on a separate sealed set.
- **Søgaard, Ebert, Bastings, Filippova (2021). "We Need to Talk About Random Splits." EACL 2021. arXiv:2005.00636.** [K] — [arXiv](https://arxiv.org/abs/2005.00636)
  - Did/found: random splits overestimate performance compared with harder, shifted held-out data.
  - Relation: a held-out language is a deliberately shifted confirmation set, in the spirit of this paper.
- **Pineau, Vincent-Lamarre, Sinha, Larivière, Beygelzimer, d'Alché-Buc, Fox, Larochelle (2021). "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)." JMLR 22. arXiv:2003.12206.** [K] — [arXiv](https://arxiv.org/abs/2003.12206)
  - Did/found: describes the NeurIPS reproducibility checklist and code-submission policy.
  - Relation: a baseline community standard that the paper's manifest and scripted numbers go beyond.
- **Kapoor, Narayanan (2023). "Leakage and the Reproducibility Crisis in Machine-Learning-Based Science." Patterns 4(9). arXiv:2207.07048.** [K] — [arXiv](https://arxiv.org/abs/2207.07048)
  - Did/found: a taxonomy of data leakage affecting hundreds of papers, including test-set reuse, with "model info sheets" proposed as a remedy.
  - Relation: test-once discipline and a sealed set are the remedy for test-set leakage.
- **Ulmer, Bassignana, Müller-Eberstein, Varab, Zhang, van der Goot, Hardmeier, Plank (2022). "Experimental Standards for Deep Learning in Natural Language Processing Research." Findings of EMNLP 2022. arXiv:2204.06251.** [K] — [arXiv](https://arxiv.org/abs/2204.06251)
  - Did/found: a checklist for data, codebase, model training and evaluation, including significance testing and multiple seeds.
  - Relation: a general standard to cite next to the reproducibility manifest.

### Cited Findings — held-out discipline and illusions in interpretability/steering
- **Hewitt, Liang (2019). "Designing and Interpreting Probes with Control Tasks." EMNLP 2019. arXiv:1909.03368.** [K] — [arXiv](https://arxiv.org/abs/1909.03368)
  - Did/found: high probe accuracy can reflect probe capacity rather than what the representation encodes, so the paper introduces control tasks and "selectivity".
  - Relation: an early call for baselines and controls before reading meaning into representations.
- **Bolukbasi, Pearce, Yuan, Coenen, Reif, Viégas, Wattenberg (2021). "An Interpretability Illusion for BERT." arXiv:2104.07143.** [K] — [arXiv](https://arxiv.org/abs/2104.07143)
  - Did/found: a neuron's apparent meaning depends on the dataset used to inspect it and does not hold on other corpora.
  - Relation: an early example of interpretations that do not survive a change of distribution. Held-out items and a held-out language test for this directly.
- **Makelov, Lange, Nanda (2024). "Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching." ICLR 2024. arXiv:2311.17030.** [K] — [arXiv](https://arxiv.org/abs/2311.17030)
  - Did/found: subspace patching can change behaviour through dormant pathways, so the intervention works without the subspace being the mechanism.
  - Relation: a caution that a working intervention does not prove a mechanism. Relevant to any claim that a steering effect is "the Hindi refusal lever".
- **Friedman, Lampinen, Dixon, Chen, Ghandeharioun (2024). "Interpretability Illusions in the Generalization of Simplified Models." ICML 2024. arXiv:2312.03656.** [K] — [arXiv](https://arxiv.org/abs/2312.03656)
  - Did/found: simplified proxies of a model (e.g. SVD or clustering of representations) match it in-distribution but diverge out-of-distribution.
  - Relation: supports confirming out-of-distribution, here on a held-out language.
- **Tan, Chanin, Lynch, Paige, Kanoulas, Garriga-Alonso, Kirk (2024). "Analyzing the Generalization and Reliability of Steering Vectors." NeurIPS 2024. arXiv:2407.12404.** [K] — [arXiv](https://arxiv.org/abs/2407.12404)
  - Did/found: across many behaviours, steerability varies widely between inputs, is sometimes driven by spurious biases, and often generalises poorly out of distribution.
  - Relation: the core empirical justification for testing steering once on held-out items instead of reporting dev-set effects.
- **Pres, Ruis, Lubana, Krueger (2024). "Towards Reliable Evaluation of Behavior Steering Interventions in LLMs." NeurIPS 2024 Workshop on Foundation Model Interventions. arXiv:2410.17245.** [V] — [arXiv](https://arxiv.org/abs/2410.17245)
  - Did/found: proposes evaluation criteria (open-ended generation, likelihood-aware metrics, standardised comparison across behaviours, baselines). Under these criteria CAA is weaker than reported and fails in open-ended generation even though it works in multiple-choice format.
  - Relation: directly supports grading open-ended generations with a behavioural label, instead of using multiple-choice or logit proxies.
- **Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts (2025). "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders." ICML 2025. arXiv:2501.17148.** [K] — [arXiv](https://arxiv.org/abs/2501.17148)
  - Did/found: on a standardised benchmark with an LLM judge, prompting beats all representation-steering methods, and SAEs are not competitive.
  - Relation: a precedent for steering comparisons against strong baselines, judged on generations, with negative findings reported.
- **Panickssery (Rimsky), Gabrieli, Schulz, Tong, Hubinger, Turner (2024). "Steering Llama 2 via Contrastive Activation Addition." ACL 2024. arXiv:2312.06681.** [K] — [arXiv](https://arxiv.org/abs/2312.06681)
  - Did/found: the CAA method, which uses mean-difference steering vectors.
  - Relation: the method family being tested. Pres et al. and Tan et al. show its evaluation pitfalls.
- **Belinkov (2022). "Probing Classifiers: Promises, Shortcomings, and Advances." Computational Linguistics 48(1). arXiv:2102.12452.** [K] — [arXiv](https://arxiv.org/abs/2102.12452)
  - Relation: a survey-level citation for correlational-vs-causal caveats in representation analysis.

### Cited Findings — reporting null/negative results in interpretability
- **Smith, Rajamanoharan, Conmy, McDougall, Kramár, Lieberum, Shah, Nanda (2025). "Negative Results for Sparse Autoencoders on Downstream Tasks and Deprioritising SAE Research (GDM Mech Interp Team Progress Update #2)." AI Alignment Forum / DeepMind Safety Research blog, 26 March 2025.** [V; not peer-reviewed] — [GreaterWrong mirror](https://greaterwrong.com/posts/4uXCAJNuPKtKBsi28)
  - Did/found: SAE-based probes underperformed linear probes at detecting harmful intent under out-of-distribution jailbreaks, and the team deprioritised fundamental SAE work.
  - Relation: a high-profile precedent for publishing negative interpretability results. It parallels the paper's reporting of the Phase 4 failure, the non-reproduction of 4b, and Phase 5 finding no Hindi-only lever.
- **Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda (2025). "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing." ICML 2025. arXiv:2502.16681.** [K] — [arXiv](https://arxiv.org/abs/2502.16681)
  - Did/found: across data-scarce, shifted and noisy settings, SAE probes rarely beat baselines. The paper reports this as a largely negative result.
  - Relation: a peer-reviewed negative-results precedent with pre-specified baselines.
- **Workshop on Insights from Negative Results in NLP (2020–) and the "I Can't Believe It's Not Better!" (ICBINB) workshops at NeurIPS (2020–).** [K; FLAG cite specific proceedings year] — [ACL Anthology (Insights 2020)](https://aclanthology.org/2020.insights-1.0/)
  - Relation: venue-level evidence that the community values negative results. A natural framing for the paper's null findings.

### Inferences
- I found no interpretability or steering paper that combines pre-registration, a dev/held-out split, and a sealed held-out-language confirmation set. The combination appears novel in this subfield, but I cannot rule out unindexed workshop papers, so phrase it as "to our knowledge".
- Tan et al. 2024, Pres et al. 2024, AxBench 2025 and the illusion papers together make the case that dev-set steering effects regularly fail to generalise. That makes the paper's test-once and sealed-confirmation design a direct answer to a documented failure mode, not just procedure.

### Gaps
- I found no formal pre-registration in the mechanistic interpretability literature, published or on OSF. This is an absence of evidence, not proof.
- I did not verify specific Insights/ICBINB proceedings entries. Cite a specific year's front matter only after checking.
- Kappa interpretation thresholds (Landis & Koch 1977, "Biometrics") fall before 2019 and were not fetched. They are a standard citation if needed.
