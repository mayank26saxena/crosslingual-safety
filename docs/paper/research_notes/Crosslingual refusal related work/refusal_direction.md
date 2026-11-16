# Refusal direction line of work and its cross-lingual extensions

Scope: annotated bibliography for the related-work section. The study under discussion fits an English-only difference-of-means (DiM) refusal direction at the last prompt token (harmful minus harmless), picks the layer by directional ablation, validates it by how much of the first-token refusal-score gap ablation removes, and then measures how Hindi and English versions of the same harmful requests project onto it in Qwen3-8B and Llama-3.1-8B-Instruct. It also fits a second DiM direction for explicit vs soft refusals, scored by cross-validated AUROC.

Verification legend: **[V]** means I checked it this session on the arXiv abstract/HTML page or the proceedings page. **[V-meta]** means title, authors and date were confirmed but the body was not read. **[U]** means it comes from memory or a secondary source and was not verified this session; check before citing.

## Q1. Arditi et al. 2024: method details and models

### Takeaway
Arditi et al. fit a DiM direction for each layer and each post-instruction position (128 harmful / 128 harmless training prompts). They select one candidate using ablation and addition criteria on 32 validation prompts, plus a KL limit and a layer limit, and they score refusal with a first-token log-odds metric over family-specific refusal tokens. The study's pipeline is a near-direct port of this recipe. What is new in the study is the cross-lingual projection analysis, not the direction extraction.

### Cited Findings
- **[V] Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. (2024). "Refusal in Language Models Is Mediated by a Single Direction." NeurIPS 2024. arXiv:2406.11717** (v1 17 Jun 2024, v3 30 Oct 2024). — [arXiv](https://arxiv.org/abs/2406.11717); [NeurIPS proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf)
  - *What they did / found:* in 13 open chat models, one residual-stream direction both suppresses refusal of harmful prompts (when ablated) and induces refusal of harmless ones (when added). Weight orthogonalization along it gives a white-box jailbreak with little loss of capability. — [arXiv](https://arxiv.org/abs/2406.11717)
  - *Relation to the study:* this is the method the study adopts. The study adds cross-lingual (Hindi) projection measurements and a second refusal-style direction. Arditi et al. are English-only.
- Models: Qwen Chat 1.8B/7B/14B/72B; Yi 6B/34B; Gemma IT 2B/7B; Llama-2 Chat 7B/13B/70B; Llama-3 Instruct 8B/70B. — [arXiv HTML v3](https://arxiv.org/html/2406.11717v3)
- Data: 128 harmful training prompts (AdvBench, MaliciousInstruct, TDC2023), 32 harmful validation prompts (HarmBench standard), and 128/32 harmless from Alpaca. Evaluation used JailbreakBench (100), 100 Alpaca prompts for induction, and the HarmBench test set (159). — [arXiv HTML v3](https://arxiv.org/html/2406.11717v3)
- Extraction: DiM of harmful minus harmless mean activations, computed for each layer and each post-instruction token position. — [arXiv HTML v3](https://arxiv.org/html/2406.11717v3)
- Selection: the candidate with the lowest bypass_score (refusal metric under ablation on harmful validation prompts), subject to induce_score > 0, kl_score < 0.1 on harmless prompts, and layer < 0.8L. — [arXiv HTML v3](https://arxiv.org/html/2406.11717v3)
- Refusal metric: log-odds of the probability mass on a small per-family set of refusal tokens at the first generated position. Behavioral refusal is also scored by substring matching, and safety by Llama Guard 2. — [arXiv HTML v3](https://arxiv.org/html/2406.11717v3)
- Interventions:
  - Directional ablation x' = x − r̂r̂ᵀx at all layers and positions.
  - Activation addition x' = x + r at the source layer.
  - Weight orthogonalization W'_out = W_out − r̂r̂ᵀW_out, which is equivalent to directional ablation.
  - [arXiv HTML v3](https://arxiv.org/html/2406.11717v3)
- On Qwen-1.8B, adversarial (GCG) suffixes suppress the direction's expression at the last token and pull top attention heads onto the suffix. — [arXiv HTML v3](https://arxiv.org/html/2406.11717v3)
- An appendix reports that the direction is also present in the corresponding base models. Cosine similarity with base-model activations on harmful prompts is high, which suggests safety tuning repurposes an existing direction. — [arXiv HTML v3](https://arxiv.org/html/2406.11717v3) (appendix title seen; base-model detail from search summary, not read directly)
- **[U] Precursor LessWrong / Alignment Forum post:** Arditi, Obeso, Syed, Paleka, Rimsky, Gurnee, Nanda, "Refusal in LLMs is mediated by a single direction," LessWrong, April 2024. Its primary early write-up was on LessWrong. I did not fetch it this session.

### Inferences
- The study's layer choice by directional ablation and its first-token refusal-score gap closely match Arditi's bypass_score and refusal metric. Two differences should be stated explicitly:
  - The study uses only the last prompt token, not all post-instruction positions.
  - The study reports "fraction of the score gap removed" rather than the selection score itself.
- Neither Qwen3-8B nor Llama-3.1-8B-Instruct is among Arditi's 13 models. Llama-3.1-8B-Instruct is, however, used in Wang et al. 2025 and Aziz et al. 2026 (below).

### Gaps
- I did not read the exact refusal-token IDs for Llama-3. The paper lists per-family token sets.
- I did not read the Appendix J numbers for base models.

## Q2. Follow-ups: cones, harmfulness vs refusal, ReFAT, base vs chat, jailbreaks

### Takeaway
Since 2024, follow-ups have qualified the "single direction" picture in several ways:
- Refusal can be mediated by multi-dimensional cones.
- Different refusal categories give distinct but functionally similar directions.
- Harmfulness and refusal are encoded at different positions and along different directions.
- Most jailbreaks act by suppressing the refusal direction rather than the harm signal.

The study's single English DiM direction, read at the last prompt token, is closest to what Zhao et al. call the "refusal" (post-instruction) signal, not the "harmfulness" signal.

### Cited Findings
- **[V] Wollschläger, T., Elstner, J., Geisler, S., Cohen-Addad, V., Günnemann, S., Gasteiger, J. (2025). "The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence." ICML 2025 (PMLR 267). arXiv:2502.17420.** — [arXiv](https://arxiv.org/abs/2502.17420); [PMLR](https://proceedings.mlr.press/v267/wollschlager25a.html)
  - *Did / found:* they use gradient-based representation engineering (RDO) to find refusal directions with fewer side effects than DiM. They find multi-dimensional polyhedral cones in which every direction mediates refusal. They also show that orthogonal directions are not necessarily independent under intervention.
  - *Relation:* the study uses a single DiM vector. Its low Hindi projection could reflect a Hindi refusal component that lies elsewhere in a refusal cone, which is a caveat worth stating.
- **[V] Zhao, J., Huang, J., Wu, Z., Bau, D., Shi, W. (2025). "LLMs Encode Harmfulness and Refusal Separately." arXiv:2507.11878** (v1 16 Jul 2025; v4 Dec 2025). Venue: **[U]** possibly NeurIPS 2025, not verified. — [arXiv](https://arxiv.org/abs/2507.11878)
  - *Did / found:* harmfulness is encoded mainly at the last instruction token (t_inst), and refusal at the last post-instruction (chat-template) token (t_post-inst). Steering along the harmfulness direction changes the model's harm judgment, while steering along the refusal direction produces refusal without changing that judgment. Some jailbreaks lower the refusal signal while leaving harmfulness intact. Their "Latent Guard" matches or beats Llama Guard 3 8B. — [arXiv HTML](https://arxiv.org/html/2507.11878v2)
  - *Relation:* this matters most for interpreting the study's results. If "last prompt token" means the last template token, the study is reading the refusal signal. A weaker Hindi projection there could then mean "harm detected but refusal not triggered" rather than "harm not detected." The study should say which token it uses.
- **[V] Yu, L., Do, V., Hambardzumyan, K., Cancedda, N. (2024/2025). "Robust LLM Safeguarding via Refusal Feature Adversarial Training" (ReFAT). ICLR 2025. arXiv:2409.20089.** — [arXiv](https://arxiv.org/abs/2409.20089v2); [ICLR](https://iclr.cc/virtual/2025/poster/28144)
  - *Did / found:* many input-level attacks work by ablating the refusal feature (a DiM direction). Training with refusal-feature ablation (ReFAT) improves robustness across three LLMs at low cost.
  - *Relation:* the study treats a language switch to Hindi as a natural attack that may weaken the refusal feature. Adversarial training with ReFAT is the defense counterpart.
- **[V] Joad, F., Hawasly, M., Boughorbel, S., Durrani, N., Sencar, H. T. (2026). "There Is More to Refusal in Large Language Models than a Single Direction." arXiv:2602.02132** (v2 15 Sep 2026). An author page lists it as EMNLP 2026 ([PDF](https://alt.qcri.org/~ndurrani/pubs/EMNLP26a.pdf)); arXiv lists no venue. — [arXiv HTML v2](https://arxiv.org/html/2602.02132v2)
  - *Did / found:* they fit DiM directions for 11 refusal categories (safety, capability limits, underspecification, over-refusal) on Gemma-2-9B-it, Llama-3-8B-Instruct and Qwen-7B-Chat. The directions are geometrically distinct (pairwise cosine about 0.4–0.6) but steer refusal in nearly the same way. Ablating the safety direction leaves about 30–50% residual refusal on non-safety categories. SAEs show a shared core of refusal latents plus a long tail of style/domain-specific latents.
  - *Relation:* this is the closest work to the study's second direction. It separates refusal *categories* by reason, not *style* (explicit vs soft).
- **[V] Hildebrandt et al. (2025). Nonlinear refusal behavior. arXiv:2501.08145.** — cited in [BabelSteering](https://arxiv.org/html/2608.16577v1). [V-meta only via that citation; I did not read the abstract.] *Relation:* questions whether refusal is linear at all.
- **[V-meta] COSMIC: "Generalized Refusal Direction Identification in LLM Activations." Findings of ACL 2025.** — [ACL Anthology PDF](https://aclanthology.org/2025.findings-acl.1310.pdf)
  - *Did / found (search summary):* selects refusal directions using cosine-similarity criteria, without relying on refusal-token matching. It works even when a model refuses everything. Authors not verified.
  - *Relation:* an alternative to the study's token-based selection, which could matter for Hindi if refusal tokens differ in Hindi outputs.
- **[V-meta] Base models:** "Base LLMs refuse too," LessWrong post (primary write-up on LessWrong). Authors **[U]**: Kissane, Krzyzanowski, Conmy, Nanda (2024). — [LessWrong](https://www.lesswrong.com/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too)
  - *Found (search summary):* base models refuse 48% of harmful vs 3% of harmless requests, while chat models refuse 90% vs 3%. Ablating the chat model's refusal direction also removes base-model refusals. Base refusals often take the form of claimed incompetence or misunderstanding, which is a style difference.
  - *Relation:* this supports the idea that refusal has distinct surface styles sharing one mechanism.
- **[V-meta] Jailbreaks / suppression, 2026 preprints found by search (abstracts not read):**
  - "HARC: Coupling Harmfulness and Refusal Directions for Robust Safety Alignment," arXiv:2607.00572 ([link](https://arxiv.org/html/2607.00572)).
  - "Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection," arXiv:2605.02958 ([link](https://arxiv.org/html/2605.02958)).
  - "A Mechanistic Study of the Prefill Jailbreak," arXiv:2607.14147 ([link](https://arxiv.org/pdf/2607.14147)).
  - "Refusal Lives Downstream of Persona in Chat Models," arXiv:2606.26161 ([link](https://arxiv.org/html/2606.26161v1)).
  - "Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction," arXiv:2509.15202 ([link](https://arxiv.org/pdf/2509.15202)).
  - "SAEs are highly dataset dependent: a case study on the refusal direction," Alignment Forum post ([link](https://www.alignmentforum.org/posts/rtp6n7Z23uJpEH7od/saes-are-highly-dataset-dependent-a-case-study-on-the)).
- **[U] Ball, S., Kreuter, F., Panickssery, N. (2024). "Understanding Jailbreak Success: A Study of Latent Space Dynamics in Large Language Models." arXiv:2406.09289.** From memory: jailbreaks of many types reduce the refusal-direction (harmfulness) signal. Verify before citing.
- **[V] Wang, X. et al. (2024/2025). "Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation." arXiv:2410.03415** (cited by [BabelSteering](https://arxiv.org/html/2608.16577v1); venue **[U]**, ICLR 2025 from memory).
  - *Did:* extract a *false-refusal* DiM vector, orthogonalize it against the true refusal direction, and ablate it.
  - *Relation:* this is a precedent for fitting a second DiM direction that is distinct from the main refusal direction, as the study's explicit-vs-soft direction is.

### Inferences
- Reviewers will probably ask whether a single English DiM vector is sufficient. Wollschläger (cones) and Joad (category directions) both suggest that projection onto one vector understates refusal-relevant signal. Zhao suggests the readout position determines whether "harm" or "refusal" is being measured.

### Gaps
- I did not verify the Zhao et al. venue or the Ball et al. citation.
- I only have titles for the 2026 jailbreak-mechanism preprints.

## Q3. Cross-lingual: is the refusal direction shared across languages?

### Takeaway
Wang et al. (NeurIPS 2025) report that English DiM directions transfer almost perfectly to other safety-aligned languages and that directions are approximately parallel across languages. They also report that **harmful and harmless prompts separate less in non-English languages**: silhouette scores are lower than English for every model, including Llama-3.1-8B-Instruct. Later work extends this:
- Aziz et al. 2026, which **includes Hindi and Llama-3.1-8B-Instruct**, finds that low-resource harmful prompts *project lower* onto the high-resource direction but remain linearly separable (AUC > 0.85). They call this a calibration ("action") failure.
- Ivanova 2026 finds that universality is graded and fails for strongly misaligned low-resource source languages.

No paper found studies Qwen3-8B, or Hindi paired prompts projected onto an English-only direction, as the main object. That combination appears to be the study's novelty.

### Cited Findings
- **[V] Wang, X., Wang, M., Liu, Y., Schütze, H., Plank, B. (2025). "Refusal Direction is Universal Across Safety-Aligned Languages." NeurIPS 2025. arXiv:2505.17306** (v1 22 May 2025; v2 25 Feb 2026). — [arXiv](https://arxiv.org/abs/2505.17306v1); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/116911); [proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/2e94772e3c079d83f79c311d07456111-Paper-Conference.pdf)
  - *Data:* PolyRefuse, which machine-translates (Google Translate) harmful and harmless prompts into 13 languages plus English: de, es, fr, it, nl, ja, pl, ru, zh, ko, ar, th, yo. **There is no Hindi or other Indic language.** Each language has 128 training prompts per class, 32 harmful validation prompts, and 572 harmful test prompts. — [arXiv HTML v2](https://arxiv.org/html/2505.17306v2)
  - *Method:* Arditi-style DiM. Candidates come from all layers at final instruction-token positions, and the one whose ablation most lowers the refusal score is selected. For Llama-3.1-8B the best positions were −5 and −1, peaking near layer 12. — [arXiv HTML v2](https://arxiv.org/html/2505.17306v2)
  - *Models:* Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct and gemma-2-9B-it for the main compliance and transfer results. English-vector ablation is also run on the Yi, Qwen2.5, Llama-3 and Gemma-2 families, including Qwen2.5-14B. — [arXiv HTML v2](https://arxiv.org/html/2505.17306v2)
  - *Finding:* ablating the English direction removes refusal in other languages almost perfectly. Directions from any safety-aligned language transfer to the others. Cross-language similarity is shown only as heatmaps, with no numeric cosines given, and peaks near the extraction layer. — [arXiv HTML v2](https://arxiv.org/html/2505.17306v2)
  - *Low-resource (Yoruba):* baseline compliance on harmful Yoruba prompts is 82.9% for Llama-3.1-8B, about 74% for Qwen2.5-7B, and 56.6% for gemma-2-9B.
    - English-vector ablation raises gemma's compliance from 0.57 to 0.87.
    - Adding the English vector lowers gemma's Yoruba compliance from 58% to 28%, which is still higher than for other languages.
    - [arXiv HTML v2](https://arxiv.org/html/2505.17306v2)
  - *Separation, which directly answers the key question:* using PCA and silhouette scores (no projection metric), **English has the highest harmful/harmless silhouette score for every model.**
    - Llama-3.1-8B: en 0.496 vs 0.218 (de) to 0.317 (yo).
    - Qwen2.5-7B: en 0.389 vs 0.196 (zh) to 0.288 (yo).
    - gemma-2-9B: en 0.306 vs 0.230 (yo) to 0.288 (de).
    - ja, ko, ru and th score about 0.22–0.29.
    - Yoruba's relatively high score is explained by compact but poorly separated clusters.
    - Jailbroken samples fall between the harmful and harmless clusters.
    - The authors attribute cross-lingual jailbreaks to this representational weakness rather than to a missing refusal mechanism.
    - [arXiv HTML v2](https://arxiv.org/html/2505.17306v2)
  - *Relation:* this is the most direct precedent and the one the study should position against. Similarities: same method, and one shared model (Llama-3.1-8B-Instruct). The study differs in four ways: (a) Hindi, which is absent from PolyRefuse; (b) Qwen3-8B; (c) a projection metric onto the English direction for *paired* translations, rather than silhouette scores or PCA; (d) a separate refusal-style axis.
- **[V] Aziz, R., Hanif, I. A., Koto, F. (2026). "Low-Resource Safety Failures Are Action Failures, Not Representation Failures." arXiv:2606.01196** (31 May 2026, preprint, MBZUAI). — [arXiv HTML](https://arxiv.org/html/2606.01196v1)
  - *Setup:* extends PolyRefuse to 23 languages in three tiers. **Hindi is included, in the medium-resource tier.** The low-resource tier is Swahili, Amharic, Burmese, Khmer, Sinhala and Yoruba. Models are Qwen2.5-7B-Instruct, Gemma-2-9B-it and **Llama-3.1-8B-Instruct**. The direction is fit by DiM at the final token on a *pooled 10-high-resource-language set* (layers: Qwen 15, Gemma 20, Llama 10), with an English-only direction as a baseline.
  - *Finding:* harmful refusal averages 87.9% for high-resource, 78.9% for medium-resource and 43.9% for low-resource languages. **Low-resource harmful prompts project lower onto the high-resource direction (the mean harmful score shifts downward) but remain linearly separable (most AUC > 0.85).** The authors frame this as miscalibration of the refusal threshold. A few-shot latent gate using 1–4 target-language examples per class recovers selective refusal.
  - *Hindi:* harmful refusal is 66.3% (Qwen2.5-7B), 92.7% (Gemma-2-9B) and 67.0% (Llama-3.1-8B). Harmless refusal is 1.2–3.4%. No Hindi-specific AUC or projection shift appears in the text; there is a per-language AUC figure.
  - *Relation:* this is the closest prior result to the study's Hindi projection measurement, and it uses the same Llama model. The study differs by using an English-only direction, paired same-request translations, Qwen3-8B, and the style axis. The study should check whether its Hindi findings agree with "lower projection but still separable."
- **[V] Ivanova, A. (2026). "Is Refusal Direction Universal across Low-Resourced Languages?" BlackboxNLP @ EMNLP 2026 (Reproducibility Track). arXiv:2610.28897** (20 Oct 2026). — [arXiv](https://arxiv.org/abs/2610.28897)
  - *Did / found:* replicates Wang et al. on Qwen2.5-14B-Instruct, adding Belarusian, Bashkir and Tajik. Universality is "graded rather than binary." Directions from safety-aligned and partially aligned languages transfer broadly, but directions from strongly misaligned languages fail to modulate refusal. Geometric alignment and behavioral transfer are only partially linked.
  - *Relation:* this supports the study's choice to measure projection (geometry) separately from behavior.
- **[V] Stein, E. V., Meier, D., Ruas, T., Wahle, J. P., Gipp, B. (2026). "BabelSteering: Multilingual Safety Alignment via English Steering Vectors." arXiv:2608.16577** (17 Aug 2026, preprint). — [arXiv HTML](https://arxiv.org/html/2608.16577v1)
  - *Did / found:* adds an English DiM refusal vector (128 prompts per class, KL < 0.1 layer selection) and ablates an orthogonalized false-refusal vector, testing Gemma-7B-it, Llama-3.1-8B-Instruct and Qwen-7B-Chat in 8 languages (Bengali included, no Hindi). Harmful refusal rises (+11 pp on average for Gemma; Bengali +17 pp). Over-refusal rises more in low-resource languages, and MMLU drops by less than 3 pp.
  - *Relation:* this is the intervention-side counterpart, which uses the English direction as a fix. The study is diagnostic.
- **[V] Faruna, G. A. (2026). "Latent Space Refusal Anchoring for Low-Resource African Languages: Mechanistic Safety Recovery Without Retraining." ICML 2026 Workshop on Global South in ML. arXiv:2608.18089.** — [arXiv HTML](https://arxiv.org/html/2608.18089)
  - *Did / found:* clamps an English refusal direction at layer 12 on Llama-3-8B, Llama-3.1-70B, Mistral-7B and Qwen2.5-7B, and recovers refusals in Yoruba, Igbo, Igala and Hausa. Gains depend on the unsteered baseline (ceiling effects below about 15%, inverse transfer above about 60%). Arabic fails everywhere. "Refusal Centroid Drift" (the cosine distance between English and target harmful centroids) is 0.95 for Yoruba and Swahili. Single author, single annotator.
  - *Relation:* a related geometric metric. It uses centroid distance rather than projection.
- **[V] Kompella, R. P., Mahajan, A. (2026). "Decided Upstream, Written Late: Locating and Pricing the Cross-Lingual Refusal Circuit of a Multilingual MoE." Actionable Interpretability workshop @ COLM 2026. arXiv:2608.08032.** — [arXiv](https://arxiv.org/abs/2608.08032)
  - *Did / found:* studies the Indic-multilingual Sarvam MoE. The harm direction is nearly language-invariant mid-network (English vs Indic cosine about 0.9 at L11), and steering it controls refusal. The refusal is "written late," during generation, by a direction orthogonal to the detection direction, with an attention "opposer" holding back the writer. The specific Indic languages are not named on the abstract page.
  - *Relation:* this is the only Indic-language mechanistic refusal paper I found. Its distinction between detection and writing parallels Zhao et al. and is a caution for the study's single-token readout.
- **[V] Oppong, A. et al. (2026). "The Illusion of Cross-Lingual Safety in Low-Resource Languages." arXiv:2608.11146** (11 Aug 2026; LoDNA dataset; Twi, Hausa, Amharic, Swahili). — [arXiv](https://arxiv.org/abs/2608.11146)
  - *Found:* harmful prompts "retain less than 10% of the English refusal signal across most language-model pairs." Representations drift across layers despite high semantic similarity. The authors argue against a universal harm representation.
  - *Relation:* this is the strongest counter-claim to Wang et al. and supports expecting a weaker Hindi projection.
- **[V] Upadhyaya, A., Sikdar, S. (2026). "When Safety Speaks a Language: A Mechanistic Analysis of Safety-Language Identity Entanglement in LLMs." arXiv:2608.29936.** — [arXiv](https://arxiv.org/abs/2608.29936)
  - *Found:* using SAEs over 3 instruction-tuned models and 8 languages (the languages are not listed on the abstract page), safety features are entangled with language identity, and universality depends on architecture.
  - *Relation:* this suggests results may differ between Qwen3 and Llama, which supports the study's two-model design.
- **[V] Miao, S. et al. (2026). "Who Bridges Safety? Identifying and Targeting Cross-Lingual Shared Safety Pathways." arXiv:2608.09095.** — [arXiv](https://arxiv.org/abs/2608.09095)
  - *Found:* shared cross-layer safety pathways carry safety from high-resource to non-high-resource languages. Pathway-only updates improve non-high-resource safety.
- **[V] Verma, N., Bharadwaj, M. (+1) (2025). "The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual Context." arXiv:2504.02708** (preprint). — [arXiv](https://arxiv.org/abs/2504.02708)
  - *Found:* alignment-induced separation in a "safety space" is weaker for low-resource languages. This is embedding-level, not a refusal direction. Aziz et al. dispute the overlap explanation. ([Aziz et al.](https://arxiv.org/html/2606.01196v1))
- **[V-meta] Ifebi, C., Kong, B., Mehrotra, A. (2026). "Minionese: Comprehensive Benchmark and Mechanistic Study of Multilingual LLM Safety." arXiv:2607.10112.** — [arXiv PDF](https://arxiv.org/pdf/2607.10112). The body could not be parsed, so its content is unverified.
- Behavioral multilingual-safety background, cited inside the papers above and not verified here: MultiJail (Deng et al. 2024); "All Languages Matter" (Wang et al., arXiv:2310.00905); Yong et al. 2023/2025 on low-resource jailbreaks. — via [BabelSteering](https://arxiv.org/html/2608.16577v1)

### Inferences
- To the key question "do they report harmful prompts in other languages separate less on the direction?":
  - Wang et al. report *weaker clustering* (silhouette), not lower projection.
  - Aziz et al. explicitly report *lower projection* onto the high-resource direction, with separability preserved.
  - Oppong et al. report less than 10% of the English refusal signal retained.
- So a finding that Hindi harmful prompts project less than English ones would *replicate* the direction of these results on a new language and model. The novel parts are Hindi, Qwen3-8B, same-request pairing and the English-only fit, and the study should frame them that way.
- Aziz et al.'s action-vs-representation distinction gives a natural framing. Lower projection alone does not show that Hindi harm is undetected. The study's AUROC or separability numbers for Hindi along the English axis would settle which case applies.

### Gaps
- I found no paper fitting refusal directions on Qwen3 models in a cross-lingual setting.
- I found no Hindi-specific projection or AUC numbers in the text of any paper. Aziz et al. may have them in Figure 5, which I could not read.
- The Kompella & Mahajan abstract does not name its Indic languages.
- Most 2026 papers are unreviewed preprints.

## Q4. Has prior work separated refusal *style* (explicit vs soft/evasive) as its own direction?

### Takeaway
I found no paper that fits a direction separating explicit from soft or evasive refusals. The nearest precedents are:
- refusal directions split by *category or reason* (Joad et al. 2026);
- a separate *false-refusal* direction (Wang et al. 2410.03415, BabelSteering);
- observations that refusal *framing* varies with the steering direction or between base and chat models.

The study's explicit-vs-soft direction appears to be novel. Phrase this as "to our knowledge."

### Cited Findings
- Joad et al. 2026 classify refusals into five *framing* types (capability limitation, safety/ethical, underspecification, generic, non-refusal) and find that the choice of steering direction changes *how* the model refuses. They do not treat soft, hedged or partial refusals as a category, and they attribute style to a long tail of SAE latents rather than to a dedicated direction. — [arXiv HTML v2](https://arxiv.org/html/2602.02132v2)
- The false-refusal vector, orthogonalized against the refusal direction (Wang et al. 2024, arXiv:2410.03415, used again in BabelSteering), is a second DiM direction distinct from refusal, but it separates false refusal from true refusal, not refusal styles. — [BabelSteering](https://arxiv.org/html/2608.16577v1)
- Base-model refusals differ in style from chat-model refusals (claimed incompetence or misunderstanding) yet are removed by the same direction. This is informal evidence that style varies within one mechanism. — [LessWrong, "Base LLMs refuse too"](https://www.lesswrong.com/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too)
- Zhao et al.'s refusal-vs-harmfulness split and Kompella & Mahajan's detection-vs-writing split are separations along other axes, not style. — [Zhao](https://arxiv.org/abs/2507.11878); [Kompella](https://arxiv.org/abs/2608.08032)
- A search for "soft refusal vs explicit refusal direction" returned no direct match. — (search this session)

### Inferences
- The study should cite Joad et al. (category directions) and the false-refusal direction as the closest relatives, and state that refusal *style* has not been isolated. The AUROC result would be a new observation.
- Wollschläger's representational-independence criterion suggests one check: is the style direction independent of the refusal direction under intervention, not merely low in cosine? Reviewers may ask.

### Gaps
- Behavioral literature on partial or hedged refusal taxonomies (for example, CoCoNot or other refusal taxonomies) was not covered in depth. I did not verify any paper that fits activation directions for "partial compliance."
- **[U] "Refusal Tokens" (Jain et al., arXiv:2412.06748, 2024)**, from memory, calibrates refusal categories with special tokens. It is not a direction and was not verified.
