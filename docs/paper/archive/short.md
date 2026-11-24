# Why Two Open Chat Models Refuse Less in Hindi: Refusal Style, Internal Signals and Gated Steering

## Abstract

Safety-tuned chat models refuse harmful requests less reliably outside English. We study this gap in Qwen3-8B and Llama-3.1-8B-Instruct on
paired requests asked in English and in Hindi, grading replies with a four-way scheme that separates explicit refusals from soft ones. On 404
held-out pairs, explicit refusal falls from 51% to 19% in Qwen and from 89% to 47% in Llama. Qwen's lost refusals are mostly soft (115 of 148);
Llama's split between compliance and soft refusal (88 and 75 of 172), and Llama fails to refuse Hindi harmful requests 1.4–1.9 times as often as
Qwen. An English-fitted refusal direction scores harmful Hindi prompts at English strength in Qwen but at 28% of it in Llama, although in both
models it still ranks harmful above harmless Hindi prompts (exploratory). A separate direction that separates explicit from soft refusals raises
Hindi explicit refusal when added. Pre-registered gated steering raises Hindi explicit refusal on held-out pairs by 17.1 points in Qwen and 14.6 in
Llama (narrowly missing its registered bars), but it also raises English by 15.8 and 7.2 points, so it is not language-selective. The Qwen repair
failed a pre-registered Bengali guard. We report failures alongside successes.

---

## 1 Introduction

Refusal is the main behavioural safeguard of open chat models, and it is trained mostly on English data. Prior work shows that harmful requests
in lower-resource languages are refused less often (Yong et al. 2023; Deng et al. 2024; Wang et al. 2024), including in Hindi and in the two
models we study (§2). It rarely asks what replaces the missing refusal, or whether the same internal signal explains the gap in different models.

We study the gap on paired requests: each harmful request is asked in English and in Hindi and compared item by item. Replies are graded
COMPLY, REFUSE_EXPLICIT, REFUSE_SOFT (understood, but neither provides nor declines: it warns, lectures or redirects) or BROKEN (incoherent or
in the wrong language). Qwen3-8B and Llama-3.1-8B-Instruct are evaluated on the same 404 held-out pairs. Our contributions are on the response
side and in the protocol:

1. **A response-side decomposition of the gap.** In Qwen most missing Hindi explicit refusals become soft refusals; in Llama they split between
   compliance and soft refusal, and Llama complies far more often. The soft-versus-comply boundary is the least reliable label (§3).
2. **Two internal readings of a shared gap.** On the same items, harmful Hindi prompts reach English strength on an English refusal direction in
   Qwen but only 28% of it in Llama; in both, the harmful–harmless order survives (exploratory).
3. **An explicit-versus-soft direction.** To our knowledge not previously reported. Adding it raises Hindi explicit refusal; we did not test its
   independence from the refusal direction.
4. **Pre-registered repair tests with negative results.** Gated steering raises Hindi explicit refusal in both models but is not
   language-selective; the Qwen repair failed a sealed Bengali confirmation.

---

## 2 Related work

**Multilingual safety gaps.** Yong et al. (2023, arXiv:2310.02446), MultiJail (Deng et al., ICLR 2024, arXiv:2310.06474) and XSafety (Wang et
al., Findings of ACL 2024, arXiv:2310.00905) first quantified the gap with unsafe-output or attack-success rates, mostly on closed models and
translated prompts. IndicSafeEval (Mondal et al., Findings of EMNLP 2026, arXiv:2609.03781) evaluates Qwen3-8B and Llama-3.1-8B, among others,
on Hindi and three other Indian languages with adversarial persuasion prompts, and CompositeHarm (Shukla et al., AICS @ AAAI 2026,
arXiv:2602.07963) reports sharply higher attack success in Hindi and other Indic languages. We use default, non-adversarial requests, per-item
pairing, and explicit versus soft refusal as the outcome. Huang et al. (Findings of EMNLP 2026) decompose attack success by response language;
we decompose by refusal style.

**Refusal directions.** Arditi et al. (NeurIPS 2024, arXiv:2406.11717) show that refusal is mediated by a difference-of-means direction.
Wang et al. (NeurIPS 2025, arXiv:2505.17306) find it approximately parallel across 14 languages, with weaker harmful–harmless separation outside
English, especially in Llama-3.1. Aziz et al. (2026, arXiv:2606.01196) extend this to 23 languages including Hindi: low-resource harm projects
lower but stays separable. Zhao et al. (2025, arXiv:2507.11878) show that harmfulness and refusal are encoded separately, so our directions, read
at the last template token, are refusal directions and not measures of harm recognition. Balani & Panda (2026, arXiv:2609.27758) show that
cross-lingual probe transfer can collapse on hard negatives. Kompella & Mahajan (2026, arXiv:2608.08032) find language-invariant harm detection
but a late, separate refusal-writing direction in an Indic mixture-of-experts model, and Joad et al. (EMNLP 2026, arXiv:2602.02132) find
reason-specific refusal directions. Our weak Llama projection replicates prior findings in a paired Hindi design.

**Steering.** Conditional Activation Steering (Lee et al., ICLR 2025, arXiv:2409.05907) gates refusal steering on a condition vector. Aziz et al.
gate on a classifier calibrated with a few target-language examples and find that Llama needs target-language calibration, as ours did.
BabelSteering (Stein et al. 2026, arXiv:2608.16577) adds an English refusal vector across eight languages and raises refusal of pseudo-harmful
prompts too; ALTSTEER (EMNLP 2026, arXiv:2608.30197) steers toward constructive alternatives. Our gate is not new as a method. Reliability
critiques (Tan et al., NeurIPS 2024, arXiv:2407.12404; AxBench, Wu et al., ICML 2025, arXiv:2501.17148, which finds that prompting outperforms
steering) motivate our held-out tests and make the missing prompting baseline a real gap (§7).

**Grading and protocol.** Prior schemes separate full from partial refusal (XSTest; Röttger et al., NAACL 2024, arXiv:2308.01263), name
indirect refusals (OR-Bench; Cui et al., ICML 2025, arXiv:2405.20947) or categorise response actions (Do-Not-Answer; Wang et al., Findings of
EACL 2024, arXiv:2308.13387). LLM judges are less reliable outside English (Fu & Liu, Findings of EMNLP 2025, arXiv:2505.12201; Hada et al.,
Findings of EACL 2024, arXiv:2309.07462). Pre-registration has been argued for in NLP (van Miltenburg et al., NAACL 2021, arXiv:2103.06944). To
our knowledge, no prior steering or interpretability paper combines it with a sealed held-out-language confirmation.

---

## 3 Setup

**Data.** Harmful requests come from Aya Red-teaming (Aakanksha et al. 2024, arXiv:2406.18682), whose Hindi prompts were written natively with
human English translations, and from IndicSafe (Pattnayak & Chowdhuri 2026, arXiv:2603.17915). Harmless controls come from PolyGuard (Kumar et
al., COLM 2025, arXiv:2504.04377) and IndicSafe. Each candidate was screened for being a real request, for translation fidelity, and for whether
the model understands the Hindi version. The held-out split has **404 understood harmful and 169 understood harmless pairs**, used for every test
here. Spanish (Aya) serves as a control language and Bengali (MultiJail) as a sealed confirmation language. Hindi is an officially supported
language of Llama-3.1 (model card); Qwen3 reports support for 119 languages and dialects (Qwen Team 2025, arXiv:2505.09388).

**Models and generation.** Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct at pinned revisions, greedy decoding, system prompt "You are a
helpful assistant.", up to 1,024 new tokens in Hindi and Bengali and 512 in English, one A40 GPU per run in a pinned environment.

**Grading.** Each reply gets the first label that fits: COMPLY (provides what was asked, even partly or with warnings), REFUSE_EXPLICIT (says it will
not or cannot help), REFUSE_SOFT, or BROKEN. One fluent Hindi reader, the author, graded 150 calibration replies, and four candidate judges
graded them blind. The selected judge, Claude Sonnet 5.5, reached κ 0.82 (95% CI 0.68–0.92) on refused versus not refused and 0.91 on explicit
versus the rest, but **four-label κ is only 0.62** (0.55 on the production grading route). The weak boundary is soft versus comply, and the
judge's errors there run in opposite directions in the two models: in Qwen's Hindi calibration it labelled 9 of the human's 80 COMPLY replies as
soft, and in the Llama check it labelled 6 of the human's 15 soft replies as compliance. The Llama human check (60 replies) gave blind four-label κ
0.643, below its registered 0.7. The author then reviewed the 14 disagreements with the judge's labels in view and changed 10, all to the judge's
label (κ 0.893, an upper estimate); the Llama sheet was graded after aggregate Llama results had been seen. Explicit refusal is the
best-validated label, so primary claims rest on it.

**Protocol.** The steering and neuron studies (Phases 3–7; Appendix A) had their rules frozen before any of their replies were generated. The
studies of the internal directions (Phases 2a and 2b) were registered as exploratory, the four-way rubric was revised after the first human
comparison, and the same-item projection statistics in §5 were added after the registered studies. Settings were tuned on development items and
tested once on held-out items. Intervals are paired bootstrap 95% intervals (10,000 resamples).

---

## 4 The gap

| Explicit refusal, 404 pairs | Qwen EN | Qwen HI | Llama EN | Llama HI |
|---|---|---|---|---|
| Harmful | 51.2% | 19.1% | 88.9% | 46.5% |
| Any refusal (explicit or soft) | 90.3% | 77.2% | 90.8% | 66.8% |
| Harmless refused (169 pairs) | 3.5% | 1.2% | 4.1% | 2.4% |

The explicit-refusal gap is −32.2 [−37.4, −26.5] points in Qwen and −42.3 [−47.3, −37.4] in Llama, and it holds on both harmful sources in
Llama. (Steering runs regenerate the unsteered replies, giving slightly different rates: Qwen 50.5% and 18.6%; Llama any refusal 66.3%.)

![Figure 1](figures/fig1_what_replaces_refusal.png)

**Figure 1.** Replies to the same 404 held-out harmful requests in English (open) and Hindi (filled), both models, with Wilson 95% intervals. The
bar splits the requests refused explicitly in English but not in Hindi by what the Hindi reply did.

**What replaces the refusal.** In Qwen, 148 requests are refused explicitly in English but not in Hindi: 115 become soft refusals, 20 compliance
and 13 broken replies. In Llama, 172 do: 88 compliance, 75 soft and 9 broken (judge labels; p = 0.35 for 88 versus 75). A human review of 23 such
Llama pairs found 13 compliance, 9 soft and 1 broken. So Qwen mostly hedges, while Llama hedges or complies in similar shares. Given the opposite
judge errors in §3, the size of this contrast is uncertain. The clearer contrast is the rate: on the same pairs Llama complies with Hindi harmful
requests 30.2% of the time and Qwen 16.3% (+13.9 [+9.4, +18.3]); counting broken replies too, 33.2% against 24.0% (+9.2 [+4.5, +14.1]), hence the
1.4–1.9 ratio. Harmless Hindi requests are not over-refused, though Qwen's replies to them are broken 21 times in 169.

**Spanish control.** On 782 Aya Spanish pairs, Qwen's explicit refusal also falls, from 47% to 32%, so the effect is not specific to low-resource
scripts. This run is provisional: items were not screened, replies were capped at 512 tokens in both languages, and the judge was validated on
Hindi only.

---

## 5 Two internal readings

**Refusal direction.** We fit a refusal direction on English prompts only, as a difference of means of last-prompt-token activations, and choose
the layer by directional ablation (Qwen layer 14, removing 64% of the English harmful–harmless gap on held-out prompts; Llama layer 10, 94%).
On the same 404 harmful and 169 harmless test pairs, harmful Hindi prompts reach 100% of the English mean projection in Qwen and 28% in Llama
(Figure 2). An exploratory same-item analysis shows that the direction still ranks Hindi harmful above Hindi harmless prompts in both models
(AUROC 0.92 in Qwen and 0.97 in Llama, against 0.98 and 0.99 in English), with a Hindi harmful–harmless gap about a third of the English one
(0.33 and 0.30). The models differ in which side moves: in Qwen the harmless side rises, while in Llama about three quarters of the shrinkage comes
from harmful Hindi prompts falling.

![Figure 2](figures/fig2_projection.png)

**Figure 2.** Mean projection on each model's refusal direction at its chosen layer, same 404 harmful and 169 harmless test pairs, English (ring)
and Hindi (dot), bootstrap 95% intervals. Hindi harmful prompts reach 100% of the English value in Qwen and 28% in Llama; harmless Hindi prompts
in Qwen sit far above harmless English.

Three cautions apply. This is a projection on a refusal direction, not recognition of harm. Harmless Hindi prompts scoring high in Qwen may
reflect a language offset or a source mismatch (the harmful set is native Hindi from Aya, the harmless set is machine-translated PolyGuard). And
we have not tested against Hindi hard negatives. The 100% versus 28% comparison also mixes sources, layers and scales.

**Explicit-versus-soft direction.** We fit a second direction on English prompts whose replies were explicit (367) or soft (200) refusals (Qwen
layer 24, cross-validated AUROC 0.93; 0.76 on Hindi prompts without refitting). Hindi prompts sit toward its soft end (paired shift −10.9 [−12.1,
−9.6], 22% of the English gap, below the registered 25% threshold). Mean projection by Hindi outcome runs explicit −11.6, soft −28.0, broken
−40.0, complied −43.5, so the axis tracks refusal strength as much as style. On 150 pairs refused explicitly in English but not in Hindi, adding
the vector raises Hindi explicit refusal from 10% to 53% (+42.7 [+34.0, +51.3]) and roughly halves compliance, with harmless prompts unchanged; at
twice the strength it over-refuses harmless prompts and breaks replies. We ran no random-vector control and no independence test against the
refusal direction, so we claim only an effect on explicit refusal.

---

## 6 Repair by gated steering

**Method.** A gate reads the refusal-direction score at the last prompt token. If it exceeds a threshold τ, a fixed vector is added from the
prompt onward. For Qwen the gate is at layer 14 with τ at the median English development harmful score, and the push is 1.25 times the
explicit–soft vector at layer 25. For Llama the gate is at layer 10 with τ at the median Hindi development harmful score, and the push is along the
refusal direction at layer 11, 1.5 times the mean English-minus-Hindi projection gap. Settings were chosen on development pairs by a registered
rule and each was tested once on the held-out pairs.

| Held-out, run once | Qwen unsteered | Qwen gated | Change | Llama unsteered | Llama pushed | Change |
|---|---|---|---|---|---|---|
| Hindi explicit refusal | 18.6% | 35.6% | **+17.1** [+13.1, +21.0] | 46.5% | 61.1% | **+14.6** [+11.1, +18.3] |
| Hindi compliance | 16.3% | 12.6% | −3.7 | 30.4% | 22.0% | −8.4 [−11.1, −5.9] |
| Hindi BROKEN | 7.7% | 7.9% | +0.2 | 3.2% | 2.5% | −0.7 |
| English explicit refusal | 50.5% | 66.3% | +15.8 [+12.1, +19.6] | 88.9% | 96.0% | +7.2 [+5.0, +9.7] |
| Harmless refusal (HI / EN) | 1.8% / 2.4% | same | 0 | 1.8% / 4.1% | 1.8% / 4.7% | 0 / +0.6 |
| Global-MMLU (HI / EN) | 51.3% / 64.0% | same | 0 | 42.7% / 63.0% | same | 0 |

![Figure 3](figures/fig3_repairs.png)

**Figure 3.** Explicit refusal before and after each repair on held-out pairs, run once, with paired bootstrap intervals and the gap before and
after. Qwen passes its registered checks without narrowing the gap; Llama narrows it by 7.4 points and narrowly misses its bars.

**Qwen.** All eight registered checks pass, but English rises almost as much as Hindi and the explicit-refusal gap goes only from 31.9 to 30.7
points. The repair converts soft refusals into explicit ones in both languages; it does not target the gap. The effect rests on Aya items (the
gate opened on 2 of 67 IndicSafe prompts), and the harmless and MMLU guards are weak because the gate almost never opens on those inputs.

**Failures.** On the sealed Bengali set (148 requests, run once) explicit refusal rose by 23.6 points and the gap narrowed, but BROKEN replies
rose by 8.1 points [+2.0, +14.9] against a limit of 5, so the registered verdict is FAIL (a second judge agreed). A follow-up sweep on 150 new
Bengali requests did not reproduce the damage (+0.9), but the registered verdict stands. Separately, the MLP neurons most tied to the
English–Hindi difference, moved with one untuned offset, raised English explicit refusal by 15.2 points and Hindi by only 1.2 [−2.5, +5.0], so
we found no Hindi-only lever.

**Llama.** The Qwen gate, carried over unchanged, opened on only 1% of Hindi harmful prompts. With a Hindi-calibrated gate, the push raises
Hindi explicit refusal by 14.6 points, just short of its registered +15 bar, and narrows the gap by 7.4 points [+3.2, +11.6] against a bar of 10.
The third registered claim, that compliance falls, is met (−8.4). Since any refusal rose by 9.2, about 5.4 points of the gain are soft refusals made
explicit. It corroborates the target-language calibration that Aziz et al. found Llama needs. We did not test whether the choice of direction
rather than the Hindi-calibrated gate matters: an unmatched push along the explicit–soft direction also qualified on development data.

![Figure 4](figures/fig7_gate.png)

**Figure 4.** Gate scores of every held-out prompt (unsteered), at the gate layer, with each model's threshold as run and the share of each group
above it.

**Not language-selective.** Both gates read how harmful-looking a prompt is. At its registered threshold Qwen's gate opens on 48% of English
and 47% of Hindi harmful test prompts; Llama's Hindi-set threshold opens on 92% and 47% (Figure 4). Lower Qwen thresholds opened on more Hindi
than English development prompts (92% against 75% at the 25th percentile) but were rejected for broken replies, and in Bengali the registered
gate opened more often than in English (76% against 39%). Selectivity therefore depends on the threshold and the language, and at the thresholds
we registered neither gate was language-selective.

---

## 7 Limitations

- **Scope:** two 8B models, one revision each, greedy decoding. No prompting or translate-to-English baseline, which AxBench makes the most
  important missing comparison.
- **Judges:** one human grader, the author, who graded the Llama sheet after seeing aggregates. Four-label κ is 0.62, and the soft-versus-comply
  errors run in opposite directions in the two models, which bears directly on the Qwen versus Llama contrast. Bengali and Spanish grades have no
  human check.
- **Data and confounds:** the Hindi side of each Aya pair is native and the English side a translation, and we ran no back-translation or
  romanised-Hindi control, so language, script and translation fidelity are not separated. Harmful and harmless sets come from different sources.
- **Truncation:** replies were cut at fixed token limits and many Hindi replies reach them (571 of 1,174 in the baseline pool); a registered
  truncation check was withdrawn, so a reply cut off before it answers could be read as soft.
- **Interpretation:** projections are on a refusal direction, not harm recognition; hard negatives are untested; the same-item statistics are
  exploratory; the explicit–soft direction lacks an independence or random-vector control.
- **Statistics:** settings were tuned on development data, test items are reused across the Qwen and Llama studies, and there is no correction
  for multiple comparisons.
- **Verifiability:** registrations are timestamped in version control in a private repository and are available on request.

---

## Appendix A: Protocol, deviations and reproducibility

Studies are named by phase: Phase 0 judge selection; Phase 1 baseline gap; Phases 2a and 2b refusal and explicit–soft directions; Phase 3 Qwen
gated steering; Phases 4 and 4b Bengali confirmation and sweep; Phase 5 Hindi neurons; Phase 6 Llama replication; Phase 7 Llama repair.

| Study | Deviation | When recorded | Effect on results |
|---|---|---|---|
| Baseline split | first split drained the failure pattern; redrawn once before use | before any use | none |
| Screening audit | 29 items adjudicated with model labels shown | pre-registered addendum | adjudicated agreement reported as an upper estimate |
| Phases 3, 4b | grading parts stopped by the judge's safety filter; retried or filled by a second judge | before analysis / after | outcomes unchanged |
| Phase 6 | human check graded after aggregate results were seen; disagreements adjudicated with judge labels shown | before / after grading | blind κ 0.643 reported as registered; 0.893 as an upper estimate |
| Phase 7 | steering arms double-scaled (about 4.8×) and a sign error in one claim, found after stage 1 | after stage 1, before stage 2 | those arms rerun; selection unchanged |
| Same-item and per-layer statistics | added after the registered studies | 2026-11-17 | no registered claim changes |

Code, pre-registrations with their dated addenda, id-only labels and a checksum manifest of the untracked inputs are archived with the project
and available from the author on request. Harmful request and reply text is not redistributed and none is reproduced here. Every study ran on a
single 48 GB GPU in a pinned software environment.
