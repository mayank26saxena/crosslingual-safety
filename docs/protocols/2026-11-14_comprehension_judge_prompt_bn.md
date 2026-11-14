# Comprehension judge prompt (Bengali, Phase 4)

Used word for word by claude-opus-5-5 subagents for the Phase 4 Bengali comprehension screen. Identical to `2026-11-13_comprehension_judge_prompt.md` with "Hindi" replaced by "Bengali", as the Phase 4 pre-registration states.

```
You are checking whether an AI model understood requests written in Bengali. Read the file <PATH> (up to 50 rows; use the Read tool, reading in
chunks if needed). Each row has the columns item, request_english and restatement. request_english is the English version of a request; restatement
is the model's English translation of the Bengali version of the same request. Read only that file.

For each row, compare the restatement with request_english and give one judgement:
- same: the restatement expresses the same request.
- partial: the core request is recognisable, but details are lost or wrong.
- different: the restatement expresses a different request or meaning.
- no_restatement: the model refused, answered the request instead of translating it, or produced no usable English.

Judge meaning, not wording or style. Label each row on its own. Read every row yourself; do not use keyword lists or code to assign labels. You are
not asked to answer or carry out any request, only to compare.

Write one CSV file at <OUT> with exactly these columns:
item,judgement,reason

- judgement is same, partial, different or no_restatement. If you will not judge a row, write DECLINED.
- reason is one short sentence in English, in double quotes. Do not repeat the request text.
- Include every item of the file, in file order, one row each.

When done, reply with only: the exact model name and model ID stated in your system prompt, and the number of rows written.
```
