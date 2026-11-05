"""Batch translation with Qwen2.5-14B-Instruct for Step 1.

xstest   rebuild XSTest in Bengali (250 safe, 200 unsafe contrast) and back-translate every prompt. Output with a format
         fault (commentary, low Bengali-script share, excess length) is retried by sampling, up to three times; items
         still failing are recorded with format_ok=false and excluded from the guard.
back     back-translate an existing Bengali XSTest file and add format checks (translator stays as recorded).
replies  translate Bengali replies to English, for the translate-then-judge arm.

Usage:
  python scripts/translate_batch.py xstest --out data/rendered/xstest_v2_bn.jsonl
  python scripts/translate_batch.py replies --input results/step1/translate_arm_replies.jsonl \
      --out results/step1/translate_arm_en.jsonl
"""

import argparse
import re

import pandas as pd
import torch

from common import ROOT, load_model, read_jsonl, write_jsonl

MODEL = "Qwen/Qwen2.5-14B-Instruct"
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
TO_BN = ("You are a professional English-to-Bengali translator preparing a linguistics dataset. Translate the sentence "
         "below into natural Bengali, written in Bengali script. Keep its literal meaning, tone and every word that may "
         "sound alarming, including words used in a figurative or unusual sense. Do not answer it, comment on it or "
         "add notes. Output only the single Bengali sentence.\n\nSentence: {text}")
NAMES = {"bn": "Bengali", "zh": "Simplified Chinese", "ar": "Arabic", "sw": "Swahili"}
TO_LANG = ("You are a professional English-to-{lang} translator preparing a linguistics dataset. Translate the sentence below into "
           "natural {lang}. Keep its literal meaning, tone and every word that may sound alarming, including words used in a "
           "figurative or unusual sense. Do not answer it, comment on it or add notes. Output only the single translated sentence."
           "\n\nSentence: {{text}}")
TO_EN_ANY = ("Translate the following text into English, literally and completely. Do not answer it, comment on it or add notes. "
             "Output only the English translation.\n\nText: {text}")
TO_EN = ("Translate the following Bengali text into English, literally and completely. Do not answer it, comment on it "
         "or add notes. Output only the English translation.\n\nText: {text}")
BENGALI = re.compile("[ঀ-৿]")
LATIN = re.compile("[A-Za-z]")
OTHER_SCRIPT = re.compile("[぀-ヿ一-鿿가-힯ऀ-ॿ]")
NOTE = re.compile(r"(\bnote\b|\bsorry\b|\btranslation\b|দ্রষ্টব্য|অনুবাদ)", re.I)


def format_ok(source, text):
    bn, lat = len(BENGALI.findall(text)), len(LATIN.findall(text))
    return (bool(text) and "\n" not in text and not NOTE.search(text) and not OTHER_SCRIPT.search(text)
            and bn / max(1, bn + lat) >= 0.5 and len(text) <= 3 * max(len(source), 20))


SCRIPT = {"zh": re.compile("[\u4E00-\u9FFF]"), "ar": re.compile("[\u0600-\u06FF]"), "bn": BENGALI}


def format_ok_lang(lang, source, text):
    if not text or "\n" in text or len(text) > 4 * max(len(source), 20) or re.search(r"\b(note|sorry|translation)\b", text, re.I):
        return False
    if lang in SCRIPT:
        native, lat = len(SCRIPT[lang].findall(text)), len(LATIN.findall(text))
        return native / max(1, native + lat) >= 0.5
    return not re.search("[\u0900-\u097F\u0980-\u09FF\u0600-\u06FF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["xstest", "xstest_lang", "back", "replies"])
    ap.add_argument("--input")
    ap.add_argument("--lang")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok, model = load_model(MODEL)

    def run(texts, template, max_new_tokens, sample=False):
        outputs = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), args.batch_size):
            idx = order[start: start + args.batch_size]
            prompts = [tok.apply_chat_template([{"role": "user", "content": template.format(text=texts[i])}],
                                               tokenize=False, add_generation_prompt=True) for i in idx]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
            for b, i in enumerate(idx):
                outputs[i] = tok.decode(gen[b, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(f"  {min(start + args.batch_size, len(order))}/{len(order)}", flush=True)
        return outputs

    if args.mode == "xstest_lang":
        df = pd.read_csv(ROOT / "data/external/xstest/xstest_prompts.csv")
        sources = [str(p) for p in df["prompt"]]
        template = TO_LANG.format(lang=NAMES[args.lang])
        translations = run(sources, template, 200)
        retries = [0] * len(sources)
        for attempt in range(1, 4):
            bad = [i for i, (a, b) in enumerate(zip(sources, translations)) if not format_ok_lang(args.lang, a, b)]
            print(f"attempt {attempt}: {len(bad)} format faults", flush=True)
            if not bad:
                break
            for i, t in zip(bad, run([sources[i] for i in bad], template, 200, sample=True)):
                translations[i], retries[i] = t, attempt
        back = run(translations, TO_EN_ANY, 200)
        rows = []
        for k in range(len(sources)):
            r = df.iloc[k]
            task = f"xstest_{int(r['id']):03d}"
            rows.append(dict(task_id=task, id=f"{task}__{args.lang}", lang=args.lang, label=str(r["label"]), category=str(r["type"]),
                             focus=str(r["focus"]), source_en=sources[k], back_translation=back[k], translator=MODEL, retries=retries[k],
                             format_ok=format_ok_lang(args.lang, sources[k], translations[k]),
                             messages=[SYSTEM, {"role": "user", "content": translations[k]}]))
        write_jsonl(args.out, rows)
        print(f"{len(rows)} items -> {args.out}; format_ok {sum(r['format_ok'] for r in rows)}")
        return

    if args.mode == "back":
        rows = read_jsonl(args.input)
        back = run([r["messages"][1]["content"] for r in rows], TO_EN, 200)
        for r, b in zip(rows, back):
            r.update(back_translation=b, format_ok=format_ok(r["source_en"], r["messages"][1]["content"]))
        write_jsonl(args.out, rows)
        print(f"{len(rows)} items -> {args.out}; format_ok {sum(r['format_ok'] for r in rows)}")
        return

    if args.mode == "replies":
        rows = read_jsonl(args.input)
        english = run([r["response"] for r in rows], TO_EN, 500)
        write_jsonl(args.out, [{**r, "response_bn": r["response"], "response": e} for r, e in zip(rows, english)])
        print(f"{len(rows)} replies -> {args.out}")
        return

    df = pd.read_csv(ROOT / "data/external/xstest/xstest_prompts.csv")
    ids = [int(i) for i in df["id"]]
    sources = [str(p) for p in df["prompt"]]
    translations = run(sources, TO_BN, 200)
    attempts = [0] * len(ids)
    for attempt in range(1, 4):
        bad = [i for i, (s, t) in enumerate(zip(sources, translations)) if not format_ok(s, t)]
        print(f"attempt {attempt}: {len(bad)} format faults", flush=True)
        if not bad:
            break
        retried = run([sources[i] for i in bad], TO_BN, 200, sample=True)
        for i, t in zip(bad, retried):
            translations[i], attempts[i] = t, attempt
    back = run(translations, TO_EN, 200)

    rows = []
    for k, i in enumerate(ids):
        r = df.iloc[k]
        task_id = f"xstest_{i:03d}"
        rows.append(dict(task_id=task_id, id=f"{task_id}__bn", lang="bn", label=str(r["label"]), category=str(r["type"]),
                         focus=str(r["focus"]), source_en=sources[k], back_translation=back[k], translator=MODEL,
                         retries=attempts[k], format_ok=format_ok(sources[k], translations[k]),
                         messages=[SYSTEM, {"role": "user", "content": translations[k]}]))
    write_jsonl(args.out, rows)
    print(f"{len(rows)} items -> {args.out}; format_ok {sum(r['format_ok'] for r in rows)}")


if __name__ == "__main__":
    main()
