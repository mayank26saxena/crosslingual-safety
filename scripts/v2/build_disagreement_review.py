"""Build the human-vs-judge disagreement sheet for the Llama human checks (Phase 6 and Phase 7).

Usage: python3 -I scripts/v2/build_disagreement_review.py
Writes notes/review/llama_disagreements.csv (not tracked; it holds request and reply text), for tools/review_disagreements.html.
Each row is an item where the author's label and the judge's label differ, with the judge's one-line reason.
The reviewed file (llama_disagreements_reviewed.csv) is an adjudication made with the judge's label in view: any agreement computed from
it is an upper estimate and is reported next to the registered, blind value (as for the screening audit). Text is never printed.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "notes/grading"


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def judge_index(folder):
    """uid|condition -> (label, reason), following copied rows to the reply that was actually graded."""
    key = json.load(open(G / folder / "key.json"))
    graded = {}
    for f in sorted((G / folder / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
        for r in rows(f):
            k = key["graded"][r["item"].strip()]
            graded[f'{k["uid"]}|{k["condition"]}'] = (r["label"].strip(), r.get("reason", "").strip())
    for a, b in key.get("same_as", {}).items():
        if b in graded:
            graded[a] = (graded[b][0], graded[b][1] + " (label copied from an identical or gate-closed unsteered reply)")
    return graded


def human(sheet):
    out = []
    for r in rows(sheet):
        lab = next(v for k, v in r.items() if k.startswith("HUMAN_label")).strip().upper()
        out.append((r, lab))
    return out


def main():
    j6 = judge_index("phase6B")
    k6 = json.load(open(G / "phase6B_human/key.json"))
    j7_key = json.load(open(G / "phase7_human/key.json"))
    # Phase 7 judge labels: graded stage 2 rows, copies, and Phase 6 labels for identical unsteered replies
    j7 = judge_index("phase7_2")
    out = []
    for r, lab in human(G / "phase6B_human/phase6_llama_human_graded.csv"):
        uid = k6[r["item"]]
        jl, why = j6[uid + "|hi_unsteered"]
        if jl != lab:
            out.append(dict(phase="Phase 6 (unsteered)", item=r["item"], set=uid.split("|")[0].replace("test_", ""), uid=uid + "|hi_unsteered",
                            request_english=r["request_english"], message_hindi=r["message_hindi"], reply_hindi=r["reply_hindi"],
                            your_label=lab, judge_label=jl, judge_reason=why))
    for r, lab in human(G / "phase7_human/phase7_llama_human_graded.csv"):
        uc = j7_key[r["item"]]
        jl, why = j7[uc]
        if jl != lab:
            out.append(dict(phase="Phase 7 (steered R3)", item=r["item"], set=uc.split("|")[0].replace("test_", ""), uid=uc,
                            request_english=r["request_english"], message_hindi=r["message_hindi"], reply_hindi=r["reply_hindi"],
                            your_label=lab, judge_label=jl, judge_reason=why))
    for i, d in enumerate(out, 1):
        d["review_id"] = f"D{i:02d}"
        d["REVIEW_label"] = ""
        d["REVIEW_note"] = ""
    cols = ["review_id", "phase", "item", "set", "uid", "request_english", "message_hindi", "reply_hindi", "your_label", "judge_label",
            "judge_reason", "REVIEW_label", "REVIEW_note"]
    dest = ROOT / "notes/review/llama_disagreements.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(f"{len(out)} disagreements -> {dest.relative_to(ROOT)}")
    print(dict(Counter((d["phase"].split()[1], d["your_label"] + "->" + d["judge_label"]) for d in out)))


def analyse():
    """After review: adjudicated agreement (upper estimate) next to the registered blind value, and the reviewed harmful breakdown."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from collections import Counter
    from phase6_llama import _kappa
    rev = {r["uid"]: r["REVIEW_label"].strip().upper() for r in rows(ROOT / "notes/review/llama_disagreements_reviewed.csv")
           if r["REVIEW_label"].strip()}
    j6 = judge_index("phase6B")
    k6 = json.load(open(G / "phase6B_human/key.json"))
    hum, mod, harm = [], [], Counter()
    for r, lab in human(G / "phase6B_human/phase6_llama_human_graded.csv"):
        uc = k6[r["item"]] + "|hi_unsteered"
        final = rev.get(uc, lab)
        hum.append(final)
        mod.append(j6[uc][0])
        if uc.startswith("test_harmful"):
            harm[final] += 1
    changed = sum(1 for r, lab in human(G / "phase6B_human/phase6_llama_human_graded.csv")
                  if rev.get(k6[r["item"]] + "|hi_unsteered", lab) != lab)
    print(f"Phase 6: {changed} of the author's labels changed on review")
    print(f"adjudicated four-label kappa {_kappa(hum, mod):.3f} (upper estimate; registered blind value 0.643 stands)")
    print("reviewed author labels, 45 harmful:", dict(harm))


if __name__ == "__main__":
    import sys
    analyse() if sys.argv[1:] == ["analyse"] else main()
