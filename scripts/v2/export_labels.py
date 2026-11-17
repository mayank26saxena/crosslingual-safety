"""Export every human and model label to tracked, id-only tables (no request, reply or reason text).

Usage: python3 scripts/v2/export_labels.py
Writes results/v2/labels/*.csv. Each row is keyed by the dataset item id (and language where relevant), so the results can be recomputed from
these tables plus the public datasets, without the untracked notes/ folder.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N = ROOT / "notes"
OUT = ROOT / "results/v2/labels"


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def write(name, fields, data):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)
    print(f"{name}: {len(data)}")


def blind_run(run_csv, key, grader, extra=None):
    out = []
    for r in rows(run_csv):
        k = key[r["item"].strip()]
        d = {"item_id": k.get("pair") or k.get("item_id"), "lang": k.get("lang", ""), "grader": grader, "label": r["label"].strip()}
        if extra:
            d.update({x: k.get(x, "") for x in extra})
        out.append(d)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Phase 0: human four-way grades on the 150 Hindi calibration items
    ck = json.load(open(N / "hindi_calibration/key.json"))
    col = "HUMAN_label (COMPLY/REFUSE_EXPLICIT/REFUSE_SOFT/BROKEN)"
    write("calibration_human.csv", ["calib_id", "source_id", "half", "benchmark_label", "label"],
          [{"calib_id": r["item"], "source_id": ck[r["item"]]["source_id"], "half": ck[r["item"]]["half"],
            "benchmark_label": ck[r["item"]]["label"], "label": r[col].strip()}
           for r in rows(N / "hindi_calibration/human/hindi_calibration_1024_human.csv")])
    # Phase 0: model graders on the calibration sheet (Kiro runs and the subagent bridge run)
    rk = json.load(open(N / "hindi_calibration/model_grading_key_1024.json"))
    cal = []
    for g in ("claude-haiku-5.5", "claude-sonnet-5.5", "gpt-5.6-terra", "gpt-5.6-luna"):
        cal += blind_run(N / f"hindi_calibration/model_runs/{g}/r1.csv", rk, f"{g} (kiro)")
    cal += blind_run(N / "hindi_calibration/kiro_grading/runs/claude-sonnet-5.5-cc/r1.csv", rk, "claude-sonnet-5.5 (subagent)")
    for d in cal:
        d["calib_id"] = d.pop("item_id")
    write("calibration_models.csv", ["calib_id", "lang", "grader", "label"], cal)
    # Hindi baseline grades (tier A + PolyGuard), selected grader
    hk = json.load(open(N / "grading/hindi_tierA_pg/key.json"))
    write("hindi_baseline_grades.csv", ["item_id", "lang", "grader", "label"],
          blind_run(N / "grading/hindi_tierA_pg/kiro_grading/runs/claude-sonnet-5.5-cc/r1.csv", hk, "claude-sonnet-5.5 (subagent)"))
    # Spanish grades
    sk = json.load(open(N / "grading/spanish_aya_pg/key.json"))
    sp = []
    for g in ("claude-sonnet-5.5", "gpt-5.6-luna"):
        sp += blind_run(N / f"grading/spanish_aya_pg/kiro_grading/runs/{g}/r1.csv", sk, f"{g} (kiro)")
    write("spanish_grades.csv", ["item_id", "lang", "grader", "label"], sp)
    # Phase 2b steering grades
    ek = json.load(open(N / "grading/es_dir_steer/key.json"))
    st = []
    for r in rows(N / "grading/es_dir_steer/runs/claude-sonnet-5.5-cc/r1.csv"):
        k = ek[r["item"].strip()]
        st.append({"item_id": k["pair"], "set": k["set"], "alpha": k["alpha"], "label": r["label"].strip()})
    write("steering_grades.csv", ["item_id", "set", "alpha", "label"], st)
    # Phase 3 tuning and test grades (graded rows plus steered rows that inherit the unsteered label)
    for stage in ("tune", "test"):
        gdir = N / f"grading/phase3_{stage}"
        if not any((gdir / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
            continue
        pk = json.load(open(gdir / "key.json"))
        p3 = {}
        for f in sorted((gdir / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
            for r in rows(f):
                k = pk["graded"][r["item"].strip()]
                p3[f'{k["uid"]}|{k["condition"]}'] = dict(item_id=k["pair"], set=k["set"], lang=k["lang"], condition=k["condition"],
                                                          label=r["label"].strip(), inherited="no")
        for a, b in pk["same_as"].items():
            uid, cond = a.rsplit("|", 1)
            p3[a] = dict(p3[b], condition=cond, inherited="yes")
        write(f"phase3_{stage}_grades.csv", ["item_id", "set", "lang", "condition", "label", "inherited"],
              sorted(p3.values(), key=lambda d: (d["set"], d["lang"], d["item_id"], d["condition"])))
    # Phase 4: reply grades (Sonnet, with inherited rows), second-grader sample (Opus), Bengali comprehension judgements
    g4 = N / "grading/phase4"
    if any((g4 / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
        for gdir, run, name, inherit in ((g4, "claude-sonnet-5.5-cc", "phase4_grades.csv", True),
                                         (N / "grading/phase4_opus", "claude-opus-5.5-cc", "phase4_second_grader.csv", False)):
            if not any((gdir / "runs" / run).glob("part_*.csv")):
                continue
            pk = json.load(open(gdir / "key.json"))
            p4 = {}
            for f in sorted((gdir / "runs" / run).glob("part_*.csv")):
                for r in rows(f):
                    k = pk["graded"][r["item"].strip()]
                    p4[f'{k["uid"]}|{k["condition"]}'] = dict(item_id=k["pair"], set=k["set"], lang=k["lang"], condition=k["condition"],
                                                              label=r["label"].strip(), inherited="no")
            if inherit:
                for a, b in pk["same_as"].items():
                    p4[a] = dict(p4[b], condition=a.rsplit("|", 1)[1], inherited="yes")
            write(name, ["item_id", "set", "lang", "condition", "label", "inherited"],
                  sorted(p4.values(), key=lambda d: (d["set"], d["lang"], d["item_id"], d["condition"])))
        ck = json.load(open(N / "screening/comprehension_bn/key.json"))
        cj = sorted(({"set": ck[r["item"].strip()].split("/")[0], "item_id": ck[r["item"].strip()].split("/", 1)[1],
                      "judgement": r["judgement"].strip()}
                     for f in sorted((N / "screening/comprehension_bn/runs/opus").glob("part_*.csv")) for r in rows(f)),
                    key=lambda d: (d["set"], d["item_id"]))
        write("phase4_comprehension_bn.csv", ["set", "item_id", "judgement"], cj)
    # Phase 4b sweep grades (graded rows plus copied rows whose source is graded)
    g4b = N / "grading/phase4b"
    if any((g4b / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
        pk = json.load(open(g4b / "key.json"))
        sw = {}
        for f in sorted((g4b / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
            for r in rows(f):
                k = pk["graded"][r["item"].strip()]
                sw[f'{k["uid"]}|{k["condition"]}'] = dict(item_id=k["pair"], alpha=k["alpha"], label=r["label"].strip(), inherited="no")
        for a, b in pk["same_as"].items():
            if b in sw:
                sw[a] = dict(sw[b], alpha=float(a.rsplit("_", 1)[1]), inherited="yes")
        write("phase4b_sweep_grades.csv", ["item_id", "alpha", "label", "inherited"],
              sorted(sw.values(), key=lambda d: (d["item_id"], d["alpha"])))
        op = [dict(item_id=pk["graded"][r["item"].strip()]["pair"], alpha=pk["graded"][r["item"].strip()]["alpha"], label=r["label"].strip())
              for f in sorted((g4b / "runs/claude-opus-5.5-cc").glob("part_*.csv")) for r in rows(f)]
        if op:
            write("phase4b_sweep_opus_parts.csv", ["item_id", "alpha", "label"], sorted(op, key=lambda d: (d["item_id"], d["alpha"])))
    # Phase 5 Hindi-neuron behaviour grades (graded rows plus copied rows whose source is graded)
    g5 = N / "grading/phase5"
    if any((g5 / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
        k5 = json.load(open(g5 / "key.json"))
        p5 = {}
        for f in sorted((g5 / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
            for r in rows(f):
                k = k5["graded"][r["item"].strip()]
                p5[f'{k["uid"]}|{k["condition"]}'] = dict(set=k["set"], item_id=k["pair"], lang=k["lang"],
                                                          condition=k["condition"], label=r["label"].strip(), inherited="no")
        for a, b in k5["same_as"].items():
            if b in p5:
                p5[a] = dict(p5[b], condition=a.rsplit("|", 1)[1], inherited="yes")
        write("phase5_grades.csv", ["set", "item_id", "lang", "condition", "label", "inherited"],
              sorted(p5.values(), key=lambda d: (d["set"], d["item_id"], d["lang"], d["condition"])))
    # Phase 6 Llama-3.1-8B grades: stage A (English fit replies) and stage B (test replies, with copied gated rows)
    for stage in ("phase6A", "phase6B"):
        g6 = N / f"grading/{stage}"
        if not any((g6 / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
            continue
        k6 = json.load(open(g6 / "key.json"))
        p6 = {}
        for f in sorted((g6 / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
            for r in rows(f):
                k = k6["graded"][r["item"].strip()]
                p6[f'{k["uid"]}|{k["condition"]}'] = dict(set=k["set"], item_id=k["pair"], lang=k["lang"],
                                                          condition=k["condition"], label=r["label"].strip(), inherited="no")
        for a, b in k6.get("same_as", {}).items():
            if b in p6:
                p6[a] = dict(p6[b], condition=a.rsplit("|", 1)[1], inherited="yes")
        write(f"{stage}_llama_grades.csv", ["set", "item_id", "lang", "condition", "label", "inherited"],
              sorted(p6.values(), key=lambda d: (d["set"], d["item_id"], d["lang"], d["condition"])))
    # Phase 7 Llama fix grades: stage 1 (dev), 1e (corrected E arms), 2 (test). Rows are graded, copied from an identical or gate-closed
    # unsteered reply ("same_as"), or take the Phase 6 label of an identical unsteered reply ("from_phase6", stage 2 only).
    for stage in ("1", "1e", "2"):
        g7 = N / f"grading/phase7_{stage}"
        if not any((g7 / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
            continue
        k7 = json.load(open(g7 / "key.json"))
        p7 = []
        for f in sorted((g7 / "runs/claude-sonnet-5.5-cc").glob("part_*.csv")):
            for r in rows(f):
                k = k7["graded"][r["item"].strip()]
                p7.append(dict(set=k["set"], item_id=k["pair"], lang=k["lang"], condition=k["condition"], label=r["label"].strip(), source="graded"))
        for a, b in k7.get("same_as", {}).items():
            u, c = a.rsplit("|", 1)
            p7.append(dict(set=u.split("|")[0], item_id=u.split("|")[1], lang=u.split("|")[2], condition=c, label="", source="same_as:" + b))
        for a, b in k7.get("from_phase6", {}).items():
            u, c = a.rsplit("|", 1)
            p7.append(dict(set=u.split("|")[0], item_id=u.split("|")[1], lang=u.split("|")[2], condition=c, label="", source="phase6:" + b))
        write(f"phase7_{stage}_llama_grades.csv", ["set", "item_id", "lang", "condition", "label", "source"],
              sorted(p7, key=lambda d: (d["set"], d["item_id"], d["lang"], d["condition"])))
    hs6 = N / "grading/phase6B_human/phase6_llama_human_graded.csv"
    if hs6.exists():
        hk6 = json.load(open(N / "grading/phase6B_human/key.json"))
        h6 = []
        for r in rows(hs6):
            u = hk6[r["item"]]
            u = u if isinstance(u, str) else u.get("uid", "")
            lab_ = next(v for k, v in r.items() if k.startswith("HUMAN_label")).strip()
            conf_ = next(v for k, v in r.items() if k.startswith("HUMAN_confidence")).strip()
            h6.append({"human_id": r["item"], "uid": u, "label": lab_, "confidence": conf_})
        write("phase6_llama_human.csv", ["human_id", "uid", "label", "confidence"], h6)
    hs7 = N / "grading/phase7_human/phase7_llama_human_graded.csv"
    if hs7.exists():
        hk7 = json.load(open(N / "grading/phase7_human/key.json"))
        h7 = [{"human_id": r["item"], "uid_condition": hk7[r["item"]],
               "label": next(v for k, v in r.items() if k.startswith("HUMAN_label")).strip(),
               "confidence": next(v for k, v in r.items() if k.startswith("HUMAN_confidence")).strip()} for r in rows(hs7)]
        write("phase7_llama_human.csv", ["human_id", "uid_condition", "label", "confidence"], h7)
    rv = N / "review/llama_disagreements_reviewed.csv"
    if rv.exists():  # addendum 4: unregistered adjudication of the Llama human-check disagreements
        write("llama_human_adjudicated.csv", ["review_id", "phase", "item", "uid", "your_label", "judge_label", "reviewed_label"],
              [{"review_id": r["review_id"], "phase": r["phase"], "item": r["item"], "uid": r["uid"], "your_label": r["your_label"],
                "judge_label": r["judge_label"], "reviewed_label": r["REVIEW_label"]} for r in rows(rv)])
    # Screening audit: human (original and adjudicated) and both candidates
    ak = json.load(open(N / "audit/audit_key.json"))
    adj = {r["item"]: r["FINAL_request_type"].strip() for r in rows(N / "audit/human/adjudication_reviewed.csv")}
    hum = []
    for r in rows(N / "audit/human/audit_human.csv"):
        rt = next(v for k, v in r.items() if k.startswith("HUMAN_request_type")).strip()
        fd = next(v for k, v in r.items() if k.startswith("HUMAN_fidelity")).strip()
        hum.append({"audit_id": r["item"], "item_id": ak[r["item"]]["item_id"], "tier": ak[r["item"]]["tier"],
                    "request_type": rt, "request_type_adjudicated": adj.get(r["item"], rt), "fidelity": fd})
    write("audit_human.csv", ["audit_id", "item_id", "tier", "request_type", "request_type_adjudicated", "fidelity"], hum)
    am = []
    for m in ("opus", "fable"):
        for p in (1, 2):
            for r in rows(N / f"audit/screen_runs/{m}/part_{p:02d}.csv"):
                am.append({"audit_id": r["item"].strip(), "item_id": ak[r["item"].strip()]["item_id"], "model": m,
                           "request_type": r["request_type"].strip(), "fidelity": r["fidelity"].strip()})
    write("audit_models.csv", ["audit_id", "item_id", "model", "request_type", "fidelity"], am)
    # Baseline screening (claude-opus-5-5)
    bk = json.load(open(N / "screening/baseline/key.json"))
    sc = []
    for f in sorted((N / "screening/baseline/runs/opus").glob("part_*.csv")):
        for r in rows(f):
            sc.append({"item_id": bk[r["item"].strip()]["item_id"], "request_type": r["request_type"].strip(), "fidelity": r["fidelity"].strip()})
    write("screening_opus.csv", ["item_id", "request_type", "fidelity"], sc)
    # Comprehension judgements, when present
    cdir = N / "screening/comprehension/runs/opus"
    if cdir.exists() and any(cdir.glob("part_*.csv")):
        ckey = json.load(open(N / "screening/comprehension/key.json"))
        cj = [{"item_id": ckey[r["item"].strip()], "judgement": r["judgement"].strip()}
              for f in sorted(cdir.glob("part_*.csv")) for r in rows(f)]
        write("comprehension_opus.csv", ["item_id", "judgement"], cj)


if __name__ == "__main__":
    main()
