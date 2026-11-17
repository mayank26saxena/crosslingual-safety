"""Figures for the paper and blog, from tracked, id-only label tables and result files only.

Usage: uv run --no-project --with numpy --with matplotlib python -I scripts/v2/make_figures.py
Writes docs/paper/figures/fig{1,2,3}_*.{png,svg}.
  fig1: what replaces the refusal: label shares on the 404 held-out harmful pairs, English vs Hindi, both models (judge labels).
  fig2: repairs: explicit refusal unsteered -> steered, Hindi and English, Qwen (Phase 3) and Llama (Phase 7), with the gap.
  fig3: internal signatures: Hindi harmful projection on the English refusal direction as a share of English, both models.
Colours: Okabe-Ito (colour-blind safe), fixed order by label.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "results/v2/labels"
OUT = ROOT / "docs/paper/figures"
ORDER = ["REFUSE_EXPLICIT", "REFUSE_SOFT", "COMPLY", "BROKEN"]
NAMES = {"REFUSE_EXPLICIT": "Explicit refusal", "REFUSE_SOFT": "Soft refusal", "COMPLY": "Complied", "BROKEN": "Broken"}
COL = {"REFUSE_EXPLICIT": "#0072B2", "REFUSE_SOFT": "#56B4E9", "COMPLY": "#D55E00", "BROKEN": "#999999"}
INK, MUTED, GRID = "#1c1b19", "#6a6862", "#dcdad3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def labels(path, keep):
    out = {}
    for r in csv.DictReader(open(path)):
        k = keep(r)
        if k:
            out.setdefault(k, {})[r["item_id"]] = r["label"]
    return out


def shares(d):
    c = Counter(d.values())
    n = sum(c.values())
    return {k: 100 * c[k] / n for k in ORDER}, n


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def fig1():
    q = labels(LAB / "phase3_test_grades.csv",
               lambda r: r["lang"] if r["set"] == "test_harmful" and r["condition"].endswith("unsteered") else None)
    l6 = labels(LAB / "phase6B_llama_grades.csv",
                lambda r: r["lang"] if r["set"] == "test_harmful" and r["condition"].endswith("unsteered") else None)
    bars = [("Qwen3-8B\nEnglish", q["en"]), ("Qwen3-8B\nHindi", q["hi"]), ("Llama-3.1-8B\nEnglish", l6["en"]),
            ("Llama-3.1-8B\nHindi", l6["hi"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ys = [3.3, 2.5, 1.0, 0.2]
    for (name, d), y in zip(bars, ys):
        s, n = shares(d)
        left = 0
        for k in ORDER:
            w = s[k]
            ax.barh(y, w - 0.4 if w > 0.6 else w, left=left + (0.2 if w > 0.6 else 0), height=0.62, color=COL[k],
                    label=NAMES[k] if y == ys[0] else None)
            if w >= 7:
                ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if k in ("REFUSE_EXPLICIT", "COMPLY") else INK)
            left += w
        ax.text(-1.5, y, name, ha="right", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of replies to the same 404 harmful requests (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0), frameon=False, fontsize=8.5)
    ax.set_title("What replaces the refusal: the same requests in English and Hindi", loc="left", fontsize=10.5, color=INK, pad=24)
    fig.text(0.0, -0.04, "Held-out test split; LLM-judge labels (Llama labels human-validated after adjudication). "
             "Qwen loses explicit refusals mostly to soft refusals; Llama to soft refusal and compliance.", fontsize=7.5, color=MUTED)
    save(fig, "fig1_label_shares")


def fig2():
    p3 = labels(LAB / "phase3_test_grades.csv",
                lambda r: f'{r["lang"]}|{r["condition"].split("_")[1]}' if r["set"] == "test_harmful" else None)
    s7 = json.load(open(ROOT / "results/v2/phase7/stage2/stage2_results.json"))
    ex = lambda d: 100 * sum(v == "REFUSE_EXPLICIT" for v in d.values()) / len(d)
    rows = [("Qwen3-8B (Phase 3 repair)", ex(p3["en|unsteered"]), ex(p3["en|gated"]), ex(p3["hi|unsteered"]), ex(p3["hi|gated"])),
            ("Llama-3.1-8B (Phase 7 repair)", s7["enharm"]["explicit"]["base"], s7["enharm"]["explicit"]["arm"],
             s7["hiharm"]["explicit"]["base"], s7["hiharm"]["explicit"]["arm"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, (name, e0, e1, h0, h1) in enumerate(rows):
        y = 1 - i
        for lang, a, b, yy, col in (("English", e0, e1, y + 0.13, "#555555"), ("Hindi", h0, h1, y - 0.13, "#0072B2")):
            ax.annotate("", xy=(b, yy), xytext=(a, yy), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
            ax.plot([a], [yy], "o", color=col, ms=6, mfc="white", mew=1.6)
            ax.text(b + (1.2 if b >= a else -1.2), yy, f"{lang} {a:.0f}→{b:.0f}%", va="center", fontsize=8.5,
                    ha="left" if b >= a else "right", color=col)
        g0, g1 = e0 - h0, e1 - h1
        ax.text(-2, y, f"{name}\ngap {g0:.1f} → {g1:.1f} pts", ha="right", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Explicit refusal of harmful requests (%), unsteered → gated push")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Repairs raise Hindi refusal, but also English", loc="left", fontsize=10.5, color=INK)
    fig.text(0.0, -0.07, "Held-out, 404 pairs, run once. Qwen: passed its checks, gap barely narrowed. "
             "Llama: narrowly missed its registered bars (+15 points; gap 10 points).", fontsize=7.5, color=MUTED)
    save(fig, "fig2_repairs")


def fig3():
    # Qwen: Phase 2a tier-A means (results/v2/direction); Llama: Phase 6 C2 (results/v2/phase6/stageB/phase6_results.json)
    l6 = json.load(open(ROOT / "results/v2/phase6/stageB/phase6_results.json"))["C2"]
    q = json.load(open(ROOT / "results/v2/direction/direction_results.json"))["per_language"]["hi"]["mean_proj"]
    data = [("Qwen3-8B", q["harm_target"] / q["harm_en"]), ("Llama-3.1-8B", l6["mean_hi"] / l6["mean_en"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, (name, v) in enumerate(data):
        ax.barh(1 - i, 100 * v, height=0.5, color="#0072B2" if i == 0 else "#E69F00")
        ax.text(100 * v + 2, 1 - i, f"{100 * v:.0f}%", va="center", fontsize=9, color=INK)
    ax.axvline(100, color=MUTED, lw=0.8, ls="--")
    ax.set_ylim(-0.5, 1.75)
    ax.text(101, 1.55, "English level", ha="left", va="center", fontsize=8, color=MUTED)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([d[0] for d in data])
    ax.set_xlim(0, 125)
    ax.set_xlabel("Hindi harmful projection, % of English (refusal direction)")
    ax.set_title("Same gap, different internal signatures", loc="left", fontsize=10.5, color=INK)
    fig.text(0.0, -0.18, "Mean projection of harmful Hindi prompts relative to the same requests in English, last template token. "
             "\nQwen: layer 14, 1,174 tier-A pairs; Llama: layer 10, 404 test pairs.", fontsize=7, color=MUTED)
    save(fig, "fig3_projection")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
