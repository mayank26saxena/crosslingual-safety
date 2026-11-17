"""Static (paper) versions of the blog figures, drawn from the same numbers and design tokens.

Usage: uv run --no-project --with numpy --with matplotlib python -I scripts/v2/make_figures.py
Reads docs/blog/data/figures.json (written and checked by scripts/v2/export_figure_data.py; run that first).
Writes docs/paper/figures/fig1_what_replaces_refusal, fig2_projection, fig3_repairs (.pdf, .png). PDFs embed TrueType fonts (pdf.fonttype 42) for camera-ready use.
Design: the research-blog tokens (light surface, hairline grid, direct labels, interval whiskers, mono numbers) and the same palette as
docs/blog/assets/tokens.css (validated with the dataviz validator).
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
F = json.load(open(ROOT / "docs/blog/data/figures.json"))
OUT = ROOT / "docs/paper/figures"
INK, INK2, MUTED, RULE, AXIS, ACCENT = "#1d272a", "#4a5458", "#687275", "#e3e1da", "#b9bec6", "#2563b0"
COL = {"REFUSE_EXPLICIT": "#2563b0", "REFUSE_SOFT": "#d09017", "COMPLY": "#c9503a", "BROKEN": "#8a9194"}
NAME = {"REFUSE_EXPLICIT": "Explicit refusal", "REFUSE_SOFT": "Soft refusal", "COMPLY": "Complied", "BROKEN": "Broken"}
MONO = {"family": "monospace", "size": 7.5, "color": MUTED}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": INK, "xtick.labelsize": 7.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
                     "savefig.facecolor": "white"})


def m(x, d=1):
    return f"{x:.{d}f}".replace("-", "−")


def s(x, d=1):
    return ("+" if x > 0 else "") + m(x, d)


def base(ax, xlim, ticks, fmt=lambda t: f"{t:g}%"):
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt(t) for t in ticks], fontfamily="monospace")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)


def whisk(ax, lo, hi, y, c):
    ax.plot([lo, hi], [y, y], color=c, lw=1, alpha=0.55, solid_capstyle="butt")
    for v in (lo, hi):
        ax.plot([v, v], [y - 0.09, y + 0.09], color=c, lw=1, alpha=0.55)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, gridspec_kw={"wspace": 0.18})
    for ax, (key, title) in zip(axes, (("qwen", "Qwen3-8B"), ("llama", "Llama-3.1-8B"))):
        d = F["fig1"][key]
        base(ax, (-2, 118), [0, 25, 50, 75, 100])
        for y, r in zip([4, 3, 2, 1], d["rows"]):
            c = COL[r["label"]]
            whisk(ax, *r["en"]["ci"], y, INK2)
            whisk(ax, *r["hi"]["ci"], y, c)
            ax.plot([r["en"]["pct"], r["hi"]["pct"]], [y, y], color=c, lw=1.8, alpha=0.85, zorder=2)
            ax.scatter([r["en"]["pct"]], [y], s=30, facecolor="white", edgecolor=INK2, lw=1.3, zorder=3)
            ax.scatter([r["hi"]["pct"]], [y], s=34, color=c, edgecolor="white", lw=1, zorder=4)
            ax.text(104, y, s(r["change"]["pts"], 0) + " pts", va="center", ha="left", fontdict=dict(MONO, color=INK, weight="bold"))
        tot, acc = d["lost_n"], 0
        for k in ("REFUSE_SOFT", "COMPLY", "BROKEN"):
            v = d["lost_refusals"].get(k, 0)
            w = 100 * v / tot
            ax.barh(-0.1, max(0, w - 0.6), left=acc + (0.3 if acc else 0), height=0.55, color=COL[k], edgecolor="none")
            if w > 12:
                ax.text(acc + w / 2, -0.1, str(v), ha="center", va="center", fontsize=7.5, weight="bold",
                        color=INK if k == "REFUSE_SOFT" else "white")
            acc += w
        ax.text(104, -0.1, str(tot), va="center", fontdict=MONO)
        ax.set_title(title, loc="left", fontsize=9.5, weight="bold", pad=16)
        ax.set_ylim(-0.6, 4.6)
    axes[0].set_yticks([4, 3, 2, 1, -0.1])
    axes[0].set_yticklabels([NAME[r["label"]] for r in F["fig1"]["qwen"]["rows"]] + ["Lost refusals"])
    fig.text(0.125, -0.04, "Open: English · filled: Hindi · whiskers: Wilson 95% · right: change (pts) · bar: requests refused explicitly in "
             "English but not in Hindi, by Hindi outcome", fontsize=7, color=MUTED)
    save(fig, "fig1_what_replaces_refusal")


def fig2():
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.4), gridspec_kw={"hspace": 1.25})
    for ax, (key, title, sub) in zip(axes, (("qwen", "Qwen3-8B", "push toward explicit refusal · passed its registered checks"),
                                             ("llama", "Llama-3.1-8B", "push along the refusal direction, Hindi-set gate · narrowly missed its bars"))):
        p = F["fig2"][key]
        base(ax, (0, 128), [0, 25, 50, 75, 100])
        for y, (lang, name, c) in zip((1, 0), (("en", "English", INK2), ("hi", "Hindi", ACCENT))):
            a = p["arms"][lang]
            whisk(ax, *a["before_ci"], y, c)
            whisk(ax, *a["after_ci"], y, c)
            ax.annotate("", xy=(a["after"] - 0.8, y), xytext=(a["before"] + 0.8, y),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, mutation_scale=8), zorder=2)
            ax.scatter([a["before"]], [y], s=30, facecolor="white", edgecolor=c, lw=1.3, zorder=3)
            ax.scatter([a["after"]], [y], s=34, color=c, edgecolor="white", lw=1, zorder=4)
            ax.text(103, y, f"{s(a['change'])} [{m(a['ci'][0])}, {m(a['ci'][1])}]", va="center", fontdict=dict(MONO, color=INK))
        if p.get("bar_explicit"):
            xb = p["arms"]["hi"]["before"] + p["bar_explicit"]
            ax.plot([xb, xb], [-0.3, 0.3], color="#c9503a", lw=1.1, ls=(0, (2, 1.5)))
            ax.text(xb, -0.55, "bar +15", ha="center", fontsize=6.5, color="#9b3a27")
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["English", "Hindi"])
        ax.set_ylim(-0.8, 1.6)
        ax.text(0, 1.48, title, transform=ax.transAxes, fontsize=9.5, weight="bold", va="bottom")
        ax.text(0, 1.30, sub, transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
        gap = (f"gap {m(p['gap_before'])} → {m(p['gap_after'])} pts · narrowing {s(p['narrowing'])} "
               f"[{m(p['narrowing_ci'][0])}, {m(p['narrowing_ci'][1])}]")
        ax.text(0, -0.78, gap + (f" · bar {p['bar_gap']:g}" if p.get("bar_gap") else ""), fontdict=dict(MONO, size=7, color=INK2), va="top")
    fig.text(0.125, -0.06, "Explicit refusal of harmful requests, unsteered (open) → repaired (filled), held-out 404 pairs, run once. "
             "Right: paired change with bootstrap 95% CI.", fontsize=7, color=MUTED)
    save(fig, "fig3_repairs")


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"wspace": 0.3})
    for ax, (key, title) in zip(axes, (("qwen", "Qwen3-8B"), ("llama", "Llama-3.1-8B"))):
        p = F["fig3"][key]
        vals = [v for c in p["cells"].values() for v in c["ci"]]
        pad = 0.08 * (max(vals) - min(0, min(vals)))
        lo, hi = min(0, min(vals)) - pad, max(vals) + pad
        step = 2 if hi - lo > 6 else 1
        ticks = list(range(math.ceil(lo / step) * step, int(hi) + 1, step))
        base(ax, (lo, hi), ticks, fmt=lambda t: m(t, 0))
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        for y, k in ((1, "harmless"), (0, "harmful")):
            en, hi_ = p["cells"][f"{k}_en"], p["cells"][f"{k}_hi"]
            whisk(ax, *en["ci"], y, INK2)
            whisk(ax, *hi_["ci"], y, ACCENT)
            ax.plot([en["mean"], hi_["mean"]], [y, y], color=ACCENT, lw=1.2, alpha=0.5, ls=(0, (3, 2)))
            ax.scatter([hi_["mean"]], [y], s=34, color=ACCENT, edgecolor="white", lw=1, zorder=4)
            ax.scatter([en["mean"]], [y], s=80, facecolor="none", edgecolor=INK2, lw=1.3, zorder=5)
            if abs(en["mean"] - hi_["mean"]) < 0.08 * (hi - lo):
                ax.text((en["mean"] + hi_["mean"]) / 2, y + 0.3, f"EN {m(en['mean'], 2)} · HI {m(hi_['mean'], 2)}", ha="center",
                        fontdict=dict(MONO, size=6.5))
            else:
                ax.text(en["mean"], y + 0.3, m(en["mean"], 2), ha="center", fontdict=dict(MONO, size=6.5))
                ax.text(hi_["mean"], y + 0.3, m(hi_["mean"], 2), ha="center", fontdict=dict(MONO, size=6.5, color=ACCENT))
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["Harmless", "Harmful"])
        ax.set_ylim(-0.45, 1.75)
        ax.set_title(f"{title} (layer {p['layer']})", loc="left", fontsize=9.5, weight="bold", pad=14)
        ax.text(0.0, -0.14, f"harmful Hindi at {100 * p['ratio']:.0f}% of English", transform=ax.transAxes, fontsize=7.5, color=INK,
                weight="bold", va="top")
    fig.text(0.125, -0.12, "Mean projection on each model's English-fitted refusal direction (last template token), same 404 harmful and "
             "169 harmless test pairs. Ring: English · dot: Hindi · whiskers: bootstrap 95% CI.", fontsize=7, color=MUTED)
    save(fig, "fig2_projection")


if __name__ == "__main__":
    fig1()
    fig3()  # projections -> Figure 2
    fig2()  # repairs -> Figure 3
