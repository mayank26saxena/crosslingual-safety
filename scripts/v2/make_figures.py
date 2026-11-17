"""Static (paper) versions of the blog figures, drawn from the same numbers and design tokens.

Usage: uv run --no-project --with numpy --with matplotlib python -I scripts/v2/make_figures.py
Reads docs/blog/data/figures.json (written and checked by scripts/v2/export_figure_data.py; run that first).
Writes docs/paper/figures/fig1_what_replaces_refusal, fig2_projection, fig3_repairs, fig4_layers, fig5_mechanism, fig6_dose,
fig7_gate (.pdf, .png). PDFs embed TrueType fonts (pdf.fonttype 42) for camera-ready use.
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


MODELS = (("qwen", "Qwen3-8B"), ("llama", "Llama-3.1-8B"))


def fig_layers():
    L = F["layers"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex="col", gridspec_kw={"hspace": 0.55, "wspace": 0.22, "height_ratios": [1, 1]})
    for col, (key, title) in enumerate(MODELS):
        d = L[key]
        top, bot = axes[0, col], axes[1, col]
        for ax in (top, bot):
            ax.axvspan(d["chosen"] - 0.5, d["chosen"] + 0.5, color="#eef3f9", zorder=0)
            ax.grid(axis="y", color=RULE, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            ax.set_xlim(-0.5, d["layers"] - 0.5)
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                t.set_fontfamily("monospace")
        ab = d["ablation_removed"]
        top.plot(range(len(ab)), [100 * v for v in ab], color=INK, lw=1.4)
        top.set_ylim(min(-5, 100 * min(ab) - 5), 105)
        top.set_yticks([0, 50, 100])
        top.set_yticklabels(["0%", "50%", "100%"], fontfamily="monospace", fontsize=7, color=MUTED)
        top.text(d["chosen"] + 0.9, 96, f"layer {d['chosen']}", fontsize=7.5, weight="bold", color=ACCENT, va="top")
        top.set_title(title, loc="left", fontsize=9.5, weight="bold", pad=16)
        top.text(0, 1.04, "refusal removed by ablating this layer's direction", transform=top.transAxes, fontsize=7, color=MUTED)
        ends = []
        for k, c, name in (("en", INK2, "English"), ("hi", ACCENT, "Hindi")):
            v = d.get(f"auroc_{k}")
            if v:
                bot.plot(range(len(v)), v, color=c, lw=1.4)
                ends.append([len(v) - 1, v[-1], name, c])
        ends.sort(key=lambda e: e[1])
        if len(ends) == 2 and ends[1][1] - ends[0][1] < 0.05:
            mid = (ends[0][1] + ends[1][1]) / 2
            ends[0][1], ends[1][1] = mid - 0.025, mid + 0.025
        for x, y, name, c in ends:
            bot.text(x + 0.8, y, name, fontsize=7, weight="bold", color=c, va="center")
        bot.set_ylim(0.48, 1.02)
        bot.set_yticks([0.5, 0.75, 1.0])
        bot.set_yticklabels(["0.50", "0.75", "1.00"], fontfamily="monospace", fontsize=7, color=MUTED)
        bot.text(0, 1.04, "harmful vs harmless separation (AUROC)", transform=bot.transAxes, fontsize=7, color=MUTED)
        bot.set_xlabel("layer", fontsize=7.5, color=MUTED)
        bot.tick_params(axis="x", labelsize=7)
    fig.text(0.125, -0.07, "Per-layer English refusal direction (difference of means). Top: share of the harmful–harmless refusal-score gap removed\n"
             "by ablation (Qwen: fit prompts; Llama: held-out). Bottom: AUROC within each language. Band: chosen layer.", fontsize=7, color=MUTED)
    save(fig, "fig4_layers")


def fig_mechanism():
    spec = (("Qwen3-8B", 36, 14, 25, F["gate"]["qwen"]["tau"], "English-set", "1.25 × explicit–soft direction"),
            ("Llama-3.1-8B", 32, 10, 11, F["gate"]["llama"]["tau"], "Hindi-set", "1.5 × shortfall × refusal direction"))
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 2.9), gridspec_kw={"hspace": 0.15})
    for ax, (title, L, g, pu, tau, tset, what) in zip(axes, spec):
        ax.set_xlim(-10.5, 37.5)
        ax.set_ylim(-2.3, 2.3)
        ax.axis("off")
        ax.text(-10.5, 0, title, fontsize=9, weight="bold", va="center")
        ax.plot([-0.4, L - 0.6], [0, 0], color=AXIS, lw=4, solid_capstyle="round", zorder=1)
        for l in range(L):
            fc, ec = ("#eef3f9", ACCENT) if l == g else (("#fbf0e8", "#c9503a") if l == pu else ("white", RULE))
            ax.add_patch(plt.Rectangle((l - 0.4, -0.35), 0.8, 0.7, facecolor=fc, edgecolor=ec, lw=0.8, zorder=2))
        ax.text(-0.4, -0.75, "prompt →", fontsize=6.5, color=MUTED, va="top")
        ax.text(L - 0.6, -0.75, "→ reply", fontsize=6.5, color=MUTED, va="top", ha="right")
        bbox = lambda c: dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=c, lw=0.8)
        ax.annotate(f"read: score on refusal direction\nlayer {g} · open if > {tau:.2f} ({tset})", xy=(g, 0.4), xytext=(g, 1.55),
                    ha="center", va="center", fontsize=6.5, color="#1c4f8f", bbox=bbox(ACCENT),
                    arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1))
        ax.annotate(f"add: {what}\nat layer {pu} input, every position", xy=(pu, -0.4), xytext=(min(max(pu, 13), L - 6), -1.6),
                    ha="center", va="center", fontsize=6.5, color="#9b3a27", bbox=bbox("#c9503a"),
                    arrowprops=dict(arrowstyle="-|>", color="#c9503a", lw=1, mutation_scale=7))
    fig.text(0.125, 0.0, "Each box is a decoder layer. If the gate score at the last prompt token exceeds the threshold, a fixed vector is "
             "added for the whole reply.", fontsize=7, color=MUTED)
    save(fig, "fig5_mechanism")


def fig_dose():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"wspace": 0.25})
    subs = {"qwen": "9 settings · 401 dev pairs", "llama": "5 pushes · 150 dev pairs"}
    for ax, (key, title) in zip(axes, MODELS):
        p = F["dose"][key]
        xs, ys = [q["explicit"] for q in p["points"]], [q["broken"] for q in p["points"]]
        xlo, xhi = min(-10, math.floor(min(xs) / 10) * 10), max(50, math.ceil(max(xs) / 10) * 10)
        ylo, yhi = min(-2, math.floor(min(ys))), max(10, math.ceil(max(ys) / 5) * 5)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi + 1)
        ax.grid(color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=7)
        ax.set_xticks(range(xlo, xhi + 1, 10))
        ax.set_xticklabels([s(t, 0) if t else "0" for t in range(xlo, xhi + 1, 10)], fontfamily="monospace", color=MUTED)
        yt = list(range(math.ceil(ylo / 5) * 5, yhi + 1, 5))
        ax.set_yticks(yt)
        ax.set_yticklabels([s(t, 0) if t else "0" for t in yt], fontfamily="monospace", color=MUTED)
        gx = p["guards"]["gain"] if p["guards"]["gain"] is not None else 0
        ax.add_patch(plt.Rectangle((gx, ylo), xhi - gx, p["guards"]["broken"] - ylo, facecolor="#eef3f9", edgecolor="none", zorder=0))
        ax.axhline(p["guards"]["broken"], color="#c9503a", lw=1, ls=(0, (3, 2)))
        ax.text(xhi - 1, p["guards"]["broken"] + 0.4, f"broken-output guard (+{p['guards']['broken']})", ha="right", fontsize=6.5, color="#9b3a27")
        if p["guards"]["gain"] is not None:
            ax.axvline(gx, color=ACCENT, lw=1, ls=(0, (3, 2)))
            ax.text(gx + 1, yhi, f"selection bar (+{gx})", fontsize=6.5, color="#1c4f8f", va="top")
        placed = []
        for q in sorted(p["points"], key=lambda q: not q["chosen"]):
            c = ACCENT if q["qualifies"] else "#8a9194"
            ax.scatter([q["explicit"]], [q["broken"]], s=55 if q["chosen"] else 26, facecolor=c if q["chosen"] else "white", edgecolor=c,
                       lw=1.2, zorder=4)
            px, py = q["explicit"], q["broken"]
            clash = any(abs(px - a) < 18 and abs(py - b) < 0.12 * (yhi - ylo) for a, b in placed)
            if q["chosen"] or (len(p["points"]) <= 5 and not clash):
                right = px > xhi - 15
                ax.text(px + (-1.5 if right else 1.5), py + 0.05 * (yhi - ylo), q["label"] + (" · chosen" if q["chosen"] else ""),
                        fontsize=6.5, weight="bold" if q["chosen"] else "normal", color=INK2, ha="right" if right else "left")
                placed.append((px, py))
        ax.set_title(title, loc="left", fontsize=9.5, weight="bold", pad=14)
        ax.text(0, 1.03, subs[key], transform=ax.transAxes, fontdict=dict(MONO, size=7))
        ax.set_xlabel("Hindi explicit refusal, change (pts)", fontsize=7.5, color=MUTED)
    axes[0].set_ylabel("broken replies, change (pts)", fontsize=7.5, color=MUTED)
    fig.text(0.125, -0.1, "Development requests. Shaded: settings within the registered guards. Filled: setting taken to the held-out test. "
             "Blue: qualifies · grey: does not.", fontsize=7, color=MUTED)
    save(fig, "fig6_dose")


def fig_gate():
    import random
    rows = (("harmless_en", "Harmless · EN", INK2), ("harmless_hi", "Harmless · HI", ACCENT), ("harmful_en", "Harmful · EN", INK2),
            ("harmful_hi", "Harmful · HI", ACCENT))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True, gridspec_kw={"wspace": 0.32})
    for ax, (key, title) in zip(axes, MODELS):
        g = F["gate"][key]
        rng = random.Random(0)
        for y, (k, name, c) in zip((3, 2, 1, 0), rows):
            v = g["cells"][k]["scores"]
            ax.scatter(v, [y + rng.uniform(-0.28, 0.28) for _ in v], s=3, color=c, alpha=0.45, lw=0)
            share = sum(x > g["tau"] for x in v) / len(v)
            ax.text(1.02, y, f"{100 * share:.0f}% open", transform=ax.get_yaxis_transform(), va="center", fontdict=dict(MONO, size=6.5, color=INK))
        ax.axvline(g["tau"], color="#c9503a", lw=1.2)
        ax.text(g["tau"], -0.75, f"threshold {g['tau']:.2f}", ha="center", va="top", fontsize=6.5, color="#9b3a27")
        ax.set_yticks([3, 2, 1, 0])
        ax.set_yticklabels([r[1] for r in rows], fontsize=7.5)
        ax.tick_params(axis="y", labelleft=(key == "qwen"))
        ax.set_ylim(-0.9, 3.5)
        ax.grid(axis="x", color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=7)
        for t in ax.get_xticklabels():
            t.set_fontfamily("monospace")
        ax.xaxis.tick_top()
        ax.set_title(f"{title}\nlayer {g['layer']}, {'English' if key == 'qwen' else 'Hindi'}-set threshold", loc="left", fontsize=8.5,
                     weight="bold", pad=16)
    fig.text(0.125, -0.1, "Gate score of each held-out request (refusal-direction projection at the last prompt token).\nRight: share above "
             "the threshold as run, i.e. requests that received the push.", fontsize=7, color=MUTED)
    save(fig, "fig7_gate")


if __name__ == "__main__":
    fig1()
    fig3()  # projections -> Figure 2
    fig2()  # repairs -> Figure 3
    fig_layers()
    fig_mechanism()
    fig_dose()
    fig_gate()
