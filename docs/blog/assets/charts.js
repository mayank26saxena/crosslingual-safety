/* Small, dependency-free SVG chart components for research blog posts.
   Each component takes a container element and a plain data object, draws an SVG with hairline grid, direct labels,
   interval whiskers and a hover tooltip, and fills a data table for accessibility. Styling comes from tokens.css. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const COLOR = { REFUSE_EXPLICIT: "--c-explicit", REFUSE_SOFT: "--c-soft", COMPLY: "--c-comply", BROKEN: "--c-broken" };
  const NAME = { REFUSE_EXPLICIT: "Explicit refusal", REFUSE_SOFT: "Soft refusal", COMPLY: "Complied", BROKEN: "Broken" };
  const fmt = (x, d = 1) => (Math.abs(x) < 0.05 && d === 1 ? "0.0" : x.toFixed(d)).replace("-", "−");
  const sgn = (x, d = 1) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x), d);

  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, str, attrs = {}) {
    const t = el("text", Object.assign({ x, y }, attrs), parent);
    t.textContent = str;
    return t;
  }
  function svg(container, w, h) {
    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
    container.appendChild(s);
    return s;
  }
  const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

  // shared tooltip
  let tip;
  function tooltip(target, html) {
    if (!tip) { tip = document.createElement("div"); tip.className = "tip"; document.body.appendChild(tip); }
    target.addEventListener("mousemove", (e) => {
      tip.innerHTML = html; tip.style.opacity = 1;
      const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
      tip.style.left = x + "px"; tip.style.top = e.clientY + 14 + "px";
    });
    target.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
  }
  function xgrid(g, x, ticks, y0, y1, fmtTick = (t) => t + "%") {
    const grid = el("g", { class: "grid" }, g), tk = el("g", { class: "tick" }, g);
    ticks.forEach((t) => {
      el("line", { x1: x(t), x2: x(t), y1: y0, y2: y1 }, grid);
      text(tk, x(t), y0 - 8, fmtTick(t), { "text-anchor": "middle" });
    });
  }
  function whisker(g, x0, x1, y, color, h = 5) {
    el("line", { x1: x0, x2: x1, y1: y, y2: y, stroke: color, "stroke-width": 1.5, opacity: 0.55 }, g);
    [x0, x1].forEach((x) => el("line", { x1: x, x2: x, y1: y - h / 2, y2: y + h / 2, stroke: color, "stroke-width": 1.5, opacity: 0.55 }, g));
  }
  function table(container, head, rows) {
    const t = document.createElement("table"); t.className = "data";
    t.innerHTML = "<thead><tr>" + head.map((h) => `<th>${h}</th>`).join("") + "</tr></thead>";
    const b = document.createElement("tbody");
    rows.forEach((r) => { const tr = document.createElement("tr"); tr.innerHTML = r.map((c, i) => `<td class="${i ? "n" : ""}">${c}</td>`).join(""); b.appendChild(tr); });
    t.appendChild(b); container.appendChild(t);
  }

  /* Paired dumbbells: for each category, English -> Hindi share with Wilson whiskers, one panel per model. */
  function pairedShift(container, panels, opts = {}) {
    const narrow = container.clientWidth < 640;
    const W = narrow ? Math.max(320, container.clientWidth) : 920, rowH = 44, top = 74, panelGap = 34, labelW = narrow ? 104 : 118, padR = 58;
    const colW = narrow ? W - labelW : (W - labelW - panelGap) / 2, innerW = colW - padR;
    const panelH = top + 4 * rowH + 74, H = narrow ? panelH * panels.length : panelH;
    const s = svg(container, W, H);
    const x = scale(0, 100, 0, innerW);
    panels.forEach((p, pi) => {
      const g = el("g", { transform: narrow ? `translate(${labelW},${pi * panelH})` : `translate(${labelW + pi * (colW + panelGap)},0)` }, s);
      text(g, 0, 18, p.title, { "font-weight": 650, "font-size": 15 });
      text(g, 0, 36, p.subtitle, { class: "num", "font-size": 11 });
      xgrid(g, x, [0, 25, 50, 75, 100], top - 4, top + 4 * rowH - 10);
      p.rows.forEach((r, i) => {
        const y = top + 14 + i * rowH, c = css(COLOR[r.label]);
        if (pi === 0 || narrow) text(g, -14, y + 4, NAME[r.label], { "text-anchor": "end", "font-size": 13.5, "font-weight": 600 });
        whisker(g, x(r.en.ci[0]), x(r.en.ci[1]), y, css("--ink-2"));
        whisker(g, x(r.hi.ci[0]), x(r.hi.ci[1]), y, c);
        el("line", { x1: x(r.en.pct), x2: x(r.hi.pct), y1: y, y2: y, stroke: c, "stroke-width": 2.5, opacity: 0.85 }, g);
        el("circle", { cx: x(r.en.pct), cy: y, r: 6, fill: "#fff", stroke: css("--ink-2"), "stroke-width": 2 }, g);
        el("circle", { cx: x(r.hi.pct), cy: y, r: 6.5, fill: c, stroke: "#fff", "stroke-width": 2 }, g);
        const d = r.change.pts;
        text(g, innerW + 10, y + 4, sgn(d, 0) + " pts", { class: "num", "font-weight": 600, fill: css("--ink") });
        const hit = el("rect", { class: "hit", x: -6, y: y - rowH / 2, width: innerW + padR, height: rowH }, g);
        tooltip(hit, `<b>${p.title}: ${NAME[r.label]}</b><br>English ${fmt(r.en.pct)}% (${r.en.k}/${p.n})<br>Hindi ${fmt(r.hi.pct)}% (${r.hi.k}/${p.n})<br>Change ${sgn(d)} pts [${fmt(r.change.ci[0])}, ${fmt(r.change.ci[1])}]`);
      });
      // composition of lost refusals
      const yb = top + 4 * rowH + 12, total = p.lost_n;
      if (pi === 0 || narrow) text(g, -14, yb + 15, "Lost refusals", { "text-anchor": "end", "font-size": 13, "font-weight": 600, fill: css("--ink-2") });
      let acc = 0;
      ["REFUSE_SOFT", "COMPLY", "BROKEN"].forEach((k) => {
        const v = p.lost[k] || 0, w = (v / total) * innerW;
        const r = el("rect", { x: x(0) + (acc / total) * innerW + (acc ? 1 : 0), y: yb, width: Math.max(0, w - 2), height: 22, rx: 3, fill: css(COLOR[k]) }, g);
        if (w > 46) text(g, x(0) + (acc / total) * innerW + w / 2, yb + 15, `${v}`, { "text-anchor": "middle", "font-size": 12, "font-weight": 650, fill: k === "REFUSE_SOFT" ? css("--ink") : "#fff" });
        tooltip(r, `<b>${p.title}</b>: of ${total} requests refused explicitly in English but not in Hindi, ${v} became <b>${NAME[k].toLowerCase()}</b> in Hindi`);
        acc += v;
      });
      text(g, innerW + 10, yb + 15, `${total}`, { class: "num" });
    });
    if (opts.table) table(opts.table, ["Model / label", "English %", "Hindi %", "Change (pts)", "95% CI"],
      panels.flatMap((p) => p.rows.map((r) => [`${p.title}: ${NAME[r.label]}`, fmt(r.en.pct), fmt(r.hi.pct), sgn(r.change.pts), `${fmt(r.change.ci[0])} to ${fmt(r.change.ci[1])}`])));
  }

  /* Before -> after arrows per language, with the gap and a registered bar; one panel per model. */
  function repairs(container, panels, opts = {}) {
    const narrow = container.clientWidth < 640;
    const W = narrow ? Math.max(320, container.clientWidth) : 920, top = 78, rowH = 46, labelW = narrow ? 64 : 150, padR = narrow ? 70 : 160;
    const panelH = top + 2 * rowH + (narrow ? 74 : 58);
    const H = panels.length * panelH;
    const s = svg(container, W, H);
    const innerW = W - labelW - padR, x = scale(0, 100, 0, innerW);
    panels.forEach((p, pi) => {
      const g = el("g", { transform: `translate(${labelW},${pi * panelH})` }, s);
      text(g, -labelW, 20, p.title, { "font-weight": 650, "font-size": 15 });
      text(g, -labelW, 38, narrow ? p.subtitle.split(" · ")[0] : p.subtitle, { "font-size": 12.5, fill: css("--muted") });
      xgrid(g, x, [0, 25, 50, 75, 100], top - 4, top + 2 * rowH - 6);
      [["English", "en", css("--ink-2")], ["Hindi", "hi", css("--accent")]].forEach(([name, k, c], i) => {
        const a = p.arms[k], y = top + 18 + i * rowH;
        text(g, -12, y + 4, name, { "text-anchor": "end", "font-size": 13.5, "font-weight": 600 });
        whisker(g, x(a.before_ci[0]), x(a.before_ci[1]), y, c);
        whisker(g, x(a.after_ci[0]), x(a.after_ci[1]), y, c);
        el("line", { x1: x(a.before) + 7, x2: x(a.after) - 9, y1: y, y2: y, stroke: c, "stroke-width": 2.5 }, g);
        el("path", { d: `M${x(a.after) - 10},${y - 5} L${x(a.after) - 2},${y} L${x(a.after) - 10},${y + 5} Z`, fill: c }, g);
        el("circle", { cx: x(a.before), cy: y, r: 5, fill: "#fff", stroke: c, "stroke-width": 2 }, g);
        el("circle", { cx: x(a.after), cy: y, r: 5.5, fill: c, stroke: "#fff", "stroke-width": 2 }, g);
        text(g, innerW + 12, y + 4, `${sgn(a.change)} pts`, { class: "num", "font-weight": 600, fill: css("--ink") });
        if (!narrow) text(g, innerW + 82, y + 4, `[${fmt(a.ci[0])}, ${fmt(a.ci[1])}]`, { class: "num", "font-size": 10.5 });
        const hit = el("rect", { class: "hit", x: -labelW, y: y - rowH / 2, width: W, height: rowH }, g);
        tooltip(hit, `<b>${p.title}, ${name}</b><br>explicit refusal ${fmt(a.before)}% → ${fmt(a.after)}%<br>change ${sgn(a.change)} pts [${fmt(a.ci[0])}, ${fmt(a.ci[1])}], n = ${a.n}`);
      });
      if (p.bar_explicit) {
        const xb = x(p.arms.hi.before + p.bar_explicit), yb = top + 18 + rowH;
        el("line", { x1: xb, x2: xb, y1: yb - 13, y2: yb + 13, stroke: css("--c-comply"), "stroke-width": 1.5, "stroke-dasharray": "3 2" }, g);
        text(g, xb, yb + 26, `registered bar (+${p.bar_explicit})`, { "text-anchor": "middle", "font-size": 11, fill: css("--t-comply") });
      }
      const ng1 = `Gap ${fmt(p.gap_before)} → ${fmt(p.gap_after)} pts`, ng2 = `narrowing ${sgn(p.narrowing)} [${fmt(p.narrowing_ci[0])}, ${fmt(p.narrowing_ci[1])}]` + (p.bar_gap ? ` · bar ${p.bar_gap}` : "");
      if (narrow) { text(g, -labelW, top + 2 * rowH + 30, ng1, { class: "num", "font-size": 11.5, fill: css("--ink-2") }); text(g, -labelW, top + 2 * rowH + 48, ng2, { class: "num", "font-size": 11.5, fill: css("--ink-2") }); }
      else text(g, 0, top + 2 * rowH + 30, ng1 + " · " + ng2, { class: "num", "font-size": 11.5, fill: css("--ink-2") });
    });
    if (opts.table) table(opts.table, ["Model / language", "Before %", "After %", "Change (pts)", "95% CI"],
      panels.flatMap((p) => [["English", "en"], ["Hindi", "hi"]].map(([n, k]) => [`${p.title}: ${n}`, fmt(p.arms[k].before), fmt(p.arms[k].after), sgn(p.arms[k].change), `${fmt(p.arms[k].ci[0])} to ${fmt(p.arms[k].ci[1])}`])));
  }

  /* Projection small multiples: harmless and harmful, English and Hindi, each model on its own axis. */
  function projections(container, panels, opts = {}) {
    const narrow = container.clientWidth < 640;
    const W = narrow ? Math.max(320, container.clientWidth) : 920, panelGap = 40, labelW = 92, padR = 18, top = 78, rowH = 60;
    const colW = narrow ? W : (W - panelGap) / 2, panelH = top + 2 * rowH + 52, H = narrow ? panelH * panels.length : panelH;
    const s = svg(container, W, H);
    panels.forEach((p, pi) => {
      const innerW = colW - labelW - padR;
      const vals = Object.values(p.cells).flatMap((c) => c.ci);
      const lo = Math.floor(Math.min(0, ...vals)), hi = Math.ceil(Math.max(...vals));
      const x = scale(lo, hi, 0, innerW);
      const g = el("g", { transform: narrow ? `translate(${labelW},${pi * panelH})` : `translate(${pi * (colW + panelGap) + labelW},0)` }, s);
      text(g, -labelW, 18, p.title, { "font-weight": 650, "font-size": 15 });
      text(g, -labelW, 36, p.subtitle, { class: "num", "font-size": 11 });
      const step = (hi - lo) > 6 ? 2 : 1, ticks = [];
      for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
      xgrid(g, x, ticks, top - 4, top + 2 * rowH + 4, (t) => t);
      el("line", { x1: x(0), x2: x(0), y1: top - 4, y2: top + 2 * rowH + 4, stroke: css("--axis"), "stroke-width": 1.2 }, g);
      [["harmless", "Harmless"], ["harmful", "Harmful"]].forEach(([k, name], i) => {
        const y = top + 30 + i * rowH, en = p.cells[`${k}_en`], hi_ = p.cells[`${k}_hi`];
        text(g, -12, y + 4, name, { "text-anchor": "end", "font-size": 13.5, "font-weight": 600 });
        whisker(g, x(en.ci[0]), x(en.ci[1]), y, css("--ink-2"));
        whisker(g, x(hi_.ci[0]), x(hi_.ci[1]), y, css("--accent"));
        el("line", { x1: x(en.mean), x2: x(hi_.mean), y1: y, y2: y, stroke: css("--accent"), "stroke-width": 2, opacity: 0.5, "stroke-dasharray": "4 3" }, g);
        // Hindi filled dot first, English as a larger ring on top, so coincident means stay visible as overlap
        el("circle", { cx: x(hi_.mean), cy: y, r: 6, fill: css("--accent"), stroke: "#fff", "stroke-width": 2 }, g);
        el("circle", { cx: x(en.mean), cy: y, r: 9, fill: "none", stroke: css("--ink-2"), "stroke-width": 2 }, g);
        const close = Math.abs(x(en.mean) - x(hi_.mean)) < 34;
        if (close) {
          text(g, (x(en.mean) + x(hi_.mean)) / 2, y - 15, `EN ${fmt(en.mean, 2)} · HI ${fmt(hi_.mean, 2)}`, { class: "num", "font-size": 10.5, "text-anchor": "middle" });
        } else {
          text(g, x(en.mean), y - 15, fmt(en.mean, 2), { class: "num", "font-size": 10.5, "text-anchor": "middle" });
          text(g, x(hi_.mean), y - 15, fmt(hi_.mean, 2), { class: "num", "font-size": 10.5, "text-anchor": "middle", fill: css("--accent") });
        }
        const hit = el("rect", { class: "hit", x: -labelW, y: y - rowH / 2, width: colW, height: rowH }, g);
        tooltip(hit, `<b>${p.title}, ${name.toLowerCase()} requests</b><br>English mean ${fmt(en.mean, 2)}<br>Hindi mean ${fmt(hi_.mean, 2)}<br>n = ${en.n} pairs`);
      });
      text(g, 0, top + 2 * rowH + 30, p.note, { "font-size": 13, "font-weight": 600, fill: css("--ink") });
    });
    if (opts.table) table(opts.table, ["Model / set", "English mean", "Hindi mean", "n pairs"],
      panels.flatMap((p) => ["harmless", "harmful"].map((k) => [`${p.title}: ${k}`, fmt(p.cells[`${k}_en`].mean, 2), fmt(p.cells[`${k}_hi`].mean, 2), p.cells[`${k}_en`].n])));
  }

  // responsive: redraw every chart when the viewport crosses widths (debounced)
  const registry = [];
  function responsive(fn) {
    return (container, data, opts = {}) => {
      const draw = () => { container.innerHTML = ""; if (opts.table) opts.table.innerHTML = ""; fn(container, data, opts); };
      registry.push(draw); draw();
    };
  }
  let lastW = window.innerWidth, timer;
  window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(() => {
    if (Math.abs(window.innerWidth - lastW) > 40) { lastW = window.innerWidth; registry.forEach((d) => d()); } }, 150); });
  window.Charts = { pairedShift: responsive(pairedShift), repairs: responsive(repairs), projections: responsive(projections) };
})();
