// SVG chart renderer.
//
// Written by hand rather than pulled from a library for one reason: the platform
// has to run air-gapped, and a CDN <script> tag is the first thing to break there.
// It covers exactly the chart types a QuerySpec can produce and nothing else.

const NS = "http://www.w3.org/2000/svg";
const SERIES_COLORS = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7", "--c8"];

function el(name, attrs = {}, text = null) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  if (text !== null) node.textContent = text;
  return node;
}

function color(index) {
  return `var(${SERIES_COLORS[index % SERIES_COLORS.length]})`;
}

export function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (abs >= 1e3) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Number.isInteger(n)) return String(n);
  return n.toLocaleString(undefined, { maximumFractionDigits: abs < 1 ? 4 : 2 });
}

// Metric tiles need fixed precision, which axis labels do not.
// `formatNumber` drops trailing zeros — good for an axis tick, bad next to a
// comparison figure, where an MAE of 82.997 rendering as "83" beside a baseline
// of "443.91" reads as though it were measured to a different accuracy.
export function formatMetric(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e6) return formatNumber(n);
  const digits = abs >= 100 ? 2 : abs >= 1 ? 3 : 4;
  return n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function formatCell(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    // Deliberately NOT formatNumber: an axis label may compact 1234.5 to "1,235",
    // but a table cell is the answer to the question that was asked, and silently
    // dropping the cents there is a wrong number rather than a tidy one.
    if (!Number.isFinite(value)) return String(value);
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toLocaleString(undefined, {
      maximumFractionDigits: Math.abs(value) >= 1 ? 2 : 6,
    });
  }
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}T/.test(value)) {
    // Timestamps arrive ISO from pandas; midnight means it is really a date.
    return value.endsWith("T00:00:00.000Z") || value.includes("T00:00:00")
      ? value.slice(0, 10)
      : value.slice(0, 16).replace("T", " ");
  }
  return String(value);
}

export function isNumeric(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function labelFor(value) {
  const text = formatCell(value);
  return text.length > 22 ? text.slice(0, 21) + "…" : text;
}

// "Nice" axis bounds — round numbers beat exact data extents for readability.
function niceScale(min, max, ticks = 5) {
  if (min === max) {
    if (min === 0) return { lo: 0, hi: 1, step: 0.25 };
    const pad = Math.abs(min) * 0.2;
    min -= pad;
    max += pad;
  }
  const lo = Math.min(0, min); // bars must be anchored at zero to not mislead
  const raw = (max - lo) / ticks;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  return { lo: Math.floor(lo / step) * step, hi: Math.ceil(max / step) * step, step };
}

function axes(svg, { x0, y0, width, height, scale, xTicks }) {
  const group = el("g", { class: "axis" });
  for (let v = scale.lo; v <= scale.hi + 1e-9; v += scale.step) {
    const y = y0 - ((v - scale.lo) / (scale.hi - scale.lo)) * height;
    group.appendChild(el("line", { x1: x0, x2: x0 + width, y1: y, y2: y, class: "gridline" }));
    group.appendChild(el("text", { x: x0 - 8, y: y + 4, "text-anchor": "end" }, formatNumber(v)));
  }
  group.appendChild(el("line", { x1: x0, x2: x0 + width, y1: y0, y2: y0 }));

  if (xTicks) {
    // Thin the labels until they fit rather than letting them collide.
    const stride = Math.max(1, Math.ceil(xTicks.length / Math.floor(width / 74)));
    xTicks.forEach((tick, i) => {
      if (i % stride !== 0 && i !== xTicks.length - 1) return;
      group.appendChild(
        el("text", { x: tick.x, y: y0 + 17, "text-anchor": "middle" }, labelFor(tick.label))
      );
    });
  }
  svg.appendChild(group);
}

// --------------------------------------------------------------------- shapes
function drawSeriesChart(rows, dims, metrics, kind) {
  const W = 760, H = 320, PAD = { t: 14, r: 18, b: 34, l: 62 };
  const width = W - PAD.l - PAD.r;
  const height = H - PAD.t - PAD.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  const xKey = dims[0];
  const seriesKey = dims[1];
  const metric = metrics[0];

  // One line per (extra dimension) value, or one line per metric if there is none.
  let series;
  if (seriesKey) {
    const groups = new Map();
    for (const row of rows) {
      const name = formatCell(row[seriesKey]);
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(row);
    }
    series = [...groups.entries()].map(([name, points]) => ({ name, points, key: metric }));
  } else {
    series = metrics.map((m) => ({ name: m, points: rows, key: m }));
  }

  const categories = [...new Set(rows.map((r) => formatCell(r[xKey])))];
  const values = series.flatMap((s) => s.points.map((p) => Number(p[s.key]) || 0));
  const scale = niceScale(Math.min(...values), Math.max(...values));

  const step = categories.length > 1 ? width / (categories.length - 1) : 0;
  const xAt = (label) => {
    const i = categories.indexOf(label);
    return categories.length === 1 ? PAD.l + width / 2 : PAD.l + i * step;
  };
  const yAt = (v) => PAD.t + height - ((v - scale.lo) / (scale.hi - scale.lo)) * height;

  axes(svg, {
    x0: PAD.l, y0: PAD.t + height, width, height, scale,
    xTicks: categories.map((label) => ({ x: xAt(label), label })),
  });

  series.forEach((s, index) => {
    const points = s.points
      .map((p) => ({ x: xAt(formatCell(p[xKey])), y: yAt(Number(p[s.key]) || 0) }))
      .sort((a, b) => a.x - b.x);
    if (!points.length) return;
    const path = points.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");

    if (kind === "area") {
      const base = yAt(scale.lo);
      svg.appendChild(
        el("path", {
          d: `${path} L${points.at(-1).x.toFixed(1)} ${base} L${points[0].x.toFixed(1)} ${base} Z`,
          fill: color(index), opacity: 0.16,
        })
      );
    }
    svg.appendChild(
      el("path", { d: path, fill: "none", stroke: color(index), "stroke-width": 2,
                   "stroke-linejoin": "round", "stroke-linecap": "round" })
    );
    if (points.length <= 60) {
      for (const p of points) {
        svg.appendChild(el("circle", { cx: p.x, cy: p.y, r: 2.6, fill: color(index) }));
      }
    }
  });

  return { svg, legend: series.length > 1 ? series.map((s) => s.name) : null };
}

function drawBar(rows, dims, metrics, horizontal) {
  const count = rows.length;
  const W = 760;
  const H = horizontal ? Math.max(180, 26 * count + 46) : 320;
  const PAD = horizontal ? { t: 10, r: 46, b: 26, l: 150 } : { t: 14, r: 18, b: 40, l: 62 };
  const width = W - PAD.l - PAD.r;
  const height = H - PAD.t - PAD.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  const xKey = dims[0];
  const used = metrics.slice(0, horizontal ? 1 : metrics.length);
  const values = rows.flatMap((r) => used.map((m) => Number(r[m]) || 0));
  const scale = niceScale(Math.min(...values, 0), Math.max(...values, 0));

  if (horizontal) {
    const band = height / Math.max(count, 1);
    const barH = Math.min(18, band * 0.68);
    const xAt = (v) => PAD.l + ((v - scale.lo) / (scale.hi - scale.lo)) * width;
    const group = el("g", { class: "axis" });
    for (let v = scale.lo; v <= scale.hi + 1e-9; v += scale.step) {
      group.appendChild(el("line", { x1: xAt(v), x2: xAt(v), y1: PAD.t, y2: PAD.t + height, class: "gridline" }));
      group.appendChild(el("text", { x: xAt(v), y: PAD.t + height + 16, "text-anchor": "middle" }, formatNumber(v)));
    }
    svg.appendChild(group);

    rows.forEach((row, i) => {
      const value = Number(row[used[0]]) || 0;
      const y = PAD.t + i * band + (band - barH) / 2;
      const x = xAt(Math.min(0, value));
      svg.appendChild(
        el("rect", { x, y, width: Math.abs(xAt(value) - xAt(0)), height: barH, rx: 2, fill: color(0) })
      );
      svg.appendChild(
        el("text", { x: PAD.l - 9, y: y + barH / 2 + 4, "text-anchor": "end", class: "axis" },
           labelFor(row[xKey]))
      );
      svg.appendChild(
        el("text", { x: xAt(value) + 6, y: y + barH / 2 + 4, class: "axis" }, formatNumber(value))
      );
    });
    return { svg, legend: null };
  }

  const band = width / Math.max(count, 1);
  const groupW = band * 0.72;
  const barW = groupW / used.length;
  const yAt = (v) => PAD.t + height - ((v - scale.lo) / (scale.hi - scale.lo)) * height;

  axes(svg, {
    x0: PAD.l, y0: PAD.t + height, width, height, scale,
    xTicks: rows.map((row, i) => ({ x: PAD.l + i * band + band / 2, label: row[xKey] })),
  });

  rows.forEach((row, i) => {
    used.forEach((metric, m) => {
      const value = Number(row[metric]) || 0;
      const x = PAD.l + i * band + (band - groupW) / 2 + m * barW;
      const top = Math.min(yAt(value), yAt(0));
      svg.appendChild(
        el("rect", { x, y: top, width: Math.max(1, barW - 2), height: Math.abs(yAt(value) - yAt(0)),
                     rx: 2, fill: color(m) })
      );
    });
  });

  return { svg, legend: used.length > 1 ? used : null };
}

function drawPie(rows, dims, metrics) {
  const W = 760, H = 300, R = 118, cx = 200, cy = H / 2;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const xKey = dims[0];
  const metric = metrics[0];

  const slices = rows
    .map((row) => ({ label: formatCell(row[xKey]), value: Math.max(0, Number(row[metric]) || 0) }))
    .filter((s) => s.value > 0);
  const total = slices.reduce((sum, s) => sum + s.value, 0);
  if (!total) return { svg, legend: null };

  let angle = -Math.PI / 2;
  slices.forEach((slice, i) => {
    const sweep = (slice.value / total) * Math.PI * 2;
    const end = angle + sweep;
    const large = sweep > Math.PI ? 1 : 0;
    const [x1, y1] = [cx + R * Math.cos(angle), cy + R * Math.sin(angle)];
    const [x2, y2] = [cx + R * Math.cos(end), cy + R * Math.sin(end)];
    // A full circle cannot be expressed as a single arc — degenerate to a circle.
    const d = slices.length === 1
      ? null
      : `M${cx} ${cy} L${x1.toFixed(1)} ${y1.toFixed(1)} A${R} ${R} 0 ${large} 1 ${x2.toFixed(1)} ${y2.toFixed(1)} Z`;
    svg.appendChild(
      d ? el("path", { d, fill: color(i) }) : el("circle", { cx, cy, r: R, fill: color(i) })
    );

    const ly = 40 + i * 22;
    svg.appendChild(el("rect", { x: 400, y: ly - 9, width: 11, height: 11, rx: 2, fill: color(i) }));
    svg.appendChild(
      el("text", { x: 420, y: ly, class: "axis" },
         `${labelFor(slice.label)} — ${formatNumber(slice.value)} (${((slice.value / total) * 100).toFixed(1)}%)`)
    );
    angle = end;
  });

  return { svg, legend: null };
}

function drawBigNumber(rows, metrics) {
  const svg = el("svg", { viewBox: "0 0 760 150", role: "img" });
  const value = rows.length ? rows[0][metrics[0]] : null;
  svg.appendChild(el("text", { x: 380, y: 84, "text-anchor": "middle", class: "bignum" }, formatNumber(value)));
  svg.appendChild(el("text", { x: 380, y: 112, "text-anchor": "middle", class: "bignum-label" }, metrics[0] || ""));
  return { svg, legend: null };
}

function drawScatter(rows, metrics) {
  const W = 760, H = 330, PAD = { t: 14, r: 18, b: 40, l: 62 };
  const width = W - PAD.l - PAD.r, height = H - PAD.t - PAD.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const [xKey, yKey] = metrics;

  const xs = rows.map((r) => Number(r[xKey]) || 0);
  const ys = rows.map((r) => Number(r[yKey]) || 0);
  const xScale = niceScale(Math.min(...xs), Math.max(...xs));
  const yScale = niceScale(Math.min(...ys), Math.max(...ys));
  const xAt = (v) => PAD.l + ((v - xScale.lo) / (xScale.hi - xScale.lo)) * width;
  const yAt = (v) => PAD.t + height - ((v - yScale.lo) / (yScale.hi - yScale.lo)) * height;

  axes(svg, { x0: PAD.l, y0: PAD.t + height, width, height, scale: yScale, xTicks: null });
  const group = el("g", { class: "axis" });
  for (let v = xScale.lo; v <= xScale.hi + 1e-9; v += xScale.step) {
    group.appendChild(el("text", { x: xAt(v), y: PAD.t + height + 17, "text-anchor": "middle" }, formatNumber(v)));
  }
  svg.appendChild(group);

  rows.forEach((row) => {
    svg.appendChild(
      el("circle", { cx: xAt(Number(row[xKey]) || 0), cy: yAt(Number(row[yKey]) || 0), r: 3.2,
                     fill: color(0), opacity: 0.66 })
    );
  });
  svg.appendChild(el("text", { x: PAD.l + width / 2, y: H - 4, "text-anchor": "middle", class: "axis" }, xKey));
  return { svg, legend: [yKey + " vs " + xKey] };
}

function drawHeatmap(rows, dims, metrics) {
  const xs = [...new Set(rows.map((r) => formatCell(r[dims[0]])))];
  const ys = [...new Set(rows.map((r) => formatCell(r[dims[1]])))];
  const cell = Math.min(52, Math.max(20, 640 / Math.max(xs.length, 1)));
  const PAD = { t: 12, r: 12, b: 46, l: 130 };
  const W = PAD.l + xs.length * cell + PAD.r;
  const H = PAD.t + ys.length * cell + PAD.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  const metric = metrics[0];
  const values = rows.map((r) => Number(r[metric]) || 0);
  const lo = Math.min(...values), hi = Math.max(...values);
  const lookup = new Map(rows.map((r) => [`${formatCell(r[dims[0]])} ${formatCell(r[dims[1]])}`, Number(r[metric]) || 0]));

  ys.forEach((y, row) => {
    xs.forEach((x, col) => {
      const value = lookup.get(`${x} ${y}`);
      const t = value === undefined || hi === lo ? 0 : (value - lo) / (hi - lo);
      svg.appendChild(
        el("rect", {
          x: PAD.l + col * cell, y: PAD.t + row * cell, width: cell - 1.5, height: cell - 1.5, rx: 2,
          fill: value === undefined ? "var(--surface-2)" : color(0),
          opacity: value === undefined ? 0.4 : 0.15 + t * 0.85,
        })
      );
    });
    svg.appendChild(
      el("text", { x: PAD.l - 8, y: PAD.t + row * cell + cell / 2 + 4, "text-anchor": "end", class: "axis" },
         labelFor(y))
    );
  });

  const stride = Math.max(1, Math.ceil(xs.length / Math.floor((W - PAD.l) / 70)));
  xs.forEach((x, col) => {
    if (col % stride !== 0) return;
    svg.appendChild(
      el("text", { x: PAD.l + col * cell + cell / 2, y: PAD.t + ys.length * cell + 16,
                   "text-anchor": "middle", class: "axis" }, labelFor(x))
    );
  });
  return { svg, legend: [`${metric}: ${formatNumber(lo)} → ${formatNumber(hi)}`] };
}

// -------------------------------------------------------------------- public
export function renderChart(container, { chart, columns, rows, dimensions, metrics }) {
  container.innerHTML = "";
  if (!rows || !rows.length) {
    container.innerHTML = '<div class="empty">No rows to plot.</div>';
    return;
  }

  const dims = (dimensions || []).filter((d) => columns.includes(d));
  const mets = (metrics || []).filter((m) => columns.includes(m));

  // Fall back to a table whenever the requested chart needs a shape the result
  // does not have. Drawing a wrong chart is worse than drawing none.
  const unsupported =
    (chart !== "table" && chart !== "big_number" && (!dims.length || !mets.length)) ||
    (chart === "big_number" && !mets.length) ||
    (chart === "scatter" && mets.length < 2) ||
    (chart === "heatmap" && dims.length < 2);
  if (chart === "table" || unsupported) {
    container.appendChild(renderTable(columns, rows, 100));
    return;
  }

  let result;
  switch (chart) {
    case "line": result = drawSeriesChart(rows, dims, mets, "line"); break;
    case "area": result = drawSeriesChart(rows, dims, mets, "area"); break;
    case "bar": result = drawBar(rows, dims, mets, false); break;
    case "horizontal_bar": result = drawBar(rows, dims, mets, true); break;
    case "pie": result = drawPie(rows, dims, mets); break;
    case "big_number": result = drawBigNumber(rows, mets); break;
    case "scatter": result = drawScatter(rows, mets); break;
    case "heatmap": result = drawHeatmap(rows, dims, mets); break;
    default: container.appendChild(renderTable(columns, rows, 100)); return;
  }

  const shell = document.createElement("div");
  shell.className = "chart-shell";
  shell.appendChild(result.svg);
  container.appendChild(shell);

  if (result.legend) {
    const legend = document.createElement("div");
    legend.className = "legend";
    result.legend.forEach((name, i) => {
      const item = document.createElement("span");
      item.innerHTML = `<span class="swatch" style="background:${color(i)}"></span>`;
      item.append(name);
      legend.appendChild(item);
    });
    container.appendChild(legend);
  }
}

export function renderTable(columns, rows, limit = 100) {
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");

  const head = table.createTHead().insertRow();
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = column;
    head.appendChild(th);
  }

  const body = table.createTBody();
  for (const row of rows.slice(0, limit)) {
    const tr = body.insertRow();
    for (const column of columns) {
      const td = tr.insertCell();
      td.textContent = formatCell(row[column]);
      if (isNumeric(row[column])) td.className = "num";
    }
  }
  wrap.appendChild(table);

  if (rows.length > limit) {
    const note = document.createElement("div");
    note.className = "table-note";
    note.textContent = `Showing ${limit} of ${rows.length.toLocaleString()} rows.`;
    wrap.appendChild(note);
  }
  return wrap;
}
