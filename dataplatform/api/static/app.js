// Insight Platform UI.
//
// Plain ES modules against the same FastAPI process that serves this file, so
// there is no build step and no second thing to deploy.

import { formatCell, formatMetric, formatNumber, renderChart, renderTable } from "/static/charts.js";

const state = {
  config: null,
  datasets: [],
  sources: [],
  lastAsk: null,
  dbDatasets: [], // tables selected for the Dashboard view, in pick order
};

// ------------------------------------------------------------------ plumbing
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    // FastAPI puts the message in `detail`; surface that rather than a status code.
    throw new Error(payload?.detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

const $ = (id) => document.getElementById(id);

function setStatus(target, kind, message) {
  const node = $(target);
  if (!message) {
    node.innerHTML = "";
    return;
  }
  node.innerHTML =
    kind === "busy"
      ? `<div class="notice info"><span class="spinner"></span> ${escapeHtml(message)}</div>`
      : `<div class="notice ${kind}">${escapeHtml(message)}</div>`;
}

function titleCase(name) {
  return String(name).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

function card(title, bodyNode, { flush = false } = {}) {
  const node = document.createElement("div");
  node.className = flush ? "card card-flush" : "card";
  if (title) {
    const heading = document.createElement("h3");
    heading.textContent = title;
    heading.style.padding = flush ? "14px 16px 0" : "";
    node.appendChild(heading);
  }
  node.appendChild(bodyNode);
  return node;
}

function html(markup) {
  const template = document.createElement("template");
  template.innerHTML = markup.trim();
  return template.content.firstElementChild;
}

// ----------------------------------------------------------------- boot/env
async function boot() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  $("theme-toggle").addEventListener("click", toggleTheme);
  const stored = localStorage.getItem("insight-theme");
  if (stored) document.documentElement.dataset.theme = stored;

  // sidebar collapse
  const app = document.querySelector(".app");
  const sidebarToggle = $("sidebar-toggle");
  if (localStorage.getItem("nav-collapsed") === "1") {
    app.classList.add("nav-collapsed");
    sidebarToggle.innerHTML = "&#8250;";
  }
  sidebarToggle.addEventListener("click", () => {
    const collapsed = app.classList.toggle("nav-collapsed");
    sidebarToggle.innerHTML = collapsed ? "&#8250;" : "&#8249;";
    localStorage.setItem("nav-collapsed", collapsed ? "1" : "0");
  });

  wireAsk();
  wireAskNlp();
  wireSources();
  wireData();
  wireDashboard();
  wireAnalyze();
  wireSql();

  try {
    state.config = await api("/config");
    $("env-llm").textContent = state.config.llm_available ? state.config.model : "keyword fallback";
    $("env-llm").title = state.config.llm_available
      ? `effort: ${state.config.effort}`
      : "No ANTHROPIC_API_KEY — the NL layer uses its deterministic parser";

    // Don't offer a control that cannot work. Agent mode needs credentials, and
    // letting it be ticked without them just trades a useful answer for a 503.
    if (!state.config.llm_available) {
      const agent = $("ask-agent");
      agent.checked = false;
      agent.disabled = true;
      const label = agent.closest(".checkline");
      label.title = "Needs ANTHROPIC_API_KEY. Everything else works without it.";
      label.style.opacity = "0.55";
      label.lastChild.textContent = " Agent mode (needs ANTHROPIC_API_KEY)";
    }
    $("env-warehouse").textContent = state.config.warehouse.replace(/^duckdb:\/\/\//, "");
    $("env-warehouse").title = state.config.warehouse;
    $("env-build").textContent = state.config.ui_build || "—";
    $("env-build").title =
      "Fingerprint of the UI files this page loaded. If it does not change after " +
      "an update, your browser served a cached copy — hard-refresh with Ctrl+F5.";
  } catch (error) {
    setStatus("ask-status", "error", `Cannot reach the API: ${error.message}`);
  }

  await refreshDatasets();
  refreshSources();
  checkSuperset();
}

window.switchViewPublic = (name) => switchView(name);
function switchView(name) {
  document.querySelectorAll(".view").forEach((view) => {
    view.hidden = view.id !== `view-${name}`;
  });
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.setAttribute("aria-current", String(button.dataset.view === name));
  });
  location.hash = name;
  if (name === "dashboard") loadDashboardActivity();
  if (name === "ask") loadExploreActivity();
  if (name === "ask-nlp") loadAskActivity();
}

function toggleTheme() {
  const root = document.documentElement;
  const isDark =
    root.dataset.theme === "dark" ||
    (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  root.dataset.theme = isDark ? "light" : "dark";
  localStorage.setItem("insight-theme", root.dataset.theme);
}

async function checkSuperset() {
  const node = $("env-superset");
  try {
    const result = await api("/superset/check");
    node.textContent = result.connected ? "connected" : "offline";
    node.style.color = result.connected ? "var(--ok)" : "var(--faint)";
    node.title = result.connected ? "" : result.error || "";
  } catch {
    node.textContent = "offline";
  }
}

async function refreshDatasets() {
  try {
    state.datasets = await api("/datasets");
  } catch {
    state.datasets = [];
  }
  const names = state.datasets.map((d) => d.name);

  fillSelect($("ask-dataset"), names, { placeholder: "any dataset" });
  fillSelect($("ask-nlp-dataset"), names, { placeholder: "any dataset" });
  fillSelect($("data-dataset"), names);
  fillSelect($("an-dataset"), names);

  // A picked table that got dropped from the catalog (re-ingest, rename) can't
  // stay selected silently — drop it here so the chip list matches reality.
  state.dbDatasets = state.dbDatasets.filter((name) => names.includes(name));
  renderDbDatasetChips();

  // Seed the SQL console with something that actually runs here, rather than a
  // sample naming a table this warehouse may not have.
  const sqlBox = $("sql-text");
  if (!sqlBox.value.trim() && names.length) {
    sqlBox.value = `SELECT *\nFROM ${names[0]}\nLIMIT 20`;
  }

  renderDataDetail();
  renderTargets();
  renderExamples();
}

function fillSelect(select, values, { placeholder = null } = {}) {
  const previous = select.value;
  select.innerHTML = "";
  if (placeholder) select.appendChild(new Option(placeholder, ""));
  for (const value of values) select.appendChild(new Option(value, value));
  if (values.includes(previous)) select.value = previous;
}

// --------------------------------------------------------------------- ASK
function wireAsk() {
  $("ask-run").addEventListener("click", runAsk);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter") runAsk();
  });
  $("ask-publish").addEventListener("change", (event) => {
    $("ask-dashboard").hidden = !event.target.checked;
  });
  $("ask-activity-clear").addEventListener("click", clearExploreActivity);
}

// ------------------------------------------------------------------- ASK NLP
function wireAskNlp() {
  $("ask-nlp-run").addEventListener("click", runAskNlp);
  $("ask-nlp-question").addEventListener("keydown", (event) => {
    if (event.key === "Enter") runAskNlp();
  });
  $("ask-nlp-activity-clear").addEventListener("click", clearAskActivity);
}

async function runAskNlp() {
  const question = $("ask-nlp-question").value.trim();
  if (!question) return;

  const dataset = $("ask-nlp-dataset").value;

  setStatus("ask-nlp-status", "busy", "Analyzing your data…");
  $("ask-nlp-result").innerHTML = "";
  $("ask-nlp-run").disabled = true;

  try {
    const result = await api("/ask-nlp", {
      method: "POST",
      body: {
        question,
        datasets: dataset ? [dataset] : null,
      },
    });
    setStatus("ask-nlp-status", null);
    renderAskNlpResult(result);
    loadAskActivity();
  } catch (error) {
    setStatus("ask-nlp-status", "error", error.message);
  } finally {
    $("ask-nlp-run").disabled = false;
  }
}

function renderAskNlpResult(result) {
  const container = $("ask-nlp-result");
  container.innerHTML = "";

  container.appendChild(
    card("Answer", html(`<div style="white-space:pre-wrap;line-height:1.6">${escapeHtml(result.answer)}</div>`))
  );

  if (result.insights?.length) {
    const body = document.createElement("div");
    result.insights.forEach((insight) => {
      body.appendChild(html(`<div class="rec-meta" style="margin-bottom:8px">• ${escapeHtml(insight)}</div>`));
    });
    container.appendChild(card("Key Insights", body));
  }
}

function renderExamples() {
  const container = $("ask-examples");
  container.innerHTML = "";
  if (!state.datasets.length) return;

  const examples = [
    "monthly revenue by region",
    "top 5 product categories by total revenue in 2023",
    "average delivery days for Online orders in the West",
    "how many orders by channel",
  ];
  for (const text of examples) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = text;
    chip.addEventListener("click", () => {
      $("question").value = text;
      runAsk();
    });
    container.appendChild(chip);
  }
}

async function runAsk() {
  const question = $("question").value.trim();
  if (!question) return;

  const dataset = $("ask-dataset").value;
  const agent = $("ask-agent").checked;
  const publish = $("ask-publish").checked;

  setStatus("ask-status", "busy", agent ? "Agent is exploring the data…" : "Translating…");
  $("ask-result").innerHTML = "";
  $("ask-run").disabled = true;

  try {
    const result = await api("/ask", {
      method: "POST",
      body: {
        question,
        datasets: dataset ? [dataset] : null,
        agent,
        publish,
        dashboard: $("ask-dashboard").value || null,
        max_rows: 500,
      },
    });
    setStatus("ask-status", null);
    if (result.mode === "agent") renderAgentResult(result);
    else renderAskResult(result);
    loadExploreActivity();
  } catch (error) {
    setStatus("ask-status", "error", error.message);
  } finally {
    $("ask-run").disabled = false;
  }
}

function renderAskResult(result) {
  state.lastAsk = result;
  const container = $("ask-result");
  container.innerHTML = "";

  const spec = result.spec;

  // A dashboard request gets an action, not an instruction. Telling someone to go
  // and use a different view is a worse answer than doing it for them.
  if (result.suggestion?.type === "dashboard") {
    const prompt = html(`
      <div class="suggest">
        <div class="suggest-text">${escapeHtml(result.suggestion.message)}</div>
        <button class="primary" id="suggest-dashboard">${escapeHtml(result.suggestion.action)}</button>
      </div>`);
    container.appendChild(prompt);
    prompt.querySelector("#suggest-dashboard").addEventListener("click", () => {
      switchView("dashboard");
      state.dbDatasets = [result.suggestion.dataset];
      renderDbDatasetChips();
      $("db-request").value = result.suggestion.request;
      if (!$("db-title").value) $("db-title").value = titleCase(result.suggestion.dataset);
      runDashboard(false);
    });
  }

  if (spec.explanation) {
    container.appendChild(
      card("Interpretation", html(`<p style="margin:0">${escapeHtml(spec.explanation)}</p>`))
    );
  }
  for (const warning of result.warnings || []) {
    container.appendChild(html(`<div class="notice warn">${escapeHtml(warning)}</div>`));
  }

  // Chart + table share one card behind tabs — the same result, two readings.
  const dims = spec.dimensions.map((d) => (d.time_grain !== "none" ? `${d.column}_${d.time_grain}` : d.alias || d.column));
  const metrics = spec.metrics.map((m) => m.alias || `${m.func}_${m.column}`);

  const body = document.createElement("div");
  const tabs = html(`
    <div class="tabs">
      <button class="tab" data-tab="chart" aria-selected="true">Chart</button>
      <button class="tab" data-tab="table" aria-selected="false">Table (${result.row_count.toLocaleString()})</button>
      <button class="tab" data-tab="sql" aria-selected="false">SQL</button>
      <button class="tab" data-tab="spec" aria-selected="false">Spec</button>
    </div>`);
  const panel = document.createElement("div");
  body.append(tabs, panel);

  const draw = (tab) => {
    panel.innerHTML = "";
    if (tab === "chart") {
      renderChart(panel, {
        chart: spec.chart,
        columns: result.columns,
        rows: result.rows,
        dimensions: dims.filter((d) => result.columns.includes(d)),
        metrics: metrics.filter((m) => result.columns.includes(m)),
      });
    } else if (tab === "table") {
      panel.appendChild(renderTable(result.columns, result.rows, 200));
    } else if (tab === "sql") {
      panel.appendChild(html(`<pre class="sql">${escapeHtml(result.sql)}</pre>`));
    } else {
      panel.appendChild(html(`<pre class="code">${escapeHtml(JSON.stringify(spec, null, 2))}</pre>`));
    }
    tabs.querySelectorAll(".tab").forEach((button) => {
      button.setAttribute("aria-selected", String(button.dataset.tab === tab));
    });
  };
  tabs.addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (tab) draw(tab.dataset.tab);
  });
  draw("chart");

  const shell = card(spec.title || "Result", body, { flush: true });
  shell.style.padding = "14px 16px 16px";
  container.appendChild(shell);

  if (result.superset) {
    const links = html(`
      <div>
        <a href="${escapeHtml(result.superset.chart_url)}" target="_blank" rel="noopener">Open chart in Superset</a>
        ${result.superset.dashboard_url
          ? ` · <a href="${escapeHtml(result.superset.dashboard_url)}" target="_blank" rel="noopener">Open dashboard</a>`
          : ""}
        ${(result.superset.notes || []).map((n) => `<div class="rec-meta">${escapeHtml(n)}</div>`).join("")}
      </div>`);
    container.appendChild(card("Published", links));
  }
}

function renderAgentResult(result) {
  const container = $("ask-result");
  container.innerHTML = "";
  container.appendChild(
    card("Answer", html(`<div style="white-space:pre-wrap">${escapeHtml(result.answer)}</div>`))
  );
  if (result.queries?.length) {
    const body = document.createElement("div");
    result.queries.forEach((sql, i) => {
      body.appendChild(html(`<div class="rec-meta" style="margin-top:8px">query ${i + 1}</div>`));
      body.appendChild(html(`<pre class="sql">${escapeHtml(sql)}</pre>`));
    });
    container.appendChild(card(`Queries it ran (${result.queries.length})`, body));
  }
}

// -------------------------------------------------------------------- DATA
function wireData() {
  $("data-dataset").addEventListener("change", renderDataDetail);
}

async function renderDataDetail() {
  const container = $("data-detail");
  const name = $("data-dataset").value;
  if (!name) {
    container.innerHTML = '<div class="card"><div class="empty">Nothing ingested yet. Add a source first.</div></div>';
    return;
  }

  const meta = state.datasets.find((d) => d.name === name);
  if (!meta) return;

  container.innerHTML = "";
  container.appendChild(
    card(
      "Overview",
      html(`
        <div class="metric-grid">
          <div class="metric"><div class="label">Rows</div><div class="value">${meta.n_rows.toLocaleString()}</div></div>
          <div class="metric"><div class="label">Columns</div><div class="value">${meta.columns.length}</div></div>
          <div class="metric"><div class="label">Source</div><div class="value" style="font-size:14px">${escapeHtml(meta.source)}</div>
            <div class="against">${escapeHtml(meta.origin_object)}</div></div>
          <div class="metric"><div class="label">Grain</div><div class="value" style="font-size:14px">${escapeHtml(meta.grain || "—")}</div></div>
        </div>`)
    )
  );

  const columns = document.createElement("div");
  columns.className = "col-list";
  for (const column of meta.columns) {
    const facts = [];
    if (column.n_unique != null) facts.push(`${column.n_unique.toLocaleString()} distinct`);
    if (column.null_fraction) facts.push(`${(column.null_fraction * 100).toFixed(1)}% null`);
    if (column.min != null) facts.push(`${formatCell(column.min)} … ${formatCell(column.max)}`);
    if (column.sample_values?.length) facts.push(column.sample_values.slice(0, 5).join(", "));

    columns.appendChild(
      html(`
        <div class="col-row">
          <span class="name">${escapeHtml(column.name)}</span>
          <span class="type-tag type-${escapeHtml(column.semantic_type)}">${escapeHtml(column.semantic_type)}</span>
          <span class="facts" title="${escapeHtml(facts.join(" · "))}">${escapeHtml(facts.join(" · "))}</span>
        </div>`)
    );
  }
  container.appendChild(card(`Schema — as the language layer sees it`, columns, { flush: true }));

  try {
    const preview = await api(`/datasets/${encodeURIComponent(name)}/preview?limit=25`);
    container.appendChild(card("Preview", renderTable(preview.columns, preview.rows, 25), { flush: true }));
  } catch (error) {
    container.appendChild(html(`<div class="notice error">${escapeHtml(error.message)}</div>`));
  }
}

// ----------------------------------------------------------------- SOURCES
function wireSources() {
  $("src-add").addEventListener("click", addSource);
}

async function addSource() {
  const name = $("src-name").value.trim();
  const uri = $("src-uri").value.trim();
  if (!name || !uri) {
    setStatus("sources-status", "warn", "Name and URI are both required.");
    return;
  }
  setStatus("sources-status", "busy", `Connecting to ${name}…`);
  try {
    await api("/sources", {
      method: "POST",
      body: { name, uri, type: $("src-type").value || null },
    });
    $("src-name").value = "";
    $("src-uri").value = "";
    setStatus("sources-status", "info", `Registered ${name}. Pick objects to ingest below.`);
    await refreshSources();
  } catch (error) {
    setStatus("sources-status", "error", error.message);
  }
}

async function refreshSources() {
  try {
    state.sources = await api("/sources");
  } catch {
    state.sources = [];
  }
  const container = $("sources-list");
  container.innerHTML = "";

  if (!state.sources.length) {
    container.appendChild(card(null, html('<div class="empty">No sources registered yet.</div>')));
    return;
  }

  for (const source of state.sources) {
    const body = document.createElement("div");
    body.appendChild(
      html(`
        <div class="row" style="align-items:center">
          <div class="grow">
            <strong>${escapeHtml(source.name)}</strong>
            <span class="badge">${escapeHtml(source.type)}</span>
            <div class="rec-meta" style="font-family:var(--mono)">${escapeHtml(source.uri)}</div>
          </div>
          <button class="tiny" data-act="objects">List objects</button>
          <button class="tiny primary" data-act="ingest">Ingest all</button>
          <button class="tiny ghost" data-act="remove">Remove</button>
        </div>`)
    );
    const detail = document.createElement("div");
    detail.style.marginTop = "10px";
    body.appendChild(detail);

    body.addEventListener("click", async (event) => {
      const action = event.target.dataset?.act;
      if (!action) return;

      if (action === "remove") {
        if (!confirm(`Remove source "${source.name}" and its ingested datasets?`)) return;
        await api(`/sources/${encodeURIComponent(source.name)}`, { method: "DELETE" });
        await refreshSources();
        await refreshDatasets();
        return;
      }

      detail.innerHTML = '<span class="spinner"></span>';
      try {
        if (action === "objects") {
          const objects = await api(`/sources/${encodeURIComponent(source.name)}/objects`);
          detail.innerHTML = "";
          const chips = document.createElement("div");
          chips.className = "chips";
          for (const name of objects) {
            const chip = document.createElement("button");
            chip.className = "chip";
            chip.textContent = name;
            chip.title = "Ingest just this object";
            chip.addEventListener("click", async () => {
              detail.innerHTML = `<span class="spinner"></span> ingesting ${escapeHtml(name)}…`;
              const results = await api("/ingest", { method: "POST", body: { source: source.name, object: name } });
              detail.innerHTML = results.map(ingestLine).join("");
              await refreshDatasets();
            });
            chips.appendChild(chip);
          }
          detail.appendChild(chips);
        } else {
          const results = await api("/ingest", { method: "POST", body: { source: source.name } });
          detail.innerHTML = results.map(ingestLine).join("");
          await refreshDatasets();
        }
      } catch (error) {
        detail.innerHTML = `<div class="notice error">${escapeHtml(error.message)}</div>`;
      }
    });

    container.appendChild(card(null, body));
  }
}

function ingestLine(result) {
  const failed = result.dataset.startsWith("!");
  return failed
    ? `<div class="notice error">${escapeHtml(result.origin)}</div>`
    : `<div class="rec-meta">${escapeHtml(result.dataset)} — ${result.rows.toLocaleString()} rows, ${result.columns} columns</div>`;
}

// --------------------------------------------------------------- DASHBOARD
function wireDashboard() {
  $("db-preview").addEventListener("click", () => runDashboard(false));
  $("db-publish").addEventListener("click", () => runDashboard(true));
  $("db-activity-clear").addEventListener("click", clearDashboardActivity);

  // Composing is a 5-30ms round trip with no warehouse query behind it, so the
  // plan can track the text as you type. Publishing stays an explicit click —
  // that one writes to Superset.
  let timer = null;
  const live = () => {
    clearTimeout(timer);
    timer = setTimeout(() => runDashboard(false, { live: true }), 250);
  };
  $("db-request").addEventListener("input", live);

  $("db-dataset-add").addEventListener("change", (event) => {
    const name = event.target.value;
    if (!name) return;
    event.target.value = "";
    if (!state.dbDatasets.includes(name)) {
      state.dbDatasets.push(name);
      renderDbDatasetChips();
      live();
    }
  });

  renderDbDatasetChips();
}

// The "add a table" dropdown only ever offers tables not already picked —
// re-adding one is a removal, not a no-op, and a second entry would just
// duplicate every tile it produces.
function renderDbDatasetChips() {
  const names = state.datasets.map((d) => d.name);
  fillSelect($("db-dataset-add"), names.filter((n) => !state.dbDatasets.includes(n)), {
    placeholder: "+ add a table…",
  });

  const chips = $("db-dataset-chips");
  chips.innerHTML = "";
  for (const name of state.dbDatasets) {
    const chip = html(`
      <span class="chip-selected">${escapeHtml(name)}<button type="button" aria-label="remove ${escapeHtml(name)}">×</button></span>
    `);
    chip.querySelector("button").addEventListener("click", () => {
      state.dbDatasets = state.dbDatasets.filter((n) => n !== name);
      renderDbDatasetChips();
      runDashboard(false, { live: true });
    });
    chips.appendChild(chip);
  }
}

// Monotonic guard: a slow early response must not overwrite a newer plan.
let dashboardRequestId = 0;

async function runDashboard(publish, { live = false } = {}) {
  const datasets = state.dbDatasets;
  if (!datasets.length) {
    if (!live) setStatus("db-status", "warn", "Add at least one table first.");
    $("db-result").innerHTML = "";
    return;
  }

  const ticket = ++dashboardRequestId;
  if (!live) setStatus("db-status", "busy", publish ? "Composing and publishing…" : "Composing…");
  if (publish) $("db-preview").disabled = $("db-publish").disabled = true;

  try {
    const plan = await api("/dashboard", {
      method: "POST",
      body: { datasets, request: $("db-request").value, title: $("db-title").value, publish, live },
    });
    if (ticket !== dashboardRequestId) return; // superseded while in flight
    setStatus("db-status", null);
    renderDashboardPlan(plan, { live });
    if (!live) loadDashboardActivity();
  } catch (error) {
    if (ticket !== dashboardRequestId) return;
    setStatus("db-status", "error", error.message);
  } finally {
    $("db-preview").disabled = $("db-publish").disabled = false;
  }
}

function renderDashboardPlan(plan, { live = false } = {}) {
  const container = $("db-result");
  container.innerHTML = "";

  const byTitle = new Map(plan.tiles.map((t) => [t.title, t]));
  const wrap = document.createElement("div");

  if (plan.interpretation) {
    wrap.appendChild(html(`<div class="notice info" style="margin-bottom:12px">${escapeHtml(plan.interpretation)}</div>`));
  }
  if (plan.description) {
    wrap.appendChild(html(`<div class="rec-meta" style="margin-bottom:12px">${escapeHtml(plan.description)}</div>`));
  }

  // Visual grid — each tile is clickable
  const grid = document.createElement("div");
  for (const row of plan.rows) {
    const rowNode = document.createElement("div");
    rowNode.className = "layout-row";
    for (const title of row) {
      const tile = byTitle.get(title) || { width: 6, role: "breakdown", chart: "table" };
      const cell = html(`
        <div class="layout-tile role-${escapeHtml(tile.role)}" style="flex:${tile.width}" data-title="${escapeHtml(title)}" tabindex="0" role="button" aria-label="View SQL for ${escapeHtml(title)}">
          <div class="layout-title">${escapeHtml(title)}</div>
          <div class="layout-meta">${escapeHtml(tile.chart)} · ${tile.width}/12</div>
        </div>`);
      rowNode.appendChild(cell);
    }
    grid.appendChild(rowNode);
  }
  wrap.appendChild(grid);

  // SQL panel — revealed when a tile is clicked
  const sqlPanel = html(`
    <div class="db-sql-panel">
      <div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:6px;gap:8px">
        <div style="min-width:0">
          <span class="db-sql-tile-name" style="font-weight:600;font-size:14px"></span>
          <span class="db-sql-tile-meta" style="color:var(--faint);font-size:12px;margin-left:8px"></span>
        </div>
        <button class="tiny ghost db-sql-copy">Copy SQL</button>
      </div>
      <div class="db-sql-explanation" style="color:var(--muted);font-size:13px;margin-bottom:8px"></div>
      <textarea class="db-sql-editor" spellcheck="false" rows="6"></textarea>
    </div>`);
  wrap.appendChild(sqlPanel);

  const combinedCard = card(`${plan.title} — ${plan.tiles.length} tiles`, wrap);
  container.appendChild(combinedCard);

  // Wire tile clicks
  function selectTile(titleKey) {
    for (const b of combinedCard.querySelectorAll(".layout-tile")) b.classList.remove("active");
    const btn = combinedCard.querySelector(`.layout-tile[data-title="${CSS.escape(titleKey)}"]`);
    if (btn) btn.classList.add("active");
    const tile = byTitle.get(titleKey);
    if (!tile) return;
    sqlPanel.style.display = "block";
    sqlPanel.querySelector(".db-sql-tile-name").textContent = tile.title;
    sqlPanel.querySelector(".db-sql-tile-meta").textContent = `${tile.chart} · ${tile.width}/12`;
    sqlPanel.querySelector(".db-sql-explanation").textContent = tile.explanation || "";
    sqlPanel.querySelector(".db-sql-editor").value = tile.sql || "";
  }

  for (const cell of combinedCard.querySelectorAll(".layout-tile")) {
    cell.addEventListener("click", () => selectTile(cell.dataset.title));
    cell.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") selectTile(cell.dataset.title); });
  }

  sqlPanel.querySelector(".db-sql-copy").addEventListener("click", () => {
    const txt = sqlPanel.querySelector(".db-sql-editor").value;
    navigator.clipboard.writeText(txt).catch(() => {});
  });

  // Auto-select first tile
  const firstTile = combinedCard.querySelector(".layout-tile");
  if (firstTile) selectTile(firstTile.dataset.title);

  for (const problem of plan.problems || []) {
    container.appendChild(html(`<div class="notice warn">${escapeHtml(problem)}</div>`));
  }

  if (plan.superset) {
    container.appendChild(
      card(
        "Published",
        html(`
          <div>
            <a href="${escapeHtml(plan.superset.dashboard_url)}" target="_blank" rel="noopener">Open dashboard in Superset</a>
            <div class="rec-meta">${plan.superset.chart_ids.length} charts created</div>
            ${(plan.superset.notes || []).map((n) => `<div class="rec-meta">${escapeHtml(n)}</div>`).join("")}
            <div class="rec-meta" style="margin-top:6px">If Superset shows a login page, sign in there first — it keeps its own browser session.</div>
          </div>`)
      )
    );
  }
}

// --------------------------------------------------------- DASHBOARD ACTIVITY
async function clearDashboardActivity() {
  if (!confirm("Clear all dashboard activity? This cannot be undone.")) return;
  try {
    await api("/dashboard-activity", { method: "DELETE" });
    await loadDashboardActivity();
  } catch (error) {
    setStatus("db-status", "error", error.message);
  }
}

async function loadDashboardActivity() {
  const container = $("db-activity");
  try {
    const entries = await api("/dashboard-activity", { method: "GET" });
    renderDashboardActivity(entries, container);
  } catch (_) {
    // silently skip — activity is non-critical
  }
}

function renderDashboardActivity(entries, container) {
  if (!entries || !entries.length) {
    container.innerHTML = `<span style="color:var(--faint);font-size:13px">No activity yet.</span>`;
    return;
  }
  container.innerHTML = "";
  for (const entry of entries) {
    const date = fmtActivityDate(entry.created_at);
    const datasets = (entry.datasets || []).join(", ");
    const isPublish = entry.action === "publish";
    const titleNode = isPublish && entry.dashboard_url
      ? `<a href="${escapeHtml(entry.dashboard_url)}" target="_blank" rel="noopener" class="activity-title">${escapeHtml(entry.title)}</a>`
      : `<span class="activity-title">${escapeHtml(entry.title)}</span>`;
    const row = html(`
      <div class="activity-item clickable" title="Click to preview this dashboard again">
        <div class="activity-top">
          <span class="activity-badge ${isPublish ? "publish" : "preview"}">${isPublish ? "Published" : "Preview"}</span>
          ${titleNode}
        </div>
        ${entry.request ? `<div class="activity-request">“${escapeHtml(entry.request)}”</div>` : ""}
        <div class="activity-meta">
          ${datasets ? escapeHtml(datasets) : ""}${datasets && entry.n_charts ? " · " : ""}${entry.n_charts ? `${entry.n_charts} chart${entry.n_charts !== 1 ? "s" : ""}` : ""}
        </div>
        <div class="activity-meta">${escapeHtml(date)}</div>
      </div>`);
    row.addEventListener("click", (event) => {
      if (event.target.closest("a")) return; // let the published-dashboard link behave normally
      state.dbDatasets = [...(entry.datasets || [])];
      renderDbDatasetChips();
      $("db-title").value = entry.title;
      $("db-request").value = entry.request || "";
      runDashboard(entry.action === "publish"); // replay the same action it logged
    });
    container.appendChild(row);
  }
}

function fmtActivityDate(createdAt) {
  return createdAt
    ? new Date(createdAt).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
    : "";
}

// ----------------------------------------------------------- EXPLORE ACTIVITY
async function clearExploreActivity() {
  if (!confirm("Clear all Explore activity? This cannot be undone.")) return;
  try {
    await api("/explore-activity", { method: "DELETE" });
    await loadExploreActivity();
  } catch (error) {
    setStatus("ask-status", "error", error.message);
  }
}

async function loadExploreActivity() {
  const container = $("ask-activity");
  try {
    const entries = await api("/explore-activity", { method: "GET" });
    renderExploreActivity(entries, container);
  } catch (_) {
    // silently skip — activity is non-critical
  }
}

function renderExploreActivity(entries, container) {
  if (!entries || !entries.length) {
    container.innerHTML = `<span style="color:var(--faint);font-size:13px">No activity yet.</span>`;
    return;
  }
  container.innerHTML = "";
  for (const entry of entries) {
    const date = fmtActivityDate(entry.created_at);
    // A publish without a dashboard title creates a standalone chart, which
    // has no dashboard_url — fall back to chart_url so a published entry
    // always shows the badge, never looking unpublished.
    const publishLink = entry.dashboard_url || entry.chart_url;
    const publishedNote = entry.published
      ? (publishLink
          ? `<a href="${escapeHtml(publishLink)}" target="_blank" rel="noopener" class="activity-badge publish" style="text-decoration:none">Published</a>`
          : `<span class="activity-badge publish">Published</span>`)
      : "";
    const row = html(`
      <div class="activity-item clickable" title="Click to run this question again">
        <div class="activity-top">
          <span class="activity-badge ${entry.mode}">${escapeHtml(entry.mode)}</span>
          ${publishedNote}
        </div>
        <div class="activity-request">“${escapeHtml(entry.question)}”</div>
        <div class="activity-meta">${entry.row_count != null ? `${entry.row_count} rows` : ""}</div>
        <div class="activity-meta">${escapeHtml(date)}</div>
      </div>`);
    row.addEventListener("click", (event) => {
      if (event.target.closest("a")) return; // let the "Published" link behave normally
      $("question").value = entry.question;
      $("ask-agent").checked = entry.mode === "agent";
      $("ask-publish").checked = !!entry.published;
      $("ask-dashboard").hidden = !entry.published; // matches the checkbox's own change handler
      runAsk();
    });
    container.appendChild(row);
  }
}

// --------------------------------------------------------------- ASK ACTIVITY
async function clearAskActivity() {
  if (!confirm("Clear all Ask activity? This cannot be undone.")) return;
  try {
    await api("/ask-activity", { method: "DELETE" });
    await loadAskActivity();
  } catch (error) {
    setStatus("ask-nlp-status", "error", error.message);
  }
}

async function loadAskActivity() {
  const container = $("ask-nlp-activity");
  try {
    const entries = await api("/ask-activity", { method: "GET" });
    renderAskActivity(entries, container);
  } catch (_) {
    // silently skip — activity is non-critical
  }
}

function renderAskActivity(entries, container) {
  if (!entries || !entries.length) {
    container.innerHTML = `<span style="color:var(--faint);font-size:13px">No activity yet.</span>`;
    return;
  }
  container.innerHTML = "";
  for (const entry of entries) {
    const date = fmtActivityDate(entry.created_at);
    const row = html(`
      <div class="activity-item clickable" title="Click to ask this question again">
        <div class="activity-request">“${escapeHtml(entry.question)}”</div>
        <div class="activity-meta">${escapeHtml(entry.dataset || "any dataset")}</div>
        <div class="activity-meta">${escapeHtml(date)}</div>
      </div>`);
    row.addEventListener("click", () => {
      $("ask-nlp-question").value = entry.question;
      $("ask-nlp-dataset").value = entry.dataset || "";
      runAskNlp();
    });
    container.appendChild(row);
  }
}

// ----------------------------------------------------------------- ANALYZE
function wireAnalyze() {
  $("an-dataset").addEventListener("change", renderTargets);
  $("an-run").addEventListener("click", runAnalyze);
  $("an-baseline").addEventListener("click", runBaseline);
}

function renderTargets() {
  const meta = state.datasets.find((d) => d.name === $("an-dataset").value);
  const select = $("an-target");
  const previous = select.value;
  select.innerHTML = "";
  select.appendChild(new Option("none — unsupervised", ""));
  if (!meta) return;
  for (const column of meta.columns) {
    // Identifiers are never a sensible prediction target.
    if (column.semantic_type === "identifier") continue;
    select.appendChild(new Option(`${column.name} (${column.semantic_type})`, column.name));
  }
  if ([...select.options].some((o) => o.value === previous)) select.value = previous;
}

async function runAnalyze() {
  const dataset = $("an-dataset").value;
  if (!dataset) return;
  setStatus("an-status", "busy", "Profiling, detecting patterns, ranking algorithms…");
  $("an-result").innerHTML = "";
  $("an-run").disabled = true;

  try {
    const report = await api("/analyze", {
      method: "POST",
      body: { dataset, target: $("an-target").value || null },
    });
    setStatus("an-status", null);
    renderReport(report);
  } catch (error) {
    setStatus("an-status", "error", error.message);
  } finally {
    $("an-run").disabled = false;
  }
}

function renderReport(report) {
  const container = $("an-result");
  container.innerHTML = "";
  const profile = report.profile;

  container.appendChild(
    card(
      "Profile",
      html(`
        <div class="metric-grid">
          <div class="metric"><div class="label">Rows</div><div class="value">${profile.n_rows.toLocaleString()}</div></div>
          <div class="metric"><div class="label">Columns</div><div class="value">${profile.n_columns}</div></div>
          <div class="metric"><div class="label">Duplicate rows</div><div class="value">${profile.n_duplicate_rows.toLocaleString()}</div></div>
          <div class="metric"><div class="label">Inferred task</div><div class="value" style="font-size:15px">${escapeHtml(report.task)}</div>
            <div class="against">${report.target ? "target: " + escapeHtml(report.target) : "no target given"}</div></div>
        </div>`)
    )
  );

  // --- patterns, strongest first (the API already sorts them) ---------------
  const counts = { strong: 0, notable: 0, info: 0 };
  for (const pattern of report.patterns) counts[pattern.severity] += 1;

  const list = document.createElement("div");
  for (const pattern of report.patterns) {
    const stat = pattern.statistic != null ? `<span class="stat">${formatNumber(pattern.statistic)}</span>` : "";
    const p = pattern.p_value != null ? `<span class="stat"> p=${Number(pattern.p_value).toExponential(1)}</span>` : "";
    list.appendChild(
      html(`
        <div class="pattern severity-${pattern.severity}">
          <span class="kind">${escapeHtml(pattern.kind)}</span>
          <span class="body">
            <div>${escapeHtml(pattern.description)} ${stat}${p}</div>
            ${pattern.implication ? `<div class="implication">${escapeHtml(pattern.implication)}</div>` : ""}
          </span>
        </div>`)
    );
  }
  const patternCard = card(null, list, { flush: true });
  patternCard.prepend(
    html(`
      <div style="padding:14px 16px 10px;display:flex;gap:8px;align-items:center">
        <h3 style="margin:0;flex:1">Patterns detected (${report.patterns.length})</h3>
        <span class="badge strong"><span class="dot"></span>${counts.strong} strong</span>
        <span class="badge notable"><span class="dot"></span>${counts.notable} notable</span>
        <span class="badge info"><span class="dot"></span>${counts.info} info</span>
      </div>`)
  );
  container.appendChild(patternCard);

  // --- recommendations -----------------------------------------------------
  const recs = document.createElement("div");
  for (const rec of report.recommendations) {
    const details = document.createElement("details");
    details.className = "rec";
    if (rec.rank <= 2) details.open = true;

    details.appendChild(
      html(`
        <summary class="rec-head">
          <span class="rec-rank">${rec.rank}</span>
          <span class="rec-name">${escapeHtml(rec.algorithm)}</span>
          <span class="rec-meta">${escapeHtml(rec.task)} · ${escapeHtml(rec.library)} · confidence ${escapeHtml(rec.confidence)}</span>
          <span class="chev">›</span>
        </summary>`)
    );

    const body = document.createElement("div");
    body.className = "rec-body";
    const section = (title, items, cls = "") =>
      items?.length
        ? `<h4>${title}</h4><ul>${items.map((i) => `<li class="${cls}">${escapeHtml(i)}</li>`).join("")}</ul>`
        : "";

    body.innerHTML =
      section("Why this, on this data", rec.rationale) +
      section("Preprocessing", rec.preprocessing) +
      (rec.evaluation ? `<h4>Evaluate with</h4><div>${escapeHtml(rec.evaluation)}</div>` : "") +
      section("Caveats", rec.caveats, "caveat") +
      (rec.starter_code ? `<h4>Starter code</h4><pre class="code">${escapeHtml(rec.starter_code)}</pre>` : "");
    details.appendChild(body);
    recs.appendChild(details);
  }
  container.appendChild(card(`Recommended algorithms (${report.recommendations.length})`, recs));
}

async function runBaseline() {
  const dataset = $("an-dataset").value;
  const target = $("an-target").value;
  if (!target) {
    setStatus("an-status", "warn", "Pick a target column — a baseline needs something to predict.");
    return;
  }
  setStatus("an-status", "busy", "Fitting and cross-validating…");
  $("an-baseline").disabled = true;

  try {
    const result = await api("/baseline", { method: "POST", body: { dataset, target } });
    setStatus("an-status", null);
    renderBaseline(result);
  } catch (error) {
    setStatus("an-status", "error", error.message);
  } finally {
    $("an-baseline").disabled = false;
  }
}

function renderBaseline(result) {
  const container = $("an-result");
  const body = document.createElement("div");

  // Pair each metric with its dummy counterpart — the model number is
  // uninterpretable on its own, which is the whole point of showing both.
  const grid = document.createElement("div");
  grid.className = "metric-grid";
  for (const [name, value] of Object.entries(result.metrics)) {
    if (name.startsWith("baseline_")) continue;
    const reference = result.metrics[`baseline_${name}`];
    grid.appendChild(
      html(`
        <div class="metric">
          <div class="label">${escapeHtml(name.replace(/_/g, " "))}</div>
          <div class="value">${formatMetric(value)}</div>
          <div class="against">${reference != null ? `baseline ${formatMetric(reference)}` : "&nbsp;"}</div>
        </div>`)
    );
  }
  body.appendChild(
    html(`<div class="rec-meta" style="margin-bottom:10px">${escapeHtml(result.algorithm)} · ${result.n_train.toLocaleString()} rows · ${result.n_features} features · ${result.cv_folds}-fold CV</div>`)
  );
  body.appendChild(grid);

  const importance = Object.entries(result.feature_importance || {});
  if (importance.length) {
    const max = Math.max(...importance.map(([, v]) => Math.abs(v))) || 1;
    const bars = document.createElement("div");
    bars.style.marginTop = "14px";
    bars.appendChild(html('<h4 style="margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)">Permutation importance</h4>'));
    for (const [name, value] of importance) {
      bars.appendChild(
        html(`
          <div class="bar-row">
            <span style="font-family:var(--mono);font-size:12px">${escapeHtml(name)}</span>
            <span class="bar-track"><span class="bar-fill" style="width:${(Math.abs(value) / max) * 100}%"></span></span>
            <span class="pct">${(value * 100).toFixed(1)}%</span>
          </div>`)
      );
    }
    body.appendChild(bars);
  }

  for (const note of result.notes || []) {
    body.appendChild(html(`<div class="rec-meta" style="margin-top:8px">${escapeHtml(note)}</div>`));
  }

  container.prepend(card("Baseline fit", body));
}

// --------------------------------------------------------------------- SQL
function wireSql() {
  $("sql-run").addEventListener("click", runSql);
  $("sql-text").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) runSql();
  });
}

async function runSql() {
  const sql = $("sql-text").value.trim();
  if (!sql) return;
  setStatus("sql-status", "busy", "Running…");
  $("sql-result").innerHTML = "";
  try {
    const result = await api("/sql", { method: "POST", body: { sql } });
    setStatus("sql-status", null);
    $("sql-result").appendChild(
      card(`${result.row_count.toLocaleString()} rows`, renderTable(result.columns, result.rows, 200), { flush: true })
    );
  } catch (error) {
    setStatus("sql-status", "error", error.message);
  }
}

// ------------------------------------------------------------------- start
switchView(location.hash ? location.hash.slice(1) : "home");
boot();
