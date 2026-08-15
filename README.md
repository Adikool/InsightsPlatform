# Insight Platform

A Python data platform with three stacked layers:

```
  ┌──────────────────────────────────────────────────────────────┐
  │  UI — one page, served by the same process (insight serve)    │
  │  Ask · Data · Sources · Analyze · SQL                          │
  ├──────────────────────────────────────────────────────────────┤
  │  LAYER 3 — Data Science          patterns → algorithm advice  │
  │  profiling · trend/seasonality/correlation/outliers/drift     │
  │  → ranked model recommendations → baseline training           │
  ├──────────────────────────────────────────────────────────────┤
  │  LAYER 2 — NLP                   question → QuerySpec → SQL   │
  │  Claude (structured outputs + tool use) over a semantic       │
  │  catalog; SQL is *built by us*, never emitted by the model    │
  │  → auto-published Apache Superset datasets/charts/dashboards  │
  ├──────────────────────────────────────────────────────────────┤
  │  LAYER 1 — Connectors            SQL · Excel · CSV · Parquet  │
  │  → ingested into a DuckDB (or Postgres) warehouse             │
  └──────────────────────────────────────────────────────────────┘
```

Python is the binding language throughout: connectors, catalog, NL layer, Superset
publishing, and the ML advisory layer are all one importable package with a CLI and
a FastAPI service.

---

## Why the NL layer emits a *spec*, not SQL

The model never writes SQL text that we execute. It fills a strictly-schema'd
`QuerySpec` (dimensions, metrics, filters, grain, limit). We then compile that spec
to SQL ourselves against the catalog. Consequences:

* every identifier is validated against the catalog allow-list before it reaches SQL;
* the query is read-only by construction — there is no code path that emits DML/DDL;
* a `LIMIT` is always injected;
* the same spec compiles to DuckDB **and** Postgres, and doubles as the Superset
  chart definition, so the dashboard and the answer can never drift apart.

A raw-SQL escape hatch exists (`--allow-raw-sql`) and goes through
`nlp/sql_guard.py`, which rejects anything that is not a single read-only statement
over allow-listed tables.

---

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[ml]"        # scikit-learn / scipy baselines + pattern tests
pip install -e ".[api]"       # the web UI and REST service
pip install -e ".[postgres]"  # psycopg driver for a Postgres warehouse
```

Set your key (the NL layer degrades to a rule-based parser without it):

```bash
setx ANTHROPIC_API_KEY sk-ant-...
```

---

## 60-second demo (no API key needed)

Generates a synthetic retail SQLite DB and an Excel workbook under `./sample`:

```bash
python -m dataplatform.examples.make_sample_data
```

```bash
insight source add --name sales_db --type sql --uri "sqlite:///sample/retail.db"
```

```bash
insight ingest --source sales_db
```

```bash
insight ask "top 5 product categories by total revenue in 2023"
```

```bash
insight analyze --dataset sales_db_orders --target revenue --verbose
```

Without `ANTHROPIC_API_KEY` the NL layer falls back to a keyword parser, which
handles the common `<agg> <measure> by <dimension> for <filter>` shapes and says so
in its `explanation`. Everything else — ingestion, compilation, the guard, Superset
publishing, and all of Layer 3 — works identically without a key.

The demo data is deliberately not clean. It carries a growth trend, December
seasonality, an 18% price rise partway through, right-skewed revenue, structured
missingness, a 28:1 imbalanced return flag, and a column derived from the target.
Layer 3 finds them, which makes the demo a test of the detectors rather than a
decoration:

```
!! target_leakage: cost is almost perfectly correlated with revenue (r=0.994)
!! changepoint: unit_price steps by 20% around 2023-12-02, over and above the trend
!! skewed_distribution: revenue is right-skewed (skew=17.23)
 ! seasonality: revenue repeats every 7 D-periods (autocorrelation=0.44)
 ! structured_missingness: cost and gross_margin are missing together (r=1.00)
```

```bash
insight baseline --dataset sales_db_orders --target revenue
```

fits the top recommendation, drops the leaking columns first, and splits with
`TimeSeriesSplit` because the data is time-ordered — reporting the model's score
next to the dummy's, since the first number means nothing without the second.

---

## The UI

```bash
insight serve
```

Opens the whole platform at `http://127.0.0.1:8000` — same process, same catalog,
same warehouse as the CLI. Five views:

| View | What it does |
|---|---|
| **Ask** | Question → interpretation, chart, table, the generated SQL, and the raw QuerySpec, behind four tabs. Optional agent mode and a publish-to-Superset toggle. One question, one chart. |
| **Dashboard** | Several questions at once. Previews the 12-column grid before publishing, then creates one Superset dashboard: KPI cards along the top, graphs beneath, detail table at the bottom. |
| **Data** | Every ingested dataset, its schema *as the language layer sees it* (semantic type, cardinality, nulls, sample values), and a live preview. |
| **Sources** | Register SQL/Excel/CSV/Parquet sources, list their objects, ingest all or one at a time. |
| **Analyze** | Layer 3 end to end: profile, patterns colour-coded by severity, and expandable recommendations showing rationale, preprocessing, evaluation metric, caveats, and starter code. "Fit baseline" trains the top recommendation and shows its score next to the dummy's. |
| **SQL** | The guarded console. Rejections show the guard's reason. |

**No build step, no npm, no CDN.** Vanilla ES modules and a hand-written SVG chart
renderer, because a `<script src="https://…">` is the first thing to break on an
air-gapped host — there is a test (`test_ui_references_no_external_hosts`) that
fails if anything in the UI reaches off-box. Charts are driven by the same `chart`
field the QuerySpec already carries, so the UI, the CLI and the Superset dashboard
all visualise a result the same way. Light/dark follows the OS with a manual
override; the layout collapses to a single column on mobile.

The one thing worth knowing: the UI is a *client* of the REST API, nothing more.
Every view maps to an endpoint you can drive yourself (`/ask`, `/analyze`,
`/baseline`, `/sql`, `/ingest`), and interactive API docs are at `/docs`.

## Dashboards: several questions, one report

`ask` answers one question with one chart. A dashboard is a different shape — many
queries composed onto a grid — so it has its own path:

```bash
insight dashboard "sales overview with kpi cards, graphs and a detail table" --dataset sales_db_orders --title "Sales Overview"
```

```
row 1: Total revenue [3/12, kpi], Records [3/12, kpi], Average revenue [3/12, kpi], Distinct customer id [3/12, kpi]
row 2: Revenue over time [12/12, trend]
row 3: Revenue by channel [6/12, breakdown], Revenue by region [6/12, breakdown]
row 4: Detail [12/12, detail]
```

`--dry-run` plans without publishing; `--show-sql` prints each tile's query.

**The composer works without an API key.** For "give me a sales dashboard" the
conventional shape — headline numbers, a trend, breakdowns, detail underneath — is
usually the right answer, and deriving it from the catalog's semantic types is
reproducible in a way a model is not. With a key, the request text drives which
tiles get built instead.

Two choices worth knowing about:

* **Tiles are ordered by role, not packed for density.** Reordering to fill rows
  more tightly would move the detail table off the bottom, and "detail table below
  the graphs" is the whole point of the layout.
* **Breakdown dimensions are ranked, not sorted by cardinality.** Sorting by
  cardinality picks the *least* informative columns first — a free-text field that
  is 94% empty has two distinct values and would sort to the top, producing a chart
  with one visible bar. The catalog records each column's top-value share so those
  can be excluded outright.

## Publishing to Superset

```bash
insight superset check
```

```bash
insight ask "monthly revenue by region" --publish --dashboard "Revenue Review"
```

`superset/publisher.py` registers the warehouse as a Superset database, creates a
*virtual dataset* from the compiled SQL, creates a chart whose `viz_type` comes from
the same `QuerySpec`, and pins it onto a dashboard. Verified end to end against a
live Superset 4.x docker-compose stack: the published chart returns the same numbers
the CLI printed, because it is the same compiled SQL.

### Superset must be able to reach the warehouse

This is the one setup step that catches everyone. **Superset normally runs in a
container, so it does not share your filesystem or your `localhost`.** Publishing a
DuckDB-file warehouse fails with a confusing `No such file or directory` for a path
that plainly exists — Superset resolved it inside its own container
(`/app/C:/Users/...`). The publisher now detects that case and refuses up front with
the real reason instead.

Use a networked database both sides can reach, and tell each side its own address:

```bash
docker run -d --name insight_warehouse --network superset_default -p 55432:5432 -e POSTGRES_USER=insight -e POSTGRES_PASSWORD=insight -e POSTGRES_DB=insight postgres:16
```

| Variable | Value | Whose view |
|---|---|---|
| `DP_WAREHOUSE_URI` | `postgresql+psycopg://insight:insight@localhost:55432/insight` | ours, from the host |
| `DP_SUPERSET_WAREHOUSE_URI` | `postgresql+psycopg2://insight:insight@insight_warehouse:5432/insight` | Superset's, from inside its network |

Needs the postgres extra: `pip install -e ".[postgres]"`. Everything else is
unchanged — the same `QuerySpec` compiles to Postgres and DuckDB alike.

---

## Layer 3 in one call

```python
from dataplatform import Platform

p = Platform()
report = p.analyze("sales_db_orders", target="revenue")

report.task                                    # 'regression'
{pat.kind for pat in report.patterns}          # {'trend', 'seasonality', 'target_leakage', ...}
report.recommendations[0].algorithm            # the dummy baseline — always first
report.recommendations[1].rationale            # why, in terms of the measured patterns
p.baseline("sales_db_orders", target="revenue")
```

### What it detects

| Group | Patterns |
|---|---|
| Hygiene | missing data, structured missingness, duplicate rows, constant columns, high cardinality, free text |
| Distribution | skew, outliers (IQR), zero-inflation |
| Relationships | linear correlation, monotone-but-nonlinear pairs, multicollinearity (VIF) |
| Time | coverage and gaps, trend (OLS + t-test), seasonality (autocorrelation), level shifts (Chow test), variance drift |
| Target | class imbalance, rare classes, leakage, predictive features, categorical group effects (η²) |
| Unsupervised | cluster tendency (silhouette sweep) |

### How it recommends

`ds/recommender.py` is a transparent rule engine — every recommendation carries the
evidence that produced it, the preprocessing it needs, the metric to judge it by, and
its caveats. It is meant to be read and argued with, not trusted blindly.

Tasks covered: regression, binary and multiclass classification, time-series
forecasting, clustering, anomaly detection, dimensionality reduction, association
rules, and text classification. Two rules run through all of them:

* **The trivial baseline is always recommendation #1.** A model that cannot beat
  `DummyRegressor` or `predict the majority class` has not been shown to work, and
  `insight baseline` reports both numbers side by side for exactly that reason.
* **Nothing is recommended without a measured reason.** If a rule cannot point at a
  pattern, it does not fire.

Three worked examples of the coupling between the two layers:

| Measured | Recommendation changes to |
|---|---|
| outliers in ≥1 column | Huber / quantile regression, `RobustScaler`, and a warning that if the outliers *are* the phenomenon this should be anomaly detection instead |
| class imbalance beyond ~20:1 | PR-AUC over accuracy, `class_weight='balanced'`, threshold tuning — plus reframing as anomaly detection |
| seasonality + >2 years of history | SARIMA at the detected period, ETS, and Prophet only if a changepoint was also found |

---

## Layout

| Path | What it is |
|---|---|
| `dataplatform/connectors/` | SQL (SQLAlchemy), Excel/CSV/Parquet readers |
| `dataplatform/warehouse/` | DuckDB / Postgres landing zone |
| `dataplatform/catalog/` | source registry, column profiler, semantic model |
| `dataplatform/nlp/` | QuerySpec, SQL compiler, Claude client, agent, guard |
| `dataplatform/superset/` | REST client + dataset/chart/dashboard publisher |
| `dataplatform/ds/` | profiling, pattern detection, algorithm recommender, baselines |
| `dataplatform/api/` | FastAPI service + the single-page UI in `api/static/` |
| `dataplatform/cli.py` | `insight` command |
| `tests/` | 79 tests: compiler + guard, detectors against planted ground truth, API payloads, end-to-end |

Run them with `pytest tests -q`. The detector tests are the interesting ones — each
plants exactly one property in synthetic data and asserts both that the right
detector fires *and* that the others stay quiet, which is what stops a
changepoint detector from quietly reporting every trend.

## Known limits

* The keyword fallback parser is a fallback. It handles the common analytical
  shapes and gives up honestly rather than inventing a query; comparisons across
  periods, ratios of metrics, and joins need the model path (or a hand-written
  `QuerySpec`).
* There are no joins. Each question resolves against one ingested table — model
  them as views in the source, or ingest a pre-joined query.
* Changepoint detection finds one step per series. On a heavy-tailed series the
  magnitude is reliable but the located date can drift by weeks.
* Superset publishing creates a virtual dataset per chart. That keeps the answer
  and the dashboard consistent, at the cost of dataset sprawl if you publish
  every ad-hoc question.
* `AnalystAgent` and the LLM narrator require credentials and are exercised
  manually, not by the test suite.
* The UI has no auth and binds to localhost by default. It is a single-user
  analyst tool; putting it on a shared host means putting a reverse proxy with
  authentication in front of it first.
* The UI is verified by driving the live page (assets served, queries run, charts
  built, guard messages surfaced, mobile layout clean) — but there are no
  pixel-level snapshot tests, so visual regressions would not be caught
  automatically.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `DP_HOME` | `~/.insight-platform` | state directory (catalog + warehouse) |
| `DP_WAREHOUSE_URI` | `duckdb:///<DP_HOME>/warehouse.duckdb` | warehouse target |
| `ANTHROPIC_API_KEY` | — | enables the LLM path |
| `DP_MODEL` | `claude-opus-5` | model id |
| `DP_EFFORT` | `medium` | `low`/`medium`/`high`/`xhigh`/`max` |
| `DP_MAX_ROWS` | `50000` | hard row cap on every query |
| `SUPERSET_URL` | `http://localhost:8088` | Superset base URL |
| `SUPERSET_USERNAME` / `SUPERSET_PASSWORD` | `admin`/`admin` | Superset creds |
| `DP_SUPERSET_WAREHOUSE_URI` | *(falls back to `DP_WAREHOUSE_URI`)* | how **Superset** reaches the warehouse, when that differs from how we do |
