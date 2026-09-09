# Insight Platform

Connect your database, ask questions in plain English, build dashboards, and run ML analysis — all from a single browser interface.

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux

# 2. Install (choose the extras for your database)
pip install -e ".[api]"          # DuckDB / SQLite
pip install -e ".[api,postgres]" # PostgreSQL
pip install -e ".[api,mssql]"    # SQL Server

# 3. Configure (optional — both keys can be set in the app instead)
cp .env.example .env
# Edit .env — set DP_WAREHOUSE_URI if you are not using the bundled DuckDB

# 4. Run
uvicorn dataplatform.api.main:app --reload
# Open http://localhost:8000 and create your account
```

The first screen asks you to sign in. Create an account — the **first** one
created takes ownership of any data already ingested (see
[Accounts and workspaces](#accounts-and-workspaces)).

Then open **Configure**, which has two things to set up:

* **Register a data source** — a database connection string, or a path to an
  Excel / CSV / Parquet file.
* **AI Key** — your own Anthropic key, so the plain-English tabs use the model
  rather than the built-in parser. See
  [AI key](#ai-key-bring-your-own).

## What's inside

| Tab | What it does |
|---|---|
| **Configure** | Register data sources, and add your own AI key |
| **Data Quality** | Per-column completeness, distinct counts and value ranges, plus a live sample |
| **Ask** | Plain-English question → descriptive NLP answer |
| **Quick Insight** | One plain-English question → a chart and its rows, with the compiled SQL |
| **Create Dashboard** | Describe a report → preview the tiles, then publish to Superset |
| **ML Readiness** | Can this table support a prediction? Ranked algorithms with reasoning |
| **SQL** | Run read-only SQL directly |

### Working history

Ask, Quick Insight and Create Dashboard each keep an **Activity** panel on the right,
recording everything you run there. It survives restarts and sign-outs, and is
private to your account.

* **Click any entry to reopen it.** Quick Insight re-runs the question, bringing
  back agent mode and the *Publish to Superset* setting exactly as you had them.
  Create Dashboard restores the tables, title and request and previews them — it
  never republishes, so reopening history cannot write to Superset. A published
  entry keeps an **Open dashboard** button for that.
* **Repeats update in place.** Asking the same thing again refreshes the
  existing entry and moves it to the top rather than filling the list with
  duplicates.
* **Clear** empties the panel for that tab.

On the Create Dashboard tab the plan and its SQL are one view: click a tile in the
layout to see the query behind it, editable in place. After publishing, the link
to the dashboard appears next to the button.

If a dashboard of that name already exists with its own layout, publishing stops
and asks rather than quietly leaving a numbered copy behind: open the existing
one, overwrite it (tick **Replace existing**), or publish under the suggested
name.

## Configuration

Copy `.env.example` to `.env` and fill in your values:

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | No* | — | Fallback key, used by any account without its own |
| `DP_WAREHOUSE_URI` | No* | DuckDB at `~/.insight-platform/` | SQLAlchemy connection string |
| `DP_SUPERSET_WAREHOUSE_URI` | Dashboard only | — | How Superset (Docker) reaches your DB |
| `SUPERSET_URL` | No | `http://localhost:8088` | Superset base URL |
| `DP_MODEL` | No | `claude-opus-5` | Claude model |
| `DP_MAX_ROWS` | No | `50000` | Max rows per query |
| `DP_COOKIE_SECURE` | No | `0` | Set to `1` when serving over HTTPS |
| `DP_SESSION_TTL_DAYS` | No | `30` | How long a sign-in lasts |
| `DP_MIN_PASSWORD_LENGTH` | No | `8` | Minimum password length |

\*Neither key variable is required. Leave `DP_WAREHOUSE_URI` blank and a local
DuckDB file is created automatically. Leave `ANTHROPIC_API_KEY` unset and each
user supplies their own key in **Configure → AI Key**; with neither, the
plain-English tabs fall back to the built-in parser.

## Accounts and workspaces

The first time you open the app it asks you to create an account. Signup is
open — anyone who can reach the port can register — so do not expose it to an
untrusted network without putting something in front of it.

Each user gets a **private workspace**: their own sources, datasets, warehouse
tables and activity history. Nobody can see or query anyone else's data.

* The **first** account adopts whatever was already there, so upgrading an
  existing install does not appear to lose your ingested data or history.
* Later accounts start empty. On DuckDB each gets its own warehouse file under
  `~/.insight-platform/users/<id>/`; on Postgres/SQL Server each gets its own
  schema (`u<id>`), so table names stay unqualified.
* Users, sessions and activity live in `~/.insight-platform/app.db` (SQLite in
  WAL mode). Sources and datasets stay in a `catalog.json` per workspace.

Set `DP_COOKIE_SECURE=1` if you serve the app over HTTPS. Leave it off for
plain `http://localhost`, or the session cookie is dropped and sign-in fails.

### AI key (bring your own)

The plain-English tabs — Ask, Quick Insight, and the composer behind Create
Dashboard — call Anthropic's API. Each account can hold **its own key**, set in
**Configure → AI Key**, and it is used only for that account's workspace, so
questions are billed to whoever asked them.

Resolution order is: your own key, then the server's `ANTHROPIC_API_KEY`, then
the built-in deterministic parser. The parser handles straightforward questions
("revenue by region") without any key at all; it is the nuanced ones that need
the model. Agent mode always needs a key and is disabled without one.

The key is stored in `app.db` **in plain text**, is never sent back to the
browser (only a `...abcd` stub is shown), and is cleared from the form once
saved. Anyone who can read that file can read the key, exactly as with `.env` —
so treat `DP_HOME` as sensitive.

### Known limitations

* **Run a single server process.** The catalog is read once at startup and
  rewritten whole, so `uvicorn --workers 2` (or two servers on one
  `DP_HOME`) will have them overwrite each other's catalog changes. Activity
  and accounts are in SQLite and safe either way; the catalog is not.
* **The CLI is not user-aware.** `insight ingest ...` operates on the default
  workspace at `DP_HOME`, not on any web user's. Running CLI writes against a
  live server can clobber that server's in-memory catalog.
* **Keep `DP_HOME` on a local disk.** SQLite's WAL mode does not work over
  network shares (SMB/NFS).

## Connection string examples

```
# PostgreSQL
DP_WAREHOUSE_URI=postgresql+psycopg://user:password@host:5432/dbname

# SQL Server
DP_WAREHOUSE_URI=mssql+pyodbc://user:password@host/dbname?driver=ODBC+Driver+18+for+SQL+Server

# DuckDB file
DP_WAREHOUSE_URI=duckdb:///C:/path/to/warehouse.duckdb

# SQLite
DP_WAREHOUSE_URI=sqlite:///C:/path/to/database.db
```

## SQL Server — extra step

SQL Server requires the Microsoft ODBC Driver installed on your OS:

- **Windows**: [Download from Microsoft](https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server)
- **Mac**: `brew install msodbcsql18`
- **Linux**: see Microsoft docs above

## Superset dashboards (optional)

Superset is only needed for the **Create Dashboard** tab. Skip this if you don't need it.

```bash
# From the superset/ folder
docker compose up -d
# Open http://localhost:8088  (admin / admin)
```

Set `DP_SUPERSET_WAREHOUSE_URI` in `.env` to the address Superset can reach from inside Docker.  
On Windows/Mac use `host.docker.internal` instead of `localhost`:

```
DP_SUPERSET_WAREHOUSE_URI=postgresql+psycopg2://user:password@host.docker.internal:5432/dbname
```

## Requirements

- Python 3.10+
- Docker Desktop *(Superset only)*

See `Insight_Platform_Setup_Guide.docx` for the full illustrated setup guide.
