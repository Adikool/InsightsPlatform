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

# 3. Configure
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY and DP_WAREHOUSE_URI

# 4. Run
uvicorn dataplatform.api.main:app --reload
# Open http://localhost:8000
```

## What's inside

| Tab | What it does |
|---|---|
| **Configure** | Register database connections |
| **Data** | Browse tables and column statistics |
| **Ask** | Plain-English question → descriptive NLP answer |
| **Explore** | Plain-English question → SQL + result table |
| **Dashboard** | Describe a report → auto-publish to Superset |
| **Analyze** | Profile data and get ML algorithm recommendations |
| **SQL** | Run read-only SQL directly |

## Configuration

Copy `.env.example` to `.env` and fill in your values:

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `DP_WAREHOUSE_URI` | No* | DuckDB at `~/.insight-platform/` | SQLAlchemy connection string |
| `DP_SUPERSET_WAREHOUSE_URI` | Dashboard only | — | How Superset (Docker) reaches your DB |
| `SUPERSET_URL` | No | `http://localhost:8088` | Superset base URL |
| `DP_MODEL` | No | `claude-opus-5` | Claude model |
| `DP_MAX_ROWS` | No | `50000` | Max rows per query |
| `DP_COOKIE_SECURE` | No | `0` | Set to `1` when serving over HTTPS |
| `DP_SESSION_TTL_DAYS` | No | `30` | How long a sign-in lasts |
| `DP_MIN_PASSWORD_LENGTH` | No | `8` | Minimum password length |

*Leave `DP_WAREHOUSE_URI` blank and a local DuckDB file is created automatically — no database setup needed.

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

Superset is only needed for the **Dashboard** tab. Skip this if you don't need it.

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
