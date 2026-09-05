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

*Leave `DP_WAREHOUSE_URI` blank and a local DuckDB file is created automatically — no database setup needed.

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
