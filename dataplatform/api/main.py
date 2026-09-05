"""FastAPI service.

    uvicorn dataplatform.api.main:app --reload

One Platform instance is shared across requests. The warehouse opens a connection
per operation, so that is safe; the catalog is read-mostly and rewritten
atomically on change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import settings
from ..errors import (
    CatalogError,
    ConnectorError,
    LLMUnavailable,
    PlatformError,
    QueryValidationError,
    SupersetError,
)
from ..nlp import QuerySpec
from ..platform import Platform

app = FastAPI(
    title="Insight Platform",
    version="0.1.0",
    description="Connectors -> NL query layer -> Superset -> data-science advisory layer",
)

STATIC = Path(__file__).parent / "static"


class RevalidatingStaticFiles(StaticFiles):
    """Serve assets with `Cache-Control: no-cache`.

    Without an explicit header browsers apply heuristic caching, so after the UI
    is updated you keep getting the previous HTML and JS until a hard refresh —
    a confusing failure, because the server is serving the new file correctly.
    `no-cache` still permits ETag revalidation, so repeat loads stay cheap 304s;
    it only forbids using a cached copy without asking.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


if STATIC.is_dir():
    app.mount("/static", RevalidatingStaticFiles(directory=STATIC), name="static")

_platform: Platform | None = None


def platform() -> Platform:
    global _platform
    if _platform is None:
        _platform = Platform()
    return _platform


_STATUS = {
    QueryValidationError: 400,
    CatalogError: 404,
    ConnectorError: 502,
    SupersetError: 502,
    LLMUnavailable: 503,
}


def _fail(exc: PlatformError) -> HTTPException:
    return HTTPException(status_code=_STATUS.get(type(exc), 500), detail=str(exc))


def _records(frame: pd.DataFrame, limit: int | None = None) -> list[dict]:
    """DataFrame -> JSON-safe records.

    Going through pandas' own serialiser rather than `to_dict` because a result set
    routinely contains Timestamps and NaN, and neither survives `json.dumps`.
    """
    subset = frame.head(limit) if limit is not None else frame
    return json.loads(subset.to_json(orient="records", date_format="iso"))


# ------------------------------------------------------------------ schemas
class SourceRequest(BaseModel):
    name: str
    uri: str
    type: str | None = None
    description: str = ""
    options: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    source: str
    object: str | None = None
    limit: int | None = None


class AskRequest(BaseModel):
    question: str
    datasets: list[str] | None = None
    publish: bool = False
    dashboard: str | None = None
    agent: bool = False
    max_rows: int = 500


class AskNlpRequest(BaseModel):
    question: str
    datasets: list[str] | None = None


class SQLRequest(BaseModel):
    sql: str


class DashboardRequest(BaseModel):
    datasets: list[str]
    request: str = ""
    title: str = ""
    publish: bool = True
    # True for the debounced recompose-as-you-type calls; those aren't logged
    # to the activity feed, only explicit "Preview plan" / "Publish" clicks are.
    live: bool = False


class AnalyzeRequest(BaseModel):
    dataset: str
    target: str | None = None
    sample: int | None = None
    narrate: bool = False


class BaselineRequest(BaseModel):
    dataset: str
    target: str
    sample: int | None = None


# ------------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def index():
    page = STATIC / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="UI assets are not installed")
    return FileResponse(page, headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "warehouse": settings.warehouse_uri,
        "llm": settings.has_llm,
        "datasets": len(platform().list_datasets()),
    }


def _ui_build() -> str:
    """Short fingerprint of the served UI assets.

    Shown in the sidebar so "am I looking at a stale page?" is answerable at a
    glance — browser caching has twice made a fixed UI look unfixed.
    """
    import hashlib

    if not STATIC.is_dir():
        return "none"
    digest = hashlib.sha256()
    for path in sorted(STATIC.glob("*")):
        digest.update(path.name.encode())
        digest.update(str(int(path.stat().st_mtime)).encode())
    return digest.hexdigest()[:8]


@app.get("/config")
def config() -> dict:
    """Everything the UI needs to describe its own environment to the user."""
    return {
        "ui_build": _ui_build(),
        "warehouse": settings.warehouse_uri,
        "catalog": str(settings.catalog_path),
        "llm_available": settings.has_llm,
        "model": settings.model,
        "effort": settings.effort,
        "superset_url": settings.superset_url,
        "max_rows": settings.max_rows,
    }


@app.get("/superset/check")
def superset_check() -> dict:
    from ..superset import check_connection

    try:
        return {"connected": True, "user": check_connection()}
    except PlatformError as exc:
        # Not an HTTP error: "Superset is not running" is a normal state for this
        # platform, and the UI shows it as a badge rather than a failure dialog.
        return {"connected": False, "error": str(exc)}


@app.get("/sources")
def list_sources() -> list[dict]:
    return [meta.model_dump() for meta in platform().list_sources()]


@app.post("/sources")
def add_source(request: SourceRequest) -> dict:
    try:
        meta = platform().add_source(
            request.name,
            request.uri,
            type=request.type,
            description=request.description,
            **request.options,
        )
    except PlatformError as exc:
        raise _fail(exc) from exc
    return meta.model_dump()


@app.get("/sources/{name}/objects")
def list_objects(name: str) -> list[str]:
    try:
        return platform().list_objects(name)
    except PlatformError as exc:
        raise _fail(exc) from exc


@app.post("/ingest")
def ingest(request: IngestRequest) -> list[dict]:
    try:
        results = platform().ingest(request.source, obj=request.object, limit=request.limit)
    except PlatformError as exc:
        raise _fail(exc) from exc
    return [result.__dict__ for result in results]


@app.get("/datasets")
def list_datasets() -> list[dict]:
    return [meta.model_dump() for meta in platform().list_datasets()]


@app.get("/datasets/{name}")
def get_dataset(name: str) -> dict:
    try:
        return platform().dataset(name).model_dump()
    except PlatformError as exc:
        raise _fail(exc) from exc


@app.get("/datasets/{name}/preview")
def preview_dataset(name: str, limit: int = 50) -> dict:
    try:
        frame = platform().frame(name, limit=min(limit, 500))
    except PlatformError as exc:
        raise _fail(exc) from exc
    return {"columns": list(frame.columns), "rows": _records(frame, limit)}


@app.delete("/sources/{name}")
def remove_source(name: str) -> dict:
    platform().remove_source(name)
    return {"removed": name}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    engine = platform()
    try:
        if request.agent:
            agent_result = engine.agent_ask(request.question)
            last = agent_result.last_query
            return {
                "mode": "agent",
                "answer": agent_result.answer,
                "queries": [q.sql for q in agent_result.queries],
                "spec": last.spec.model_dump() if last else None,
            }

        result = engine.ask(
            request.question,
            datasets=request.datasets,
            publish=request.publish,
            dashboard=request.dashboard,
        )
    except PlatformError as exc:
        raise _fail(exc) from exc

    return {
        "mode": "spec",
        "question": result.question,
        "spec": result.spec.model_dump(),
        "sql": result.sql,
        "columns": list(result.data.columns),
        "row_count": len(result.data),
        "rows": _records(result.data, request.max_rows),
        "warnings": result.warnings,
        "suggestion": result.suggestion,
        "superset": result.published.__dict__ if result.published else None,
    }


@app.post("/ask-nlp")
def ask_nlp(request: AskNlpRequest) -> dict:
    """Answer a natural language question about data with a descriptive response.

    Uses the LLM to generate a natural language answer based on the specified datasets.
    """
    engine = platform()
    try:
        answer = engine.ask_nlp(
            request.question,
            datasets=request.datasets,
        )
    except PlatformError as exc:
        raise _fail(exc) from exc

    return {
        "answer": answer.get("answer", ""),
        "insights": answer.get("insights", []),
    }


@app.post("/spec")
def run_spec(spec: QuerySpec) -> dict:
    try:
        result = platform().run_spec(spec)
    except PlatformError as exc:
        raise _fail(exc) from exc
    return {"sql": result.sql, "columns": list(result.data.columns), "rows": _records(result.data)}


@app.post("/sql")
def run_sql(request: SQLRequest) -> dict:
    try:
        frame = platform().run_sql(request.sql)
    except PlatformError as exc:
        raise _fail(exc) from exc
    return {"row_count": len(frame), "columns": list(frame.columns), "rows": _records(frame)}


@app.get("/dashboard-history")
def get_dashboard_history() -> list[dict]:
    return [e.model_dump() for e in platform().catalog.list_dashboard_history()]


@app.get("/dashboard-activity")
def get_dashboard_activity() -> list[dict]:
    return [e.model_dump() for e in platform().catalog.list_dashboard_activity()]


@app.post("/dashboard")
def dashboard(request: DashboardRequest) -> dict:
    """Compose several tiles into one Superset dashboard.

    Distinct from /ask, which is one question -> one chart. Set publish=false to
    preview the plan and each tile's SQL without touching Superset.
    """
    from datetime import datetime, timezone

    from ..catalog.models import DashboardActivityEntry, DashboardHistoryEntry
    from ..superset.layout import pack_rows

    if not request.datasets:
        raise _fail(PlatformError("select at least one dataset"))

    try:
        spec, compiled, published, problems = platform().build_dashboard(
            request.datasets,
            request=request.request,
            title=request.title,
            publish=request.publish,
        )
    except PlatformError as exc:
        raise _fail(exc) from exc

    if published:
        platform().catalog.add_dashboard_history(
            DashboardHistoryEntry(
                title=spec.title,
                dashboard_url=published.dashboard_url,
                chart_url=None,
                datasets=request.datasets,
                n_charts=len(published.chart_ids),
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )

    if not request.live:
        platform().catalog.add_dashboard_activity(
            DashboardActivityEntry(
                action="publish" if published else "preview",
                title=spec.title,
                request=request.request,
                datasets=request.datasets,
                dashboard_url=published.dashboard_url if published else None,
                n_charts=len(published.chart_ids) if published else 0,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )

    tiles = [tile for tile, _ in compiled]
    return {
        "title": spec.title,
        "description": spec.description,
        "interpretation": spec.interpretation,
        "datasets": request.datasets,
        "tiles": [
            {
                "title": tile.title,
                "role": tile.role,
                "width": tile.width,
                "chart": tile.spec.chart,
                "sql": query.sql,
                "explanation": tile.spec.explanation,
            }
            for tile, query in compiled
        ],
        "rows": [[tile.title for tile in row] for row in pack_rows(tiles)],
        "problems": problems,
        "superset": published.__dict__ if published else None,
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    try:
        report = platform().analyze(
            request.dataset,
            target=request.target,
            sample=request.sample,
            narrate=request.narrate,
        )
    except PlatformError as exc:
        raise _fail(exc) from exc
    return report.model_dump()


@app.post("/baseline")
def baseline(request: BaselineRequest) -> dict:
    try:
        result = platform().baseline(request.dataset, request.target, sample=request.sample)
    except PlatformError as exc:
        raise _fail(exc) from exc
    return result.model_dump()
