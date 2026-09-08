"""FastAPI service.

    uvicorn dataplatform.api.main:app --reload

Every route below the auth gate runs against the signed-in user's own
workspace: their own catalog file and their own warehouse storage. There is no
shared Platform - `platform_for` resolves one per user and caches it, so two
users never see each other's sources, datasets or tables.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from .. import auth, workspace
from ..config import settings
from ..errors import (
    CatalogError,
    DashboardExists,
    ConnectorError,
    LLMUnavailable,
    PlatformError,
    QueryValidationError,
    SupersetError,
)
from ..nlp import QuerySpec
from ..platform import Platform
from ..store import activity, db

@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init()
    # Capture the pre-auth activity lists before anything can rewrite
    # catalog.json without them. Rows land unowned; the first signup adopts.
    activity.import_pre_auth(settings.catalog_path)
    yield


app = FastAPI(
    title="Insight Platform",
    version="0.1.0",
    description="Connectors -> NL query layer -> Superset -> data-science advisory layer",
    lifespan=lifespan,
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

# Test seam only: when set, every user resolves to this Platform instead of
# their own workspace. It bypasses workspace *selection*, never the auth gate.
_platform: Platform | None = None


def current_user(http: Request) -> auth.User:
    """The signed-in user, as established by the gate below."""
    user = getattr(http.state, "user", None)
    if user is None:  # pragma: no cover - the middleware precedes every route
        raise HTTPException(status_code=401, detail="not signed in")
    return user


def platform_for(user: auth.User) -> Platform:
    if _platform is not None:
        return _platform
    return workspace.platform_for(user)


# Reachable without signing in. `/` and `/health` are matched exactly - a
# `startswith("/")` here would quietly expose the entire app.
_PUBLIC_EXACT = {"/", "/health", "/favicon.ico"}
_PUBLIC_PREFIXES = ("/static/", "/auth/")


class AuthGate(BaseHTTPMiddleware):
    """Require a session for everything except the login page and its assets."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _PUBLIC_EXACT or path.startswith(_PUBLIC_PREFIXES):
            return await call_next(request)

        user = auth.resolve_session(request.cookies.get(auth.SESSION_COOKIE))
        if user is None:
            return JSONResponse({"detail": "not signed in"}, status_code=401)
        request.state.user = user
        return await call_next(request)


app.add_middleware(AuthGate)


# --------------------------------------------------------------------- auth
class CredentialsRequest(BaseModel):
    username: str
    password: str


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        auth.SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        # False by default: the app is normally served over plain http on
        # localhost, where a Secure cookie is silently discarded and login
        # would fail with nothing in the logs. DP_COOKIE_SECURE=1 under TLS.
        secure=settings.cookie_secure,
        max_age=settings.session_ttl_days * 24 * 3600,
    )


def _auth_failed(exc: auth.AuthError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=str(exc))


@app.post("/auth/signup")
def signup(body: CredentialsRequest, http: Request, response: Response) -> dict:
    client = http.client.host if http.client else "?"
    try:
        auth.throttle.check(f"signup:{client}")
        user = auth.create_user(body.username, body.password)
    except auth.AuthError as exc:
        auth.throttle.record_failure(f"signup:{client}")
        raise _auth_failed(exc) from exc
    _set_session_cookie(response, auth.create_session(user.id))
    return {"username": user.username}


@app.post("/auth/login")
def login(body: CredentialsRequest, http: Request, response: Response) -> dict:
    client = http.client.host if http.client else "?"
    keys = (f"login:{client}", f"user:{auth.normalize_username(body.username)}")
    try:
        for key in keys:
            auth.throttle.check(key)
        user = auth.verify_credentials(body.username, body.password)
    except auth.AuthError as exc:
        for key in keys:
            auth.throttle.record_failure(key)
        raise _auth_failed(exc) from exc
    for key in keys:
        auth.throttle.clear(key)
    _set_session_cookie(response, auth.create_session(user.id))
    return {"username": user.username}


@app.post("/auth/logout")
def logout(http: Request, response: Response) -> dict:
    auth.delete_session(http.cookies.get(auth.SESSION_COOKIE))
    response.delete_cookie(auth.SESSION_COOKIE)
    return {"signed_out": True}


@app.get("/auth/me")
def me(http: Request) -> dict:
    user = auth.resolve_session(http.cookies.get(auth.SESSION_COOKIE))
    if user is None:
        raise HTTPException(status_code=401, detail="not signed in")
    return {"username": user.username}


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
    # Overwrite a Superset dashboard of the same name instead of publishing
    # beside it as "... (2)". Off by default: taking over a dashboard someone
    # laid out by hand orphans every chart on it.
    replace: bool = False


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
def list_sources(http: Request) -> list[dict]:
    return [meta.model_dump() for meta in platform_for(current_user(http)).list_sources()]


@app.post("/sources")
def add_source(request: SourceRequest, http: Request) -> dict:
    try:
        meta = platform_for(current_user(http)).add_source(
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
def list_objects(name: str, http: Request) -> list[str]:
    try:
        return platform_for(current_user(http)).list_objects(name)
    except PlatformError as exc:
        raise _fail(exc) from exc


@app.post("/ingest")
def ingest(request: IngestRequest, http: Request) -> list[dict]:
    try:
        results = platform_for(current_user(http)).ingest(
            request.source, obj=request.object, limit=request.limit
        )
    except PlatformError as exc:
        raise _fail(exc) from exc
    return [result.__dict__ for result in results]


@app.get("/datasets")
def list_datasets(http: Request) -> list[dict]:
    return [meta.model_dump() for meta in platform_for(current_user(http)).list_datasets()]


@app.get("/datasets/{name}")
def get_dataset(name: str, http: Request) -> dict:
    try:
        return platform_for(current_user(http)).dataset(name).model_dump()
    except PlatformError as exc:
        raise _fail(exc) from exc


@app.get("/datasets/{name}/preview")
def preview_dataset(name: str, http: Request, limit: int = 50) -> dict:
    try:
        frame = platform_for(current_user(http)).frame(name, limit=min(limit, 500))
    except PlatformError as exc:
        raise _fail(exc) from exc
    return {"columns": list(frame.columns), "rows": _records(frame, limit)}


@app.delete("/sources/{name}")
def remove_source(name: str, http: Request) -> dict:
    platform_for(current_user(http)).remove_source(name)
    return {"removed": name}


@app.get("/explore-activity")
def get_explore_activity(http: Request) -> list[dict]:
    return [e.model_dump() for e in activity.list_explore(current_user(http).id)]


@app.delete("/explore-activity")
def clear_explore_activity(http: Request) -> dict:
    activity.clear(current_user(http).id, "explore_activity")
    return {"cleared": True}


@app.post("/ask")
def ask(request: AskRequest, http: Request) -> dict:
    from ..catalog.models import ExploreActivityEntry

    user = current_user(http)
    engine = platform_for(user)
    try:
        if request.agent:
            agent_result = engine.agent_ask(request.question)
            last = agent_result.last_query
            activity.add_explore(
                user.id,
                ExploreActivityEntry(
                    question=request.question,
                    mode="agent",
                    sql=agent_result.queries[-1].sql if agent_result.queries else None,
                ),
            )
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

    activity.add_explore(
        user.id,
        ExploreActivityEntry(
            question=request.question,
            mode="spec",
            sql=result.sql,
            row_count=len(result.data),
            published=bool(result.published),
            dashboard_url=result.published.dashboard_url if result.published else None,
            chart_url=result.published.chart_url if result.published else None,
        ),
    )

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


@app.get("/ask-activity")
def get_ask_activity(http: Request) -> list[dict]:
    return [e.model_dump() for e in activity.list_ask(current_user(http).id)]


@app.delete("/ask-activity")
def clear_ask_activity(http: Request) -> dict:
    activity.clear(current_user(http).id, "ask_activity")
    return {"cleared": True}


@app.post("/ask-nlp")
def ask_nlp(request: AskNlpRequest, http: Request) -> dict:
    """Answer a natural language question about data with a descriptive response.

    Uses the LLM to generate a natural language answer based on the specified datasets.
    """
    from ..catalog.models import AskActivityEntry

    user = current_user(http)
    engine = platform_for(user)
    try:
        answer = engine.ask_nlp(
            request.question,
            datasets=request.datasets,
        )
    except PlatformError as exc:
        raise _fail(exc) from exc

    activity.add_ask(
        user.id,
        AskActivityEntry(
            question=request.question,
            dataset=request.datasets[0] if request.datasets else None,
        ),
    )

    return {
        "answer": answer.get("answer", ""),
        "insights": answer.get("insights", []),
    }


@app.post("/spec")
def run_spec(spec: QuerySpec, http: Request) -> dict:
    try:
        result = platform_for(current_user(http)).run_spec(spec)
    except PlatformError as exc:
        raise _fail(exc) from exc
    return {"sql": result.sql, "columns": list(result.data.columns), "rows": _records(result.data)}


@app.post("/sql")
def run_sql(request: SQLRequest, http: Request) -> dict:
    try:
        frame = platform_for(current_user(http)).run_sql(request.sql)
    except PlatformError as exc:
        raise _fail(exc) from exc
    return {"row_count": len(frame), "columns": list(frame.columns), "rows": _records(frame)}


@app.get("/dashboard-history")
def get_dashboard_history(http: Request) -> list[dict]:
    return [e.model_dump() for e in activity.list_history(current_user(http).id)]


@app.get("/dashboard-activity")
def get_dashboard_activity(http: Request) -> list[dict]:
    return [e.model_dump() for e in activity.list_dashboard(current_user(http).id)]


@app.delete("/dashboard-activity")
def clear_dashboard_activity(http: Request) -> dict:
    activity.clear(current_user(http).id, "dashboard_activity")
    return {"cleared": True}


@app.post("/dashboard")
def dashboard(request: DashboardRequest, http: Request) -> dict:
    """Compose several tiles into one Superset dashboard.

    Distinct from /ask, which is one question -> one chart. Set publish=false to
    preview the plan and each tile's SQL without touching Superset.
    """
    from ..catalog.models import DashboardActivityEntry, DashboardHistoryEntry
    from ..superset.layout import pack_rows

    user = current_user(http)

    if not request.datasets:
        raise _fail(PlatformError("select at least one dataset"))

    try:
        spec, compiled, published, problems = platform_for(user).build_dashboard(
            request.datasets,
            request=request.request,
            title=request.title,
            publish=request.publish,
            replace=request.replace,
            # The browser can ask the person what to do; don't leave a numbered
            # copy behind on their behalf.
            on_conflict="ask",
        )
    except DashboardExists as exc:
        # A structured 409 rather than a message: the UI needs the existing
        # dashboard's link and a free title to offer real choices.
        raise HTTPException(
            status_code=409,
            detail={
                "error": "dashboard_exists",
                "message": (
                    f"A dashboard named {exc.title!r} already exists and has its own "
                    "layout. Overwrite it, or publish under a different name."
                ),
                "title": exc.title,
                "existing_url": exc.existing_url,
                "suggested_title": exc.suggested_title,
            },
        ) from exc
    except PlatformError as exc:
        raise _fail(exc) from exc

    if published:
        activity.add_history(
            user.id,
            DashboardHistoryEntry(
                title=spec.title,
                dashboard_url=published.dashboard_url,
                chart_url=None,
                datasets=request.datasets,
                n_charts=len(published.chart_ids),
            ),
        )

    if not request.live:
        activity.add_dashboard(
            user.id,
            DashboardActivityEntry(
                action="publish" if published else "preview",
                title=spec.title,
                request=request.request,
                datasets=request.datasets,
                dashboard_url=published.dashboard_url if published else None,
                n_charts=len(published.chart_ids) if published else 0,
            ),
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
def analyze(request: AnalyzeRequest, http: Request) -> dict:
    try:
        report = platform_for(current_user(http)).analyze(
            request.dataset,
            target=request.target,
            sample=request.sample,
            narrate=request.narrate,
        )
    except PlatformError as exc:
        raise _fail(exc) from exc
    return report.model_dump()


@app.post("/baseline")
def baseline(request: BaselineRequest, http: Request) -> dict:
    try:
        result = platform_for(current_user(http)).baseline(
            request.dataset, request.target, sample=request.sample
        )
    except PlatformError as exc:
        raise _fail(exc) from exc
    return result.model_dump()
