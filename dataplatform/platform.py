"""The façade every entry point uses: CLI, API, notebooks.

Holds one catalog, one warehouse, and the layers stacked on top of them, so
`Platform().ask(...)` works without wiring anything up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .catalog import Catalog, DatasetMeta, SourceMeta
from .config import settings
from .connectors import infer_source_type, open_connector
from .ds import (
    AnalysisReport,
    BaselineResult,
    detect_all,
    infer_task,
    profile,
    recommend,
    train_baseline,
    try_narrate,
)
from .errors import CatalogError, PlatformError
from .ingest import Ingestor, IngestResult
from .nlp import CompiledQuery, NL2SQL, QuerySpec, reconcile, validate_sql
from .reports import DashboardSpec, compose, validate
from .superset import DashboardResult, PublishResult, SupersetPublisher, publish_dashboard
from .warehouse import open_warehouse


_DASHBOARD_WORDS = re.compile(
    r"\b(dashboard|kpi|kpis|scorecard|report|cards?|tiles?|overview page)\b", re.I
)
_MULTI_VIZ = re.compile(r"\b(charts?|graphs?|plots?|tables?|visuali[sz]ations?)\b", re.I)


def wants_dashboard(question: str) -> bool:
    """Does this question describe a report rather than a single number?

    `ask` answers one question with one chart. Without this check the platform
    silently returns a single tile for "kpi cards, graphs and a detail table" — an
    answer to a question that was not asked.
    """
    named = bool(_DASHBOARD_WORDS.search(question))
    several = len(set(m.group(0).lower() for m in _MULTI_VIZ.finditer(question))) >= 2
    return named or several


def _multi_chart_hint(question: str) -> list[str]:
    if not wants_dashboard(question):
        return []
    return [
        "This looks like a request for several charts. `ask` returns one chart "
        "per question — use the Create Dashboard view (or `insight dashboard`) "
        "to compose KPI cards, graphs and a detail table into one Superset dashboard."
    ]


@dataclass
class AskResult:
    question: str
    compiled: CompiledQuery
    data: pd.DataFrame
    published: PublishResult | None = None
    answer: str = ""
    warnings: list[str] = field(default_factory=list)
    # A structured next step the caller can offer as an action. Kept structured
    # rather than embedded in `warnings` so the UI can render a button instead of
    # asking the user to go and do it themselves.
    suggestion: dict | None = None

    @property
    def sql(self) -> str:
        return self.compiled.sql

    @property
    def spec(self) -> QuerySpec:
        return self.compiled.spec


class Platform:
    def __init__(
        self,
        warehouse_uri: str | None = None,
        catalog_path: Path | None = None,
        use_llm: bool | None = None,
        warehouse_schema: str | None = None,
    ) -> None:
        self.catalog = Catalog(catalog_path)
        self.warehouse = open_warehouse(warehouse_uri, schema=warehouse_schema)
        self.ingestor = Ingestor(self.catalog, self.warehouse)
        self.nl2sql = NL2SQL(self.catalog, dialect=self.warehouse.dialect, use_llm=use_llm)

    # ---------------------------------------------------------------- sources
    def add_source(
        self,
        name: str,
        uri: str,
        type: str | None = None,
        description: str = "",
        **options,
    ) -> SourceMeta:
        meta = SourceMeta(
            name=name,
            type=type or infer_source_type(uri),  # type: ignore[arg-type]
            uri=uri,
            options=options,
            description=description,
        )
        with open_connector(meta) as connector:
            connector.test()
        self.catalog.add_source(meta)
        return meta

    def list_sources(self) -> list[SourceMeta]:
        return self.catalog.list_sources()

    def list_objects(self, source: str) -> list[str]:
        meta = self.catalog.get_source(source)
        with open_connector(meta) as connector:
            return [obj.qualified for obj in connector.list_objects()]

    def remove_source(self, name: str) -> None:
        self.catalog.remove_source(name)

    # -------------------------------------------------------------- ingestion
    def ingest(
        self, source: str, obj: str | None = None, limit: int | None = None
    ) -> list[IngestResult]:
        if obj:
            return [self.ingestor.ingest_object(source, obj, limit=limit)]
        return self.ingestor.ingest_source(source, limit=limit)

    def list_datasets(self) -> list[DatasetMeta]:
        return self.catalog.list_datasets()

    def dataset(self, name: str) -> DatasetMeta:
        return self.catalog.get_dataset(name)

    def frame(self, dataset: str, limit: int | None = None) -> pd.DataFrame:
        meta = self.catalog.get_dataset(dataset)
        cap = limit or settings.max_rows
        return self.warehouse.query(f'SELECT * FROM "{meta.name}" LIMIT {int(cap)}')

    # ------------------------------------------------------------- NL queries
    def ask(
        self,
        question: str,
        datasets: list[str] | None = None,
        publish: bool = False,
        dashboard: str | None = None,
        chart_name: str | None = None,
    ) -> AskResult:
        compiled = self.nl2sql.translate(question, datasets=datasets)
        data = self.warehouse.query(compiled.sql)

        # Now that the row count is known, confirm the chart still makes sense.
        meta = self.catalog.get_dataset(compiled.spec.dataset)
        compiled.spec.chart = reconcile(compiled, meta, n_rows=len(data))

        result = AskResult(
            question=question,
            compiled=compiled,
            data=data,
            warnings=list(compiled.warnings),
        )
        if wants_dashboard(question):
            result.suggestion = {
                "type": "dashboard",
                "dataset": compiled.spec.dataset,
                "request": question,
                "message": (
                    "That reads like a request for a whole report, not a single "
                    "chart. Below is the one chart this question resolved to — or "
                    "build the full dashboard from the same question."
                ),
                "action": "Create this as a dashboard",
            }
        if publish:
            result.published = self.publish(compiled, dashboard=dashboard, chart_name=chart_name)
        return result

    def run_spec(self, spec: QuerySpec) -> AskResult:
        compiled = self.nl2sql.compile_spec(spec)
        data = self.warehouse.query(compiled.sql)
        return AskResult(question="(spec)", compiled=compiled, data=data)

    def run_sql(self, sql: str) -> pd.DataFrame:
        """Raw SQL, through the guard. Read-only, catalog tables only."""
        return self.warehouse.query(validate_sql(sql, self.catalog))

    def agent_ask(self, question: str):
        from .nlp.agent import AnalystAgent

        return AnalystAgent(self.catalog, self.warehouse).ask(question)

    def ask_nlp(self, question: str, datasets: list[str] | None = None) -> dict:
        """Answer a natural language question with an NLP-generated response.

        Uses the LLM to generate a descriptive answer based on data summaries and context.
        """
        from .nlp.llm import LLMClient

        # Get dataset metadata
        all_datasets = self.catalog.list_datasets()
        if datasets:
            selected = [d for d in all_datasets if d.name in datasets]
        else:
            selected = all_datasets

        if not selected:
            raise PlatformError("No datasets available to analyze")

        # Build data context
        context_lines = []
        for dataset in selected:
            context_lines.append(f"Dataset: {dataset.name}")
            context_lines.append(f"  Rows: {dataset.n_rows:,}")
            cols = ", ".join(c.name for c in dataset.columns[:10])
            context_lines.append(f"  Columns: {cols}")
            if len(dataset.columns) > 10:
                context_lines.append(f"  ... and {len(dataset.columns) - 10} more columns")

        data_context = "\n".join(context_lines)

        # Generate answer using LLM
        llm = LLMClient()
        instructions = "You are a helpful data analyst. Provide clear, insightful answers about data."
        prompt = f"""Based on the following data context, answer this question in 2-3 sentences:

Data Context:
{data_context}

Question: {question}"""

        response = llm.text(instructions=instructions, prompt=prompt, max_tokens=1024)

        # Split response into answer and insights
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        answer = paragraphs[0] if paragraphs else response
        insights = [p for p in paragraphs[1:] if len(p) < 200]

        return {
            "answer": answer,
            "insights": insights,
        }

    # -------------------------------------------------------------- dashboard
    def compose_dashboard(
        self,
        datasets: list[str],
        request: str = "",
        title: str = "",
        use_llm: bool | None = None,
    ) -> tuple[DashboardSpec, list, list[str]]:
        """Plan a multi-tile dashboard — from one table or several — and compile every tile.

        Returns the spec, the surviving (tile, compiled) pairs, and any problems —
        so callers can show a partial dashboard rather than failing whole.
        """
        spec = compose(self.catalog, datasets, request=request, title=title, use_llm=use_llm)
        kept, problems = validate(spec, self.nl2sql.compiler)
        spec.tiles = kept
        compiled = [(tile, self.nl2sql.compile_spec(tile.spec)) for tile in spec.ordered]
        return spec, compiled, problems

    def build_dashboard(
        self,
        datasets: list[str],
        request: str = "",
        title: str = "",
        publish: bool = True,
        use_llm: bool | None = None,
        replace: bool = False,
    ) -> tuple[DashboardSpec, list, DashboardResult | None, list[str]]:
        spec, compiled, problems = self.compose_dashboard(
            datasets, request=request, title=title, use_llm=use_llm
        )
        if not compiled:
            names = ", ".join(datasets)
            raise PlatformError(f"no usable tiles for {names}: {'; '.join(problems) or 'unknown'}")

        result = None
        if publish:
            spec.tiles = [tile for tile, _ in compiled]
            result = publish_dashboard(
                SupersetPublisher(), spec, compiled, self.warehouse, replace=replace
            )
        return spec, compiled, result, problems

    # ---------------------------------------------------------------- publish
    def publish(
        self,
        compiled: CompiledQuery,
        dashboard: str | None = None,
        chart_name: str | None = None,
    ) -> PublishResult:
        publisher = SupersetPublisher()
        return publisher.publish(
            compiled, self.warehouse, chart_name=chart_name, dashboard=dashboard
        )

    # -------------------------------------------------------- data science ---
    def analyze(
        self,
        dataset: str,
        target: str | None = None,
        sample: int | None = None,
        narrate: bool = False,
        include_clustering: bool = True,
    ) -> AnalysisReport:
        meta = self.catalog.get_dataset(dataset)
        df = self.frame(meta.name, limit=sample or settings.profile_sample)
        if df.empty:
            raise PlatformError(f"{meta.name} has no rows to analyse")

        if target:
            resolved = meta.resolve(target)
            if resolved is None:
                raise CatalogError(f"target {target!r} is not a column of {meta.name}")
            target = resolved

        stats = profile(df, name=meta.name)
        patterns = detect_all(df, stats, target=target, include_clustering=include_clustering)
        task = infer_task(df, stats, target)
        recommendations = recommend(df, stats, patterns, target=target, task=task)

        report = AnalysisReport(
            dataset=meta.name,
            target=target,
            task=task,
            profile=stats,
            patterns=patterns,
            recommendations=recommendations,
        )
        if narrate:
            report.narrative = try_narrate(report)
        return report

    def baseline(
        self, dataset: str, target: str, task: str | None = None, sample: int | None = None
    ) -> BaselineResult:
        meta = self.catalog.get_dataset(dataset)
        df = self.frame(meta.name, limit=sample or settings.profile_sample)
        resolved = meta.resolve(target)
        if resolved is None:
            raise CatalogError(f"target {target!r} is not a column of {meta.name}")

        stats = profile(df, name=meta.name)
        inferred = task or infer_task(df, stats, resolved)
        time_column = stats.temporal_columns[0] if stats.temporal_columns else None
        return train_baseline(df, stats, resolved, inferred, time_column=time_column)  # type: ignore[arg-type]
