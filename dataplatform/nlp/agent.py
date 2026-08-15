"""Agentic analyst.

`NL2SQL` is one-shot: question in, one query out. That is the right default — it is
cheap, auditable, and easy to cache. Some questions genuinely need exploration
("why did revenue drop in Q3?"): look at one cut, then decide the next.

This uses the SDK tool runner over four read-only tools. Every tool is a wrapper
around the same validated compiler path the one-shot route uses, so the agent has
no more authority than a single translation does — it just gets more turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..catalog import Catalog, describe_catalog, describe_dataset
from ..config import settings
from ..errors import LLMUnavailable, PlatformError, QueryValidationError
from ..warehouse import Warehouse
from .compiler import SQLCompiler
from .spec import CompiledQuery, Dimension, Metric, QuerySpec, Sort

MAX_TOOL_ROWS = 200

SYSTEM = """\
You are a data analyst with read-only access to a warehouse.

Work by running queries, not by guessing. Start from the catalog, check the values
in a column before filtering on it, and run a follow-up query when a result raises
an obvious question. Prefer two or three sharp queries to one sprawling one.

When you have the answer, state it in plain prose: the number or finding first,
then what you did to get it, then any caveat about the data that a reader would
need. Do not paste raw JSON back at the user.
"""


@dataclass
class AgentResult:
    answer: str
    queries: list[CompiledQuery] = field(default_factory=list)
    frames: list[pd.DataFrame] = field(default_factory=list)

    @property
    def last_query(self) -> CompiledQuery | None:
        return self.queries[-1] if self.queries else None


class AnalystAgent:
    def __init__(self, catalog: Catalog, warehouse: Warehouse, model: str | None = None) -> None:
        self.catalog = catalog
        self.warehouse = warehouse
        self.compiler = SQLCompiler(catalog, dialect=warehouse.dialect)
        self.model = model or settings.model

    # ------------------------------------------------------------- tool impls
    def _list_datasets(self) -> str:
        return describe_catalog(self.catalog)

    def _describe(self, dataset: str) -> str:
        return describe_dataset(self.catalog.get_dataset(dataset))

    def _distinct_values(self, dataset: str, column: str, limit: int = 50) -> str:
        meta = self.catalog.get_dataset(dataset)
        resolved = meta.resolve(column)
        if resolved is None:
            return f"error: {dataset} has no column {column!r}"
        spec = QuerySpec(
            dataset=meta.name,
            dimensions=[Dimension(column=resolved)],
            metrics=[Metric(column="*", func="count", alias="n")],
            sort=[Sort(field="n", descending=True)],
            limit=min(int(limit), 200),
        )
        compiled = self.compiler.compile(spec)
        return self.warehouse.query(compiled.sql).to_csv(index=False)

    def _run_query(self, spec_json: str) -> str:
        try:
            spec = QuerySpec.model_validate_json(spec_json)
        except Exception as exc:
            return f"error: that is not a valid QuerySpec: {exc}"
        try:
            compiled = self.compiler.compile(spec)
        except QueryValidationError as exc:
            return f"error: {exc}"
        frame = self.warehouse.query(compiled.sql)
        self._queries.append(compiled)
        self._frames.append(frame)
        preview = frame.head(MAX_TOOL_ROWS)
        note = "" if len(frame) <= MAX_TOOL_ROWS else f"\n({len(frame)} rows, showing {MAX_TOOL_ROWS})"
        return f"SQL:\n{compiled.sql}\n\nRESULT:\n{preview.to_csv(index=False)}{note}"

    # ------------------------------------------------------------------- run
    def ask(self, question: str, max_turns: int = 12) -> AgentResult:
        try:
            import anthropic
            from anthropic import beta_tool
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("the `anthropic` package is not installed") from exc
        if not settings.has_llm:
            raise LLMUnavailable("no ANTHROPIC_API_KEY in the environment")

        self._queries: list[CompiledQuery] = []
        self._frames: list[pd.DataFrame] = []

        @beta_tool
        def list_datasets() -> str:
            """List every table in the catalog with its columns, types and sample values."""
            return self._list_datasets()

        @beta_tool
        def describe_table(dataset: str) -> str:
            """Show the full column detail for one table.

            Args:
                dataset: Table name as it appears in the catalog.
            """
            return self._describe(dataset)

        @beta_tool
        def distinct_values(dataset: str, column: str, limit: int = 50) -> str:
            """List the distinct values of a column with row counts.

            Call this before filtering on a categorical column so the filter uses
            the spelling that is actually in the data.

            Args:
                dataset: Table name.
                column: Column to enumerate.
                limit: Maximum distinct values to return.
            """
            return self._distinct_values(dataset, column, limit)

        @beta_tool
        def run_query(spec_json: str) -> str:
            """Run a QuerySpec and return the generated SQL plus the result as CSV.

            Args:
                spec_json: A JSON QuerySpec object with keys: dataset, dimensions
                    (column, time_grain, alias), metrics (column, func, alias),
                    filters (column, op, values), sort (field, descending), limit,
                    chart, title, explanation.
            """
            return self._run_query(spec_json)

        client = anthropic.Anthropic()
        runner = client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=settings.max_tokens,
            system=[
                {"type": "text", "text": SYSTEM},
                {
                    "type": "text",
                    "text": "CATALOG\n=======\n" + describe_catalog(self.catalog),
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            thinking={"type": "adaptive"},
            output_config={"effort": settings.effort},
            tools=[list_datasets, describe_table, distinct_values, run_query],
            messages=[{"role": "user", "content": question}],
            max_iterations=max_turns,
        )

        final: Any = None
        for message in runner:
            final = message

        if final is None:
            raise PlatformError("the agent produced no response")
        if getattr(final, "stop_reason", None) == "refusal":
            raise PlatformError("the model declined this request")

        answer = "\n".join(b.text for b in final.content if b.type == "text").strip()
        return AgentResult(answer=answer, queries=self._queries, frames=self._frames)
