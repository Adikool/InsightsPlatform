"""Question -> QuerySpec -> validated SQL.

The compile step is also the verification step: if the model hallucinates a column,
the compiler raises with the near-miss suggestions, and we hand that error straight
back to the model as a repair turn. Two attempts, then we stop — a third rarely
converges and the failure is more useful than a wrong answer.
"""

from __future__ import annotations

from ..catalog import Catalog, describe_catalog
from ..config import settings
from ..errors import PlatformError, QueryValidationError
from . import heuristic
from .compiler import SQLCompiler
from . import llm as llm_module
from .llm import LLMClient, available
from .spec import CompiledQuery, QuerySpec

INSTRUCTIONS = """\
You translate business questions into a QuerySpec against a data catalog.

Rules:
- Use only table and column names that appear in the catalog below. Never invent one.
- Prefer the smallest spec that answers the question. Do not add dimensions,
  metrics, or filters the question did not ask for.
- Set time_grain only on temporal columns, and only when the question implies a
  time series ("monthly", "over time", "trend").
- Filter values go in `values` as strings; the compiler casts them by column type.
  Use the exact spelling shown in the catalog's sample values.
- For "top N" questions, set limit to N and sort by the metric descending.
- Choose `chart` from the shape of the result: time on an axis -> line; one
  category -> bar; no grouping -> big_number; two categories -> heatmap;
  otherwise table.
- `explanation` is one sentence describing how you read the question, including
  any assumption you had to make.
"""


class NL2SQL:
    def __init__(
        self,
        catalog: Catalog,
        dialect: str = "duckdb",
        llm: LLMClient | None = None,
        use_llm: bool | None = None,
        api_key: str | None = None,
    ) -> None:
        self.catalog = catalog
        self.compiler = SQLCompiler(catalog, dialect=dialect)
        self.api_key = api_key
        self.llm = llm or LLMClient(api_key=api_key)
        self.use_llm = available(api_key) if use_llm is None else use_llm

    # ------------------------------------------------------------------ main
    def translate(self, question: str, datasets: list[str] | None = None) -> CompiledQuery:
        # `disabled_reason` short-circuits before describe_catalog runs: with a
        # rejected key there is no point rendering the whole schema for a call
        # that will be refused.
        if not self.use_llm or llm_module.disabled_reason(self.api_key):
            return self._compile_or_raise(
                heuristic.parse(question, self.catalog, settings.default_limit)
            )
        try:
            return self._translate_with_llm(question, datasets)
        except PlatformError:
            # Same reasoning as reports/composer.py: a refusal, a rate limit, or
            # a dropped connection should degrade to the heuristic parser exactly
            # like no key at all, not surface as an unhandled exception.
            return self._compile_or_raise(
                heuristic.parse(question, self.catalog, settings.default_limit)
            )

    def _translate_with_llm(self, question: str, datasets: list[str] | None) -> CompiledQuery:
        context = "CATALOG\n=======\n" + describe_catalog(self.catalog, only=datasets)
        spec = self.llm.structured(
            instructions=INSTRUCTIONS,
            context=context,
            question=question,
            output_model=QuerySpec,
        )
        try:
            return self.compiler.compile(spec)
        except QueryValidationError as first_error:
            repair = (
                f"{question}\n\n"
                f"Your previous spec was rejected by the compiler:\n"
                f"  {first_error}\n\n"
                f"It was:\n{spec.model_dump_json(indent=2)}\n\n"
                f"Return a corrected spec that uses only catalog columns."
            )
            repaired = self.llm.structured(
                instructions=INSTRUCTIONS,
                context=context,
                question=repair,
                output_model=QuerySpec,
            )
            compiled = self.compiler.compile(repaired)
            compiled.warnings.append(f"repaired after: {first_error}")
            return compiled

    def _compile_or_raise(self, spec: QuerySpec) -> CompiledQuery:
        return self.compiler.compile(spec)

    # ------------------------------------------------------------- utilities
    def compile_spec(self, spec: QuerySpec) -> CompiledQuery:
        """Compile a spec supplied directly (API clients, saved queries)."""
        return self.compiler.compile(spec)
