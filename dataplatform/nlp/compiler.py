"""QuerySpec -> SQL.

Security model, stated plainly because this is the trust boundary:

* **Identifiers never come from the model.** Every column/table name in the spec is
  resolved against the catalog first; the string that reaches the SQL text is the
  catalog's canonical name, quoted. An unresolvable name is an error, not a guess.
* **Literals are escaped by type.** Numeric and date operands must parse as such
  before they are inlined; string operands are single-quote escaped. Literals are
  inlined rather than bound because the same SQL text has to be handed to Superset
  as a virtual dataset, where there is nowhere to bind parameters.
* **The statement shape is fixed.** SELECT / FROM / WHERE / GROUP BY / HAVING /
  ORDER BY / LIMIT, assembled from validated parts. There is no code path here
  that can emit DML or DDL.
"""

from __future__ import annotations

import difflib
import re
from datetime import datetime

from ..catalog import Catalog, DatasetMeta
from ..config import settings
from ..errors import QueryValidationError
from .spec import CompiledQuery, Dimension, Filter, Metric, QuerySpec, Sort

_IDENT_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MULTI_VALUE_OPS = {"in", "not_in"}
_NO_VALUE_OPS = {"is_null", "is_not_null"}
_RELATIVE_OPS = {"last_n_days", "last_n_months"}


def _quote_ident(name: str) -> str:
    if not _IDENT_OK.match(name):
        # Catalog names are normalised at ingest, so this only fires if the
        # catalog itself was hand-edited. Refuse rather than escape-and-hope.
        raise QueryValidationError(f"refusing to emit non-identifier column name: {name!r}")
    return f'"{name}"'


def _quote_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _numeric_literal(value: str, column: str) -> str:
    try:
        return repr(float(value)) if "." in str(value) or "e" in str(value).lower() else str(int(value))
    except (TypeError, ValueError):
        raise QueryValidationError(
            f"filter on numeric column {column!r} needs a number, got {value!r}"
        ) from None


def _date_literal(value: str, column: str) -> str:
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%Y":
                return f"DATE '{parsed.year}-01-01'"
            if fmt == "%Y-%m":
                return f"DATE '{parsed.year}-{parsed.month:02d}-01'"
            return f"DATE '{parsed.date().isoformat()}'"
        except ValueError:
            continue
    try:
        return f"TIMESTAMP '{datetime.fromisoformat(text).isoformat(sep=' ')}'"
    except ValueError:
        raise QueryValidationError(
            f"filter on temporal column {column!r} needs a date, got {value!r}"
        ) from None


class SQLCompiler:
    def __init__(self, catalog: Catalog, dialect: str = "duckdb") -> None:
        self.catalog = catalog
        self.dialect = dialect

    # ------------------------------------------------------------ resolution
    def _dataset(self, name: str) -> DatasetMeta:
        try:
            return self.catalog.get_dataset(name)
        except Exception:
            known = [d.name for d in self.catalog.list_datasets()]
            close = difflib.get_close_matches(name, known, n=3, cutoff=0.4)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise QueryValidationError(f"unknown dataset {name!r}.{hint}") from None

    def _column(self, dataset: DatasetMeta, name: str) -> str:
        resolved = dataset.resolve(name)
        if resolved is None:
            close = difflib.get_close_matches(
                name, [c.name for c in dataset.columns], n=3, cutoff=0.4
            )
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise QueryValidationError(
                f"column {name!r} does not exist on {dataset.name}.{hint}"
            )
        return resolved

    @staticmethod
    def _safe_alias(alias: str) -> str:
        """Aliases are model-authored, so they are sanitised rather than trusted."""
        cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(alias)).strip("_").lower()
        if not cleaned:
            cleaned = "value"
        if cleaned[0].isdigit():
            cleaned = f"c_{cleaned}"
        return cleaned[:63]

    # ------------------------------------------------------------ expressions
    def _dimension_sql(self, dataset: DatasetMeta, dim: Dimension) -> tuple[str, str, bool]:
        column = self._column(dataset, dim.column)
        meta = dataset.column(column)
        quoted = _quote_ident(column)
        is_time = False

        if dim.time_grain != "none":
            if meta and meta.semantic_type != "temporal":
                raise QueryValidationError(
                    f"time_grain={dim.time_grain!r} requested on non-temporal column {column!r}"
                )
            expr = f"date_trunc('{dim.time_grain}', {quoted})"
            alias = dim.alias or f"{column}_{dim.time_grain}"
            is_time = True
        else:
            expr = quoted
            alias = dim.alias or column
            is_time = bool(meta and meta.semantic_type == "temporal")

        return expr, self._safe_alias(alias), is_time

    def _metric_sql(self, dataset: DatasetMeta, metric: Metric) -> tuple[str, str]:
        if metric.column.strip() == "*":
            if metric.func not in ("count",):
                raise QueryValidationError("'*' is only valid with func='count'")
            return "COUNT(*)", self._safe_alias(metric.alias or "row_count")

        column = self._column(dataset, metric.column)
        meta = dataset.column(column)
        quoted = _quote_ident(column)

        numeric_only = {"sum", "avg", "median", "stddev"}
        if metric.func in numeric_only and meta and not meta.is_numeric_dtype:
            raise QueryValidationError(
                f"cannot apply {metric.func}() to {meta.semantic_type} column "
                f"{column!r} of type {meta.dtype}"
            )
        if metric.func == "sum" and meta and meta.semantic_type == "identifier":
            raise QueryValidationError(f"summing the identifier column {column!r} is meaningless")

        if metric.func == "count_distinct":
            expr = f"COUNT(DISTINCT {quoted})"
        elif metric.func == "median":
            expr = (
                f"median({quoted})"
                if self.dialect == "duckdb"
                else f"percentile_cont(0.5) WITHIN GROUP (ORDER BY {quoted})"
            )
        elif metric.func == "stddev":
            expr = f"stddev_samp({quoted})"
        else:
            expr = f"{metric.func.upper()}({quoted})"

        alias = metric.alias or f"{metric.func}_{column}"
        return expr, self._safe_alias(alias)

    def _filter_sql(self, dataset: DatasetMeta, flt: Filter, expr_override: str = "") -> str:
        if expr_override:
            expr, semantic = expr_override, "numeric"
        else:
            column = self._column(dataset, flt.column)
            meta = dataset.column(column)
            expr = _quote_ident(column)
            semantic = meta.semantic_type if meta else "text"

        op = flt.op
        values = list(flt.values)

        if op in _NO_VALUE_OPS:
            return f"{expr} IS NULL" if op == "is_null" else f"{expr} IS NOT NULL"

        if op in _RELATIVE_OPS:
            if len(values) != 1:
                raise QueryValidationError(f"{op} needs exactly one number, got {values}")
            n = _numeric_literal(values[0], flt.column)
            unit = "days" if op == "last_n_days" else "months"
            return f"{expr} >= CURRENT_DATE - INTERVAL '{n} {unit}'"

        if not values:
            raise QueryValidationError(f"filter {flt.column} {op} is missing operands")

        def lit(v: str) -> str:
            if semantic in ("numeric", "currency"):
                return _numeric_literal(v, flt.column)
            if semantic == "temporal":
                return _date_literal(v, flt.column)
            if semantic == "boolean":
                return "TRUE" if str(v).strip().lower() in ("true", "1", "yes", "y") else "FALSE"
            return _quote_string(v)

        if op in _MULTI_VALUE_OPS:
            rendered = ", ".join(lit(v) for v in values)
            keyword = "IN" if op == "in" else "NOT IN"
            return f"{expr} {keyword} ({rendered})"

        if op == "between":
            if len(values) != 2:
                raise QueryValidationError(f"'between' needs two operands, got {values}")
            return f"{expr} BETWEEN {lit(values[0])} AND {lit(values[1])}"

        if op in ("contains", "starts_with"):
            pattern = str(values[0]).replace("%", r"\%").replace("_", r"\_")
            pattern = f"%{pattern}%" if op == "contains" else f"{pattern}%"
            return f"LOWER(CAST({expr} AS VARCHAR)) LIKE LOWER({_quote_string(pattern)})"

        return f"{expr} {op} {lit(values[0])}"

    # ----------------------------------------------------------------- public
    def compile(self, spec: QuerySpec) -> CompiledQuery:
        dataset = self._dataset(spec.dataset)
        warnings: list[str] = []

        select_parts: list[str] = []
        group_parts: list[str] = []
        dimension_aliases: list[str] = []
        time_column = ""

        for dim in spec.dimensions:
            expr, alias, is_time = self._dimension_sql(dataset, dim)
            select_parts.append(f"{expr} AS {_quote_ident(alias)}")
            group_parts.append(expr)
            dimension_aliases.append(alias)
            if is_time and not time_column:
                time_column = alias

        metric_exprs: dict[str, str] = {}
        metric_aliases: list[str] = []
        for metric in spec.metrics:
            expr, alias = self._metric_sql(dataset, metric)
            select_parts.append(f"{expr} AS {_quote_ident(alias)}")
            metric_exprs[alias] = expr
            metric_aliases.append(alias)

        if not select_parts:
            select_parts.append("*")
            warnings.append("no dimensions or metrics given; selecting all columns")

        where: list[str] = []
        having: list[str] = []
        for flt in spec.filters:
            # A filter naming a metric alias is a post-aggregation condition.
            if flt.column in metric_exprs:
                having.append(self._filter_sql(dataset, flt, expr_override=metric_exprs[flt.column]))
            else:
                where.append(self._filter_sql(dataset, flt))

        produced = dimension_aliases + metric_aliases
        order: list[str] = []
        for sort in spec.sort:
            alias = self._resolve_sort(sort, produced, dataset, warnings)
            if alias:
                order.append(f"{_quote_ident(alias)} {'DESC' if sort.descending else 'ASC'}")

        if not order and metric_aliases and dimension_aliases:
            # Sensible default: biggest first for categories, chronological for time.
            if time_column:
                order.append(f"{_quote_ident(time_column)} ASC")
            else:
                order.append(f"{_quote_ident(metric_aliases[0])} DESC")

        limit = max(1, min(int(spec.limit or settings.default_limit), settings.max_rows))

        sql = f"SELECT\n  " + ",\n  ".join(select_parts)
        sql += f"\nFROM {_quote_ident(dataset.name)}"
        if where:
            sql += "\nWHERE " + "\n  AND ".join(where)
        if group_parts and metric_aliases:
            sql += "\nGROUP BY " + ", ".join(group_parts)
        if having:
            if not metric_aliases:
                raise QueryValidationError("HAVING requires at least one metric")
            sql += "\nHAVING " + "\n  AND ".join(having)
        if order:
            sql += "\nORDER BY " + ", ".join(order)
        sql += f"\nLIMIT {limit}"

        return CompiledQuery(
            spec=spec,
            sql=sql,
            columns=produced or [c.name for c in dataset.columns],
            dimension_aliases=dimension_aliases,
            metric_aliases=metric_aliases,
            time_column=time_column,
            warnings=warnings,
        )

    def _resolve_sort(
        self, sort: Sort, produced: list[str], dataset: DatasetMeta, warnings: list[str]
    ) -> str:
        if sort.field in produced:
            return sort.field
        lowered = sort.field.lower()
        for alias in produced:
            if alias.lower() == lowered:
                return alias
        # The model often sorts by the underlying column instead of the alias.
        resolved = dataset.resolve(sort.field)
        if resolved:
            for alias in produced:
                if alias == resolved or alias.endswith(f"_{resolved}") or alias.startswith(resolved):
                    return alias
        warnings.append(f"dropped sort on {sort.field!r}: not among output columns {produced}")
        return ""
