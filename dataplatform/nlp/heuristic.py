"""Deterministic fallback parser.

Used when no model credentials are present, and as the seed for prompt-free tests.
It is a keyword parser, not a language model: it handles the common analytical
shapes ("<agg> <measure> by <dimension> per <grain> for <filter>, top N") and
gives up honestly on anything else rather than inventing a query.
"""

from __future__ import annotations

import re

from ..catalog import Catalog, DatasetMeta
from ..errors import QueryValidationError
from .spec import Dimension, Filter, Metric, QuerySpec, Sort

_AGG_WORDS = {
    "sum": "sum", "total": "sum", "revenue": "sum", "sales": "sum",
    "average": "avg", "avg": "avg", "mean": "avg",
    "median": "median",
    "max": "max", "maximum": "max", "highest": "max", "peak": "max",
    "min": "min", "minimum": "min", "lowest": "min",
    "count": "count", "number": "count", "how many": "count", "volume": "count",
    "distinct": "count_distinct", "unique": "count_distinct",
    "spread": "stddev", "deviation": "stddev", "variance": "stddev",
}

_GRAIN_WORDS = {
    "daily": "day", "day": "day", "per day": "day",
    "weekly": "week", "week": "week",
    "monthly": "month", "month": "month",
    "quarterly": "quarter", "quarter": "quarter",
    "yearly": "year", "annual": "year", "annually": "year", "year": "year",
}


def _tokens(text: str) -> list[str]:
    """Word pieces, splitting on underscores too.

    Keeping `_` inside the character class made "revenue_target" a single token, so
    it never matched the words "revenue" and "target" in a question — which is the
    whole job here.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


_STOPWORDS = {
    "the", "a", "an", "of", "for", "by", "per", "in", "on", "at", "to", "and", "or",
    "show", "me", "give", "what", "which", "how", "many", "much", "is", "are", "was",
    "were", "top", "bottom", "last", "first", "total", "all", "with", "from", "over",
    "each", "every", "across", "split", "grouped", "between", "during", "get", "list",
}


def pick_dataset(question: str, catalog: Catalog) -> DatasetMeta:
    datasets = catalog.list_datasets()
    if not datasets:
        raise QueryValidationError("the catalog is empty; ingest something first")
    if len(datasets) == 1:
        return datasets[0]

    lowered = question.lower()
    content = {t for t in _tokens(question) if t not in _STOPWORDS and not t.isdigit()}

    def richness(dataset: DatasetMeta) -> tuple[int, int, int]:
        """How central a table is, used only to break ties.

        When a question carries no discriminating signal — "create a sales
        dashboard with kpi cards" names no column at all — every dataset scores
        zero and insertion order decides, which is arbitrary. The fact table is
        the better default: it has a time axis, the most measures, and the most
        rows, which is what someone asking for an overview almost always means.
        """
        has_time = 1 if dataset.temporal_columns else 0
        return has_time, len(dataset.measures), dataset.n_rows

    def score(dataset: DatasetMeta) -> tuple[int, int, int, int, tuple[int, int, int]]:
        """(name hits, tokens explained, phrase matches, weak hits, richness).

        Coverage first, because it is what actually discriminates: "revenue target
        by region" and an orders table both share *revenue* and *region*, and only
        the targets table accounts for *target*. Whichever dataset leaves fewer of
        the user's words unexplained is the one they meant. Richness only applies
        when the question itself cannot tell them apart.
        """
        explained: set[str] = set()
        named: set[str] = set()
        phrases = 0
        weak = 0

        # A table's own name is the strongest signal about what it holds. Matched
        # through the same singular/plural variants as columns, so "customer
        # dashboard" reaches `sales_db_customers` — and scored separately, because
        # `sales_db_orders` also contains a `customer_id` column and would
        # otherwise tie on the word "customer" despite not being the customer table.
        for token in _tokens(dataset.name):
            for word in content:
                if token == word or token in _variants(word) or word in _variants(token):
                    named.add(word)
                    explained.add(word)

        for col in dataset.columns:
            readable = col.name.replace("_", " ").lower()
            if _mentions(lowered, readable) or _mentions(lowered, col.name.lower()):
                phrases += 1
                explained |= {t for t in _tokens(col.name) if t in content}
            else:
                hits = {t for t in _tokens(col.name) if t in content}
                if hits:
                    weak += 1
                    explained |= hits
            for value in col.sample_values[:8]:
                text = str(value).strip().lower()
                if len(text) >= 3 and re.search(rf"\b{re.escape(text)}\b", lowered):
                    explained |= {t for t in _tokens(text) if t in content}
                    phrases += 1
                    break

        return len(named), len(explained), phrases, weak, richness(dataset)

    return max(datasets, key=score)


def _variants(phrase: str) -> set[str]:
    """Singular/plural spellings of a column name as a person would type it.

    "product_category" has to match "product categories"; a bare `s?` suffix does not.
    """
    forms = {phrase}
    if phrase.endswith("y"):
        forms.add(phrase[:-1] + "ies")
    if phrase.endswith(("s", "x", "ch", "sh")):
        forms.add(phrase + "es")
    else:
        forms.add(phrase + "s")
    if phrase.endswith("ies"):
        forms.add(phrase[:-3] + "y")
    if phrase.endswith("s") and len(phrase) > 3:
        forms.add(phrase[:-1])
    return forms


def _mentions(question: str, phrase: str) -> bool:
    return any(
        re.search(rf"\b{re.escape(form)}\b", question) for form in _variants(phrase) if form
    )


def _match_columns(question: str, dataset: DatasetMeta) -> list[str]:
    """Columns explicitly named in the question, longest name first."""
    lowered = question.lower()
    hits: list[tuple[int, str]] = []
    for col in dataset.columns:
        readable = col.name.replace("_", " ").lower()
        for candidate in {col.name.lower(), readable, *(s.lower() for s in col.synonyms)}:
            if _mentions(lowered, candidate):
                hits.append((len(candidate), col.name))
                break
    hits.sort(reverse=True)
    seen: set[str] = set()
    ordered = []
    for _, name in hits:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def parse(question: str, catalog: Catalog, default_limit: int = 1000) -> QuerySpec:
    dataset = pick_dataset(question, catalog)
    lowered = question.lower()
    named = _match_columns(question, dataset)

    # --- aggregate --------------------------------------------------------
    func = "sum"
    for word, mapped in _AGG_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            func = mapped
            break

    # A column the question actually names wins, even if it was inferred
    # categorical — "average delivery days" means avg(delivery_days).
    named_numeric = [
        c for c in dataset.columns if c.name in named and c.is_numeric_dtype and not c.semantic_type == "identifier"
    ]
    measures = named_numeric or dataset.measures
    if func in ("count", "count_distinct") and not named_numeric:
        ids = [c for c in dataset.columns if c.semantic_type == "identifier" and c.name in named]
        metric = (
            Metric(column=ids[0].name, func="count_distinct")
            if ids and func == "count_distinct"
            else Metric(column="*", func="count", alias="row_count")
        )
    elif measures:
        metric = Metric(column=measures[0].name, func=func)
    else:
        metric = Metric(column="*", func="count", alias="row_count")

    # --- dimensions -------------------------------------------------------
    dimensions: list[Dimension] = []
    grain = "none"
    for word, mapped in _GRAIN_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            grain = mapped
            break

    temporal = dataset.temporal_columns
    if grain != "none" and temporal:
        preferred = next((c for c in temporal if c.name in named), temporal[0])
        dimensions.append(Dimension(column=preferred.name, time_grain=grain))

    # Group-by candidates: any non-temporal dimension the question names. A "by X"
    # fragment narrows the field, but the dimension is as often *before* the "by"
    # ("top 5 product categories by revenue"), so it is a preference, not a filter.
    candidates = [
        col
        for col in dataset.dimensions
        if col.semantic_type not in ("temporal", "identifier")
        and col.name in named
        and col.name != metric.column  # the measure is not also the grouping
    ]
    grouped = re.search(r"\b(?:by|per|across|split by|grouped by)\s+([a-z0-9_ ,]+)", lowered)
    if grouped and candidates:
        fragment = grouped.group(1)
        preferred = [
            col
            for col in candidates
            if _mentions(fragment, col.name.replace("_", " ").lower())
            or _mentions(fragment, col.name.lower())
        ]
        candidates = preferred or candidates

    for col in candidates:
        if all(d.column != col.name for d in dimensions):
            dimensions.append(Dimension(column=col.name))

    if not dimensions:
        for name in named:
            col = dataset.column(name)
            if col and col.semantic_type == "temporal":
                dimensions.append(Dimension(column=col.name, time_grain="month"))
                break

    # --- filters ----------------------------------------------------------
    filters: list[Filter] = []
    relative = re.search(r"last\s+(\d+)\s+(day|week|month|year)s?", lowered)
    if relative and temporal:
        n, unit = int(relative.group(1)), relative.group(2)
        days = {"day": 1, "week": 7}.get(unit)
        if days:
            filters.append(
                Filter(column=temporal[0].name, op="last_n_days", values=[str(n * days)])
            )
        else:
            months = n * (12 if unit == "year" else 1)
            filters.append(
                Filter(column=temporal[0].name, op="last_n_months", values=[str(months)])
            )

    year = re.search(r"\b(?:in|during|for)\s+(20\d{2})\b", lowered)
    if year and temporal:
        start = f"{year.group(1)}-01-01"
        end = f"{year.group(1)}-12-31"
        filters.append(Filter(column=temporal[0].name, op="between", values=[start, end]))

    grouped_columns = {d.column for d in dimensions}
    for col in dataset.columns:
        if col.semantic_type not in ("categorical", "geo", "boolean"):
            continue
        if col.name in grouped_columns:
            continue  # grouping by a column and filtering it to one value is rarely meant
        for value in col.sample_values[:20]:
            text = str(value).strip().lower()
            # Short, empty or numeric sample values match incidental words and digits
            # in the question ("top 5" became delivery_days = 5). Require a real word.
            if len(text) < 3 or text.replace(".", "").isdigit():
                continue
            if re.search(rf"\b{re.escape(text)}\b", lowered):
                filters.append(Filter(column=col.name, op="=", values=[str(value)]))
                break

    # --- shaping ----------------------------------------------------------
    top = re.search(r"\b(?:top|bottom|first)\s+(\d+)", lowered)
    limit = int(top.group(1)) if top else default_limit
    descending = not lowered.strip().startswith("bottom") and "bottom" not in lowered

    metric_alias = metric.alias or f"{metric.func}_{metric.column}"
    time_dim = next((d for d in dimensions if d.time_grain != "none"), None)
    sort = [Sort(field=f"{time_dim.column}_{time_dim.time_grain}", descending=False)] if time_dim else [
        Sort(field=metric_alias, descending=descending)
    ]

    if not dimensions:
        chart = "big_number"
    elif time_dim:
        chart = "line"
    elif len(dimensions) >= 2:
        chart = "heatmap"
    else:
        chart = "bar"

    return QuerySpec(
        dataset=dataset.name,
        dimensions=dimensions,
        metrics=[metric],
        filters=filters,
        sort=sort,
        limit=limit,
        chart=chart,
        title=question.strip()[:80].rstrip("?").title(),
        # Say what was understood, not what was unavailable. The old wording led
        # with "Parsed without a language model", which reads as an error even
        # though the query succeeded — and it told the reader nothing about
        # whether the interpretation was right, which is the only thing they can
        # actually check.
        explanation=_describe(dataset, metric, dimensions, filters),
    )


def _describe(dataset: DatasetMeta, metric: Metric, dimensions, filters) -> str:
    """Plain-English readback of the spec, so a wrong reading is obvious."""
    verb = {
        "sum": "Total", "avg": "Average", "median": "Median", "min": "Minimum",
        "max": "Maximum", "count": "Count of", "count_distinct": "Distinct",
        "stddev": "Spread of",
    }.get(metric.func, metric.func)
    measure = "rows" if metric.column == "*" else metric.column.replace("_", " ")
    parts = [f"{verb} {measure}"]

    if dimensions:
        grouped = [
            f"{d.column.replace('_', ' ')}"
            + (f" per {d.time_grain}" if d.time_grain != "none" else "")
            for d in dimensions
        ]
        parts.append("grouped by " + ", ".join(grouped))

    for flt in filters:
        column = flt.column.replace("_", " ")
        if flt.op in ("last_n_days", "last_n_months"):
            unit = "days" if flt.op == "last_n_days" else "months"
            parts.append(f"limited to the last {flt.values[0]} {unit}")
        elif flt.op == "between" and len(flt.values) == 2:
            parts.append(f"where {column} is between {flt.values[0]} and {flt.values[1]}")
        else:
            parts.append(f"where {column} {flt.op} {', '.join(flt.values) or ''}".strip())

    return f"Read as: {', '.join(parts)}, from {dataset.name}."
