"""Build a multi-tile dashboard from a dataset.

Two paths, same output. The deterministic one derives a conventional report shape
from the catalog's semantic types and works with no credentials at all; the model
path reads the request and picks the tiles. The deterministic path is not a
degraded fallback here — for "give me a sales dashboard" the conventional shape
(headline numbers, trend, breakdowns, detail) is usually the right answer, and it
is reproducible.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..catalog import Catalog, DatasetMeta, describe_catalog
from ..errors import PlatformError, QueryValidationError
from ..nlp.compiler import SQLCompiler
from ..nlp.llm import LLMClient, available
from ..nlp.spec import Dimension, Filter, Metric, QuerySpec, Sort
from .spec import (
    CHART_HEIGHT,
    GRID_COLUMNS,
    KPI_HEIGHT,
    TABLE_HEIGHT,
    DashboardPlan,
    DashboardSpec,
    Tile,
)

MAX_TILES = 10
DETAIL_ROWS = 200

INSTRUCTIONS = """\
You design a Superset dashboard as a set of tiles, each an independent QuerySpec
against the catalog below.

Compose 5-8 tiles in this shape unless the request says otherwise:
- 3-4 `kpi` tiles: one metric, no dimensions, chart="big_number". These are the
  headline numbers — totals, counts, averages, rates.
- 1 `trend` tile: the primary measure over the temporal column with a sensible
  time_grain, chart="line".
- 1-3 `breakdown` tiles: the primary measure by a categorical dimension,
  chart="bar" (or "pie" when there are few categories).
- 1 `detail` tile: chart="table", the dimensions and measures a reader would want
  to inspect row by row.

Rules:
- Use only columns from the catalog. Every tile must stand on its own.
- Give each tile a short `title` — it becomes the chart name in Superset.
- Set `role` on every tile. Leave `width` and `height` alone; layout is computed.
- Do not aggregate an identifier column with sum.
"""


_GRAINS = {
    "daily": "day", "day": "day",
    "weekly": "week", "week": "week",
    "monthly": "month", "month": "month",
    "quarterly": "quarter", "quarter": "quarter",
    "yearly": "year", "annual": "year", "annually": "year", "year": "year",
}


class Request(BaseModel):
    """Directives read out of the request text without a language model.

    The deterministic composer used to ignore this text entirely, which made the
    "what should it show?" field a false promise — asking for 3 KPI cards silently
    produced 4. These are the instructions it can honour; `understood` and
    `ignored` are reported back so a misreading is visible rather than mysterious.
    """

    n_kpis: int | None = None
    measure: str | None = None
    dimensions: list[str] = Field(default_factory=list)
    grain: str | None = None
    include_trend: bool = True
    include_detail: bool = True
    # (column, value) pairs for "one KPI for north, one for east, one for west" —
    # a request for per-value cards, not a request for N generic totals. These are
    # different intents that happen to both mention a number, and conflating them
    # is exactly the bug this field exists to avoid: "3 kpi cards, one for north,
    # one for east, one for west" was previously read as "make 3 generic cards",
    # silently discarding which three and what each should show.
    per_value_kpis: list[tuple[str, str]] = Field(default_factory=list)
    understood: list[str] = Field(default_factory=list)
    ignored: list[str] = Field(default_factory=list)


def parse_request(text: str, dataset: DatasetMeta) -> Request:
    """Keyword-level reading of a dashboard request. No model involved."""
    parsed = Request()
    if not text or not text.strip():
        return parsed

    lowered = text.lower()

    count = re.search(r"\b(\d+)\s*(?:kpi|card|tile|metric|number)s?\b", lowered) or re.search(
        r"\b(?:kpi|card|tile)s?\s*[:=]?\s*(\d+)\b", lowered
    )
    if count:
        parsed.n_kpis = max(1, min(int(count.group(1)), 6))
        parsed.understood.append(f"{parsed.n_kpis} KPI cards")

    for word, grain in _GRAINS.items():
        if re.search(rf"\b{word}\b", lowered):
            parsed.grain = grain
            parsed.understood.append(f"{grain}ly trend" if grain != "day" else "daily trend")
            break

    if re.search(r"\bno\s+(detail|table|grid)\b|\bwithout\s+(a\s+)?(detail|table)\b", lowered):
        parsed.include_detail = False
        parsed.understood.append("no detail table")
    if re.search(r"\bno\s+(trend|line|time\s*series)\b|\bwithout\s+(a\s+)?trend\b", lowered):
        parsed.include_trend = False
        parsed.understood.append("no trend chart")

    # Named columns: measures to headline, dimensions to break down by.
    for column in dataset.columns:
        readable = column.name.replace("_", " ").lower()
        if not re.search(rf"\b{re.escape(readable)}s?\b", lowered):
            continue
        if column.is_numeric_dtype and column.semantic_type != "identifier" and not parsed.measure:
            parsed.measure = column.name
            parsed.understood.append(f"headline measure {column.name}")
        elif column.semantic_type in ("categorical", "geo", "boolean"):
            parsed.dimensions.append(column.name)

    # Named *values* — "one for north, one for east, one for west" — are a
    # different signal from a named *column*: they ask for a card per value, not a
    # breakdown chart. Scan every categorical/geo/boolean column's actual sample
    # values (not the column name) for literal mentions in the text.
    # Position, not sample-value order: "for north, ... east, ... west" should
    # produce cards in that reading order, not whatever order the column's own
    # most-frequent-first sample list happens to store them in.
    value_hits: dict[str, list[tuple[int, str]]] = {}
    for column in dataset.columns:
        if column.semantic_type not in ("categorical", "geo", "boolean"):
            continue
        seen: set[str] = set()
        for value in column.sample_values:
            text_value = str(value).strip()
            if len(text_value) < 2 or text_value.lower() in seen:
                continue
            match = re.search(rf"\b{re.escape(text_value.lower())}\b", lowered)
            if match:
                value_hits.setdefault(column.name, []).append((match.start(), text_value))
                seen.add(text_value.lower())

    if value_hits:
        # Several columns can coincidentally match one word each; the column with
        # the most named values is almost certainly the one meant.
        best_column = max(value_hits, key=lambda c: len(value_hits[c]))
        values = [text_value for _, text_value in sorted(value_hits[best_column])]
        if len(values) >= 2:
            parsed.per_value_kpis = [(best_column, v) for v in values]
            parsed.understood.append(
                f"one KPI card per {best_column.replace('_', ' ')}: {', '.join(values)}"
            )
            if parsed.n_kpis is not None and parsed.n_kpis != len(values):
                parsed.understood.append(
                    f"named {len(values)} values, which takes priority over the "
                    f"'{parsed.n_kpis}' count"
                )

    if parsed.dimensions:
        parsed.understood.append("breakdowns by " + ", ".join(parsed.dimensions))

    # Say plainly what was seen and not acted on, rather than failing silently.
    for phrase, note in (
        (r"\bfilter\b|\bwhere\b|\bonly\b", "filters"),
        (r"\bcompare\b|\bvs\b|\bversus\b", "comparisons"),
        (r"\bforecast\b|\bpredict\b", "forecasting"),
        (r"\bpivot\b|\bcross ?tab\b", "pivot tables"),
    ):
        if re.search(phrase, lowered):
            parsed.ignored.append(note)

    return parsed


def _rate_like(name: str) -> bool:
    return bool(re.search(r"(pct|percent|rate|ratio|price|score|days|avg|average)", name, re.I))


def _pick_measures(dataset: DatasetMeta) -> list[str]:
    """Additive money-ish columns first — those are what headline KPIs are made of."""
    currency = [c.name for c in dataset.columns if c.semantic_type == "currency" and not _rate_like(c.name)]
    numeric = [
        c.name
        for c in dataset.columns
        if c.semantic_type == "numeric" and not _rate_like(c.name) and c.name not in currency
    ]
    return currency + numeric


def _pick_dimensions(dataset: DatasetMeta, limit: int = 3) -> list[str]:
    """Rank breakdown dimensions by how much they would actually tell a reader.

    Sorting by cardinality alone picks the *least* informative columns first — a
    free-text field that is 94% empty has two distinct values and sorts to the top,
    producing a chart with one visible bar. So: drop degenerate columns outright,
    then prefer the 3-12 distinct-value range where a bar chart is legible.
    """
    def score(column) -> tuple[int, int]:
        n = column.n_unique or 0
        if 3 <= n <= 12:
            band = 0
        elif 13 <= n <= 30:
            band = 1
        else:  # binary flags are last: real, but rarely the headline breakdown
            band = 2
        return band, n

    candidates = [
        c
        for c in dataset.columns
        if c.semantic_type in ("categorical", "geo", "boolean")
        and 2 <= (c.n_unique or 0) <= 30
        # A column dominated by one value yields a chart with a single visible bar.
        and (c.pct_top_value is None or c.pct_top_value <= 0.9)
    ]
    candidates.sort(key=score)
    return [c.name for c in candidates[:limit]]


def compose_default(dataset: DatasetMeta, title: str = "", request: str = "") -> DashboardSpec:
    """The conventional report shape, adjusted by whatever the request asked for."""
    asked = parse_request(request, dataset)

    measures = _pick_measures(dataset)
    if asked.measure:
        # An explicitly named measure leads; the rest stay available for extra KPIs.
        measures = [asked.measure] + [m for m in measures if m != asked.measure]

    dimensions = asked.dimensions or _pick_dimensions(dataset)
    temporal = dataset.temporal_columns[0].name if dataset.temporal_columns else None
    identifiers = [c.name for c in dataset.columns if c.semantic_type == "identifier"]

    if not measures and not identifiers:
        raise QueryValidationError(
            f"{dataset.name} has no numeric or identifier column to summarise; "
            "a dashboard needs at least one measure"
        )

    primary = measures[0] if measures else None
    tiles: list[Tile] = []

    def kpi(spec: QuerySpec, width: int = 3) -> Tile:
        return Tile(spec=spec, role="kpi", width=width, height=KPI_HEIGHT)

    # --- headline numbers -----------------------------------------------------
    if asked.per_value_kpis and primary:
        # "one for north, one for east, one for west": exactly those cards, each
        # the primary measure filtered to that value — not the generic
        # total/records/average/distinct set, which would silently answer a
        # different question than the one asked.
        n = len(asked.per_value_kpis)
        width = GRID_COLUMNS // n if n <= 4 else 3
        for column_name, value in asked.per_value_kpis:
            tiles.append(kpi(QuerySpec(
                dataset=dataset.name,
                metrics=[Metric(column=primary, func="sum", alias=f"total_{primary}")],
                filters=[Filter(column=column_name, op="=", values=[value])],
                chart="big_number",
                title=f"{value} {primary.replace('_', ' ')}",
                explanation=f"{primary} where {column_name} = {value}",
            ), width=width))
    elif primary:
        tiles.append(kpi(QuerySpec(
            dataset=dataset.name,
            metrics=[Metric(column=primary, func="sum", alias=f"total_{primary}")],
            chart="big_number", title=f"Total {primary.replace('_', ' ')}",
            explanation=f"sum of {primary} across all rows",
        )))

        tiles.append(kpi(QuerySpec(
            dataset=dataset.name,
            metrics=[Metric(column="*", func="count", alias="records")],
            chart="big_number", title="Records",
            explanation="row count",
        )))

        tiles.append(kpi(QuerySpec(
            dataset=dataset.name,
            metrics=[Metric(column=primary, func="avg", alias=f"avg_{primary}")],
            chart="big_number", title=f"Average {primary.replace('_', ' ')}",
            explanation=f"mean {primary} per row",
        )))

        # A distinct count of a *repeating* identifier reads as "how many
        # customers". A near-unique one is the primary key, where the count just
        # restates the row count we already show.
        def repeats(column) -> bool:
            share = (column.n_unique or 0) / max(dataset.n_rows, 1)
            return column.semantic_type == "identifier" and 0 < share < 0.9

        repeating = [c for c in dataset.columns if repeats(c)]
        if repeating:
            name = repeating[0].name
            tiles.append(kpi(QuerySpec(
                dataset=dataset.name,
                metrics=[Metric(column=name, func="count_distinct", alias=f"distinct_{name}")],
                chart="big_number", title=f"Distinct {name.replace('_', ' ')}",
                explanation=f"unique values of {name}",
            )))
        elif len(measures) > 1:
            second = measures[1]
            tiles.append(kpi(QuerySpec(
                dataset=dataset.name,
                metrics=[Metric(column=second, func="sum", alias=f"total_{second}")],
                chart="big_number", title=f"Total {second.replace('_', ' ')}",
                explanation=f"sum of {second}",
            )))
    else:
        tiles.append(kpi(QuerySpec(
            dataset=dataset.name,
            metrics=[Metric(column="*", func="count", alias="records")],
            chart="big_number", title="Records",
            explanation="row count",
        )))

    # Honour an explicit count on the *generic* KPI set only — per-value KPIs
    # already have an exact, meaningful count (one per named value) and truncating
    # them would drop a named region rather than an interchangeable card.
    if asked.n_kpis is not None and not asked.per_value_kpis:
        kpis = [t for t in tiles if t.role == "kpi"]
        if len(kpis) < asked.n_kpis:
            for extra in measures[1:]:
                if len(kpis) >= asked.n_kpis:
                    break
                if any(m.column == extra for t in kpis for m in t.spec.metrics):
                    continue
                tile = kpi(QuerySpec(
                    dataset=dataset.name,
                    metrics=[Metric(column=extra, func="sum", alias=f"total_{extra}")],
                    chart="big_number", title=f"Total {extra.replace('_', ' ')}",
                    explanation=f"sum of {extra}",
                ))
                tiles.append(tile)
                kpis.append(tile)
        keep = {id(t) for t in kpis[: asked.n_kpis]}
        tiles = [t for t in tiles if t.role != "kpi" or id(t) in keep]

    # --- trend --------------------------------------------------------------
    if asked.include_trend and temporal and primary:
        grain = asked.grain or "month"
        tiles.append(Tile(
            spec=QuerySpec(
                dataset=dataset.name,
                dimensions=[Dimension(column=temporal, time_grain=grain)],
                metrics=[Metric(column=primary, func="sum", alias=f"total_{primary}")],
                sort=[Sort(field=f"{temporal}_{grain}", descending=False)],
                chart="line",
                title=f"{primary.replace('_', ' ').title()} over time",
                explanation=f"{grain}ly {primary}",
            ),
            role="trend", width=12, height=CHART_HEIGHT,
        ))

    # --- breakdowns ---------------------------------------------------------
    if primary and dimensions:
        for index, dimension in enumerate(dimensions[:2]):
            column = dataset.column(dimension)
            few = (column.n_unique or 99) <= 6
            tiles.append(Tile(
                spec=QuerySpec(
                    dataset=dataset.name,
                    dimensions=[Dimension(column=dimension)],
                    metrics=[Metric(column=primary, func="sum", alias=f"total_{primary}")],
                    sort=[Sort(field=f"total_{primary}", descending=True)],
                    limit=25,
                    chart="pie" if (few and index == 1) else "bar",
                    title=f"{primary.replace('_', ' ').title()} by {dimension.replace('_', ' ')}",
                    explanation=f"{primary} grouped by {dimension}",
                ),
                role="breakdown", width=6, height=CHART_HEIGHT,
            ))

    # --- detail table -------------------------------------------------------
    detail_dims = [Dimension(column=d) for d in dimensions[:3]]
    if temporal:
        detail_dims.insert(0, Dimension(column=temporal, time_grain="month"))
    detail_metrics = [Metric(column=m, func="sum", alias=f"total_{m}") for m in measures[:3]]
    if not detail_metrics:
        detail_metrics = [Metric(column="*", func="count", alias="records")]

    if asked.include_detail and detail_dims:
        tiles.append(Tile(
            spec=QuerySpec(
                dataset=dataset.name,
                dimensions=detail_dims,
                metrics=detail_metrics,
                sort=[Sort(field=detail_metrics[0].alias, descending=True)],
                limit=DETAIL_ROWS,
                chart="table",
                title="Detail",
                explanation="row-level breakdown behind the charts above",
            ),
            role="detail", width=12, height=TABLE_HEIGHT,
        ))

    kept = tiles[:MAX_TILES]
    return DashboardSpec(
        title=title or f"{dataset.name.replace('_', ' ').title()} overview",
        description=f"Auto-composed from {dataset.name} ({dataset.n_rows:,} rows).",
        tiles=kept,
        interpretation=_readback(dataset, kept, asked),
    )


def _readback(dataset: DatasetMeta, tiles: list[Tile], asked: Request) -> str:
    """Say what was built and what was taken from the request.

    Same reasoning as the Ask view's "Read as:" line — a wrong reading should be
    obvious on the page, not something you infer from a chart that looks off.
    """
    counts: dict[str, int] = {}
    for tile in tiles:
        counts[tile.role] = counts.get(tile.role, 0) + 1

    parts = []
    if asked.per_value_kpis:
        labels = ", ".join(value for _, value in asked.per_value_kpis)
        parts.append(f"{counts.get('kpi', 0)} KPI cards ({labels})")
    elif counts.get("kpi"):
        parts.append(f"{counts['kpi']} KPI card{'s' if counts['kpi'] != 1 else ''}")
    if counts.get("trend"):
        parts.append("a trend chart")
    if counts.get("breakdown"):
        parts.append(f"{counts['breakdown']} breakdown{'s' if counts['breakdown'] != 1 else ''}")
    if counts.get("detail"):
        parts.append("a detail table")

    text = f"Built {', '.join(parts) or 'nothing'} from {dataset.name}."
    if asked.understood:
        text += " From your request: " + "; ".join(asked.understood) + "."
    if asked.ignored:
        text += (
            " Not applied without a model: "
            + ", ".join(sorted(set(asked.ignored)))
            + " — set ANTHROPIC_API_KEY for those."
        )
    return text


def compose_with_llm(
    request: str, catalog: Catalog, dataset: DatasetMeta | None, llm: LLMClient | None = None
) -> DashboardSpec:
    llm = llm or LLMClient()
    context = "CATALOG\n=======\n" + describe_catalog(
        catalog, only=[dataset.name] if dataset else None
    )
    plan = llm.structured(
        instructions=INSTRUCTIONS,
        context=context,
        question=request,
        output_model=DashboardPlan,
    )
    return DashboardSpec(
        title=plan.title,
        description=plan.description,
        tiles=[_size(tile) for tile in plan.tiles][:MAX_TILES],
    )


def _size(tile: Tile) -> Tile:
    """Widths come from the role, not the model — rows have to add up to 12."""
    sizes = {
        "kpi": (3, KPI_HEIGHT),
        "trend": (12, CHART_HEIGHT),
        "breakdown": (6, CHART_HEIGHT),
        "distribution": (6, CHART_HEIGHT),
        "detail": (12, TABLE_HEIGHT),
    }
    tile.width, tile.height = sizes.get(tile.role, (6, CHART_HEIGHT))
    return tile


def compose(
    catalog: Catalog,
    dataset_name: str,
    request: str = "",
    title: str = "",
    use_llm: bool | None = None,
) -> DashboardSpec:
    dataset = catalog.get_dataset(dataset_name)
    wants_llm = available() if use_llm is None else use_llm

    if wants_llm and request:
        try:
            return compose_with_llm(request, catalog, dataset)
        except PlatformError:
            # Covers no-credentials, but also a refusal, a rate limit, an
            # overload, a dropped connection, or a malformed structured-output
            # response — every failure mode of the call, not just "no key
            # configured". A flaky model call should degrade to the
            # deterministic composer, the same as no key at all, not 500 the
            # /dashboard endpoint.
            pass
    return compose_default(dataset, title=title, request=request)


def validate(spec: DashboardSpec, compiler: SQLCompiler) -> tuple[list[Tile], list[str]]:
    """Drop tiles that will not compile, and say which and why.

    A dashboard is worth more partially built than not at all: one bad tile out of
    eight should cost you that tile, not the report.
    """
    kept: list[Tile] = []
    problems: list[str] = []
    for tile in spec.tiles:
        try:
            compiler.compile(tile.spec)
            kept.append(tile)
        except QueryValidationError as exc:
            problems.append(f"dropped {tile.title!r}: {exc}")
    return kept, problems
