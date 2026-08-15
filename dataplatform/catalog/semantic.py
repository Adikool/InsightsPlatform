"""Renders the catalog into the compact text the language layer is grounded on.

This is deliberately terse: schema context is resent on every question, so it sits
in the cached prefix and every token is paid for repeatedly. Sample values matter
more than prose here — they are what let the model map "west coast" onto
`region = 'West'` without a round trip.
"""

from __future__ import annotations

from .models import DatasetMeta
from .store import Catalog

_MAX_SAMPLES = 8


def _column_line(col) -> str:
    bits = [f"  - {col.name} ({col.semantic_type}/{col.dtype})"]
    if col.description:
        bits.append(f"  # {col.description}")
    facts = []
    if col.n_unique is not None:
        facts.append(f"{col.n_unique} distinct")
    if col.null_fraction:
        facts.append(f"{col.null_fraction:.0%} null")
    if col.semantic_type in ("numeric", "currency", "temporal") and col.min is not None:
        facts.append(f"range {col.min}..{col.max}")
    if col.sample_values and col.semantic_type in ("categorical", "boolean", "geo", "text"):
        shown = ", ".join(str(v) for v in col.sample_values[:_MAX_SAMPLES])
        facts.append(f"values: {shown}")
    if col.synonyms:
        facts.append("aka " + ", ".join(col.synonyms))
    if facts:
        bits.append(" [" + "; ".join(facts) + "]")
    return "".join(bits)


def describe_dataset(dataset: DatasetMeta) -> str:
    header = f"TABLE {dataset.name}  ({dataset.n_rows:,} rows)"
    if dataset.description:
        header += f"\n  {dataset.description}"
    if dataset.grain:
        header += f"\n  grain: {dataset.grain}"
    lines = [header, "COLUMNS:"]
    lines.extend(_column_line(col) for col in dataset.columns)
    return "\n".join(lines)


def describe_catalog(catalog: Catalog, only: list[str] | None = None) -> str:
    datasets = catalog.list_datasets()
    if only:
        wanted = {name.lower() for name in only}
        datasets = [d for d in datasets if d.name.lower() in wanted]
    if not datasets:
        return "(no datasets have been ingested yet)"

    blocks = [describe_dataset(dataset) for dataset in datasets]
    if catalog.state.metrics:
        metric_lines = [f"  - {k}: {v}" for k, v in catalog.state.metrics.items()]
        blocks.append("DEFINED METRICS:\n" + "\n".join(metric_lines))
    return "\n\n".join(blocks)
