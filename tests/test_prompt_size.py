"""Guards on the cost of grounding the language layer.

Schema context is rebuilt and resent on every question, and the heuristic
parser walks every column of every table. Both were quadratic-ish in content
that nobody had bounded: a single text column holding an XML document or an
encoded image produced a 150k-character "sample value", which then went into
the model prompt and into a regex, per column, per table, per question.
"""

from __future__ import annotations

import pathlib
import time

from dataplatform.catalog import Catalog
from dataplatform.catalog.inference import profile_column
from dataplatform.catalog.models import CatalogState, ColumnMeta, DatasetMeta
from dataplatform.catalog.semantic import describe_catalog
from dataplatform.config import settings
from dataplatform.nlp import heuristic

import pandas as pd
import pytest


def _blob(n: int) -> str:
    return "<xml>" + ("payload " * (n // 8))


def _catalog_with_blob_samples(tmp_path, n_tables: int = 20) -> Catalog:
    """A catalog shaped like the real one: a few columns holding huge values."""
    datasets = {}
    for i in range(n_tables):
        name = f"table_{i}"
        datasets[name] = DatasetMeta(
            name=name, source="s", origin_object=name, n_rows=1000,
            columns=[
                ColumnMeta(name="region", dtype="str", semantic_type="categorical",
                           sample_values=["North", "South"]),
                ColumnMeta(name="document", dtype="str", semantic_type="text",
                           sample_values=[_blob(20_000) for _ in range(8)]),
            ],
        )
    catalog = Catalog(tmp_path / "catalog.json")
    catalog.state = CatalogState(datasets=datasets)
    return catalog


def test_sample_values_are_clipped_when_profiled():
    """A 150k-character value is not a sample; it is the whole document."""
    series = pd.Series([_blob(50_000), _blob(50_000), "short"])
    col = profile_column("document", series)
    assert col.sample_values, "text columns should still carry samples"
    assert all(len(v) <= 80 for v in col.sample_values), (
        f"longest sample was {max(len(v) for v in col.sample_values)} chars"
    )


def test_prompt_stays_bounded_despite_huge_values(tmp_path):
    """The rendered schema must not scale with the size of the data itself."""
    catalog = _catalog_with_blob_samples(tmp_path, n_tables=20)
    rendered = describe_catalog(catalog)
    # Unclipped this is ~20 tables x 8 samples x 20k chars = ~3.2M.
    assert len(rendered) < 60_000, f"schema context ballooned to {len(rendered):,} chars"


def test_parser_does_not_slow_down_with_huge_sample_values(tmp_path):
    """The regex per sample value was the slowest thing in the parser.

    A whole-word match cannot occur unless the value is a substring of the
    question, so the cheap check gates the expensive one.
    """
    catalog = _catalog_with_blob_samples(tmp_path, n_tables=20)
    start = time.perf_counter()
    heuristic.parse("revenue by region", catalog, settings.default_limit)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"parsing took {elapsed:.2f}s against blob-valued columns"


def test_gating_still_matches_real_sample_values(tmp_path):
    """The optimisation must not stop short values from being matched."""
    datasets = {
        "orders": DatasetMeta(
            name="orders", source="s", origin_object="orders", n_rows=100,
            columns=[
                ColumnMeta(name="region", dtype="str", semantic_type="categorical",
                           sample_values=["West", "North"]),
                ColumnMeta(name="revenue", dtype="float", semantic_type="currency"),
            ],
        )
    }
    catalog = Catalog(tmp_path / "catalog.json")
    catalog.state = CatalogState(datasets=datasets)

    spec = heuristic.parse("revenue in the West", catalog, settings.default_limit)
    assert spec.dataset == "orders"
    assert any(f.column == "region" and "West" in f.values for f in spec.filters), (
        "a sample value named in the question should still become a filter"
    )


# --------------------------------------------------- activity replay is read-only
def test_activity_click_never_publishes():
    """Loading a history entry must not write to Superset.

    Clicking an entry restores what was asked for and previews it; publishing
    stays a separate, deliberate click. Pinned here because the opposite -
    replaying a logged "publish" - silently created dashboards from a click
    that reads like navigation.
    """
    import re

    app_js = pathlib.Path("dataplatform/api/static/app.js").read_text(encoding="utf-8")
    start = app_js.index("function renderDashboardActivity")
    body = app_js[start : app_js.index("\nfunction ", start + 1)]

    calls = re.findall(r"runDashboard\(([^)]*)\)", body)
    assert calls, "the entry click should still preview"
    assert all(c.strip() == "false" for c in calls), (
        f"activity replay must never publish, found runDashboard({calls})"
    )
    assert "Open dashboard" in body, "a published entry needs its own explicit open action"
