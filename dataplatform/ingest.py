"""Ingestion: source object -> warehouse table -> catalog entry.

Ingestion is the only writer to the warehouse, and it always registers what it
wrote. That invariant is what lets the SQL compiler treat the catalog as a
trustworthy allow-list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .catalog import Catalog, DatasetMeta
from .catalog.inference import guess_grain, profile_frame
from .config import settings
from .connectors import normalise_columns, open_connector
from .errors import ConnectorError
from .warehouse import Warehouse, WriteMode


def safe_table_name(*parts: str) -> str:
    joined = "_".join(p for p in parts if p)
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", joined).strip("_").lower()
    if not name:
        name = "dataset"
    if name[0].isdigit():
        name = f"t_{name}"
    return name[:63]


@dataclass
class IngestResult:
    dataset: str
    rows: int
    columns: int
    source: str
    origin: str

    def __str__(self) -> str:
        return f"{self.source}:{self.origin} -> {self.dataset} ({self.rows:,} rows, {self.columns} cols)"


class Ingestor:
    def __init__(self, catalog: Catalog, warehouse: Warehouse) -> None:
        self.catalog = catalog
        self.warehouse = warehouse

    def ingest_object(
        self,
        source_name: str,
        obj: str,
        dataset_name: str | None = None,
        limit: int | None = None,
        mode: WriteMode = "replace",
        description: str = "",
    ) -> IngestResult:
        meta = self.catalog.get_source(source_name)
        with open_connector(meta) as connector:
            df = connector.read(obj, limit=limit)

        if df.empty:
            raise ConnectorError(f"{source_name}:{obj} returned no rows")

        df = normalise_columns(df)
        df = self._coerce_types(df)

        table = dataset_name or safe_table_name(source_name, obj.replace(".", "_"))
        n_rows = self.warehouse.write(df, table, mode=mode)

        columns = profile_frame(df, sample=settings.profile_sample)
        self.catalog.add_dataset(
            DatasetMeta(
                name=table,
                source=source_name,
                origin_object=obj,
                n_rows=n_rows,
                columns=columns,
                description=description,
                grain=guess_grain(columns, n_rows),
                ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )
        return IngestResult(table, n_rows, len(columns), source_name, obj)

    def ingest_source(
        self, source_name: str, limit: int | None = None, prefix: bool = True
    ) -> list[IngestResult]:
        meta = self.catalog.get_source(source_name)
        with open_connector(meta) as connector:
            objects = connector.list_objects()

        results: list[IngestResult] = []
        for info in objects:
            name = safe_table_name(source_name if prefix else "", info.qualified.replace(".", "_"))
            try:
                results.append(
                    self.ingest_object(source_name, info.qualified, dataset_name=name, limit=limit)
                )
            except ConnectorError as exc:
                # One unreadable sheet should not abort a whole workbook.
                results.append(IngestResult(f"!{name}", 0, 0, source_name, str(exc)))
        return results

    @staticmethod
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        """Promote obvious date strings to real timestamps before they land.

        Doing this at ingest rather than at query time means date_trunc works, the
        profiler sees a temporal column, and Superset offers a time axis.
        """
        from pandas.api import types as ptypes

        from .catalog.inference import _TIME_NAME, _looks_temporal

        out = df.copy()
        for col in out.columns:
            series = out[col]
            # pandas 3 gives string columns a StringDtype rather than object, so
            # testing against `object` alone silently skips every date-as-text column.
            if not (ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series)):
                continue
            if _TIME_NAME.search(str(col)) or _looks_temporal(series):
                converted = pd.to_datetime(series, errors="coerce", format="mixed")
                if converted.notna().mean() > 0.9:
                    out[col] = converted
        return out
