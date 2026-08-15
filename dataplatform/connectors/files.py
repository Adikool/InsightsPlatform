"""Excel / CSV / Parquet connectors.

Excel is the messy one: a workbook is a *collection* of tables, headers are often
not on row 1, and column names arrive with trailing spaces and newlines. The
normalisation below is the difference between a usable catalog and one full of
`Unnamed: 4`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..catalog.models import SourceMeta
from ..errors import ConnectorError
from .base import Connector, ObjectInfo

_UNNAMED = re.compile(r"^unnamed:?\s*\d*$", re.I)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    names: list[str] = []
    for i, raw in enumerate(df.columns):
        name = re.sub(r"\s+", " ", str(raw)).strip()
        if not name or _UNNAMED.match(name):
            name = f"column_{i + 1}"
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower() or f"column_{i + 1}"
        if name[0].isdigit():
            name = f"c_{name}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    df = df.copy()
    df.columns = names
    return df


def _find_header_row(raw: pd.DataFrame, scan: int = 10) -> int:
    """Pick the first row that looks like a header: mostly non-null strings."""
    best_row, best_score = 0, -1.0
    for i in range(min(scan, len(raw))):
        row = raw.iloc[i]
        non_null = row.notna().sum()
        if non_null < 2:
            continue
        strings = sum(isinstance(v, str) and str(v).strip() != "" for v in row)
        score = strings / max(non_null, 1) * (non_null / max(len(row), 1))
        if score > best_score:
            best_row, best_score = i, score
    return best_row


class ExcelConnector(Connector):
    type = "excel"

    def __init__(self, meta: SourceMeta) -> None:
        super().__init__(meta)
        self.path = Path(meta.uri)

    def test(self) -> None:
        if not self.path.exists():
            raise ConnectorError(f"workbook not found: {self.path}")

    def list_objects(self) -> list[ObjectInfo]:
        self.test()
        try:
            book = pd.ExcelFile(self.path)
        except Exception as exc:
            raise ConnectorError(f"could not open {self.path}: {exc}") from exc
        return [ObjectInfo(name=sheet, kind="sheet") for sheet in book.sheet_names]

    def read(self, obj: str, limit: int | None = None) -> pd.DataFrame:
        self.test()
        header = self.meta.options.get("header")
        try:
            if header is None:
                probe = pd.read_excel(self.path, sheet_name=obj, header=None, nrows=15)
                header = _find_header_row(probe)
            df = pd.read_excel(self.path, sheet_name=obj, header=int(header), nrows=limit)
        except Exception as exc:
            raise ConnectorError(f"could not read sheet {obj!r} from {self.path}: {exc}") from exc
        return normalise_columns(df).dropna(how="all")


class CSVConnector(Connector):
    type = "csv"

    def __init__(self, meta: SourceMeta) -> None:
        super().__init__(meta)
        self.path = Path(meta.uri)

    def test(self) -> None:
        if not self.path.exists():
            raise ConnectorError(f"path not found: {self.path}")

    def _files(self) -> list[Path]:
        if self.path.is_dir():
            pattern = self.meta.options.get("glob", "*.csv")
            return sorted(self.path.glob(pattern))
        return [self.path]

    def list_objects(self) -> list[ObjectInfo]:
        self.test()
        return [ObjectInfo(name=p.stem, kind="file") for p in self._files()]

    def read(self, obj: str, limit: int | None = None) -> pd.DataFrame:
        self.test()
        matches = [p for p in self._files() if p.stem == obj] or self._files()
        if not matches:
            raise ConnectorError(f"no CSV matching {obj!r} under {self.path}")
        try:
            df = pd.read_csv(
                matches[0],
                nrows=limit,
                sep=self.meta.options.get("sep", ","),
                encoding=self.meta.options.get("encoding", "utf-8"),
            )
        except Exception as exc:
            raise ConnectorError(f"could not read {matches[0]}: {exc}") from exc
        return normalise_columns(df)


class ParquetConnector(CSVConnector):
    type = "parquet"

    def _files(self) -> list[Path]:
        if self.path.is_dir():
            return sorted(self.path.glob(self.meta.options.get("glob", "*.parquet")))
        return [self.path]

    def read(self, obj: str, limit: int | None = None) -> pd.DataFrame:
        self.test()
        matches = [p for p in self._files() if p.stem == obj] or self._files()
        if not matches:
            raise ConnectorError(f"no parquet matching {obj!r} under {self.path}")
        try:
            df = pd.read_parquet(matches[0])
        except Exception as exc:
            raise ConnectorError(f"could not read {matches[0]}: {exc}") from exc
        if limit:
            df = df.head(limit)
        return normalise_columns(df)
