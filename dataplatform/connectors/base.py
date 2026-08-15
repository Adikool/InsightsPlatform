"""Connector contract.

A connector knows how to *list* addressable objects in a source and *read* one of
them into a DataFrame. It knows nothing about the warehouse, the catalog, or SQL
generation — ingestion orchestrates those.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import pandas as pd

from ..catalog.models import SourceMeta


@dataclass
class ObjectInfo:
    """One readable thing inside a source: a table, a sheet, a file."""

    name: str
    kind: str  # table | view | sheet | file
    schema: str | None = None
    estimated_rows: int | None = None

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name


class Connector(abc.ABC):
    type: str = "base"

    def __init__(self, meta: SourceMeta) -> None:
        self.meta = meta

    @abc.abstractmethod
    def test(self) -> None:
        """Raise ConnectorError if the source is unreachable."""

    @abc.abstractmethod
    def list_objects(self) -> list[ObjectInfo]:
        ...

    @abc.abstractmethod
    def read(self, obj: str, limit: int | None = None) -> pd.DataFrame:
        ...

    def close(self) -> None:  # pragma: no cover - most connectors are stateless
        pass

    def __enter__(self) -> "Connector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
