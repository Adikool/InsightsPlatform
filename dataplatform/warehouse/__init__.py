"""Warehouse abstraction: the single place every layer reads analytical data from.

Two backends ship: DuckDB (zero-setup, file-based, great for a laptop demo) and
anything SQLAlchemy speaks (Postgres in particular, which is what you want the
moment Superset needs to reach the same data over the network).

The dialect string each backend exposes is what the SQL compiler branches on for
the handful of expressions that genuinely differ (median, stddev, date_trunc).
"""

from __future__ import annotations

import abc
from typing import Literal

import pandas as pd

from ..config import settings
from ..errors import WarehouseError

WriteMode = Literal["replace", "append"]


class Warehouse(abc.ABC):
    dialect: str = "ansi"

    @abc.abstractmethod
    def write(self, df: pd.DataFrame, table: str, mode: WriteMode = "replace") -> int:
        ...

    @abc.abstractmethod
    def query(self, sql: str) -> pd.DataFrame:
        ...

    @abc.abstractmethod
    def list_tables(self) -> list[str]:
        ...

    @abc.abstractmethod
    def drop(self, table: str) -> None:
        ...

    @property
    @abc.abstractmethod
    def sqlalchemy_uri(self) -> str:
        """The URI Superset should use to reach this warehouse."""

    def count(self, table: str) -> int:
        result = self.query(f'SELECT COUNT(*) AS n FROM "{table}"')
        return int(result.iloc[0]["n"])

    def sample(self, table: str, n: int = 1000) -> pd.DataFrame:
        return self.query(f'SELECT * FROM "{table}" LIMIT {int(n)}')


def open_warehouse(uri: str | None = None, schema: str | None = None) -> Warehouse:
    """Open the warehouse, optionally confined to one schema.

    `schema` is how per-user isolation is done on a SQL warehouse: each user's
    tables live in their own schema, so table names stay unqualified and the
    catalog/compiler/guard need no notion of who is asking. DuckDB ignores it -
    there, isolation is a separate file per user.
    """
    uri = uri or settings.warehouse_uri
    if uri.startswith("duckdb://"):
        from .duckdb_backend import DuckDBWarehouse

        return DuckDBWarehouse(uri)
    if "://" in uri:
        from .sql_backend import SQLWarehouse

        return SQLWarehouse(uri, schema=schema)
    raise WarehouseError(f"unrecognised warehouse URI: {uri!r}")


__all__ = ["Warehouse", "WriteMode", "open_warehouse"]
