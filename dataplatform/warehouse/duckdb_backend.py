"""DuckDB warehouse backend."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from ..errors import WarehouseError
from . import Warehouse, WriteMode


class DuckDBWarehouse(Warehouse):
    dialect = "duckdb"

    def __init__(self, uri: str) -> None:
        raw = uri.replace("duckdb:///", "").replace("duckdb://", "")
        self.path = Path(raw)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self, read_only: bool = False):
        # DuckDB allows a single writer, so connections are per-operation rather
        # than held open. Read-only connections can still be concurrent.
        try:
            conn = duckdb.connect(str(self.path), read_only=read_only)
        except duckdb.Error as exc:
            raise WarehouseError(f"cannot open warehouse {self.path}: {exc}") from exc
        try:
            yield conn
        finally:
            conn.close()

    def write(self, df: pd.DataFrame, table: str, mode: WriteMode = "replace") -> int:
        if df.empty and mode == "append":
            return 0
        with self._connect() as conn:
            conn.register("_incoming", df)
            try:
                if mode == "replace":
                    conn.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _incoming')
                else:
                    exists = conn.execute(
                        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                        [table],
                    ).fetchone()[0]
                    if exists:
                        conn.execute(f'INSERT INTO "{table}" SELECT * FROM _incoming')
                    else:
                        conn.execute(
                            f'CREATE TABLE "{table}" AS SELECT * FROM _incoming'
                        )
            except duckdb.Error as exc:
                raise WarehouseError(f"write to {table} failed: {exc}") from exc
            finally:
                conn.unregister("_incoming")
            return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])

    def query(self, sql: str) -> pd.DataFrame:
        with self._connect(read_only=True) as conn:
            try:
                return conn.execute(sql).fetch_df()
            except duckdb.Error as exc:
                raise WarehouseError(f"query failed: {exc}\n---\n{sql}") from exc

    def list_tables(self) -> list[str]:
        if not self.path.exists():
            return []
        with self._connect(read_only=True) as conn:
            rows = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        return [r[0] for r in rows]

    def drop(self, table: str) -> None:
        with self._connect() as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')

    @property
    def sqlalchemy_uri(self) -> str:
        return f"duckdb:///{self.path.as_posix()}"
