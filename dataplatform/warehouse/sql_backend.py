"""SQLAlchemy warehouse backend (Postgres, MySQL, SQL Server, ...)."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from ..errors import WarehouseError
from . import Warehouse, WriteMode


class SQLWarehouse(Warehouse):
    def __init__(self, uri: str, schema: str | None = None) -> None:
        self.uri = uri
        self.schema = schema
        try:
            self._engine = create_engine(uri, pool_pre_ping=True)
        except SQLAlchemyError as exc:
            raise WarehouseError(f"cannot open warehouse {uri}: {exc}") from exc
        self.dialect = self._engine.dialect.name
        if schema:
            self._bind_schema(schema)

    def _bind_schema(self, schema: str) -> None:
        """Confine this warehouse to one schema, creating it if needed.

        `write()` and `list_tables()` already take `schema=` explicitly, but
        `query()` runs arbitrary compiled SQL with unqualified table names, so
        the schema has to be bound at the connection level via search_path.
        Set on every pooled connection, not once, because the pool hands out
        fresh connections over time.
        """
        if not schema.replace("_", "").isalnum():
            raise WarehouseError(f"unsafe schema name {schema!r}")
        try:
            with self._engine.begin() as conn:
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        except SQLAlchemyError as exc:
            raise WarehouseError(f"cannot create schema {schema}: {exc}") from exc

        if self.dialect == "postgresql":

            @event.listens_for(self._engine, "connect")
            def _set_search_path(dbapi_conn, _record):  # pragma: no cover - driver level
                cursor = dbapi_conn.cursor()
                cursor.execute(f'SET search_path TO "{schema}"')
                cursor.close()

            # The pool may already hold connections opened before the listener
            # was attached; drop them so every future one is bound.
            self._engine.dispose()

    # Postgres (and most drivers) cap a single statement at 65535 bind parameters.
    # `method="multi"` packs chunksize x n_columns parameters into one INSERT, so a
    # fixed chunk size silently works on a narrow table and fails on a wide one.
    MAX_BIND_PARAMS = 60_000  # headroom under the 65535 protocol limit

    def _chunksize(self, n_columns: int) -> int:
        return max(1, self.MAX_BIND_PARAMS // max(n_columns, 1))

    def write(self, df: pd.DataFrame, table: str, mode: WriteMode = "replace") -> int:
        try:
            df.to_sql(
                table,
                self._engine,
                schema=self.schema,
                if_exists="replace" if mode == "replace" else "append",
                index=False,
                chunksize=self._chunksize(len(df.columns)),
                method="multi",
            )
        except SQLAlchemyError as exc:
            raise WarehouseError(f"write to {table} failed: {exc}") from exc
        return self.count(table)

    def query(self, sql: str) -> pd.DataFrame:
        try:
            with self._engine.connect() as conn:
                return pd.read_sql(text(sql), conn)
        except SQLAlchemyError as exc:
            raise WarehouseError(f"query failed: {exc}\n---\n{sql}") from exc

    def list_tables(self) -> list[str]:
        try:
            return sorted(inspect(self._engine).get_table_names(schema=self.schema))
        except SQLAlchemyError as exc:
            raise WarehouseError(f"cannot list tables: {exc}") from exc

    def drop(self, table: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))

    @property
    def sqlalchemy_uri(self) -> str:
        return self.uri
