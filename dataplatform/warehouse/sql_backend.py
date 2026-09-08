"""SQLAlchemy warehouse backend (Postgres, MySQL, SQL Server, ...)."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from ..errors import WarehouseError
from . import Warehouse, WriteMode


class SQLWarehouse(Warehouse):
    def __init__(self, uri: str, schema: str | None = None) -> None:
        self.uri = uri
        self.schema = schema
        if schema and not schema.replace("_", "").isalnum():
            raise WarehouseError(f"unsafe schema name {schema!r}")

        connect_args: dict = {}
        if schema and uri.startswith(("postgresql", "postgres")):
            # Set at connection startup, not with a later `SET`. `SET
            # search_path` is transactional: the pool issues a rollback when a
            # connection is returned, which silently reverts it, so the first
            # use of each physical connection worked and every reuse failed to
            # find the user's tables. A startup option cannot be rolled back.
            connect_args["options"] = f"-csearch_path={schema}"

        try:
            self._engine = create_engine(uri, pool_pre_ping=True, connect_args=connect_args)
        except SQLAlchemyError as exc:
            raise WarehouseError(f"cannot open warehouse {uri}: {exc}") from exc
        self.dialect = self._engine.dialect.name
        if schema:
            self._ensure_schema(schema)

    def _ensure_schema(self, schema: str) -> None:
        """Create this user's schema if it is not there yet.

        Only creation happens here - the connection is already pointed at the
        schema by `connect_args`, set in __init__. Naming a schema that does
        not exist in search_path is not an error in Postgres, so this DDL runs
        fine on such a connection.
        """
        try:
            with self._engine.begin() as conn:
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        except SQLAlchemyError as exc:
            raise WarehouseError(f"cannot create schema {schema}: {exc}") from exc

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
