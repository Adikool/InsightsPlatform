"""SQLAlchemy connector — Postgres, MySQL, SQL Server, SQLite, Snowflake, anything
with a dialect installed."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from ..catalog.models import SourceMeta
from ..errors import ConnectorError
from .base import Connector, ObjectInfo


class SQLConnector(Connector):
    type = "sql"

    def __init__(self, meta: SourceMeta) -> None:
        super().__init__(meta)
        try:
            self._engine = create_engine(meta.uri, pool_pre_ping=True)
        except SQLAlchemyError as exc:
            raise ConnectorError(f"could not build an engine for {meta.name}: {exc}") from exc

    def test(self) -> None:
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise ConnectorError(f"source {self.meta.name} is unreachable: {exc}") from exc

    def list_objects(self) -> list[ObjectInfo]:
        try:
            inspector = inspect(self._engine)
            default_schema = inspector.default_schema_name
            schemas = self.meta.options.get("schemas")
            if not schemas:
                schemas = [default_schema] if default_schema else [None]

            objects: list[ObjectInfo] = []
            for schema in schemas:
                # Qualifying with the default schema buys nothing and leaks into
                # every downstream name ("shop_main_orders" instead of "shop_orders").
                reported = None if schema == default_schema else schema
                for table in inspector.get_table_names(schema=schema):
                    objects.append(ObjectInfo(name=table, kind="table", schema=reported))
                for view in inspector.get_view_names(schema=schema):
                    objects.append(ObjectInfo(name=view, kind="view", schema=reported))
            return objects
        except SQLAlchemyError as exc:
            raise ConnectorError(f"could not list objects in {self.meta.name}: {exc}") from exc

    def read(self, obj: str, limit: int | None = None) -> pd.DataFrame:
        """`obj` is a table name, optionally schema-qualified.

        Identifiers are quoted through the dialect's own preparer rather than
        interpolated, so a table name from a hostile catalog cannot inject SQL.
        """
        preparer = self._engine.dialect.identifier_preparer
        parts = obj.split(".")
        quoted = ".".join(preparer.quote(part) for part in parts)
        sql = f"SELECT * FROM {quoted}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        try:
            with self._engine.connect() as conn:
                return pd.read_sql(text(sql), conn)
        except SQLAlchemyError as exc:
            raise ConnectorError(f"could not read {obj} from {self.meta.name}: {exc}") from exc

    def query(self, sql: str) -> pd.DataFrame:
        """Escape hatch for ingesting the result of a hand-written source query."""
        try:
            with self._engine.connect() as conn:
                return pd.read_sql(text(sql), conn)
        except SQLAlchemyError as exc:
            raise ConnectorError(f"query failed against {self.meta.name}: {exc}") from exc

    def close(self) -> None:
        self._engine.dispose()
