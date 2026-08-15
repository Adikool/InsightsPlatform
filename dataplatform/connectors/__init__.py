from ..catalog.models import SourceMeta
from ..errors import ConnectorError
from .base import Connector, ObjectInfo
from .files import CSVConnector, ExcelConnector, ParquetConnector, normalise_columns
from .sql import SQLConnector

_REGISTRY: dict[str, type[Connector]] = {
    "sql": SQLConnector,
    "excel": ExcelConnector,
    "csv": CSVConnector,
    "parquet": ParquetConnector,
}


def open_connector(meta: SourceMeta) -> Connector:
    try:
        cls = _REGISTRY[meta.type]
    except KeyError:
        raise ConnectorError(
            f"no connector for type {meta.type!r}; known: {sorted(_REGISTRY)}"
        ) from None
    return cls(meta)


def infer_source_type(uri: str) -> str:
    lowered = uri.lower()
    if "://" in lowered and not lowered.startswith("file://"):
        return "sql"
    if lowered.endswith((".xlsx", ".xls", ".xlsm")):
        return "excel"
    if lowered.endswith(".parquet"):
        return "parquet"
    return "csv"


__all__ = [
    "CSVConnector",
    "Connector",
    "ExcelConnector",
    "ObjectInfo",
    "ParquetConnector",
    "SQLConnector",
    "infer_source_type",
    "normalise_columns",
    "open_connector",
]
