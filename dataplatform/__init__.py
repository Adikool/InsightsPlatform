"""Insight Platform — connectors, an NL layer, Superset publishing, and a
data-science advisory layer, bound together in Python.

    from dataplatform import Platform

    p = Platform()
    p.add_source("sales", "sqlite:///retail.db")
    p.ingest("sales")
    result = p.ask("monthly revenue by region")
    report = p.analyze("sales_orders", target="revenue")
"""

from .catalog import Catalog, DatasetMeta, SourceMeta
from .config import settings
from .errors import (
    CatalogError,
    ConnectorError,
    LLMUnavailable,
    PlatformError,
    QueryValidationError,
    SupersetError,
    WarehouseError,
)
from .platform import AskResult, Platform

__version__ = "0.1.0"

__all__ = [
    "AskResult",
    "Catalog",
    "CatalogError",
    "ConnectorError",
    "DatasetMeta",
    "LLMUnavailable",
    "Platform",
    "PlatformError",
    "QueryValidationError",
    "SourceMeta",
    "SupersetError",
    "WarehouseError",
    "__version__",
    "settings",
]
