"""Exception hierarchy. Every layer raises one of these so the CLI/API can map
them onto exit codes and HTTP statuses without guessing."""

from __future__ import annotations


class PlatformError(Exception):
    """Base class for everything this package raises deliberately."""


class ConnectorError(PlatformError):
    """A source could not be reached, read, or understood."""


class CatalogError(PlatformError):
    """The requested dataset/source is not registered."""


class WarehouseError(PlatformError):
    """The warehouse rejected an operation."""


class QueryValidationError(PlatformError):
    """A query spec or raw SQL statement failed validation.

    This is the security boundary: nothing that raises here reaches the engine.
    """


class LLMUnavailable(PlatformError):
    """No credentials or SDK for the language layer; caller may fall back."""


class SupersetError(PlatformError):
    """Superset returned an error or is unreachable."""
