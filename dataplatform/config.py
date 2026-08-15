"""Runtime configuration, resolved from the environment once at import time."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional convenience, never required
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _home() -> Path:
    return Path(os.environ.get("DP_HOME", Path.home() / ".insight-platform"))


@dataclass
class Settings:
    home: Path = field(default_factory=_home)

    # --- warehouse -------------------------------------------------------
    warehouse_uri: str = ""

    # --- llm -------------------------------------------------------------
    model: str = os.environ.get("DP_MODEL", "claude-opus-5")
    effort: str = os.environ.get("DP_EFFORT", "medium")
    max_tokens: int = int(os.environ.get("DP_MAX_TOKENS", "8000"))

    # --- limits ----------------------------------------------------------
    max_rows: int = int(os.environ.get("DP_MAX_ROWS", "50000"))
    default_limit: int = int(os.environ.get("DP_DEFAULT_LIMIT", "1000"))
    profile_sample: int = int(os.environ.get("DP_PROFILE_SAMPLE", "200000"))

    # --- superset --------------------------------------------------------
    superset_url: str = os.environ.get("SUPERSET_URL", "http://localhost:8088")
    superset_username: str = os.environ.get("SUPERSET_USERNAME", "admin")
    superset_password: str = os.environ.get("SUPERSET_PASSWORD", "admin")
    superset_provider: str = os.environ.get("SUPERSET_PROVIDER", "db")

    # How *Superset* reaches the warehouse, which is not always how we reach it.
    # Superset normally runs in a container: our `localhost:55432` is its own
    # loopback, and a Windows file path is meaningless inside its filesystem.
    # Set this to the address that resolves from Superset's side, e.g.
    # postgresql+psycopg2://insight:insight@insight_warehouse:5432/insight
    superset_warehouse_uri: str = os.environ.get("DP_SUPERSET_WAREHOUSE_URI", "")

    def __post_init__(self) -> None:
        self.home = Path(self.home)
        self.home.mkdir(parents=True, exist_ok=True)
        if not self.warehouse_uri:
            default = f"duckdb:///{(self.home / 'warehouse.duckdb').as_posix()}"
            self.warehouse_uri = os.environ.get("DP_WAREHOUSE_URI", default)

    # --- derived ---------------------------------------------------------
    @property
    def catalog_path(self) -> Path:
        return self.home / "catalog.json"

    @property
    def has_llm(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


settings = Settings()
