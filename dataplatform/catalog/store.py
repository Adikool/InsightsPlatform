"""JSON-backed catalog persistence."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

from ..config import settings
from ..errors import CatalogError
from .models import CatalogState, DatasetMeta, SourceMeta


class Catalog:
    """One JSON file's worth of catalog state, guarded by a re-entrant lock.

    FastAPI runs the sync route handlers in a threadpool, so several requests
    share this object concurrently. Without the lock, `save()` serialising
    `sources`/`datasets` while another thread ingests into them raises
    "dictionary changed size during iteration" - reachable today, because
    `ingest_source` calls `add_dataset` (and so `save`) once per table.
    The lock is re-entrant so the mutate-then-save pairs can nest freely.

    It does NOT make the file safe across processes: state is read once at
    construction and never re-read, so a second process writing the same file
    (a CLI command against a live server, or `uvicorn --workers 2`) still
    clobbers. See the README.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or settings.catalog_path)
        self._lock = threading.RLock()
        self.state = self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> CatalogState:
        if not self.path.exists():
            return CatalogState()
        try:
            return CatalogState.model_validate_json(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # corrupt catalog should not be silently reset
            raise CatalogError(f"catalog at {self.path} is unreadable: {exc}") from exc

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # model_dump(mode="json") gives a JSON-safe dict directly - no round-trip
            # through a string. atomic write so a crash never leaves a partial catalog.
            payload = self.state.model_dump(mode="json")
            with tempfile.NamedTemporaryFile(
                "w", dir=self.path.parent, delete=False, encoding="utf-8", suffix=".tmp"
            ) as handle:
                json.dump(payload, handle, indent=2)
                tmp = Path(handle.name)
            tmp.replace(self.path)

    # -------------------------------------------------------------- sources
    def add_source(self, source: SourceMeta) -> None:
        with self._lock:
            self.state.sources[source.name] = source
            self.save()

    def get_source(self, name: str) -> SourceMeta:
        with self._lock:
            try:
                return self.state.sources[name]
            except KeyError:
                raise CatalogError(
                    f"unknown source {name!r}; registered: {sorted(self.state.sources) or 'none'}"
                ) from None

    def list_sources(self) -> list[SourceMeta]:
        with self._lock:
            return list(self.state.sources.values())

    def remove_source(self, name: str) -> None:
        with self._lock:
            self.state.sources.pop(name, None)
            for ds_name in [d.name for d in self.state.datasets.values() if d.source == name]:
                self.state.datasets.pop(ds_name, None)
            self.save()

    # ------------------------------------------------------------- datasets
    def add_dataset(self, dataset: DatasetMeta) -> None:
        with self._lock:
            self.state.datasets[dataset.name] = dataset
            self.save()

    def get_dataset(self, name: str) -> DatasetMeta:
        with self._lock:
            if name in self.state.datasets:
                return self.state.datasets[name]
            lowered = name.lower()
            for key, dataset in self.state.datasets.items():
                if key.lower() == lowered:
                    return dataset
            raise CatalogError(
                f"unknown dataset {name!r}; available: {sorted(self.state.datasets) or 'none'}"
            )

    def list_datasets(self) -> list[DatasetMeta]:
        with self._lock:
            return list(self.state.datasets.values())

    def has_dataset(self, name: str) -> bool:
        with self._lock:
            return any(key.lower() == name.lower() for key in self.state.datasets)

    # ---------------------------------------------------------- semantic aid
    def define_metric(self, name: str, expression: str) -> None:
        """Register a named business metric, e.g. aov = sum(revenue)/count(order_id)."""
        with self._lock:
            self.state.metrics[name] = expression
            self.save()
