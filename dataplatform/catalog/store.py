"""JSON-backed catalog persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ..config import settings
from ..errors import CatalogError
from .models import CatalogState, DatasetMeta, SourceMeta


class Catalog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or settings.catalog_path)
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # model_dump(mode="json") gives a JSON-safe dict directly — no round-trip
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
        self.state.sources[source.name] = source
        self.save()

    def get_source(self, name: str) -> SourceMeta:
        try:
            return self.state.sources[name]
        except KeyError:
            raise CatalogError(
                f"unknown source {name!r}; registered: {sorted(self.state.sources) or 'none'}"
            ) from None

    def list_sources(self) -> list[SourceMeta]:
        return list(self.state.sources.values())

    def remove_source(self, name: str) -> None:
        self.state.sources.pop(name, None)
        for ds_name in [d.name for d in self.state.datasets.values() if d.source == name]:
            self.state.datasets.pop(ds_name, None)
        self.save()

    # ------------------------------------------------------------- datasets
    def add_dataset(self, dataset: DatasetMeta) -> None:
        self.state.datasets[dataset.name] = dataset
        self.save()

    def get_dataset(self, name: str) -> DatasetMeta:
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
        return list(self.state.datasets.values())

    def has_dataset(self, name: str) -> bool:
        return any(key.lower() == name.lower() for key in self.state.datasets)

    # ---------------------------------------------------------- semantic aid
    def define_metric(self, name: str, expression: str) -> None:
        """Register a named business metric, e.g. aov = sum(revenue)/count(order_id)."""
        self.state.metrics[name] = expression
        self.save()
