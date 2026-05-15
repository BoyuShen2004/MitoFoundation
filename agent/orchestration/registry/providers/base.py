"""Provider protocol and shared types."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class AssetSpec:
    """Normalized description of one downloadable asset for a dataset."""

    stable_dataset_id: str
    asset_type: str          # "em_volume" | "mito_seg"
    remote_url: str
    voxel_size_nm: list[float] | None = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class BaseProvider(Protocol):
    """Interface every dataset provider must satisfy."""

    name: str
    base_url: str

    def ingest_discovery(self, conn: sqlite3.Connection) -> list[str]:
        """Read stage-1 discovery outputs → upsert providers/datasets in registry.

        Returns list of stable dataset IDs that were newly added or updated.
        """
        ...

    def resolve_assets(self, conn: sqlite3.Connection) -> list[AssetSpec]:
        """Read stage-2 DB → return normalized assets for download planning."""
        ...

    def get_spacing_nm(self, stable_id: str) -> list[float] | None:
        """Return ZYX voxel spacing (nm) for a dataset, or None if unknown."""
        ...

    def generate_downloader_script(
        self,
        *,
        mode: str,
        chunk_zyx: str,
        n_crops: int,
        voxel_size_nm_zyx: str,
        **kwargs: Any,
    ) -> Path | None:
        """Generate a provider-native downloader script and return its path.

        Return ``None`` if the provider does not support native script generation
        (callers should fall back to the generic volume-stub script).
        """
        ...

    def studio_env_vars(
        self,
        *,
        chunk_zyx: str,
        n_crops: int,
        voxel_size_nm_zyx: str,
        physical_zyx: str,
        no_foundation: bool = False,
    ) -> dict[str, str]:
        """Return extra environment variables for the downloader subprocess.

        Returns an empty dict for providers that do not need custom env vars.
        """
        ...
