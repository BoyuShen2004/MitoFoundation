"""Central ingestion registry for mitoFoundation2.

data/registry.sqlite is the source of truth for:
  providers, datasets, assets, downloads, preprocess_runs
"""
from .schema import connect, migrate, DEFAULT_REGISTRY_PATH

__all__ = ["connect", "migrate", "DEFAULT_REGISTRY_PATH"]
