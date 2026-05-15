"""Provider registry — factory and lookup for dataset catalog providers.

To add a new provider:
  1. Create ``registry/providers/<name>.py`` implementing ``BaseProvider``.
  2. Add an entry to ``_PROVIDER_CONSTRUCTORS`` below.
  3. Done — no other file needs to change.
"""

from __future__ import annotations

from typing import Any

from .base import AssetSpec, BaseProvider
from .bossdb import BossDBProvider
from .openorganelle import OpenOrganelleProvider

# Keyed by normalized (lowercase) provider name.
# Values are callables (class constructors or factory functions).
_PROVIDER_CONSTRUCTORS: dict[str, Any] = {
    "openorganelle": OpenOrganelleProvider,
    "bossdb": BossDBProvider,
}


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    """Return a provider instance for the given site name (case-insensitive).

    ``**kwargs`` are forwarded to the provider constructor, allowing callers to
    override defaults (e.g., ``probe_path=``, ``catalog_db=`` for OpenOrganelle).

    Raises ``KeyError`` for unknown provider names.
    """
    key = name.strip().lower()
    cls = _PROVIDER_CONSTRUCTORS.get(key)
    if cls is None:
        raise KeyError(
            f"Unknown provider {name!r}. "
            f"Known: {sorted(_PROVIDER_CONSTRUCTORS)}"
        )
    return cls(**kwargs)


def register_provider(name: str, constructor: Any) -> None:
    """Register a provider constructor at runtime (for plugins and tests)."""
    _PROVIDER_CONSTRUCTORS[name.strip().lower()] = constructor


def known_providers() -> list[str]:
    """Return sorted list of registered provider names."""
    return sorted(_PROVIDER_CONSTRUCTORS.keys())


__all__ = [
    "AssetSpec",
    "BaseProvider",
    "BossDBProvider",
    "OpenOrganelleProvider",
    "get_provider",
    "known_providers",
    "register_provider",
]
