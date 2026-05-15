"""Smoke tests for provider factory routing.

Runnable as:
  python -m pytest registry/providers/tests/
  python registry/providers/tests/test_factory_routing.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure the repo root is on the path when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from agent.orchestration.registry.providers import get_provider, known_providers, register_provider
from agent.orchestration.registry.providers.base import BaseProvider
from agent.orchestration.registry.providers.bossdb import BossDBProvider
from agent.orchestration.registry.providers.openorganelle import OpenOrganelleProvider


def _load_bridge_module():
    ws01 = _REPO_ROOT / "1web_scraper_01"
    if not (ws01 / "master" / "mito_foundation_bridge.py").is_file():
        pytest.skip(f"bridge module missing under: {ws01}")
    if str(ws01) not in sys.path:
        sys.path.insert(0, str(ws01))
    # Force re-import from 1web_scraper_01/master (avoid collision with 2database_builder/master).
    for k in list(sys.modules.keys()):
        if k == "master" or k.startswith("master."):
            del sys.modules[k]
    return importlib.import_module("master.mito_foundation_bridge")


# ── factory basics ─────────────────────────────────────────────────────────────

def test_get_known_provider():
    p = get_provider("openorganelle")
    assert isinstance(p, OpenOrganelleProvider)
    assert p.name == "OpenOrganelle"


def test_get_known_provider_case_insensitive():
    p1 = get_provider("OpenOrganelle")
    p2 = get_provider("OPENORGANELLE")
    assert type(p1) is type(p2)


def test_get_bossdb_provider():
    p = get_provider("bossdb")
    assert isinstance(p, BossDBProvider)
    assert p.name == "BossDB"


def test_get_unknown_provider_raises_key_error():
    with pytest.raises(KeyError, match="bossdb_stub"):
        get_provider("bossdb_stub")


def test_known_providers_includes_openorganelle():
    assert "openorganelle" in known_providers()
    assert "bossdb" in known_providers()


def test_openorganelle_implements_base_provider_protocol():
    p = get_provider("openorganelle")
    assert isinstance(p, BaseProvider)


# ── protocol extension (generate_downloader_script / studio_env_vars) ─────────

def test_openorganelle_has_generate_downloader_script():
    p = get_provider("openorganelle")
    assert hasattr(p, "generate_downloader_script")
    assert callable(p.generate_downloader_script)


def test_openorganelle_has_studio_env_vars():
    p = get_provider("openorganelle")
    assert hasattr(p, "studio_env_vars")
    env = p.studio_env_vars(
        chunk_zyx="512,512,512",
        n_crops=1,
        voxel_size_nm_zyx="8,8,8",
        physical_zyx="4096,4096,4096",
    )
    assert isinstance(env, dict)
    assert "MITO2_STUDIO_OG_CHUNK" in env
    assert env["MITO2_STUDIO_OG_CHUNK"] == "512,512,512"


# ── custom provider registration ───────────────────────────────────────────────

def test_register_custom_provider():
    import sqlite3

    class FakeProvider:
        name = "FakeTest"
        base_url = "https://example.com"

        def ingest_discovery(self, conn: sqlite3.Connection):
            return []

        def resolve_assets(self, conn: sqlite3.Connection):
            return []

        def get_spacing_nm(self, stable_id: str):
            return None

        def generate_downloader_script(self, **kw):
            return None

        def studio_env_vars(self, **kw):
            return {}

    register_provider("faketest", FakeProvider)
    assert "faketest" in known_providers()
    p = get_provider("faketest")
    assert p.name == "FakeTest"


def test_generic_provider_without_downloader_method():
    """A provider that omits generate_downloader_script → studio falls back to stub."""
    import sqlite3

    class MinimalProvider:
        name = "MinTest"
        base_url = "https://example.com"

        def ingest_discovery(self, conn: sqlite3.Connection):
            return []

        def resolve_assets(self, conn: sqlite3.Connection):
            return []

        def get_spacing_nm(self, stable_id: str):
            return None

    register_provider("mintest_no_dl", MinimalProvider)
    p = get_provider("mintest_no_dl")
    # Studio code guards with `hasattr(provider, "generate_downloader_script")`.
    assert not hasattr(p, "generate_downloader_script"), (
        "MinimalProvider should NOT have generate_downloader_script — "
        "callers should fall back to the generic stub when it's absent"
    )


# ── bridge dispatcher ──────────────────────────────────────────────────────────

def test_bridge_dispatch_unknown_provider_returns_error_dict():
    """run_provider_deep_scrape returns ok=False for unknown provider, does not raise."""
    bridge = _load_bridge_module()
    run_provider_deep_scrape = bridge.run_provider_deep_scrape

    result = run_provider_deep_scrape("empiar", Path("."), workspace_slug="test")
    assert result["ok"] is False
    assert result["error"] == "provider_not_implemented"
    assert "empiar" in result["message"].lower()


def test_bridge_provider_name_for_url():
    bridge = _load_bridge_module()
    provider_name_for_url = bridge.provider_name_for_url

    assert provider_name_for_url("https://openorganelle.janelia.org/") == "openorganelle"
    assert provider_name_for_url("https://www.openorganelle.janelia.org/datasets") == "openorganelle"
    assert provider_name_for_url("https://bossdb.org/") == "bossdb"
    assert provider_name_for_url("") is None


# ── self-runnable ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [(n, fn) for n, fn in sorted(globals().items()) if n.startswith("test_")]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
