"""Tests for soft hide/unhide: datasets hidden from training are excluded from datalist.

Verifies:
- hide_dataset sets hidden_from_training=1
- unhide_dataset clears it
- is_dataset_hidden reflects the correct state
- list_hidden_datasets returns only hidden dataset IDs
- Hidden flag persists across registry open/close cycles
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent.orchestration.registry.schema import migrate
from agent.orchestration.registry.api import (
    upsert_provider,
    upsert_dataset,
    get_provider_id,
    get_dataset_id,
    hide_dataset,
    unhide_dataset,
    is_dataset_hidden,
    list_hidden_datasets,
)


def _mem() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return conn


_PROVIDER = "OpenOrganelle"
_DATASETS = ["jrc_hela-2", "jrc_jurkat-1", "jrc_macrophage-2"]


def _seed(conn) -> tuple[int, dict[str, int]]:
    pid = upsert_provider(conn, name=_PROVIDER, base_url="https://openorganelle.janelia.org")
    dids: dict[str, int] = {}
    for name in _DATASETS:
        dids[name] = upsert_dataset(conn, provider_id=pid, stable_id=name)
    conn.commit()
    return pid, dids


class TestHideUnhide:
    def test_new_dataset_not_hidden(self):
        conn = _mem()
        pid, dids = _seed(conn)
        for name, did in dids.items():
            assert not is_dataset_hidden(conn, did), f"{name} should not be hidden by default"

    def test_hide_sets_flag(self):
        conn = _mem()
        pid, dids = _seed(conn)
        name = _DATASETS[0]
        hide_dataset(conn, dids[name])
        conn.commit()
        assert is_dataset_hidden(conn, dids[name])

    def test_unhide_clears_flag(self):
        conn = _mem()
        pid, dids = _seed(conn)
        name = _DATASETS[0]
        hide_dataset(conn, dids[name])
        conn.commit()
        unhide_dataset(conn, dids[name])
        conn.commit()
        assert not is_dataset_hidden(conn, dids[name])

    def test_hide_only_affects_target_dataset(self):
        conn = _mem()
        pid, dids = _seed(conn)
        target = _DATASETS[0]
        hide_dataset(conn, dids[target])
        conn.commit()
        for name in _DATASETS[1:]:
            assert not is_dataset_hidden(conn, dids[name]), (
                f"{name} must not be hidden when only {target} was hidden"
            )

    def test_list_hidden_returns_correct_ids(self):
        conn = _mem()
        pid, dids = _seed(conn)
        hide_dataset(conn, dids[_DATASETS[0]])
        hide_dataset(conn, dids[_DATASETS[2]])
        conn.commit()
        hidden_ids = set(list_hidden_datasets(conn, pid))
        assert dids[_DATASETS[0]] in hidden_ids
        assert dids[_DATASETS[2]] in hidden_ids
        assert dids[_DATASETS[1]] not in hidden_ids

    def test_list_hidden_empty_when_none_hidden(self):
        conn = _mem()
        pid, dids = _seed(conn)
        assert list_hidden_datasets(conn, pid) == []

    def test_double_hide_idempotent(self):
        conn = _mem()
        pid, dids = _seed(conn)
        did = dids[_DATASETS[0]]
        hide_dataset(conn, did)
        hide_dataset(conn, did)
        conn.commit()
        assert is_dataset_hidden(conn, did)

    def test_double_unhide_idempotent(self):
        conn = _mem()
        pid, dids = _seed(conn)
        did = dids[_DATASETS[0]]
        hide_dataset(conn, did)
        conn.commit()
        unhide_dataset(conn, did)
        unhide_dataset(conn, did)
        conn.commit()
        assert not is_dataset_hidden(conn, did)

    def test_hide_persists_across_reconnect(self, tmp_path: Path):
        """Hidden flag survives close + reopen of a file-backed registry."""
        import sys
        import os
        repo_root = Path(__file__).resolve().parents[5]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from agent.orchestration.registry.schema import open_registry

        db_path = tmp_path / "test_registry.sqlite"
        conn1 = open_registry(db_path)
        pid = upsert_provider(conn1, name=_PROVIDER, base_url="")
        did = upsert_dataset(conn1, provider_id=pid, stable_id="jrc_test")
        hide_dataset(conn1, did)
        conn1.commit()
        conn1.close()

        conn2 = open_registry(db_path)
        pid2 = get_provider_id(conn2, _PROVIDER)
        did2 = get_dataset_id(conn2, pid2, "jrc_test")
        assert is_dataset_hidden(conn2, did2), "hidden_from_training must persist on disk"
        conn2.close()
