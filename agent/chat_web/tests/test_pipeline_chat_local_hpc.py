from __future__ import annotations

from pathlib import Path

from agent.chat_web.app import pipeline_chat
from agent.orchestration.session_pipeline import PipelineSession


def test_source_alias_normalization_local_hpc() -> None:
    assert pipeline_chat._canonical_site_name("local hpc") == "LocalHPC"
    assert pipeline_chat._canonical_site_name("hpc-local") == "LocalHPC"
    assert pipeline_chat._canonical_site_name("mitole local") == "LocalHPC"
    assert pipeline_chat._canonical_site_name("open organelle") == "OpenOrganelle"
    assert pipeline_chat._canonical_site_name("boss db") == "BossDB"


def test_coerce_plan_keeps_unsupported_sources() -> None:
    plan = pipeline_chat._coerce_pipeline_plan(
        {
            "sites": ["local hpc", "bossdb", "mysterysource"],
            "stages": ["download"],
            "n_crops": 2,
        },
        user_message="run local hpc and bossdb",
    )
    assert plan["sites"] == ["LocalHPC", "BossDB"]
    assert "mysterysource" in plan["unsupported_sources"]


def test_execute_local_hpc_download_dispatch(monkeypatch) -> None:
    import agent.chat_web.app.studio_api as studio_api

    calls: dict[str, object] = {}

    monkeypatch.setattr(
        studio_api,
        "studio_mitole_catalogue",
        lambda regenerate=False: {
            "ok": True,
            "rows": [
                {
                    "dataset": "hela_set",
                    "sample_type": "hela",
                    "source": "mitole",
                    "image_path": "/tmp/im_1.nii.gz",
                    "label_path": "/tmp/seg_1.nii.gz",
                }
            ],
        },
    )

    def _run_mitole(body):
        calls["dataset_splits"] = dict(body.dataset_splits)
        calls["dataset_pairs"] = [dict(x) for x in body.dataset_pairs]
        return {"ok": True, "message": "ok", "downloader_log": "done"}

    monkeypatch.setattr(studio_api, "studio_run_mitole_downloader", _run_mitole)
    monkeypatch.setattr(studio_api, "studio_run_training", lambda *_args, **_kwargs: {"ok": True, "message": "ok"})
    monkeypatch.setattr(studio_api, "studio_downloader_preview", lambda *_args, **_kwargs: {"datasets": [], "dataset_rows": []})
    monkeypatch.setattr(studio_api, "studio_run_downloader", lambda *_args, **_kwargs: {"ok": True, "message": "ok"})
    monkeypatch.setattr(studio_api, "studio_run_database_build", lambda *_args, **_kwargs: {"ok": True, "message": "ok"})
    monkeypatch.setattr(studio_api, "studio_scrape_website", lambda *_args, **_kwargs: {"ok": True, "message": "ok"})

    body, meta = pipeline_chat.execute_pipeline_plan(
        Path(__file__).resolve().parents[3],
        "chat_test",
        PipelineSession(),
        {
            "sites": ["LocalHPC"],
            "stages": ["download"],
            "n_crops": 2,
            "download": {"target": "split", "total_crops": 2, "training_ratio": 0.5},
        },
    )

    assert meta["ok"] is True
    assert any(s.get("site") == "LocalHPC" and s.get("stage") == "download" for s in meta["steps"])
    assert "LocalHPC" in body
    assert calls["dataset_splits"]["hela_set"] == {"training": 1, "inference": 1}

