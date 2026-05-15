"""Pipeline Studio: run stage CLIs from the web UI (no approval queue)."""

from __future__ import annotations

import ast
import asyncio
import atexit
import csv
import json
import logging
import os
import queue
import signal
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Response
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from agent.orchestration.time_utils import US_EASTERN_TZ, now_us_eastern, now_us_eastern_iso, to_us_eastern
from starlette.responses import StreamingResponse

from agent.orchestration.session_pipeline import PipelineSession, PipelineStep, STEP_LABELS
from config.paths import (
    NNUNET_DATASET_NAME,
    data_outputs_bc,
    data_outputs_postprocessed,
    nnunet_dataset_root,
    nnunet_preprocessed_root,
    nnunet_results_root,
    rel_nnunet_labels_ts_instance,
    resolve_under_project,
)
from hpc_data_pipeline.mitole_pipeline import (
    MITOLE_DEFAULT_REL_FOLDERS,
    MITOLE_ROOT,
    abs_from_rel as mitole_abs_from_rel,
    clean_rel_folder as mitole_clean_rel_folder,
    load_selected_rel_folders as mitole_load_selected_rel_folders,
    list_all_subfolders as mitole_list_all_subfolders,
    save_selected_rel_folders as mitole_save_selected_rel_folders,
    scan_folder_rows as mitole_scan_folder_rows,
    stage3_copy_selected_pairs as mitole_stage3_copy_selected_pairs,
    mitole_pending_and_requested_crop_counts,
)

from .agent_turn import update_session_after_scrape
from .downloader_log_progress import DownloaderLogProgressParser
from .executor import run_command
from .llm import build_scrape_subprocess_llm_env

router = APIRouter(prefix="/api/studio", tags=["studio"])

# Populated by routes module on startup
_get_project_root: Any = None
_get_session: Any = None
_downloader_procs: dict[str, Any] = {}
_downloader_procs_lock = threading.Lock()
_downloader_state: dict[str, dict[str, Any]] = {}
_downloader_state_lock = threading.RLock()
_downloader_kill_requested: set[str] = set()
_downloader_kill_requested_lock = threading.Lock()
# script_path marker when studio_run_downloader runs with execute=True (sync mirror for UI)
_STUDIO_RUN_DOWNLOADER_SYNC_TAG = "[pipeline] studio_run_downloader --execute"


def _downloader_proc_is_alive(session_id: str) -> bool:
    """True when a live downloader subprocess is bound to this session."""
    sid = (session_id or "default").strip() or "default"
    with _downloader_procs_lock:
        proc = _downloader_procs.get(sid)
    if proc is None:
        return False
    try:
        return proc.poll() is None
    except Exception:
        return False


def _self_heal_stale_downloader_running_flag(session_id: str) -> None:
    """Clear stale running state when no live subprocess exists."""
    sid = (session_id or "default").strip() or "default"
    if _downloader_proc_is_alive(sid):
        return
    with _downloader_state_lock:
        st = _downloader_state.get(sid)
        if st and st.get("running"):
            st["running"] = False
            st["updated_at"] = time.time()
_preprocess_state: dict[str, dict[str, Any]] = {}
_preprocess_state_lock = threading.RLock()
# `subprocess.Popen` handles are per-process; see `_write_preprocess_selective_pid_file` so kill still
# works when another worker handles HTTP or the in-memory handle is missing.
_preprocess_procs: dict[str, Any] = {}
_preprocess_procs_lock = threading.Lock()
_preprocess_kill_requested: set[str] = set()
_preprocess_kill_requested_lock = threading.Lock()
_database_build_state: dict[str, dict[str, Any]] = {}
_database_build_state_lock = threading.Lock()
_slurm_run_state: dict[str, dict[str, dict[str, Any]]] = {}
_slurm_run_state_lock = threading.RLock()



def _studio_run_message(
    ok: bool,
    ok_message: str,
    fail_title: str,
    result: dict[str, Any],
    *,
    max_len: int = 12000,
) -> str:
    """User-facing ``message`` field: on failure append captured stdout/stderr from ``run_command``."""
    if ok:
        return ok_message
    err = (result.get("stderr") or "").strip()
    out = (result.get("stdout") or "").strip()
    if err and out:
        if out in err or err in out or len(err) >= len(out):
            tail = err
        else:
            tail = f"{err}\n\n--- stdout ---\n{out}"
    else:
        tail = err or out
    if not tail:
        tail = "(no subprocess output captured)"
    body = f"{fail_title}\n\n{tail}"
    if len(body) > max_len:
        return body[: max_len - 24] + "\n… (truncated)"
    return body


def _emit_run_failure_to_terminal(tag: str, result: dict[str, Any]) -> None:
    """Print failed subprocess output to the host terminal (stderr) for operators."""
    if int(result.get("returncode") or 0) == 0:
        return
    err = (result.get("stderr") or "").strip()
    out = (result.get("stdout") or "").strip()
    if err and out and out not in err and err not in out:
        blob = f"{err}\n{out}"
    else:
        blob = err or out
    if not blob:
        return
    print(f"\n[mitoFoundation2:{tag}] exit={result.get('returncode')}\n{blob}\n", file=sys.stderr, flush=True)


def _slurm_session_key(session_id: str) -> str:
    return (session_id or "default").strip() or "default"


def _slurm_kind_key(kind: str) -> str:
    return "inference" if kind == "inference" else "training"


def _get_or_init_slurm_run_state(session_id: str, kind: str) -> dict[str, Any]:
    sid = _slurm_session_key(session_id)
    kk = _slurm_kind_key(kind)
    with _slurm_run_state_lock:
        per_sid = _slurm_run_state.get(sid)
        if per_sid is None:
            per_sid = {}
            _slurm_run_state[sid] = per_sid
        row = per_sid.get(kk)
        if row is None:
            row = {
                "running": False,
                "job_id": "",
                "out_path": "",
                "err_path": "",
                "out_log": "",
                "err_log": "",
                "result": None,
                "updated_at": time.time(),
            }
            per_sid[kk] = row
        return row


def _read_slurm_log(path: str, *, max_chars: int = 200000) -> str:
    p = Path((path or "").strip())
    if not path or not p.is_file():
        return ""
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return txt[-max_chars:]


def _slurm_job_running(job_id: str) -> bool | None:
    jid = (job_id or "").strip()
    if not jid:
        return None
    if shutil.which("squeue") is None:
        return None
    try:
        cp = subprocess.run(
            ["squeue", "-h", "-j", jid, "-o", "%A"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    return any((ln or "").strip() == jid for ln in (cp.stdout or "").splitlines())


def _sync_slurm_run_logs(session_id: str, kind: str) -> dict[str, Any]:
    s = _get_or_init_slurm_run_state(session_id, kind)
    out_log = _read_slurm_log(str(s.get("out_path") or ""))
    err_log = _read_slurm_log(str(s.get("err_path") or ""))

    running = bool(s.get("running"))
    if running:
        sq = _slurm_job_running(str(s.get("job_id") or ""))
        finished_marker = "Total runtime:" in out_log
        if sq is False or finished_marker:
            running = False
    s["running"] = running
    s["out_log"] = out_log
    s["err_log"] = err_log
    s["updated_at"] = time.time()
    return s


def _slurm_log_root_from_path(path: str) -> str:
    p = Path((path or "").strip())
    name = p.name
    if name.endswith(".out"):
        return name[: -len(".out")]
    if name.endswith(".err"):
        return name[: -len(".err")]
    return ""


def _slurm_logs_dir(root: Path, kind: str) -> Path:
    sub = "infer" if kind == "inference" else "train"
    return (root / "5model_training" / "slurm" / "logs" / sub).resolve()


def _slurm_kind_matches_root(kind: str, log_root: str) -> bool:
    low = (log_root or "").strip().lower()
    if not low:
        return False
    if kind == "training":
        return low.startswith("train_")
    return low.startswith("predict_") or low.startswith("infer_")


def _list_slurm_log_roots(root: Path, kind: str) -> list[str]:
    logs_dir = _slurm_logs_dir(root, kind)
    roots_with_mtime: dict[str, float] = {}
    if logs_dir.is_dir():
        for ext in ("*.out", "*.err"):
            for p in logs_dir.glob(ext):
                if not p.is_file():
                    continue
                rr = _slurm_log_root_from_path(p.name)
                if not _slurm_kind_matches_root(kind, rr):
                    continue
                try:
                    mt = float(p.stat().st_mtime)
                except Exception:
                    mt = 0.0
                prev = roots_with_mtime.get(rr, 0.0)
                if mt >= prev:
                    roots_with_mtime[rr] = mt
    return [k for k, _ in sorted(roots_with_mtime.items(), key=lambda kv: (-kv[1], kv[0].lower()))]


def _training_out_has_terminal_runtime_marker(out_log: str) -> bool:
    lines = [ln.strip() for ln in (out_log or "").splitlines() if ln.strip()]
    if not lines:
        return False
    return bool(re.match(r"^Total runtime:\s+.+$", lines[-1]))


def _extract_training_log_summary(out_log: str) -> dict[str, Any] | None:
    lines = (out_log or "").splitlines()
    is_complete = _training_out_has_terminal_runtime_marker(out_log)

    def _last_float(pattern: str) -> float | None:
        ms = re.findall(pattern, out_log, flags=re.MULTILINE)
        if not ms:
            return None
        try:
            return float(ms[-1])
        except Exception:
            return None

    def _last_int(pattern: str) -> int | None:
        ms = re.findall(pattern, out_log, flags=re.MULTILINE)
        if not ms:
            return None
        try:
            return int(ms[-1])
        except Exception:
            return None

    runtime_line = ""
    ended_at_line = ""
    if lines:
        for ln in reversed(lines):
            s = ln.strip()
            if not runtime_line and s.startswith("Total runtime:"):
                runtime_line = s
                continue
            if not ended_at_line and s.startswith("Job ended at "):
                ended_at_line = s
            if runtime_line and ended_at_line:
                break

    mean_validation_dice = _last_float(r"Mean Validation Dice:\s*([-+0-9.eE]+)")
    best_ema = _last_float(r"Yayy!\s+New best EMA pseudo Dice:\s*([-+0-9.eE]+)")
    final_epoch = _last_int(r"Epoch\s+(\d+)\s*$")
    final_train_loss = _last_float(r"train_loss\s+([-+0-9.eE]+)")
    final_val_loss = _last_float(r"val_loss\s+([-+0-9.eE]+)")

    pseudo_last: dict[str, Any] = {}
    m_pd = list(re.finditer(r"Pseudo dice\s+\[(.+?)\]\s*$", out_log, flags=re.MULTILINE))
    if m_pd:
        raw = m_pd[-1].group(1)
        vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)]
        if vals:
            pseudo_last = {
                "values": vals,
                "mean": float(sum(vals) / len(vals)),
            }

    headline_bits: list[str] = []
    if mean_validation_dice is not None:
        headline_bits.append(f"Mean Validation Dice {mean_validation_dice:.4f}")
    if best_ema is not None:
        headline_bits.append(f"Best EMA pseudo Dice {best_ema:.4f}")
    if final_epoch is not None:
        headline_bits.append(f"Final epoch {final_epoch}")
    headline = " | ".join(headline_bits)

    return {
        "complete": is_complete,
        "headline": headline,
        "runtime": runtime_line,
        "ended_at": ended_at_line,
        "mean_validation_dice": mean_validation_dice,
        "best_ema_pseudo_dice": best_ema,
        "final_epoch": final_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_pseudo_dice": pseudo_last if pseudo_last else None,
    }


def _parse_failed_datasets_from_logs(stdout: str, stderr: str) -> list[str]:
    text = f"{stdout or ''}\n{stderr or ''}"
    m = re.search(r"Failed datasets:\s*(\[[^\]]*\])", text, re.IGNORECASE)
    if not m:
        return []
    try:
        obj = ast.literal_eval(m.group(1).strip())
        if isinstance(obj, list):
            return sorted({str(x).strip() for x in obj if str(x).strip()})
    except Exception:
        pass
    return []


def _downloader_quality_gate(result: dict[str, Any]) -> tuple[bool, str]:
    """Map stage-3 result to dataset-level truth (not only process returncode)."""
    rc_ok = int(result.get("returncode") or 0) == 0
    out = str(result.get("stdout") or "")
    err = str(result.get("stderr") or "")
    failed = _parse_failed_datasets_from_logs(out, err)
    marker_errors = bool(re.search(r"Integrated preprocess verification failed", out + "\n" + err, re.IGNORECASE))
    if (not rc_ok) or failed or marker_errors:
        if failed:
            return False, f"Failed datasets: {', '.join(failed)}."
        if marker_errors:
            return False, "Integrated preprocess marker verification failed for one or more datasets."
        return False, f"Downloader exited with code {int(result.get('returncode') or 1)}."
    return True, ""


def _scrape_result_message(payload: dict[str, Any]) -> str:
    """Normalize Stage-1 scrape message so callers always get a useful top-level string."""
    msg = str(payload.get("message") or "").strip()
    if msg:
        return msg
    fetch = payload.get("fetch")
    if isinstance(fetch, dict):
        ferr = str(fetch.get("error") or "").strip()
        if ferr:
            return ferr
    bridge = payload.get("mito_foundation_bridge")
    if isinstance(bridge, dict):
        berr = str(bridge.get("message") or bridge.get("error") or "").strip()
        if berr:
            return berr
    if payload.get("ok"):
        return "Scrape finished."
    return "Scrape failed with no explicit message."

# Isolate preprocess state/cancel from Starlette's default sync pool (``data/inspect`` and other
# routes can occupy every worker for a long time, starving polling + kill so the UI freezes).
_preprocess_studio_executor = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="mito2-pre",
)
# Kill/cancel must never sit behind a full queue of ``preprocess-selective-state`` polls (same pool
# would delay SIGKILL by many seconds or minutes on a busy tab).
_preprocess_kill_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="mito2-kill",
)
atexit.register(lambda: _preprocess_studio_executor.shutdown(wait=False))
atexit.register(lambda: _preprocess_kill_executor.shutdown(wait=False))


def configure_studio(*, get_project_root: Any, get_session: Any) -> None:
    global _get_project_root, _get_session
    _get_project_root = get_project_root
    _get_session = get_session


SCRAPE_PRESETS: list[dict[str, str]] = [
    {"id": "example", "label": "Example.org", "url": "http://example.org"},
    {"id": "openorganelle", "label": "OpenOrganelle", "url": "https://openorganelle.janelia.org/"},
    {"id": "allen_cell", "label": "Allen Cell Explorer", "url": "http://www.allencell.org/"},
    {"id": "custom", "label": "Custom URL", "url": ""},
]


def _root() -> Path:
    if _get_project_root is None:
        raise RuntimeError("studio_api not configured")
    return _get_project_root()


def _resolve_studio_training_config(root: Path) -> Path:
    """Same resolution as ``training_config_api._config_path`` (YAML editor on disk)."""
    env = (os.environ.get("MITO_TRAIN_CONFIG_YAML") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    config_dir = root / "5model_training" / "configuration"
    # Prefer the provider-agnostic name if it exists; fall back to the OpenOrganelle-specific
    # file that is present in the current repo layout.
    generic = config_dir / "mito_foundation.yaml"
    if generic.is_file():
        return generic.resolve()
    return (config_dir / "mito_openorganelle_foundation.yaml").resolve()


def _default_slurm_training_script(root: Path) -> Path:
    env = (os.environ.get("MITO2_SLURM_TRAIN_SCRIPT") or "").strip()
    if env:
        env_path = Path(env).expanduser().resolve()
        if env_path.is_file():
            return env_path
    return (root / "5model_training" / "slurm/scripts/train_nnunet_mito_foundation.sl").resolve()


def _default_slurm_inference_script(root: Path) -> Path:
    env = (os.environ.get("MITO2_SLURM_INFER_SCRIPT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (root / "5model_training" / "slurm/scripts/infer_nnunet_mito_foundation.sl").resolve()


def _resolve_project_path(root: Path, raw: str) -> Path:
    p = Path((raw or "").strip()).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def _slurm_batch_file_patterns(slurm_script: Path) -> tuple[str, str, str]:
    """Return ``(job_name, output_template, error_template)`` from ``#SBATCH`` lines."""
    job_name = "job"
    out_t = ""
    err_t = ""
    text = slurm_script.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("#SBATCH"):
            continue
        m = re.search(r"--job-name=([^\s#]+)", line)
        if m:
            job_name = m.group(1).strip()
        m = re.search(r"--output=(\S+)", line)
        if m:
            out_t = m.group(1).strip()
        m = re.search(r"--error=(\S+)", line)
        if m:
            err_t = m.group(1).strip()
    return job_name, out_t, err_t


def _expand_slurm_path_template(tmpl: str, job_name: str, job_id: str) -> str:
    return (
        tmpl.replace("%x", job_name)
        .replace("%j", job_id)
        .replace("%u", os.environ.get("USER", "user"))
    )


def _parse_slurm_submitted_job_id(stdout: str) -> str | None:
    m = re.search(r"Submitted batch job\s+(\d+)", stdout, re.IGNORECASE)
    return m.group(1) if m else None


def _database_build_timeout_sec() -> float:
    """Stage 2 may probe many S3 prefixes; default must exceed large-catalog wall time."""
    raw = (
        os.environ.get("MITO2_STUDIO_DATABASE_TIMEOUT_SEC")
        or os.environ.get("MITO2_STUDIO_SCHEMA_TIMEOUT_SEC")
        or ""
    ).strip()
    if raw:
        try:
            return max(60.0, float(raw))
        except ValueError:
            pass
    return 3600.0


def _downloader_timeout_sec() -> float | None:
    """Stage 3 full downloads can be very long-running.

    Default: no timeout (None). Set MITO2_STUDIO_DOWNLOADER_TIMEOUT_SEC to enforce a cap.
    """
    raw = (os.environ.get("MITO2_STUDIO_DOWNLOADER_TIMEOUT_SEC") or "").strip()
    if raw:
        try:
            return max(300.0, float(raw))
        except ValueError:
            pass
    return None


def _downloader_heartbeat_sec() -> float:
    """Optional progress heartbeat interval for long downloader runs.

    Default 0 disables synthetic heartbeat logs; rely on script logs only.
    """
    raw = (os.environ.get("MITO2_STUDIO_DOWNLOADER_HEARTBEAT_SEC") or "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 0.0


def _session(sid: str) -> PipelineSession:
    if _get_session is None:
        raise RuntimeError("studio_api not configured")
    return _get_session(sid)


def _pipeline_dict(session: PipelineSession) -> dict[str, Any]:
    return {
        "current_step": int(session.current_step),
        "step_label": STEP_LABELS.get(int(session.current_step), "idle"),
        "last_url": session.last_url,
        "last_site_stem": session.last_site_stem,
    }


def _list_probe_rel_paths(root: Path) -> list[str]:
    out = root / "1web_scraper_01" / "outputs"
    if not out.is_dir():
        return []
    paths = sorted(out.glob("*.probe.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.relative_to(root)).replace("\\", "/") for p in paths]


def _site_stems_from_probes(root: Path) -> list[str]:
    out = root / "1web_scraper_01" / "outputs"
    if not out.is_dir():
        return []
    stems: list[str] = []
    for p in sorted(out.glob("*.probe.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        name = p.name
        if name.endswith(".probe.json"):
            stems.append(name[: -len(".probe.json")])
    return stems


def _catalog_db_for_site(root: Path, site_stem: str) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in site_stem.strip())
    return root / "2database_builder" / "outputs" / "databases" / f"{safe}.db"


def _inspect_path_norm(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def _mito2_fallback_spacing_nm_zyx(path: Path) -> list[float]:
    """OpenOrganelle foundation crops are resampled to isotropic 16 nm (z,y,x); attrs are often absent."""
    s = _inspect_path_norm(path)
    if "openorganelle_mito_" in s:
        return [16.0, 16.0, 16.0]
    if "/data/nnunet_raw/dataset001_mito2/" in s and s.endswith(".nii.gz"):
        return [16.0, 16.0, 16.0]
    if "/data/raw/" in s and (s.endswith("_im.h5") or s.endswith("_seg.h5")):
        return [16.0, 16.0, 16.0]
    return []


def _h5_list_all_datasets(h5f: Any) -> list[tuple[str, Any]]:
    import h5py  # noqa: PLC0415

    out: list[tuple[str, Any]] = []

    def walk(grp: Any, prefix: str) -> None:
        for k in grp.keys():
            obj = grp[k]
            full = f"{prefix}/{k}" if prefix else k
            if isinstance(obj, h5py.Dataset):
                out.append((full, obj))
            elif isinstance(obj, h5py.Group):
                walk(obj, full)

    walk(h5f, "")
    return out


def _h5_ranked_3d_datasets(items: list[tuple[str, Any]]) -> list[tuple[int, str, Any]]:
    import numpy as np  # noqa: PLC0415

    ranked: list[tuple[int, str, Any]] = []
    for full, ds in items:
        if not hasattr(ds, "ndim") or int(ds.ndim) < 3:
            continue
        try:
            n = int(np.prod(np.array(ds.shape[:3], dtype=np.int64)))
        except Exception:
            continue
        ranked.append((n, full, ds))
    ranked.sort(key=lambda t: -t[0])
    return ranked


def _h5_pick_volume_dataset(
    name_l: str,
    ranked: list[tuple[int, str, Any]],
    *,
    path_for_hints: Path | None = None,
) -> Any | None:
    """Pick the main 3D dataset inside an HDF5 file.

    For EM crops we prefer internal paths containing ``_im``. For segmentations (``*_seg`` basename or any ``.h5``
    under ``…/labels/``) we prefer ``_seg`` internal paths and **must not** pick an ``_im`` dataset inside a label file
    (common cause of empty / bogus ``label_summary`` in the Studio table).
    """
    if not ranked:
        return None
    pl = str(path_for_hints).replace("\\", "/").lower() if path_for_hints is not None else ""
    under_labels = "/labels/" in pl
    treat_as_seg = ("_seg" in name_l) or under_labels
    if treat_as_seg:
        for _n, full, ds in ranked:
            if "_seg" in full.lower():
                return ds
        return ranked[0][2]
    for _n, full, ds in ranked:
        if "_im" in full.lower():
            return ds
    return ranked[0][2]


def _h5_seg_label_summary(ds: Any) -> str:
    try:
        import numpy as np  # noqa: PLC0415
    except Exception:
        return ""
    try:
        shape = tuple(int(ds.shape[i]) for i in range(min(3, int(ds.ndim))))
    except Exception:
        return ""
    if len(shape) != 3:
        return ""
    uniq_all: set[int] = set()
    # Fast bounded sampling: avoid full-volume scans on huge tensors.
    z_count = int(shape[0])
    n_samples = min(12, max(3, z_count))
    if z_count <= n_samples:
        z_indices = list(range(z_count))
    else:
        z_indices = sorted({int(round(i * (z_count - 1) / max(1, n_samples - 1))) for i in range(n_samples)})
    try:
        for z in z_indices:
            block = np.asarray(ds[z : z + 1, :, :])
            if np.issubdtype(block.dtype, np.bool_) or np.issubdtype(block.dtype, np.integer):
                u = np.unique(block)
            else:
                u = np.unique(np.rint(block).astype(np.int64, copy=False))
            uniq_all.update(int(x) for x in u.tolist())
    except Exception:
        return ""
    nz = {x for x in uniq_all if x != 0}
    if not nz:
        return "0"
    return str(len(nz))


def _inspect_path_is_label_h5(path: Path) -> bool:
    """Treat OpenOrganelle-style ``*_seg`` names or any ``.h5`` under a ``labels/`` directory as segmentation volumes."""
    name_l = path.name.lower()
    pl = str(path).replace("\\", "/").lower()
    if not name_l.endswith(".h5"):
        return False
    if "_seg" in name_l or "_mito" in name_l:
        return True
    return "/labels/" in pl or "/mito/" in pl


def _inspect_path_is_label_volume(path: Path) -> bool:
    """Heuristic for segmentation-like files across formats (nii/nrrd/h5)."""
    name_l = path.name.lower()
    pl = str(path).replace("\\", "/").lower()
    if any(tok in pl for tok in ("/labelstr/", "/labelsts/", "/labelstr-instance/", "/labelsts-instance/", "/labels/")):
        return True
    # Predicted masks in studio postprocessing dirs are segmentation volumes.
    if any(tok in pl for tok in ("/data/outputs/bc/", "/data/outputs/postprocessed/")):
        return True
    return ("_seg" in name_l) or ("_label" in name_l) or ("_mito" in name_l)


def _nii_seg_label_summary(img: Any) -> str:
    try:
        import numpy as np  # noqa: PLC0415
    except Exception:
        return ""
    try:
        shp = tuple(int(x) for x in list(img.shape[:3]))
    except Exception:
        return ""
    if len(shp) != 3:
        return ""
    uniq_all: set[int] = set()
    z_count = int(shp[0])
    n_samples = min(12, max(3, z_count))
    if z_count <= n_samples:
        z_indices = list(range(z_count))
    else:
        z_indices = sorted({int(round(i * (z_count - 1) / max(1, n_samples - 1))) for i in range(n_samples)})
    try:
        dataobj = img.dataobj
        for z in z_indices:
            block = np.asarray(dataobj[z : z + 1, :, :])
            if np.issubdtype(block.dtype, np.bool_) or np.issubdtype(block.dtype, np.integer):
                u = np.unique(block)
            else:
                u = np.unique(np.rint(block).astype(np.int64, copy=False))
            uniq_all.update(int(x) for x in u.tolist())
    except Exception:
        return ""
    nz = {x for x in uniq_all if x != 0}
    if not nz:
        return "0"
    return str(len(nz))


def _inspect_collapse_numpy_scalars(row: dict[str, Any]) -> None:
    """numpy.ndarray / numpy scalars break truthiness checks and sometimes JSON encoding."""
    try:
        import numpy as np  # noqa: PLC0415
    except Exception:
        return
    sp = row.get("spacing")
    if isinstance(sp, np.ndarray):
        row["spacing"] = [float(x) for x in sp.ravel().tolist()[:8]]
    elif isinstance(sp, (list, tuple)):
        flat: list[float] = []
        for x in sp[:8]:
            if isinstance(x, np.generic):
                flat.append(float(x.item()))
            else:
                try:
                    flat.append(float(x))
                except Exception:
                    pass
        row["spacing"] = flat
    dims = row.get("dimensions")
    if isinstance(dims, np.ndarray):
        row["dimensions"] = [int(x) for x in dims.ravel().tolist()[:8]]
    elif isinstance(dims, (list, tuple)):
        out_d: list[int] = []
        for x in dims[:8]:
            if isinstance(x, np.generic):
                out_d.append(int(x.item()))
            else:
                try:
                    out_d.append(int(x))
                except Exception:
                    pass
        row["dimensions"] = out_d


def _inspect_spacing_incomplete(row: dict[str, Any]) -> bool:
    sp = row.get("spacing")
    if not isinstance(sp, (list, tuple)):
        return True
    if len(sp) < 3:
        return True
    try:
        return not all(float(sp[i]) > 0.0 for i in range(3))
    except Exception:
        return True


def _inspect_coerce_row_json(row: dict[str, Any]) -> None:
    _inspect_collapse_numpy_scalars(row)
    dims = row.get("dimensions") or []
    out_d: list[int] = []
    for x in dims[:8] if isinstance(dims, (list, tuple)) else []:
        try:
            out_d.append(int(x))
        except Exception:
            pass
    row["dimensions"] = out_d
    sp = row.get("spacing") or []
    out_s: list[float] = []
    for x in sp[:3] if isinstance(sp, (list, tuple)) else []:
        try:
            out_s.append(float(x))
        except Exception:
            pass
    row["spacing"] = out_s
    if "label_summary" not in row or row["label_summary"] is None:
        row["label_summary"] = ""
    else:
        row["label_summary"] = str(row["label_summary"])


def _inspect_patch_row_gaps(path: Path, row: dict[str, Any]) -> None:
    """Fill spacing / label_summary when inspect missed (e.g. odd HDF5 layout); JSON-safe scalars."""
    _inspect_collapse_numpy_scalars(row)
    pl = _inspect_path_norm(path)
    if pl.endswith(".h5"):
        if _inspect_spacing_incomplete(row):
            fb = _mito2_fallback_spacing_nm_zyx(path)
            if fb:
                row["spacing"] = [float(x) for x in fb]
        if _inspect_path_is_label_volume(path) and not str(row.get("label_summary") or "").strip():
            try:
                import h5py  # noqa: PLC0415

                with h5py.File(path, "r") as hf:
                    items = _h5_list_all_datasets(hf)
                    ranked = _h5_ranked_3d_datasets(items)
                    ds2 = _h5_pick_volume_dataset(path.name.lower(), ranked, path_for_hints=path)
                    if ds2 is not None:
                        row["label_summary"] = _h5_seg_label_summary(ds2)
            except Exception:
                pass
    _inspect_coerce_row_json(row)
    # Final guard: coerce can drop bad values — re-apply known OpenOrganelle spacing once more.
    if pl.endswith(".h5") and _inspect_spacing_incomplete(row):
        fb = _mito2_fallback_spacing_nm_zyx(path)
        if fb:
            row["spacing"] = [float(x) for x in fb]
    if pl.endswith(".h5") and _inspect_path_is_label_volume(path) and not str(row.get("label_summary") or "").strip():
        try:
            import h5py  # noqa: PLC0415

            with h5py.File(path, "r") as hf:
                items = _h5_list_all_datasets(hf)
                ranked = _h5_ranked_3d_datasets(items)
                ds2 = _h5_pick_volume_dataset(path.name.lower(), ranked, path_for_hints=path)
                if ds2 is not None:
                    row["label_summary"] = _h5_seg_label_summary(ds2)
        except Exception:
            pass
    _inspect_coerce_row_json(row)


def _inspect_dataset_file(path: Path) -> dict[str, Any]:
    def _read_nrrd_header_fallback(p: Path) -> dict[str, Any]:
        """Parse NRRD text header without pynrrd dependency."""
        hdr: dict[str, Any] = {}
        try:
            with p.open("rb") as f:
                raw = f.read(65536)
        except Exception:
            return hdr
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            return hdr
        lines = text.splitlines()
        if not lines or not lines[0].startswith("NRRD"):
            return hdr
        for line in lines[1:]:
            s = line.strip()
            if not s:
                break
            if s.startswith("#") or ":" not in s:
                continue
            key, val = s.split(":", 1)
            k = key.strip().lower()
            v = val.strip()
            if k == "sizes":
                parts = [x for x in re.split(r"\s+", v) if x]
                out: list[int] = []
                for part in parts:
                    try:
                        out.append(int(part))
                    except Exception:
                        pass
                hdr["sizes"] = out
            elif k == "spacings":
                vals: list[float] = []
                for part in re.split(r"\s+", v):
                    p1 = part.strip()
                    if not p1:
                        continue
                    try:
                        vals.append(float(p1))
                    except Exception:
                        pass
                hdr["spacings"] = vals
            elif k == "space directions":
                vecs: list[list[float]] = []
                for m in re.finditer(r"\(([^)]*)\)", v):
                    nums: list[float] = []
                    for part in m.group(1).split(","):
                        p1 = part.strip()
                        if not p1:
                            continue
                        try:
                            nums.append(float(p1))
                        except Exception:
                            pass
                    if nums:
                        vecs.append(nums)
                if vecs:
                    hdr["space directions"] = vecs
        return hdr

    def _float_list(raw: Any) -> list[float]:
        try:
            import numpy as np  # noqa: PLC0415
        except Exception:
            np = None  # type: ignore[assignment]
        vals: list[float] = []
        if raw is None:
            return vals
        if np is not None and isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if isinstance(raw, (list, tuple)):
            for x in raw:
                if isinstance(x, (list, tuple)):
                    continue
                try:
                    vals.append(float(x))
                except Exception:
                    pass
            return [v for v in vals if v == v]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        if isinstance(raw, str):
            txt = raw.strip().strip("[]()")
            for part in txt.split(","):
                p = part.strip()
                if not p:
                    continue
                try:
                    vals.append(float(p))
                except Exception:
                    pass
            return [v for v in vals if v == v]
        try:
            return [float(raw)]
        except Exception:
            return []

    def _int_list(raw: Any) -> list[int]:
        out: list[int] = []
        for v in _float_list(raw):
            try:
                out.append(int(round(v)))
            except Exception:
                pass
        return out

    def _spacing_from_space_directions(raw: Any) -> list[float]:
        try:
            import numpy as np  # noqa: PLC0415
        except Exception:
            np = None  # type: ignore[assignment]
        if raw is None:
            return []
        if np is not None and isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if not isinstance(raw, (list, tuple)):
            return []
        spacing: list[float] = []
        for vec in raw:
            if np is not None and isinstance(vec, np.ndarray):
                vec = vec.tolist()
            if isinstance(vec, (list, tuple)) and vec:
                nums = _float_list(vec)
                if nums:
                    spacing.append(max(abs(x) for x in nums))
        return spacing

    def _vector3_positive_spacing(raw_any: Any) -> list[float]:
        vals = _float_list(raw_any)
        if len(vals) >= 3 and all(v > 0 for v in vals[:3]):
            return [float(vals[0]), float(vals[1]), float(vals[2])]
        if isinstance(raw_any, (bytes, bytearray)):
            try:
                raw_any = raw_any.decode("utf-8", errors="ignore")
            except Exception:
                return []
        if isinstance(raw_any, str):
            txt = raw_any.strip()
            for fn in (json.loads, ast.literal_eval):
                try:
                    parsed = fn(txt)
                except Exception:
                    continue
                vals = _float_list(parsed)
                if len(vals) >= 3 and all(v > 0 for v in vals[:3]):
                    return [float(vals[0]), float(vals[1]), float(vals[2])]
        return []

    def _spacing_from_h5_attr_dict(attrs: dict[str, Any]) -> list[float]:
        spacing_keys = (
            "spacing",
            "voxel_size",
            "voxel_size_nm",
            "resolution",
            "element_size_um",
            "pixel_size",
            "voxel_size_nm_zyx",
            "out_spacing_nm",
            "out_spacing_nm_zyx",
        )
        for k in spacing_keys:
            if k not in attrs:
                continue
            v = _vector3_positive_spacing(attrs[k])
            if v:
                return v
        for ak, av in attrs.items():
            lk = str(ak).lower()
            if any(s in lk for s in ("voxel_size", "voxel", "spacing_nm", "resolution", "element_size", "pixel_size")):
                v = _vector3_positive_spacing(av)
                if v:
                    return v
        return []

    def _peer_im_for_seg_by_layout(seg_path: Path) -> Path | None:
        """Find matching EM image for a segmentation volume using the standard layout.

        Expects ``data/raw/<run>/labels/<tag>_seg.h5`` with a sibling
        ``data/raw/<run>/images/<tag>_im.h5`` — a convention used by the OpenOrganelle
        downloader that any provider can follow.
        """
        s = str(seg_path).replace("\\", "/")
        low = s.lower()
        if "/labels/" not in low or not low.endswith("_seg.h5"):
            return None
        im_s = s.replace("/labels/", "/images/").replace("_seg.h5", "_im.h5").replace("_seg.H5", "_im.h5")
        cand = Path(im_s)
        return cand if cand.is_file() else None

    # Backward-compatible alias (will be removed in a future cleanup pass).
    _openorganelle_peer_im_for_seg = _peer_im_for_seg_by_layout

    def _read_h5_spacing_vector(p: Path) -> list[float]:
        try:
            import h5py  # noqa: PLC0415

            with h5py.File(p, "r") as hf:
                merged: dict[str, Any] = {}
                merged.update(dict(hf.attrs.items()))
                for kk in hf.keys():
                    obj = hf[kk]
                    if isinstance(obj, h5py.Dataset):
                        merged.update(dict(obj.attrs.items()))
                return _spacing_from_h5_attr_dict(merged)
        except Exception:
            return []

    name_l = path.name.lower()
    row: dict[str, Any] = {
        "name": path.name,
        "path": str(path),
        "type": "unknown",
        "dimensions": [],
        "spacing": [],
        "label_summary": "",
    }
    if name_l.endswith(".h5"):
        try:
            import h5py  # noqa: PLC0415

            with h5py.File(path, "r") as f:
                items = _h5_list_all_datasets(f)
                ranked = _h5_ranked_3d_datasets(items)
                ds = _h5_pick_volume_dataset(name_l, ranked, path_for_hints=path)
                if ds is None:
                    return row
                row["type"] = "h5"
                row["dimensions"] = [int(x) for x in ds.shape[:3]]
                attrs: dict[str, Any] = {}
                attrs.update(dict(f.attrs.items()))
                attrs.update(dict(ds.attrs.items()))
                for _full, dso in items:
                    if isinstance(dso, h5py.Dataset):
                        attrs.update(dict(dso.attrs.items()))
                row["spacing"] = _spacing_from_h5_attr_dict(attrs)
                if not row["spacing"]:
                    nrrd_peer = path.with_suffix(".nrrd")
                    if nrrd_peer.is_file():
                        try:
                            import nrrd  # noqa: PLC0415

                            hdr = nrrd.read_header(str(nrrd_peer))
                        except Exception:
                            hdr = _read_nrrd_header_fallback(nrrd_peer)
                        row["spacing"] = _float_list(hdr.get("spacings")) or _spacing_from_space_directions(
                            hdr.get("space directions")
                        )
                        if not row["dimensions"]:
                            alt = _int_list(hdr.get("sizes"))
                            if alt:
                                row["dimensions"] = alt[:3]
                if not row["spacing"]:
                    peer = _openorganelle_peer_im_for_seg(path)
                    if peer is not None:
                        row["spacing"] = _read_h5_spacing_vector(peer)
                if not row["spacing"]:
                    row["spacing"] = _mito2_fallback_spacing_nm_zyx(path)
                if _inspect_path_is_label_volume(path):
                    row["label_summary"] = _h5_seg_label_summary(ds)
        except Exception:
            pass
        return row
    if name_l.endswith(".nrrd"):
        try:
            import nrrd  # noqa: PLC0415

            hdr = nrrd.read_header(str(path))
        except Exception:
            hdr = _read_nrrd_header_fallback(path)
        row["type"] = "nrrd"
        row["dimensions"] = _int_list(hdr.get("sizes"))
        row["spacing"] = _float_list(hdr.get("spacings")) or _spacing_from_space_directions(
            hdr.get("space directions")
        )
        return row
    if name_l.endswith(".nii") or name_l.endswith(".nii.gz"):
        row["type"] = "nii"
        try:
            import nibabel as nib  # noqa: PLC0415

            img = nib.load(str(path))
            shp = list(img.shape[:3])
            row["dimensions"] = [int(x) for x in shp]
            zooms = [float(x) for x in img.header.get_zooms()[:3]]
            if len(zooms) == 3:
                # Keep z,y,x convention in UI
                row["spacing"] = [zooms[2], zooms[1], zooms[0]]
            if _inspect_path_is_label_volume(path):
                row["label_summary"] = _nii_seg_label_summary(img)
        except Exception:
            pass
        return row
    if name_l.endswith(".tif") or name_l.endswith(".tiff"):
        row["type"] = "tiff"
        try:
            import numpy as np  # noqa: PLC0415
            import tifffile  # noqa: PLC0415

            with tifffile.TiffFile(str(path)) as tf:
                arr = tf.asarray()
                shp = list(arr.shape)
                if len(shp) >= 3:
                    row["dimensions"] = [int(shp[-3]), int(shp[-2]), int(shp[-1])]
                elif len(shp) == 2:
                    row["dimensions"] = [1, int(shp[-2]), int(shp[-1])]
                elif len(shp) == 1:
                    row["dimensions"] = [1, 1, int(shp[-1])]
                page0 = tf.pages[0] if tf.pages else None
                if page0 is not None:
                    tags = page0.tags
                    xres_tag = tags.get("XResolution")
                    yres_tag = tags.get("YResolution")
                    unit_tag = tags.get("ResolutionUnit")

                    def _to_float_res(v: Any) -> float | None:
                        try:
                            if isinstance(v, (tuple, list)) and len(v) == 2:
                                den = float(v[1])
                                if den == 0:
                                    return None
                                return float(v[0]) / den
                            return float(v)
                        except Exception:
                            return None

                    xres = _to_float_res(getattr(xres_tag, "value", None)) if xres_tag is not None else None
                    yres = _to_float_res(getattr(yres_tag, "value", None)) if yres_tag is not None else None
                    unit_val = int(getattr(unit_tag, "value", 1)) if unit_tag is not None else 1
                    unit_nm = 0.0
                    if unit_val == 2:  # inch
                        unit_nm = 25_400_000.0
                    elif unit_val == 3:  # cm
                        unit_nm = 10_000_000.0
                    if unit_nm > 0 and xres and yres and xres > 0 and yres > 0:
                        row["spacing"] = [1.0, unit_nm / yres, unit_nm / xres]
                if _inspect_path_is_label_volume(path):
                    uniq_all: set[int] = set()
                    if arr.ndim == 2:
                        work = arr[np.newaxis, ...]
                    elif arr.ndim >= 3:
                        work = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
                    else:
                        work = arr.reshape((1, 1, -1))
                    z_count = int(work.shape[0])
                    n_samples = min(12, max(3, z_count))
                    if z_count <= n_samples:
                        z_indices = list(range(z_count))
                    else:
                        z_indices = sorted({int(round(i * (z_count - 1) / max(1, n_samples - 1))) for i in range(n_samples)})
                    for z in z_indices:
                        block = np.asarray(work[z : z + 1, :, :])
                        if np.issubdtype(block.dtype, np.bool_) or np.issubdtype(block.dtype, np.integer):
                            u = np.unique(block)
                        else:
                            u = np.unique(np.rint(block).astype(np.int64, copy=False))
                        uniq_all.update(int(x) for x in u.tolist())
                    nz = {x for x in uniq_all if x != 0}
                    row["label_summary"] = "0" if not nz else str(len(nz))
        except Exception:
            pass
        return row
    if name_l.endswith(".pt"):
        row["type"] = "pt"
        return row
    if name_l.endswith(".json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            row["type"] = "json"
            if isinstance(data, list):
                row["dimensions"] = [len(data)]
                if data and isinstance(data[0], dict) and isinstance(data[0].get("spacing"), list):
                    row["spacing"] = [float(x) for x in data[0]["spacing"] if isinstance(x, (int, float))]
            elif isinstance(data, dict):
                for key in ("training", "validation", "test"):
                    if isinstance(data.get(key), list):
                        row["dimensions"].append(len(data[key]))
        except Exception:
            pass
        return row
    return row


def _inspect_dataset_file_shallow(path: Path) -> dict[str, Any]:
    """Path + extension only (no HDF5 / volume reads) for a responsive Studio file tree."""
    name_l = path.name.lower()
    row: dict[str, Any] = {
        "name": path.name,
        "path": str(path),
        "type": "unknown",
        "dimensions": [],
        "spacing": [],
        "label_summary": "",
    }
    if name_l.endswith(".h5"):
        row["type"] = "h5"
    elif name_l.endswith(".nrrd"):
        row["type"] = "nrrd"
    elif name_l.endswith(".nii.gz"):
        row["type"] = "nii"
    elif name_l.endswith(".nii"):
        row["type"] = "nii"
    elif name_l.endswith(".pt"):
        row["type"] = "pt"
    elif name_l.endswith(".json"):
        row["type"] = "json"
    _inspect_coerce_row_json(row)
    return row


def _studio_norm_inspect_deep_under(raw_base: Path, deep_under: str | None) -> str | None:
    """Relative path under ``data/raw``; only used with shallow inspect to open HDF5 for one folder."""
    if deep_under is None:
        return None
    s = str(deep_under).strip().replace("\\", "/").strip("/")
    if not s:
        return None
    if any(part == ".." for part in s.split("/")):
        return None
    try:
        cand = (raw_base / s).resolve()
        raw_res = raw_base.resolve()
        if raw_base.is_dir() and not str(cand).startswith(str(raw_res) + os.sep) and cand != raw_res:
            return None
    except Exception:
        return None
    return s


def _inspect_raw_path_is_under_deep_prefix(rel_posix: str, deep_prefix: str) -> bool:
    return rel_posix == deep_prefix or rel_posix.startswith(f"{deep_prefix}/")


def _download_run_images_dir(run_dir: Path) -> Path | None:
    """OpenOrganelle-style layout: EM stacks live under ``images`` or ``Images``."""
    for name in ("images", "Images"):
        d = run_dir / name
        if d.is_dir():
            return d
    return None


def _count_em_h5_for_preprocess(img_dir: Path) -> int:
    """Count EM stacks for selective preprocess discovery."""
    if not img_dir.is_dir():
        return 0
    count = 0
    for p in img_dir.iterdir():
        name_l = p.name.lower()
        if p.is_file() and (name_l.endswith("_im.h5") or name_l.endswith("_0000.nii.gz")):
            count += 1
    return count


def _preprocess_em_stack_dir(run_dir: Path) -> Path | None:
    """Stacks for Stage 4: nested legacy ``…/images/`` or Dataset001_mito2 split dirs."""
    sub = _download_run_images_dir(run_dir)
    if sub is not None:
        return sub
    if run_dir.is_dir() and _count_em_h5_for_preprocess(run_dir) > 0:
        return run_dir
    return None


def _is_training_alias_duplicate(path: Path) -> bool:
    """Hide canonical aliases when raw downloader twins are present.

    Example: ``foo.h5`` is treated as an alias of ``foo_im.h5`` (images) or
    ``foo_seg.h5`` (labels) when those raw files exist in the same folder.
    """
    name = path.name
    low = name.lower()
    if not low.endswith(".h5"):
        return False
    if low.endswith("_im.h5") or low.endswith("_seg.h5"):
        return False
    base = path.with_suffix("")
    try:
        return (path.parent / f"{base.name}_im.h5").is_file() or (path.parent / f"{base.name}_seg.h5").is_file()
    except Exception:
        return False


@router.get("/data/inspect")
def studio_data_inspect(
    shallow: bool = Query(
        False,
        description="If true, list files without opening HDF5/NIfTI (fast). Studio UI sends shallow=1; omit for full metadata.",
    ),
    deep_under: str | None = Query(
        None,
        description=(
            "When shallow=1, still run full HDF5/NIfTI inspect for files whose path under data/raw starts with "
            "this relative prefix (e.g. ``my_run/labels``). Keeps the file list fast while filling dimensions / "
            "label counts for the Stage-4 raw viewer folder."
        ),
    ),
) -> dict[str, Any]:
    root = _root()
    raw_base = root / "data" / "raw"
    dataset_base = nnunet_dataset_root(root)
    training_base = dataset_base / "imagesTr"
    inference_base = dataset_base / "imagesTs"
    training_labels_base = dataset_base / "labelsTr"
    training_labels_instance_base = dataset_base / "labelsTr-instance"
    inference_labels_base = dataset_base / "labelsTs"
    inference_labels_instance_base = dataset_base / "labelsTs-instance"
    deep_under_q = str(deep_under).strip() if deep_under is not None else ""
    deep_prefix = _studio_norm_inspect_deep_under(raw_base, deep_under)
    if deep_under_q and deep_prefix is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid deep_under (use a path relative to data/raw, no '..').",
        )

    raw_rows: list[dict[str, Any]] = []
    if raw_base.is_dir():
        for p in sorted(raw_base.glob("**/*")):
            if not p.is_file():
                continue
            if p.name.lower().endswith((".h5", ".nrrd", ".nii", ".nii.gz", ".pt", ".json")):
                rel_posix = p.relative_to(raw_base).as_posix()
                use_full = (not shallow) or (
                    shallow and deep_prefix is not None and _inspect_raw_path_is_under_deep_prefix(rel_posix, deep_prefix)
                )
                if use_full:
                    row = _inspect_dataset_file(p)
                    _inspect_patch_row_gaps(p, row)
                    raw_rows.append(row)
                else:
                    raw_rows.append(_inspect_dataset_file_shallow(p))

    training_rows: list[dict[str, Any]] = []
    for scan_base in (training_base, training_labels_base):
        if not scan_base.is_dir():
            continue
        for p in sorted(scan_base.glob("**/*")):
            if not p.is_file():
                continue
            if p.name.lower().endswith((".nrrd", ".nii", ".nii.gz", ".pt", ".json")):
                if _is_training_alias_duplicate(p):
                    continue
                if shallow:
                    training_rows.append(_inspect_dataset_file_shallow(p))
                else:
                    row = _inspect_dataset_file(p)
                    _inspect_patch_row_gaps(p, row)
                    training_rows.append(row)

    inference_rows: list[dict[str, Any]] = []
    for scan_base in (inference_base, inference_labels_base):
        if not scan_base.is_dir():
            continue
        for p in sorted(scan_base.glob("**/*")):
            if not p.is_file():
                continue
            if p.name.lower().endswith((".nrrd", ".nii", ".nii.gz", ".pt", ".json")):
                if shallow:
                    inference_rows.append(_inspect_dataset_file_shallow(p))
                else:
                    row = _inspect_dataset_file(p)
                    _inspect_patch_row_gaps(p, row)
                    inference_rows.append(row)

    instance_rows: list[dict[str, Any]] = []
    for scan_base in (training_labels_instance_base, inference_labels_instance_base):
        if not scan_base.is_dir():
            continue
        for p in sorted(scan_base.glob("**/*")):
            if not p.is_file():
                continue
            if p.name.lower().endswith((".nrrd", ".nii", ".nii.gz", ".pt", ".json")):
                if shallow:
                    instance_rows.append(_inspect_dataset_file_shallow(p))
                else:
                    row = _inspect_dataset_file(p)
                    _inspect_patch_row_gaps(p, row)
                    instance_rows.append(row)

    return {
        "raw_base": str(raw_base),
        "training_base": str(training_base),
        "inference_base": str(inference_base),
        # Backward-compat aliases used by existing UI code.
        "preprocessed_base": str(dataset_base),
        "raw_datasets": raw_rows,
        "training_datasets": training_rows,
        "inference_datasets": inference_rows,
        "instance_datasets": instance_rows,
        "preprocessed_datasets": training_rows,
        "inspect_shallow": shallow,
        "inspect_deep_under": deep_prefix,
    }


@router.get("/data/raw-em-stacks")
def studio_raw_em_stacks(
    run: str = Query(..., min_length=1, description="Top-level folder name under data/raw"),
) -> dict[str, Any]:
    """Filesystem check for EM stacks used by stage-4 whole-run preprocess (authoritative vs data/inspect index)."""
    root = _root()
    raw_base = (root / "data" / "raw").resolve()
    run_name = str(run).strip()
    if not run_name or ".." in run_name:
        raise HTTPException(status_code=400, detail="Invalid run name.")
    if "/" in run_name.replace("\\", "/"):
        raise HTTPException(status_code=400, detail="Run name must be a single path segment.")
    run_dir = (raw_base / run_name).resolve()
    try:
        run_dir.relative_to(raw_base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Run path escapes data/raw.")
    if not run_dir.is_dir():
        return {
            "ok": False,
            "run": run_name,
            "count": 0,
            "reason": "missing_run_dir",
            "detail": f"Folder not found: {run_dir}",
        }
    img_dir = _download_run_images_dir(run_dir)
    if img_dir is None:
        return {
            "ok": False,
            "run": run_name,
            "count": 0,
            "reason": "no_images_dir",
            "detail": f"No images/ or Images/ under data/raw/{run_name}.",
        }
    n = _count_em_h5_for_preprocess(img_dir)
    sample_other: list[str] = []
    if n == 0:
        for p in sorted(img_dir.iterdir(), key=lambda x: str(x).lower()):
            if p.is_file() and len(sample_other) < 16:
                sample_other.append(p.name)
    return {
        "ok": True,
        "run": run_name,
        "count": n,
        "images_dir": str(img_dir),
        "sample_other_files": sample_other,
        "detail": (
            ""
            if n > 0
            else (
                f"No EM stack files found in {img_dir}. "
                f"Expected nnUNet-style …/imagesTr/<tag>_0000.nii.gz (or legacy *_im.h5). "
                f"Found other files (sample): {', '.join(sample_other)}" if sample_other else f"No files in {img_dir}."
            )
        ),
    }


@router.get("/postprocessing/files")
def studio_postprocessing_files() -> dict[str, Any]:
    """List files from fixed postprocessing input/output directories for table display."""
    root = _root()
    input_dir = data_outputs_bc(root)
    output_dir = data_outputs_postprocessed(root)

    def _scan(base: Path, source: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not base.is_dir():
            return rows
        files = sorted(
            [p for p in base.iterdir() if p.is_file() and p.name.lower().endswith((".nii.gz", ".nii"))],
            key=lambda p: p.name.lower(),
        )
        for p in files:
            try:
                row = _inspect_dataset_file(p)
                row["source"] = source
                _inspect_coerce_row_json(row)
                rows.append(row)
            except Exception:
                rows.append(
                    {
                        "name": p.name,
                        "path": str(p),
                        "type": "",
                        "dimensions": [],
                        "spacing": [],
                        "label_summary": "",
                        "source": source,
                    }
                )
        return rows

    return {
        "ok": True,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": _scan(input_dir, "input") + _scan(output_dir, "output"),
    }


def _downloader_preview_from_db(db_path: Path, *, data_scope: str) -> dict[str, Any]:
    if not db_path.is_file():
        return {"ok": False, "message": f"Schema DB not found: {db_path}", "datasets": [], "dataset_rows": []}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        # Match stage-3 labeled semantics: datasets with good non-prediction mito masks.
        cur.execute(
            """
            SELECT dataset_name
            FROM datasets
            WHERE COALESCE(TRIM(dataset_name), '') <> ''
            ORDER BY dataset_name COLLATE NOCASE
            """
        )
        _ = cur.fetchall()
        cur.execute(
            """
            SELECT dataset_name, sample_type
            FROM datasets
            WHERE COALESCE(TRIM(dataset_name), '') <> ''
              AND LOWER(COALESCE(TRIM(download_mito_mask_quality), '')) = 'good'
              AND COALESCE(TRIM(download_mito_mask_url), '') <> ''
              AND LOWER(COALESCE(TRIM(download_mito_mask_url), '')) NOT LIKE '%pred%'
              AND LOWER(COALESCE(TRIM(download_mito_mask_url), '')) NOT LIKE '%inference%'
            ORDER BY dataset_name COLLATE NOCASE
            """
        )
        rows = cur.fetchall()
        labeled_names = {(r["dataset_name"] or "").strip() for r in rows if (r["dataset_name"] or "").strip()}
        names = sorted(labeled_names, key=str.lower)
        dataset_rows = [
            {
                "dataset_name": (r["dataset_name"] or "").strip(),
                "sample_type": ((r["sample_type"] or "").strip() or "unknown"),
            }
            for r in rows
            if (r["dataset_name"] or "").strip()
        ]
        return {"ok": True, "message": "ok", "datasets": names, "dataset_rows": dataset_rows}
    finally:
        conn.close()


def _downloader_sample_rows_for_names(db_path: Path, names: list[str]) -> list[dict[str, str]]:
    """Resolve sample_type for an explicit dataset name list from one catalog DB."""
    if not db_path.is_file() or not names:
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(datasets)")
        dcols = {str(r[1]) for r in cur.fetchall()}
        sample_col = "sample_type" if "sample_type" in dcols else None
        if sample_col is None:
            return [{"dataset_name": n, "sample_type": "unknown"} for n in names]

        out: list[dict[str, str]] = []
        for n in names:
            row = cur.execute(
                f"SELECT {sample_col} AS sample_type FROM datasets WHERE dataset_name = ? LIMIT 1",
                (n,),
            ).fetchone()
            st = str((row["sample_type"] if row else "") or "").strip() or "unknown"
            out.append({"dataset_name": n, "sample_type": st})
        return out
    finally:
        conn.close()


def _dataset_has_complete_labeled_outputs(root: Path, dataset_name: str, n_crops: int) -> bool:
    """True when all expected nnUNet image/label pairs exist for this dataset.

    Checks both training (imagesTr/labelsTr) and inference (imagesTs/labelsTs) trees,
    matching Stage-3 script naming ``<stem>_vol{i}`` where *stem* matches downloader
    sanitization (slashes → underscores, etc.).
    """
    dataset = (dataset_name or "").strip()
    if not dataset:
        return False
    stem = dataset.replace("/", "_").replace("-", "_")
    expected = max(1, int(n_crops))
    base = nnunet_dataset_root(root)
    candidates = [
        (base / "imagesTr", base / "labelsTr"),
        (base / "imagesTs", base / "labelsTs"),
    ]
    for img_dir, lbl_dir in candidates:
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        ok = True
        for i in range(1, expected + 1):
            tag = f"{stem}_vol{i}"
            img = img_dir / f"{tag}_0000.nii.gz"
            lbl = lbl_dir / f"{tag}.nii.gz"
            if not img.is_file() or not lbl.is_file():
                ok = False
                break
        if ok:
            return True
    return False


class StudioSessionBody(BaseModel):
    session_id: str = "default"


class MitoLeFoldersBody(BaseModel):
    folders: list[str] = Field(default_factory=list)


@router.get("/mitole/config")
def studio_mitole_config() -> dict[str, Any]:
    root = _root()
    folders = mitole_load_selected_rel_folders(root)
    return {
        "ok": True,
        "base_path": str(MITOLE_ROOT),
        "default_folders": list(MITOLE_DEFAULT_REL_FOLDERS),
        "folders": folders,
    }


@router.get("/mitole/subfolders")
def studio_mitole_subfolders() -> dict[str, Any]:
    return {
        "ok": True,
        "base_path": str(MITOLE_ROOT),
        "subfolders": mitole_list_all_subfolders(),
    }


@router.post("/mitole/config")
def studio_mitole_config_save(body: MitoLeFoldersBody) -> dict[str, Any]:
    root = _root()
    clean: list[str] = []
    for raw in body.folders:
        rel = mitole_clean_rel_folder(raw)
        if rel and rel not in clean:
            clean.append(rel)
    mitole_save_selected_rel_folders(root, clean)
    return {
        "ok": True,
        "base_path": str(MITOLE_ROOT),
        "folders": mitole_load_selected_rel_folders(root),
    }


@router.get("/mitole/inspect")
def studio_mitole_inspect(folder: str = Query("__all__")) -> dict[str, Any]:
    root = _root()
    selected = mitole_load_selected_rel_folders(root)
    target = str(folder or "").strip()
    if target and target != "__all__":
        clean = mitole_clean_rel_folder(target)
        if not clean:
            raise HTTPException(status_code=400, detail="Invalid folder.")
        selected = [clean]
    rows: list[dict[str, Any]] = []
    for rel in selected:
        try:
            rows.extend(
                mitole_scan_folder_rows(
                    mitole_abs_from_rel(rel),
                    rel,
                    inspect_dataset_file=_inspect_dataset_file,
                    inspect_dataset_file_shallow=_inspect_dataset_file_shallow,
                    inspect_coerce_row_json=_inspect_coerce_row_json,
                )
            )
        except Exception:
            continue
    rows = sorted(rows, key=lambda r: (str(r.get("folder") or ""), str(r.get("path") or "").lower()))
    return {
        "ok": True,
        "base_path": str(MITOLE_ROOT),
        "folders": selected,
        "rows": rows,
    }


@router.get("/mitole/catalogue")
def studio_mitole_catalogue(regenerate: bool = Query(default=False)) -> dict[str, Any]:
    def _write_stage2_outputs(project_root: Path, out_rows: list[dict[str, Any]], out_filters: dict[str, Any]) -> None:
        out_dir = (project_root / "hpc_data_pipeline" / "stage2" / "outputs").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        generated_at = now_us_eastern_iso()

        payload = {
            "ok": True,
            "generated_at": generated_at,
            "rows": out_rows,
            "filters": out_filters,
        }
        (out_dir / "mitole_catalogue.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        csv_path = out_dir / "mitole_catalogue.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "folder",
                    "source",
                    "organism",
                    "sample_type",
                    "image_file",
                    "label_file",
                    "image_path",
                    "label_path",
                    "dimensions",
                    "spacing",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(out_rows)

        db_path = out_dir / "mitole_catalogue.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DROP TABLE IF EXISTS catalogue")
            conn.execute(
                """
                CREATE TABLE catalogue (
                    dataset TEXT,
                    folder TEXT,
                    source TEXT,
                    organism TEXT,
                    sample_type TEXT,
                    image_file TEXT,
                    label_file TEXT,
                    image_path TEXT,
                    label_path TEXT,
                    dimensions TEXT,
                    spacing TEXT
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO catalogue
                  (dataset, folder, source, organism, sample_type, image_file, label_file, image_path, label_path, dimensions, spacing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(r.get("dataset") or ""),
                        str(r.get("folder") or ""),
                        str(r.get("source") or ""),
                        str(r.get("organism") or ""),
                        str(r.get("sample_type") or ""),
                        str(r.get("image_file") or ""),
                        str(r.get("label_file") or ""),
                        str(r.get("image_path") or ""),
                        str(r.get("label_path") or ""),
                        json.dumps(r.get("dimensions") or []),
                        json.dumps(r.get("spacing") or []),
                    )
                    for r in out_rows
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def _norm_stem(name: str) -> str:
        s = str(name or "").lower()
        s = re.sub(r"\.nii\.gz$", "", s)
        s = re.sub(r"\.(h5|nii|tif|tiff)$", "", s)
        s = re.sub(r"([._-])(im|image|img)$", "", s)
        s = re.sub(r"([._-])(mito|seg|label|labels|mask|gt|pred)$", "", s)
        return s

    def _split_tokens(s: str) -> list[str]:
        return [t for t in re.split(r"[^a-z0-9]+", str(s or "").lower()) if t]

    def _infer_organism_sample(dataset: str, file_names: list[str]) -> tuple[str, str]:
        token_stream: list[str] = []
        dataset_tokens = _split_tokens(dataset)
        for nm in file_names:
            token_stream.extend(_split_tokens(_norm_stem(nm)))
        if not token_stream:
            token_stream = _split_tokens(dataset)
        if not token_stream:
            return ("unknown", "unknown")

        org_alias = {
            "drosophila": "fly",
            "fly": "fly",
            "mouse": "mouse",
            "mice": "mouse",
            "rat": "rat",
            "human": "human",
            "hela": "human",
            "jurkat": "human",
            "macrophage": "human",
            "elegans": "c_elegans",
            "celegans": "c_elegans",
            "worm": "c_elegans",
            "zebrafish": "zebrafish",
            "yeast": "yeast",
        }
        sample_vocab = {
            "brain",
            "muscle",
            "heart",
            "cardiac",
            "kidney",
            "liver",
            "cell",
            "cells",
            "gland",
            "neuron",
            "astrocyte",
            "tissue",
            "fib",
        }

        organism = "unknown"
        for tok in token_stream:
            if tok in org_alias:
                organism = org_alias[tok]
                break
        if organism == "unknown":
            for tok in token_stream:
                if tok not in sample_vocab and len(tok) > 2:
                    organism = tok
                    break
        if (
            not organism
            or organism == "unknown"
            or organism.isdigit()
            or (any(ch.isdigit() for ch in organism) and len(organism) >= 8)
        ):
            organism = dataset_tokens[0] if dataset_tokens else "unknown"

        sample_type = "unknown"
        for tok in token_stream:
            if tok in sample_vocab:
                sample_type = "cardiac" if tok == "heart" else tok
                break
        if sample_type == "unknown" and len(token_stream) >= 2:
            sample_type = token_stream[1]
        elif sample_type == "unknown" and token_stream:
            sample_type = token_stream[0]
        if (not sample_type) or sample_type.isdigit():
            sample_type = dataset_tokens[1] if len(dataset_tokens) > 1 else (dataset_tokens[0] if dataset_tokens else "unknown")

        return (organism, sample_type)

    def _read_stage2_outputs(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        out_dir = (project_root / "hpc_data_pipeline" / "stage2" / "outputs").resolve()
        json_path = out_dir / "mitole_catalogue.json"
        if json_path.is_file():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                rows = payload.get("rows", []) if isinstance(payload, dict) else []
                filters = payload.get("filters", {}) if isinstance(payload, dict) else {}
                if isinstance(rows, list) and isinstance(filters, dict):
                    return rows, filters
            except Exception:
                pass
        return None

    root = _root()
    if not regenerate:
        loaded = _read_stage2_outputs(root)
        if loaded is not None:
            rows, filters = loaded
            return {
                "ok": True,
                "base_path": str(MITOLE_ROOT),
                "rows": rows,
                "filters": filters,
            }
        return {
            "ok": True,
            "base_path": str(MITOLE_ROOT),
            "rows": [],
            "filters": {
                "folders": [],
                "sources": [],
                "organisms": [],
                "sample_types": [],
            },
        }

    def _is_label_row(r: dict[str, Any]) -> bool:
        n = str(r.get("name") or "").lower()
        p = str(r.get("path") or "").lower()
        return bool(re.search(r"(label|labels|mask|_seg|segmentation|_gt|_mito)", n) or re.search(r"(\/labels?\/|\/masks?\/|\/mito\/)", p))

    folders = mitole_load_selected_rel_folders(root)
    rows: list[dict[str, Any]] = []
    for rel in folders:
        try:
            abs_folder = mitole_abs_from_rel(rel)
        except Exception:
            continue
        files = mitole_scan_folder_rows(
            abs_folder,
            rel,
            inspect_dataset_file=_inspect_dataset_file,
            inspect_dataset_file_shallow=_inspect_dataset_file_shallow,
            inspect_coerce_row_json=_inspect_coerce_row_json,
        )
        images = [r for r in files if not _is_label_row(r)]
        labels = [r for r in files if _is_label_row(r)]
        label_by_key: dict[str, list[dict[str, Any]]] = {}
        for lb in labels:
            k = _norm_stem(str(lb.get("name") or ""))
            label_by_key.setdefault(k, []).append(lb)
        for im in images:
            k = _norm_stem(str(im.get("name") or ""))
            lbs = label_by_key.get(k, [])
            if not lbs:
                continue
            lb = lbs.pop(0)
            label_by_key[k] = lbs
            dataset = k or rel.split("/")[-1]
            organism, sample_type = _infer_organism_sample(
                dataset,
                [str(im.get("name") or ""), str(lb.get("name") or "")],
            )
            dimensions = im.get("dimensions") or lb.get("dimensions") or []
            spacing = im.get("spacing") or lb.get("spacing") or []
            rows.append(
                {
                    "dataset": dataset,
                    "folder": rel,
                    "source": rel,
                    "organism": organism,
                    "sample_type": sample_type,
                    "image_file": str(im.get("name") or ""),
                    "label_file": str(lb.get("name") or ""),
                    "image_path": str(im.get("path") or ""),
                    "label_path": str(lb.get("path") or ""),
                    "dimensions": dimensions if isinstance(dimensions, list) else [],
                    "spacing": spacing if isinstance(spacing, list) else [],
                }
            )
    filters = {
        "folders": folders,
        "sources": sorted({str(r["source"]) for r in rows}),
        "organisms": sorted({str(r["organism"]) for r in rows}),
        "sample_types": sorted({str(r["sample_type"]) for r in rows}),
    }
    try:
        _write_stage2_outputs(root, rows, filters)
    except Exception:
        # Output persistence should not break Stage-2 browsing API.
        pass

    return {
        "ok": True,
        "base_path": str(MITOLE_ROOT),
        "rows": rows,
        "filters": filters,
    }


@router.post("/model/reset-downloaded-data-history")
def studio_model_reset_downloaded_data_history(body: StudioSessionBody) -> dict[str, Any]:
    """Clear model outputs and slurm log history used by Studio model pages."""
    root = _root()

    # Do not clear history while training/inference jobs are still active.
    active_runs: list[dict[str, str]] = []
    with _slurm_run_state_lock:
        snap = {
            sid: {
                kind: dict(state)
                for kind, state in (per_sid or {}).items()
            }
            for sid, per_sid in _slurm_run_state.items()
        }
    for sid, per_sid in snap.items():
        for kind in ("training", "inference"):
            st = per_sid.get(kind) or {}
            if not bool(st.get("running")):
                continue
            jid = str(st.get("job_id") or "").strip()
            if jid and _slurm_job_running(jid):
                active_runs.append({"session_id": sid, "kind": kind, "job_id": jid})
    if active_runs:
        raise HTTPException(status_code=409, detail=f"Cannot reset while runs are active: {active_runs}")

    prune_children_targets = [
        (root / "data" / "outputs" / "bc").resolve(),
        (root / "data" / "outputs" / "postprocessed").resolve(),
        (root / "5model_training" / "slurm" / "logs" / "infer").resolve(),
        (root / "5model_training" / "slurm" / "logs" / "train").resolve(),
    ]
    raw_dataset_root = nnunet_dataset_root(root)
    raw_dataset_subdirs = [
        (raw_dataset_root / "imagesTr").resolve(),
        (raw_dataset_root / "imagesTs").resolve(),
        (raw_dataset_root / "labelsTr").resolve(),
        (raw_dataset_root / "labelsTr-instance").resolve(),
        (raw_dataset_root / "labelsTs").resolve(),
        (raw_dataset_root / "labelsTs-instance").resolve(),
    ]
    raw_dataset_json_path = (raw_dataset_root / "dataset.json").resolve()
    remove_tree_targets = [
        (nnunet_preprocessed_root(root) / NNUNET_DATASET_NAME).resolve(),
        (nnunet_results_root(root) / NNUNET_DATASET_NAME).resolve(),
    ]

    deleted_paths: list[str] = []
    delete_errors: list[str] = []
    deleted_files = 0
    deleted_dirs = 0
    for base in prune_children_targets:
        if not base.is_dir():
            base.mkdir(parents=True, exist_ok=True)
            continue
        for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                    deleted_dirs += 1
                else:
                    child.unlink(missing_ok=True)
                    deleted_files += 1
                deleted_paths.append(str(child))
            except Exception as exc:
                delete_errors.append(f"{child}: {exc}")
        base.mkdir(parents=True, exist_ok=True)

    for base in raw_dataset_subdirs:
        if base.is_dir():
            for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                        deleted_dirs += 1
                    else:
                        child.unlink(missing_ok=True)
                        deleted_files += 1
                    deleted_paths.append(str(child))
                except Exception as exc:
                    delete_errors.append(f"{child}: {exc}")
        base.mkdir(parents=True, exist_ok=True)
    raw_dataset_root.mkdir(parents=True, exist_ok=True)
    clean_dataset_json = {
        "channel_names": {"0": "em"},
        "labels": {"background": 0, "mitochondria": 1, "contour": 2},
        "numTraining": 0,
        "file_ending": ".nii.gz",
        "name": "Dataset001_mito2",
        "description": "Clean reset template for Dataset001_mito2",
        # Shared crop-profile marker used by Local-HPC / OpenOrganelle / BossDB dedupe.
        "mito2_global_crop_profiles": {},
    }
    raw_dataset_json_path.write_text(
        json.dumps(clean_dataset_json, indent=2) + "\n",
        encoding="utf-8",
    )

    for base in remove_tree_targets:
        if not base.exists():
            continue
        try:
            if base.is_symlink():
                base.unlink(missing_ok=True)
                deleted_files += 1
            elif base.is_dir():
                shutil.rmtree(base)
                deleted_dirs += 1
            else:
                base.unlink(missing_ok=True)
                deleted_files += 1
            deleted_paths.append(str(base))
        except Exception as exc:
            delete_errors.append(f"{base}: {exc}")

    # Clear in-memory slurm history snapshots for all sessions.
    with _slurm_run_state_lock:
        _slurm_run_state.clear()

    return {
        "ok": True,
        "message": "Reset model outputs/slurm history; cleared Dataset001_mito2 raw contents; removed preprocessed/results trees.",
        "targets": [str(p) for p in [*prune_children_targets, *raw_dataset_subdirs, *remove_tree_targets]],
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "deleted_paths": deleted_paths,
        "delete_errors": delete_errors,
    }


@router.post("/inventory/reset-downloaded-training")
def studio_inventory_reset_downloaded_training(body: StudioSessionBody) -> dict[str, Any]:
    """Delete files under ``data/nnUNet_raw/Dataset001_mito2`` and clear registry download history.

    Intended for Stage-0 "start over" behavior so the project looks like no downloads
    have ever been completed for training/inference data.
    """
    root = _root()

    # Avoid mutating on-disk state while stage-3/4 workers are active.
    with _downloader_state_lock:
        if any(bool(v.get("running")) for v in _downloader_state.values()):
            raise HTTPException(status_code=409, detail="Cannot reset while a downloader run is active.")
    with _preprocess_state_lock:
        if any(bool(v.get("running")) for v in _preprocess_state.values()):
            raise HTTPException(status_code=409, detail="Cannot reset while a preprocess run is active.")

    dataset_root = nnunet_dataset_root(root)
    training_root = (dataset_root / "imagesTr").resolve()
    inference_root = (dataset_root / "imagesTs").resolve()
    labels_tr = (dataset_root / "labelsTr").resolve()
    labels_ts = (dataset_root / "labelsTs").resolve()
    labels_tr_instance = (dataset_root / "labelsTr-instance").resolve()
    labels_ts_instance = (dataset_root / "labelsTs-instance").resolve()
    dataset_json_path = (dataset_root / "dataset.json").resolve()
    deleted_paths: list[str] = []
    delete_errors: list[str] = []
    deleted_files = 0
    deleted_dirs = 0

    for base in (
        training_root,
        inference_root,
        labels_tr,
        labels_ts,
        labels_tr_instance,
        labels_ts_instance,
    ):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                    deleted_dirs += 1
                else:
                    child.unlink(missing_ok=True)
                    deleted_files += 1
                deleted_paths.append(str(child))
            except Exception as exc:
                delete_errors.append(f"{child}: {exc}")
    training_root.mkdir(parents=True, exist_ok=True)
    inference_root.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)
    labels_tr_instance.mkdir(parents=True, exist_ok=True)
    labels_ts_instance.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    clean_dataset_json = {
        "channel_names": {"0": "em"},
        "labels": {"background": 0, "mitochondria": 1, "contour": 2},
        "numTraining": 0,
        "file_ending": ".nii.gz",
        "name": "Dataset001_mito2",
        "description": "Clean reset template for Dataset001_mito2",
        # Shared crop-profile marker used by Local-HPC / OpenOrganelle / BossDB dedupe.
        "mito2_global_crop_profiles": {},
    }
    dataset_json_path.write_text(
        json.dumps(clean_dataset_json, indent=2) + "\n",
        encoding="utf-8",
    )

    registry_stats: dict[str, Any] = {
        "registry_exists": False,
        "registry_paths_cleared": [],
        "downloads_deleted": 0,
        "preprocess_runs_deleted": 0,
        "batch_items_deleted": 0,
        "download_batches_deleted": 0,
        "deletion_events_deleted": 0,
        "datasets_unhidden": 0,
    }
    registry_paths = [_registry_path(root)]
    seen_registry_paths: set[str] = set()
    for rp in registry_paths:
        rp_s = str(rp)
        if rp_s in seen_registry_paths or not rp.is_file():
            continue
        seen_registry_paths.add(rp_s)
        registry_stats["registry_exists"] = True
        registry_stats["registry_paths_cleared"].append(rp_s)
        conn = sqlite3.connect(str(rp))
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            registry_stats["preprocess_runs_deleted"] += int(conn.execute("SELECT COUNT(*) FROM preprocess_runs").fetchone()[0])
            registry_stats["downloads_deleted"] += int(conn.execute("SELECT COUNT(*) FROM downloads").fetchone()[0])
            registry_stats["batch_items_deleted"] += int(conn.execute("SELECT COUNT(*) FROM batch_items").fetchone()[0])
            registry_stats["download_batches_deleted"] += int(conn.execute("SELECT COUNT(*) FROM download_batches").fetchone()[0])
            has_deletion_events = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='deletion_events' LIMIT 1"
            ).fetchone() is not None
            if has_deletion_events:
                registry_stats["deletion_events_deleted"] += int(
                    conn.execute("SELECT COUNT(*) FROM deletion_events").fetchone()[0]
                )
            registry_stats["datasets_unhidden"] += int(
                conn.execute("SELECT COUNT(*) FROM datasets WHERE COALESCE(hidden_from_training, 0) != 0").fetchone()[0]
            )
            conn.execute("DELETE FROM preprocess_runs")
            conn.execute("DELETE FROM downloads")
            conn.execute("DELETE FROM batch_items")
            conn.execute("DELETE FROM download_batches")
            try:
                conn.execute("DELETE FROM deletion_events")
            except sqlite3.OperationalError:
                pass
            conn.execute("UPDATE datasets SET hidden_from_training = 0 WHERE COALESCE(hidden_from_training, 0) != 0")
            conn.commit()
        finally:
            conn.close()

    return {
        "ok": True,
        "message": "Reset Dataset001_mito2 data (training + inference/testing) and history complete.",
        "training_root": str(training_root),
        "inference_root": str(inference_root),
        "labels_tr_instance_root": str(labels_tr_instance),
        "labels_ts_instance_root": str(labels_ts_instance),
        "dataset_json": str(dataset_json_path),
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "deleted_paths": deleted_paths,
        "delete_errors": delete_errors,
        "registry": registry_stats,
    }


class StudioScrapeBody(StudioSessionBody):
    url: str = Field(..., min_length=4, description="http://… or https://… landing page to probe")


class StudioDatabaseBuildBody(StudioSessionBody):
    probe: str = Field(default="", description="Relative path to .probe.json, or empty for newest")


def _site_stem_from_probe_arg(probe: str) -> str:
    """Return provider/site stem from a probe path like ``.../<stem>.probe.json``."""
    p = (probe or "").strip()
    if not p:
        return ""
    name = Path(p).name
    if name.endswith(".probe.json"):
        return name[: -len(".probe.json")].strip()
    if name.endswith(".json"):
        return name[: -len(".json")].strip()
    return Path(name).stem.strip()


class StudioDownloaderBody(StudioSessionBody):
    site: str = Field(default="site", min_length=1)
    n_crops: int = Field(default=1, ge=1, le=256)
    voxel_size_nm: str = Field(default="16,16,16", description="Physical voxel size (x,y,z) in nm")
    crop_dimensions_voxels: str = Field(
        default="128,128,128",
        description="Crop dimensions (x,y,z) in voxel counts",
    )
    data_scope: str = Field(
        default="labeled",
        pattern="^(labeled)$",
        description="labeled=good mito masks + matching EM (default).",
    )
    dataset_splits: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("dataset_splits", "datasetSplits"),
        description="Optional per-dataset split mapping: {training: int, inference: int}.",
    )
    execute: bool = False


class StudioMitoLePairBody(BaseModel):
    dataset: str = Field(..., min_length=1)
    source: str = Field(default="")
    image_path: str = Field(default="")
    label_path: str = Field(default="")


class StudioMitoLeDownloaderBody(StudioSessionBody):
    dataset_splits: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("dataset_splits", "datasetSplits"),
    )
    dataset_pairs: list[StudioMitoLePairBody] = Field(
        default_factory=list,
        validation_alias=AliasChoices("dataset_pairs", "datasetPairs"),
    )


class StudioRunDownloaderScriptBody(StudioSessionBody):
    script_path: str = Field(..., min_length=1, description="Path to generated downloader script under 3data_downloader/outputs")


class StudioDownloaderCancelBody(StudioSessionBody):
    pass


class StudioPreprocessSelectiveBody(StudioSessionBody):
    model_config = ConfigDict(extra="ignore")

    dataset_paths: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("dataset_paths", "datasetPaths"),
        description="Absolute raw file paths selected in UI",
    )
    task: str = Field(default="supervised", pattern="^(supervised)$")
    output_format: str = Field(
        default="nifti",
        validation_alias=AliasChoices("output_format", "outputFormat"),
        pattern="^(h5|nifti)$",
        description="Training volume format under data/nnUNet_raw/Dataset001_mito2",
    )
    split_label_cc: bool = Field(
        default=True,
        validation_alias=AliasChoices("split_label_cc", "splitLabelCc"),
        description="Split disconnected components of each label value into unique integer ids (per label file).",
    )
    raw_download_folder: str = Field(
        default="",
        validation_alias=AliasChoices("raw_download_folder", "rawDownloadFolder"),
        description=(
            "Scope for preprocess when dataset_paths is empty: ``training`` for Dataset001_mito2/imagesTr, "
            "or a legacy basename under data/raw."
        ),
    )


class StudioPostprocessBody(StudioSessionBody):
    input_dir: str = Field(
        default="data/outputs/bc",
        description="Input contour prediction directory (relative to project root or absolute path).",
    )
    output_dir: str = Field(
        default="data/outputs/postprocessed",
        description="Output watershed directory (relative to project root or absolute path).",
    )


class StudioEvaluateBody(StudioSessionBody):
    pred_dir: str = Field(
        default="data/outputs/postprocessed",
        description="Postprocessed prediction directory (relative to project root or absolute path).",
    )
    gt_dir: str = Field(
        default="data/nnUNet_raw/Dataset001_mito2/labelsTs-instance",
        description="Ground-truth instance label directory (relative to project root or absolute path).",
    )


def _get_or_init_preprocess_state(sid: str) -> dict[str, Any]:
    with _preprocess_state_lock:
        s = _preprocess_state.get(sid)
        if s is None:
            s = {
                "running": False,
                "log": "",
                "progress": None,
                "result": None,
                "updated_at": time.time(),
            }
            _preprocess_state[sid] = s
        return s


def _preprocess_selective_pid_path(sid: str) -> Path:
    """Stable path so any API worker can SIGKILL the real ``agent.py`` OS process."""
    safe = re.sub(r"[^a-zA-Z0-9._@-]+", "_", (sid or "default").strip() or "default")
    return _root() / ".mito2" / f"preprocess_selective_{safe}.pid"


def _write_preprocess_selective_pid_file(sid: str, pid: int) -> None:
    try:
        p = _preprocess_selective_pid_path(sid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(int(pid)), encoding="utf-8")
    except OSError:
        pass


def _read_preprocess_selective_pid(sid: str) -> int | None:
    try:
        p = _preprocess_selective_pid_path(sid)
        if not p.is_file():
            return None
        return int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _unlink_preprocess_selective_pid_file(sid: str) -> None:
    try:
        _preprocess_selective_pid_path(sid).unlink(missing_ok=True)
    except OSError:
        pass


def _sigkill_child_pids_first(parent_pid: int) -> None:
    """SIGKILL direct children of ``parent_pid`` (and their subtrees) on POSIX — see ``man ps`` ``--ppid``."""
    if int(parent_pid) <= 0 or os.name == "nt":
        return
    try:
        proc_ps = subprocess.run(
            ["ps", "-o", "pid=", "--no-headers", "--ppid", str(int(parent_pid))],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        raw = (proc_ps.stdout or "").strip()
    except (subprocess.TimeoutExpired, Exception):
        return
    for line in raw.splitlines():
        cp = line.strip()
        if not cp.isdigit():
            continue
        cid = int(cp)
        if cid == int(parent_pid):
            continue
        _sigkill_child_pids_first(cid)
        try:
            os.kill(cid, signal.SIGKILL)
        except Exception:
            pass


def _sigkill_preprocess_os_pid(pid: int) -> bool:
    """SIGKILL ``pid``, its descendants, and its process group (POSIX)."""
    if int(pid) <= 0:
        return False
    try:
        if os.name != "nt":
            _sigkill_child_pids_first(int(pid))
            try:
                pgid = os.getpgid(int(pid))
                if pgid > 0:
                    os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                return True
            except PermissionError as exc:
                logging.warning("preprocess SIGKILL pid=%s: permission denied (%s)", pid, exc)
                return False
        else:
            os.kill(int(pid), getattr(signal, "SIGKILL", signal.SIGTERM))
        return True
    except Exception:
        return False


def _studio_project_path_variants(project_root: Path) -> set[str]:
    """Lowercased path spellings for argv matching (resolve vs realpath / symlinks)."""
    out: set[str] = set()
    try:
        p = project_root.resolve()
        for c in (p, Path(os.path.realpath(str(p)))):
            s = str(c).replace("\\", "/").lower().rstrip("/")
            if s:
                out.add(s)
    except OSError:
        pass
    return out


def _stage4_preprocessor_dirs(project_root: Path) -> set[Path]:
    """``3data_downloader`` directory paths (resolve + realpath) for ``/proc/…/cwd`` matching."""
    out: set[Path] = set()
    try:
        s4 = (project_root / "3data_downloader").resolve()
        for c in (s4, Path(os.path.realpath(str(s4)))):
            out.add(c)
    except OSError:
        pass
    return out


def _list_stage4_supervised_by_procfs_cwd(project_root: Path, budget_sec: float = 3.0) -> list[tuple[int, int]]:
    """Linux: supervised ``agent.py`` via ``/proc`` — **cwd** under stage-4 **or** ``--project-root`` in argv."""
    if os.name != "posix" or not os.path.isdir("/proc"):
        return []
    s4_dirs = _stage4_preprocessor_dirs(project_root)
    root_hits = {v for v in _studio_project_path_variants(project_root) if len(v) >= 8}
    deadline = time.monotonic() + float(budget_sec)
    out: list[tuple[int, int]] = []
    try:
        for name in os.listdir("/proc"):
            if time.monotonic() > deadline:
                break
            if not name.isdigit():
                continue
            pid = int(name)
            if pid == os.getpid():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace").lower()
            except OSError:
                continue
            if "downloader_master/preprocess_agent.py" not in cmd or "--task" not in cmd or "supervised" not in cmd:
                continue
            cwd_ok = False
            try:
                cwd = Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
                cwd_real = Path(os.path.realpath(f"/proc/{pid}/cwd"))
                cwd_ok = cwd in s4_dirs or cwd_real in s4_dirs
            except OSError:
                pass
            cmd_ok = any(rv in cmd for rv in root_hits)
            if not cwd_ok and not cmd_ok:
                continue
            try:
                pgid = os.getpgid(pid)
            except OSError:
                continue
            out.append((pid, pgid))
    except Exception:
        pass
    return out


def _list_stage4_agent_pids_s4_cwd_any_task(project_root: Path, budget_sec: float = 3.0) -> list[tuple[int, int]]:
    """Linux: any ``agent.py --task …`` whose **cwd** is this repo's ``3data_downloader``.

    Catches processes that ``supervised`` / ``--project-root`` matching missed (truncated ``ps``/cmdline).
    """
    if os.name != "posix" or not os.path.isdir("/proc"):
        return []
    s4_dirs = _stage4_preprocessor_dirs(project_root)
    deadline = time.monotonic() + float(budget_sec)
    out: list[tuple[int, int]] = []
    try:
        for name in os.listdir("/proc"):
            if time.monotonic() > deadline:
                break
            if not name.isdigit():
                continue
            pid = int(name)
            if pid == os.getpid():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace").lower()
            except OSError:
                continue
            if "downloader_master/preprocess_agent.py" not in cmd or "--task" not in cmd:
                continue
            try:
                cwd = Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
                cwd_real = Path(os.path.realpath(f"/proc/{pid}/cwd"))
                cwd_ok = cwd in s4_dirs or cwd_real in s4_dirs
            except OSError:
                continue
            if not cwd_ok:
                continue
            try:
                pgid = os.getpgid(pid)
            except OSError:
                continue
            out.append((pid, pgid))
    except Exception:
        pass
    return out


def _append_preprocess_log(sid: str, text: str) -> None:
    with _preprocess_state_lock:
        s = _get_or_init_preprocess_state(sid)
        merged = (s.get("log") or "") + text
        s["log"] = merged[-200000:]
        s["updated_at"] = time.time()


def _set_preprocess_progress(sid: str, progress: dict[str, Any] | None) -> None:
    with _preprocess_state_lock:
        s = _get_or_init_preprocess_state(sid)
        s["progress"] = progress
        s["updated_at"] = time.time()


def _canonical_preprocess_image_key(path_s: str) -> str:
    """Match ``3data_downloader/downloader_master/preprocess_agent._canonical_key`` so disk-based progress matches written files."""

    path = Path(path_s.replace("\\", "/"))
    if path.name.endswith(".nii.gz"):
        name = path.name[:-7]
    else:
        name = path.stem
    name = re.sub(r"_(im|seg)$", "", name, flags=re.I)
    name = re.sub(r"\.(im|seg)$", "", name, flags=re.I)
    return name


def _preprocess_selected_total_from_argv(argv: list[str]) -> int:
    """Unique EM volume keys selected for preprocess."""
    return len(_preprocess_selected_image_keys_from_argv(argv))


def _preprocess_selected_image_keys_from_argv(argv: list[str]) -> list[str]:
    try:
        i = argv.index("--dataset-paths-json")
    except ValueError:
        return []
    if i + 1 >= len(argv):
        return []
    try:
        raw = json.loads(argv[i + 1])
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue
        xs = x.replace("\\", "/")
        if "/images/" not in xs.lower() and "/imagestr/" not in xs.lower():
            continue
        if not (xs.lower().endswith("_im.h5") or xs.lower().endswith("_0000.nii.gz")):
            continue
        key = _canonical_preprocess_image_key(xs)
        if key:
            out.append(key)
    return sorted(set(out))


def _preprocess_output_format_from_argv(argv: list[str]) -> str:
    try:
        i = argv.index("--output-format")
    except ValueError:
        return "h5"
    if i + 1 >= len(argv):
        return "h5"
    v = str(argv[i + 1]).strip().lower()
    return "nifti" if v == "nifti" else "h5"


def _project_root_from_preprocess_argv(argv: list[str]) -> Path | None:
    """Same tree ``agent.py`` uses for Dataset001_mito2 (must match ``--project-root``)."""
    try:
        i = argv.index("--project-root")
        if i + 1 >= len(argv):
            return None
        return Path(str(argv[i + 1])).resolve()
    except ValueError:
        return None


def _build_preprocess_selective_argv(root: Path, body: StudioPreprocessSelectiveBody) -> list[str]:
    """Build ``downloader_master/preprocess_agent.py`` argv for selective supervised preprocess; raises ``HTTPException`` on invalid input."""
    s4 = root / "3data_downloader"
    if not (s4 / "downloader_master/preprocess_agent.py").is_file():
        raise HTTPException(status_code=500, detail="Stage 4 agent.py missing")
    py = sys.executable
    task = "supervised"
    argv = [py, "-u", "downloader_master/preprocess_agent.py", "--task", task, "--project-root", str(root.resolve())]
    raw_base = (root / "data" / "raw").resolve()
    train_img = (nnunet_dataset_root(root) / "imagesTr").resolve()
    train_lbl = (nnunet_dataset_root(root) / "labelsTr").resolve()
    du_raw = (root / "data" / "data_unlabeled" / "data_raw").resolve()
    dl_raw = (root / "data" / "data_labeled" / "data_raw").resolve()
    preprocess_source_bases = [
        raw_base,
        train_img,
        train_lbl,
        du_raw,
        dl_raw,
    ]

    def _is_allowed_preprocess_source(p: Path) -> bool:
        pr = p.resolve()
        for b in preprocess_source_bases:
            try:
                pr.relative_to(b.resolve())
                return True
            except ValueError:
                continue
        return False

    def _discover_single_download_run() -> str | None:
        if not raw_base.is_dir():
            return None
        hits: list[tuple[str, int]] = []
        for child in sorted(raw_base.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            img_dir = _preprocess_em_stack_dir(child)
            if img_dir is None:
                continue
            n = _count_em_h5_for_preprocess(img_dir)
            if n > 0:
                hits.append((child.name, n))
        if len(hits) == 1:
            return hits[0][0]
        return None

    def _resolve_preprocess_run_dir(name: str) -> Path | None:
        n = name.strip()
        if not n:
            return None
        low = n.lower().replace("\\", "/").strip("/")
        if low in ("training", "train", "data/nnunet_raw/dataset001_mito2/imagestr"):
            return train_img if train_img.is_dir() else None
        raw_res = raw_base.resolve()
        rd = (raw_base / n).resolve()
        try:
            rd.relative_to(raw_res)
        except ValueError:
            return None
        return rd if rd.is_dir() else None

    paths = [Path(p).resolve() for p in (body.dataset_paths or []) if str(p).strip()]
    run_name = (body.raw_download_folder or "").strip()
    if not paths and not run_name:
        if train_img.is_dir():
            ims = sorted(
                (
                    p
                    for p in train_img.iterdir()
                    if p.is_file()
                    and (p.name.lower().endswith("_im.h5") or p.name.lower().endswith("_0000.nii.gz"))
                ),
                key=lambda p: str(p).lower(),
            )
            if ims:
                paths = [p.resolve() for p in ims]
        if not paths:
            guessed = _discover_single_download_run()
            if guessed:
                run_name = guessed
    if run_name:
        run_dir = _resolve_preprocess_run_dir(run_name)
        if run_dir is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Download folder not found: {run_name!r} "
                    "(use a subfolder name under data/raw for legacy runs, or training for flat stacks)."
                ),
            )
        if not paths:
            img_dir = _preprocess_em_stack_dir(run_dir)
            if img_dir is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No EM stacks found for {run_name!r} "
                        "(expected …/images/*_im.h5 under a raw run, or *_0000.nii.gz directly under imagesTr)."
                    ),
                )
            paths = sorted(
                (
                    p
                    for p in img_dir.iterdir()
                    if p.is_file()
                    and (p.name.lower().endswith("_im.h5") or p.name.lower().endswith("_0000.nii.gz"))
                ),
                key=lambda p: str(p).lower(),
            )
            paths = [p.resolve() for p in paths]
        else:
            run_s = str(run_dir).replace("\\", "/")
            paths = [p for p in paths if str(p).replace("\\", "/").startswith(run_s + "/")]
    filtered: list[Path] = []
    for p in paths:
        if _is_allowed_preprocess_source(p):
            filtered.append(p)
            p_s = str(p).replace("\\", "/")
            if "/images/" in p_s:
                label_s = p_s.replace("/images/", "/labels/")
                label_s = re.sub(r"_im(\.[^.]+(?:\.gz)?)$", r"_seg\1", label_s)
                lp = Path(label_s).resolve()
                if lp.is_file():
                    filtered.append(lp)
            elif "/imagesTr/" in p_s.lower():
                label_s = re.sub(r"/imagesTr/", "/labelsTr/", p_s, flags=re.IGNORECASE)
                label_s = re.sub(r"_im(\.[^.]+(?:\.gz)?)$", r"_seg\1", label_s)
                lp = Path(label_s).resolve()
                if lp.is_file():
                    filtered.append(lp)
    uniq = sorted({str(p): p for p in filtered}.values(), key=lambda p: str(p).lower())
    if not uniq:
        n_in = len(paths)
        n_gate = sum(1 for p in paths if _is_allowed_preprocess_source(p))
        extra = (
            f" raw_download_folder={run_name!r}, input_paths={n_in}, passed_source_gate={n_gate}, "
            f"train_images={train_img}."
        )
        if run_name and not (body.dataset_paths or []):
            rd = _resolve_preprocess_run_dir(run_name)
            img_dir = _preprocess_em_stack_dir(rd) if rd is not None else None
            n_glob = _count_em_h5_for_preprocess(img_dir) if img_dir is not None else 0
            extra += f" Glob EM stacks in resolved run: {n_glob}."
        raise HTTPException(status_code=400, detail=f"No valid datasets selected for task={task}.{extra}")
    argv.extend(["--dataset-paths-json", json.dumps([str(p) for p in uniq])])
    out_fmt = (body.output_format or "h5").strip().lower()
    argv.extend(["--output-format", out_fmt if out_fmt in ("h5", "nifti") else "h5"])
    if body.split_label_cc:
        argv.append("--split-label-cc")
    return argv


def _preprocess_selective_worker(sid: str, argv: list[str], cwd: Path) -> None:
    import subprocess

    proc: Any | None = None
    out_tail: list[str] = []
    out_tail_lock = threading.Lock()
    rc = 1
    was_kill = False
    payload: dict[str, Any] | None = None
    selected_total = _preprocess_selected_total_from_argv(argv)
    selected_keys = _preprocess_selected_image_keys_from_argv(argv)
    out_fmt = _preprocess_output_format_from_argv(argv)
    monitor_root = _project_root_from_preprocess_argv(argv) or cwd.resolve().parent
    try:
        _append_preprocess_log(sid, "[mito2] Selective preprocess subprocess starting…\n")
        if selected_total > 0:
            _set_preprocess_progress(
                sid,
                {"completed": 0, "total": selected_total, "current": 1, "dataset": "starting"},
            )
        with _preprocess_kill_requested_lock:
            if sid in _preprocess_kill_requested:
                _preprocess_kill_requested.discard(sid)
                session = _session(sid)
                payload = {
                    "ok": False,
                    "message": "Selective preprocess stopped (killed).",
                    "returncode": 130,
                    "stdout": "",
                    "stderr": "",
                    "pipeline": _pipeline_dict(session),
                }
            else:
                # New session so ``killpg`` from the Studio cancel handler only affects this subtree (POSIX).
                popen_kw: dict[str, Any] = {
                    "cwd": str(cwd),
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "bufsize": 1,
                    "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
                }
                if os.name != "nt":
                    popen_kw["start_new_session"] = True
                proc = subprocess.Popen(argv, **popen_kw)
                with _preprocess_procs_lock:
                    _preprocess_procs[sid] = proc
                _write_preprocess_selective_pid_file(sid, proc.pid)
                with _preprocess_kill_requested_lock:
                    if sid in _preprocess_kill_requested:
                        try:
                            proc.kill()
                        except Exception:
                            pass

                def _pump_stdout() -> None:
                    if proc.stdout is None:
                        return
                    try:
                        for line in iter(proc.stdout.readline, ""):
                            if not line:
                                break
                            m_prog = re.search(r"\[PROGRESS\]\s+dataset\s+(\d+)/(\d+):\s*(.+)\s*$", line)
                            if m_prog:
                                cur = int(m_prog.group(1))
                                total = int(m_prog.group(2))
                                _set_preprocess_progress(
                                    sid,
                                    {
                                        "completed": max(0, cur - 1),
                                        "total": total,
                                        "current": cur,
                                        "dataset": m_prog.group(3).strip(),
                                    },
                                )
                            m_done = re.search(r"\[DONE\]\s+dataset\s+(\d+)/(\d+):\s*(.+)\s*$", line)
                            if m_done:
                                cur = int(m_done.group(1))
                                total = int(m_done.group(2))
                                _set_preprocess_progress(
                                    sid,
                                    {
                                        "completed": cur,
                                        "total": total,
                                        "current": min(cur + 1, total),
                                        "dataset": m_done.group(3).strip(),
                                    },
                                )
                            with out_tail_lock:
                                out_tail.append(line)
                                if len(out_tail) > 4000:
                                    del out_tail[: len(out_tail) - 4000]
                            _append_preprocess_log(sid, line)
                    finally:
                        try:
                            proc.stdout.close()
                        except Exception:
                            pass

                stop_mon = threading.Event()

                def _monitor_outputs() -> None:
                    if selected_total <= 0 or not selected_keys:
                        return
                    img_dir = monitor_nnunet_dataset_root(root) / "imagesTr"
                    keys_lower = [k.lower() for k in selected_keys]
                    # Match output stems case-insensitively (some FS / path normalizations differ).
                    while not stop_mon.is_set():
                        stems_lower: set[str] = set()
                        try:
                            for p in img_dir.glob("*.h5"):
                                stems_lower.add(p.stem.lower())
                            for p in img_dir.glob("*.nii.gz"):
                                stems_lower.add(p.name[: -len(".nii.gz")].lower())
                        except OSError:
                            pass
                        done = sum(1 for kl in keys_lower if kl in stems_lower)
                        done = max(0, min(done, selected_total))
                        cur = selected_total if done >= selected_total else max(1, done + 1)
                        _set_preprocess_progress(
                            sid,
                            {"completed": done, "total": selected_total, "current": cur, "dataset": "writing outputs"},
                        )
                        stop_mon.wait(0.4)

                pump_t = threading.Thread(target=_pump_stdout, name=f"preprocess-pump-{sid}", daemon=True)
                mon_t = threading.Thread(target=_monitor_outputs, name=f"preprocess-monitor-{sid}", daemon=True)
                pump_t.start()
                mon_t.start()
                start = time.monotonic()
                next_heartbeat = start + 3.0
                while True:
                    polled = proc.poll()
                    if polled is not None:
                        rc = int(polled)
                        break
                    now = time.monotonic()
                    if now >= next_heartbeat:
                        elapsed = int(now - start)
                        _append_preprocess_log(sid, f"[mito2] Preprocess still running… elapsed {elapsed}s\n")
                        next_heartbeat = now + 3.0
                    time.sleep(0.5)
                stop_mon.set()
                mon_t.join(timeout=1.0)
                pump_t.join(timeout=10.0)
                ok = rc == 0
                with _preprocess_kill_requested_lock:
                    was_kill = sid in _preprocess_kill_requested
                session = _session(sid)
                if ok:
                    _apply_step(session, PipelineStep.PREPROCESS)
                with out_tail_lock:
                    full_out = "".join(out_tail)
                sel_result = {"returncode": rc, "stdout": full_out[-8000:], "stderr": ""}
                if not ok and not was_kill:
                    _emit_run_failure_to_terminal("stage4-preprocess-selective", sel_result)
                if ok:
                    sel_msg = "Selective preprocess completed (supervised)."
                elif was_kill:
                    sel_msg = "Selective preprocess stopped (killed)."
                else:
                    sel_msg = _studio_run_message(False, "", "Selective preprocess failed.", sel_result)
                payload = {
                    "ok": ok,
                    "message": sel_msg,
                    "returncode": rc,
                    "stdout": sel_result["stdout"],
                    "stderr": sel_result["stderr"],
                    "pipeline": _pipeline_dict(session),
                }
    except Exception as exc:  # noqa: BLE001
        session = _session(sid)
        err = str(exc)
        _append_preprocess_log(sid, f"\n[mito2] Preprocess error: {err}\n")
        with out_tail_lock:
            tail_txt = "".join(out_tail)[-8000:]
        ex_payload = {"returncode": 1, "stdout": tail_txt, "stderr": err}
        _emit_run_failure_to_terminal("stage4-preprocess-selective-exception", ex_payload)
        payload = {
            "ok": False,
            "message": _studio_run_message(False, "", "Selective preprocess failed (exception).", ex_payload),
            "returncode": 1,
            "stdout": tail_txt,
            "stderr": err,
            "pipeline": _pipeline_dict(session),
        }
    finally:
        _unlink_preprocess_selective_pid_file(sid)
        with _preprocess_kill_requested_lock:
            _preprocess_kill_requested.discard(sid)
        with _preprocess_state_lock:
            s = _get_or_init_preprocess_state(sid)
            s["running"] = False
            s["progress"] = None
            if payload is not None:
                s["result"] = payload
            s["updated_at"] = time.time()
        with _preprocess_procs_lock:
            cur = _preprocess_procs.get(sid)
            if proc is not None and cur is proc:
                _preprocess_procs.pop(sid, None)


def _clear_preprocess_selective_state_for_session(sid: str) -> dict[str, Any]:
    with _preprocess_state_lock:
        s = _get_or_init_preprocess_state(sid)
        if s.get("running"):
            raise HTTPException(status_code=409, detail="Cannot clear preprocess output while a run is active")
        s.update(
            {
                "log": "",
                "progress": None,
                "result": None,
                "updated_at": time.time(),
            }
        )
    return {"ok": True, "cleared": True}


def _get_or_init_downloader_state(sid: str) -> dict[str, Any]:
    with _downloader_state_lock:
        s = _downloader_state.get(sid)
        if s is None:
            s = {
                "running": False,
                "script_path": "",
                "log": "",
                "progress": None,
                "result": None,
                "updated_at": time.time(),
            }
            _downloader_state[sid] = s
        return s


def _append_downloader_log(sid: str, text: str) -> None:
    with _downloader_state_lock:
        s = _get_or_init_downloader_state(sid)
        merged = (s.get("log") or "") + text
        s["log"] = merged[-200000:]
        s["updated_at"] = time.time()


def _set_downloader_progress(sid: str, progress: dict[str, Any]) -> None:
    with _downloader_state_lock:
        s = _get_or_init_downloader_state(sid)
        s["progress"] = progress
        s["updated_at"] = time.time()


def _planned_crop_pairs_from_splits(dataset_splits: dict[str, dict[str, int]] | None) -> int:
    total = 0
    for _, split in (dataset_splits or {}).items():
        try:
            tr = max(0, int((split or {}).get("training", 0) or 0))
            ts = max(0, int((split or {}).get("inference", 0) or 0))
            total += tr + ts
        except Exception:
            continue
    return int(total)


def _mitole_dataset_total_crops(dataset_splits: dict[str, dict[str, int]], dataset: str) -> int:
    split = (dataset_splits or {}).get(dataset, {}) or {}
    try:
        tr = max(0, int(split.get("training", 0) or 0))
        ts = max(0, int(split.get("inference", 0) or 0))
        return int(tr + ts)
    except Exception:
        return 0


def _mitole_noop_downloader_log_and_progress(
    *,
    root: Path,
    pairs_for_counts: list[dict[str, str]],
    dataset_splits: dict[str, dict[str, int]],
    skipped_incremental: list[str],
) -> tuple[str, dict[str, Any]]:
    """Web-scrape-style console lines + full progress when there is nothing left to materialize."""
    _, req_e = mitole_pending_and_requested_crop_counts(root, pairs_for_counts, dataset_splits)
    names: list[str] = []
    for p in pairs_for_counts:
        ds = str(p.get("dataset") or "").strip()
        if ds and _mitole_dataset_total_crops(dataset_splits, ds) > 0 and ds not in names:
            names.append(ds)
    if not names and skipped_incremental:
        for ln in skipped_incremental:
            pre = str(ln).split(":", 1)[0].strip()
            if pre and pre not in names:
                names.append(pre)
    n_total = max(1, len(names)) if (names or skipped_incremental) else 1
    split_plan = max(1, int(req_e), int(_planned_crop_pairs_from_splits(dataset_splits)))
    skip_disp = ", ".join(names) if names else "—"
    parts: list[str] = [
        "[mito2] Local HPC downloader started...\n",
        f"[INFO] source=local-hpc planned crop pairs={split_plan}\n",
        "[INFO] new crops to write=0 (progress bar denominator only)\n",
    ]
    for ln in skipped_incremental:
        parts.append(f"[INFO] {ln}\n")
    parts.extend(
        [
            f"- Planned image/label pairs: {split_plan}\n",
            f"- Datasets: {n_total}\n",
            f"[PLAN] 0 of {n_total} dataset(s) need download; {n_total} already complete.\n",
            f"[SKIP] Already complete: {skip_disp}\n",
            "[NOOP] No new assets to download — all "
            f"{split_plan} crop pair(s) are already complete for this profile.\n",
            "[NOOP] No output folder will be created. "
            "Re-run with different n_crops/chunk/voxel for a new profile.\n",
        ]
    )
    prog: dict[str, Any] = {
        "completed": int(split_plan),
        "total": int(split_plan),
        "current": int(split_plan),
        "dataset": "done",
    }
    return "".join(parts), prog


def _mitole_infer_progress_from_log(log_text: str, done: bool) -> dict[str, Any] | None:
    """Infer Local-HPC stage-3 progress from log text when explicit state is missing."""
    txt = str(log_text or "")
    if not txt.strip():
        return None
    m_total = (
        re.search(r"new crops to write=(\d+)", txt, flags=re.IGNORECASE)
        or re.search(r"Pairs to materialize now:\s*(\d+)", txt, flags=re.IGNORECASE)
        or re.search(r"\b(\d+)\s+new to write this run\b", txt, flags=re.IGNORECASE)
    )
    total = int(m_total.group(1)) if m_total else 0
    if total <= 0:
        return None
    done_hits = re.findall(r"\[DONE\][^\n]*(?:training|inference|global)\s+vol\s+\d+/\d+", txt, flags=re.IGNORECASE)
    completed = len(done_hits)
    if done:
        completed = total
    completed = max(0, min(total, int(completed)))
    current = total if done else min(total, max(1, completed + (0 if done else 1)))
    return {"completed": int(completed), "total": int(total), "current": int(current), "dataset": "done" if done else ""}


def _count_dataset_units_on_disk(project_root: Path, dataset_stable: str) -> int:
    """Count already-materialized nnUNet image units for a stable dataset key."""
    tr, ts = _mitole_disk_split_counts(project_root, dataset_stable)
    return int(tr + ts)


def _mitole_disk_split_counts(project_root: Path, dataset_stable: str) -> tuple[int, int]:
    """Count training (imagesTr) vs inference (imagesTs) units matching ``dataset_stable``."""
    dataset_root = nnunet_dataset_root(project_root)
    counts: dict[str, int] = {"training": 0, "inference": 0}
    for sub, key in (("imagesTr", "training"), ("imagesTs", "inference")):
        base = (dataset_root / sub).resolve()
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if not p.is_file():
                continue
            n = p.name.lower()
            if not n.endswith("_0000.nii.gz"):
                continue
            n_stable = _normalize_stable_id(p.name)
            if not (n_stable == dataset_stable or n_stable.startswith(f"{dataset_stable}_")):
                continue
            counts[key] = int(counts[key]) + 1
    return int(counts["training"]), int(counts["inference"])


def _register_mitole_manifest_into_registry(
    *,
    project_root: Path,
    manifest: dict[str, Any],
    dataset_splits: dict[str, dict[str, int]],
) -> None:
    """Create registry download_batch + batch_items for Local-HPC downloader runs."""
    run_name = str(manifest.get("run_name") or "").strip() or f"mitole_local_{now_us_eastern().strftime('%Y%m%d_%H%M%S')}"
    run_folder = str(manifest.get("run_dir") or "")
    rows = list(manifest.get("rows") or [])
    if not rows:
        return
    from agent.orchestration.registry.api import (  # noqa: PLC0415
        create_download_batch,
        update_batch_status,
        upsert_batch_item,
        upsert_dataset,
        upsert_provider,
    )
    from agent.orchestration.registry.schema import open_registry  # noqa: PLC0415
    reg = _registry_path(project_root)
    conn = open_registry(reg)
    try:
        provider_name = "Local HPC"
        per_dataset_totals: dict[str, int] = {}
        train_pairs = 0
        infer_pairs = 0
        for r in rows:
            ds = str(r.get("dataset") or "").strip()
            if not ds:
                continue
            sid = _normalize_stable_id(ds)
            per_dataset_totals[sid] = _mitole_dataset_total_crops(dataset_splits, ds)
            for m in (r.get("nnunet_materialized") or []):
                spl = str(m.get("split") or "training").strip().lower()
                if spl not in ("training", "inference"):
                    spl = "training"
                if spl == "inference":
                    infer_pairs += 1
                else:
                    train_pairs += 1
        asset_units = int(2 * (train_pairs + infer_pairs))
        profile_json = {
            "source": "mitole_local",
            "run_name": run_name,
            "datasets_this_run": len(per_dataset_totals),
            "dataset_totals": per_dataset_totals,
            "training_pairs_this_run": int(train_pairs),
            "inference_pairs_this_run": int(infer_pairs),
            "training_units_this_run": int(2 * train_pairs),
            "inference_units_this_run": int(2 * infer_pairs),
            "download_asset_completions": asset_units,
        }
        bid = create_download_batch(
            conn,
            batch_id=run_name,
            provider=provider_name,
            profile_hash=None,
            profile_json=profile_json,
            run_folder=run_folder,
        )
        conn.execute(
            "UPDATE download_batches SET download_asset_completions = ? WHERE id = ?",
            (asset_units, int(bid)),
        )
        pid = upsert_provider(conn, name=provider_name, base_url="")
        for r in rows:
            ds = str(r.get("dataset") or "").strip()
            if not ds:
                continue
            stable = _normalize_stable_id(ds)
            did = upsert_dataset(
                conn,
                provider_id=pid,
                stable_id=stable,
                metadata={"source": "mitole_local", "run_name": run_name},
                changed=False,
            )
            mats = list(r.get("nnunet_materialized") or [])
            split_idx: dict[str, int] = {"training": 0, "inference": 0}
            for m in mats:
                split = str(m.get("split") or "training").strip().lower()
                if split not in ("training", "inference"):
                    split = "training"
                split_idx[split] = int(split_idx.get(split, 0)) + 1
                unit = int(split_idx[split])
                # Preserve one row per materialized unit (crop pair), not one row
                # per dataset. This keeps Inventory counts/summaries aligned with
                # downloader output (pairs and train/inference units).
                stable_unit = f"{stable}__{split}_{unit}"
                img_path = str(m.get("nnunet_image_dst") or "")
                lbl_path = str(m.get("nnunet_label_dst") or "")
                upsert_batch_item(
                    conn,
                    batch_db_id=int(bid),
                    dataset_id=int(did),
                    stable_id=stable_unit,
                    asset_type="em_volume",
                    local_path=img_path or None,
                    status="present",
                )
                upsert_batch_item(
                    conn,
                    batch_db_id=int(bid),
                    dataset_id=int(did),
                    stable_id=stable_unit,
                    asset_type="mito_seg",
                    local_path=lbl_path or None,
                    status="present",
                )
        update_batch_status(conn, int(bid), "complete")
        conn.commit()
    finally:
        conn.close()


def _run_downloader_command_live(
    sid: str,
    argv: list[str],
    *,
    cwd: Path,
    timeout_sec: float | None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run downloader command with line-by-line log mirroring (same output feel as streamed manual run)."""
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=({**os.environ, **env} if env else None),
        )
    except Exception as e:  # noqa: BLE001
        err = str(e)
        _append_downloader_log(sid, f"[mito2] Downloader launch failed: {err}\n")
        return {"returncode": 1, "stdout": "", "stderr": err}

    with _downloader_procs_lock:
        _downloader_procs[sid] = proc

    out_tail: list[str] = []
    err_tail: list[str] = []
    dl_progress = DownloaderLogProgressParser(lambda d: _set_downloader_progress(sid, d))

    def _pump(stream: Any, kind: str) -> None:
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                if dl_progress.filter_noise_line(line):
                    continue
                dl_progress.consume_line(line)
                if kind == "stdout":
                    out_tail.append(line)
                    if len(out_tail) > 500:
                        del out_tail[: len(out_tail) - 500]
                else:
                    err_tail.append(line)
                    if len(err_tail) > 500:
                        del err_tail[: len(err_tail) - 500]
                _append_downloader_log(sid, line)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

    start = time.monotonic()
    hb = _downloader_heartbeat_sec()
    next_heartbeat = start + hb if hb > 0 else float("inf")
    rc: int | None = None
    while rc is None:
        rc = proc.poll()
        now = time.monotonic()
        if rc is not None:
            break
        if hb > 0 and now >= next_heartbeat:
            elapsed = int(now - start)
            _append_downloader_log(sid, f"[mito2] Downloader still running… elapsed {elapsed}s\n")
            next_heartbeat = now + hb
        if timeout_sec is not None and (now - start) >= timeout_sec:
            proc.kill()
            rc = 124
            _append_downloader_log(sid, f"\n[mito2] Timed out after {timeout_sec}s.\n")
            break
        time.sleep(1.0)

    t_out.join(timeout=2.0)
    t_err.join(timeout=2.0)
    with _downloader_procs_lock:
        cur = _downloader_procs.get(sid)
        if cur is proc:
            _downloader_procs.pop(sid, None)

    return {
        "returncode": int(rc if rc is not None else 1),
        "stdout": "".join(out_tail)[-200000:],
        "stderr": "".join(err_tail)[-100000:],
    }


def _database_build_state_begin(sid: str) -> None:
    with _database_build_state_lock:
        _database_build_state[sid] = {
            "running": True,
            "log": "[mito2] Starting database / inventory build…\n",
            "result": None,
            "updated_at": time.time(),
        }


def _database_build_log_append(sid: str, text: str) -> None:
    if not text:
        return
    with _database_build_state_lock:
        row = _database_build_state.get(sid)
        if not row:
            return
        merged = (row.get("log") or "") + text
        row["log"] = merged[-500000:]
        row["updated_at"] = time.time()


def _database_build_state_finish(sid: str, payload: dict[str, Any]) -> None:
    with _database_build_state_lock:
        row = _database_build_state.get(sid)
        if not row:
            return
        row["running"] = False
        row["result"] = payload
        row["updated_at"] = time.time()


def _database_build_state_fail(sid: str, err: str) -> None:
    with _database_build_state_lock:
        row = _database_build_state.get(sid)
        if not row:
            return
        row["running"] = False
        row["log"] = (row.get("log") or "") + f"\n[mito2] {err}\n"
        row["updated_at"] = time.time()


def _run_database_build_command_live(
    sid: str,
    argv: list[str],
    *,
    cwd: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    """Run stage-2 build command and mirror output to `/run/database-state` incrementally."""
    proc: subprocess.Popen[str] | None = None
    out_tail: list[str] = []
    err_tail: list[str] = []
    out_lock = threading.Lock()
    err_lock = threading.Lock()
    rc: int | None = None

    try:
        popen_kw: dict[str, Any] = {
            "cwd": str(cwd),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
            "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
        }
        if os.name != "nt":
            popen_kw["start_new_session"] = True
        proc = subprocess.Popen(argv, **popen_kw)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        _database_build_log_append(sid, f"[mito2] Database build launch failed: {err}\n")
        return {"returncode": 1, "stdout": "", "stderr": err}

    def _pump(stream: Any, kind: str) -> None:
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                _database_build_log_append(sid, line)
                if kind == "stdout":
                    with out_lock:
                        out_tail.append(line)
                        if len(out_tail) > 6000:
                            del out_tail[: len(out_tail) - 6000]
                else:
                    with err_lock:
                        err_tail.append(line)
                        if len(err_tail) > 3000:
                            del err_tail[: len(err_tail) - 3000]
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

    start = time.monotonic()
    while rc is None:
        rc = proc.poll()
        if rc is not None:
            break
        if timeout_sec is not None and (time.monotonic() - start) >= float(timeout_sec):
            try:
                proc.kill()
            except Exception:
                pass
            rc = 124
            _database_build_log_append(sid, f"\n[mito2] Database build timed out after {timeout_sec}s.\n")
            break
        time.sleep(0.5)

    t_out.join(timeout=3.0)
    t_err.join(timeout=3.0)
    return {
        "returncode": int(rc if rc is not None else 1),
        "stdout": "".join(out_tail)[-200000:],
        "stderr": "".join(err_tail)[-200000:],
    }


def _downloader_sync_execute_begin(sid: str, headline: str, script_path: str = "", planned_total: int = 0) -> None:
    prog: dict[str, Any] | None = None
    if int(planned_total) > 0:
        prog = {"completed": 0, "total": int(planned_total), "current": 1, "dataset": ""}
    with _downloader_state_lock:
        st = _get_or_init_downloader_state(sid)
        st.update(
            {
                "running": True,
                "script_path": (script_path or _STUDIO_RUN_DOWNLOADER_SYNC_TAG),
                "log": headline,
                "progress": prog,
                "result": None,
                "updated_at": time.time(),
            }
        )


def _downloader_sync_execute_clear_running(sid: str) -> None:
    with _downloader_state_lock:
        st = _downloader_state.get(sid)
        if st and st.get("running"):
            st["running"] = False
            st["updated_at"] = time.time()


def _downloader_sync_execute_finalize_from_out(sid: str, session: PipelineSession, out: dict[str, Any]) -> None:
    res: dict[str, Any] = {
        "ok": bool(out.get("ok")),
        "message": str(out.get("message") or ""),
        "returncode": int(out.get("returncode", 1)),
        "stdout": str(out.get("stdout") or ""),
        "stderr": str(out.get("stderr") or ""),
        "pipeline": _pipeline_dict(session),
    }
    if out.get("downloader_generations"):
        res["downloader_generations"] = out["downloader_generations"]
        res["openorganelle_generations"] = out.get("openorganelle_generations") or out["downloader_generations"]
    with _downloader_state_lock:
        st = _get_or_init_downloader_state(sid)
        st["running"] = False
        st["result"] = res
        st["updated_at"] = time.time()


@router.get("/run/downloader-script-state")
def studio_run_downloader_script_state(session_id: str = "default") -> dict[str, Any]:
    sid = (session_id or "default").strip() or "default"
    with _downloader_procs_lock:
        proc = _downloader_procs.get(sid)
    proc_dead = False
    if proc is not None:
        try:
            proc_dead = proc.poll() is not None
        except Exception:
            proc_dead = True
    with _downloader_state_lock:
        s_ref = _get_or_init_downloader_state(sid)
        # Self-heal stale "running" flags only for subprocess-backed downloader runs.
        # Local-HPC Stage-3 uses in-process execution (no Popen handle), so forcing
        # running=False here would hide live progress/log updates while work continues.
        is_inprocess_mitole = str(s_ref.get("script_path") or "") == "[pipeline] mitole_stage3"
        if bool(s_ref.get("running")) and not is_inprocess_mitole and (proc is None or proc_dead):
            s_ref["running"] = False
            if s_ref.get("result") is None:
                session = _session(sid)
                s_ref["result"] = {
                    "ok": False,
                    "message": "Downloader stopped (process no longer running).",
                    "returncode": 130,
                    "stdout": "",
                    "stderr": "",
                    "pipeline": _pipeline_dict(session),
                }
            s_ref["updated_at"] = time.time()
        # Local-HPC stage-3 can complete in-process before the frontend gets a POST-body snapshot.
        # Guarantee non-empty, parseable state here so both old and new frontends render progress/log.
        if is_inprocess_mitole:
            _res = s_ref.get("result") or {}
            _done_ok = (not bool(s_ref.get("running"))) and bool(_res.get("ok"))
            _log_now = str(s_ref.get("log") or "")
            if not _log_now.strip():
                if _done_ok:
                    _msg = str(_res.get("message") or "Local HPC downloader finished.")
                    _log_now = f"[DONE] {_msg}\n"
                elif bool(s_ref.get("running")):
                    _log_now = "[mito2] Local HPC downloader started...\n"
                if _log_now:
                    s_ref["log"] = _log_now
                    s_ref["updated_at"] = time.time()
            _p = s_ref.get("progress")
            _p_total = 0
            try:
                _p_total = int((_p or {}).get("total") or 0)
            except Exception:
                _p_total = 0
            if _p_total <= 0:
                _inf = _mitole_infer_progress_from_log(str(s_ref.get("log") or ""), _done_ok)
                if _inf is not None:
                    s_ref["progress"] = _inf
                    s_ref["updated_at"] = time.time()
        s = s_ref.copy()
    return {"ok": True, **s}


@router.api_route(
    "/run/downloader-script-state/clear",
    methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"],
    include_in_schema=False,
)
@router.api_route(
    "/run/downloader-script-state/clear/",
    methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"],
    include_in_schema=False,
)
def studio_run_downloader_script_state_clear_any(
    body: StudioDownloaderCancelBody | None = None,
    session_id: str = Query("default"),
) -> dict[str, Any]:
    sid = ((body.session_id if body and body.session_id else session_id) or "default").strip() or "default"
    return _clear_downloader_script_state_for_session(sid)


def _clear_downloader_script_state_for_session(sid: str) -> dict[str, Any]:
    with _downloader_state_lock:
        s = _get_or_init_downloader_state(sid)
        if s.get("running"):
            raise HTTPException(status_code=409, detail="Cannot clear downloader output while a run is active")
        s.update(
            {
                "script_path": "",
                "log": "",
                "progress": None,
                "result": None,
                "updated_at": time.time(),
            }
        )
    return {"ok": True, "cleared": True}


@router.get("/downloader/preview")
def studio_downloader_preview(site: str, data_scope: str = "labeled") -> dict[str, Any]:
    root = _root()
    s = (site or "").strip()
    if not s:
        return {"ok": False, "message": "site required", "datasets": []}
    db_path = _catalog_db_for_site(root, s)
    scope_l = (data_scope or "labeled").strip().lower()
    out: dict[str, Any]
    # Provider-native labeled inventory: keep preview aligned with script generation.
    if scope_l == "labeled":
        try:
            import sys as _sys
            if str(root) not in _sys.path:
                _sys.path.insert(0, str(root))
            from agent.orchestration.registry.providers import get_provider  # noqa: PLC0415
            provider = get_provider(s.lower())
            inv_fn = getattr(provider, "labeled_inventory_stable_ids", None)
            if callable(inv_fn):
                inv = sorted(list(inv_fn()), key=str.lower)
                out = {
                    "ok": True,
                    "message": "ok",
                    "datasets": inv,
                    "dataset_rows": _downloader_sample_rows_for_names(db_path, inv),
                }
            else:
                out = _downloader_preview_from_db(db_path, data_scope=data_scope)
        except Exception:
            out = _downloader_preview_from_db(db_path, data_scope=data_scope)
    else:
        out = _downloader_preview_from_db(db_path, data_scope=data_scope)
    out["site"] = s
    out["data_scope"] = "labeled"
    out["db_path"] = str(db_path)
    out["count"] = len(out.get("datasets") or [])
    return out


@router.get("/pending-downloads")
def studio_pending_downloads(
    site: str = Query(default="openorganelle"),
    n_crops: int = Query(default=1, ge=1, le=256),
    chunk_zyx: str = Query(default="128,128,128"),
    voxel_nm_zyx: str = Query(default="16,16,16"),
    mode: str = Query(default="labeled"),
    foundation: bool = Query(default=True),
) -> dict[str, Any]:
    """Return datasets still missing a complete download for the given profile.

    For OpenOrganelle ``mode=labeled``, only datasets in the same inventory as
    the generated labeled script (``inventory_from_db``) are counted — aligned
    with ``GET /downloader/preview`` for that site.
    """
    root = _root()
    reg_path = root / "data" / "registry.sqlite"
    if not reg_path.is_file():
        return {
            "ok": False,
            "message": "Registry not built yet — run stage 2 first.",
            "pending_count": None,
            "pending_datasets": [],
            "profile_hash": None,
        }
    try:
        import sys as _sys

        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from agent.orchestration.registry.schema import open_registry
        from agent.orchestration.registry.providers import get_provider
        from agent.orchestration.registry.api import make_download_profile_hash
        from agent.orchestration.registry.planners import plan_downloads

        def _parse_xyz(s: str) -> list[float]:
            parts = [float(v.strip()) for v in s.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Expected 3 comma-separated values, got: {s!r}")
            return parts

        chunk_xyz = _parse_xyz(chunk_zyx)
        voxel_xyz = _parse_xyz(voxel_nm_zyx)
        chunk_zyx_t = (int(chunk_xyz[2]), int(chunk_xyz[1]), int(chunk_xyz[0]))
        voxel_zyx_t = (float(voxel_xyz[2]), float(voxel_xyz[1]), float(voxel_xyz[0]))

        mode_s = (mode or "labeled").strip().lower()
        site_s = (site or "openorganelle").strip().lower() or "openorganelle"
        provider = get_provider(site_s)
        _fd = getattr(provider, "download_profile_foundation", None)
        foundation_for_profile = (
            bool(_fd(mode=mode_s, foundation_query=bool(foundation)))
            if callable(_fd)
            else bool(foundation)
        )
        profile_hash = make_download_profile_hash(
            n_crops=n_crops,
            chunk_zyx=chunk_zyx_t,
            voxel_nm_zyx=voxel_zyx_t,
            mode=mode_s,
            foundation=foundation_for_profile,
        )

        conn = open_registry(reg_path)

        allowlist: frozenset[str] | None = None
        if mode_s == "labeled" and callable(getattr(provider, "labeled_inventory_stable_ids", None)):
            allowlist = provider.labeled_inventory_stable_ids()

        pending = plan_downloads(
            conn,
            provider,
            download_profile_hash=profile_hash,
            stable_id_allowlist=allowlist,
        )
        conn.close()

        seen: set[str] = set()
        pending_ds: list[str] = []
        for p in pending:
            if p.stable_dataset_id not in seen:
                seen.add(p.stable_dataset_id)
                pending_ds.append(p.stable_dataset_id)

        # Keep pre-run count aligned with downloader runtime preflight:
        # when files already exist locally (but registry has not adopted them yet),
        # runtime skips those datasets immediately.
        if mode_s == "labeled":
            pending_ds = [
                ds for ds in pending_ds
                if not _dataset_has_complete_labeled_outputs(root, ds, n_crops)
            ]

        return {
            "ok": True,
            "pending_count": len(pending_ds),
            "pending_datasets": pending_ds,
            "profile_hash": profile_hash,
            "profile": {
                "n_crops": n_crops,
                "chunk_zyx": chunk_zyx,
                "voxel_nm_zyx": voxel_nm_zyx,
                "mode": mode_s,
                "foundation": foundation_for_profile,
                "site": site_s,
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "message": str(exc),
            "pending_count": None,
            "pending_datasets": [],
            "profile_hash": None,
        }


@router.get("/downloader/scripts")
def studio_downloader_scripts(
    response: Response,
    site: str = "",
    data_scope: str = "",
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    root = _root()
    out_dir = root / "3data_downloader" / "outputs"
    if not out_dir.is_dir():
        return {"ok": True, "scripts": []}
    # Intentionally list all generated downloader scripts in outputs; UI handles selection.
    files = sorted(out_dir.glob("download_*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    scripts: list[str] = []
    for p in files:
        scripts.append(str(p.relative_to(root)).replace("\\", "/"))
    return {"ok": True, "scripts": scripts}


def _apply_step(session: PipelineSession, step: PipelineStep) -> None:
    session.current_step = step


_ww_mod: Any = None


def _website_workspace_mod() -> Any:
    """Import ``master.website_workspace`` with ``1web_scraper_01`` on ``sys.path`` (relative imports work)."""
    global _ww_mod
    if _ww_mod is None:
        s1 = str(_root() / "1web_scraper_01")
        if s1 not in sys.path:
            sys.path.insert(0, s1)
        import master.website_workspace as ww  # noqa: PLC0415

        _ww_mod = ww
    return _ww_mod


_sc_mod: Any = None


def _scrape_control_mod() -> Any:
    """Import ``master.scrape_control`` (same ``sys.path`` as website workspace)."""
    global _sc_mod
    if _sc_mod is None:
        s1 = str(_root() / "1web_scraper_01")
        if s1 not in sys.path:
            sys.path.insert(0, s1)
        import master.scrape_control as sc  # noqa: PLC0415

        _sc_mod = sc
    return _sc_mod


class StudioWebsiteSaveBody(StudioSessionBody):
    display_name: str = Field(default="", max_length=200)
    url: str = Field(..., min_length=4)
    description: str = Field(default="", max_length=20000)
    data_focus: str = Field(default="", max_length=20000)
    slug: str = Field(default="", max_length=120, description="Optional folder slug hint; new sites get _01, _02, …")
    editing_slug: str = Field(
        default="",
        max_length=120,
        description="Only when UI sends this (overwrite mode): update this folder in place instead of allocating _NN+1.",
    )


class StudioWebsiteDeleteBody(BaseModel):
    slug: str = Field(..., min_length=1, max_length=120)


class StudioWebsiteScrapeBody(StudioSessionBody):
    """Send **slug** only to scrape from saved ``site.md``, or send **url** (+ profile fields) like the form."""

    slug: str = Field(default="", max_length=120)
    url: str = Field(default="", max_length=8000)
    display_name: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=20000)
    data_focus: str = Field(default="", max_length=20000)


class StudioScrapeCancelBody(StudioSessionBody):
    pass


@router.get("/websites")
def studio_list_websites() -> dict[str, Any]:
    ww = _website_workspace_mod()
    root = _root()
    slugs = ww.list_website_slugs(root)
    items = []
    for s in slugs:
        summ = ww.load_website_summary(root, s)
        if summ:
            items.append(summ)
    return {"websites": items}


@router.get("/websites/scrape-state")
def studio_scrape_website_state(session_id: str = Query("default")) -> dict[str, Any]:
    """Live or last-completed scrape log for a Studio session (sync scrape, stream scrape, or chat pipeline).

    NOTE: This must be declared before the dynamic ``/websites/{slug}`` route so Starlette does not
    treat ``scrape-state`` as a website slug.
    """
    sc = _scrape_control_mod()
    sid = (session_id or "default").strip() or "default"
    running, log = sc.session_snapshot(sid)
    return {"running": running, "log": log}


@router.post("/websites/scrape-state/clear")
def studio_scrape_website_state_clear(body: StudioSessionBody) -> dict[str, Any]:
    sid = (body.session_id or "default").strip() or "default"
    sc = _scrape_control_mod()
    running, _ = sc.session_snapshot(sid)
    if running:
        raise HTTPException(status_code=409, detail="Cannot clear scrape output while a run is active")
    sc.session_clear(sid)
    return {"ok": True, "cleared": True}


@router.get("/websites/{slug}")
def studio_get_website(slug: str) -> dict[str, Any]:
    ww = _website_workspace_mod()
    root = _root()
    summ = ww.load_website_summary(root, slug)
    if not summ:
        raise HTTPException(status_code=404, detail="Unknown website slug")
    return summ


@router.post("/websites/save")
def studio_save_website(body: StudioWebsiteSaveBody) -> dict[str, Any]:
    ww = _website_workspace_mod()
    root = _root()
    slug = (body.slug or "").strip() or None
    edit = (body.editing_slug or "").strip() or None
    out = ww.save_website_meta_only(
        root,
        display_name=body.display_name,
        url=body.url.strip(),
        description=body.description,
        data_focus=body.data_focus,
        slug_override=slug,
        editing_slug=edit,
    )
    return {"ok": True, **out}


@router.delete("/websites/{slug}")
def studio_delete_website(slug: str) -> dict[str, Any]:
    ww = _website_workspace_mod()
    root = _root()
    out = ww.delete_website_workspace(root, slug)
    if not out.get("ok"):
        if out.get("error") == "not_found":
            raise HTTPException(status_code=404, detail="Unknown website slug")
        if out.get("error") == "protected_slug":
            raise HTTPException(
                status_code=403,
                detail=str(out.get("message", "This workspace cannot be deleted.")),
            )
        raise HTTPException(status_code=400, detail=str(out.get("error", "delete_failed")))
    return {"ok": True, **out}


@router.post("/websites/delete")
def studio_delete_website_post(body: StudioWebsiteDeleteBody) -> dict[str, Any]:
    """Same as ``DELETE /websites/{slug}`` — POST avoids clients and proxies that block DELETE."""
    ww = _website_workspace_mod()
    root = _root()
    slug = (body.slug or "").strip()
    if not slug:
        raise HTTPException(status_code=422, detail="slug is required")
    out = ww.delete_website_workspace(root, slug)
    if not out.get("ok"):
        err = str(out.get("error", "delete_failed"))
        if err == "not_found":
            raise HTTPException(status_code=404, detail="Unknown website slug")
        if err == "protected_slug":
            raise HTTPException(
                status_code=403,
                detail=str(out.get("message", "This workspace cannot be deleted.")),
            )
        if err == "remove_failed":
            raise HTTPException(status_code=500, detail=str(out.get("message", err)))
        raise HTTPException(status_code=400, detail=err)
    return {"ok": True, **out, "project_root": str(root.resolve())}


@router.post("/websites/scrape")
def studio_scrape_website(body: StudioWebsiteScrapeBody) -> dict[str, Any]:
    ww = _website_workspace_mod()
    root = _root()
    sid = (body.session_id or "default").strip() or "default"
    session = _session(sid)
    llm_sub = build_scrape_subprocess_llm_env()
    sc = _scrape_control_mod()
    cancel_ev = threading.Event()
    sc.session_start(sid, cancel_ev)

    url_in = (body.url or "").strip()
    slug_in = (body.slug or "").strip()
    target_label = slug_in or url_in or "(unspecified)"
    # Ensure the Studio "Scrape output" pane shows activity even when the underlying
    # workspace scraper emits minimal log lines (chat/agent sync runs rely on this).
    sc.session_log_append(sid, f"[mito2] Starting scrape: {target_label}\n")

    def _log_line(t: str) -> None:
        sc.session_log_append(sid, t)

    try:

        if url_in:
            out = ww.run_website_workspace_scrape(
                root,
                display_name=body.display_name,
                url=url_in,
                description=body.description,
                data_focus=body.data_focus,
                slug_override=slug_in or None,
                write_probe_legacy=True,
                log_line=_log_line,
                cancel_event=cancel_ev,
                scrape_session_id=sid,
                subprocess_llm_env=llm_sub,
            )
        elif slug_in:
            summ = ww.load_website_summary(root, slug_in)
            if not summ:
                raise HTTPException(status_code=404, detail="Unknown website slug")
            url_disk = (summ.get("url") or "").strip()
            if not url_disk:
                raise HTTPException(
                    status_code=400,
                    detail="Saved site has no URL in site.md — set URL in Site profile, then Save.",
                )
            out = ww.run_website_workspace_scrape(
                root,
                display_name=(summ.get("display_name") or "").strip() or slug_in,
                url=url_disk,
                description=summ.get("description") or "",
                data_focus=summ.get("data_focus") or "",
                slug_override=summ["slug"],
                write_probe_legacy=True,
                log_line=_log_line,
                cancel_event=cancel_ev,
                scrape_session_id=sid,
                subprocess_llm_env=llm_sub,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide slug (scrape from saved site) or url (scrape from form).",
            )
        if out.get("ok"):
            fu = ""
            if isinstance(out.get("fetch"), dict):
                fu = str(out["fetch"].get("url") or "").strip()
            if not fu:
                fu = url_in
            if not fu and slug_in:
                s = ww.load_website_summary(root, slug_in)
                if s:
                    fu = (s.get("url") or "").strip()
            update_session_after_scrape(root, session, fu)

        full = {
            **out,
            "pipeline": _pipeline_dict(session),
        }
        full["message"] = _scrape_result_message(full)
        sc.session_log_append(sid, f"[mito2] Scrape finished: ok={bool(full.get('ok'))} target={target_label}\n")
        return full
    finally:
        sc.session_end(sid)


@router.post("/websites/scrape-stream")
async def studio_scrape_website_stream(body: StudioWebsiteScrapeBody) -> StreamingResponse:
    """Same as ``POST /websites/scrape`` but streams merged subprocess stdout as SSE ``log`` events, then ``done`` or ``error``."""

    root = _root()
    q: queue.Queue[tuple[str, Any]] = queue.Queue()
    sid = (body.session_id or "default").strip() or "default"

    def worker() -> None:
        sc = _scrape_control_mod()
        cancel_ev = threading.Event()
        sc.session_start(sid, cancel_ev)
        try:
            ww = _website_workspace_mod()
            session = _session(sid)
            url_in = (body.url or "").strip()
            slug_in = (body.slug or "").strip()
            target_label = slug_in or url_in or "(unspecified)"
            sc.session_log_append(sid, f"[mito2] Starting scrape: {target_label}\n")
            llm_sub = build_scrape_subprocess_llm_env()
            try:

                def log_cb(t: str) -> None:
                    q.put(("log", t))
                    sc.session_log_append(sid, t)

                if url_in:
                    out = ww.run_website_workspace_scrape(
                        root,
                        display_name=body.display_name,
                        url=url_in,
                        description=body.description,
                        data_focus=body.data_focus,
                        slug_override=slug_in or None,
                        write_probe_legacy=True,
                        log_line=log_cb,
                        cancel_event=cancel_ev,
                        scrape_session_id=sid,
                        subprocess_llm_env=llm_sub,
                    )
                elif slug_in:
                    summ = ww.load_website_summary(root, slug_in)
                    if not summ:
                        q.put(("error", "Unknown website slug"))
                        return
                    url_disk = (summ.get("url") or "").strip()
                    if not url_disk:
                        q.put(
                            (
                                "error",
                                "Saved site has no URL in site.md — set URL in Site profile, then Save.",
                            ),
                        )
                        return
                    out = ww.run_website_workspace_scrape(
                        root,
                        display_name=(summ.get("display_name") or "").strip() or slug_in,
                        url=url_disk,
                        description=summ.get("description") or "",
                        data_focus=summ.get("data_focus") or "",
                        slug_override=summ["slug"],
                        write_probe_legacy=True,
                        log_line=log_cb,
                        cancel_event=cancel_ev,
                        scrape_session_id=sid,
                        subprocess_llm_env=llm_sub,
                    )
                else:
                    q.put(("error", "Provide slug (scrape from saved site) or url (scrape from form)."))
                    return

                if cancel_ev.is_set():
                    out = {**out, "ok": False, "cancelled": True}
                    full: dict[str, Any] = {
                        **out,
                        "pipeline": _pipeline_dict(session),
                    }
                    q.put(("done", full))
                    return

                if out.get("ok"):
                    fu = ""
                    if isinstance(out.get("fetch"), dict):
                        fu = str(out["fetch"].get("url") or "").strip()
                    if not fu:
                        fu = url_in
                    if not fu and slug_in:
                        s = ww.load_website_summary(root, slug_in)
                        if s:
                            fu = (s.get("url") or "").strip()
                    update_session_after_scrape(root, session, fu)

                full = {
                    **out,
                    "pipeline": _pipeline_dict(session),
                }
                sc.session_log_append(
                    sid,
                    f"[mito2] Scrape finished: ok={bool(full.get('ok'))} target={target_label}\n",
                )
                q.put(("done", full))
            except Exception as e:  # noqa: BLE001
                q.put(("error", str(e)))
        finally:
            sc.session_end(sid)

    threading.Thread(target=worker, daemon=True).start()

    async def event_gen():
        while True:
            kind, data = await asyncio.to_thread(q.get)
            if kind == "log":
                yield f"data: {json.dumps({'type': 'log', 'text': data}, ensure_ascii=False)}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': str(data)}, ensure_ascii=False)}\n\n"
                break
            elif kind == "done":
                yield f"data: {json.dumps({'type': 'done', 'payload': data}, ensure_ascii=False, default=str)}\n\n"
                break

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/websites/scrape-cancel")
def studio_scrape_cancel(body: StudioScrapeCancelBody) -> dict[str, Any]:
    """Signal cancel and kill the OpenOrganelle agent subprocess for this Studio session (if any)."""
    sc = _scrape_control_mod()
    sid = (body.session_id or "default").strip() or "default"
    killed = sc.session_kill(sid)
    return {"ok": True, "killed": killed}


@router.get("/presets/scrape")
def studio_scrape_presets() -> dict[str, Any]:
    return {"presets": SCRAPE_PRESETS}


@router.get("/probes")
def studio_list_probes() -> dict[str, Any]:
    root = _root()
    return {"probes": _list_probe_rel_paths(root)}


@router.get("/sites")
def studio_list_sites() -> dict[str, Any]:
    root = _root()
    return {"sites": _site_stems_from_probes(root)}


@router.get("/summary")
def studio_summary(session_id: str = "default") -> dict[str, Any]:
    root = _root()
    s2_out = root / "2database_builder" / "outputs"
    session = _session(session_id or "default")
    probes = _list_probe_rel_paths(root)
    inv = root / "data" / "inventory.sqlite"
    reg = root / "data" / "registry.sqlite"
    # Find the first catalog DB (prefer OpenOrganelle.db for backward compat; otherwise
    # pick any *.db that is not the registry or inventory meta-databases).
    _db_dir = s2_out / "databases"
    _meta_dbs = {"registry.sqlite", "inventory.sqlite"}
    _catalog_db_candidates = (
        sorted(p for p in _db_dir.glob("*.db") if p.name not in _meta_dbs)
        if _db_dir.is_dir() else []
    )
    _oo_db = _db_dir / "OpenOrganelle.db"
    catalog_db = _oo_db if _oo_db.is_file() else (_catalog_db_candidates[0] if _catalog_db_candidates else _oo_db)
    gen = root / "3data_downloader" / "outputs"
    n_gen = len(list(gen.glob("download_*.py"))) if gen.is_dir() else 0
    dataset_mito2 = nnunet_dataset_root(root)
    legacy_pre = root / "data" / "training"
    training_cfg = dataset_mito2 / "finetune_datalist.json"
    if not training_cfg.is_file():
        training_cfg = legacy_pre / "finetune_datalist.json"

    registry_stats: dict[str, Any] = {"exists": False}
    if reg.is_file():
        try:
            import sqlite3 as _sq3
            rc = _sq3.connect(str(reg))
            rc.row_factory = _sq3.Row
            n_providers = rc.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
            n_datasets  = rc.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
            n_assets    = rc.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
            n_dl_done   = rc.execute("SELECT COUNT(*) FROM downloads WHERE status='complete'").fetchone()[0]
            n_pp_done   = rc.execute("SELECT COUNT(*) FROM preprocess_runs WHERE status='complete'").fetchone()[0]
            rc.close()
            registry_stats = {
                "exists": True,
                "providers": n_providers,
                "datasets": n_datasets,
                "assets": n_assets,
                "complete_downloads": n_dl_done,
                "complete_preprocess_runs": n_pp_done,
            }
        except Exception:
            registry_stats = {"exists": True, "error": "could not read registry"}

    labeled_ready = 0
    if catalog_db.is_file():
        try:
            import sqlite3 as _sq3
            cc = _sq3.connect(str(catalog_db))
            labeled_ready = cc.execute(
                "SELECT COUNT(*) FROM dataset_resolved WHERE ready_labeled=1"
            ).fetchone()[0]
            cc.close()
        except Exception:
            pass

    return {
        "pipeline": _pipeline_dict(session),
        "probe_count": len(probes),
        "latest_probe": probes[0] if probes else "",
        "inventory_sqlite_exists": inv.is_file(),
        "catalog_db_labeled_ready": labeled_ready,
        "generated_download_scripts": n_gen,
        "preprocessed_dir_exists": dataset_mito2.is_dir() and any(dataset_mito2.iterdir()),
        "training_config_exists": training_cfg.is_file(),
        "registry": registry_stats,
    }


@router.post("/run/scrape")
def studio_run_scrape(body: StudioScrapeBody) -> dict[str, Any]:
    root = _root()
    sid = body.session_id or "default"
    session = _session(sid)
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    s1 = root / "1web_scraper_01"
    if not (s1 / "agent.py").is_file():
        raise HTTPException(status_code=500, detail="Stage 1 agent.py missing")
    py = sys.executable
    result = run_command([py, "agent.py", url], cwd=s1, timeout_sec=180.0)
    ok = result["returncode"] == 0
    if ok:
        update_session_after_scrape(root, session, url)
    if not ok:
        _emit_run_failure_to_terminal("stage1-scrape", result)
    return {
        "ok": ok,
        "message": _studio_run_message(ok, "Scrape finished.", "Scrape failed.", result),
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "pipeline": _pipeline_dict(session),
    }


@router.get("/run/database-state")
def studio_run_database_state(session_id: str = Query("default")) -> dict[str, Any]:
    sid = (session_id or "default").strip() or "default"
    with _database_build_state_lock:
        row = _database_build_state.get(sid)
        if not row:
            return {"running": False, "log": "", "result": None}
        return {
            "running": bool(row.get("running")),
            "log": str(row.get("log") or ""),
            "result": row.get("result"),
        }


@router.post("/run/database-state/clear")
def studio_run_database_state_clear(body: StudioSessionBody) -> dict[str, Any]:
    sid = (body.session_id or "default").strip() or "default"
    with _database_build_state_lock:
        row = _database_build_state.get(sid)
        if row and bool(row.get("running")):
            raise HTTPException(status_code=409, detail="Cannot clear database build output while a run is active")
        _database_build_state[sid] = {
            "running": False,
            "log": "",
            "result": None,
            "updated_at": time.time(),
        }
    return {"ok": True, "cleared": True}


@router.post("/run/database")
def studio_run_database_build(body: StudioDatabaseBuildBody) -> dict[str, Any]:
    root = _root()
    sid = (body.session_id or "default").strip() or "default"
    session = _session(sid)
    s2 = root / "2database_builder"
    if not (s2 / "agent.py").is_file():
        raise HTTPException(status_code=500, detail="Stage 2 agent.py missing")
    _database_build_state_begin(sid)
    try:
        py = sys.executable
        argv = [py, "agent.py"]
        probe = (body.probe or "").strip()
        if probe:
            argv.extend(["--probe", probe])
            site_stem = _site_stem_from_probe_arg(probe)
            # Keep stage-2 provider-scoped when the user picks a specific probe JSON.
            # Without this, ``agent.py`` builds all ``*.md`` catalogs and logs can look
            # inconsistent (e.g. BossDB selected but OpenOrganelle probe activity shown).
            if site_stem:
                argv.extend(["--catalog-sites", site_stem])
        # Registry sync is on by default (agent.py reads MITO2_DISABLE_REGISTRY).
        # Pass --no-registry only when the env var explicitly disables it, so Studio
        # never needs its own toggle — the environment variable is the single opt-out.
        result = _run_database_build_command_live(
            sid,
            argv,
            cwd=s2,
            timeout_sec=_database_build_timeout_sec(),
        )
        ok = result["returncode"] == 0
        if ok:
            _apply_step(session, PipelineStep.DATABASE)
        if not ok:
            _emit_run_failure_to_terminal("stage2-database", result)
        payload = {
            "ok": ok,
            "message": _studio_run_message(
                ok, "Database / inventory updated.", "Database build failed.", result
            ),
            "returncode": result["returncode"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "pipeline": _pipeline_dict(session),
        }
        _database_build_state_finish(sid, payload)
        return payload
    except HTTPException:
        _database_build_state_fail(sid, "HTTP error during database build")
        raise
    except Exception as exc:  # noqa: BLE001
        _database_build_state_fail(sid, str(exc))
        raise
    finally:
        with _database_build_state_lock:
            row = _database_build_state.get(sid)
            if row and row.get("running"):
                row["running"] = False
                row["updated_at"] = time.time()


@router.post("/run/downloader")
def studio_run_downloader(body: StudioDownloaderBody) -> dict[str, Any]:
    root = _root()
    sid = (body.session_id or "default").strip() or "default"
    session = _session(sid)
    # Stage 3 UX is one-step Download (generate + execute).
    body.execute = True
    s3 = root / "3data_downloader"
    if not (s3 / "downloader_master/agent.py").is_file():
        raise HTTPException(status_code=500, detail="Stage 3 downloader_master/agent.py missing")
    try:
        return _studio_run_downloader_impl(body, root, sid, session, s3)
    except BaseException:
        with _downloader_state_lock:
            st = _downloader_state.get(sid)
            if st and st.get("running"):
                st["running"] = False
                st["updated_at"] = time.time()
        raise


@router.post("/mitole/run/downloader")
def studio_run_mitole_downloader(body: StudioMitoLeDownloaderBody) -> dict[str, Any]:
    root = _root()
    sid = (body.session_id or "default").strip() or "default"
    selected_pairs = []
    dataset_splits = dict(body.dataset_splits or {})
    for p in body.dataset_pairs:
        selected_pairs.append(
            {
                "dataset": str(p.dataset or "").strip(),
                "source": str(p.source or "").strip(),
                "image_path": str(p.image_path or "").strip(),
                "label_path": str(p.label_path or "").strip(),
            }
        )
    if not selected_pairs:
        return {
            "ok": False,
            "message": "No dataset pairs were provided from Local HPC Stage 2.",
            "returncode": 1,
            "stdout": "",
            "stderr": "dataset_pairs is empty",
        }
    submitted_pairs_snapshot = [dict(p) for p in selected_pairs]
    filtered_pairs: list[dict[str, str]] = []
    skipped_incremental: list[str] = []
    for p in selected_pairs:
        ds = str(p.get("dataset") or "").strip()
        if not ds:
            continue
        requested_total = _mitole_dataset_total_crops(dataset_splits, ds)
        if requested_total <= 0:
            continue
        stable = _normalize_stable_id(ds)
        split_cfg = (dataset_splits or {}).get(ds, {}) or {}
        try:
            tr_req = max(0, int(split_cfg.get("training", 0) or 0))
            ts_req = max(0, int(split_cfg.get("inference", 0) or 0))
        except Exception:
            tr_req = ts_req = 0
        pt_ds, _rq_ds = mitole_pending_and_requested_crop_counts(root, [p], dataset_splits)
        if int(pt_ds) <= 0:
            tr_disk, ts_disk = _mitole_disk_split_counts(root, stable)
            skipped_incremental.append(
                f"{ds}: global foundation layout satisfied for training={tr_req} inference={ts_req} "
                f"(on-disk training files={tr_disk}, inference files={ts_disk}; "
                f"total nnUNet units={_count_dataset_units_on_disk(root, stable)}/{requested_total}); skipping dataset"
            )
            continue
        filtered_pairs.append(p)
    selected_pairs = filtered_pairs

    if not selected_pairs:
        msg = "All selected Local-HPC datasets already satisfy requested train/inference crop counts; nothing new to download."
        noop_log, noop_prog = _mitole_noop_downloader_log_and_progress(
            root=root,
            pairs_for_counts=submitted_pairs_snapshot,
            dataset_splits=dataset_splits,
            skipped_incremental=skipped_incremental,
        )
        with _downloader_state_lock:
            _downloader_state[sid] = {
                "running": False,
                "script_path": "[pipeline] mitole_stage3",
                "log": noop_log,
                "progress": noop_prog,
                "result": {"ok": True, "message": msg, "returncode": 0, "stdout": "", "stderr": ""},
                "updated_at": time.time(),
            }
            snap = dict(_downloader_state[sid])
        return {
            "ok": True,
            "message": msg,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "downloader_log": str(snap.get("log") or ""),
            "downloader_progress": snap.get("progress"),
        }

    pending_total, req_eligible = mitole_pending_and_requested_crop_counts(root, selected_pairs, dataset_splits)
    if pending_total <= 0:
        msg = "Every requested crop file already exists under Dataset001_mito2; nothing new to materialize."
        noop_log, noop_prog = _mitole_noop_downloader_log_and_progress(
            root=root,
            pairs_for_counts=selected_pairs,
            dataset_splits=dataset_splits,
            skipped_incremental=skipped_incremental,
        )
        with _downloader_state_lock:
            _downloader_state[sid] = {
                "running": False,
                "script_path": "[pipeline] mitole_stage3",
                "log": noop_log + f"[INFO] {msg}\n",
                "progress": noop_prog,
                "result": {"ok": True, "message": msg, "returncode": 0, "stdout": "", "stderr": ""},
                "updated_at": time.time(),
            }
            snap = dict(_downloader_state[sid])
        return {
            "ok": True,
            "message": msg,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "downloader_log": str(snap.get("log") or ""),
            "downloader_progress": snap.get("progress"),
        }

    prog_total = max(1, int(pending_total))
    already_mat = max(0, int(req_eligible) - int(pending_total))
    with _downloader_state_lock:
        _downloader_state[sid] = {
            "running": True,
            "script_path": "[pipeline] mitole_stage3",
            "log": (
                "[mito2] Local HPC downloader started...\n"
                # Stable substring consumed by PipelineStudio ``parseDownloaderProgress`` for mitole.
                f"[INFO] new crops to write={int(pending_total)} (progress bar denominator only)\n"
                f"[INFO] source=local-hpc crop plan (same idea as web Stage 3 pending check): "
                f"{req_eligible} eligible crop slot(s) from your selections; "
                f"{already_mat} already materialized on disk under Dataset001_mito2; "
                f"{pending_total} new to write this run (progress bar uses only the last count).\n"
            ),
            "progress": {"completed": 0, "total": int(prog_total), "current": 1, "dataset": ""},
            "result": None,
            "updated_at": time.time(),
        }
    with _downloader_kill_requested_lock:
        _downloader_kill_requested.discard(sid)
    def _log(msg: str) -> None:
        with _downloader_state_lock:
            st = _downloader_state.get(sid) or {}
            st["log"] = str(st.get("log") or "") + str(msg)
            st["updated_at"] = time.time()
            _downloader_state[sid] = st
    def _progress(completed: int, total: int, current: int, dataset: str) -> None:
        with _downloader_state_lock:
            st = _downloader_state.get(sid) or {}
            st["progress"] = {
                "completed": int(completed),
                "total": int(total),
                "current": int(current),
                "dataset": str(dataset or ""),
            }
            st["updated_at"] = time.time()
            _downloader_state[sid] = st
    def _cancel_requested() -> bool:
        with _downloader_kill_requested_lock:
            return sid in _downloader_kill_requested
    try:
        manifest = mitole_stage3_copy_selected_pairs(
            root,
            selected_pairs,
            dataset_splits,
            log_fn=_log,
            progress_fn=_progress,
            cancel_requested_fn=_cancel_requested,
        )
    except Exception as exc:
        with _downloader_kill_requested_lock:
            was_cancelled = sid in _downloader_kill_requested
            _downloader_kill_requested.discard(sid)
        err_msg = "Local HPC download cancelled." if was_cancelled else f"Local HPC download failed: {exc}"
        with _downloader_state_lock:
            _downloader_state[sid] = {
                "running": False,
                "script_path": "[pipeline] mitole_stage3",
                "log": str((_downloader_state.get(sid) or {}).get("log") or "")
                + ("[mito2] Download kill requested.\n" if was_cancelled else f"[ERR] {exc}\n"),
                "progress": None,
                "result": {
                    "ok": False,
                    "message": err_msg,
                    "returncode": 130 if was_cancelled else 1,
                    "stdout": "",
                    "stderr": "cancelled" if was_cancelled else str(exc),
                },
                "updated_at": time.time(),
            }
            snap = dict(_downloader_state[sid])
        return {
            "ok": False,
            "message": err_msg,
            "returncode": 130 if was_cancelled else 1,
            "stdout": "",
            "stderr": "cancelled" if was_cancelled else str(exc),
            "downloader_log": str(snap.get("log") or ""),
            "downloader_progress": snap.get("progress"),
        }
    with _downloader_kill_requested_lock:
        _downloader_kill_requested.discard(sid)
    copied = int(manifest.get("copied_pairs", 0) or 0)
    skipped = list(manifest.get("skipped", []) or []) + skipped_incremental
    tr_units = int(manifest.get("nnunet_training_units", 0) or 0)
    ts_units = int(manifest.get("nnunet_inference_units", 0) or 0)
    msg = (
        f"Copied {copied} dataset pair(s) from Local HPC sources and materialized "
        f"{tr_units} training + {ts_units} inference unit(s) under Dataset001_mito2."
    )
    if skipped:
        msg += f" Skipped {len(skipped)} pair(s)."
    result_payload = {
        "ok": True,
        "message": msg,
        "returncode": 0,
        "stdout": json.dumps(manifest, indent=2),
        "stderr": "",
        "run_folder": str(manifest.get("run_dir") or ""),
        "copied_pairs": copied,
    }
    try:
        _register_mitole_manifest_into_registry(
            project_root=root,
            manifest=manifest,
            dataset_splits=dataset_splits,
        )
    except Exception as reg_exc:
        # Non-fatal for download success; include a visible warning in logs.
        with _downloader_state_lock:
            st_reg = _downloader_state.get(sid) or {}
            st_reg["log"] = str(st_reg.get("log") or "") + f"[WARN] Inventory registry sync failed: {reg_exc}\n"
            st_reg["updated_at"] = time.time()
            _downloader_state[sid] = st_reg
    with _downloader_state_lock:
        st_log = str((_downloader_state.get(sid) or {}).get("log") or "")
        final_prog = {
            "completed": int(tr_units + ts_units),
            "total": max(1, int(tr_units + ts_units)),
            "current": max(1, int(tr_units + ts_units)),
            "dataset": "done",
        }
        _downloader_state[sid] = {
            "running": False,
            "script_path": "[pipeline] mitole_stage3",
            "log": st_log + f"[DONE] {msg}\n",
            "progress": final_prog,
            "result": result_payload,
            "updated_at": time.time(),
        }
        out_log = str(_downloader_state[sid].get("log") or "")
    result_payload["downloader_log"] = out_log
    result_payload["downloader_progress"] = final_prog
    return result_payload


def _studio_run_downloader_impl(
    body: StudioDownloaderBody,
    root: Path,
    sid: str,
    session: PipelineSession,
    s3: Path,
) -> dict[str, Any]:
    def _self_heal_downloader_running_flag() -> None:
        """Clear stale downloader running state when no live process exists."""
        with _downloader_procs_lock:
            proc = _downloader_procs.get(sid)
        proc_alive = False
        if proc is not None:
            try:
                proc_alive = proc.poll() is None
            except Exception:
                proc_alive = False
        if proc_alive:
            return
        with _downloader_state_lock:
            st = _downloader_state.get(sid)
            if st and st.get("running"):
                st["running"] = False
                st["updated_at"] = time.time()

    py = sys.executable
    site = body.site.strip()
    # Fully supervised: only datasets with good mitochondria labels (non-prediction masks).
    scope = "labeled"
    site_l = site.lower()
    generated_scripts: list[str] = []
    gen_out_dir = (root / "3data_downloader" / "outputs").resolve()
    gen_start_ts = time.time()
    planned_total = _planned_crop_pairs_from_splits(body.dataset_splits)
    start_headline = (
        "[mito2] Web downloader started...\n"
        f"[INFO] source={site_l or 'unknown'} planned crop pairs={planned_total if planned_total > 0 else 'auto'}\n"
    )

    # Attempt to dispatch to a registered provider that supports native script generation.
    # Falls back to the generic volume-stub generator for unknown providers.
    _provider = None
    try:
        import sys as _sys
        _reg_root = root
        if str(_reg_root) not in _sys.path:
            _sys.path.insert(0, str(_reg_root))
        from agent.orchestration.registry.providers import get_provider as _get_provider
        _provider = _get_provider(site_l)
    except (KeyError, Exception):
        _provider = None

    provider_generations: list[dict[str, Any]] = []

    if _provider is not None and hasattr(_provider, "generate_downloader_script"):
        if not body.execute:
            _self_heal_downloader_running_flag()
        # Provider-native path: OpenOrganelle (and future providers) use real script generation.
        modes = ["labeled"]
        out_dir = s3 / "outputs"
        # Remove legacy stub duplicates so there is one canonical script per mode.
        for mode in modes:
            legacy = out_dir / f"download_OpenOrganelle_{mode}.py"
            if legacy.is_file():
                legacy.unlink()
        runs: list[dict[str, Any]] = []
        crop_raw = (body.crop_dimensions_voxels.strip() or "128,128,128").split(",")
        if len(crop_raw) != 3:
            raise HTTPException(status_code=400, detail="crop_dimensions_voxels must be x,y,z")
        try:
            cx, cy, cz = (int(float(x.strip())) for x in crop_raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Bad crop_dimensions_voxels: {body.crop_dimensions_voxels}") from exc
        chunk_zyx = f"{cz},{cy},{cx}"
        vox_raw = (body.voxel_size_nm.strip() or "16,16,16").split(",")
        if len(vox_raw) != 3:
            raise HTTPException(status_code=400, detail="voxel_size_nm must be x,y,z")
        try:
            vx, vy, vz = (float(x.strip()) for x in vox_raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Bad voxel_size_nm: {body.voxel_size_nm}") from exc
        voxel_zyx = f"{vz},{vy},{vx}"
        physical_zyx = f"{vz * cz},{vy * cy},{vx * cx}"
        use_no_foundation = False
        _fd = getattr(_provider, "download_profile_foundation", None)
        foundation_for_profile = (
            bool(_fd(mode=scope, foundation_query=not use_no_foundation))
            if callable(_fd)
            else ((not use_no_foundation) if site_l == "openorganelle" else False)
        )

        # ── Incremental guard: return no-op if all assets already downloaded ──────
        # NOTE: With per-dataset split crops, each dataset side can use a different
        # n_crops/profile hash. The generated script performs split-aware preflight,
        # so skip this coarse global guard when explicit splits are provided.
        if not body.dataset_splits:
            try:
                from agent.orchestration.registry.schema import open_registry as _open_reg
                from agent.orchestration.registry.api import make_download_profile_hash as _make_phash
                from agent.orchestration.registry.planners import plan_downloads as _plan_dl
                _reg_path = _registry_path(root)
                if _reg_path.is_file():
                    _voxel_zyx_t = (vz, vy, vx)
                    _chunk_zyx_t = (cz, cy, cx)
                    _profile_hash = _make_phash(
                        n_crops=max(1, int(body.n_crops)),
                        chunk_zyx=_chunk_zyx_t,
                        voxel_nm_zyx=_voxel_zyx_t,
                        mode=scope,
                        foundation=foundation_for_profile,
                    )
                    _reg_conn = _open_reg(_reg_path)
                    _allowlist: frozenset[str] | None = None
                    if scope == "labeled" and callable(
                        getattr(_provider, "labeled_inventory_stable_ids", None)
                    ):
                        _allowlist = _provider.labeled_inventory_stable_ids()
                    _pending = _plan_dl(
                        _reg_conn, _provider,
                        download_profile_hash=_profile_hash,
                        stable_id_allowlist=_allowlist,
                    )
                    _n_pending = len({p.stable_dataset_id for p in _pending})
                    _n_total = len(_allowlist) if _allowlist is not None else _n_pending
                    _reg_conn.close()
                    logging.info(
                        "[stage3] plan_downloads: %d pending of %d total (profile=%s)",
                        _n_pending, _n_total, _profile_hash,
                    )
                    if body.execute and not _pending:
                        _n_win = max(1, int(body.n_crops))
                        _n_pairs_total = int(_n_total) * int(_n_win)
                        _noop_msg = (
                            f"No new assets to download — all {_n_pairs_total} crop pair(s) are already "
                            f"complete for this profile (n_crops={body.n_crops}, "
                            f"profile={_profile_hash})."
                        )
                        _apply_step(session, PipelineStep.DOWNLOAD_SCRIPT)
                        return {
                            "ok": True,
                            "message": _noop_msg,
                            "returncode": 0,
                            "stdout": _noop_msg,
                            "stderr": "",
                            "pipeline": _pipeline_dict(session),
                            "generated_scripts": [],
                            "noop": True,
                            "pending_count": 0,
                        }
            except Exception as _guard_exc:
                logging.debug("plan_downloads guard failed (proceeding): %s", _guard_exc)
        # ── End incremental guard ──────────────────────────────────────────────────

        # Get provider-specific subprocess env vars (e.g. MITO2_STUDIO_OG_* for OpenOrganelle).
        _env_fn = getattr(_provider, "studio_env_vars", None)
        provider_env: dict[str, str] = (
            _env_fn(
                chunk_zyx=chunk_zyx,
                n_crops=max(1, int(body.n_crops)),
                voxel_size_nm_zyx=voxel_zyx,
                physical_zyx=physical_zyx,
                no_foundation=use_no_foundation,
            )
            if callable(_env_fn) else {}
        )
        if not body.execute:
            # Generate scripts in-process so the Web UI always matches (no subprocess drift).
            for mode in modes:
                argv = [
                    py, "downloader_master/agent.py", "--site", site, "--",
                    "--mode", mode,
                    "--n-crops", str(max(1, int(body.n_crops))),
                    "--chunk", chunk_zyx,
                    "--voxel-size-nm", voxel_zyx,
                ]
                if use_no_foundation:
                    argv.append("--no-foundation")
                provider_generations.append(
                    {"mode": mode, "argv": argv, "env": provider_env, "method": "in_process"}
                )
                try:
                    path = _provider.generate_downloader_script(
                        mode=mode,
                        chunk_zyx=chunk_zyx,
                        n_crops=max(1, int(body.n_crops)),
                        voxel_size_nm_zyx=voxel_zyx,
                        dataset_splits=body.dataset_splits,
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"{site} script generation failed ({mode}): {exc}",
                    ) from exc
                msg = (
                    f"[OK] Generated downloader script: {path}\n"
                    f"Run with: python {path} --dataset <name> --window-policy center "
                    f"(default: foundation uses script chunk/spacing/n-crops settings; "
                    f"add --no-foundation for legacy raw crop)\n"
                    "Generate-only mode complete. No data downloaded.\n"
                )
                try:
                    generated_scripts.append(str(Path(path).resolve().relative_to(root)).replace("\\", "/"))
                except Exception:
                    pass
                runs.append({"returncode": 0, "stdout": msg, "stderr": ""})
            ok = True
            stdout = "\n\n".join((r.get("stdout") or "").strip() for r in runs if (r.get("stdout") or "").strip())
            stderr = ""
            returncode = 0
            result = {"returncode": returncode, "stdout": stdout, "stderr": stderr}
        else:
            # Ensure provider script artifacts exist before execute-only modes.
            # BossDB's site agent expects generated outputs/download_*.py to exist.
            pre_generated_scripts: list[str] = []
            script_by_mode: dict[str, str] = {}
            for mode in modes:
                try:
                    path = _provider.generate_downloader_script(
                        mode=mode,
                        chunk_zyx=chunk_zyx,
                        n_crops=max(1, int(body.n_crops)),
                        voxel_size_nm_zyx=voxel_zyx,
                        dataset_splits=body.dataset_splits,
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"{site} script generation failed ({mode}): {exc}",
                    ) from exc
                if path:
                    try:
                        script_by_mode[mode] = str(Path(path).resolve())
                    except Exception:
                        script_by_mode[mode] = str(path)
                    try:
                        pre_generated_scripts.append(
                            str(Path(path).resolve().relative_to(root)).replace("\\", "/")
                        )
                    except Exception:
                        pass
            if pre_generated_scripts:
                generated_scripts = sorted(
                    set(generated_scripts + pre_generated_scripts), key=str.lower
                )
            primary_script = pre_generated_scripts[0] if pre_generated_scripts else ""
            _downloader_sync_execute_begin(
                sid,
                start_headline,
                script_path=primary_script,
                planned_total=planned_total,
            )
            for mode in modes:
                script_abs = script_by_mode.get(mode, "")
                if not script_abs:
                    raise HTTPException(
                        status_code=500,
                        detail=f"{site} execute failed: missing generated script for mode '{mode}'",
                    )
                argv = [py, "-u", script_abs]
                if use_no_foundation:
                    argv.append("--no-foundation")
                provider_generations.append(
                    {"mode": mode, "argv": argv, "env": None, "method": "subprocess"}
                )
                runs.append(
                    _run_downloader_command_live(
                        sid,
                        argv,
                        cwd=s3,
                        timeout_sec=_downloader_timeout_sec(),
                        env=None,
                    )
                )
            ok = all(r.get("returncode") == 0 for r in runs)
            stdout = "\n\n".join((r.get("stdout") or "").strip() for r in runs if (r.get("stdout") or "").strip())
            stderr = "\n\n".join((r.get("stderr") or "").strip() for r in runs if (r.get("stderr") or "").strip())
            returncode = 0 if ok else next((int(r.get("returncode", 1)) for r in runs if int(r.get("returncode", 1)) != 0), 1)
            result = {"returncode": returncode, "stdout": stdout, "stderr": stderr}
        suffix = f" (mode={scope})"
    else:
        # Generic / unknown providers use the volume-stub script generator.
        argv = [
            py, "downloader_master/agent.py", "--volume-stub", "--site", site,
            "--n-crops", str(body.n_crops),
            "--voxel-size-nm", body.voxel_size_nm.strip() or "16,16,16",
            "--crop-dims-voxels", body.crop_dimensions_voxels.strip() or "128,128,128",
            "--data-scope", scope,
        ]
        if body.n_crops <= 1:
            argv.extend(["--crop-policy", "center"])
        if body.execute:
            argv.append("--execute")
            _downloader_sync_execute_begin(
                sid,
                start_headline,
                planned_total=planned_total,
            )
        result = (
            _run_downloader_command_live(sid, argv, cwd=s3, timeout_sec=_downloader_timeout_sec())
            if body.execute
            else run_command(argv, cwd=s3, timeout_sec=_downloader_timeout_sec())
        )
        ok = result["returncode"] == 0
        suffix = " (n_crops=1 → center crop policy)." if body.n_crops <= 1 else ""

    q_ok, q_reason = _downloader_quality_gate(result)
    ok = bool(ok and q_ok)
    if ok:
        _apply_step(session, PipelineStep.DOWNLOAD_SCRIPT)
    if not ok:
        _emit_run_failure_to_terminal("stage3-downloader", result)
    ok_msg = "Downloader script generated" + (" and executed." if body.execute else ".") + suffix
    if not ok and q_reason:
        ok_msg = f"Downloader step completed with dataset-level failures. {q_reason}"
    out: dict[str, Any] = {
        "ok": ok,
        "message": _studio_run_message(ok, ok_msg, "Downloader step failed.", result),
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "pipeline": _pipeline_dict(session),
    }
    if provider_generations:
        out["downloader_generations"] = provider_generations
        out["openorganelle_generations"] = provider_generations  # backward-compat alias
    # Return recently generated/updated scripts so UI can run the exact set.
    if gen_out_dir.is_dir():
        refreshed: list[str] = []
        for p in sorted(gen_out_dir.glob("download_*.py"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                if p.stat().st_mtime >= (gen_start_ts - 1.0):
                    refreshed.append(str(p.relative_to(root)).replace("\\", "/"))
            except Exception:
                continue
        if refreshed:
            generated_scripts = sorted(set(generated_scripts + refreshed), key=str.lower)
    out["generated_scripts"] = generated_scripts
    if body.execute:
        _downloader_sync_execute_finalize_from_out(sid, session, out)
    return out


@router.post("/run/downloader-script")
def studio_run_downloader_script(body: StudioRunDownloaderScriptBody) -> dict[str, Any]:
    root = _root()
    sid = (body.session_id or "default").strip() or "default"
    session = _session(sid)
    py = sys.executable
    script = Path(body.script_path.strip())
    if not script.is_absolute():
        script = (root / script).resolve()
    allowed_root = (root / "3data_downloader" / "outputs").resolve()
    if allowed_root not in script.parents or script.suffix != ".py":
        raise HTTPException(status_code=400, detail="script_path must point to a .py under 3data_downloader/outputs")
    if not script.is_file():
        raise HTTPException(status_code=404, detail=f"script not found: {script}")
    _self_heal_stale_downloader_running_flag(sid)
    argv = [py, "-u", str(script)]
    result = run_command(argv, cwd=root / "3data_downloader", timeout_sec=_downloader_timeout_sec())
    ok = result["returncode"] == 0
    if ok:
        _apply_step(session, PipelineStep.DOWNLOAD_SCRIPT)
    if not ok:
        _emit_run_failure_to_terminal("stage3-downloader-script", result)
    return {
        "ok": ok,
        "message": _studio_run_message(
            ok, "Generated downloader script executed.", "Generated downloader script failed.", result
        ),
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "pipeline": _pipeline_dict(session),
    }


@router.post("/run/downloader-script-stream")
async def studio_run_downloader_script_stream(body: StudioRunDownloaderScriptBody) -> StreamingResponse:
    """Run a generated downloader script and stream live logs as SSE."""
    root = _root()
    sid = body.session_id or "default"
    session = _session(sid)
    py = sys.executable
    script = Path(body.script_path.strip())
    if not script.is_absolute():
        script = (root / script).resolve()
    allowed_root = (root / "3data_downloader" / "outputs").resolve()
    if allowed_root not in script.parents or script.suffix != ".py":
        raise HTTPException(status_code=400, detail="script_path must point to a .py under 3data_downloader/outputs")
    if not script.is_file():
        raise HTTPException(status_code=404, detail=f"script not found: {script}")
    _self_heal_stale_downloader_running_flag(sid)
    with _downloader_state_lock:
        st = _get_or_init_downloader_state(sid)
        if st.get("running"):
            raise HTTPException(status_code=409, detail="A downloader script is already running for this session")
        st.update(
            {
                "running": True,
                "script_path": str(script.relative_to(root)).replace("\\", "/"),
                "log": "",
                "progress": None,
                "result": None,
                "updated_at": time.time(),
            }
        )

    q: queue.Queue[tuple[str, Any]] = queue.Queue()

    def worker() -> None:
        import subprocess

        argv = [py, "-u", str(script)]

        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(root / "3data_downloader"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as e:  # noqa: BLE001
            q.put(("error", str(e)))
            with _downloader_state_lock:
                s = _get_or_init_downloader_state(sid)
                s["running"] = False
                s["result"] = {
                    "ok": False,
                    "message": str(e),
                    "returncode": 1,
                    "stdout": "",
                    "stderr": str(e),
                    "pipeline": _pipeline_dict(session),
                }
                s["updated_at"] = time.time()
            return
        with _downloader_procs_lock:
            _downloader_procs[sid] = proc

        out_tail: list[str] = []
        err_tail: list[str] = []

        def _emit_stream_progress(d: dict[str, Any]) -> None:
            q.put(("progress", d))
            _set_downloader_progress(sid, d)

        dl_stream_progress = DownloaderLogProgressParser(_emit_stream_progress)

        def _pump(stream, kind: str) -> None:
            try:
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    if dl_stream_progress.filter_noise_line(line):
                        continue
                    dl_stream_progress.consume_line(line)
                    if kind == "stdout":
                        out_tail.append(line)
                        if len(out_tail) > 400:
                            del out_tail[: len(out_tail) - 400]
                        q.put(("log", line))
                        _append_downloader_log(sid, line)
                    else:
                        err_tail.append(line)
                        if len(err_tail) > 400:
                            del err_tail[: len(err_tail) - 400]
                        q.put(("log", line))
                        _append_downloader_log(sid, line)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t_out = threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True)
        t_err = threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True)
        t_out.start()
        t_err.start()

        to = _downloader_timeout_sec()
        start = time.monotonic()
        hb = _downloader_heartbeat_sec()
        next_heartbeat = start + hb if hb > 0 else float("inf")
        rc = None
        while rc is None:
            rc = proc.poll()
            now = time.monotonic()
            if rc is not None:
                break
            if hb > 0 and now >= next_heartbeat:
                elapsed = int(now - start)
                q.put(("log", f"[mito2] Downloader still running… elapsed {elapsed}s\n"))
                next_heartbeat = now + hb
            if to is not None and (now - start) >= to:
                proc.kill()
                rc = 124
                q.put(("log", f"\n[mito2] Timed out after {to}s.\n"))
                break
            time.sleep(1.0)
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        ok = rc == 0
        if ok:
            _apply_step(session, PipelineStep.DOWNLOAD_SCRIPT)
            # Force final progress to N/N on success (some scripts end without last [DONE]).
            with _downloader_state_lock:
                _st = _get_or_init_downloader_state(sid)
                _log = str(_st.get("log") or "")
                _prev = _st.get("progress") if isinstance(_st.get("progress"), dict) else None
            _total = 0
            _dataset = ""
            _m_summary = re.search(r"-\s*Planned image/label pairs:\s*(\d+)", _log)
            if _m_summary:
                _total = max(_total, int(_m_summary.group(1)))
            _m_ds_summary = re.search(r"-\s*Datasets:\s*(\d+)", _log)
            _m_win_summary = re.search(r"-\s*Windows per dataset:\s*(\d+)", _log)
            if _m_ds_summary and _m_win_summary:
                _total = max(_total, int(_m_ds_summary.group(1)) * max(1, int(_m_win_summary.group(1))))
            _m_summary = re.search(r"-\s*Datasets:\s*(\d+)", _log)
            if _m_summary:
                _total = max(_total, int(_m_summary.group(1)))
            for _m in re.finditer(r"\[DONE\]\s+dataset\s+(\d+)/(\d+):\s*(.+)\s*$", _log, flags=re.MULTILINE):
                _total = max(_total, int(_m.group(2)))
                _dataset = _m.group(3).strip() or _dataset
            for _m in re.finditer(r"\[PROGRESS\]\s+dataset\s+(\d+)/(\d+):\s*(.+)\s*$", _log, flags=re.MULTILINE):
                _total = max(_total, int(_m.group(2)))
                _dataset = _m.group(3).strip() or _dataset
            if _total <= 0 and _prev:
                try:
                    _total = int(_prev.get("total") or 0)
                    _dataset = str(_prev.get("dataset") or _dataset)
                except Exception:
                    _total = 0
            if _total > 0:
                _final_progress = {
                    "completed": _total,
                    "total": _total,
                    "current": _total,
                    "dataset": _dataset,
                }
                q.put(("progress", _final_progress))
                _set_downloader_progress(sid, _final_progress)
        stream_result = {
            "returncode": rc,
            "stdout": "".join(out_tail)[-8000:],
            "stderr": "".join(err_tail)[-8000:],
        }
        if not ok:
            _emit_run_failure_to_terminal("stage3-downloader-script-stream", stream_result)
        payload = {
            "ok": ok,
            "message": _studio_run_message(
                ok,
                "Generated downloader script executed.",
                "Generated downloader script failed.",
                stream_result,
            ),
            "returncode": rc,
            "stdout": stream_result["stdout"],
            "stderr": stream_result["stderr"],
            "pipeline": _pipeline_dict(session),
        }
        q.put(("done", payload))
        with _downloader_state_lock:
            s = _get_or_init_downloader_state(sid)
            s["running"] = False
            s["result"] = payload
            s["updated_at"] = time.time()
        with _downloader_procs_lock:
            cur = _downloader_procs.get(sid)
            if cur is proc:
                _downloader_procs.pop(sid, None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_gen():
        while True:
            kind, data = await asyncio.to_thread(q.get)
            if kind == "log":
                yield f"data: {json.dumps({'type': 'log', 'text': data}, ensure_ascii=False)}\n\n"
            elif kind == "progress":
                yield f"data: {json.dumps({'type': 'progress', 'payload': data}, ensure_ascii=False)}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': str(data)}, ensure_ascii=False)}\n\n"
                break
            elif kind == "done":
                yield f"data: {json.dumps({'type': 'done', 'payload': data}, ensure_ascii=False, default=str)}\n\n"
                break

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/run/downloader-script-cancel")
def studio_run_downloader_script_cancel(body: StudioDownloaderCancelBody) -> dict[str, Any]:
    def _kill_stage3_by_ps(*, script_path_hint: str = "") -> bool:
        """Fallback kill when Popen handle is missing/stale."""
        root = _root().resolve()
        s3 = (root / "3data_downloader").resolve()
        hint = (script_path_hint or "").strip()
        hint_base = os.path.basename(hint) if hint else ""
        try:
            proc_ps = subprocess.run(
                ["ps", "-ww", "-eo", "pid=,args="],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            out = proc_ps.stdout or ""
        except Exception:
            return False
        killed_any = False
        for raw in out.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
            except Exception:
                continue
            cmd = parts[1]
            cmd_l = cmd.lower()
            if str(s3) not in cmd:
                continue
            # stage-3 run signatures:
            # - python -u .../outputs/download_*.py
            # - python downloader_master/agent.py ... --execute
            is_stage3 = (
                "/outputs/download_" in cmd_l
                or "downloader_master/agent.py" in cmd_l
            )
            if not is_stage3:
                continue
            if hint_base and hint_base not in cmd:
                # If we have a precise script hint, prefer that exact process.
                continue
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed_any = True
            except Exception:
                pass
        return killed_any

    sid = (body.session_id or "default").strip() or "default"
    proc = None
    with _downloader_procs_lock:
        proc = _downloader_procs.get(sid)
    script_hint = ""
    is_mitole_inprocess_running = False
    with _downloader_state_lock:
        st = _get_or_init_downloader_state(sid)
        script_hint = str(st.get("script_path") or "").strip()
        is_mitole_inprocess_running = bool(st.get("running")) and script_hint == "[pipeline] mitole_stage3"
    if is_mitole_inprocess_running:
        with _downloader_kill_requested_lock:
            _downloader_kill_requested.add(sid)
        _append_downloader_log(sid, "[mito2] Download kill requested.\n")
        return {"ok": True, "killed": True}
    if proc is None:
        killed = _kill_stage3_by_ps(script_path_hint=script_hint)
        if killed:
            _append_downloader_log(sid, "[mito2] Download kill requested.\n")
        else:
            _downloader_sync_execute_clear_running(sid)
        return {"ok": True, "killed": killed}
    try:
        if proc.poll() is None:
            proc.kill()
            killed = True
        else:
            killed = False
        if not killed:
            # Popen handle exists but process already exited; try process discovery
            # in case active downloader moved to a child/no-handle process.
            killed = _kill_stage3_by_ps(script_path_hint=script_hint)
        _append_downloader_log(sid, "[mito2] Download kill requested.\n")
    except Exception:
        killed = _kill_stage3_by_ps(script_path_hint=script_hint)
        if killed:
            _append_downloader_log(sid, "[mito2] Download kill requested.\n")
    return {"ok": True, "killed": killed}


@router.post("/run/preprocess")
def studio_run_preprocess(body: StudioSessionBody) -> dict[str, Any]:
    root = _root()
    sid = body.session_id or "default"
    session = _session(sid)
    s4 = root / "3data_downloader"
    if not (s4 / "downloader_master/preprocess_agent.py").is_file():
        raise HTTPException(status_code=500, detail="Stage 4 agent.py missing")
    py = sys.executable
    result = run_command([py, "downloader_master/preprocess_agent.py", "--task", "supervised"], cwd=s4, timeout_sec=3600.0)
    ok = result["returncode"] == 0
    if ok:
        _apply_step(session, PipelineStep.PREPROCESS)
    if not ok:
        _emit_run_failure_to_terminal("stage4-preprocess", result)
    return {
        "ok": ok,
        "message": _studio_run_message(
            ok, "Preprocess completed (fully supervised artifacts).", "Preprocess failed.", result
        ),
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "pipeline": _pipeline_dict(session),
    }


def _studio_preprocess_start_sync(body: StudioPreprocessSelectiveBody) -> dict[str, Any]:
    """Sync implementation for ``POST …/preprocess-selective`` (runs on ``_preprocess_studio_executor``)."""
    root = _root()
    sid = (body.session_id or "default").strip() or "default"
    argv = _build_preprocess_selective_argv(root, body)
    s4 = root / "3data_downloader"
    with _preprocess_state_lock:
        st = _get_or_init_preprocess_state(sid)
        if st.get("running"):
            raise HTTPException(
                status_code=409,
                detail="A selective preprocess is already running for this session",
            )
        st.update(
            {
                "running": True,
                "log": "[mito2] Selective preprocess queued…\n",
                "progress": None,
                "result": None,
                "updated_at": time.time(),
            }
        )
    threading.Thread(target=_preprocess_selective_worker, args=(sid, argv, s4.resolve()), daemon=True).start()
    return {
        "ok": True,
        "accepted": True,
        "message": "Selective preprocess started — refresh-safe; poll preprocess state for log and final result.",
    }


@router.post("/run/preprocess-selective")
async def studio_run_preprocess_selective(body: StudioPreprocessSelectiveBody) -> dict[str, Any]:
    """Queue selective preprocess in a background thread (same pattern as streaming downloader).

    Returns immediately with ``accepted``; poll ``GET /run/preprocess-selective-state`` for ``log``/``result``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_preprocess_studio_executor, _studio_preprocess_start_sync, body)


def _preprocess_selective_state_sync(sid: str) -> dict[str, Any]:
    """Sync body for ``GET …/preprocess-selective-state`` (runs on ``_preprocess_studio_executor``)."""
    # Self-heal only when we hold a live ``Popen`` that has exited. Do **not** use ``proc is None``:
    # startup has not registered ``Popen`` yet; shutdown clears state before dropping the handle.
    with _preprocess_procs_lock:
        proc = _preprocess_procs.get(sid)
    proc_dead = False
    if proc is not None:
        try:
            proc_dead = proc.poll() is not None
        except Exception:
            proc_dead = True
    with _preprocess_state_lock:
        st = _get_or_init_preprocess_state(sid)
        if st.get("running") and proc is not None and proc_dead:
            session = _session(sid)
            with _preprocess_kill_requested_lock:
                killed_stop = sid in _preprocess_kill_requested
            st["running"] = False
            st["progress"] = None
            if st.get("result") is None:
                st["result"] = {
                    "ok": False,
                    "message": (
                        "Selective preprocess stopped (killed)."
                        if killed_stop
                        else "Selective preprocess stopped (process no longer running)."
                    ),
                    "returncode": 130,
                    "stdout": "",
                    "stderr": "",
                    "pipeline": _pipeline_dict(session),
                }
            st["updated_at"] = time.time()
        s = st.copy()
    return {"ok": True, **s}


@router.get("/run/preprocess-selective-state")
async def studio_run_preprocess_selective_state(session_id: str = "default") -> dict[str, Any]:
    sid = (session_id or "default").strip() or "default"
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_preprocess_studio_executor, _preprocess_selective_state_sync, sid)


def _list_stage4_supervised_agent_pids(project_root: Path, *, ps_timeout_sec: float = 5.0) -> list[tuple[int, int]]:
    """Return ``(pid, pgid)`` for stage-4 ``downloader_master/preprocess_agent.py`` supervised runs for this project.

    Uses **one** bounded ``ps`` invocation (never walks all of ``/proc`` per PID — that can take
    minutes on busy hosts and block the HTTP cancel handler until the client aborts).

    Studio passes ``--project-root`` and ``--dataset-paths-json`` so ``ps`` args usually contain
    the repo path; if not, orphan detection may miss (direct ``Popen`` kill still works).
    """
    root_variants = _studio_project_path_variants(project_root)
    root_hits = {v for v in root_variants if len(v) >= 8}
    out: list[tuple[int, int]] = []

    def _cmdline_supervised_agent(cmd_l: str) -> bool:
        return "downloader_master/preprocess_agent.py" in cmd_l and "--task" in cmd_l and "supervised" in cmd_l

    def _cmdline_project_scoped_for_ps(cmd_l: str) -> bool:
        """``ps`` output may omit cwd; require argv hints so we do not kill unrelated projects."""
        if not _cmdline_supervised_agent(cmd_l):
            return False
        if "--dataset-paths-json" in cmd_l:
            return True
        return any(rv in cmd_l for rv in root_hits)

    ps_out = ""
    for argv in (
        ["ps", "-ww", "-eo", "pid=,pgid=,args="],
        ["ps", "-w", "-eo", "pid=,pgid=,args="],
        ["ps", "-eo", "pid=,pgid=,args="],
    ):
        try:
            proc_ps = subprocess.run(argv, capture_output=True, text=True, timeout=float(ps_timeout_sec), check=False)
            cand = proc_ps.stdout or ""
            if cand.strip():
                ps_out = cand
                break
        except (subprocess.TimeoutExpired, Exception):
            continue
    if not ps_out.strip():
        return []
    for raw in ps_out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except Exception:
            continue
        if pid == os.getpid():
            continue
        args = parts[2].replace("\\", "/").lower()
        if not _cmdline_project_scoped_for_ps(args):
            continue
        out.append((pid, pgid))
    return out


def _kill_external_preprocess_jobs(
    *,
    ps_timeout_sec: float = 5.0,
    procfs_budget_sec: float = 3.0,
    relaxed_s4_cwd_agent: bool = False,
) -> int:
    """Kill stage-4 ``agent.py`` processes for this project (tree + pg; ps + /proc)."""
    root = _root()
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []
    for item in _list_stage4_supervised_agent_pids(root, ps_timeout_sec=ps_timeout_sec):
        if item not in seen:
            seen.add(item)
            candidates.append(item)
    for item in _list_stage4_supervised_by_procfs_cwd(root, budget_sec=procfs_budget_sec):
        if item not in seen:
            seen.add(item)
            candidates.append(item)
    if relaxed_s4_cwd_agent:
        for item in _list_stage4_agent_pids_s4_cwd_any_task(root, budget_sec=procfs_budget_sec):
            if item not in seen:
                seen.add(item)
                candidates.append(item)
    n_signaled = 0
    seen_pid: set[int] = set()
    for pid, _pgid in candidates:
        if pid in seen_pid:
            continue
        seen_pid.add(pid)
        if _sigkill_preprocess_os_pid(pid):
            n_signaled += 1
    return n_signaled


def _collect_stage4_preprocess_pids(
    project_root: Path,
    *,
    ps_timeout_sec: float = 5.0,
    procfs_budget_sec: float = 3.0,
    relaxed_s4_cwd_agent: bool = False,
) -> set[int]:
    """PIDs for this repo's stage-4 ``agent.py`` (``ps`` + Linux ``/proc``)."""
    s: set[int] = set()
    for pid, _ in _list_stage4_supervised_agent_pids(project_root, ps_timeout_sec=ps_timeout_sec):
        s.add(pid)
    for pid, _ in _list_stage4_supervised_by_procfs_cwd(project_root, budget_sec=procfs_budget_sec):
        s.add(pid)
    if relaxed_s4_cwd_agent:
        for pid, _ in _list_stage4_agent_pids_s4_cwd_any_task(project_root, budget_sec=procfs_budget_sec):
            s.add(pid)
    return s


def _preprocess_cancel_straggler_sweep(sid: str) -> None:
    """Extra SIGKILL rounds off the single-thread kill executor (daemon) so HTTP can return quickly."""
    try:
        root = _root()
        time.sleep(0.15)
        for i in range(40):
            relaxed = i >= 2
            remaining = _collect_stage4_preprocess_pids(root, relaxed_s4_cwd_agent=relaxed)
            if not remaining:
                return
            _kill_external_preprocess_jobs(relaxed_s4_cwd_agent=relaxed)
            time.sleep(0.28)
        remaining = _collect_stage4_preprocess_pids(root, relaxed_s4_cwd_agent=True)
        if remaining:
            msg_pids = ", ".join(str(p) for p in sorted(remaining)[:14])
            _append_preprocess_log(
                sid,
                f"[mito2] Preprocess cancel: follow-up sweeps still see stage-4 PIDs [{msg_pids}]. "
                "Try killing those PIDs manually or check permissions.\n",
            )
    except Exception:
        logging.exception("preprocess cancel straggler sweep failed for session %s", sid)


def _preprocess_selective_cancel_impl(sid: str) -> dict[str, Any]:
    """Stop preprocess: prefer in-memory ``Popen``, then durable PID file (multi-worker safe), then ``ps`` sweep."""

    def _mark_stopped_after_kill(*, message: str = "Selective preprocess stopped (killed).") -> None:
        session = _session(sid)
        with _preprocess_state_lock:
            st = _get_or_init_preprocess_state(sid)
            st["running"] = False
            st["progress"] = None
            st["result"] = {
                "ok": False,
                "message": message,
                "returncode": 130,
                "stdout": "",
                "stderr": "",
                "pipeline": _pipeline_dict(session),
            }
            st["updated_at"] = time.time()
        _append_preprocess_log(sid, "[mito2] Preprocess cancel: kill signal sent (SIGKILL).\n")

    with _preprocess_kill_requested_lock:
        _preprocess_kill_requested.add(sid)

    with _preprocess_procs_lock:
        proc = _preprocess_procs.get(sid)
    registered_proc = proc
    on_disk_pid = _read_preprocess_selective_pid(sid)

    with _preprocess_state_lock:
        running = bool(_get_or_init_preprocess_state(sid).get("running"))

    root = _root()
    _append_preprocess_log(
        sid,
        f"[mito2] Preprocess cancel: begin sid={sid!r} project_root={root} "
        f"popen_pid={getattr(proc, 'pid', None)!r} pid_file={on_disk_pid!r} session_running={running}\n",
    )

    killed_direct = False
    if proc is not None:
        killed_direct = _sigkill_preprocess_os_pid(proc.pid)
    if not killed_direct and on_disk_pid is not None:
        killed_direct = _sigkill_preprocess_os_pid(on_disk_pid)

    if proc is not None:
        with _preprocess_procs_lock:
            cur = _preprocess_procs.get(sid)
            if cur is proc:
                _preprocess_procs.pop(sid, None)

    # Tight budgets on the **HTTP-bound** path; second pass uses **relaxed** /proc matching (cwd in
    # ``3data_downloader`` only) so truncated cmdline cannot hide the real writer.
    fast_ps = 2.0
    fast_pf = 0.75
    swept = _kill_external_preprocess_jobs(ps_timeout_sec=fast_ps, procfs_budget_sec=fast_pf)
    time.sleep(0.12)
    remaining = _collect_stage4_preprocess_pids(root, ps_timeout_sec=fast_ps, procfs_budget_sec=fast_pf)
    if remaining:
        swept += _kill_external_preprocess_jobs(
            ps_timeout_sec=fast_ps,
            procfs_budget_sec=fast_pf,
            relaxed_s4_cwd_agent=True,
        )
        time.sleep(0.12)
        remaining = _collect_stage4_preprocess_pids(
            root,
            ps_timeout_sec=fast_ps,
            procfs_budget_sec=fast_pf,
            relaxed_s4_cwd_agent=True,
        )
    if remaining:
        threading.Thread(
            target=_preprocess_cancel_straggler_sweep,
            args=(sid,),
            daemon=True,
            name=f"mito2-pre-kill-followup-{sid}",
        ).start()
    had_handle = registered_proc is not None or on_disk_pid is not None
    we_signaled = bool(killed_direct or swept > 0)

    if not remaining:
        if not had_handle and not running and not we_signaled:
            with _preprocess_kill_requested_lock:
                _preprocess_kill_requested.discard(sid)
            return {"ok": True, "killed": False}
        # Worker set ``running`` before ``Popen``; do not clear state until the worker observes ``kill_requested``.
        if registered_proc is None and on_disk_pid is None and running and not we_signaled:
            _append_preprocess_log(
                sid,
                "[mito2] Preprocess cancel: stop requested (subprocess not registered yet — worker will exit).\n",
            )
            _append_preprocess_log(sid, "[mito2] Preprocess cancel: kill_requested flag set.\n")
            return {"ok": True, "killed": True}
        _unlink_preprocess_selective_pid_file(sid)
        _mark_stopped_after_kill()
        if swept > 0:
            _append_preprocess_log(
                sid,
                f"[mito2] Preprocess cancel: force-killed {swept} stage-4 process(es) via discovery sweep.\n",
            )
        return {"ok": True, "killed": True}

    if remaining and we_signaled:
        _append_preprocess_log(
            sid,
            f"[mito2] Preprocess cancel: {len(remaining)} PID(s) still listed after fast sweeps — follow-up thread running.\n",
        )
        return {
            "ok": True,
            "killed": False,
            "warning": (
                "SIGKILL was sent; extra sweeps are running in the background (a few seconds). "
                "If new outputs still appear, check the preprocessor log for PIDs."
            ),
        }

    msg_pids = ", ".join(str(p) for p in sorted(remaining)[:12])
    if len(remaining) > 12:
        msg_pids += ", …"
    _append_preprocess_log(
        sid,
        f"[mito2] Preprocess cancel: stage-4 process(es) still present after SIGKILL (PIDs {msg_pids}). "
        "Check permissions or kill manually.\n",
    )
    warn = (
        "Could not terminate all stage-4 processes for this project. "
        "See preprocessor log for PIDs; you may need to kill them manually or fix permissions."
    )
    return {"ok": True, "killed": False, "warning": warn}


def _preprocess_selective_cancel_sync(sid: str) -> dict[str, Any]:
    try:
        return _preprocess_selective_cancel_impl(sid)
    except Exception:
        logging.exception("preprocess selective cancel failed for session %s", sid)
        return {"ok": False, "killed": False, "error": "cancel_internal_error"}


@router.post("/run/preprocess-selective-cancel")
async def studio_run_preprocess_selective_cancel_post(body: StudioDownloaderCancelBody) -> dict[str, Any]:
    """Block until SIGKILL + discovery sweeps finish so the client never sees \"Kill sent\" while writers live."""
    sid = (body.session_id or "default").strip() or "default"
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_preprocess_kill_executor, _preprocess_selective_cancel_sync, sid)


@router.api_route(
    "/run/preprocess-selective/cancel",
    methods=["POST", "GET", "DELETE", "PUT", "PATCH", "OPTIONS"],
    include_in_schema=False,
)
@router.api_route(
    "/run/preprocess-selective/cancel/",
    methods=["POST", "GET", "DELETE", "PUT", "PATCH", "OPTIONS"],
    include_in_schema=False,
)
async def studio_run_preprocess_selective_cancel_any(
    body: StudioDownloaderCancelBody | None = Body(default=None),
    session_id: str = Query("default"),
) -> dict[str, Any]:
    """Legacy alias (slash path); prefer ``POST …/preprocess-selective-cancel`` (matches downloader URL shape)."""
    sid = ((body.session_id if body and body.session_id else session_id) or "default").strip() or "default"
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_preprocess_kill_executor, _preprocess_selective_cancel_sync, sid)


@router.api_route(
    "/run/preprocess-selective-state/clear",
    methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"],
    include_in_schema=False,
)
@router.api_route(
    "/run/preprocess-selective-state/clear/",
    methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"],
    include_in_schema=False,
)
def studio_run_preprocess_selective_state_clear_any(
    body: StudioDownloaderCancelBody | None = None,
    session_id: str = Query("default"),
) -> dict[str, Any]:
    sid = ((body.session_id if body and body.session_id else session_id) or "default").strip() or "default"
    return _clear_preprocess_selective_state_for_session(sid)


@router.post("/run/training")
def studio_run_training(body: StudioSessionBody) -> dict[str, Any]:
    """Submit nnUNet training via Slurm (``sbatch`` on ``train_nnunet_mito_foundation.sl``)."""
    root = _root()
    sid = (body.session_id or "default").strip() or "default"
    session = _session(sid)
    if shutil.which("sbatch") is None:
        raise HTTPException(
            status_code=500,
            detail="sbatch not found on PATH; run this action from a cluster login node or install Slurm client tools.",
        )
    slurm_script = _default_slurm_training_script(root)
    if not slurm_script.is_file():
        raise HTTPException(status_code=500, detail=f"Slurm training script not found: {slurm_script}")
    _slurm_logs_dir(root, "training").mkdir(parents=True, exist_ok=True)

    argv = ["sbatch", str(slurm_script.resolve())]
    result = run_command(argv, cwd=root, timeout_sec=120.0, env={"MITO2_PROJECT_ROOT": str(root.resolve())})
    ok = result["returncode"] == 0
    out = (result.get("stdout") or "").strip()
    job_id = _parse_slurm_submitted_job_id(out) if ok else None
    if ok and not job_id:
        ok = False
        result = {
            **result,
            "stderr": (result.get("stderr") or "").strip()
            + ("\n" if result.get("stderr") else "")
            + "Could not parse Slurm job id from sbatch output.",
        }
    jn, out_tmpl, err_tmpl = _slurm_batch_file_patterns(slurm_script)
    out_path = _expand_slurm_path_template(out_tmpl, jn, job_id) if ok and job_id and out_tmpl else ""
    err_path = _expand_slurm_path_template(err_tmpl, jn, job_id) if ok and job_id and err_tmpl else ""
    if ok and job_id and not out_path:
        out_path = str((_slurm_logs_dir(root, "training") / f"{jn}_{job_id}.out").resolve())
    if ok and job_id and not err_path:
        err_path = str((_slurm_logs_dir(root, "training") / f"{jn}_{job_id}.err").resolve())

    if ok:
        _apply_step(session, PipelineStep.SSL)
        st = _get_or_init_slurm_run_state(sid, "training")
        st["running"] = True
        st["job_id"] = str(job_id or "")
        st["out_path"] = out_path
        st["err_path"] = err_path
        st["result"] = None
        st["updated_at"] = time.time()
        msg = (
            f"Submitted Slurm job {job_id}.\n"
            f"Out: {out_path}\n"
            f"Err: {err_path}"
        )
        return {
            "ok": True,
            "message": msg,
            "returncode": 0,
            "stdout": msg,
            "stderr": "",
            "pipeline": _pipeline_dict(session),
            "slurm_job_id": job_id,
            "slurm_out_path": out_path,
            "slurm_err_path": err_path,
            "training_log_path": out_path,
        }

    _emit_run_failure_to_terminal("stage5-sbatch-training", result)
    st = _get_or_init_slurm_run_state(sid, "training")
    st["running"] = False
    st["result"] = {
        "ok": False,
        "message": _studio_run_message(ok, "", "Slurm sbatch failed.", result),
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "pipeline": _pipeline_dict(session),
    }
    st["updated_at"] = time.time()
    return {
        "ok": False,
        "message": _studio_run_message(ok, "", "Slurm sbatch failed.", result),
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "pipeline": _pipeline_dict(session),
    }


@router.post("/run/inference")
def studio_run_inference(body: StudioSessionBody) -> dict[str, Any]:
    """Submit nnUNet inference via Slurm (``sbatch`` on ``infer_nnunet_mito_foundation.sl``)."""
    root = _root()
    sid = (body.session_id or "default").strip() or "default"
    session = _session(sid)
    if shutil.which("sbatch") is None:
        raise HTTPException(
            status_code=500,
            detail="sbatch not found on PATH; run this action from a cluster login node or install Slurm client tools.",
        )
    slurm_script = _default_slurm_inference_script(root)
    if not slurm_script.is_file():
        raise HTTPException(status_code=500, detail=f"Slurm inference script not found: {slurm_script}")
    _slurm_logs_dir(root, "inference").mkdir(parents=True, exist_ok=True)

    argv = ["sbatch", str(slurm_script.resolve())]
    result = run_command(argv, cwd=root, timeout_sec=120.0, env={"MITO2_PROJECT_ROOT": str(root.resolve())})
    ok = result["returncode"] == 0
    out = (result.get("stdout") or "").strip()
    job_id = _parse_slurm_submitted_job_id(out) if ok else None
    if ok and not job_id:
        ok = False
        result = {
            **result,
            "stderr": (result.get("stderr") or "").strip()
            + ("\n" if result.get("stderr") else "")
            + "Could not parse Slurm job id from sbatch output.",
        }
    jn, out_tmpl, err_tmpl = _slurm_batch_file_patterns(slurm_script)
    out_path = _expand_slurm_path_template(out_tmpl, jn, job_id) if ok and job_id and out_tmpl else ""
    err_path = _expand_slurm_path_template(err_tmpl, jn, job_id) if ok and job_id and err_tmpl else ""
    if ok and job_id and not out_path:
        out_path = str((_slurm_logs_dir(root, "inference") / f"{jn}_{job_id}.out").resolve())
    if ok and job_id and not err_path:
        err_path = str((_slurm_logs_dir(root, "inference") / f"{jn}_{job_id}.err").resolve())

    if ok:
        _apply_step(session, PipelineStep.SSL)
        st = _get_or_init_slurm_run_state(sid, "inference")
        st["running"] = True
        st["job_id"] = str(job_id or "")
        st["out_path"] = out_path
        st["err_path"] = err_path
        st["result"] = None
        st["updated_at"] = time.time()
        msg = (
            f"Submitted Slurm job {job_id}.\n"
            f"Out: {out_path}\n"
            f"Err: {err_path}"
        )
        return {
            "ok": True,
            "message": msg,
            "returncode": 0,
            "stdout": msg,
            "stderr": "",
            "pipeline": _pipeline_dict(session),
            "slurm_job_id": job_id,
            "slurm_out_path": out_path,
            "slurm_err_path": err_path,
            "training_log_path": out_path,
        }

    _emit_run_failure_to_terminal("stage5-sbatch-inference", result)
    st = _get_or_init_slurm_run_state(sid, "inference")
    st["running"] = False
    st["result"] = {
        "ok": False,
        "message": _studio_run_message(ok, "", "Slurm sbatch failed.", result),
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "pipeline": _pipeline_dict(session),
    }
    st["updated_at"] = time.time()
    return {
        "ok": False,
        "message": _studio_run_message(ok, "", "Slurm sbatch failed.", result),
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "pipeline": _pipeline_dict(session),
    }


@router.post("/run/postprocessing")
def studio_run_postprocessing(body: StudioPostprocessBody) -> dict[str, Any]:
    """Run watershed postprocessing from border-contour predictions."""
    root = _root()
    sid = (body.session_id or "default").strip() or "default"
    session = _session(sid)

    script = (root / "5model_training" / "postprocess_evaluation" / "run_watershed_postprocess.py").resolve()
    if not script.is_file():
        raise HTTPException(status_code=500, detail=f"Postprocessing script not found: {script}")

    # Fixed project paths for Studio Postprocessing page.
    input_dir = data_outputs_bc(root)
    output_dir = data_outputs_postprocessed(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    argv = [
        py,
        str(script),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
    ]
    result = run_command(argv, cwd=root, timeout_sec=3600.0)
    ok = result["returncode"] == 0
    if ok:
        _apply_step(session, PipelineStep.SSL)
    else:
        _emit_run_failure_to_terminal("stage5-postprocessing", result)

    msg = _studio_run_message(ok, "Postprocessing finished.", "Postprocessing failed.", result)
    return {
        "ok": ok,
        "message": msg,
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "pipeline": _pipeline_dict(session),
    }


@router.post("/run/evaluation")
def studio_run_evaluation(body: StudioEvaluateBody) -> dict[str, Any]:
    """Evaluate postprocessed predictions against labelsTs-instance."""
    root = _root()
    sid = (body.session_id or "default").strip() or "default"
    session = _session(sid)

    script = (root / "5model_training" / "postprocess_evaluation" / "run_instance_evaluation.py").resolve()
    if not script.is_file():
        raise HTTPException(status_code=500, detail=f"Evaluation script not found: {script}")

    # Fixed project paths for Studio Evaluation page.
    pred_dir = data_outputs_postprocessed(root)
    gt_dir = resolve_under_project(rel_nnunet_labels_ts_instance(), root)
    py = sys.executable
    argv = [
        py,
        str(script),
        "--pred_dir",
        str(pred_dir),
        "--gt_dir",
        str(gt_dir),
    ]
    result = run_command(argv, cwd=root, timeout_sec=3600.0)
    ok = result["returncode"] == 0
    if ok:
        _apply_step(session, PipelineStep.SSL)
    else:
        _emit_run_failure_to_terminal("stage5-evaluation", result)

    summary: dict[str, Any] | None = None
    cases: list[dict[str, Any]] | None = None
    try:
        for line in str(result.get("stdout") or "").splitlines():
            if line.startswith("EVAL_SUMMARY_JSON="):
                parsed = json.loads(line.split("=", 1)[1].strip())
                if isinstance(parsed, dict):
                    summary = parsed
            elif line.startswith("EVAL_CASES_JSON="):
                parsed = json.loads(line.split("=", 1)[1].strip())
                if isinstance(parsed, list):
                    cases = [x for x in parsed if isinstance(x, dict)]
    except Exception:
        summary = summary if isinstance(summary, dict) else None
        cases = cases if isinstance(cases, list) else None

    msg = _studio_run_message(ok, "Evaluation finished.", "Evaluation failed.", result)
    return {
        "ok": ok,
        "message": msg,
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "evaluation_summary": summary,
        "evaluation_cases": cases,
        "pipeline": _pipeline_dict(session),
    }


@router.get("/run/training-state")
def studio_run_training_state(
    session_id: str = Query("default"),
    log_root: str = Query(default="", description="Optional log root (without .out/.err)"),
) -> dict[str, Any]:
    root = _root()
    sid = _slurm_session_key(session_id)
    s = _sync_slurm_run_logs(sid, "training")
    requested = (log_root or "").strip()
    roots = _list_slurm_log_roots(root, "training")
    if requested and requested in roots:
        selected_root = requested
    elif roots:
        selected_root = roots[0]
    else:
        selected_root = ""

    out_path = ""
    err_path = ""
    out_log = ""
    err_log = ""
    if selected_root:
        logs_dir = _slurm_logs_dir(root, "training")
        out_path = str((logs_dir / f"{selected_root}.out").resolve())
        err_path = str((logs_dir / f"{selected_root}.err").resolve())
        out_log = _read_slurm_log(out_path)
        err_log = _read_slurm_log(err_path)
    summary = _extract_training_log_summary(out_log)
    return {
        "ok": True,
        "running": bool(s.get("running")),
        "selected_log_root": selected_root,
        "log_roots": roots,
        "out_path": out_path,
        "err_path": err_path,
        "out_log": out_log,
        "err_log": err_log,
        "summary": summary,
        "result": s.get("result"),
        "updated_at": float(s.get("updated_at") or time.time()),
    }


@router.post("/run/training-state/clear")
def studio_run_training_state_clear(body: StudioSessionBody) -> dict[str, Any]:
    sid = _slurm_session_key(body.session_id)
    s = _sync_slurm_run_logs(sid, "training")
    if bool(s.get("running")):
        raise HTTPException(status_code=409, detail="Cannot clear training output while a run is active")
    s.update({
        "running": False,
        "job_id": "",
        "out_path": "",
        "err_path": "",
        "out_log": "",
        "err_log": "",
        "result": None,
        "updated_at": time.time(),
    })
    return {"ok": True, "cleared": True}


@router.get("/run/inference-state")
def studio_run_inference_state(
    session_id: str = Query("default"),
    log_root: str = Query(default="", description="Optional log root (without .out/.err)"),
) -> dict[str, Any]:
    root = _root()
    sid = _slurm_session_key(session_id)
    s = _sync_slurm_run_logs(sid, "inference")
    requested = (log_root or "").strip()
    roots = _list_slurm_log_roots(root, "inference")
    if requested and requested in roots:
        selected_root = requested
    elif roots:
        selected_root = roots[0]
    else:
        selected_root = ""

    out_path = ""
    err_path = ""
    out_log = ""
    err_log = ""
    if selected_root:
        logs_dir = _slurm_logs_dir(root, "inference")
        out_path = str((logs_dir / f"{selected_root}.out").resolve())
        err_path = str((logs_dir / f"{selected_root}.err").resolve())
        out_log = _read_slurm_log(out_path)
        err_log = _read_slurm_log(err_path)
    return {
        "ok": True,
        "running": bool(s.get("running")),
        "selected_log_root": selected_root,
        "log_roots": roots,
        "out_path": out_path,
        "err_path": err_path,
        "out_log": out_log,
        "err_log": err_log,
        "result": s.get("result"),
        "updated_at": float(s.get("updated_at") or time.time()),
    }


@router.post("/run/inference-state/clear")
def studio_run_inference_state_clear(body: StudioSessionBody) -> dict[str, Any]:
    sid = _slurm_session_key(body.session_id)
    s = _sync_slurm_run_logs(sid, "inference")
    if bool(s.get("running")):
        raise HTTPException(status_code=409, detail="Cannot clear inference output while a run is active")
    s.update({
        "running": False,
        "job_id": "",
        "out_path": "",
        "err_path": "",
        "out_log": "",
        "err_log": "",
        "result": None,
        "updated_at": time.time(),
    })
    return {"ok": True, "cleared": True}


# ── Download batch provenance endpoints ───────────────────────────────────────

def _registry_path(root: Path) -> Path:
    return root / "data" / "registry.sqlite"


def _open_reg(root: Path):
    """Open registry; raise 503 if not yet built."""
    rp = _registry_path(root)
    if not rp.is_file():
        raise HTTPException(status_code=503, detail="Registry not built yet — run stage 2 first.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from agent.orchestration.registry.schema import open_registry as _open  # noqa: PLC0415
    return _open(rp)


@router.get("/download-batches")
def studio_list_download_batches(
    provider: str = Query(default="", description="Filter by provider name (empty = all)"),
) -> dict[str, Any]:
    """List all recorded download batches with their item-level statuses."""
    root = _root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from agent.orchestration.registry.api import list_download_batches as _list_batches, list_batch_items  # noqa: PLC0415
    from agent.orchestration.registry.schema import open_registry  # noqa: PLC0415

    rp = _registry_path(root)
    if not rp.is_file():
        return {"ok": True, "batches": []}

    conn = open_registry(rp)
    try:
        batches_raw = _list_batches(conn, provider.strip() or None)
        batches_out: list[dict[str, Any]] = []
        for b in batches_raw:
            items_raw = list_batch_items(conn, int(b["id"]))
            items_out = [
                {
                    "id": int(i["id"]),
                    "stable_id": i["stable_id"],
                    "asset_type": i["asset_type"],
                    "local_path": i["local_path"],
                    "status": i["status"],
                    "completed_at": i["completed_at"],
                }
                for i in items_raw
            ]
            n_present = sum(1 for i in items_out if i["status"] == "present")
            n_missing = sum(1 for i in items_out if i["status"] == "missing_or_deleted_local")
            try:
                profile = json.loads(b["profile_json"] or "{}")
            except Exception:
                profile = {}
            batches_out.append({
                "id": int(b["id"]),
                "batch_id": b["batch_id"],
                "provider": b["provider"],
                "profile_hash": b["profile_hash"],
                "profile": profile,
                "run_folder": b["run_folder"],
                "status": b["status"],
                "created_at": b["created_at"],
                "finished_at": b["finished_at"],
                "n_items": len(items_out),
                "n_present": n_present,
                "n_missing": n_missing,
                "items": items_out,
            })
        return {"ok": True, "batches": batches_out}
    finally:
        conn.close()


class DatasetDeleteBody(BaseModel):
    """Hard-delete one or more training datasets from disk and update registry."""
    stable_ids: list[str] = Field(default_factory=list, description="Dataset stable_ids to delete")
    file_paths: list[str] = Field(default_factory=list, description="Exact dataset file paths to delete")
    provider: str = Field(default="OpenOrganelle")


def _normalize_stable_id(raw: str) -> str:
    s = (raw or "").strip()
    lower = s.lower()
    if lower.endswith(".nii.gz"):
        s = s[: -len(".nii.gz")]
    elif lower.endswith(".h5"):
        s = s[: -len(".h5")]
    elif lower.endswith(".nrrd"):
        s = s[: -len(".nrrd")]
    elif lower.endswith(".nii"):
        s = s[: -len(".nii")]
    s = re.sub(r"(_im|_seg|\.im|\.seg)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_0000$", "", s, flags=re.IGNORECASE)
    return s


def _asset_type_from_local_path(path: Path) -> str | None:
    s = str(path).replace("\\", "/").lower()
    stem_l = path.stem.lower()
    if stem_l.endswith("_im") or stem_l.endswith(".im") or stem_l.endswith("_0000") or "/images/" in s or "/imagestr/" in s or "/imagests/" in s:
        return "em_volume"
    if stem_l.endswith("_seg") or stem_l.endswith(".seg") or "/labels/" in s or "/labelstr/" in s or "/labelsts/" in s:
        return "mito_seg"
    return None


def _data_source_from_local_path(path_like: str | Path | None) -> str:
    s = str(path_like or "").replace("\\", "/").lower()
    # ``imagesTr``.lower() is ``imagestr``; ``labelsTr`` → ``labelstr``.
    if "/data/nnunet_raw/dataset001_mito2/imagestr/" in s or "/data/nnunet_raw/dataset001_mito2/labelstr/" in s:
        return "training"
    if "/data/nnunet_raw/dataset001_mito2/imagests/" in s or "/data/nnunet_raw/dataset001_mito2/labelsts/" in s:
        return "inference"
    return "unknown"


def _data_source_from_row(
    local_path: str | Path | None,
    run_folder: str | None,
    profile: dict[str, Any] | None,
) -> str:
    src = _data_source_from_local_path(local_path)
    if src != "unknown":
        return src
    src = _data_source_from_local_path(run_folder)
    if src != "unknown":
        return src
    raw = str((profile or {}).get("data_source") or "").strip().lower()
    if raw in ("training", "inference"):
        return raw
    return "unknown"


def _stable_key_from_filename(path: Path) -> str:
    return _normalize_stable_id(path.name).lower()


def _data_h5_paired_paths(
    root: Path,
) -> tuple[set[str], set[str], list[tuple[Path, Path, str]]]:
    """Pair nnUNet image/label files from Dataset001_mito2.

    Rebuilding paths as ``f\"{key}_im.h5\"`` from a normalized key can miss files when
    casing or spelling no longer round-trips; we always use the exact ``Path``s seen on disk.
    """
    all_im_keys: set[str] = set()
    all_seg_keys: set[str] = set()
    pair_paths: list[tuple[Path, Path, str]] = []

    dataset = nnunet_dataset_root(root)
    splits = {
        "training": (dataset / "imagesTr", dataset / "labelsTr"),
        "inference": (dataset / "imagesTs", dataset / "labelsTs"),
    }
    for source, (src_img, src_lbl) in splits.items():
        im_by_stem: dict[str, Path] = {}
        seg_by_stem: dict[str, Path] = {}
        if src_img.is_dir():
            for p in src_img.iterdir():
                if p.is_file() and p.name.lower().endswith("_0000.nii.gz"):
                    k = _normalize_stable_id(p.name)
                    im_by_stem[k] = p
                    all_im_keys.add(f"{source}:{k}")
        if src_lbl.is_dir():
            for p in src_lbl.iterdir():
                if p.is_file() and p.name.lower().endswith(".nii.gz"):
                    k = _normalize_stable_id(p.name)
                    seg_by_stem[k] = p
                    all_seg_keys.add(f"{source}:{k}")
        common = im_by_stem.keys() & seg_by_stem.keys()
        for k in sorted(common, key=str.lower):
            im_p = im_by_stem[k]
            seg_p = seg_by_stem[k]
            if im_p.is_file() and seg_p.is_file():
                pair_paths.append((im_p.resolve(), seg_p.resolve(), source))
    return all_im_keys, all_seg_keys, pair_paths


def _inventory_batch_display_title(batch_id: str, provider: str) -> str:
    """Human-readable batch line: site-style id + date/time when encoded as ``…_YYYYMMDD_HHMMSS``."""
    bid = (batch_id or "").strip()
    prov = (provider or "").strip() or "unknown"
    if bid == "on_disk":
        return "Legacy on-disk batch (pre-renamed synthetic id)"
    m = re.search(r"_(\d{8})_(\d{6})$", bid)
    if m:
        date_part, clock_part = m.group(1), m.group(2)
        ds = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}"
        ts = f"{clock_part[0:2]}:{clock_part[2:4]}:{clock_part[4:6]}"
        prefix = bid[: m.start()].strip("_") or prov
        site_disp = prefix.replace("_", " ").strip()
        if site_disp:
            site_disp = site_disp[0].upper() + site_disp[1:]
        else:
            site_disp = prov
        return f"{site_disp} · {ds} {ts} ET · {prov}"
    return f"{prov} · {bid}"


def _inventory_batch_sort_tuple(batch_id: str, created_at: str | None, mtime_ts: float) -> tuple[float]:
    """Sort key ascending: newest batch first (largest epoch → smallest negative)."""
    from datetime import datetime as _dt

    ts = float(mtime_ts or 0.0)
    if ts <= 0 and created_at:
        try:
            ca = str(created_at).replace("Z", "+00:00")
            ts = _dt.fromisoformat(ca).timestamp()
        except Exception:
            ts = 0.0
    if ts <= 0:
        m = re.search(r"_(\d{8})_(\d{6})(?:_|$)", batch_id or "")
        if m:
            try:
                ts = _dt.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
            except Exception:
                ts = 0.0
    return (-ts,)


def _provider_base_url(provider_name: str) -> str:
    p = (provider_name or "").strip().lower()
    if p == "openorganelle":
        return "https://openorganelle.janelia.org/"
    if p == "bossdb":
        return "https://api.bossdb.io/"
    return ""


def _stable_match_key(raw: str) -> str:
    s = _normalize_stable_id(raw or "").lower()
    s = re.sub(r"_vol\d+$", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _infer_provider_for_stable(conn: sqlite3.Connection, stable_id: str) -> str:
    """Infer provider from existing registry rows; default to OpenOrganelle."""
    sid = _normalize_stable_id(stable_id)
    # 1) Exact provenance from existing non-bootstrap batch items.
    row = conn.execute(
        """
        SELECT db.provider
        FROM batch_items bi
        JOIN download_batches db ON db.id = bi.batch_db_id
        WHERE bi.stable_id = ?
          AND INSTR(COALESCE(db.profile_json, ''), 'studio_inventory_bootstrap') = 0
        ORDER BY bi.id DESC
        LIMIT 1
        """,
        (sid,),
    ).fetchone()
    if row and row["provider"]:
        return str(row["provider"])

    # 2) Exact dataset stable_id match.
    row = conn.execute(
        """
        SELECT p.name AS provider
        FROM datasets d
        JOIN providers p ON p.id = d.provider_id
        WHERE d.stable_id = ?
          AND INSTR(COALESCE(d.metadata_json, ''), 'studio_inventory_bootstrap') = 0
          AND INSTR(COALESCE(d.metadata_json, ''), 'studio_inventory_sync') = 0
        ORDER BY d.id DESC
        LIMIT 1
        """,
        (sid,),
    ).fetchone()
    if row and row["provider"]:
        return str(row["provider"])

    # 3) Relaxed stable-id key match (handles transformed stems like *_vol1).
    # Prefer a unique provider match across existing dataset keys; this is far
    # more reliable than filename-shape heuristics.
    target_key = _stable_match_key(sid)
    if not target_key:
        return "OpenOrganelle"
    rows = conn.execute(
        """
        SELECT d.stable_id, p.name AS provider
        FROM datasets d
        JOIN providers p ON p.id = d.provider_id
        WHERE INSTR(COALESCE(d.metadata_json, ''), 'studio_inventory_bootstrap') = 0
          AND INSTR(COALESCE(d.metadata_json, ''), 'studio_inventory_sync') = 0
        """
    ).fetchall()
    provider_hits: dict[str, int] = {}
    for r in rows:
        dsid = str(r["stable_id"] or "")
        if _stable_match_key(dsid) == target_key:
            prov = str(r["provider"] or "Unknown")
            provider_hits[prov] = provider_hits.get(prov, 0) + 1
    if len(provider_hits) == 1:
        return next(iter(provider_hits.keys()))
    if provider_hits:
        # Ambiguous: keep deterministic by highest hit count, then name.
        return sorted(provider_hits.items(), key=lambda kv: (-kv[1], kv[0].lower()))[0][0]
    # 4) URL/shape heuristic only as a last resort.
    if "/" in sid:
        return "BossDB"
    # Fallback: avoid surfacing synthetic "Unknown" for common Stage-0 bootstrap rows.
    return "OpenOrganelle"


def _get_or_create_bootstrap_batch_id(
    conn: sqlite3.Connection,
    *,
    provider_name: str,
    dt_et: datetime,
    run_folder: str = "data/nnUNet_raw/Dataset001_mito2",
) -> int:
    from agent.orchestration.registry.api import (  # noqa: PLC0415
        DEFAULT_DOWNLOAD_PROFILE_HASH,
        create_download_batch,
    )

    prov = (provider_name or "").strip() or "Unknown"
    prov_slug = re.sub(r"[^a-z0-9]+", "_", prov.lower()).strip("_") or "unknown"
    prefix = "openorganelle_mito" if prov_slug == "openorganelle" else f"{prov_slug}_mito"
    batch_id = f"{prefix}_{to_us_eastern(dt_et).strftime('%Y%m%d_%H%M%S')}"
    profile: dict[str, Any] = {
        "batch_id": batch_id,
        "chunk_shape_zyx": [128, 128, 128],
        "n_crops": 1,
        "voxel_nm_zyx": [16.0, 16.0, 16.0],
        "foundation": (prov.lower() == "openorganelle"),
        "source": "studio_inventory_bootstrap",
        "provider": prov,
        "data_source": "inference" if "inference" in run_folder else "training",
    }
    return create_download_batch(
        conn,
        batch_id=batch_id,
        provider=prov,
        profile_hash=DEFAULT_DOWNLOAD_PROFILE_HASH,
        profile_json=profile,
        run_folder=run_folder,
    )


def _repair_bootstrap_batch_provider_labels(conn: sqlite3.Connection) -> None:
    """Relabel/split bootstrap batches so provider labels match actual source."""
    from agent.orchestration.registry.api import (  # noqa: PLC0415
        upsert_batch_item,
        upsert_dataset,
        upsert_provider,
    )

    rows = conn.execute(
        """
        SELECT id, batch_id, provider, created_at, profile_json
        FROM download_batches
        WHERE INSTR(COALESCE(profile_json, ''), 'studio_inventory_bootstrap') > 0
        ORDER BY id ASC
        """
    ).fetchall()

    batch_target_cache: dict[tuple[int, str], int] = {}

    def _batch_dt_eastern(batch_id: str, created_at: str | None) -> datetime:
        m = re.search(r"_(\d{8})_(\d{6})$", batch_id or "")
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=US_EASTERN_TZ)
            except Exception:
                pass
        if created_at:
            try:
                return to_us_eastern(datetime.fromisoformat(str(created_at).replace("Z", "+00:00")))
            except Exception:
                pass
        return now_us_eastern()

    def _target_batch_id_for_provider(row: sqlite3.Row, provider_name: str) -> int:
        key = (int(row["id"]), provider_name)
        if key in batch_target_cache:
            return int(batch_target_cache[key])
        cur_provider = str(row["provider"] or "")
        cur_bid = str(row["batch_id"] or "")
        if cur_provider == provider_name:
            batch_target_cache[key] = int(row["id"])
            return int(row["id"])
        dt_et = _batch_dt_eastern(cur_bid, row["created_at"])
        new_id = _get_or_create_bootstrap_batch_id(conn, provider_name=provider_name, dt_et=dt_et)
        batch_target_cache[key] = int(new_id)
        return int(new_id)

    for r in rows:
        bid = int(r["id"])
        irows = conn.execute(
            """
            SELECT id, stable_id, asset_type, local_path, status, completed_at
            FROM batch_items
            WHERE batch_db_id = ?
            """,
            (bid,),
        ).fetchall()
        if not irows:
            continue
        for ir in irows:
            stable = str(ir["stable_id"] or "")
            provider_name = _infer_provider_for_stable(conn, stable)
            if not provider_name:
                provider_name = "Unknown"
            target_bid = _target_batch_id_for_provider(r, provider_name)
            if target_bid == bid:
                continue
            pid = upsert_provider(conn, name=provider_name, base_url=_provider_base_url(provider_name))
            did = upsert_dataset(
                conn,
                provider_id=pid,
                stable_id=stable,
                metadata={"source": "studio_inventory_bootstrap_repair"},
                changed=False,
            )
            upsert_batch_item(
                conn,
                batch_db_id=target_bid,
                dataset_id=did,
                stable_id=stable,
                asset_type=str(ir["asset_type"] or ""),
                local_path=str(ir["local_path"] or ""),
                status=str(ir["status"] or "present"),
            )
            conn.execute("DELETE FROM batch_items WHERE id = ?", (int(ir["id"]),))

        # If no items remain in the original batch, drop it.
        left = int(conn.execute("SELECT COUNT(*) FROM batch_items WHERE batch_db_id = ?", (bid,)).fetchone()[0])
        if left == 0:
            conn.execute("DELETE FROM download_batches WHERE id = ?", (bid,))
            continue

        # Normalize provider label for remaining (single-provider) rows.
        srows = conn.execute(
            "SELECT DISTINCT stable_id FROM batch_items WHERE batch_db_id = ?",
            (bid,),
        ).fetchall()
        counts: dict[str, int] = {}
        for sr in srows:
            prov = _infer_provider_for_stable(conn, str(sr["stable_id"] or ""))
            counts[prov] = counts.get(prov, 0) + 1
        best = max(counts.items(), key=lambda kv: kv[1])[0] if counts else str(r["provider"] or "Unknown")
        if best:
            batch_id = str(r["batch_id"] or "")
            new_batch_id = batch_id
            m = re.search(r"_(\d{8})_(\d{6})$", batch_id)
            if m:
                suffix = f"{m.group(1)}_{m.group(2)}"
                prov_slug = re.sub(r"[^a-z0-9]+", "_", best.lower()).strip("_") or "unknown"
                new_batch_id = f"{prov_slug}_mito_{suffix}"
            conn.execute(
                "UPDATE download_batches SET provider = ?, batch_id = ? WHERE id = ?",
                (best, new_batch_id, bid),
            )


def _bootstrap_training_h5_pairs_into_registry(
    conn: sqlite3.Connection,
    paired_paths: list[tuple[Path, Path, str]],
) -> None:
    """Persist ``download_batches`` + ``batch_items`` for existing paired H5 files.

    Inventory rows must remain in the registry when files are deleted so status
    becomes ``missing_or_deleted_local`` and the table row count stays fixed until
    genuinely new datasets are recorded.
    """
    from datetime import datetime

    from agent.orchestration.registry.api import update_batch_status, upsert_batch_item, upsert_dataset, upsert_provider  # noqa: PLC0415

    max_ts = 0.0
    for im_p, seg_p, _data_source in paired_paths:
        for p in (im_p, seg_p):
            try:
                max_ts = max(max_ts, float(p.stat().st_mtime))
            except OSError:
                pass
    if max_ts <= 0.0:
        max_ts = now_us_eastern().timestamp()
    dt_et = datetime.fromtimestamp(max_ts, tz=US_EASTERN_TZ)
    batch_ids_by_provider: dict[tuple[str, str], int] = {}
    pair_count_by_provider: dict[tuple[str, str], int] = {}
    for im_p, seg_p, data_source in sorted(paired_paths, key=lambda t: t[0].name.lower()):
        if not im_p.is_file() or not seg_p.is_file():
            continue
        stable = _normalize_stable_id(im_p.name)
        provider_name = _infer_provider_for_stable(conn, stable)
        k = (provider_name, data_source)
        run_folder = f"data/{data_source}"
        if k not in batch_ids_by_provider:
            batch_ids_by_provider[k] = _get_or_create_bootstrap_batch_id(
                conn, provider_name=provider_name, dt_et=dt_et, run_folder=run_folder
            )
            pair_count_by_provider[k] = 0
        batch_db_id = int(batch_ids_by_provider[k])
        pid = upsert_provider(
            conn,
            name=provider_name,
            base_url=_provider_base_url(provider_name),
        )
        did = upsert_dataset(
            conn,
            provider_id=pid,
            stable_id=stable,
            metadata={"source": "studio_inventory_bootstrap"},
            changed=False,
        )
        upsert_batch_item(
            conn,
            batch_db_id=batch_db_id,
            dataset_id=did,
            stable_id=stable,
            asset_type="em_volume",
            local_path=str(im_p.resolve()),
            status="present",
        )
        upsert_batch_item(
            conn,
            batch_db_id=batch_db_id,
            dataset_id=did,
            stable_id=stable,
            asset_type="mito_seg",
            local_path=str(seg_p.resolve()),
            status="present",
        )
        pair_count_by_provider[k] = int(pair_count_by_provider.get(k, 0)) + 1
    if not pair_count_by_provider:
        for bid in batch_ids_by_provider.values():
            conn.execute("DELETE FROM download_batches WHERE id = ?", (int(bid),))
        return
    for (provider_name, _data_source), n_pairs in pair_count_by_provider.items():
        bid = int(batch_ids_by_provider[(provider_name, _data_source)])
        row = conn.execute("SELECT profile_json FROM download_batches WHERE id = ?", (bid,)).fetchone()
        try:
            profile = json.loads(str(row["profile_json"] or "{}")) if row else {}
        except Exception:
            profile = {}
        em_seg_units = 2 * int(n_pairs)
        profile["download_asset_completions"] = em_seg_units
        profile["planned_asset_downloads"] = em_seg_units
        profile["datasets_this_run"] = int(n_pairs)
        conn.execute(
            """
            UPDATE download_batches
            SET download_asset_completions = ?, profile_json = ?
            WHERE id = ?
            """,
            (int(em_seg_units), json.dumps(profile, ensure_ascii=False), bid),
        )
        update_batch_status(conn, bid, "complete")
    conn.commit()


def _sync_new_training_pairs_into_registry(
    conn: sqlite3.Connection,
    paired_paths: list[tuple[Path, Path, str]],
) -> None:
    """Register any new training/inference H5 pairs not yet present as ``batch_items``.

    The one-time bootstrap only runs when the join table is empty; this keeps
    ``tracked`` counts in sync when additional datasets are copied onto disk.
    Existing rows keep their ``batch_db_id`` (provenance); only ``local_path`` /
    ``status`` are refreshed when a row already exists.
    """
    from agent.orchestration.registry.api import upsert_batch_item, upsert_dataset, upsert_provider

    target_bid_by_provider: dict[tuple[str, str], int] = {}
    pid_by_provider: dict[str, int] = {}
    now_iso = now_us_eastern_iso(timespec="seconds")
    for im_p, seg_p, data_source in sorted(paired_paths, key=lambda t: t[0].name.lower()):
        if not im_p.is_file() or not seg_p.is_file():
            continue
        stable = _normalize_stable_id(im_p.name)
        stable_norm = stable.lower().replace("-", "_")
        provider_name = _infer_provider_for_stable(conn, stable)
        k = (provider_name, data_source)
        run_folder = f"data/{data_source}"
        if k not in target_bid_by_provider:
            br = conn.execute(
                """
                SELECT id FROM download_batches
                WHERE provider = ?
                  AND COALESCE(run_folder, '') = ?
                  AND (
                    INSTR(COALESCE(profile_json, ''), 'studio_inventory_bootstrap') > 0
                    OR batch_id LIKE ?
                  )
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    provider_name,
                    run_folder,
                    f"{re.sub(r'[^a-z0-9]+', '_', provider_name.lower()).strip('_') or 'unknown'}_mito_%",
                ),
            ).fetchone()
            if br is None:
                br = conn.execute(
                    """
                    SELECT id FROM download_batches
                    WHERE provider = ?
                      AND COALESCE(run_folder, '') = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (provider_name, run_folder),
                ).fetchone()
            if br is None:
                br_id = _get_or_create_bootstrap_batch_id(
                    conn, provider_name=provider_name, dt_et=now_us_eastern(), run_folder=run_folder
                )
            else:
                br_id = int(br["id"])
            target_bid_by_provider[k] = int(br_id)
        if provider_name not in pid_by_provider:
            pid_by_provider[provider_name] = upsert_provider(
                conn,
                name=provider_name,
                base_url=_provider_base_url(provider_name),
            )
        target_bid = int(target_bid_by_provider[k])
        pid = int(pid_by_provider[provider_name])
        did = upsert_dataset(
            conn,
            provider_id=pid,
            stable_id=stable,
            metadata={"source": "studio_inventory_sync"},
            changed=False,
        )
        for asset_type, path in (("em_volume", im_p), ("mito_seg", seg_p)):
            lp = str(path.resolve())
            existing = conn.execute(
                """
                SELECT id FROM batch_items
                WHERE REPLACE(LOWER(stable_id), '-', '_') = ? AND asset_type = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (stable_norm, asset_type),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    """
                    UPDATE batch_items
                    SET local_path = ?,
                        status = 'present',
                        completed_at = ?,
                        dataset_id = COALESCE(?, dataset_id)
                    WHERE id = ?
                    """,
                    (lp, now_iso, did, int(existing["id"])),
                )
            else:
                upsert_batch_item(
                    conn,
                    batch_db_id=target_bid,
                    dataset_id=did,
                    stable_id=stable,
                    asset_type=asset_type,
                    local_path=lp,
                    status="present",
                )


def _is_inventory_excluded_delete_audit_path(path_like: str | Path) -> bool:
    """Stage-0 inventory ignores nnUNet instance-label dirs (not primary train/infer pairs)."""
    s = str(path_like or "").replace("\\", "/").lower()
    return "/labelstr-instance/" in s or "/labelsts-instance/" in s


def _append_deletion_events(
    registry_path: Path,
    *,
    provider_name: str,
    deleted_paths: list[str],
) -> None:
    """Persist file-level delete audit rows; creates registry DB + table if needed."""
    if not deleted_paths:
        return
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    from agent.orchestration.registry.schema import open_registry  # noqa: PLC0415

    conn = open_registry(registry_path)
    try:
        now = now_us_eastern_iso(timespec="seconds")
        for fp_str in sorted(set(deleted_paths)):
            if _is_inventory_excluded_delete_audit_path(fp_str):
                continue
            p = Path(fp_str)
            conn.execute(
                """
                INSERT INTO deletion_events (provider, stable_id, asset_type, local_path, deleted_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    provider_name,
                    _normalize_stable_id(p.name),
                    _asset_type_from_local_path(p),
                    str(p),
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


@router.post("/datasets/delete")
def studio_delete_datasets(body: DatasetDeleteBody) -> dict[str, Any]:
    """Hard-delete selected training datasets from disk and mark them missing in registry.

    Deletes matching files under ``data/nnUNet_raw/Dataset001_mito2`` whose
    filename stem starts with the stable_id.  Registry batch_items for those
    datasets are updated to ``missing_or_deleted_local``.
    """
    root = _root()
    dataset_base = nnunet_dataset_root(root)
    training_base = dataset_base / "imagesTr"
    inference_base = dataset_base / "imagesTs"
    labels_training_base = dataset_base / "labelsTr"
    labels_training_instance_base = dataset_base / "labelsTr-instance"
    labels_inference_base = dataset_base / "labelsTs"
    labels_inference_instance_base = dataset_base / "labelsTs-instance"
    postprocess_input_base = root / "data" / "outputs" / "bc"
    postprocess_output_base = root / "data" / "outputs" / "postprocessed"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if not body.stable_ids and not body.file_paths:
        raise HTTPException(status_code=400, detail="Provide stable_ids or file_paths")

    deleted_files: list[str] = []
    registry_paths_to_mark_missing: set[str] = set()
    registry_stable_assets_to_mark_missing: set[tuple[str, str | None]] = set()
    registry_key_assets_to_mark_missing: list[tuple[str, str | None]] = []
    errors: list[str] = []

    # Delete explicit file paths first (single-row delete in Existing Data table).
    allowed_roots = (
        training_base.resolve(),
        inference_base.resolve(),
        labels_training_base.resolve(),
        labels_training_instance_base.resolve(),
        labels_inference_base.resolve(),
        labels_inference_instance_base.resolve(),
        postprocess_input_base.resolve(),
        postprocess_output_base.resolve(),
    )
    for fp in body.file_paths:
        fp_raw = (fp or "").strip()
        if not fp_raw:
            continue
        try:
            p = Path(fp_raw)
            p = (root / p).resolve() if not p.is_absolute() else p.resolve()
        except Exception:
            errors.append(f"Invalid file path '{fp_raw}'")
            continue
        if not any(base == p or base in p.parents for base in allowed_roots):
            errors.append(f"Path outside allowed data roots: {p}")
            continue
        registry_paths_to_mark_missing.add(str(p))
        key_guess = _stable_key_from_filename(p)
        if key_guess:
            registry_key_assets_to_mark_missing.append((key_guess, _asset_type_from_local_path(p)))
        try:
            if p.is_file():
                p.unlink()
                deleted_files.append(str(p))
            elif p.is_dir():
                import shutil as _sh
                _sh.rmtree(p)
                deleted_files.append(str(p))
            else:
                # Already absent: still mark corresponding batch_items missing in registry.
                pass
        except OSError as exc:
            errors.append(f"Could not delete {p}: {exc}")

    # Delete training files for each stable_id.
    for sid in body.stable_ids:
        sid = _normalize_stable_id(sid)
        if not sid:
            continue
        registry_stable_assets_to_mark_missing.add((sid, None))
        # Safety: no path traversal.
        if ".." in sid or "/" in sid or "\\" in sid:
            errors.append(f"Invalid stable_id '{sid}'")
            continue
        for search_dir in (training_base, labels_training_base, inference_base, labels_inference_base):
            if not search_dir.is_dir():
                continue
            for p in list(search_dir.iterdir()):
                norm = _normalize_stable_id(p.name).lower()
                stem_l = p.stem.lower()
                if norm == sid.lower() or stem_l.startswith(sid.lower() + "_"):
                    try:
                        if p.is_file():
                            p.unlink()
                        elif p.is_dir():
                            import shutil as _sh
                            _sh.rmtree(p)
                        deleted_files.append(str(p))
                        registry_paths_to_mark_missing.add(str(p.resolve()))
                    except OSError as exc:
                        errors.append(f"Could not delete {p}: {exc}")

    # Update registry: mark batch_items as missing_or_deleted_local.
    rp = _registry_path(root)
    if rp.is_file():
        try:
            from agent.orchestration.registry.schema import open_registry  # noqa: PLC0415
            from agent.orchestration.registry.api import (  # noqa: PLC0415
                delete_batch_item_by_path,
                get_provider_id as _get_pid,
                get_dataset_id as _get_did,
                hide_dataset as _hide,
            )
            conn = open_registry(rp)
            try:
                for fp in sorted(set(deleted_files) | registry_paths_to_mark_missing):
                    delete_batch_item_by_path(conn, fp)
                for sid, asset_type in sorted(registry_stable_assets_to_mark_missing):
                    if asset_type:
                        conn.execute(
                            """
                            UPDATE batch_items
                            SET status = 'missing_or_deleted_local'
                            WHERE stable_id = ? AND asset_type = ? AND status = 'present'
                            """,
                            (sid, asset_type),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE batch_items
                            SET status = 'missing_or_deleted_local'
                            WHERE stable_id = ? AND status = 'present'
                            """,
                            (sid,),
                        )
                for key_guess, asset_type in registry_key_assets_to_mark_missing:
                    rows = conn.execute(
                        """
                        SELECT stable_id
                        FROM datasets
                        WHERE REPLACE(LOWER(stable_id), '-', '_') = ?
                           OR ? LIKE REPLACE(LOWER(stable_id), '-', '_') || '_%'
                        """,
                        (key_guess, key_guess),
                    ).fetchall()
                    for r in rows:
                        sid = str(r["stable_id"])
                        if asset_type:
                            conn.execute(
                                """
                                UPDATE batch_items
                                SET status = 'missing_or_deleted_local'
                                WHERE stable_id = ? AND asset_type = ? AND status = 'present'
                                """,
                                (sid, asset_type),
                            )
                        else:
                            conn.execute(
                                """
                                UPDATE batch_items
                                SET status = 'missing_or_deleted_local'
                                WHERE stable_id = ? AND status = 'present'
                                """,
                                (sid,),
                            )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            errors.append(f"Registry update warning: {exc}")

    # Deletion audit log: always record successful file deletes, even when the
    # registry file did not exist yet (previously inserts only ran inside ``if rp.is_file()``).
    if deleted_files:
        try:
            _append_deletion_events(
                rp,
                provider_name=body.provider.strip() or "OpenOrganelle",
                deleted_paths=[
                    p for p in deleted_files if not _is_inventory_excluded_delete_audit_path(p)
                ],
            )
        except Exception as exc:
            errors.append(f"Deletion history log failed: {exc}")

    return {
        "ok": len(errors) == 0,
        "deleted_files": deleted_files,
        "errors": errors,
        "message": (
            f"Deleted {len(deleted_files)} file(s)."
            if not errors
            else f"Deleted {len(deleted_files)} file(s) with {len(errors)} error(s)."
        ),
    }


class DatasetHideBody(BaseModel):
    stable_ids: list[str] = Field(..., min_length=1)
    provider: str = Field(default="OpenOrganelle")


class DatasetUseInModelBody(BaseModel):
    stable_id: str = Field(..., min_length=1)
    source: str = Field(default="inference")
    use_in_model: bool = Field(default=True)
    provider: str = Field(default="OpenOrganelle")


def _dataset001_use_in_model_state_path(root: Path) -> Path:
    dataset_root = nnunet_dataset_root(root)
    return dataset_root / ".studio_use_in_model_state.json"


def _load_dataset001_use_in_model_state(root: Path) -> dict[str, Any]:
    path = _dataset001_use_in_model_state_path(root)
    if not path.is_file():
        return {"hidden_inference": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"hidden_inference": []}
    if not isinstance(data, dict):
        return {"hidden_inference": []}
    hidden_inference = data.get("hidden_inference", [])
    if not isinstance(hidden_inference, list):
        hidden_inference = []
    return {
        "hidden_inference": sorted({
            _normalize_stable_id(str(v))
            for v in hidden_inference
            if _normalize_stable_id(str(v))
        }),
    }


def _save_dataset001_use_in_model_state(root: Path, state: dict[str, Any]) -> None:
    path = _dataset001_use_in_model_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _load_dataset001_use_in_model_state(root)
    hidden_inference = state.get("hidden_inference", normalized.get("hidden_inference", []))
    normalized["hidden_inference"] = sorted({
        _normalize_stable_id(str(v))
        for v in (hidden_inference if isinstance(hidden_inference, list) else [])
        if _normalize_stable_id(str(v))
    })
    path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


def _sync_dataset001_dataset_json(root: Path, provider: str = "OpenOrganelle") -> dict[str, Any]:
    dataset_root = nnunet_dataset_root(root)
    dataset_json_path = (dataset_root / "dataset.json").resolve()
    images_tr = (dataset_root / "imagesTr").resolve()
    labels_tr = (dataset_root / "labelsTr").resolve()
    images_ts = (dataset_root / "imagesTs").resolve()
    for p in (dataset_root, images_tr, labels_tr, images_ts):
        p.mkdir(parents=True, exist_ok=True)

    hidden_training: set[str] = set()
    rp = _registry_path(root)
    if rp.is_file():
        try:
            from agent.orchestration.registry.schema import open_registry  # noqa: PLC0415
            from agent.orchestration.registry.api import (  # noqa: PLC0415
                get_provider_id as _gpid,
            )
            conn = open_registry(rp)
            try:
                pid = _gpid(conn, provider.strip() or "OpenOrganelle")
                if pid is not None:
                    rows = conn.execute(
                        "SELECT stable_id FROM datasets WHERE provider_id = ? AND COALESCE(hidden_from_training, 0) != 0",
                        (pid,),
                    ).fetchall()
                    hidden_training = {_normalize_stable_id(r["stable_id"]) for r in rows if r["stable_id"]}
            finally:
                conn.close()
        except Exception:
            hidden_training = set()

    use_state = _load_dataset001_use_in_model_state(root)
    hidden_inference = {
        _normalize_stable_id(v) for v in use_state.get("hidden_inference", []) if _normalize_stable_id(v)
    }

    training_entries: list[dict[str, str]] = []
    for img in sorted(images_tr.glob("*_0000.nii.gz"), key=lambda x: x.name.lower()):
        sid = _normalize_stable_id(img.name)
        if sid in hidden_training:
            continue
        lbl = labels_tr / f"{img.name[:-12]}.nii.gz"
        if not lbl.is_file():
            continue
        training_entries.append({"image": f"./imagesTr/{img.name}", "label": f"./labelsTr/{lbl.name}"})

    test_entries: list[str] = []
    for img in sorted(images_ts.glob("*_0000.nii.gz"), key=lambda x: x.name.lower()):
        sid = _normalize_stable_id(img.name)
        if sid in hidden_inference:
            continue
        test_entries.append(f"./imagesTs/{img.name}")

    prev: dict[str, Any] = {}
    if dataset_json_path.is_file():
        try:
            loaded = json.loads(dataset_json_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                prev = loaded
        except Exception:
            prev = {}

    data = dict(prev)
    data["channel_names"] = {"0": "em"}
    data["labels"] = {"background": 0, "mitochondria": 1, "contour": 2}
    data["file_ending"] = ".nii.gz"
    data["name"] = str(data.get("name") or "Dataset001_mito2")
    data["description"] = str(data.get("description") or "Dataset001_mito2 dataset")
    data["numTraining"] = len(training_entries)
    data["numTest"] = len(test_entries)
    data["training"] = training_entries
    data["test"] = test_entries
    dataset_json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {
        "dataset_json_path": str(dataset_json_path),
        "numTraining": len(training_entries),
        "numTest": len(test_entries),
    }


@router.post("/datasets/hide")
def studio_hide_datasets(body: DatasetHideBody) -> dict[str, Any]:
    """Soft-hide datasets from training datalist without deleting files."""
    root = _root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rp = _registry_path(root)
    hidden: list[str] = []
    not_found: list[str] = []
    from agent.orchestration.registry.schema import open_registry  # noqa: PLC0415
    from agent.orchestration.registry.api import (  # noqa: PLC0415
        upsert_provider as _upsert_provider,
        get_dataset_id as _gdid,
        upsert_dataset as _upsert_dataset,
        hide_dataset as _hide,
    )
    conn = open_registry(rp)
    try:
        provider_name = body.provider.strip() or "OpenOrganelle"
        provider_id = _upsert_provider(conn, name=provider_name)
        for sid in body.stable_ids:
            sid = _normalize_stable_id(sid)
            if not sid:
                continue
            did = _gdid(conn, provider_id, sid)
            if did is None:
                did = _upsert_dataset(
                    conn,
                    provider_id=provider_id,
                    stable_id=sid,
                    display_name=sid,
                    metadata={"source": "studio-data-existing"},
                    changed=False,
                )
            _hide(conn, did)
            hidden.append(sid)
        conn.commit()
    finally:
        conn.close()

    sync_info = _sync_dataset001_dataset_json(root, provider=body.provider)
    return {
        "ok": True,
        "hidden": hidden,
        "not_found": not_found,
        "message": f"Hidden {len(hidden)} dataset(s) from training.",
        "dataset_json": sync_info,
    }


@router.post("/datasets/unhide")
def studio_unhide_datasets(body: DatasetHideBody) -> dict[str, Any]:
    """Re-include datasets in training datalist generation."""
    root = _root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rp = _registry_path(root)
    unhidden: list[str] = []
    not_found: list[str] = []
    from agent.orchestration.registry.schema import open_registry  # noqa: PLC0415
    from agent.orchestration.registry.api import (  # noqa: PLC0415
        upsert_provider as _upsert_provider,
        get_dataset_id as _gdid,
        upsert_dataset as _upsert_dataset,
        unhide_dataset as _unhide,
    )
    conn = open_registry(rp)
    try:
        provider_name = body.provider.strip() or "OpenOrganelle"
        provider_id = _upsert_provider(conn, name=provider_name)
        for sid in body.stable_ids:
            sid = _normalize_stable_id(sid)
            if not sid:
                continue
            did = _gdid(conn, provider_id, sid)
            if did is None:
                did = _upsert_dataset(
                    conn,
                    provider_id=provider_id,
                    stable_id=sid,
                    display_name=sid,
                    metadata={"source": "studio-data-existing"},
                    changed=False,
                )
            _unhide(conn, did)
            unhidden.append(sid)
        conn.commit()
    finally:
        conn.close()

    sync_info = _sync_dataset001_dataset_json(root, provider=body.provider)
    return {
        "ok": True,
        "unhidden": unhidden,
        "not_found": not_found,
        "message": f"Unhid {len(unhidden)} dataset(s); re-included in training.",
        "dataset_json": sync_info,
    }


@router.post("/datasets/use-in-model")
def studio_set_dataset_use_in_model(body: DatasetUseInModelBody) -> dict[str, Any]:
    root = _root()
    source = (body.source or "").strip().lower()
    stable_id = _normalize_stable_id(body.stable_id)
    if source not in {"training", "inference"}:
        raise HTTPException(status_code=400, detail="source must be 'training' or 'inference'.")
    if not stable_id:
        raise HTTPException(status_code=400, detail="Invalid stable_id.")

    if source == "training":
        if body.use_in_model:
            studio_unhide_datasets(DatasetHideBody(stable_ids=[stable_id], provider=body.provider))
        else:
            studio_hide_datasets(DatasetHideBody(stable_ids=[stable_id], provider=body.provider))
    else:
        state = _load_dataset001_use_in_model_state(root)
        hidden_inference = {
            _normalize_stable_id(v) for v in state.get("hidden_inference", []) if _normalize_stable_id(v)
        }
        if body.use_in_model:
            hidden_inference.discard(stable_id)
        else:
            hidden_inference.add(stable_id)
        state["hidden_inference"] = sorted(hidden_inference)
        _save_dataset001_use_in_model_state(root, state)

    sync_info = _sync_dataset001_dataset_json(root, provider=body.provider)
    return {
        "ok": True,
        "stable_id": stable_id,
        "source": source,
        "use_in_model": bool(body.use_in_model),
        "dataset_json": sync_info,
        "message": f"Updated '{stable_id}' use-in-model for {source}.",
    }


@router.get("/datasets/status")
def studio_datasets_status(
    provider: str = Query(default="OpenOrganelle"),
) -> dict[str, Any]:
    """Return all known training datasets with their registry status (hidden, batch provenance).

    Merges data from registry (hidden flag, batch membership) with the
    filesystem scan of ``data/nnUNet_raw/Dataset001_mito2``.
    """
    root = _root()
    pre_base = nnunet_dataset_root(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rp = _registry_path(root)
    registry_info: dict[str, dict[str, Any]] = {}
    use_state = _load_dataset001_use_in_model_state(root)
    hidden_inference = {
        _normalize_stable_id(v)
        for v in use_state.get("hidden_inference", [])
        if _normalize_stable_id(v)
    }

    if rp.is_file():
        try:
            from agent.orchestration.registry.schema import open_registry  # noqa: PLC0415
            from agent.orchestration.registry.api import (  # noqa: PLC0415
                get_provider_id as _gpid,
                list_datasets as _ldsets,
            )
            conn = open_registry(rp)
            try:
                pid = _gpid(conn, provider.strip() or "OpenOrganelle")
                if pid is not None:
                    # Build a map: stable_id → {hidden, batch_ids}
                    rows = conn.execute(
                        """
                        SELECT d.stable_id, d.hidden_from_training,
                               GROUP_CONCAT(DISTINCT db.batch_id) AS batch_ids
                        FROM datasets d
                        LEFT JOIN batch_items bi ON bi.dataset_id = d.id
                        LEFT JOIN download_batches db ON db.id = bi.batch_db_id
                        WHERE d.provider_id = ?
                        GROUP BY d.id
                        """,
                        (pid,),
                    ).fetchall()
                    for r in rows:
                        batch_ids = [b for b in (r["batch_ids"] or "").split(",") if b]
                        registry_info[r["stable_id"]] = {
                            "hidden_from_training": bool(r["hidden_from_training"]),
                            "batch_ids": batch_ids,
                        }
            finally:
                conn.close()
        except Exception as exc:
            print(f"[WARN] studio_datasets_status: registry read error: {exc}", file=sys.stderr)

    # Scan nnUNet dataset directory for train/test image/label files.
    datasets_out: list[dict[str, Any]] = []
    seen_stems: set[str] = set()
    if pre_base.is_dir():
        for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs", ""):
            scan_dir = (pre_base / sub) if sub else pre_base
            if not scan_dir.is_dir():
                continue
            for p in sorted(scan_dir.iterdir(), key=lambda x: x.name.lower()):
                if not p.is_file():
                    continue
                if not p.suffix.lower() in (".h5", ".nrrd", ".nii", ".gz", ".pt"):
                    continue
                # Normalize from full filename so ``*.nii.gz`` and ``*_0000`` resolve
                # to the same stable_id used by hide/unhide and registry rows.
                stem = _normalize_stable_id(p.name)
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)
                reg = registry_info.get(stem, {})
                datasets_out.append({
                    "stable_id": stem,
                    "filename": p.name,
                    "path": str(p),
                    "size_bytes": p.stat().st_size if p.is_file() else 0,
                    "hidden_from_training": bool(reg.get("hidden_from_training", False)),
                    "hidden_from_inference": stem in hidden_inference,
                    "batch_ids": reg.get("batch_ids", []),
                })

    return {
        "ok": True,
        "provider": provider,
        "preprocessed_base": str(pre_base),
        "datasets": datasets_out,
    }


@router.get("/inventory/catalogue")
def studio_inventory_catalogue(
    response: Response,
    provider: str = Query(default="", description="Filter by provider name (empty = all)"),
) -> dict[str, Any]:
    """Return a normalised inventory of all tracked download batches and their item-level statuses.

    Reconciles each batch against disk before returning so ``status`` fields are
    always fresh.  Returns a graceful empty response when the registry does not
    exist yet (no 503 — callers display an informative empty-state UI).

    Response shape
    --------------
    {
      ok: bool,
      registry_exists: bool,
      summary: {
        total_items: int,              # registry batch_items only (stable per provider)
        inventory_row_count: int,     # len(rows); same as total_items for normal catalogue responses
        present: int,                 # registry items with status present
        missing_or_deleted: int,      # registry items with status missing_or_deleted_local
        deletion_events_count: int,   # rows in deletion_events (file-level delete log)
        download_completions_total: int,  # SUM(batch download_asset_completions): EM+mito_seg units (pairs×2)
        pending: int, failed: int, hidden_from_training: int,
        on_disk_pairs: int,
        on_disk_pairs_training: int,
        on_disk_pairs_inference: int,
        providers: {<name>: <count>}, # counts from registry rows only (per website, stable)
        batches: [{batch_id, display_title, provider, profile_hash, profile, created_at,
                   n_items, n_present, n_missing}]
      },
      rows: [{item_id, stable_id, provider, batch_id, asset_type, status,
              local_path, hidden_from_training, completed_at, batch_created_at,
              profile_hash, profile}]
    }
    """
    root = _root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rp = _registry_path(root)

    _EMPTY_SUMMARY: dict[str, Any] = {
        "total_items": 0,
        "inventory_row_count": 0,
        "distinct_datasets": 0,
        "present": 0,
        "missing_or_deleted": 0,
        "deletion_events_count": 0,
        "download_completions_total": 0,
        "pending": 0,
        "failed": 0,
        "hidden_from_training": 0,
        "on_disk_pairs": 0,
        "on_disk_pairs_training": 0,
        "on_disk_pairs_inference": 0,
        "on_disk_images": 0,
        "on_disk_labels": 0,
        "delete_history": [],
        "providers": {},
        "batches": [],
    }

    try:
        from agent.orchestration.registry.schema import open_registry  # noqa: PLC0415
        from agent.orchestration.registry.api import (  # noqa: PLC0415
            list_download_batches as _list_batches,
            reconcile_batch_items as _reconcile,
        )
    except Exception as exc:
        response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return {"ok": False, "registry_exists": True, "error": str(exc),
                "summary": _EMPTY_SUMMARY, "rows": []}

    conn = open_registry(rp)
    try:
        provider_filter = (provider or "").strip()

        # Clean up only stale pre-registered shells (no batch_items + still in_progress).
        # Do NOT delete finalized zero-item rows, since a real re-download run may
        # intentionally log completions via profile counters even when item-path
        # assembly could not materialize batch_items.
        try:
            with _downloader_state_lock:
                any_dl_running = any(bool(v.get("running")) for v in _downloader_state.values())
            if not any_dl_running:
                conn.execute(
                    """
                    DELETE FROM download_batches
                    WHERE id IN (
                        SELECT db.id
                        FROM download_batches db
                        LEFT JOIN batch_items bi ON bi.batch_db_id = db.id
                        GROUP BY db.id
                        HAVING COUNT(bi.id) = 0
                    )
                      AND COALESCE(status, '') = 'in_progress'
                    """
                )
                conn.commit()
        except Exception:
            # Non-fatal hygiene pass; inventory should still load.
            pass

        # Reconcile all batches so statuses reflect current disk state.
        try:
            _repair_bootstrap_batch_provider_labels(conn)
            conn.commit()
        except Exception:
            pass
        batches_raw = _list_batches(conn, provider_filter or None)
        for b in batches_raw:
            try:
                _reconcile(conn, int(b["id"]))
            except Exception:
                pass
        conn.commit()

        img_keys, lbl_keys, on_disk_paired_paths = _data_h5_paired_paths(root)

        # When the registry has no batch_items yet but paired H5s exist on disk,
        # materialize rows once in SQLite so deletes / re-downloads only flip status
        # and never shrink the inventory table. After that, each catalogue load still
        # syncs any *new* on-disk pairs so tracked counts follow the filesystem.
        if not provider_filter:
            n_joined = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM batch_items bi
                    JOIN download_batches db ON db.id = bi.batch_db_id
                    """
                ).fetchone()[0]
            )
            n_batches = int(conn.execute("SELECT COUNT(*) FROM download_batches").fetchone()[0])
            if n_joined == 0 and n_batches == 0:
                if on_disk_paired_paths:
                    try:
                        _bootstrap_training_h5_pairs_into_registry(conn, on_disk_paired_paths)
                        conn.commit()
                        for b in _list_batches(conn, None):
                            try:
                                _reconcile(conn, int(b["id"]))
                            except Exception:
                                pass
                        conn.commit()
                    except Exception as exc:
                        logging.warning(
                            "inventory bootstrap from training H5 pairs failed: %s", exc
                        )
            elif on_disk_paired_paths:
                try:
                    _sync_new_training_pairs_into_registry(conn, on_disk_paired_paths)
                    conn.commit()
                    for b in _list_batches(conn, None):
                        try:
                            _reconcile(conn, int(b["id"]))
                        except Exception:
                            pass
                    conn.commit()
                except Exception as exc:
                    logging.warning(
                        "inventory sync new training pairs failed: %s", exc
                    )

        # Single JOIN query for all inventory rows.
        query = """
            SELECT
              bi.id            AS item_id,
              bi.stable_id,
              bi.asset_type,
              bi.local_path,
              bi.status        AS item_status,
              bi.completed_at,
              db.batch_id,
              db.provider,
              db.profile_hash,
              db.profile_json,
              db.run_folder,
              db.created_at    AS batch_created_at,
              db.status        AS batch_status,
              COALESCE(d.hidden_from_training, 0) AS hidden_from_training
            FROM batch_items bi
            JOIN download_batches db ON db.id = bi.batch_db_id
            LEFT JOIN datasets d ON d.id = bi.dataset_id
            {where}
            ORDER BY db.created_at ASC, bi.stable_id
        """
        if provider_filter:
            rows_raw = conn.execute(
                query.format(where="WHERE db.provider = ?"), (provider_filter,)
            ).fetchall()
        else:
            rows_raw = conn.execute(query.format(where="")).fetchall()

        stats_sql = """
            SELECT
              COUNT(*) AS n_total,
              COALESCE(SUM(CASE WHEN bi.status = 'present' THEN 1 ELSE 0 END), 0) AS n_present,
              COALESCE(SUM(CASE WHEN bi.status = 'missing_or_deleted_local' THEN 1 ELSE 0 END), 0)
                AS n_missing,
              COALESCE(SUM(CASE WHEN bi.status = 'pending' THEN 1 ELSE 0 END), 0) AS n_pending,
              COALESCE(SUM(CASE WHEN bi.status = 'failed' THEN 1 ELSE 0 END), 0) AS n_failed,
              COALESCE(SUM(CASE WHEN COALESCE(d.hidden_from_training, 0) != 0 THEN 1 ELSE 0 END), 0)
                AS n_hidden
            FROM batch_items bi
            JOIN download_batches db ON db.id = bi.batch_db_id
            LEFT JOIN datasets d ON d.id = bi.dataset_id
            {where}
        """
        if provider_filter:
            stats_row = conn.execute(
                stats_sql.format(where="WHERE db.provider = ?"), (provider_filter,)
            ).fetchone()
        else:
            stats_row = conn.execute(stats_sql.format(where="")).fetchone()

        # Build registry rows only (stable per-site counts; synthetic rows appended later).
        registry_rows: list[dict[str, Any]] = []
        for r in rows_raw:
            try:
                profile = json.loads(r["profile_json"] or "{}")
            except Exception:
                profile = {}
            status = r["item_status"] or "pending"
            ph = r["profile_hash"]
            if isinstance(ph, (bytes, bytearray)):
                ph = ph.decode("utf-8", errors="replace")
            elif ph is not None:
                ph = str(ph)
            registry_rows.append(
                {
                    "item_id": int(r["item_id"]),
                    "stable_id": r["stable_id"],
                    "provider": r["provider"],
                    "batch_id": r["batch_id"],
                    "asset_type": r["asset_type"],
                    "status": status,
                    "local_path": r["local_path"],
                    "data_source": _data_source_from_row(
                        r["local_path"],
                        r["run_folder"],
                        profile,
                    ),
                    "hidden_from_training": bool(r["hidden_from_training"]),
                    "completed_at": r["completed_at"],
                    "batch_created_at": r["batch_created_at"],
                    "profile_hash": ph,
                    "profile": profile,
                }
            )

        # Collapse duplicate registry rows to one logical file-tracking row per
        # physical asset (path + asset_type). Older runs used slightly different
        # stable_id normalization (hyphen/underscore variants), which could create
        # duplicate registry rows for the same file and inflate Stage-0 counters.
        def _status_rank(st: str) -> int:
            s = (st or "").strip().lower()
            if s == "present":
                return 4
            if s == "pending":
                return 3
            if s == "missing_or_deleted_local":
                return 2
            if s == "failed":
                return 1
            return 0

        def _row_sort_stamp(row: dict[str, Any]) -> str:
            return str(row.get("completed_at") or row.get("batch_created_at") or "")

        def _stable_norm_key(sid: str) -> str:
            s = (sid or "").strip().lower()
            if not s:
                return ""
            return re.sub(r"[-_]+", "_", s)

        def _dedupe_key(row: dict[str, Any]) -> tuple[str, str]:
            at = str(row.get("asset_type") or "").strip().lower()
            lp = str(row.get("local_path") or "").strip()
            if lp:
                # Primary identity: one row per physical file path.
                return (lp.replace("\\", "/").lower(), at or "__unknown__")
            sid = _stable_norm_key(str(row.get("stable_id") or ""))
            if sid:
                return (f"sid:{sid}", at or "__unknown__")
            return (f"item:{row.get('item_id', '')}", at or "__unknown__")

        collapsed_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in registry_rows:
            k = _dedupe_key(row)
            prev = collapsed_by_key.get(k)
            if prev is None:
                collapsed_by_key[k] = row
                continue
            pr = _status_rank(str(prev.get("status") or ""))
            rr = _status_rank(str(row.get("status") or ""))
            if rr > pr or (rr == pr and _row_sort_stamp(row) >= _row_sort_stamp(prev)):
                collapsed_by_key[k] = row

        registry_rows = list(collapsed_by_key.values())
        # Final table rows after dedupe (one logical row per physical asset).
        rows_display = list(registry_rows)
        n_total = len(rows_display)
        n_present_reg = sum(1 for r in rows_display if str(r.get("status") or "") == "present")
        n_missing_reg = sum(1 for r in rows_display if str(r.get("status") or "") == "missing_or_deleted_local")
        n_pending_reg = sum(1 for r in rows_display if str(r.get("status") or "") == "pending")
        n_failed_reg = sum(1 for r in rows_display if str(r.get("status") or "") == "failed")
        n_hidden_reg = sum(1 for r in rows_display if bool(r.get("hidden_from_training")))

        # Reflect actual current data on disk split by training/inference.
        on_disk_pairs_training = sum(1 for _im, _seg, src in on_disk_paired_paths if src == "training")
        on_disk_pairs_inference = sum(1 for _im, _seg, src in on_disk_paired_paths if src == "inference")
        on_disk_pairs = int(on_disk_pairs_training + on_disk_pairs_inference)

        # Per-provider counts and batch cards use registry only (stable when no new datasets).
        provider_present_items: dict[str, int] = {}
        batch_agg: dict[str, dict[str, Any]] = {}
        for r in registry_rows:
            prov = (r.get("provider") or "unknown")
            st = (r.get("status") or "pending")
            if st == "present":
                provider_present_items[prov] = int(provider_present_items.get(prov, 0)) + 1
            bid = r.get("batch_id") or ""
            if bid not in batch_agg:
                ph_b = r.get("profile_hash")
                if isinstance(ph_b, (bytes, bytearray)):
                    ph_b = ph_b.decode("utf-8", errors="replace")
                elif ph_b is not None:
                    ph_b = str(ph_b)
                batch_agg[bid] = {
                    "batch_id": bid,
                    "display_title": _inventory_batch_display_title(bid, prov),
                    "provider": prov,
                    "profile": r.get("profile") or {},
                    "profile_hash": ph_b,
                    "created_at": r.get("batch_created_at"),
                    "n_items": 0,
                    "n_present": 0,
                    "n_missing": 0,
                    "n_training_present": 0,
                    "n_inference_present": 0,
                }
            ba = batch_agg[bid]
            ba["n_items"] += 1
            if st == "present":
                ba["n_present"] += 1
                src = str(r.get("data_source") or "").strip().lower()
                if src == "training":
                    ba["n_training_present"] += 1
                elif src == "inference":
                    ba["n_inference_present"] += 1
            elif st == "missing_or_deleted_local":
                ba["n_missing"] += 1

        # Batch cards: list every ``download_batches`` row (each downloader run), not only
        # batches that still own ``batch_items``. Re-download moves items to a new batch_id,
        # which would otherwise hide older runs from the stacked "Batches" UI.
        db_batches = _list_batches(conn, provider_filter or None)
        batches_sorted: list[dict[str, Any]] = []
        for br in db_batches:
            bid = str(br["batch_id"])
            prov = str(br["provider"] or "unknown")
            ph_b = br["profile_hash"]
            if isinstance(ph_b, (bytes, bytearray)):
                ph_b = ph_b.decode("utf-8", errors="replace")
            elif ph_b is not None:
                ph_b = str(ph_b)
            try:
                profile_b = json.loads(br["profile_json"] or "{}")
            except Exception:
                profile_b = {}
            # Inventory bootstrap/sync batches are internal reconciliation artifacts.
            # Keep them in registry rows, but hide from Stage-0 "Batches" cards so one
            # user-triggered Stage-3 download run appears as one batch card.
            if str(profile_b.get("source") or "").strip().lower() == "studio_inventory_bootstrap":
                continue
            ba = batch_agg.get(bid)
            if ba is not None:
                n_items = int(ba["n_items"])
                n_present = int(ba["n_present"])
                n_missing = int(ba["n_missing"])
                n_training_present = int(ba.get("n_training_present", 0))
                n_inference_present = int(ba.get("n_inference_present", 0))
            else:
                n_items = n_present = n_missing = 0
                n_training_present = n_inference_present = 0
            # Local-HPC batches: catalogue rows are deduped by filesystem path across
            # runs; surviving row may attach to another batch. Use raw ``batch_items``
            # counts for this run so cards match Materialize output.
            if str(profile_b.get("source") or "").strip().lower() == "mitole_local":
                try:
                    bdb = int(br["id"])
                    rc = conn.execute(
                        """
                        SELECT COUNT(*) AS n,
                        COALESCE(SUM(CASE WHEN bi.status = 'present' THEN 1 ELSE 0 END), 0) AS np,
                        COALESCE(SUM(CASE WHEN bi.status = 'missing_or_deleted_local' THEN 1 ELSE 0 END), 0) AS nm
                        FROM batch_items bi
                        WHERE bi.batch_db_id = ?
                        """,
                        (bdb,),
                    ).fetchone()
                    if rc and int(rc["n"] or 0) > 0:
                        n_items = int(rc["n"] or 0)
                        n_present = int(rc["np"] or 0)
                        n_missing = int(rc["nm"] or 0)
                except Exception:
                    pass
            try:
                n_dl_logged = int(br["download_asset_completions"] or 0)
            except (KeyError, TypeError, ValueError):
                n_dl_logged = 0
            if n_dl_logged == 0:
                try:
                    pj = json.loads(br["profile_json"] or "{}")
                    n_dl_logged = int(pj.get("download_asset_completions") or 0)
                except Exception:
                    n_dl_logged = 0
            try:
                n_train_units = int(profile_b.get("training_units_this_run") or 0)
            except Exception:
                n_train_units = 0
            try:
                n_infer_units = int(profile_b.get("inference_units_this_run") or 0)
            except Exception:
                n_infer_units = 0
            if n_train_units <= 0 and n_infer_units <= 0:
                # Fallback for legacy batches without explicit per-split unit counters.
                n_train_units = 2 * max(0, n_training_present)
                n_infer_units = 2 * max(0, n_inference_present)
            # Also suppress empty shell batches in the UI card list.
            if n_items <= 0 and n_dl_logged <= 0:
                continue
            profile_ui = dict(profile_b)
            _dt_tot = profile_ui.get("dataset_totals")
            if isinstance(_dt_tot, dict):
                profile_ui["dataset_totals"] = json.dumps(_dt_tot, ensure_ascii=False, sort_keys=True)
            batches_sorted.append(
                {
                    "batch_id": bid,
                    "display_title": _inventory_batch_display_title(bid, prov),
                    "provider": prov,
                    "profile": profile_ui,
                    "profile_hash": ph_b,
                    "created_at": br["created_at"],
                    "n_items": n_items,
                    "n_present": n_present,
                    "n_missing": n_missing,
                    "download_asset_completions": n_dl_logged,
                    "training_units_this_run": n_train_units,
                    "inference_units_this_run": n_infer_units,
                }
            )

        for ba in batches_sorted:
            ba["_sort_tuple"] = _inventory_batch_sort_tuple(
                ba["batch_id"], ba.get("created_at"), 0.0
            )

        batches_sorted = sorted(
            batches_sorted,
            key=lambda b: b.get("_sort_tuple", (0, 0.0)),
        )
        for ba in batches_sorted:
            ba.pop("_sort_tuple", None)

        distinct_datasets = len(
            {_stable_norm_key(str(r["stable_id"])) for r in registry_rows if r.get("stable_id")}
        )

        delete_rows: list[dict[str, Any]] = []
        # Omit nnUNet ``labelsTr-instance`` / ``labelsTs-instance`` paths from Stage-0 delete
        # history (inventory tracks EM + semantic label pairs only).
        _del_vis = (
            " AND INSTR(LOWER(REPLACE(local_path, CHAR(92), '/')), '/labelstr-instance/') = 0 "
            " AND INSTR(LOWER(REPLACE(local_path, CHAR(92), '/')), '/labelsts-instance/') = 0 "
        )
        if provider_filter:
            drows = conn.execute(
                f"""
                SELECT stable_id, asset_type, local_path, deleted_at, provider
                FROM deletion_events
                WHERE provider = ?{_del_vis}
                ORDER BY deleted_at DESC, id DESC
                LIMIT 100
                """,
                (provider_filter,),
            ).fetchall()
            deletion_events_count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM deletion_events WHERE provider = ?{_del_vis}",
                    (provider_filter,),
                ).fetchone()[0]
            )
        else:
            drows = conn.execute(
                f"""
                SELECT stable_id, asset_type, local_path, deleted_at, provider
                FROM deletion_events
                WHERE 1=1{_del_vis}
                ORDER BY deleted_at DESC, id DESC
                LIMIT 100
                """
            ).fetchall()
            deletion_events_count = int(
                conn.execute(f"SELECT COUNT(*) FROM deletion_events WHERE 1=1{_del_vis}").fetchone()[0]
            )
        delete_rows = [
            {
                "stable_id": str(r["stable_id"]),
                "asset_type": str(r["asset_type"] or ""),
                "local_path": str(r["local_path"] or ""),
                "deleted_at": str(r["deleted_at"] or ""),
                "provider": str(r["provider"] or ""),
            }
            for r in drows
        ]

        # Stage-0 "Log files (downloads)" should mirror the same deduped logical
        # asset count used by the table, while still being a separate summary card.
        download_completions_total = int(len(rows_display))

        summary: dict[str, Any] = {
            "total_items": n_total,
            "inventory_row_count": len(rows_display),
            "distinct_datasets": distinct_datasets,
            "present": n_present_reg,
            "missing_or_deleted": n_missing_reg,
            "deletion_events_count": deletion_events_count,
            "download_completions_total": download_completions_total,
            "pending": n_pending_reg,
            "failed": n_failed_reg,
            "hidden_from_training": n_hidden_reg,
            "on_disk_pairs": on_disk_pairs,
            "on_disk_pairs_training": int(on_disk_pairs_training),
            "on_disk_pairs_inference": int(on_disk_pairs_inference),
            "on_disk_images": len(img_keys),
            "on_disk_labels": len(lbl_keys),
            "delete_history": delete_rows,
            "providers": provider_present_items,
            "batches": batches_sorted,
        }

        response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return {
            "ok": True,
            "registry_exists": True,
            "summary": summary,
            "rows": rows_display,
        }
    except Exception as exc:
        response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return {"ok": False, "registry_exists": True, "error": str(exc),
                "summary": _EMPTY_SUMMARY, "rows": []}
    finally:
        conn.close()


@router.post("/run/download-and-preprocess")
def studio_run_download_and_preprocess(
    body: "StudioDownloaderBody",
) -> dict[str, Any]:
    """Download a batch AND immediately preprocess it (Stage 3 + Stage 4 merged).

    Runs the OpenOrganelle downloader with ``execute=True``, then invokes
    ``3data_downloader/downloader_master/preprocess_agent.py`` on the resulting batch folder.  Batch
    provenance is recorded in the registry.
    """
    # Reuse the existing downloader endpoint with execute=True, then auto-preprocess.
    body.execute = True
    dl_result = studio_run_downloader(body)
    if not dl_result.get("ok"):
        return dl_result

    # Extract the run folder from the downloader output.
    run_folder: str | None = None
    stdout = dl_result.get("stdout", "")
    for line in stdout.splitlines():
        if "[BATCH_ID]" in line:
            run_folder = line.split("[BATCH_ID]")[-1].strip()
            break
    if not run_folder:
        return {
            **dl_result,
            "preprocess": {"ok": False, "message": "Could not determine batch id for preprocessing."},
        }

    # New layout: downloader writes directly to Dataset001_mito2.
    if not str(run_folder).startswith("openorganelle_mito_"):
        return {
            **dl_result,
            "preprocess": {
                "ok": True,
                "skipped": True,
                "message": "Stage 4 already ran during download (direct-to-training layout).",
            },
            "run_folder": run_folder,
        }

    sys.path.insert(0, str(_root()))
    from agent.orchestration.pipeline.preprocess_runner import run_preprocess_on_batch  # noqa: PLC0415
    pre_result = run_preprocess_on_batch(
        project_root=_root(),
        download_run=run_folder,
        split_label_cc=True,
    )

    return {
        **dl_result,
        "preprocess": pre_result,
        "run_folder": run_folder,
    }
