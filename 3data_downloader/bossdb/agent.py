"""BossDB stage-3 agent with cohesive one-shot download mode.

Subcommands
-----------
  discover   Scrape BossDB metadata API and print summary (no inventory file)
  generate   Live discovery → write outputs/download_bossdb_labeled.{py,md}
  run        Discover + generate in one shot (default when no subcommand given)
  list       Print datasets from live metadata
  execute    Discover + generate + execute in one command
  execute-only  Run an existing generated download_bossdb_labeled.py

The ``main()`` function is called by downloader_master/agent.py when
``--site bossdb`` is passed; it receives whatever flags follow ``--site bossdb``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Ensure parent package (3data_downloader) is importable.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# Ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_INV_ROOT = _REPO_ROOT / "0inventory"
if str(_INV_ROOT) not in sys.path:
    sys.path.insert(0, str(_INV_ROOT))

from bossdb.metadata_client import discover as _discover_api, BossDBDataset  # noqa: E402
from bossdb_inventory import pair_labeled_datasets  # noqa: E402
from bossdb.scriptgen import write_bossdb_outputs  # noqa: E402
from download_history import nnunet_dataset_root  # noqa: E402

_OUTPUTS_DIR   = _REPO_ROOT / "3data_downloader" / "outputs"
_TRAINING_ROOT = nnunet_dataset_root(_REPO_ROOT)
_DEFAULT_CHUNK       = (128, 128, 128)
_DEFAULT_N_CROPS     = 1
_DEFAULT_VOXEL_NM    = (16.0, 16.0, 16.0)


# ── helpers ────────────────────────────────────────────────────────────────────

def _resolve_outputs_dir(project_root: Path | None) -> Path:
    root = project_root or _REPO_ROOT
    d = root / "3data_downloader" / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_training_root(project_root: Path | None) -> Path:
    return nnunet_dataset_root(project_root or _REPO_ROOT)


# ── subcommand handlers ────────────────────────────────────────────────────────

def cmd_discover(args: argparse.Namespace) -> list[BossDBDataset]:
    """Scrape the BossDB metadata API (live only)."""
    query: dict = {}
    # channels/query does not accept free-form tags; keep flag for compatibility, but ignore.
    if args.filter_tags:
        print("[WARN] --filter-tags is ignored for BossDB channels/query (schema-key filters only).")

    print(f"[PLAN] BossDB discovery from metadata API ({_discover_api.__module__})")
    datasets = _discover_api(
        query=query or None,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(f"[INFO] Discovered {len(datasets)} channel record(s).")
    print("[DONE] Discovery complete (no inventory file written).")
    return datasets


def cmd_generate(args: argparse.Namespace, datasets: list[BossDBDataset] | None = None) -> tuple[Path, Path]:
    """Live-discover and generate download_bossdb_labeled.{py,md}."""
    if datasets is None:
        datasets = cmd_discover(args)
    print(f"[INFO] Using {len(datasets)} channel record(s) from live discovery.")

    pairs = pair_labeled_datasets(datasets)
    print(f"[INFO] Paired {len(pairs)} labeled experiment(s) (image+annotation).")
    if not pairs:
        print(
            "[WARN] No paired (image+annotation) experiments found.\n"
            "       Check that your inventory contains channels with type='image' and type='annotation'"
            " in the same experiment."
        )

    chunk_shape = tuple(int(x) for x in args.chunk_shape.split(","))
    n_crops     = int(args.n_crops)
    voxel_nm    = tuple(float(x) for x in args.voxel_size_nm.split(","))
    outputs_dir = _resolve_outputs_dir(
        Path(args.project_root) if args.project_root else None
    )
    training_root = str(_resolve_training_root(
        Path(args.project_root) if args.project_root else None
    ))

    py_path, md_path = write_bossdb_outputs(
        outputs_dir,
        pairs,
        chunk_shape=chunk_shape,
        n_crops=n_crops,
        voxel_size_nm=voxel_nm,
        training_root=training_root,
    )
    print(f"[DONE] Generated downloader script: {py_path}")
    print(f"[DONE] Generated metadata doc: {md_path}")
    return py_path, md_path


def cmd_list(args: argparse.Namespace) -> None:
    """Print datasets from live discovery."""
    datasets = cmd_discover(args)
    if not datasets:
        print("[NOOP] No datasets discovered.")
        return
    print(f"[INFO] BossDB live discovery: {len(datasets)} channel record(s)")

    if args.pairs:
        pairs = pair_labeled_datasets(datasets)
        print(f"\nPaired labeled experiments ({len(pairs)}):")
        for p in pairs:
            print(
                f"  {p['project_id']}"
                f"  img={p['img_channel']}"
                f"  seg={p['seg_channel']}"
                f"  voxel={p.get('voxel_size_nm')}"
                f"  organism={p.get('organism')}"
            )
    else:
        print(f"\nAll channels ({len(datasets)}):")
        for ds in datasets[:50]:
            print(
                f"  {ds.uri}"
                f"  type={ds.channel_type}"
                f"  dtype={ds.data_type}"
                f"  organism={ds.organism}"
            )
        if len(datasets) > 50:
            print(f"  … and {len(datasets) - 50} more (use --pairs to see paired summary)")


def cmd_execute(args: argparse.Namespace) -> None:
    """Run an existing generated download script."""
    py_path = _resolve_outputs_dir(
        Path(args.project_root) if args.project_root else None
    ) / "download_bossdb_labeled.py"
    if not py_path.is_file():
        raise SystemExit(
            f"[BOSSDB] Download script not found: {py_path}\n"
            "Run `--generate` first."
        )
    extra: list[str] = []
    if args.dataset:
        extra += ["--dataset", args.dataset]
    if args.dry_run:
        extra += ["--dry-run"]
    cmd = [sys.executable, str(py_path)] + extra
    print(f"[PROGRESS] Executing generated downloader: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def cmd_run(args: argparse.Namespace) -> None:
    """Discover + generate (+ optional execute) in one cohesive flow."""
    print("[PLAN] BossDB full pipeline: discover (live) -> generate script")
    datasets = cmd_discover(args)
    py_path, _ = cmd_generate(args, datasets=datasets)
    if args.execute:
        print(f"[PROGRESS] Executing generated script: {py_path}")
        extra: list[str] = []
        if args.dataset:
            extra += ["--dataset", args.dataset]
        if args.dry_run:
            extra += ["--dry-run"]
        subprocess.check_call([sys.executable, str(py_path), *extra])
    else:
        print("[DONE] Generation-only flow complete. Run the generated script with:")
        print(f"  python {py_path} --list")


# ── argument parser ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "BossDB stage-3 downloader. "
            "Discover metadata (live), generate download script, and optionally execute."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode flags (mutually select the subcommand; default is 'run' = discover+generate)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--discover",
        action="store_true",
        help="Scrape BossDB metadata API and print a summary; stop after.",
    )
    mode.add_argument(
        "--generate",
        action="store_true",
        help="Live-discover channels and (re)generate download script; stop after.",
    )
    mode.add_argument(
        "--list",
        action="store_true",
        help="Print datasets from live metadata and exit.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run cohesive flow: discover + generate + execute.",
    )
    mode.add_argument(
        "--execute-only",
        action="store_true",
        help="Run only an existing generated download_bossdb_labeled.py script.",
    )

    # Discovery options
    p.add_argument("--filter-tags", default="", help="Comma-separated tags to filter (e.g. 'em,mouse')")
    p.add_argument("--timeout",   type=int, default=30, help="HTTP timeout per request (s)")
    p.add_argument("--retries",   type=int, default=3,  help="HTTP retries per request")

    # Generation / download options
    p.add_argument("--project-root",    default="", help="Override project root path")
    p.add_argument("--chunk-shape",     default=",".join(str(x) for x in _DEFAULT_CHUNK),
                   help="Output voxel count z,y,x  (default: %(default)s)")
    # Compatibility alias used by Studio subprocess calls.
    p.add_argument("--chunk",           default="",
                   help=argparse.SUPPRESS)
    p.add_argument("--n-crops",         type=int, default=_DEFAULT_N_CROPS)
    p.add_argument("--voxel-size-nm",   default=",".join(str(x) for x in _DEFAULT_VOXEL_NM),
                   help="Target voxel spacing nm z,y,x  (default: %(default)s)")
    # Compatibility alias used by Studio subprocess calls.
    p.add_argument("--mode",            default="labeled",
                   help=argparse.SUPPRESS)
    p.add_argument("--no-foundation",   action="store_true",
                   help=argparse.SUPPRESS)

    # Execution options
    p.add_argument("--dataset",  default="", help="Single project_id to download (collection/experiment)")
    p.add_argument("--dry-run",  action="store_true", help="Print download plan but skip actual I/O")

    # List options
    p.add_argument("--pairs", action="store_true", help="With --list: show paired experiments only")

    return p


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.chunk and not args.chunk_shape:
        args.chunk_shape = args.chunk
    if args.chunk:
        args.chunk_shape = args.chunk

    if args.mode and str(args.mode).strip().lower() not in ("labeled", ""):
        raise SystemExit(f"[BOSSDB] Unsupported mode={args.mode!r}. BossDB currently supports only labeled.")

    if args.list:
        cmd_list(args)
        return

    if args.discover:
        cmd_discover(args)
        return

    if args.generate:
        cmd_generate(args)
        return

    if args.execute_only:
        cmd_execute(args)
        return

    # Default: discover + generate. --execute upgrades this to one-shot download.
    cmd_run(args)


if __name__ == "__main__":
    main()
