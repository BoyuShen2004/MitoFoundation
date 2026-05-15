"""Stage 3 CLI: cohesive download entrypoint for supported providers.

Default behavior remains backward compatible, while ``--download`` enforces a
single-command path (generate + execute when supported by provider agents).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from .download_options import DownloadJobOptions
    from .scriptgen import write_generated_script
except ImportError:
    # Support direct execution: python downloader_master/agent.py
    _THIS_DIR = Path(__file__).resolve().parent
    _PKG_ROOT = _THIS_DIR.parent
    if str(_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PKG_ROOT))
    from downloader_master.download_options import DownloadJobOptions
    from downloader_master.scriptgen import write_generated_script


def _default_project_root() -> Path:
    from config.paths import project_root

    return project_root()


def _run_guided(argv_tail: list[str]) -> None:
    from .guided_download_agent import main as gmain

    sys.argv = [sys.argv[0]] + argv_tail
    gmain()


def _run_stub(args: argparse.Namespace, root: Path) -> None:
    import json

    vx, vy, vz = (float(x) for x in args.voxel_size_nm.split(","))
    dx, dy, dz = (int(float(x)) for x in args.crop_dims_voxels.split(","))
    centers: list[tuple[float, float, float]] = []
    if args.centers_json:
        raw = json.loads(Path(args.centers_json).read_text(encoding="utf-8"))
        centers = [tuple(float(x) for x in c) for c in raw]
    opts = DownloadJobOptions(
        n_crops=args.n_crops,
        voxel_size_nm=(vx, vy, vz),
        crop_dimensions_voxels=(dx, dy, dz),
        centers=centers,
        data_scope="labeled",
        crop_policy=(args.crop_policy or "auto").strip().lower(),
        output_dir=str(root / "data" / "raw_volumes"),
    )
    path = write_generated_script(root, args.site, opts)
    print({"written": str(path)})
    if args.execute:
        subprocess.check_call([sys.executable, str(path)], cwd=str(root))


def _forward_openorganelle(forward_argv: list[str]) -> None:
    dl_root = Path(__file__).resolve().parent.parent
    if str(dl_root) not in sys.path:
        sys.path.insert(0, str(dl_root))
    from openorganelle.agent import main as og_main

    sys.argv = [sys.argv[0]] + forward_argv
    og_main()


def _forward_bossdb(forward_argv: list[str]) -> None:
    dl_root = Path(__file__).resolve().parent.parent
    if str(dl_root) not in sys.path:
        sys.path.insert(0, str(dl_root))
    from bossdb.agent import main as bd_main

    sys.argv = [sys.argv[0]] + forward_argv
    bd_main()


def _ensure_execute_flag(argv: list[str]) -> list[str]:
    """Append --execute when the caller requested one-shot download mode."""
    for tok in argv:
        if tok == "--execute" or tok.startswith("--execute="):
            return argv
    return [*argv, "--execute"]


def main() -> None:
    if "--guided" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--guided"]
        _run_guided(rest)
        return

    pre = argparse.ArgumentParser(
        description=(
            "Stage 3 downloader. Default: --site openorganelle or bossdb runs the real generator; "
            "use --volume-stub for the Pipeline Studio stub. "
            "OpenOrganelle-specific flags: run `python agent.py --site OpenOrganelle -- --help`."
        ),
    )
    pre.add_argument("--project-root", type=Path, default=None)
    pre.add_argument("--site", default="OpenOrganelle")
    pre.add_argument(
        "--download",
        action="store_true",
        help="Run one cohesive provider download flow (generate + execute).",
    )
    pre.add_argument(
        "--volume-stub",
        action="store_true",
        help="Emit Pipeline Studio volume stub (outputs/download_<site>.py).",
    )
    pre_args, rest = pre.parse_known_args()
    root = Path(pre_args.project_root or _default_project_root()).resolve()
    site_l = pre_args.site.strip().lower()
    # OpenOrganelle/BossDB should use real site downloaders.
    # Keep stub mode for non-supported sites only.
    use_stub = site_l not in ("openorganelle", "bossdb")

    if use_stub:
        stub_parser = argparse.ArgumentParser(description="mitoFoundation2 — volume download stub")
        stub_parser.add_argument("--project-root", type=Path, default=root)
        stub_parser.add_argument("--site", default=pre_args.site)
        stub_parser.add_argument("--n-crops", type=int, default=1)
        stub_parser.add_argument("--voxel-size-nm", default="16,16,16")
        stub_parser.add_argument("--crop-dims-voxels", default="512,512,512")
        stub_parser.add_argument("--centers-json", default="")
        stub_parser.add_argument("--data-scope", choices=("labeled",), default="labeled")
        stub_parser.add_argument(
            "--crop-policy",
            choices=("auto", "center"),
            default="auto",
            help="When n_crops=1, Studio sets center automatically.",
        )
        stub_parser.add_argument("--execute", action="store_true")
        stub_args = stub_parser.parse_args(rest)
        if pre_args.download:
            stub_args.execute = True
        _run_stub(stub_args, Path(stub_args.project_root).resolve())
        return

    forward: list[str] = []
    skip = False
    for tok in rest:
        if tok == "--":
            continue
        if skip:
            skip = False
            continue
        if tok == "--site":
            skip = True
            continue
        if tok.startswith("--site="):
            continue
        forward.append(tok)

    if pre_args.download:
        forward = _ensure_execute_flag(forward)

    if site_l == "openorganelle":
        _forward_openorganelle(forward)
        return
    _forward_bossdb(forward)


if __name__ == "__main__":
    main()
