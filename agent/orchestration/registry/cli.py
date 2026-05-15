"""CLI entry point: python -m agent.orchestration.registry.cli <command> [options]

Commands:
  status         Print registry summary (providers, datasets, assets, downloads)
  sync           Run discovery + asset resolution for a provider
  plan-downloads Show which assets still need downloading
  plan-preprocess Show which raw files need preprocessing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.orchestration.registry.schema import open_registry, DEFAULT_REGISTRY_PATH
from agent.orchestration.registry.providers import get_provider, known_providers
from agent.orchestration.registry.planners import plan_discovery, plan_downloads, plan_preprocess


def _get_provider(name: str):
    # "oo" is a common shorthand alias for OpenOrganelle.
    canonical = "openorganelle" if name.lower() == "oo" else name
    try:
        return get_provider(canonical)
    except KeyError:
        raise SystemExit(
            f"Unknown provider: {name!r}. "
            f"Available: {', '.join(known_providers())}"
        )


def cmd_status(args: argparse.Namespace) -> None:
    conn = open_registry(args.db)
    providers = conn.execute("SELECT * FROM providers").fetchall()
    print(f"Registry: {args.db}")
    print(f"Providers: {len(providers)}")
    for p in providers:
        n_ds = conn.execute(
            "SELECT COUNT(*) FROM datasets WHERE provider_id = ?", (p["id"],)
        ).fetchone()[0]
        n_assets = conn.execute(
            """SELECT COUNT(*) FROM assets a
               JOIN datasets d ON d.id = a.dataset_id
               WHERE d.provider_id = ?""",
            (p["id"],),
        ).fetchone()[0]
        n_dl = conn.execute(
            """SELECT COUNT(*) FROM downloads dl
               JOIN assets a ON a.id = dl.asset_id
               JOIN datasets d ON d.id = a.dataset_id
               WHERE d.provider_id = ? AND dl.status = 'complete'""",
            (p["id"],),
        ).fetchone()[0]
        print(f"  [{p['name']}] datasets={n_ds} assets={n_assets} complete_downloads={n_dl}")
    n_pp = conn.execute(
        "SELECT COUNT(*) FROM preprocess_runs WHERE status = 'complete'"
    ).fetchone()[0]
    print(f"Preprocess runs (complete): {n_pp}")
    conn.close()


def cmd_sync(args: argparse.Namespace) -> None:
    provider = _get_provider(args.provider)
    conn = open_registry(args.db)
    plan = plan_discovery(conn, provider)
    print(f"Discovery sync complete: {len(plan.new_stable_ids)} new, {plan.total_in_registry} total")
    specs = provider.resolve_assets(conn)
    conn.commit()
    conn.close()
    print(f"Asset resolution: {len(specs)} asset specs written to registry")


def cmd_plan_downloads(args: argparse.Namespace) -> None:
    from agent.orchestration.registry.api import make_download_profile_hash

    provider = _get_provider(args.provider)
    conn = open_registry(args.db)

    profile_hash: str | None = None
    if args.n_crops is not None:
        chunk = tuple(int(v) for v in args.chunk_zyx.split(","))
        voxel = tuple(float(v) for v in args.voxel_nm.split(","))
        profile_hash = make_download_profile_hash(
            n_crops=args.n_crops,
            chunk_zyx=chunk,
            voxel_nm_zyx=voxel,
            mode=args.mode,
            foundation=not args.no_foundation,
        )
        print(f"Profile: n_crops={args.n_crops} chunk={chunk} voxel_nm={voxel} mode={args.mode} → hash={profile_hash}")

    allowlist: frozenset[str] | None = None
    if args.mode == "labeled" and callable(getattr(provider, "labeled_inventory_stable_ids", None)):
        allowlist = provider.labeled_inventory_stable_ids()

    plans = plan_downloads(
        conn,
        provider,
        download_profile_hash=profile_hash,
        stable_id_allowlist=allowlist,
    )
    conn.close()
    if not plans:
        print("Nothing to download — all assets are already satisfied for this profile.")
        return
    print(f"{len(plans)} asset(s) need downloading:")
    for p in plans[:50]:
        print(f"  [{p.asset_type}] {p.stable_dataset_id} ({p.reason})")
        print(f"    url: {p.remote_url}")
    if len(plans) > 50:
        print(f"  ... and {len(plans)-50} more")


def cmd_plan_preprocess(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        raise SystemExit(f"Raw directory not found: {raw_dir}")
    raw_files = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix in (".h5", ".nrrd", ".nii") or p.name.endswith(".nii.gz")
    )
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    conn = open_registry(args.db)
    plans = plan_preprocess(conn, raw_files, config)
    conn.close()
    if not plans:
        print("Nothing to preprocess — all inputs already satisfied.")
        return
    print(f"{len(plans)} file(s) need preprocessing:")
    for p in plans[:30]:
        print(f"  {p.input_path.name} ({p.reason})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agent.orchestration.registry.cli", description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_REGISTRY_PATH,
        help="Path to registry SQLite (default: data/registry.sqlite)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Print registry summary")

    sp = sub.add_parser("sync", help="Sync discovery + assets for a provider")
    sp.add_argument("--provider", default="openorganelle")

    sp2 = sub.add_parser("plan-downloads", help="Show assets needing download")
    sp2.add_argument("--provider", default="openorganelle")
    sp2.add_argument("--n-crops", type=int, default=None,
                     help="Filter by n_crops profile (omit for profile-agnostic check)")
    sp2.add_argument("--chunk-zyx", default="512,512,512",
                     help="Chunk shape z,y,x in voxels (default: 512,512,512)")
    sp2.add_argument("--voxel-nm", default="16.0,16.0,16.0",
                     help="Voxel size z,y,x in nm (default: 16,16,16)")
    sp2.add_argument("--mode", default="labeled", choices=["labeled", "unlabeled"])
    sp2.add_argument("--no-foundation", action="store_true")

    sp3 = sub.add_parser("plan-preprocess", help="Show raw files needing preprocess")
    sp3.add_argument("--raw-dir", default="data/raw")
    sp3.add_argument("--config", default=None, help="JSON config file for fingerprinting")

    args = parser.parse_args(argv)
    {
        "status": cmd_status,
        "sync": cmd_sync,
        "plan-downloads": cmd_plan_downloads,
        "plan-preprocess": cmd_plan_preprocess,
    }[args.command](args)


if __name__ == "__main__":
    main()
