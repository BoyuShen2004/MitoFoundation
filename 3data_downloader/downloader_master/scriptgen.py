"""Emit a standalone Python downloader stub with argparse for crop/voxel/center options."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from .download_options import DownloadJobOptions, default_centers_if_empty


def render_script(project_root: Path, site: str, options: DownloadJobOptions) -> str:
    opts = default_centers_if_empty(options)
    payload = json.dumps(opts.to_dict(), indent=2)
    root_s = str(project_root.resolve())
    return dedent(
        f'''\
        #!/usr/bin/env python3
        """Auto-generated download stub for mitoFoundation2 (site={site})."""
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path


        DEFAULTS = json.loads({json.dumps(payload)})
        WEBSITE_NAME = {json.dumps(site)}
        DATA_TYPE = DEFAULTS.get("data_scope", "both")
        DATASETS_TO_DOWNLOAD = DEFAULTS.get("datasets", [])


        def main() -> None:
            print(f"[INFO] Website: {{WEBSITE_NAME}}")
            print(f"[INFO] Data type: {{DATA_TYPE}}")
            if DATASETS_TO_DOWNLOAD:
                print(f"[INFO] Dataset entries in script: {{len(DATASETS_TO_DOWNLOAD)}}")
                for row in DATASETS_TO_DOWNLOAD:
                    ds = (row.get("dataset_name") or "").strip()
                    if not ds:
                        continue
                    img = (row.get("img_path") or "").strip()
                    seg = (row.get("seg_path") or "").strip()
                    if seg:
                        print(f"  - {{ds}} | img={{img}} | label={{seg}}")
                    else:
                        print(f"  - {{ds}} | img={{img}}")
            else:
                print("[INFO] Dataset entries are not embedded in this generic stub.")

            p = argparse.ArgumentParser(description="Generated volume download stub")
            p.add_argument("--n-crops", type=int, default=DEFAULTS["n_crops"])
            p.add_argument(
                "--voxel-size-nm",
                type=str,
                default=",".join(str(x) for x in DEFAULTS["voxel_size_nm"]),
                help="Physical voxel size in nm as x,y,z",
            )
            p.add_argument(
                "--crop-dims-voxels",
                type=str,
                default=",".join(str(int(x)) for x in DEFAULTS["crop_dimensions_voxels"]),
                help="Crop dimensions (voxel counts) as x,y,z",
            )
            p.add_argument("--centers-json", type=str, default="", help="Path to JSON list of [x,y,z]")
            p.add_argument("--label-centers-json", type=str, default="", help="Path to JSON list of label centers")
            p.add_argument("--output-dir", type=str, default=DEFAULTS["output_dir"])
            args = p.parse_args()

            vx, vy, vz = (float(x) for x in args.voxel_size_nm.split(","))
            dx, dy, dz = (int(float(x)) for x in args.crop_dims_voxels.split(","))
            centers = DEFAULTS["centers"]
            if args.centers_json:
                centers = json.loads(Path(args.centers_json).read_text(encoding="utf-8"))
            label_centers = DEFAULTS["label_centers"]
            if args.label_centers_json:
                label_centers = json.loads(Path(args.label_centers_json).read_text(encoding="utf-8"))

            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            manifest = {{
                "site": {json.dumps(site)},
                "project_root": {json.dumps(root_s)},
                "n_crops": args.n_crops,
                "voxel_size_nm": [vx, vy, vz],
                "crop_dimensions_voxels": [dx, dy, dz],
                "centers": centers,
                "label_centers": label_centers,
                "data_scope": DEFAULTS.get("data_scope", "both"),
                "crop_policy": ("center" if args.n_crops <= 1 else DEFAULTS.get("crop_policy", "auto")),
                "status": "stub_run_ok",
            }}
            (out / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(manifest, indent=2))


        if __name__ == "__main__":
            main()
        '''
    )


def write_generated_script(project_root: Path, site: str, options: DownloadJobOptions) -> Path:
    gen_dir = project_root / "3data_downloader" / "outputs"
    gen_dir.mkdir(parents=True, exist_ok=True)
    scope = (options.data_scope or "both").strip().lower()
    safe_scope = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in scope)
    path = gen_dir / f"download_{site}_{safe_scope}.py"
    path.write_text(render_script(project_root, site, options), encoding="utf-8")
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        pass
    return path
