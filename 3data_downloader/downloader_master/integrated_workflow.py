from __future__ import annotations

from pathlib import Path


def _expected_training_pair(repo_root: Path, stem: str) -> tuple[Path, Path]:
    """Expected nnUNet training pair after preprocess (same tree as Stage-3 download)."""
    from config.paths import nnunet_dataset_root

    nn_root = nnunet_dataset_root(repo_root)
    nn_img = nn_root / "imagesTr" / f"{stem}_0000.nii.gz"
    nn_seg = nn_root / "labelsTr" / f"{stem}.nii.gz"
    return nn_img, nn_seg


def _assert_split_preprocess_marker(seg_path: Path) -> None:
    if seg_path.name.endswith(".nii.gz"):
        # NIfTI outputs do not store the H5 preprocess marker attribute.
        return
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"h5py is required to verify preprocess marker for {seg_path}") from exc
    with h5py.File(seg_path, "r") as f:
        marker = f.attrs.get("mito2_split_label_cc", 0)
        try:
            marker_int = int(marker)
        except Exception:
            marker_int = 0
        if marker_int != 1:
            raise RuntimeError(
                f"Integrated preprocess verification failed for {seg_path}: "
                "missing mito2_split_label_cc=1 marker."
            )


def run_integrated_preprocess_for_downloaded_dataset(
    *,
    repo_root: Path,
    dataset_name: str,
    downloaded_images_dir: Path,
    split_label_cc: bool = True,
) -> list[Path]:
    """Run stage-4 preprocess for one downloaded dataset and verify outputs.

    Returns the list of downloaded raw image files used as stage-4 inputs.
    Raises RuntimeError on any failure (no raw images, preprocess non-zero,
    missing training outputs).
    """
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    inv_root = repo_root / "0inventory"
    if str(inv_root) not in sys.path:
        sys.path.insert(0, str(inv_root))
    from download_history import run_stage4_for_images  # noqa: PLC0415

    tag = dataset_name.replace("-", "_")
    downloaded = sorted(
        list(downloaded_images_dir.glob(f"{tag}*_0000.nii.gz"))
        + list(downloaded_images_dir.glob(f"{tag}*_im.h5")),
        key=lambda p: p.name.lower(),
    )
    if not downloaded:
        raise RuntimeError(f"No downloaded image files found for preprocessing: {dataset_name}")

    rc = run_stage4_for_images(repo_root, downloaded, split_label_cc=split_label_cc)
    if rc != 0:
        raise RuntimeError(f"Integrated preprocess failed for {dataset_name} (exit {rc})")

    for raw_im in downloaded:
        name = raw_im.name
        if name.lower().endswith("_0000.nii.gz"):
            stem = name[: -len("_0000.nii.gz")]
        elif name.lower().endswith("_im.h5"):
            stem = name[:-6]
        else:
            stem = raw_im.stem
        expect_img, expect_seg = _expected_training_pair(repo_root, stem)
        if (not expect_img.is_file()) or (not expect_seg.is_file()):
            raise RuntimeError(
                f"Integrated preprocess missing outputs for {dataset_name}: "
                f"{expect_img} and {expect_seg}"
            )
        if split_label_cc:
            _assert_split_preprocess_marker(expect_seg)
    return downloaded

