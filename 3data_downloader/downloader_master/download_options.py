"""Structured options for volumetric crop / chunk downloads (placeholders for real APIs)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List


@dataclass
class DownloadJobOptions:
    """User-tunable knobs echoed into generated downloader scripts."""

    n_crops: int = 1
    voxel_size_nm: tuple[float, float, float] = (16.0, 16.0, 16.0)
    crop_dimensions_voxels: tuple[int, int, int] = (512, 512, 512)
    centers: List[tuple[float, float, float]] = field(default_factory=list)
    label_centers: List[tuple[float, float, float]] = field(default_factory=list)
    data_scope: str = "both"
    crop_policy: str = "auto"
    output_dir: str = "data/raw_volumes"
    datasets: List[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["voxel_size_nm"] = list(self.voxel_size_nm)
        # Backward compatibility for any downstream consumer expecting the old key.
        d["voxel_nm"] = list(self.voxel_size_nm)
        d["crop_dimensions_voxels"] = [int(x) for x in self.crop_dimensions_voxels]
        d["centers"] = [list(c) for c in self.centers]
        d["label_centers"] = [list(c) for c in self.label_centers]
        d["datasets"] = list(self.datasets)
        return d


def default_centers_if_empty(opts: DownloadJobOptions) -> DownloadJobOptions:
    if not opts.centers:
        opts.centers = [(0.0, 0.0, 0.0)]
    return opts
