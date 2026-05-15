"""Streamlit UI: pick a subfolder under ``data/raw`` and inspect volume metadata (shape, spacing, dtype)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import streamlit as st

from downloader_master.preprocess_agent import (
    _default_project_root,
    list_immediate_subfolders_of_raw,
    list_volume_files_under,
    peek_volume_metadata,
    raw_data_root,
)


def _project_root() -> Path:
    env = os.environ.get("MITO2_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return _default_project_root()


def main() -> None:
    st.set_page_config(page_title="Raw data inspector", layout="wide")
    st.title("Raw volume inspector")
    st.caption("Browse data/raw subfolders and view shape, dtype, and physical spacing (Z, Y, X as used by the preprocessor).")

    root = _project_root()
    raw = raw_data_root(root)

    st.sidebar.markdown("**Project root**")
    st.sidebar.code(str(root), language="text")

    if not raw.is_dir():
        st.error(f"Raw data directory does not exist: `{raw}`")
        st.info("Create `data/raw` or set `MITO2_PROJECT_ROOT` to your mitoFoundation2 checkout.")
        return

    subdirs = list_immediate_subfolders_of_raw(root)
    options: list[tuple[str, Path]] = [("— entire data/raw (recursive) —", raw)]
    for d in subdirs:
        options.append((d.name, d))

    label_to_folder = dict(options)
    labels = [o[0] for o in options]
    choice = st.selectbox("Subfolder under data/raw", labels, index=0)
    folder = label_to_folder[choice]

    files = list_volume_files_under(folder)
    try:
        rel = folder.relative_to(root)
    except ValueError:
        rel = folder
    st.subheader(f"Volumes under `{rel}`")
    st.write(f"**{len(files)}** file(s) matched (.h5, .nrrd, .nii, .nii.gz).")

    if not files:
        return

    max_rows = st.slider("Max files to scan (for speed)", min_value=5, max_value=500, value=min(100, len(files)), step=5)
    subset = files[:max_rows]

    rows: list[dict[str, str]] = []
    prog = st.progress(0, text="Reading headers…")
    for i, p in enumerate(subset):
        meta = peek_volume_metadata(p)
        rows.append(
            {
                "file": meta["file"],
                "format": meta["format"],
                "shape": meta["shape"],
                "dtype": meta["dtype"],
                "spacing Z,Y,X": meta["spacing_zyx"],
                "error": meta["error"],
                "path": meta["path"],
            }
        )
        prog.progress((i + 1) / len(subset), text=f"Reading headers… ({i + 1}/{len(subset)})")
    prog.empty()

    st.dataframe(rows, use_container_width=True, hide_index=True)

    if len(files) > max_rows:
        st.warning(f"Showing metadata for the first **{max_rows}** files only (sorted paths). Increase the slider to scan more.")


if __name__ == "__main__":
    main()
