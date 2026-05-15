"""High-level build flow: markdown -> sqlite + schema markdown."""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from pathlib import Path

from .db import build_sqlite_db
from .parser import DatasetRecord, LayerRecord, parse_markdown_file
from .report import render_schema_markdown


def clear_output_artifacts(db_dir: Path, schema_md_dir: Path) -> tuple[int, int]:
    """Delete prior batch outputs under *db_dir* and *schema_md_dir*.

    Removes ``*.db`` and ``*.schema.md`` only so other files in those folders are kept.
    Call this at the start of a full batch run so removed/renamed inputs do not leave
    stale artifacts.
    """
    removed_db = 0
    removed_md = 0
    for p in sorted(db_dir.glob("*.db")):
        if p.is_file():
            p.unlink()
            removed_db += 1
    for p in sorted(schema_md_dir.glob("*.schema.md")):
        if p.is_file():
            p.unlink()
            removed_md += 1
    return removed_db, removed_md


def list_markdown_inputs(md_dir: Path) -> list[Path]:
    """Return sorted ``*.md`` files in *md_dir* (files only)."""
    return sorted([p for p in md_dir.glob("*.md") if p.is_file()])


def build_batch(
    md_dir: Path,
    db_dir: Path,
    schema_md_dir: Path,
    *,
    clean: bool = True,
) -> tuple[list[tuple[Path, dict[str, int]]], tuple[int, int] | None]:
    """Build one SQLite DB + schema markdown per ``*.md`` in *md_dir*.

    By default (*clean* is True), clears all ``*.db`` in *db_dir* and all
    ``*.schema.md`` in *schema_md_dir* before building so each run replaces the
    previous batch outputs.

    Returns ``(results, removed_counts)`` where *removed_counts* is
    ``(n_db, n_schema_md)`` if *clean* was true, else ``None``.
    """
    md_dir = md_dir.expanduser().resolve()
    db_dir = db_dir.expanduser().resolve()
    schema_md_dir = schema_md_dir.expanduser().resolve()

    if not md_dir.is_dir():
        raise RuntimeError(f"Input markdown folder not found: {md_dir}")

    db_dir.mkdir(parents=True, exist_ok=True)
    schema_md_dir.mkdir(parents=True, exist_ok=True)

    removed: tuple[int, int] | None = None
    if clean:
        removed = clear_output_artifacts(db_dir, schema_md_dir)

    md_files = list_markdown_inputs(md_dir)
    if not md_files:
        raise RuntimeError(f"No .md files found in: {md_dir}")

    results: list[tuple[Path, dict[str, int]]] = []
    for md_path in md_files:
        db_path = db_dir / f"{md_path.stem}.db"
        schema_md_path = schema_md_dir / f"{md_path.stem}.schema.md"
        stats = build_one_markdown(md_path, db_path, schema_md_path)
        results.append((md_path, stats))
    return results, removed


def build_one_markdown(
    markdown_path: Path,
    sqlite_out: Path,
    schema_md_out: Path,
) -> dict[str, int]:
    records = parse_markdown_file(markdown_path)
    records = _augment_records_with_probe_json(markdown_path, records)
    stats = build_sqlite_db(records, sqlite_out)
    schema_md = render_schema_markdown(markdown_path, sqlite_out, stats)
    schema_md_out.write_text(schema_md, encoding="utf-8")
    return stats


def merge_dataset_records(records: list[DatasetRecord]) -> list[DatasetRecord]:
    """Merge rows for the same ``dataset_name`` (union layers; fill missing scalars)."""
    from collections import defaultdict

    groups: dict[str, list[DatasetRecord]] = defaultdict(list)
    for r in records:
        groups[r.dataset_name].append(r)
    out = [_merge_record_group(g) for g in groups.values()]
    return sorted(out, key=lambda x: str(x.dataset_name).lower())


def _merge_record_group(group: list[DatasetRecord]) -> DatasetRecord:
    if len(group) == 1:
        return group[0]

    def richness(r: DatasetRecord) -> tuple[int, int]:
        n_scalar = sum(
            1
            for f in dataclasses.fields(r)
            if f.name != "layers" and getattr(r, f.name)
        )
        return (len(r.layers), n_scalar)

    best = max(group, key=richness)
    base = copy.deepcopy(best)
    seen = {
        (l.layer_role, l.layer_kind, l.layer_name, l.layer_url or "", l.raw_token)
        for l in base.layers
    }
    for r in group:
        if r is best:
            continue
        for layer in r.layers:
            key = (
                layer.layer_role,
                layer.layer_kind,
                layer.layer_name,
                layer.layer_url or "",
                layer.raw_token,
            )
            if key not in seen:
                base.layers.append(copy.deepcopy(layer))
                seen.add(key)
        for f in dataclasses.fields(DatasetRecord):
            if f.name == "layers":
                continue
            a = getattr(base, f.name)
            b = getattr(r, f.name)
            if f.name == "mitochondria_in_layer_names":
                if b and not a:
                    setattr(base, f.name, True)
                continue
            if not a and b:
                setattr(base, f.name, b)
    return base


def build_merged_site(
    output_stem: str,
    markdown_paths: list[Path],
    sqlite_out: Path,
    schema_md_out: Path,
) -> dict[str, int]:
    """
    Parse multiple markdown reports for one site (e.g. ``OpenOrganelle_01.md`` + ``OpenOrganelle_02.md``),
    merge dataset rows, write a single DB + schema doc named ``output_stem``.
    """
    if len(markdown_paths) < 2:
        raise ValueError("build_merged_site expects at least 2 markdown paths")
    merged: list[DatasetRecord] = []
    for p in sorted(markdown_paths):
        merged.extend(parse_markdown_file(p))
    merged = merge_dataset_records(merged)
    probe_hint = markdown_paths[0].parent / f"{output_stem}.md"
    merged = _augment_records_with_probe_json(probe_hint, merged)
    stats = build_sqlite_db(merged, sqlite_out)
    schema_md = render_schema_markdown(
        probe_hint,
        sqlite_out,
        stats,
        merged_sources=sorted(markdown_paths),
    )
    schema_md_out.write_text(schema_md, encoding="utf-8")
    return stats


def _augment_records_with_probe_json(
    markdown_path: Path,
    records: list[DatasetRecord],
) -> list[DatasetRecord]:
    """
    Add missing dataset slugs from companion probe JSON when available.

    Supports either:
      - <markdown_stem>.probe.json
      - <site_base>.probe.json for numbered files like OpenOrganelle_01.md
    """
    p = markdown_path.expanduser().resolve()
    candidates: list[Path] = [p.with_name(f"{p.stem}.probe.json")]
    m = re.match(r"^(?P<base>.+)_\d+$", p.stem)
    if m:
        candidates.append(p.with_name(f"{m.group('base')}.probe.json"))

    probe_path: Path | None = next((x for x in candidates if x.is_file()), None)
    if probe_path is None:
        return records

    try:
        payload = json.loads(probe_path.read_text(encoding="utf-8"))
        site_name = str(payload.get("site") or "").strip().lower()
        ds = payload.get("datasets")
        if not isinstance(ds, dict):
            return records
        probe_names = [str(k).strip() for k in ds.keys() if str(k).strip()]
    except Exception:
        return records

    by_name = {r.dataset_name: r for r in records}
    for name in probe_names:
        if name not in by_name:
            rec = DatasetRecord(dataset_name=name)
            records.append(rec)
            by_name[name] = rec

    # BossDB markdown currently carries channel tables (not ``- dataset_name=...`` rows),
    # so hydrate DatasetRecord fields directly from probe JSON triage keys.
    if site_name.startswith("bossdb"):
        for name, raw in ds.items():
            if not isinstance(raw, dict):
                continue
            rec = by_name.get(str(name))
            if rec is None:
                continue

            uri = str(raw.get("uri") or "").strip()
            channel = str(raw.get("channel") or "").strip() or rec.dataset_name.rsplit("/", 1)[-1]
            channel_type = str(raw.get("channel_type") or "").strip().lower()
            mito_quality = str(raw.get("mito_label_quality") or "").strip().lower()
            organism = str(raw.get("organism") or "").strip()
            modality = str(raw.get("modality") or "").strip().lower()

            if organism and not rec.sample_organism:
                rec.sample_organism = organism
            if not rec.sample_type:
                rec.sample_type = "bossdb"
            if not rec.sample_name:
                rec.sample_name = str(raw.get("experiment") or "")
            if not rec.description:
                rec.description = str(raw.get("description") or "")
            if not rec.source_url:
                rec.source_url = str(raw.get("source_url") or raw.get("source_api") or "")
            vox = raw.get("voxel_size_nm")
            if isinstance(vox, list) and vox and not rec.grid_spacing:
                rec.grid_spacing = json.dumps(vox, ensure_ascii=False)
                rec.grid_axes = "['z','y','x']"
                rec.grid_spacing_unit = "nm"
            if raw.get("labeling") == "mitochondria":
                rec.mitochondria_in_layer_names = True

            if channel_type == "image" and uri and not rec.download_volume_url:
                rec.download_volume_url = uri
            if mito_quality == "good":
                if uri and not rec.download_mito_mask_url:
                    rec.download_mito_mask_url = uri
                if not rec.download_mito_mask_quality:
                    rec.download_mito_mask_quality = "good"
                if not rec.download_mito_mask_segmentation_kind:
                    rec.download_mito_mask_segmentation_kind = "annotation"
            elif mito_quality == "prediction_like":
                if uri and not rec.download_mito_prediction_url:
                    rec.download_mito_prediction_url = uri

            # Add one synthetic layer so Studio layer counts / filters work for BossDB.
            has_layer = any((ly.layer_name == channel and (ly.layer_url or "") == uri) for ly in rec.layers)
            if not has_layer:
                if channel_type == "image":
                    role = "raw_or_other_layers"
                    ctype = "em" if any(k in modality for k in ("em", "sem", "tem", "xnh", "x-ray")) else "image"
                else:
                    role = "predictions" if mito_quality == "prediction_like" else "ground_truths"
                    ctype = "prediction" if mito_quality == "prediction_like" else "segmentation"
                rec.layers.append(
                    LayerRecord(
                        layer_role=role,
                        layer_kind="image",
                        layer_name=channel or rec.dataset_name,
                        layer_url=uri or None,
                        layer_content_type=ctype,
                        semantic=(mito_quality == "good"),
                        raw_token=f"bossdb:{channel}[url={uri}][content_type={ctype}]",
                    )
                )
    return records

