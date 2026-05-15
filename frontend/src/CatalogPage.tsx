import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  CatalogDatabase,
  CatalogDatasetDetail,
  CatalogDatasetRow,
  CatalogFilterSpec,
  CatalogLayer,
  CatalogQueryFilters,
} from "./api";
import {
  getCatalogDatabases,
  getCatalogDatasetDetail,
  getCatalogDatasets,
  getCatalogFilters,
} from "./api";
import { STUDIO_SELECT_EMPTY, STUDIO_SELECT_LOADING, StudioUpdatingBadge } from "./StudioUi";

// ── helpers ───────────────────────────────────────────────────────────────────

function parseOrganism(raw: string | null): string[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) return (parsed as unknown[]).map(String).filter(Boolean);
  } catch {
    /* ignore */
  }
  return raw.trim() ? [raw.trim()] : [];
}

function formatBytes(n: number | null): string {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

const ROLE_LABELS: Record<string, string> = {
  ground_truths: "Ground truths",
  predictions: "Predictions",
  raw_or_other_layers: "Raw / other layers",
};

const ROLE_ORDER = ["ground_truths", "predictions", "raw_or_other_layers"] as const;

const PATH_SOURCE_COLORS: Record<string, string> = {
  s3_probe: "catalog-badge-s3",
  scraped_postgrest: "catalog-badge-postgrest",
  layer_url: "catalog-badge-layer",
  template_fallback: "catalog-badge-template",
  incomplete: "catalog-badge-incomplete",
};

// ── sub-components ─────────────────────────────────────────────────────────────

function FilterSection({
  title,
  items,
  active,
  onToggle,
  labelMap,
  showWhenEmpty = false,
}: {
  title: string;
  items: (string | null)[];
  active: Set<string>;
  onToggle: (v: string) => void;
  labelMap?: Record<string, string>;
  showWhenEmpty?: boolean;
}) {
  if (items.length === 0 && !showWhenEmpty) return null;
  return (
    <div className="catalog-filter-section">
      <div className="catalog-filter-section-title">{title}</div>
      {items.map((item) => {
        const key = item ?? "null";
        const label = item == null ? "unset" : (labelMap?.[item] ?? item);
        return (
          <label key={key} className="catalog-filter-item">
            <input
              type="checkbox"
              checked={active.has(key)}
              onChange={() => onToggle(key)}
            />
            <span>{label}</span>
          </label>
        );
      })}
    </div>
  );
}

function DatasetRow({
  ds,
  selected,
  onClick,
}: {
  ds: CatalogDatasetRow;
  selected: boolean;
  onClick: () => void;
}) {
  const organisms = parseOrganism(ds.sample_organism);
  const orgLabel = organisms.length ? organisms.join(", ") : "—";
  return (
    <tr
      className={`catalog-table-row${selected ? " catalog-row-selected" : ""}`}
      onClick={onClick}
    >
      <td className="catalog-col-name">
        <span className="catalog-dataset-name">{ds.dataset_name}</span>
      </td>
      <td className="catalog-col-organism">{orgLabel}</td>
      <td>{ds.sample_type ?? "—"}</td>
      <td>
        <div className="catalog-organelle-badges">
          {ds.mitochondria_in_layer_names === 1 && (
            <span className="catalog-badge catalog-badge-mito" title="Has mitochondria label">mito</span>
          )}
          {ds.n_em > 0 && (
            <span className="catalog-badge catalog-badge-em" title="Has EM image layers">EM</span>
          )}
          {ds.n_pred > 0 && (
            <span className="catalog-badge catalog-badge-pred" title="Has prediction layers">pred</span>
          )}
          {ds.n_gt > 0 && (
            <span className="catalog-badge catalog-badge-gt" title="Has ground truth layers">GT</span>
          )}
          {ds.s3_probe_img_path && (
            <span className="catalog-badge catalog-badge-s3" title="S3 probe verified">S3✓</span>
          )}
        </div>
      </td>
      <td className="catalog-col-layers">
        <span title="Ground truths">{ds.n_gt} GT</span>
        {" / "}
        <span title="Predictions">{ds.n_pred} pred</span>
        {" / "}
        <span title="Raw/other">{ds.n_raw} raw</span>
      </td>
      <td>
        {ds.ready_labeled === 1 ? (
          <span className="catalog-ready-pill">labeled</span>
        ) : ds.ready_em_only === 1 ? (
          <span className="catalog-em-only-pill">EM only</span>
        ) : (
          <span className="catalog-no-pill">—</span>
        )}
      </td>
      <td>
        {ds.download_mito_mask_quality ? (
          <span
            className={`catalog-quality-pill catalog-quality-${ds.download_mito_mask_quality.replace(/[^a-z]/gi, "_")}`}
          >
            {ds.download_mito_mask_quality}
          </span>
        ) : (
          "—"
        )}
      </td>
    </tr>
  );
}

function DetailPanel({
  detail,
  loading,
  error,
  onClose,
}: {
  detail: CatalogDatasetDetail | null;
  loading: boolean;
  error: string;
  onClose: () => void;
}) {
  if (loading) {
    return (
      <div className="catalog-detail-panel">
        <div className="catalog-detail-placeholder">Updating ...</div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="catalog-detail-panel">
        <div className="catalog-detail-error-msg">{error}</div>
      </div>
    );
  }
  if (!detail) return null;

  const { dataset, layers, resolved, sources } = detail;
  const byRole: Record<string, CatalogLayer[]> = {
    ground_truths: [],
    predictions: [],
    raw_or_other_layers: [],
  };
  for (const l of layers) {
    (byRole[l.layer_role] ??= []).push(l);
  }

  const psrc = resolved?.path_source ?? null;
  const psrcClass = psrc ? (PATH_SOURCE_COLORS[psrc] ?? "catalog-badge-layer") : "";

  const dsFields: { label: string; value: string | null }[] = [
    { label: "Stage", value: dataset.stage as string | null },
    { label: "Organism", value: parseOrganism(dataset.sample_organism as string | null).join(", ") || null },
    { label: "Sample type", value: dataset.sample_type as string | null },
    { label: "Sample subtype", value: dataset.sample_subtype as string | null },
    { label: "Sample name", value: dataset.sample_name as string | null },
    { label: "Description", value: dataset.description as string | null },
    { label: "Grid spacing", value: dataset.grid_spacing ? `${dataset.grid_spacing as string} ${dataset.grid_spacing_unit as string ?? ""}` : null },
    { label: "Grid dimensions", value: dataset.grid_dimensions ? `${dataset.grid_dimensions as string} (${dataset.grid_dimensions_unit as string ?? "voxel"})` : null },
    { label: "Grid axes", value: dataset.grid_axes as string | null },
    { label: "Mito mask quality", value: dataset.download_mito_mask_quality as string | null },
    { label: "Mito mask URL", value: dataset.download_mito_mask_url as string | null },
    { label: "Mito prediction URL", value: dataset.download_mito_prediction_url as string | null },
    { label: "Volume URL", value: dataset.download_volume_url as string | null },
    { label: "S3 probe error", value: dataset.s3_probe_error as string | null },
    { label: "S3 label names", value: dataset.s3_probe_label_names as string | null },
  ].filter((f) => f.value != null && String(f.value).trim());

  return (
    <div className="catalog-detail-panel">
      <div className="catalog-detail-header">
        <strong className="catalog-detail-name">{String(dataset.dataset_name)}</strong>
        {psrc && <span className={`catalog-badge ${psrcClass}`}>{psrc}</span>}
        <button className="catalog-detail-close ghost" onClick={onClose} aria-label="Close detail">
          ×
        </button>
      </div>

      <div className="catalog-detail-columns">
        {/* Left: resolved + properties */}
        <div className="catalog-detail-left">
          {resolved && (
            <div className="catalog-detail-card">
              <div className="catalog-detail-card-title">Resolved paths</div>
              <dl className="catalog-detail-dl">
                <dt>EM path</dt>
                <dd><code className="catalog-path-code">{resolved.resolved_em_path ?? "—"}</code></dd>
                <dt>Mito seg path</dt>
                <dd><code className="catalog-path-code">{resolved.resolved_mito_seg_path ?? "—"}</code></dd>
                <dt>Voxel size</dt>
                <dd>{resolved.resolved_voxel_nm ?? "—"} nm</dd>
                <dt>Ready</dt>
                <dd>
                  {resolved.ready_labeled === 1 && <span className="catalog-ready-pill">labeled</span>}
                  {resolved.ready_em_only === 1 && resolved.ready_labeled !== 1 && (
                    <span className="catalog-em-only-pill">EM only</span>
                  )}
                  {!resolved.ready_labeled && !resolved.ready_em_only && "—"}
                </dd>
              </dl>
            </div>
          )}

          <div className="catalog-detail-card">
            <div className="catalog-detail-card-title">Dataset properties</div>
            <dl className="catalog-detail-dl">
              {dsFields.map((f) => (
                <div key={f.label} className="catalog-detail-dl-row">
                  <dt>{f.label}</dt>
                  <dd>
                    {f.label.toLowerCase().includes("url") || f.label.toLowerCase().includes("path") ? (
                      <code className="catalog-path-code">{f.value}</code>
                    ) : (
                      String(f.value)
                    )}
                  </dd>
                </div>
              ))}
            </dl>
            {sources.length > 0 && (
              <div className="catalog-sources">
                <span className="catalog-detail-card-title">Sources</span>
                {sources.map((s) => (
                  <div key={s} className="catalog-source-url">{s}</div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right: layers grouped by role */}
        <div className="catalog-detail-right">
          {ROLE_ORDER.map((role) => {
            const roleLayers = byRole[role];
            if (!roleLayers || roleLayers.length === 0) return null;
            return (
              <div key={role} className="catalog-detail-card">
                <div className="catalog-detail-card-title">
                  {ROLE_LABELS[role]} <span className="catalog-layer-count">({roleLayers.length})</span>
                </div>
                <div className="catalog-layer-table-wrap">
                  <table className="catalog-layer-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Kind</th>
                        <th>Content</th>
                        <th>Format</th>
                        <th>URL</th>
                      </tr>
                    </thead>
                    <tbody>
                      {roleLayers.map((l) => (
                        <tr key={l.id}>
                          <td className="catalog-layer-name">{l.layer_name}</td>
                          <td>
                            <span className={`catalog-badge catalog-badge-${l.layer_kind}`}>{l.layer_kind}</span>
                          </td>
                          <td>{l.layer_content_type ?? "—"}</td>
                          <td>{l.layer_format ?? "—"}</td>
                          <td
                            className="catalog-layer-url"
                            title={l.layer_url ?? ""}
                          >
                            {l.layer_url
                              ? l.layer_url.length > 55
                                ? `${l.layer_url.slice(0, 55)}…`
                                : l.layer_url
                              : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
          {layers.length === 0 && (
            <div className="catalog-detail-card">
              <div className="catalog-detail-placeholder">No layer records found.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── main component ─────────────────────────────────────────────────────────────

type ActiveFilters = {
  stages: Set<string>;
  organisms: Set<string>;
  sampleTypes: Set<string>;
  layerRoles: Set<string>;
  organelles: Set<string>;
  contentTypes: Set<string>;
  booleans: Map<string, boolean>;
};

const EMPTY_FILTERS: ActiveFilters = {
  stages: new Set(),
  organisms: new Set(),
  sampleTypes: new Set(),
  layerRoles: new Set(),
  organelles: new Set(),
  contentTypes: new Set(),
  booleans: new Map(),
};

function cloneFilters(f: ActiveFilters): ActiveFilters {
  return {
    stages: new Set(f.stages),
    organisms: new Set(f.organisms),
    sampleTypes: new Set(f.sampleTypes),
    layerRoles: new Set(f.layerRoles),
    organelles: new Set(f.organelles),
    contentTypes: new Set(f.contentTypes),
    booleans: new Map(f.booleans),
  };
}

function filtersToQuery(f: ActiveFilters): CatalogQueryFilters {
  const q: CatalogQueryFilters = {};
  if (f.stages.size) q.stage = [...f.stages];
  if (f.organisms.size) q.organism = [...f.organisms];
  if (f.sampleTypes.size) q.sample_type = [...f.sampleTypes];
  if (f.layerRoles.size) q.layer_role = [...f.layerRoles];
  if (f.organelles.size) q.organelle = [...f.organelles];
  if (f.contentTypes.size) q.content_type = [...f.contentTypes];
  for (const [k, v] of f.booleans) {
    (q as Record<string, unknown>)[k] = v;
  }
  return q;
}

export function CatalogPage() {
  const [databases, setDatabases] = useState<CatalogDatabase[]>([]);
  const [dbsLoading, setDbsLoading] = useState(false);
  const [selectedStem, setSelectedStem] = useState("");

  const [filterSpec, setFilterSpec] = useState<CatalogFilterSpec | null>(null);
  const [filterSpecLoading, setFilterSpecLoading] = useState(false);

  const [activeFilters, setActiveFilters] = useState<ActiveFilters>(EMPTY_FILTERS);

  const [datasetList, setDatasetList] = useState<CatalogDatasetRow[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [listLoading, setListLoading] = useState(false);

  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [detail, setDetail] = useState<CatalogDatasetDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [catalogQuietNotice, setCatalogQuietNotice] = useState("");

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load database list on mount
  useEffect(() => {
    setDbsLoading(true);
    getCatalogDatabases()
      .then((r) => {
        setCatalogQuietNotice("");
        setDatabases(r.databases);
        if (r.databases.length > 0) setSelectedStem(r.databases[0].stem);
      })
      .catch(() => {
        // Quiet empty state: no red popup when data/catalog is unavailable.
        setDatabases([]);
        setSelectedStem("");
        setCatalogQuietNotice("Catalog is not available right now. If Stage 2 has not run yet, this is expected.");
      })
      .finally(() => setDbsLoading(false));
  }, []);

  // Load filter spec when stem changes
  useEffect(() => {
    if (!selectedStem) return;
    setFilterSpec(null);
    setActiveFilters(EMPTY_FILTERS);
    setSelectedName(null);
    setDetail(null);
    setFilterSpecLoading(true);
    getCatalogFilters(selectedStem)
      .then((r) => {
        setCatalogQuietNotice("");
        setFilterSpec(r);
      })
      .catch(() => {
        setFilterSpec(null);
        setCatalogQuietNotice("Catalog filters are unavailable right now.");
      })
      .finally(() => setFilterSpecLoading(false));
  }, [selectedStem]);

  // Fetch datasets when stem or filters change (debounced)
  useEffect(() => {
    if (!selectedStem) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setListLoading(true);
      getCatalogDatasets(selectedStem, filtersToQuery(activeFilters))
        .then((r) => {
          setCatalogQuietNotice("");
          setDatasetList(r.datasets);
          setTotalCount(r.total);
        })
        .catch(() => {
          setDatasetList([]);
          setTotalCount(0);
          setCatalogQuietNotice("Catalog dataset list is unavailable right now.");
        })
        .finally(() => setListLoading(false));
    }, 180);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [selectedStem, activeFilters]);

  // Load detail when a dataset is selected
  useEffect(() => {
    if (!selectedName || !selectedStem) {
      setDetail(null);
      return;
    }
    setDetailLoading(true);
    setDetailError("");
    setDetail(null);
    getCatalogDatasetDetail(selectedStem, selectedName)
      .then(setDetail)
      .catch((e: unknown) => setDetailError(e instanceof Error ? e.message : String(e)))
      .finally(() => setDetailLoading(false));
  }, [selectedStem, selectedName]);

  const toggleSet = useCallback(
    (field: keyof Pick<ActiveFilters, "stages" | "organisms" | "sampleTypes" | "layerRoles" | "organelles" | "contentTypes">, value: string) => {
      setActiveFilters((prev) => {
        const next = cloneFilters(prev);
        const s = next[field];
        if (s.has(value)) s.delete(value); else s.add(value);
        return next;
      });
    },
    [],
  );

  const toggleBool = useCallback((key: string, checked: boolean) => {
    setActiveFilters((prev) => {
      const next = cloneFilters(prev);
      if (checked) next.booleans.set(key, true);
      else next.booleans.delete(key);
      return next;
    });
  }, []);

  const resetFilters = useCallback(() => setActiveFilters(EMPTY_FILTERS), []);

  const hasAnyFilter = useMemo(
    () =>
      activeFilters.organisms.size > 0 ||
      activeFilters.sampleTypes.size > 0 ||
      activeFilters.stages.size > 0 ||
      activeFilters.layerRoles.size > 0 ||
      activeFilters.organelles.size > 0 ||
      activeFilters.contentTypes.size > 0 ||
      activeFilters.booleans.size > 0,
    [activeFilters],
  );

  const dbSelectValue = dbsLoading
    ? STUDIO_SELECT_LOADING
    : databases.length === 0
      ? STUDIO_SELECT_EMPTY
      : selectedStem;
  const isBossDb = selectedStem.trim().toLowerCase().startsWith("bossdb");
  const organismItems = filterSpec?.organisms ?? [];
  const sampleTypeItems = filterSpec?.sample_types ?? [];
  const organelleItems = filterSpec?.organelles_present ?? [];
  const organelleLabels = filterSpec?.organelle_labels ?? {};
  const mitoFacetKeys = useMemo(() => new Set(["has_mito_gt", "has_good_mito_mask"]), []);
  const booleanFacetEntries = useMemo(
    () => Object.entries(filterSpec?.boolean_facets ?? {}),
    [filterSpec],
  );
  const generalBooleanFacets = useMemo(
    () => booleanFacetEntries.filter(([key]) => !mitoFacetKeys.has(key)),
    [booleanFacetEntries, mitoFacetKeys],
  );
  const mitoBooleanFacets = useMemo(
    () => booleanFacetEntries.filter(([key]) => mitoFacetKeys.has(key)),
    [booleanFacetEntries, mitoFacetKeys],
  );
  const nonZeroGeneralBooleanFacets = useMemo(
    () => generalBooleanFacets.filter(([, facet]) => Number(facet.count || 0) > 0),
    [generalBooleanFacets],
  );
  const nonZeroMitoBooleanFacets = useMemo(
    () => mitoBooleanFacets.filter(([, facet]) => Number(facet.count || 0) > 0),
    [mitoBooleanFacets],
  );
  // Layer role/content facets intentionally hidden (redundant).
  const effectiveOrganelleItems = isBossDb ? [] : organelleItems;

  return (
    <div className="catalog-page">
      {/* ── Sidebar ── */}
      <div className="catalog-sidebar">
        <div className="catalog-db-selector">
          <label className="studio-label" htmlFor="catalog-db-pick">Database</label>
          <div className="catalog-db-row">
            <select
              id="catalog-db-pick"
              className="field-input"
              value={dbSelectValue}
              onChange={(e) => {
                const v = e.target.value;
                if (v === STUDIO_SELECT_LOADING || v === STUDIO_SELECT_EMPTY) return;
                setSelectedStem(v);
              }}
              disabled={dbsLoading}
            >
              {dbsLoading ? (
                <option value={STUDIO_SELECT_LOADING}>Updating ...</option>
              ) : databases.length === 0 ? (
                <option value={STUDIO_SELECT_EMPTY}>No databases — run stage 2</option>
              ) : (
                databases.map((db) => (
                  <option key={db.stem} value={db.stem}>
                    {db.stem} ({formatBytes(db.size_bytes)})
                  </option>
                ))
              )}
            </select>
            <StudioUpdatingBadge active={dbsLoading} label="Updating ..." />
          </div>
        </div>
        {catalogQuietNotice ? (
          <p className="muted-note" style={{ marginTop: "0.45rem" }}>{catalogQuietNotice}</p>
        ) : null}

        {filterSpecLoading && (
          <div className="catalog-filter-loading">Updating ...</div>
        )}

        {filterSpec && (
          <div className="catalog-filter-panel">
            <div className="catalog-filter-header">
              <span>Filters</span>
              {hasAnyFilter && (
                <button type="button" className="linkish catalog-clear-btn" onClick={resetFilters}>
                  Clear all
                </button>
              )}
            </div>

            <FilterSection
              title="Organism"
              items={organismItems}
              active={activeFilters.organisms}
              onToggle={(v) => toggleSet("organisms", v)}
              showWhenEmpty
            />
            <FilterSection
              title="Sample type"
              items={sampleTypeItems}
              active={activeFilters.sampleTypes}
              onToggle={(v) => toggleSet("sampleTypes", v)}
              showWhenEmpty
            />
            <div className="catalog-filter-section">
              <div className="catalog-filter-section-title">Status / availability</div>
              {nonZeroGeneralBooleanFacets.map(([key, facet]) => (
                <label key={key} className="catalog-filter-bool-row">
                  <input
                    type="checkbox"
                    checked={activeFilters.booleans.has(key)}
                    onChange={(e) => toggleBool(key, e.target.checked)}
                  />
                  <span>{facet.label}</span>
                  <span className="catalog-filter-count">({facet.count})</span>
                </label>
              ))}
            </div>

            <div className="catalog-filter-section">
              <div className="catalog-filter-section-title">Mito-specific</div>
              <FilterSection
                title="Organelles in layers"
                items={effectiveOrganelleItems}
                active={activeFilters.organelles}
                onToggle={(v) => toggleSet("organelles", v)}
                labelMap={organelleLabels}
                showWhenEmpty
              />
              {nonZeroMitoBooleanFacets.map(([key, facet]) => (
                <label key={key} className="catalog-filter-bool-row">
                  <input
                    type="checkbox"
                    checked={activeFilters.booleans.has(key)}
                    onChange={(e) => toggleBool(key, e.target.checked)}
                  />
                  <span>{facet.label}</span>
                  <span className="catalog-filter-count">({facet.count})</span>
                </label>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Main area ── */}
      <div className="catalog-main">
        {/* Dataset list table */}
        <div className="catalog-list-area" style={{ flex: selectedName ? "0 0 auto" : "1 1 0", maxHeight: selectedName ? "45%" : undefined }}>
          <div className="catalog-list-head">
            <span className="catalog-result-count">
              {listLoading ? "Updating ..." : `${totalCount} dataset(s)`}
              {hasAnyFilter && !listLoading && (
                <span className="catalog-filter-active-note"> (filtered)</span>
              )}
            </span>
            <StudioUpdatingBadge active={listLoading} label="Updating ..." />
          </div>
          <div className="catalog-table-wrap">
            <table className="catalog-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Organism</th>
                  <th>Sample type</th>
                  <th>Labels</th>
                  <th>Layers GT/pred/raw</th>
                  <th>Ready</th>
                  <th>Mito quality</th>
                </tr>
              </thead>
              <tbody>
                {datasetList.map((ds) => (
                  <DatasetRow
                    key={ds.id}
                    ds={ds}
                    selected={selectedName === ds.dataset_name}
                    onClick={() =>
                      setSelectedName((prev) =>
                        prev === ds.dataset_name ? null : ds.dataset_name,
                      )
                    }
                  />
                ))}
                {datasetList.length === 0 && !listLoading && (
                  <tr>
                    <td colSpan={7} className="catalog-empty-cell">
                      {selectedStem
                        ? "No datasets match the current filters."
                        : "Select a database above."}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Detail panel */}
        {selectedName && (
          <div className="catalog-detail-area">
            <DetailPanel
              detail={detail}
              loading={detailLoading}
              error={detailError}
              onClose={() => setSelectedName(null)}
            />
          </div>
        )}
      </div>
    </div>
  );
}
