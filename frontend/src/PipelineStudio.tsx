import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { StudioRunResult, StudioPendingDownloads, StudioWebsiteScrapeResult, StudioWebsiteSummary, DatasetStatusItem, InventoryCatalogueResponse, CatalogueRow, StudioSlurmRunState, MitoLeInspectRow, MitoLeCatalogueRow } from "./api";
import {
  getStudioProbes,
  getStudioPendingDownloads,
  getStudioSites,
  getStudioSummary,
  getStudioWebsites,
  getStudioWebsite,
  getPipelineProgress,
  getStudioWebsiteScrapeState,
  postStudioWebsiteScrapeStateClear,
  postStudioDatabaseBuildStateClear,
  getStudioDatabaseBuildState,
  STUDIO_PIPELINE_DOWNLOADER_SYNC_SCRIPT_PATH,
  postStudioDownloader,
  postStudioMitoleDownloader,
  getStudioDownloaderPreview,
  getStudioDownloaderScripts,
  getStudioRunDownloaderScriptState,
  postStudioRunDownloaderScriptStateClear,
  postStudioRunDownloaderScriptCancel,
  postStudioTraining,
  postStudioInference,
  postStudioPostprocessing,
  postStudioEvaluation,
  getStudioTrainingState,
  postStudioTrainingStateClear,
  getStudioInferenceState,
  postStudioInferenceStateClear,
  postStudioPreprocessSelective,
  getStudioRunPreprocessSelectiveState,
  postStudioRunPreprocessSelectiveStateClear,
  postStudioRunPreprocessSelectiveCancel,
  getStudioDataInspect,
  getStudioPostprocessingFiles,
  getStudioRawEmStacks,
  postStudioDatabaseBuild,
  deleteStudioWebsite,
  postStudioWebsiteSave,
  postStudioWebsiteScrapeStream,
  postStudioScrapeCancel,
  postChatStop,
  getDatasetsStatus,
  deleteDatasetFiles,
  hideDatasets,
  unhideDatasets,
  setDatasetUseInModel,
  getInventoryCatalogue,
  postStudioResetDownloadedTraining,
  postStudioResetModelDownloadedDataHistory,
  getMitoLeConfig,
  getMitoLeSubfolders,
  postMitoLeConfig,
  getMitoLeInspect,
  getMitoLeCatalogue,
} from "./api";
import {
  DEFAULT_MITOLE_BASE_PATH,
  REL_DATA_RAW,
  REL_NNUNET_DATASET,
  REL_NNUNET_LABELS_TS_INSTANCE,
  REL_OUTPUTS_BC,
  REL_OUTPUTS_POSTPROCESSED,
} from "./paths";
import { MarkdownField } from "./MarkdownField";
import { STUDIO_SELECT_LOADING, SizeStableLabel, StudioUpdatingBadge } from "./StudioUi";
import { CatalogPage } from "./CatalogPage";

export type StudioPageId =
  | "intro"
  | "inventory"
  | "scraper"
  | "database"
  | "downloader"
  | "processor"
  | "training"
  | "inference"
  | "summary";

export type AppView =
  | "home"
  | "data-hpc"
  | "data-existing-legacy"
  | "data-scrape"
  | "model-training"
  | "model-inference"
  | "model-postprocessing"
  | "pipeline-summary";

type DownloaderDatasetSplit = { training: number; inference: number };

const MAX_CROPS_PER_DATASET = 16;

function clampDownloaderSplit(side: "training" | "inference", value: number, otherValue: number): number {
  const safeOther = Math.max(0, Math.min(MAX_CROPS_PER_DATASET, Math.floor(Number(otherValue) || 0)));
  const hardCap = Math.max(0, MAX_CROPS_PER_DATASET - safeOther);
  const safeValue = Math.max(0, Math.floor(Number(value) || 0));
  if (!Number.isFinite(safeValue)) return 0;
  if (side === "training" || side === "inference") {
    return Math.min(safeValue, hardCap);
  }
  return Math.min(safeValue, MAX_CROPS_PER_DATASET);
}

function appViewFromHash(hash: string): AppView {
  switch (hash) {
    case "#data/hpc":        return "data-hpc";
    case "#data/scrape":     return "data-scrape";
    case "#legacy-hpc":      return "data-existing-legacy";
    case "#model/inference": return "model-inference";
    case "#model/postprocessing": return "model-postprocessing";
    case "#model":           return "model-training";
    case "#summary":         return "pipeline-summary";
    default:
      if (hash === "" || hash === "#home") return "home";
      if (hash.startsWith("#data")) return "data-hpc";
      return "home";
  }
}

function hashFromAppView(v: AppView): string {
  switch (v) {
    case "data-hpc":         return "#data/hpc";
    case "data-existing-legacy": return "#legacy-hpc";
    case "data-scrape":      return "#data/scrape";
    case "model-training":   return "#model";
    case "model-inference":  return "#model/inference";
    case "model-postprocessing": return "#model/postprocessing";
    case "pipeline-summary": return "#summary";
    case "home":             return "";
  }
}

/** Browser persistence for preprocessor form (download run, split-CC, row picks). */
const STUDIO_PREPROCESS_LS_PREFIX = "mito2.studio.preprocess.v1";
function studioPreprocessStorageKey(sessionId: string) {
  return `${STUDIO_PREPROCESS_LS_PREFIX}:${sessionId}`;
}

/** Browser persistence for data downloader (stage 3): per-dataset train/inference splits. */
const STUDIO_DOWNLOADER_LS_PREFIX = "mito2.studio.downloader.v3";
function studioDownloaderStorageKey(sessionId: string) {
  return `${STUDIO_DOWNLOADER_LS_PREFIX}:${sessionId}`;
}
/** Legacy v2 stored voxel/crop strings; still read for downloader migration only. */
const STUDIO_DOWNLOADER_LS_LEGACY_V2 = "mito2.studio.downloader.v2";
function studioDownloaderLegacyV2Key(sessionId: string) {
  return `${STUDIO_DOWNLOADER_LS_LEGACY_V2}:${sessionId}`;
}

/** Fixed for script generation in this pipeline (not configurable in the UI). */
const STUDIO_DOWNLOADER_FIXED_VOXEL_NM = "16,16,16";
const STUDIO_DOWNLOADER_FIXED_CROP_VOXELS = "128,128,128";

/** Count EM stacks under ``<run>/images/`` (``Images/`` OK) matching ``*_im.h5`` — fallback until ``GET /data/raw-em-stacks`` loads. */
function countPreprocessableEmStacksInRun(
  dataInspect: { raw_base: string; raw_datasets: { name: string; path: string }[] } | null,
  run: string,
  optionNames: string[],
): number {
  if (!dataInspect || optionNames.length === 0) return 0;
  if (!run || !optionNames.includes(run)) return 0;
  const rawNorm = dataInspect.raw_base.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
  const prefix = `${rawNorm}/${run.toLowerCase()}/`;
  return dataInspect.raw_datasets.filter((d) => {
    const p = d.path.replace(/\\/g, "/").toLowerCase();
    return p.startsWith(prefix) && p.includes("/images/") && d.name.toLowerCase().endsWith("_im.h5");
  }).length;
}

function sameStringSet(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const sa = new Set(a);
  if (sa.size !== new Set(b).size) return false;
  for (const v of b) {
    if (!sa.has(v)) return false;
  }
  return true;
}

function toEasternString(rawIso: string): string {
  const raw = (rawIso || "").trim();
  if (!raw) return "";
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return raw;
  return d.toLocaleString(undefined, { timeZone: "America/New_York" });
}

const SCRAPE_NAV: { id: StudioPageId; label: string; stageNum: number }[] = [
  { id: "inventory",  label: "Inventory",            stageNum: 0 },
  { id: "scraper",    label: "Web Scraper",           stageNum: 1 },
  { id: "database",   label: "Database Builder",      stageNum: 2 },
  { id: "downloader", label: "Data Downloader", stageNum: 3 },
];

type HpcPageId = "browser" | "catalog" | "downloader";
const HPC_NAV: { id: HpcPageId; label: string; stageNum: number }[] = [
  { id: "browser", label: "Folder Browser", stageNum: 1 },
  { id: "catalog", label: "Database Builder", stageNum: 2 },
  { id: "downloader", label: "Data Downloader", stageNum: 3 },
];

type Props = {
  sessionId: string;
  onNotify: (title: string, detail?: string, kind?: "ok" | "err") => void;
  /** When the agent chat column is collapsed, studio layout uses full workspace width. */
  chatPanelCollapsed?: boolean;
};

function RunLog({ result }: { result: StudioRunResult | null }) {
  if (!result) return null;
  const og = result.downloader_generations ?? result.openorganelle_generations;
  const failed = !result.ok || result.returncode !== 0;
  return (
    <details className="studio-run-log" open={failed}>
      <summary>
        Output <span className="studio-run-exit">exit {result.returncode}</span>
      </summary>
      {result.slurm_job_id || result.slurm_out_path || result.slurm_err_path || result.training_log_path ? (
        <div className="muted-note" style={{ marginBottom: "0.75rem" }}>
          {result.slurm_job_id ? (
            <p style={{ marginTop: 0, marginBottom: "0.35rem" }}>
              Slurm job <code>{result.slurm_job_id}</code>
            </p>
          ) : null}
          {result.slurm_out_path ? (
            <p style={{ marginTop: 0, marginBottom: "0.35rem" }}>
              <code>.out</code>: <code>{result.slurm_out_path}</code>
            </p>
          ) : result.training_log_path ? (
            <p style={{ marginTop: 0, marginBottom: "0.35rem" }}>
              Log: <code>{result.training_log_path}</code>
              {typeof result.training_pid === "number" ? <> (PID {result.training_pid})</> : null}
            </p>
          ) : null}
          {result.slurm_err_path ? (
            <p style={{ margin: 0 }}>
              <code>.err</code>: <code>{result.slurm_err_path}</code>
            </p>
          ) : null}
        </div>
      ) : null}
      {og && og.length > 0 ? (
        <div className="muted-note" style={{ marginBottom: "0.75rem" }}>
          <strong>Generator subprocess (n_crops; voxel and crop size are fixed at 16 nm and 128^3):</strong>
          {og.map((row) => (
            <pre key={row.mode} className="studio-run-pre" style={{ marginTop: "0.35rem" }}>
              [{row.mode}] {row.argv.join(" ")}
              {row.method ? `\nmethod: ${row.method}` : ""}
              {row.env && Object.keys(row.env).length > 0
                ? `\nenv: ${JSON.stringify(row.env)}`
                : ""}
            </pre>
          ))}
          <p className="muted-note" style={{ marginTop: "0.35rem", marginBottom: 0 }}>
            If this block is missing after &quot;Generate script&quot;, restart <code>./mito2</code> so the API loads the
            latest <code>studio_api.py</code> (Uvicorn does not reload Python modules unless{" "}
            <code>MITO2_RELOAD=1</code>).
          </p>
        </div>
      ) : null}
      {result.stdout ? (
        <pre className="studio-run-pre">{result.stdout}</pre>
      ) : null}
      {result.stderr ? (
        <pre className="studio-run-pre studio-run-stderr">{result.stderr}</pre>
      ) : null}
    </details>
  );
}

function WorkspaceRunLog({ result }: { result: StudioWebsiteScrapeResult | null }) {
  if (!result) return null;
  const fe = result.fetch as { ok?: boolean; error?: string; title?: string };
  return (
    <details className="studio-run-log">
      <summary>Workspace scrape result</summary>
      <ul className="studio-ws-paths">
        <li>
          <code>{result.site_md}</code>
        </li>
        {result.datasets_json ? (
          <li>
            <code>{result.datasets_json}</code>
          </li>
        ) : null}
        {result.probe_path ? (
          <li>
            Probe JSON: <code>{result.probe_path}</code>
          </li>
        ) : null}
      </ul>
      <p className="muted-note">
        Fetch: {fe?.ok ? "ok" : "failed"}
        {fe?.error ? ` — ${fe.error}` : ""}
        {fe?.title ? ` — ${fe.title}` : ""}
      </p>
    </details>
  );
}

function SlurmLogPanel(props: {
  title: string;
  outPath: string;
  errPath: string;
  outLog: string;
  errLog: string;
  activeTab: "out" | "err";
  onTabChange: (next: "out" | "err") => void;
  onClear: () => void;
  clearBusy: boolean;
  clearDisabled?: boolean;
  logRoots: string[];
  selectedLogRoot: string;
  onLogRootChange: (next: string) => void;
  summary?: {
    headline?: string;
    runtime?: string;
    ended_at?: string;
    mean_validation_dice?: number | null;
    best_ema_pseudo_dice?: number | null;
    final_epoch?: number | null;
    final_train_loss?: number | null;
    final_val_loss?: number | null;
    final_pseudo_dice?: { values: number[]; mean: number } | null;
  } | null;
  showSummary?: boolean;
}) {
  const {
    title,
    outPath,
    errPath,
    outLog,
    errLog,
    activeTab,
    onTabChange,
    onClear,
    clearBusy,
    clearDisabled,
    logRoots,
    selectedLogRoot,
    onLogRootChange,
    summary,
    showSummary,
  } = props;
  const body = activeTab === "out" ? outLog : errLog;
  const path = activeTab === "out" ? outPath : errPath;
  return (
    <div className="studio-scrape-live-log-wrap" style={{ marginTop: "0.75rem" }}>
      <div className="studio-scrape-live-log-label" style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}>
        <span>
          {title}
        </span>
        <button type="button" className="ghost" disabled={clearBusy || Boolean(clearDisabled)} onClick={onClear}>
          Clear
        </button>
      </div>
      <div style={{ display: "flex", gap: "0.45rem", marginBottom: "0.5rem" }}>
        <select
          className="field-input studio-field"
          value={selectedLogRoot}
          onChange={(e) => onLogRootChange(e.target.value)}
          style={{ maxWidth: "34rem" }}
          aria-label="Run select"
        >
          <option value="">Run select</option>
          {logRoots.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
        <button type="button" className={activeTab === "out" ? "primary" : "ghost"} onClick={() => onTabChange("out")}>
          .out
        </button>
        <button type="button" className={activeTab === "err" ? "primary" : "ghost"} onClick={() => onTabChange("err")}>
          .err
        </button>
      </div>
      {path ? (
        <p className="muted-note" style={{ marginTop: 0, marginBottom: "0.35rem" }}>
          File: <code>{path}</code>
        </p>
      ) : null}
      <pre className="studio-scrape-live-log" aria-live="polite">
        {body || "— No log output yet —"}
      </pre>
      {showSummary ? (
        <div className="studio-run-log" style={{ marginTop: "0.75rem", borderColor: "var(--color-ok, #2d7a2d)" }}>
          <div className="muted-note" style={{ fontWeight: 700, marginBottom: "0.35rem" }}>
            Important training summary
          </div>
          {summary?.headline ? (
            <p style={{ marginTop: 0, marginBottom: "0.35rem" }}>
              <strong>{summary.headline}</strong>
            </p>
          ) : null}
          <p className="muted-note" style={{ marginTop: 0, marginBottom: "0.25rem" }}>
            {summary?.runtime || "Total runtime:"}
          </p>
          <p className="muted-note" style={{ marginTop: 0, marginBottom: "0.35rem" }}>
            {summary?.ended_at || "Job ended at"}
          </p>
          <ul style={{ marginTop: 0, marginBottom: 0 }}>
            <li>
              Mean Validation Dice: <code>{summary?.mean_validation_dice != null ? summary.mean_validation_dice.toFixed(4) : ""}</code>
            </li>
            <li>
              Best EMA pseudo Dice: <code>{summary?.best_ema_pseudo_dice != null ? summary.best_ema_pseudo_dice.toFixed(4) : ""}</code>
            </li>
            <li>
              Final epoch: <code>{summary?.final_epoch != null ? String(summary.final_epoch) : ""}</code>
            </li>
            <li>
              Final train loss: <code>{summary?.final_train_loss != null ? summary.final_train_loss.toFixed(4) : ""}</code>
            </li>
            <li>
              Final val loss: <code>{summary?.final_val_loss != null ? summary.final_val_loss.toFixed(4) : ""}</code>
            </li>
            <li>
              Final pseudo dice mean: <code>{summary?.final_pseudo_dice?.mean != null ? summary.final_pseudo_dice.mean.toFixed(4) : ""}</code>
            </li>
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function parseDownloaderProgress(
  log: string,
  done: boolean,
  fallbackTotal?: number,
): { total: number; completed: number; active: number } | null {
  if (!log.trim()) return null;
  // Local-HPC Stage 3 (mitole): infer total from ``[INFO] new crops to write=N`` (server banner) or
  // summary bullets; count completed units from ``materializing … vol`` lines (wording matches mitole_pipeline).
  const mitoleNew =
    log.match(/new crops to write=(\d+)/i) ??
    log.match(/Pairs to materialize now:\s*(\d+)/i) ??
    log.match(/(\d+)\s+new to write this run\b/i);
  if (mitoleNew) {
    const raw = Number(mitoleNew[1]) || 0;
    if (raw <= 0) {
      /* Mitole no-op (0 new crops): let other heuristics run. */
    } else {
      const total = raw;
      const finished =
        done ||
        /\[DONE\]\s+Copied/i.test(log) ||
        /\[SUMMARY\]\s+catalogue_rows_materialized=/i.test(log);
      const mat = log.match(/materializing (?:training|global) vol\s+(\d+)\/(\d+)/gi);
      let completed = 0;
      if (mat) {
        for (const m of mat) {
          const mm = m.match(/(\d+)\/(\d+)/);
          if (mm) completed = Math.max(completed, Number(mm[1]) || 0);
        }
      }
      const matLegacy = log.match(/materializing training crop\s+(\d+)\/(\d+)/gi);
      if (matLegacy) {
        for (const m of matLegacy) {
          const mm = m.match(/(\d+)\/(\d+)/);
          if (mm) completed = Math.max(completed, Number(mm[1]) || 0);
        }
      }
      const matInf = log.match(/materializing inference vol\s+(\d+)\/(\d+)/gi);
      if (matInf) {
        for (const m of matInf) {
          const mm = m.match(/(\d+)\/(\d+)/);
          if (mm) completed = Math.max(completed, Number(mm[1]) || 0);
        }
      }
      const matInfLegacy = log.match(/materializing inference crop\s+(\d+)\/(\d+)/gi);
      if (matInfLegacy) {
        for (const m of matInfLegacy) {
          const mm = m.match(/(\d+)\/(\d+)/);
          if (mm) completed = Math.max(completed, Number(mm[1]) || 0);
        }
      }
      const doneVolLines = log.match(/\[DONE\][^\n]*(training|inference|global)\s+vol\s+\d+\/\d+/gi);
      if (doneVolLines && doneVolLines.length > 0) {
        completed = Math.max(completed, doneVolLines.length);
      }
      if (finished) completed = total;
      return {
        total,
        completed: Math.min(total, Math.max(0, completed)),
        active: Math.min(total, Math.max(1, completed || (finished ? total : 1))),
      };
    }
  }
  const pairTotalMatch = log.match(/- Planned image\/label pairs:\s*(\d+)/);
  const datasetTotalMatch = log.match(/- Datasets:\s*(\d+)/);
  const windowsMatch = log.match(/- Windows per dataset:\s*(\d+)/);
  const totalFromPairs = pairTotalMatch ? Number(pairTotalMatch[1]) : 0;
  const totalFromDatasetWindows =
    datasetTotalMatch && windowsMatch ? Number(datasetTotalMatch[1]) * Math.max(1, Number(windowsMatch[1])) : 0;
  const totalFromSummary = datasetTotalMatch
    ? Math.max(Number(datasetTotalMatch[1]), totalFromPairs, totalFromDatasetWindows)
    : (fallbackTotal ?? 0);
  const progressMatches = [...log.matchAll(/\[PROGRESS\]\s+dataset\s+(\d+)\/(\d+):/g)];
  const doneMatches = [...log.matchAll(/\[DONE\]\s+dataset\s+(\d+)\/(\d+):/g)];
  const cropMatches = [...log.matchAll(/([a-z0-9_]+_vol\d+(?:_\d+)?)_im\.h5\b/gi)];
  const datasetHeaderCount = (log.match(/^Dataset:\s+/gm) || []).length;

  if (cropMatches.length > 0) {
    const completed = done ? Math.max(cropMatches.length, totalFromSummary) : cropMatches.length;
    const total = Math.max(totalFromSummary, cropMatches.length);
    const active = Math.min(total, Math.max(completed + (done ? 0 : 1), 1));
    return { total, completed: Math.min(completed, total), active };
  }

  // Prefer explicit DONE markers when available.
  if (doneMatches.length > 0) {
    const lastDone = doneMatches[doneMatches.length - 1];
    const completed = Number(lastDone[1]);
    const total = Math.max(Number(lastDone[2]) || 0, totalFromSummary);
    const active = Math.min(total, Math.max(completed + (done ? 0 : 1), 1));
    if (Number.isFinite(total) && total > 0) {
      return {
        total,
        completed: done ? total : Math.min(completed, total),
        active,
      };
    }
  }

  if (progressMatches.length === 0) {
    // Fallback for older/custom scripts: infer progress from "Dataset:" blocks.
    if (datasetHeaderCount > 0 && totalFromSummary > 0) {
      const completed = done ? totalFromSummary : Math.max(0, Math.min(datasetHeaderCount - 1, totalFromSummary));
      const active = Math.min(totalFromSummary, Math.max(datasetHeaderCount, 1));
      return { total: totalFromSummary, completed, active };
    }
    if (totalFromSummary > 0) {
      // Before the first ``[PROGRESS]`` line, show dataset 1 of N (not 0) so the status line reads sensibly.
      return { total: totalFromSummary, completed: done ? totalFromSummary : 0, active: done ? totalFromSummary : 1 };
    }
    return null;
  }
  const last = progressMatches[progressMatches.length - 1];
  const active = Number(last[1]);
  const total = Math.max(Number(last[2]) || 0, totalFromSummary);
  if (!Number.isFinite(active) || !Number.isFinite(total) || total <= 0) return null;
  let completed = Math.max(0, active - 1);
  if (done) completed = total;
  if (completed > total) completed = total;
  return { total, completed, active: Math.min(active, total) };
}

function downloaderResultLooksSuccessful(
  result: StudioRunResult | null | undefined,
  logText: string,
): boolean {
  if (!result) return false;
  if (result.ok) return true;
  const rc = Number((result as { returncode?: unknown }).returncode ?? 1);
  const text = `${String(result.message || "")}\n${String(result.stdout || "")}\n${String(result.stderr || "")}\n${logText}`.toLowerCase();
  if (text.includes("failed datasets:")) return false;
  if (text.includes("integrated preprocess marker verification failed")) return false;
  if (rc === 0) {
    if (/\[done\]\s+dataset\s+\d+\/\d+:/i.test(logText)) return true;
    if (text.includes("no new assets to download")) return true;
  }
  return false;
}

function mergeDownloaderProgress(
  stateProgress: { completed: number; total: number; active: number; dataset: string } | null,
  log: string,
  done: boolean,
): { total: number; completed: number; active: number } | null {
  const parsed = parseDownloaderProgress(log, done, undefined);
  if (!stateProgress && !parsed) return null;
  if (!stateProgress) return parsed;
  if (!parsed) {
    const total = Math.max(0, Number(stateProgress.total) || 0);
    if (total <= 0) return null;
    const completed = Math.max(0, Math.min(total, Number(stateProgress.completed) || 0));
    const active = Math.max(1, Math.min(total, Number(stateProgress.active) || 1));
    return { total, completed: done ? total : completed, active: done ? total : active };
  }
  const total = Math.max(0, Number(stateProgress.total) || 0, parsed.total || 0);
  if (total <= 0) return null;
  const completedRaw = Math.max(0, Number(stateProgress.completed) || 0, parsed.completed || 0);
  const completed = done ? total : Math.min(total, completedRaw);
  const activeRaw = Math.max(1, Number(stateProgress.active) || 1, parsed.active || 1);
  const active = done ? total : Math.min(total, Math.max(activeRaw, completed + 1));
  return { total, completed, active };
}

type DatasetInspectItem = {
  name: string;
  path: string;
  type: string;
  dimensions: number[];
  spacing: number[];
  /** Distinct nonzero label ids for `_seg` H5 (from API); empty for images. */
  label_summary?: string;
};

type DatasetTableRow = DatasetInspectItem & {
  groupKey: string;
  order: number;
};

function naturalCmp(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

function asCellVector(v: number[]): string {
  if (!v || v.length === 0) return "—";
  return v.join(" x ");
}

function asSpacingCell(v: number[]): string {
  if (!v || v.length === 0) return "—";
  return `${v.join(" x ")} nm`;
}

/** ``label_summary`` from inspect: distinct nonzero label IDs; empty for EM-only rows. */
function asLabelSummaryCell(row: DatasetInspectItem): string {
  const s = (row.label_summary ?? "").trim();
  return s || "—";
}

/** Coerce inspect API rows (snake_case or camelCase; sparse spacing) for the preprocessor table. */
function normalizeDatasetInspectRow(d: DatasetInspectItem & Record<string, unknown>): DatasetInspectItem {
  const path = String(d.path ?? "");
  const pl = path.replace(/\\/g, "/").toLowerCase();
  const spacingRaw = (d.spacing ?? d.Spacing) as unknown;
  const toNums = (v: unknown, max: number): number[] => {
    if (!Array.isArray(v)) return [];
    const out: number[] = [];
    for (const x of v.slice(0, max)) {
      const n = typeof x === "number" ? x : Number(x);
      if (!Number.isNaN(n)) out.push(n);
    }
    return out;
  };
  let spacing = toNums(spacingRaw, 3);
  if (spacing.length < 3 && pl.endsWith(".h5") && pl.includes("openorganelle_mito_")) {
    spacing = [16, 16, 16];
  }
  const dimRaw = (d.dimensions ?? d.Dimensions) as unknown;
  let dimensions = toNums(dimRaw, 8).map((x) => Math.round(x));
  if (dimensions.length === 0 && Array.isArray(d.dimensions)) {
    dimensions = d.dimensions.map((x) => Math.round(Number(x))).filter((x) => !Number.isNaN(x));
  }
  const lr = d.label_summary ?? d.labelSummary;
  const label_summary =
    typeof lr === "string" ? lr : lr !== undefined && lr !== null ? String(lr) : "";
  return {
    name: String(d.name ?? ""),
    path,
    type: String(d.type ?? ""),
    dimensions,
    spacing,
    label_summary,
  };
}

function toStableDatasetId(name: string): string {
  const s = String(name || "").trim();
  const lower = s.toLowerCase();
  let stem = s;
  if (lower.endsWith(".nii.gz")) stem = s.slice(0, -7);
  else if (lower.endsWith(".h5")) stem = s.slice(0, -3);
  else if (lower.endsWith(".nrrd")) stem = s.slice(0, -5);
  else if (lower.endsWith(".nii")) stem = s.slice(0, -4);
  return stem.replace(/(_im|_seg|\.im|\.seg)$/i, "");
}

/** Bundled Stage-1 workspaces — not removable from Studio (API returns 403). Case-insensitive. */
const STUDIO_PROTECTED_WEBSITE_SLUGS_LOWER = new Set(["bossdb_01", "openorganelle_01"]);

function studioIsProtectedWebsiteSlug(slug: string): boolean {
  const s = slug.trim().toLowerCase();
  return s.length > 0 && STUDIO_PROTECTED_WEBSITE_SLUGS_LOWER.has(s);
}

/** Mirrors ``GET /api/pipeline`` step_label so agent/chat-issued pipeline runs show the same Studio busy UI as manual runs. */
type RemotePipelineMirror = {
  scrape: boolean;
  database: boolean;
  download: boolean;
  preprocess: boolean;
};

function pipelineStepLabelToMirror(stepLabel: string): RemotePipelineMirror {
  const l = (stepLabel || "").trim();
  return {
    scrape: l === "scrape",
    database: l === "database_build",
    download: l === "download_script",
    preprocess: l === "preprocess",
  };
}

/** Map ``/api/pipeline`` ``last_site_stem`` (BossDB / OpenOrganelle / slug) to bundled workspace slug. */
function pipelineSiteStemToScrapeSlug(stem: string): string {
  const raw = (stem || "").trim();
  if (!raw) return "";
  const t = raw.toLowerCase().replace(/[\s_]+/g, "");
  if (t.includes("bossdb")) return "bossdb_01";
  if (t.includes("openorganelle")) return "openorganelle_01";
  if (/_\d{2}$/i.test(raw) && raw.length > 3) return raw;
  return "";
}

function pipelineSiteStemToDownloaderSite(stem: string): string {
  const t = (stem || "").trim().toLowerCase().replace(/[\s_]+/g, "");
  if (t.includes("bossdb")) return "bossdb";
  if (t.includes("openorganelle")) return "openorganelle";
  return "";
}

function pipelineSiteStemIsLocalHpc(stem: string): boolean {
  const t = (stem || "").trim().toLowerCase().replace(/[\s_]+/g, "");
  return t.includes("localhpc") || t.includes("mitole") || t.includes("localmito");
}

function studioPageFromPipelineStepLabel(stepLabel: string): StudioPageId | null {
  switch ((stepLabel || "").trim()) {
    case "scrape":
      return "scraper";
    case "database_build":
      return "database";
    case "download_script":
      return "downloader";
    case "preprocess":
      return "processor";
    case "model_training":
      return "training";
    case "eval":
      return "summary";
    default:
      return null;
  }
}

/** Prefer live subprocess flags so Studio tracks work before ``/api/pipeline`` advances. */
function resolvePipelineStudioNav(
  stepLabel: string,
  siteStem: string,
  live: {
    dlRunning: boolean;
    preRunning: boolean;
    scrapeRunning: boolean;
    dbRunning: boolean;
  },
): { appView: AppView; page?: StudioPageId; hpcPage?: HpcPageId } | null {
  const l = (stepLabel || "").trim();
  if (!l || l === "idle") return null;
  const isLocal = pipelineSiteStemIsLocalHpc(siteStem);

  if (live.preRunning) {
    return { appView: "data-scrape", page: "processor" };
  }
  if (l === "model_training") {
    return { appView: "model-training", page: "training" };
  }
  if (l === "eval") {
    return { appView: "pipeline-summary", page: "summary" };
  }
  if (live.dlRunning) {
    if (isLocal) return { appView: "data-hpc", hpcPage: "downloader", page: "downloader" };
    return { appView: "data-scrape", page: "downloader" };
  }
  if (l === "scrape" || live.scrapeRunning) {
    return { appView: "data-scrape", page: "scraper" };
  }
  if (l === "database_build" || live.dbRunning) {
    if (isLocal) return { appView: "data-hpc", hpcPage: "catalog" };
    return { appView: "data-scrape", page: "database" };
  }
  if (l === "download_script" || l === "preprocess") {
    if (isLocal) return { appView: "data-hpc", hpcPage: "downloader", page: "downloader" };
    const page = studioPageFromPipelineStepLabel(l);
    if (page) return { appView: "data-scrape", page };
  }
  return null;
}

export function PipelineStudio({ sessionId, onNotify, chatPanelCollapsed = false }: Props) {
  const [page, setPage] = useState<StudioPageId>("intro");
  const [hpcPage, setHpcPage] = useState<HpcPageId>("browser");
  const [appView, setAppView] = useState<AppView>(() =>
    appViewFromHash(typeof window !== "undefined" ? window.location.hash : ""),
  );
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [selectedInferenceRowKeys, setSelectedInferenceRowKeys] = useState<Set<string>>(new Set());

  const [wsList, setWsList] = useState<StudioWebsiteSummary[]>([]);
  const [wsListLoading, setWsListLoading] = useState(false);
  const [wsPickSlug, setWsPickSlug] = useState("");
  const [wsDisplayName, setWsDisplayName] = useState("");
  const [wsUrl, setWsUrl] = useState("");
  const [wsDescription, setWsDescription] = useState("");
  const [wsDataFocus, setWsDataFocus] = useState(
    "Any datasets, download pages, APIs, or catalogs linked from this site. Organelle or mitochondria relevance is decided later using site.md and outputs/*.probe.json.",
  );
  const [wsSlugOverride, setWsSlugOverride] = useState("");
  const [wsSaveBusy, setWsSaveBusy] = useState(false);
  /** When true and a site is loaded, Save overwrites that folder; otherwise Save allocates the next ``_NN`` folder. */
  const [wsSaveOverwrite, setWsSaveOverwrite] = useState(false);
  const [wsScrapeBusy, setWsScrapeBusy] = useState(false);
  const [wsDeleteBusy, setWsDeleteBusy] = useState(false);
  /** Two-step delete (embedded browsers often block ``window.confirm``). */
  const [deletePanelArmed, setDeletePanelArmed] = useState(false);
  const [deletePickSlug, setDeletePickSlug] = useState("");
  const [wsLast, setWsLast] = useState<StudioWebsiteScrapeResult | null>(null);
  const [wsScrapeLog, setWsScrapeLog] = useState("");
  /** Authoritative runtime state from ``GET /run/website-scrape-state``. */
  const [wsScrapeRemoteRunning, setWsScrapeRemoteRunning] = useState(false);
  const scrapeLogRef = useRef<HTMLPreElement>(null);
  const databaseBuildLogRef = useRef<HTMLPreElement>(null);
  const preprocessLogRef = useRef<HTMLPreElement>(null);
  const wsScrapeAbortRef = useRef<AbortController | null>(null);
  /** Slug selected for part 2 (scrape from saved ``site.md``). */
  const [scrapeTargetSlug, setScrapeTargetSlug] = useState("");

  const wsListDeletable = useMemo(
    () => wsList.filter((w) => !studioIsProtectedWebsiteSlug(w.slug)),
    [wsList],
  );

  const [probes, setProbes] = useState<string[]>([]);
  const [probesLoading, setProbesLoading] = useState(false);
  const [probeChoice, setProbeChoice] = useState("");
  const [databaseBuildBusy, setDatabaseBuildBusy] = useState(false);
  const [databaseBuildResult, setDatabaseBuildResult] = useState<StudioRunResult | null>(null);
  const [databaseClearBusy, setDatabaseClearBusy] = useState(false);
  /** Live DB build log from ``GET /run/database-state`` (chat/agent runs while user is on another tab). */
  const [databaseRemoteLog, setDatabaseRemoteLog] = useState("");
  /** Authoritative runtime state from ``GET /run/database-state``. */
  const [databaseRemoteRunning, setDatabaseRemoteRunning] = useState(false);

  const [sites, setSites] = useState<string[]>([]);
  const [sitesLoading, setSitesLoading] = useState(false);
  const [siteChoice, setSiteChoice] = useState("");
  const [dlDatasetSplits, setDlDatasetSplits] = useState<Record<string, DownloaderDatasetSplit>>({});
  const [dlBusy, setDlBusy] = useState(false);
  const [dlGenerateBusy, setDlGenerateBusy] = useState(false);
  const [dlResult, setDlResult] = useState<StudioRunResult | null>(null);
  const [dlRunLog, setDlRunLog] = useState("");
  const [dlRunProgress, setDlRunProgress] = useState<{ completed: number; total: number; active: number; dataset: string } | null>(null);
  const [dlGeneratedScripts, setDlGeneratedScripts] = useState<string[]>([]);
  const [dlScriptsLoading, setDlScriptsLoading] = useState(false);
  const [, setDlScriptChoice] = useState("");
  const [pendingDl, setPendingDl] = useState<StudioPendingDownloads | null>(null);
  const [pendingDlBusy, setPendingDlBusy] = useState(false);
  const [dlPreviewBusy, setDlPreviewBusy] = useState(false);
  const [dlPreview, setDlPreview] = useState<{
    ok: boolean;
    message: string;
    site: string;
    data_scope: string;
    db_path: string;
    count: number;
    datasets: string[];
    dataset_rows?: Array<{ dataset_name: string; sample_type: string }>;
  } | null>(null);
  const [dlSplitsBySite, setDlSplitsBySite] = useState<Record<string, Record<string, DownloaderDatasetSplit>>>({});
  const [dlSampleTypeSelectionsBySite, setDlSampleTypeSelectionsBySite] = useState<Record<string, string[]>>({});
  const [dlSampleTypesSelected, setDlSampleTypesSelected] = useState<string[]>([]);

  const [preSelectiveBusy, setPreSelectiveBusy] = useState(false);
  /** Authoritative runtime state from ``GET /run/preprocess-selective-state``. */
  const [preRemoteRunning, setPreRemoteRunning] = useState(false);
  /** True while the kill-cancel request is in flight (separate from ``running`` — avoids a no-op feeling). */
  const [preKillBusy, setPreKillBusy] = useState(false);
  const [preRunLog, setPreRunLog] = useState("");
  const [preRunProgress, setPreRunProgress] = useState<{ completed: number; total: number; active: number; dataset: string } | null>(null);
  /** Basename under data/raw (e.g. openorganelle_mito_…); whole-run preprocess uses every ``images/*_im.h5`` in the run. */
  const [preprocessDownloadRun, setPreprocessDownloadRun] = useState<string | null>(null);
  /** Authoritative ``*_im.h5`` count from ``GET /data/raw-em-stacks`` (null = not loaded or fetch failed). */
  const [serverEmH5Count, setServerEmH5Count] = useState<number | null>(null);
  const [preprocessSplitLabelCc, setPreprocessSplitLabelCc] = useState(true);
  const [rawViewerSelection, setRawViewerSelection] = useState<string | null>(null);
  const rawViewerSourceRoot = REL_DATA_RAW;
  const [preResult, setPreResult] = useState<StudioRunResult | null>(null);
  const [dataInspectLoading, setDataInspectLoading] = useState(false);
  const [dataInspect, setDataInspect] = useState<{
    raw_base: string;
    training_base?: string;
    inference_base?: string;
    preprocessed_base: string;
    inspect_shallow?: boolean;
    inspect_deep_under?: string | null;
    raw_datasets: {
      name: string;
      path: string;
      type: string;
      dimensions: number[];
      spacing: number[];
      label_summary?: string;
    }[];
    preprocessed_datasets: {
      name: string;
      path: string;
      type: string;
      dimensions: number[];
      spacing: number[];
      label_summary?: string;
    }[];
    training_datasets?: {
      name: string;
      path: string;
      type: string;
      dimensions: number[];
      spacing: number[];
      label_summary?: string;
    }[];
    inference_datasets?: {
      name: string;
      path: string;
      type: string;
      dimensions: number[];
      spacing: number[];
      label_summary?: string;
    }[];
    instance_datasets?: {
      name: string;
      path: string;
      type: string;
      dimensions: number[];
      spacing: number[];
      label_summary?: string;
    }[];
  } | null>(null);

  const [trainingBusy, setTrainingBusy] = useState(false);
  const [trainingResult, setTrainingResult] = useState<StudioRunResult | null>(null);
  const [trainingState, setTrainingState] = useState<StudioSlurmRunState | null>(null);
  const [trainingLogTab, setTrainingLogTab] = useState<"out" | "err">("out");
  const [trainingClearBusy, setTrainingClearBusy] = useState(false);
  const [trainingSelectedLogRoot, setTrainingSelectedLogRoot] = useState("");
  const [inferenceBusy, setInferenceBusy] = useState(false);
  const [inferenceResult, setInferenceResult] = useState<StudioRunResult | null>(null);
  const [inferenceState, setInferenceState] = useState<StudioSlurmRunState | null>(null);
  const [inferenceLogTab, setInferenceLogTab] = useState<"out" | "err">("out");
  const [inferenceClearBusy, setInferenceClearBusy] = useState(false);
  const [inferenceSelectedLogRoot, setInferenceSelectedLogRoot] = useState("");
  const FIXED_POSTPROCESS_INPUT_DIR = REL_OUTPUTS_BC;
  const FIXED_POSTPROCESS_OUTPUT_DIR = REL_OUTPUTS_POSTPROCESSED;
  const [postprocessFilesLoading, setPostprocessFilesLoading] = useState(false);
  const [postprocessFiles, setPostprocessFiles] = useState<Array<DatasetInspectItem & { source: "input" | "output" }>>([]);
  const [postprocessSelectedRowKeys, setPostprocessSelectedRowKeys] = useState<Set<string>>(new Set());
  const [postprocessBusy, setPostprocessBusy] = useState(false);
  const [postprocessResult, setPostprocessResult] = useState<StudioRunResult | null>(null);
  const FIXED_EVAL_PRED_DIR = REL_OUTPUTS_POSTPROCESSED;
  const FIXED_EVAL_GT_DIR = REL_NNUNET_LABELS_TS_INSTANCE;
  const [evalBusy, setEvalBusy] = useState(false);
  const [evalResult, setEvalResult] = useState<StudioRunResult | null>(null);

  const [summary, setSummary] = useState<Awaited<ReturnType<typeof getStudioSummary>> | null>(null);
  const [summaryBusy, setSummaryBusy] = useState(false);

  // Existing Data From HPC Cluster — dataset status, delete/hide controls.
  const [dsStatus, setDsStatus] = useState<DatasetStatusItem[] | null>(null);
  const [dsStatusLoading, setDsStatusLoading] = useState(false);
  const [dsActionBusy, setDsActionBusy] = useState(false);
  const [dsSelectedRowKeys, setDsSelectedRowKeys] = useState<Set<string>>(new Set());

  // Scrape pipeline Stage 0 — Inventory.
  const [catalogue, setCatalogue] = useState<InventoryCatalogueResponse | null>(null);
  const [catalogueLoading, setCatalogueLoading] = useState(false);
  const [catFilterStatus, setCatFilterStatus] = useState("");
  const [catFilterBatch, setCatFilterBatch] = useState("");
  const [catShowMissingOnly, setCatShowMissingOnly] = useState(false);
  const [catSortCol, setCatSortCol] = useState<keyof CatalogueRow>("batch_id");
  const [catSortAsc, setCatSortAsc] = useState(true);
  const [resetDownloadedBusy, setResetDownloadedBusy] = useState(false);
  const [modelResetBusy, setModelResetBusy] = useState(false);

  const [existingSourceFilter, setExistingSourceFilter] = useState<"all" | "training" | "inference" | "instance">("all");
  const [predictedSourceFilter, setPredictedSourceFilter] = useState<"all" | "input" | "output">("all");
  const [existingDataTab, setExistingDataTab] = useState<"training" | "predicted">("training");
  const [mitoleBasePath, setMitoleBasePath] = useState(DEFAULT_MITOLE_BASE_PATH);
  const [mitoleFolders, setMitoleFolders] = useState<string[]>([]);
  const [mitoleFolderPick, setMitoleFolderPick] = useState("__all__");
  const [mitoleAllSubfolders, setMitoleAllSubfolders] = useState<string[]>([]);
  const [mitoleSelectedSet, setMitoleSelectedSet] = useState<Set<string>>(new Set());
  const [mitoleInspectRows, setMitoleInspectRows] = useState<MitoLeInspectRow[]>([]);
  const [mitoleCatalogueRows, setMitoleCatalogueRows] = useState<MitoLeCatalogueRow[]>([]);
  const [mitoleLoading, setMitoleLoading] = useState(false);
  const [mitoleStage1BootLoading, setMitoleStage1BootLoading] = useState(false);
  const [mitoleCatalogueLoading, setMitoleCatalogueLoading] = useState(false);
  const [mitoleCatQuery, setMitoleCatQuery] = useState("");
  // Empty set means "All". This avoids recomputing defaults when options change.
  const [mitoleCatSelectedSources, setMitoleCatSelectedSources] = useState<Set<string>>(new Set());
  const [mitoleCatSelectedOrganisms, setMitoleCatSelectedOrganisms] = useState<Set<string>>(new Set());
  const [mitoleCatSelectedSampleTypes, setMitoleCatSelectedSampleTypes] = useState<Set<string>>(new Set());
  const [mitoleCatSortCol, setMitoleCatSortCol] = useState<"dataset" | "source" | "organism" | "sample_type">("dataset");
  const [mitoleCatSortAsc, setMitoleCatSortAsc] = useState(true);
  const [mitoleCatSelectedPairKey, setMitoleCatSelectedPairKey] = useState<string | null>(null);
  const [mitoleCatalogueGenerated, setMitoleCatalogueGenerated] = useState(false);
  const [hpcDlSampleTypesSelected, setHpcDlSampleTypesSelected] = useState<string[]>([]);

  const dlLastNotifiedDoneAtRef = useRef<number>(0);
  const dlPrevRunningRef = useRef<boolean>(false);
  const dlGenerateInFlightRef = useRef<boolean>(false);
  const preLastNotifiedDoneAtRef = useRef<number>(0);
  const lastAutoNavPageRef = useRef<string>("");
  const prePrevRunningRef = useRef<boolean>(false);
  const preprocessPrefsHydratedRef = useRef(false);
  const preprocessCanPersistRef = useRef(false);
  const downloaderPrefsHydratedRef = useRef(false);
  const downloaderCanPersistRef = useRef(false);
  const onNotifyRef = useRef(onNotify);
  /** After chat/agent hits Stage 1, keep polling scrape-state until tail is flushed (archived log). */
  const scrapeAgentTailRef = useRef(false);
  const scrapeAgentTailClearTimerRef = useRef<number | null>(null);
  const databaseAgentTailRef = useRef(false);
  const databaseAgentTailClearTimerRef = useRef<number | null>(null);
  const preAgentTailRef = useRef(false);
  const preAgentTailClearTimerRef = useRef<number | null>(null);

  useEffect(() => {
    onNotifyRef.current = onNotify;
  }, [onNotify]);

  useEffect(() => {
    if (wsScrapeBusy) {
      scrapeAgentTailRef.current = false;
      if (scrapeAgentTailClearTimerRef.current != null) {
        window.clearTimeout(scrapeAgentTailClearTimerRef.current);
        scrapeAgentTailClearTimerRef.current = null;
      }
    }
  }, [wsScrapeBusy]);

  useEffect(() => {
    if (databaseBuildBusy) {
      databaseAgentTailRef.current = false;
      if (databaseAgentTailClearTimerRef.current != null) {
        window.clearTimeout(databaseAgentTailClearTimerRef.current);
        databaseAgentTailClearTimerRef.current = null;
      }
    }
  }, [databaseBuildBusy]);

  const [remotePipelineMirror, setRemotePipelineMirror] = useState<RemotePipelineMirror>({
    scrape: false,
    database: false,
    download: false,
    preprocess: false,
  });

  useEffect(() => {
    if (sessionId === "chat_bootstrap") {
      setRemotePipelineMirror({ scrape: false, database: false, download: false, preprocess: false });
      return;
    }
    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      try {
        const p = await getPipelineProgress(sessionId);
        if (cancelled) return;
        const mirror = pipelineStepLabelToMirror(p.step_label);
        const labelForNav = (p.step_label || "").trim();
        const nav = resolvePipelineStudioNav(labelForNav, p.last_site_stem || "", {
          dlRunning: false,
          preRunning: false,
          scrapeRunning: false,
          dbRunning: false,
        });
        if (nav) {
          const navKey = `${nav.appView}:${nav.page || ""}:${nav.hpcPage || ""}`;
          const stageChanged = navKey !== lastAutoNavPageRef.current;
          const viewMismatch =
            appView !== nav.appView ||
            (nav.page ? page !== nav.page : false) ||
            (nav.hpcPage ? hpcPage !== nav.hpcPage : false);
          if (stageChanged || viewMismatch) {
            setAppView((prev) => (prev === nav.appView ? prev : nav.appView));
            if (nav.page) {
              const nextPage = nav.page;
              setPage((prev) => (prev === nextPage ? prev : nextPage));
            }
            if (nav.hpcPage) {
              const nextHpcPage = nav.hpcPage;
              setHpcPage((prev) => (prev === nextHpcPage ? prev : nextHpcPage));
            }
            lastAutoNavPageRef.current = navKey;
          }
        }
        if (cancelled) return;
        const pipelineBusyRemote =
          mirror.scrape || mirror.database || mirror.download || mirror.preprocess;
        if (pipelineBusyRemote) {
          const stem = (p.last_site_stem || "").trim();
          if (stem) {
            const slug = pipelineSiteStemToScrapeSlug(stem);
            if (slug) setScrapeTargetSlug((prev) => (prev === slug ? prev : slug));
            const dl = pipelineSiteStemToDownloaderSite(stem);
            // Only auto-sync downloader site while Stage-3 downloader is the
            // active remote step; avoid stomping a user's manual selection.
            if (dl && mirror.download && !pipelineSiteStemIsLocalHpc(stem) && appView === "data-scrape" && page === "downloader") {
              setSiteChoice((prev) => (prev && prev.trim() ? prev : dl));
            }
          }
        }
        if (mirror.scrape) scrapeAgentTailRef.current = true;
        if (mirror.database) databaseAgentTailRef.current = true;
        if (mirror.preprocess) preAgentTailRef.current = true;
        setRemotePipelineMirror(mirror);
        if (!wsScrapeBusy && scrapeAgentTailRef.current) {
          try {
            const ss = await getStudioWebsiteScrapeState(sessionId);
            if (cancelled) return;
            setWsScrapeRemoteRunning(Boolean(ss.running));
            if (typeof ss.log === "string") setWsScrapeLog(ss.log);
            if (!ss.running && mirror.scrape) {
              // ``/api/pipeline`` can lag on the completed stage label; clear stale scrape mirror.
              setRemotePipelineMirror((prev) => (
                prev.scrape ? { ...prev, scrape: false } : prev
              ));
            }
            if (!mirror.scrape && !ss.running) {
              if (scrapeAgentTailClearTimerRef.current != null) {
                window.clearTimeout(scrapeAgentTailClearTimerRef.current);
              }
              scrapeAgentTailClearTimerRef.current = window.setTimeout(() => {
                scrapeAgentTailRef.current = false;
                scrapeAgentTailClearTimerRef.current = null;
              }, 4000);
            }
          } catch {
            // Keep prior state on transient polling errors to avoid stepper flicker.
          }
        }
        const databaseUiBusyLocal = databaseBuildBusy || mirror.database;
        if (databaseUiBusyLocal || databaseAgentTailRef.current) {
          try {
            const ds = await getStudioDatabaseBuildState(sessionId);
            if (cancelled) return;
            setDatabaseRemoteRunning(Boolean(ds.running));
            if (typeof ds.log === "string") setDatabaseRemoteLog(ds.log);
            if (ds.result) setDatabaseBuildResult(ds.result);
            if (!ds.running && mirror.database) {
              // ``/api/pipeline`` can reflect the last completed stage label
              // even after stage-2 exits; clear stale "database busy" mirror.
              setRemotePipelineMirror((prev) => (
                prev.database ? { ...prev, database: false } : prev
              ));
            }
            if (!databaseUiBusyLocal && !ds.running) {
              if (databaseAgentTailClearTimerRef.current != null) {
                window.clearTimeout(databaseAgentTailClearTimerRef.current);
              }
              databaseAgentTailClearTimerRef.current = window.setTimeout(() => {
                databaseAgentTailRef.current = false;
                databaseAgentTailClearTimerRef.current = null;
              }, 4000);
            }
          } catch {
            // Keep prior state on transient polling errors to avoid stepper flicker.
          }
        }
        // Downloader has its own dedicated poller effect (below). Keeping it out of the
        // pipeline-mirror effect avoids duplicate state writers racing on log/progress UI.
        const preUiBusyLocal = preSelectiveBusy || mirror.preprocess;
        if (preUiBusyLocal || preAgentTailRef.current) {
          try {
            const s = await getStudioRunPreprocessSelectiveState(sessionId);
            if (cancelled) return;
            setPreSelectiveBusy(Boolean(s.running));
            setPreRemoteRunning(Boolean(s.running));
            if (typeof s.log === "string") setPreRunLog(s.log);
            if (s.progress) {
              setPreRunProgress({
                completed: s.progress.completed,
                total: s.progress.total,
                active: s.progress.current,
                dataset: s.progress.dataset,
              });
            } else if (!s.running) {
              setPreRunProgress(null);
            }
            if (s.result) setPreResult(s.result);
            if (!s.running && mirror.preprocess) {
              // ``/api/pipeline`` can lag on the completed stage label; clear stale preprocess mirror.
              setRemotePipelineMirror((prev) => (
                prev.preprocess ? { ...prev, preprocess: false } : prev
              ));
            }
            const prevPre = prePrevRunningRef.current;
            const msgL = String(s.result?.message ?? "").toLowerCase();
            const killedStop =
              msgL.includes("stopped (killed)") ||
              msgL.includes("kill requested") ||
              Number(s.result?.returncode) === 130;
            if (
              prevPre &&
              !s.running &&
              s.result &&
              s.updated_at > preLastNotifiedDoneAtRef.current &&
              !killedStop
            ) {
              preLastNotifiedDoneAtRef.current = s.updated_at;
              onNotifyRef.current(
                "Selective preprocessor finished",
                s.result.message,
                s.result.ok ? "ok" : "err",
              );
            }
            prePrevRunningRef.current = Boolean(s.running);
            if (!preUiBusyLocal && !s.running) {
              if (preAgentTailClearTimerRef.current != null) {
                window.clearTimeout(preAgentTailClearTimerRef.current);
              }
              preAgentTailClearTimerRef.current = window.setTimeout(() => {
                preAgentTailRef.current = false;
                preAgentTailClearTimerRef.current = null;
              }, 4000);
            }
          } catch {
            // Keep prior state on transient polling errors to avoid stepper flicker.
          }
        }
      } catch {
        // Keep prior mirror on transient ``/api/pipeline`` read failures.
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 1300);
    return () => {
      cancelled = true;
      window.clearInterval(id);
      if (scrapeAgentTailClearTimerRef.current != null) {
        window.clearTimeout(scrapeAgentTailClearTimerRef.current);
        scrapeAgentTailClearTimerRef.current = null;
      }
      if (databaseAgentTailClearTimerRef.current != null) {
        window.clearTimeout(databaseAgentTailClearTimerRef.current);
        databaseAgentTailClearTimerRef.current = null;
      }
      if (preAgentTailClearTimerRef.current != null) {
        window.clearTimeout(preAgentTailClearTimerRef.current);
        preAgentTailClearTimerRef.current = null;
      }
    };
  }, [sessionId, wsScrapeBusy, databaseBuildBusy, preSelectiveBusy, appView, page, hpcPage]);

  const scrapeUiBusy = wsScrapeBusy || wsScrapeRemoteRunning;
  const databaseUiBusy = databaseBuildBusy || databaseRemoteRunning;
  // Downloader button busy state should follow actual stage-3 runtime state,
  // not coarse pipeline step labels (which may remain on "download_script").
  // Keep downloader UI in busy state for the full request lifecycle, even if
  // state polling briefly lags or reports stale idle right after launch.
  const dlUiBusy = dlBusy || dlGenerateBusy;
  const preUiBusy = preSelectiveBusy || preRemoteRunning;
  const trainingUiBusy = trainingBusy || Boolean(trainingState?.running);
  const inferenceUiBusy = inferenceBusy || Boolean(inferenceState?.running);

  const scrapeNavRemoteWorking = (id: StudioPageId): boolean => {
    // Keep stage circles stable while local/remote runners are active, even if
    // ``/api/pipeline`` step_label briefly lags or flips between polls.
    if (id === "scraper") return remotePipelineMirror.scrape || scrapeUiBusy;
    if (id === "database") return remotePipelineMirror.database || databaseUiBusy;
    if (id === "downloader") return remotePipelineMirror.download || dlUiBusy;
    if (id === "processor") return remotePipelineMirror.preprocess || preUiBusy;
    return false;
  };
  const hpcNavRemoteWorking = (id: HpcPageId): boolean => {
    if (id === "browser") return false;
    if (id === "catalog") return remotePipelineMirror.database || mitoleLoading || mitoleStage1BootLoading;
    if (id === "downloader") return remotePipelineMirror.download || dlUiBusy;
    return false;
  };

  useEffect(() => {
    if (typeof window === "undefined") return;
    const h = hashFromAppView(appView);
    if (window.location.hash !== h) window.location.hash = h;
  }, [appView]);

  useEffect(() => {
    const handler = () => setAppView(appViewFromHash(window.location.hash));
    window.addEventListener("popstate", handler);
    return () => window.removeEventListener("popstate", handler);
  }, []);

  useEffect(() => {
    // Keep data-scrape pages on a valid subpage when switching between top-level tabs.
    if (appView !== "data-scrape") return;
    if (page === "inventory" || page === "scraper" || page === "database" || page === "downloader" || page === "processor") return;
    setPage("inventory");
  }, [appView, page]);

  useEffect(() => {
    // Reuse the same Stage 3 downloader state machine when Local HPC enters its downloader step.
    if (appView !== "data-hpc" || hpcPage !== "downloader") return;
    if (page !== "downloader") setPage("downloader");
  }, [appView, hpcPage, page]);

  useEffect(() => {
    const el = scrapeLogRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [wsScrapeLog]);

  useEffect(() => {
    const el = databaseBuildLogRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [databaseRemoteLog, page]);

  useEffect(() => {
    const el = preprocessLogRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [preRunLog, page]);

  const refreshSummary = useCallback(async () => {
    setSummaryBusy(true);
    try {
      setSummary(await getStudioSummary(sessionId));
    } catch {
      setSummary(null);
    } finally {
      setSummaryBusy(false);
    }
  }, [sessionId]);

  useEffect(() => {
    if (appView !== "pipeline-summary") return;
    void refreshSummary();
  }, [appView, refreshSummary]);

  useEffect(() => {
    if (page !== "scraper") return;
    void (async () => {
      setWsListLoading(true);
      try {
        const w = await getStudioWebsites();
        setWsList(w.websites);
      } catch {
        setWsList([]);
      } finally {
        setWsListLoading(false);
      }
    })();
  }, [page]);

  useEffect(() => {
    if (page !== "scraper") return;
    if (wsList.length === 0) {
      setScrapeTargetSlug("");
      setDeletePickSlug("");
      return;
    }
    setScrapeTargetSlug((prev) => {
      if (prev && wsList.some((w) => w.slug === prev)) return prev;
      return "";
    });
    setDeletePickSlug((prev) => {
      if (prev && wsList.some((w) => w.slug === prev)) {
        if (studioIsProtectedWebsiteSlug(prev)) return "";
        return prev;
      }
      return "";
    });
  }, [page, wsList]);

  useEffect(() => {
    setDeletePanelArmed(false);
  }, [deletePickSlug]);

  useEffect(() => {
    if (page !== "database") return;
    void (async () => {
      setProbesLoading(true);
      try {
        const r = await getStudioProbes();
        setProbes(r.probes);
        setProbeChoice((prev) => (prev && r.probes.includes(prev) ? prev : ""));
      } catch {
        setProbes([]);
      } finally {
        setProbesLoading(false);
      }
    })();
  }, [page]);

  useEffect(() => {
    const inDownloaderStage =
      (appView === "data-scrape" && page === "downloader") ||
      (appView === "data-hpc" && hpcPage === "downloader");
    if (!inDownloaderStage) return;
    void (async () => {
      setSitesLoading(true);
      try {
        const r = await getStudioSites();
        setSites(r.sites);
        setSiteChoice((prev) => {
          const prevNorm = String(prev || "").trim().toLowerCase();
          if (!prevNorm) return "";
          if (prevNorm) {
            const matched = r.sites.find((s) => String(s || "").trim().toLowerCase() === prevNorm);
            if (matched) return matched;
          }
          return "";
        });
      } catch {
        setSites([]);
        setSiteChoice("");
      } finally {
        setSitesLoading(false);
      }
    })();
  }, [page, appView]);

  const loadWebsiteProfile = async (slug: string) => {
    if (!slug) return;
    try {
      const w = await getStudioWebsite(slug);
      setWsDisplayName(w.display_name);
      setWsUrl(w.url);
      setWsDescription(w.description);
      setWsDataFocus(
        w.data_focus?.trim()
          ? w.data_focus
          : "Any datasets, download pages, APIs, or catalogs linked from this site. Organelle or mitochondria relevance is decided later using site.md and outputs/*.probe.json.",
      );
      setWsSlugOverride(w.slug);
      setWsPickSlug(slug);
    } catch (e) {
      onNotify("Load failed", e instanceof Error ? e.message : String(e), "err");
    }
  };

  const runWorkspaceSave = async () => {
    const u = wsUrl.trim();
    if (!u || u === "http://" || u === "https://") {
      onNotify("URL required", "Enter a landing page URL before saving.", "err");
      return;
    }
    setWsSaveBusy(true);
    try {
      const out = await postStudioWebsiteSave(sessionId, {
        display_name: wsDisplayName.trim(),
        url: wsUrl.trim(),
        description: wsDescription,
        data_focus: wsDataFocus,
        slug: wsSlugOverride.trim() || undefined,
        editing_slug: wsSaveOverwrite && wsPickSlug.trim() ? wsPickSlug.trim() : undefined,
      });
      onNotify("Saved website profile", `${out.site_md}`, "ok");
      const refreshed = (await getStudioWebsites()).websites;
      setWsList(refreshed);
      setWsSlugOverride(out.slug);
      setWsPickSlug(out.slug);
      setScrapeTargetSlug(out.slug);
    } catch (e) {
      onNotify("Save failed", e instanceof Error ? e.message : String(e), "err");
    } finally {
      setWsSaveBusy(false);
    }
  };

  const executeWorkspaceDelete = async (rawSlug: string) => {
    const s = rawSlug.trim();
    if (!s) {
      onNotify("Pick a site", "Choose which workspace folder to remove.", "err");
      return;
    }
    if (studioIsProtectedWebsiteSlug(s)) {
      onNotify(
        "Cannot delete",
        "Built-in workspaces bossdb_01 and openorganelle_01 cannot be removed from Studio. Edit their files in the repository instead.",
        "err",
      );
      setDeletePanelArmed(false);
      return;
    }
    setWsDeleteBusy(true);
    try {
      const data = await deleteStudioWebsite(s);
      const removed = data.slug || s;
      const pathLine = data.folder ?? `1web_scraper_01/websites/${removed}/`;
      const probeNote = data.removed_probe ? " Probe JSON removed." : "";
      onNotify("Deleted", `${pathLine}.${probeNote}`, "ok");
      const refreshed = (await getStudioWebsites()).websites;
      setWsList(refreshed);
      const matchesRemoved = (x: string) => x === s || x === removed;
      if (matchesRemoved(wsPickSlug)) {
        setWsPickSlug("");
      }
      if (matchesRemoved(scrapeTargetSlug)) {
        setScrapeTargetSlug("");
      }
      if (matchesRemoved(deletePickSlug)) {
        setDeletePickSlug("");
      }
      if (matchesRemoved(wsSlugOverride)) {
        setWsSlugOverride("");
      }
      setDeletePanelArmed(false);
    } catch (e) {
      onNotify("Delete failed", e instanceof Error ? e.message : String(e), "err");
    } finally {
      setWsDeleteBusy(false);
    }
  };

  const runWorkspaceDelete = async () => {
    const s = deletePickSlug.trim();
    if (!s) {
      onNotify("Pick a site", "Choose which workspace folder to remove.", "err");
      return;
    }
    if (studioIsProtectedWebsiteSlug(s)) {
      onNotify(
        "Cannot delete",
        "Built-in workspaces bossdb_01 and openorganelle_01 cannot be removed from Studio.",
        "err",
      );
      setDeletePanelArmed(false);
      return;
    }
    if (!deletePanelArmed) {
      setDeletePanelArmed(true);
      return;
    }
    setDeletePanelArmed(false);
    await executeWorkspaceDelete(s);
  };

  const killWorkspaceScrape = () => {
    void postChatStop(sessionId).catch(() => {});
    void postStudioScrapeCancel(sessionId).catch(() => {});
    setWsScrapeLog((prev) => `${prev}\n[mito2] Stop requested — cancelling scrape…\n`);
    wsScrapeAbortRef.current?.abort();
  };

  const runWorkspaceScrape = async () => {
    const slug = scrapeTargetSlug.trim();
    if (!slug) {
      onNotify("Pick a website", "Save a site profile first (part 1), then choose it here.", "err");
      return;
    }
    const ac = new AbortController();
    wsScrapeAbortRef.current = ac;
    setWsScrapeBusy(true);
    setWsLast(null);
    setWsScrapeLog("");
    try {
      await postStudioWebsiteScrapeStream(
        sessionId,
        { slug },
        {
          onLog: (text) => setWsScrapeLog((prev) => prev + text),
          onComplete: (r) => {
            setWsLast(r);
            setWsSlugOverride(r.slug);
            setWsPickSlug(r.slug);
            setScrapeTargetSlug(r.slug);
            void getStudioWebsites().then((w) => setWsList(w.websites));
            if (r.cancelled) {
              onNotify("Scrape stopped", "The scrape subprocess was cancelled.", "err");
              return;
            }
            const fe = r.fetch as { error?: string } | undefined;
            const bridge = r.mito_foundation_bridge as { message?: string; stderr?: string } | undefined;
            const errTail =
              !r.ok && bridge?.stderr
                ? ` ${String(bridge.stderr).slice(-500)}`
                : !r.ok && bridge?.message
                  ? ` ${bridge.message}`
                  : !r.ok && fe?.error
                    ? ` ${fe.error}`
                    : "";
            const wrote = [r.site_md, r.probe_path].filter(Boolean).join(", ");
            notifyJobOutcome("Website scrape", r.ok, `${r.folder}${errTail}`.trim() || wrote || r.folder);
          },
          onError: (msg) => {
            setWsScrapeLog((prev) => `${prev}\n[mito2] ${msg}\n`);
            notifyJobOutcome("Website scrape", false, msg);
          },
        },
        { signal: ac.signal },
      );
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        return;
      }
      notifyJobOutcome("Website scrape", false, e instanceof Error ? e.message : String(e));
    } finally {
      wsScrapeAbortRef.current = null;
      setWsScrapeBusy(false);
    }
  };

  const runDatabaseBuild = async () => {
    setDatabaseBuildBusy(true);
    setDatabaseBuildResult(null);
    setDatabaseRemoteLog("");
    try {
      const r = await postStudioDatabaseBuild(sessionId, probeChoice);
      setDatabaseBuildResult(r);
      notifyJobOutcome("Database builder", r.ok, r.message);
    } catch (e) {
      notifyJobOutcome("Database builder", false, e instanceof Error ? e.message : String(e));
    } finally {
      setDatabaseBuildBusy(false);
    }
  };

  function notifyDesktopPopup(title: string, message: string) {
    if (typeof window === "undefined" || !("Notification" in window)) return;
    if (Notification.permission !== "granted") return;
    new Notification(title, { body: message });
  }

  function notifyJobOutcome(jobLabel: string, ok: boolean, message: string) {
    const title = `${jobLabel} ${ok ? "finished" : "failed"}`;
    onNotifyRef.current(title, message, ok ? "ok" : "err");
    notifyDesktopPopup(title, message);
  }

  const runDownloaderGenerate = async () => {
    if (dlGenerateInFlightRef.current) return;
    const isHpcLocalDownloader = appView === "data-hpc" && hpcPage === "downloader";
    if (!isHpcLocalDownloader && !siteChoice.trim()) {
      onNotify("Site required", "Pick a site stem from probes or type one.", "err");
      return;
    }
    dlGenerateInFlightRef.current = true;
    setDlGenerateBusy(true);
    // Stage 3 is now one-click download (generate+execute).
    setDlBusy(true);
    dlPrevRunningRef.current = false;
    // New run must not inherit prior run UI state.
    setDlRunProgress(null);
    setDlRunLog("");
    setDlResult(null);
    try {
      const datasetSplits: Record<string, DownloaderDatasetSplit> = {};
      const allowed = new Set(activeDlFilteredPreviewDatasets);
      const datasetUniverse = isHpcLocalDownloader
        ? Array.from(new Set(mitolePairCatalogueRows.map((r) => r.dataset)))
        : (dlPreview?.datasets ?? []);
      for (const ds of datasetUniverse) {
        if (!allowed.has(ds)) {
          datasetSplits[ds] = { training: 0, inference: 0 };
          continue;
        }
        const split = dlDatasetSplits[ds] ?? { training: 1, inference: 0 };
        const training = clampDownloaderSplit("training", split.training, split.inference);
        const inference = clampDownloaderSplit("inference", split.inference, training);
        datasetSplits[ds] = { training, inference };
      }
      const selectedCount = Object.values(datasetSplits).filter((s) => (s.training + s.inference) > 0).length;
      if (datasetUniverse.length > 0 && selectedCount === 0) {
        onNotify("No datasets selected", "Set at least one dataset to download at least 1 crop.", "err");
        return;
      }
      let r: StudioRunResult;
      if (isHpcLocalDownloader) {
        const pairByDataset = new Map<string, { dataset: string; source: string; image_path: string; label_path: string }>();
        for (const row of mitolePairCatalogueRows) {
          if (!pairByDataset.has(row.dataset)) {
            pairByDataset.set(row.dataset, {
              dataset: row.dataset,
              source: row.source,
              image_path: "",
              label_path: "",
            });
          }
        }
        const selectedDatasets = Object.entries(datasetSplits)
          .filter(([, s]) => (s.training + s.inference) > 0)
          .map(([ds]) => ds);
        const pairs = selectedDatasets
          .map((ds) => pairByDataset.get(ds))
          .filter((p): p is { dataset: string; source: string; image_path: string; label_path: string } => Boolean(p));
        r = await postStudioMitoleDownloader(sessionId, {
          dataset_splits: datasetSplits,
          dataset_pairs: pairs,
        });
      } else {
        const nCrops = Math.max(
          1,
          ...Object.values(datasetSplits).map((s) => Math.max(0, s.training) + Math.max(0, s.inference)),
        );
        r = await postStudioDownloader(sessionId, {
          site: siteChoice.trim(),
          n_crops: nCrops,
          voxel_size_nm: STUDIO_DOWNLOADER_FIXED_VOXEL_NM,
          crop_dimensions_voxels: STUDIO_DOWNLOADER_FIXED_CROP_VOXELS,
          dataset_splits: datasetSplits,
          data_scope: "labeled",
          execute: true,
        });
      }
      setDlResult(r);
      // Local HPC Stage 3 runs in-process on the POST worker: prefer log/progress embedded in the
      // response (reliable), then fall back to the shared downloader-state GET used by polling.
      if (isHpcLocalDownloader) {
        const rExt = r as StudioRunResult & {
          downloader_log?: string;
          downloaderLog?: string;
          downloader_progress?: { completed: number; total: number; current: number; dataset?: string };
          downloaderProgress?: { completed: number; total: number; current: number; dataset?: string };
        };
        const bodyLog = String(rExt.downloader_log ?? rExt.downloaderLog ?? "").trim();
        if (bodyLog.length > 0) {
          setDlRunLog(bodyLog);
        }
        const pg = rExt.downloader_progress ?? rExt.downloaderProgress;
        const settledOkPost = Boolean(r.ok);
        if (pg && Number(pg.total) > 0) {
          setDlRunProgress({
            completed: settledOkPost ? Number(pg.total) : Number(pg.completed) || 0,
            total: Number(pg.total) || 0,
            active: settledOkPost ? Number(pg.total) : Number(pg.current) || 0,
            dataset: String(pg.dataset ?? ""),
          });
        } else if (bodyLog.length > 0) {
          const parsed = parseDownloaderProgress(bodyLog, settledOkPost, undefined);
          if (parsed && parsed.total > 0) {
            setDlRunProgress({
              completed: settledOkPost ? parsed.total : parsed.completed,
              total: parsed.total,
              active: settledOkPost ? parsed.total : parsed.active,
              dataset: "",
            });
          }
        }
        if (bodyLog.length === 0 || !(pg && Number(pg.total) > 0)) {
          // Some reverse proxies can briefly return an empty POST payload body for large responses.
          // Retry shared state a few times so finished Local-HPC runs never stay on placeholder UI.
          for (let i = 0; i < 4; i += 1) {
            try {
              const s = await getStudioRunDownloaderScriptState(sessionId);
              const hasLog = typeof s.log === "string" && s.log.length > 0;
              const hasProgress = Boolean(s.progress) && Number(s.progress?.total) > 0;
              if (hasLog) setDlRunLog(String(s.log));
              if (hasProgress) {
                const settledOk = !s.running && Boolean(s.result?.ok);
                setDlRunProgress({
                  completed: settledOk ? Number(s.progress?.total) || 0 : Number(s.progress?.completed) || 0,
                  total: Number(s.progress?.total) || 0,
                  active: settledOk ? Number(s.progress?.total) || 0 : Number(s.progress?.current) || 0,
                  dataset: String(s.progress?.dataset ?? ""),
                });
              } else if (hasLog) {
                const parsed = parseDownloaderProgress(String(s.log), !s.running && Boolean(s.result?.ok), undefined);
                if (parsed && parsed.total > 0) {
                  setDlRunProgress({
                    completed: parsed.completed,
                    total: parsed.total,
                    active: parsed.active,
                    dataset: "",
                  });
                }
              }
              if (hasLog || hasProgress) break;
            } catch {
              /* continue retry loop */
            }
            await new Promise<void>((resolve) => window.setTimeout(resolve, 250));
          }
        }
      }
      const rAnyLog = r as StudioRunResult & { downloader_log?: string; downloaderLog?: string };
      const effectiveOk = downloaderResultLooksSuccessful(
        r,
        `${String(rAnyLog.downloader_log ?? rAnyLog.downloaderLog ?? "")}\n${String(r.stdout || "")}\n${String(r.stderr || "")}`,
      );
      notifyJobOutcome("Downloader", effectiveOk, r.message);
      if (!isHpcLocalDownloader && r.ok) {
        const fromRun = r.generated_scripts ?? [];
        if (fromRun.length > 0) {
          setDlGeneratedScripts((prev) => Array.from(new Set([...prev, ...fromRun])));
          setDlScriptChoice((prev) => (fromRun.includes(prev) ? prev : fromRun[0] ?? prev));
        } else {
          // Do not clear the dropdown when this generation produced no script
          // (e.g. provider inventory empty). Refresh from disk and keep previous options.
          try {
            const ls = await getStudioDownloaderScripts("", "labeled");
            const all = ls.scripts ?? [];
            setDlGeneratedScripts((prev) => {
              if (all.length === 0) return prev;
              return Array.from(new Set([...prev, ...all]));
            });
          } catch {
            /* keep current options on refresh failure */
          }
        }
      }
    } catch (e) {
      notifyJobOutcome("Downloader", false, e instanceof Error ? e.message : String(e));
    } finally {
      dlGenerateInFlightRef.current = false;
      setDlGenerateBusy(false);
      setDlBusy(false);
      dlPrevRunningRef.current = false;
      if (isHpcLocalDownloader) return;
      // Always refresh script list from disk, even when generation fails or returns no scripts.
      // This prevents the dropdown from getting stuck empty while files still exist in outputs/.
      void getStudioDownloaderScripts("", "labeled")
        .then((ls) => {
          const all = ls.scripts ?? [];
          if (all.length > 0) {
            setDlGeneratedScripts(all);
            setDlScriptChoice((prev) => (all.includes(prev) ? prev : all[0] ?? prev));
          }
        })
        .catch(() => {
          /* keep existing list on refresh failure */
        });
    }
  };

  const killDownloaderRun = async () => {
    void postChatStop(sessionId).catch(() => {});
    try {
      const r = await postStudioRunDownloaderScriptCancel(sessionId);
      if (r.killed) {
        onNotify("Download killed", "Stopped current downloader process.", "ok");
        return;
      }
      // Avoid false "No active download" when state is stale/lagging.
      const s = await getStudioRunDownloaderScriptState(sessionId);
      if (s.running) {
        onNotify("Kill requested", "Downloader still reports running; retry in a moment.", "err");
      } else {
        onNotify("No active download", "Nothing was running.", "err");
      }
    } catch (e) {
      onNotify("Kill download failed", e instanceof Error ? e.message : String(e), "err");
    }
  };

  const killPreprocessRun = async () => {
    setPreKillBusy(true);
    const applyState = (s: Awaited<ReturnType<typeof getStudioRunPreprocessSelectiveState>>) => {
      setPreSelectiveBusy(Boolean(s.running));
      setPreRemoteRunning(Boolean(s.running));
      if (typeof s.log === "string") {
        setPreRunLog(s.log);
      }
      if (s.progress) {
        setPreRunProgress({
          completed: s.progress.completed,
          total: s.progress.total,
          active: s.progress.current,
          dataset: s.progress.dataset,
        });
      } else if (!s.running) {
        setPreRunProgress(null);
      }
      if (s.result) {
        setPreResult(s.result);
      }
    };
    let r: Awaited<ReturnType<typeof postStudioRunPreprocessSelectiveCancel>>;
    try {
      r = await postStudioRunPreprocessSelectiveCancel(sessionId);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      onNotify("Kill preprocess failed", msg, "err");
      return;
    } finally {
      setPreKillBusy(false);
    }
    if (!r.ok) {
      onNotify(
        "Kill preprocess failed",
        r.error === "cancel_internal_error" ? "Internal server error during cancel." : String(r.error ?? "Unknown error"),
        "err",
      );
      return;
    }
    if (r.killed) {
      onNotify(
        "Preprocess stopped",
        "SIGKILL finished and no matching stage-4 preprocess processes remain for this project.",
        "ok",
      );
    } else if (r.warning) {
      onNotify("Could not stop preprocess", r.warning, "err");
    } else {
      onNotify(
        "No active preprocess",
        "The server reported nothing to stop for this session. Refresh if the UI looks wrong.",
        "err",
      );
    }
    void (async () => {
      try {
        for (let i = 0; i < 30; i++) {
          const s = await getStudioRunPreprocessSelectiveState(sessionId);
          applyState(s);
          if (!s.running) {
            break;
          }
          await new Promise<void>((resolve) => {
            window.setTimeout(resolve, 200);
          });
        }
      } catch {
        /* ignore background refresh errors */
      }
    })();
  };

  const clearScrapeOutput = async () => {
    try {
      const st = await getStudioWebsiteScrapeState(sessionId);
      if (st.running) {
        await postStudioScrapeCancel(sessionId);
        for (let i = 0; i < 20; i += 1) {
          const s2 = await getStudioWebsiteScrapeState(sessionId);
          if (!s2.running) break;
          await new Promise<void>((resolve) => window.setTimeout(resolve, 200));
        }
      }
      await postStudioWebsiteScrapeStateClear(sessionId);
      setWsScrapeLog("");
      setWsLast(null);
      onNotify("Scrape output cleared", "Cleared scrape log and last scrape result.", "ok");
    } catch (e) {
      onNotify("Clear scrape output failed", e instanceof Error ? e.message : String(e), "err");
    }
  };

  const clearDatabaseOutput = async () => {
    if (databaseUiBusy || databaseClearBusy) return;
    setDatabaseClearBusy(true);
    try {
      await postStudioDatabaseBuildStateClear(sessionId);
      setDatabaseRemoteLog("");
      setDatabaseBuildResult(null);
      onNotify("Database output cleared", "Cleared database build log and last result.", "ok");
    } catch (e) {
      onNotify("Clear database output failed", e instanceof Error ? e.message : String(e), "err");
    } finally {
      setDatabaseClearBusy(false);
    }
  };

  const clearDownloaderOutput = async () => {
    try {
      const st = await getStudioRunDownloaderScriptState(sessionId);
      if (st.running) {
        await postStudioRunDownloaderScriptCancel(sessionId);
        for (let i = 0; i < 20; i += 1) {
          const s2 = await getStudioRunDownloaderScriptState(sessionId);
          if (!s2.running) break;
          await new Promise<void>((resolve) => window.setTimeout(resolve, 200));
        }
      }
      await postStudioRunDownloaderScriptStateClear(sessionId);
      setDlRunLog("");
      setDlRunProgress(null);
      setDlResult(null);
      onNotify("Downloader output cleared", "Cleared downloader log/progress/result.", "ok");
    } catch (e) {
      onNotify("Clear downloader output failed", e instanceof Error ? e.message : String(e), "err");
    }
  };

  const clearPreprocessorOutput = async () => {
    try {
      const st = await getStudioRunPreprocessSelectiveState(sessionId);
      if (st.running) {
        await postStudioRunPreprocessSelectiveCancel(sessionId);
        for (let i = 0; i < 30; i += 1) {
          const s2 = await getStudioRunPreprocessSelectiveState(sessionId);
          if (!s2.running) break;
          await new Promise<void>((resolve) => window.setTimeout(resolve, 200));
        }
      }
      await postStudioRunPreprocessSelectiveStateClear(sessionId);
      setPreRunLog("");
      setPreRunProgress(null);
      setPreResult(null);
      onNotify("Preprocessor output cleared", "Cleared preprocess log and last result.", "ok");
    } catch (e) {
      onNotify("Clear preprocessor output failed", e instanceof Error ? e.message : String(e), "err");
    }
  };

  useEffect(() => {
    const inDownloader =
      page === "downloader" && (appView === "data-scrape" || (appView === "data-hpc" && hpcPage === "downloader"));
    if (!inDownloader) return;
    if (!siteChoice.trim()) {
      setDlPreview(null);
      return;
    }
    let cancelled = false;
    setDlPreviewBusy(true);
    void getStudioDownloaderPreview(siteChoice.trim(), "labeled")
      .then((r) => {
        if (cancelled) return;
        setDlPreview(r);
      })
      .catch((e) => {
        if (cancelled) return;
        setDlPreview({
          ok: false,
          message: e instanceof Error ? e.message : String(e),
          site: siteChoice.trim(),
          data_scope: "labeled",
          db_path: "",
          count: 0,
          datasets: [],
        });
      })
      .finally(() => {
        if (cancelled) return;
        setDlPreviewBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [page, siteChoice, appView]);

  useEffect(() => {
    if (appView !== "data-scrape") return;
    const datasets = dlPreview?.datasets ?? [];
    const site = siteChoice.trim();
    if (!site || datasets.length === 0) {
      return;
    }
    setDlDatasetSplits((prev) => {
      const next: Record<string, DownloaderDatasetSplit> = {};
      const remembered = dlSplitsBySite[site] ?? {};
      for (const ds of datasets) {
        const prior = prev[ds] ?? remembered[ds] ?? { training: 1, inference: 0 };
        const training = clampDownloaderSplit("training", prior.training, prior.inference);
        const inference = clampDownloaderSplit("inference", prior.inference, training);
        next[ds] = { training, inference };
      }
      return next;
    });
  }, [dlPreview, dlSplitsBySite, siteChoice, appView]);

  useEffect(() => {
    if (appView !== "data-scrape") return;
    const site = siteChoice.trim();
    if (!site) return;
    setDlSplitsBySite((prev) => ({ ...prev, [site]: { ...dlDatasetSplits } }));
  }, [dlDatasetSplits, siteChoice, appView]);

  const dlSampleTypeByDataset = useMemo<Record<string, string>>(() => {
    const out: Record<string, string> = {};
    for (const row of dlPreview?.dataset_rows ?? []) {
      const name = String(row.dataset_name || "").trim();
      if (!name) continue;
      out[name] = String(row.sample_type || "").trim() || "unknown";
    }
    for (const ds of dlPreview?.datasets ?? []) {
      if (!out[ds]) out[ds] = "unknown";
    }
    return out;
  }, [dlPreview]);

  const dlSampleTypeOptions = useMemo<string[]>(() => {
    const uniq = new Set<string>();
    for (const ds of dlPreview?.datasets ?? []) {
      uniq.add(dlSampleTypeByDataset[ds] || "unknown");
    }
    return Array.from(uniq).sort((a, b) => a.localeCompare(b));
  }, [dlPreview, dlSampleTypeByDataset]);

  useEffect(() => {
    if (appView !== "data-scrape") return;
    const site = siteChoice.trim();
    if (!site) return;
    const all = dlSampleTypeOptions;
    if (all.length === 0) {
      setDlSampleTypesSelected((prev) => (prev.length === 0 ? prev : []));
      return;
    }
    const hasRemembered = Object.prototype.hasOwnProperty.call(dlSampleTypeSelectionsBySite, site);
    const remembered = hasRemembered ? (dlSampleTypeSelectionsBySite[site] ?? []) : [];
    const rememberedValid = remembered.filter((x) => all.includes(x));
    // Default to "all selected" on first visit.
    // Preserve explicit "clear all" (remembered empty array), but if remembered
    // values are stale/invalid after option changes, fall back to all.
    const next = !hasRemembered
      ? all
      : remembered.length === 0
        ? []
        : rememberedValid.length > 0
          ? rememberedValid
          : all;
    setDlSampleTypesSelected((prev) => (sameStringSet(prev, next) ? prev : next));
  }, [siteChoice, dlSampleTypeOptions, dlSampleTypeSelectionsBySite, appView]);

  useEffect(() => {
    if (appView !== "data-scrape") return;
    const site = siteChoice.trim();
    if (!site) return;
    setDlSampleTypeSelectionsBySite((prev) => {
      const cur = prev[site] ?? [];
      if (sameStringSet(cur, dlSampleTypesSelected)) return prev;
      return { ...prev, [site]: [...dlSampleTypesSelected] };
    });
  }, [siteChoice, dlSampleTypesSelected, appView]);

  /** Discover existing download_*.py on disk so "Run generated script" works after reload or CLI generation. */
  useEffect(() => {
    if (page !== "downloader" || appView !== "data-scrape") return;
    let cancelled = false;
    setDlScriptsLoading(true);
    void getStudioDownloaderScripts("", "labeled")
      .then((r) => {
        if (cancelled) return;
        const all = r.scripts ?? [];
        setDlGeneratedScripts(all);
      })
      .catch(() => {
        if (cancelled) return;
      })
      .finally(() => {
        if (!cancelled) setDlScriptsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [page, appView]);

  // Safety net: if list is empty on downloader page, retry from disk so dropdown
  // cannot remain stuck after transient API errors or local state clears.
  useEffect(() => {
    if (page !== "downloader" || appView !== "data-scrape") return;
    if (dlScriptsLoading || dlGeneratedScripts.length > 0) return;
    let cancelled = false;
    void getStudioDownloaderScripts("", "labeled")
      .then((r) => {
        if (cancelled) return;
        const all = r.scripts ?? [];
        if (all.length > 0) {
          setDlGeneratedScripts(all);
          setDlScriptChoice((prev) => (all.includes(prev) ? prev : all[0] ?? prev));
        }
      })
      .catch(() => {
        if (cancelled) return;
      });
    return () => {
      cancelled = true;
    };
  }, [page, appView, dlScriptsLoading, dlGeneratedScripts.length]);

  useEffect(() => {
    setDlScriptChoice((prev) => {
      if (dlGeneratedScripts.length === 0) return "";
      if (prev && dlGeneratedScripts.includes(prev)) return prev;
      return dlGeneratedScripts[0] ?? "";
    });
  }, [dlGeneratedScripts]);

  const dlSucceeded = downloaderResultLooksSuccessful(dlResult, dlRunLog);
  const dlProgressDone = dlSucceeded && !dlUiBusy;
  const rawDlProgress = mergeDownloaderProgress(dlRunProgress, dlRunLog, dlProgressDone);
  const dlProgress =
    rawDlProgress && dlSucceeded && !dlUiBusy && rawDlProgress.total > 0
      ? { ...rawDlProgress, completed: rawDlProgress.total, active: rawDlProgress.total }
      : rawDlProgress;
  const dlProgressPercent = dlProgress && dlProgress.total > 0 ? Math.round((dlProgress.completed / dlProgress.total) * 100) : 0;
  // Avoid stale orange bars during active runs.
  const dlShowFailureStyle = !dlUiBusy && Boolean(dlResult) && !dlSucceeded;
  const dlStateLabel = dlUiBusy ? "running" : dlSucceeded ? "finished" : dlResult ? "failed" : "idle";

  useEffect(() => {
    const pollDownloaderUi =
      page === "downloader" && (appView === "data-scrape" || (appView === "data-hpc" && hpcPage === "downloader"));
    if (!pollDownloaderUi) return;
    // Poll downloader state for every session id so interrupted streams can resume
    // and UI status reflects actual backend process state (web subprocess + Local HPC in-process).
    let cancelled = false;
    let pollingDisabled = false;
    let warnedMissingStateRoute = false;
    let timer: number | null = null;
    const schedule = (ms: number) => {
      if (cancelled || pollingDisabled) return;
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        void tick();
      }, ms);
    };
    const tick = async () => {
      if (pollingDisabled) return;
      try {
        if (typeof document !== "undefined" && document.visibilityState === "hidden") {
          // Back off aggressively for hidden tabs to avoid noisy server access logs.
          schedule(30000);
          return;
        }
        const s = await getStudioRunDownloaderScriptState(sessionId);
        if (cancelled) return;
        setDlBusy(Boolean(s.running));
        if (s.running) {
          // Prevent stale "finished" state from a prior run while a new run is active.
          setDlResult(null);
        }
        if (
          appView === "data-scrape" &&
          s.script_path &&
          s.script_path !== STUDIO_PIPELINE_DOWNLOADER_SYNC_SCRIPT_PATH
        ) {
          setDlGeneratedScripts((prev) => {
            if (prev.includes(s.script_path)) return prev;
            return [...prev, s.script_path];
          });
          setDlScriptChoice((c) => c || s.script_path);
        }
        if (typeof s.log === "string" && s.log.length > 0) {
          setDlRunLog(s.log);
        }
        // Avoid overwriting a good in-process POST snapshot with { total: 0 } idle state.
        if (s.progress && Number(s.progress.total) > 0) {
          const settledOk = !s.running && Boolean(s.result?.ok);
          setDlRunProgress({
            completed: settledOk ? s.progress.total : s.progress.completed,
            total: s.progress.total,
            active: settledOk ? s.progress.total : s.progress.current,
            dataset: s.progress.dataset,
          });
        }
        if (!s.running && s.result) {
          setDlResult(s.result);
        }
        const prevRunning = dlPrevRunningRef.current;
        if (prevRunning && !s.running && s.result && s.updated_at > dlLastNotifiedDoneAtRef.current) {
          dlLastNotifiedDoneAtRef.current = s.updated_at;
          const success = downloaderResultLooksSuccessful(s.result, s.log || "");
          notifyJobOutcome("Downloader", success, String(s.result?.message || (success ? "Download completed." : "Download failed.")));
        }
        dlPrevRunningRef.current = Boolean(s.running);
        // Keep polling even when idle so a newly-started downloader run is picked
        // up immediately (otherwise UI can stay stale until execute() returns).
        schedule(s.running ? 1500 : 5000);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (!warnedMissingStateRoute && msg.includes("404")) {
          warnedMissingStateRoute = true;
          pollingDisabled = true;
          onNotifyRef.current(
            "Downloader resume unavailable",
            "Backend is missing /run/downloader-script-state. Restart mito2 to load latest backend.",
            "err",
          );
          return;
        }
        // Transient network errors: retry conservatively.
        schedule(15000);
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [page, sessionId, appView, hpcPage]);

  const runPreSelected = async () => {
    const dataset_paths: string[] = [];
    let raw_download_folder = "";
    /** Match the download-run `<select>` value (see `downloadRunSelectValue` below). */
    const rawFolderForWholeRun =
      !rawDatasetInspectLoading &&
      preprocessDownloadRunOptions.length > 0 &&
      downloadRunSelectValue !== STUDIO_SELECT_LOADING
        ? downloadRunSelectValue
        : "";
    if (rawFolderForWholeRun) {
      raw_download_folder = rawFolderForWholeRun;
    } else {
      onNotify(
        "Nothing to preprocess",
        "Pick a download run under data/raw (see the list in the panel below if needed), then run preprocessing.",
        "err",
      );
      return;
    }
    let emInfo: Awaited<ReturnType<typeof getStudioRawEmStacks>>;
    try {
      emInfo = await getStudioRawEmStacks(raw_download_folder);
    } catch (e) {
      onNotify("Cannot verify EM stacks", e instanceof Error ? e.message : String(e), "err");
      return;
    }
    if (emInfo.count < 1) {
      onNotify(
        "Cannot run preprocessing",
        emInfo.detail?.trim() ||
          "No *_im.h5 files under this run's images/ folder. Finish the download first (expects data/raw/<run>/images/<tag>_im.h5).",
        "err",
      );
      return;
    }
    setPreResult(null);
    setPreRunLog("");
    setPreRunProgress(null);
    try {
      setPreRunLog("Sending preprocess request…\n");
      await postStudioPreprocessSelective(sessionId, {
        dataset_paths,
        task: "supervised",
        output_format: "h5",
        split_label_cc: preprocessSplitLabelCc,
        raw_download_folder,
      });
      // Match server ``running`` — do not show "Running…" until the job is accepted (avoids Kill racing an empty proc table).
      setPreSelectiveBusy(true);
      void (async () => {
        for (let i = 0; i < 6; i++) {
          try {
            const s = await getStudioRunPreprocessSelectiveState(sessionId);
            setPreSelectiveBusy(Boolean(s.running));
            setPreRemoteRunning(Boolean(s.running));
            if (typeof s.log === "string") {
              setPreRunLog(s.log);
            }
            if (s.progress) {
              setPreRunProgress({
                completed: s.progress.completed,
                total: s.progress.total,
                active: s.progress.current,
                dataset: s.progress.dataset,
              });
            }
            if (s.result) {
              setPreResult(s.result);
            }
            if (s.running && (s.progress || (s.log && s.log.length > 80))) {
              break;
            }
          } catch {
            break;
          }
          await new Promise<void>((r) => {
            window.setTimeout(r, 120);
          });
        }
      })();
    } catch (e) {
      onNotify("Selective preprocessor error", e instanceof Error ? e.message : String(e), "err");
      setPreSelectiveBusy(false);
    }
  };

  const refreshDataInspect = useCallback(async (showLoading = true, shallow = false, deepUnder?: string | null) => {
    if (showLoading) setDataInspectLoading(true);
    const ctl = new AbortController();
    let tid: number | null = null;
    if (typeof window !== "undefined") {
      tid = window.setTimeout(() => ctl.abort(), 120_000);
    }
    try {
      setDataInspect(
        await getStudioDataInspect({
          shallow,
          deepUnder: deepUnder ?? undefined,
          signal: ctl.signal,
        }),
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const aborted =
        msg.toLowerCase().includes("abort") ||
        (typeof e === "object" && e !== null && (e as { name?: string }).name === "AbortError");
      onNotifyRef.current(
        "Dataset inspect error",
        aborted ? "Timed out after 120s — try again or reduce files under data/raw." : msg,
        "err",
      );
    } finally {
      if (tid !== null) window.clearTimeout(tid);
      if (showLoading) setDataInspectLoading(false);
    }
  }, []);

  const refreshPostprocessingFiles = useCallback(async (showLoading = true) => {
    if (showLoading) setPostprocessFilesLoading(true);
    const ctl = new AbortController();
    let tid: number | null = null;
    if (typeof window !== "undefined") {
      tid = window.setTimeout(() => ctl.abort(), 30_000);
    }
    try {
      const res = await getStudioPostprocessingFiles(ctl.signal);
      const rows = (res.files ?? []).map((d) => ({
        ...normalizeDatasetInspectRow(d),
        source: d.source,
      }));
      setPostprocessFiles(rows);
      setPostprocessSelectedRowKeys((prev) => {
        const valid = new Set(rows.map((r) => `${r.source}:${r.path}`));
        if (!prev.size) {
          // Default first-load behavior: all rows are enabled for postprocessing.
          return valid;
        }
        const next = new Set<string>();
        prev.forEach((k) => {
          if (valid.has(k)) next.add(k);
        });
        return next;
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const aborted =
        msg.toLowerCase().includes("abort") ||
        (typeof e === "object" && e !== null && (e as { name?: string }).name === "AbortError");
      const notFound = msg.toLowerCase().includes("not found");
      if (!aborted && !notFound) {
        onNotifyRef.current("Postprocessing table error", msg, "err");
      }
      setPostprocessFiles([]);
      setPostprocessSelectedRowKeys(new Set());
    } finally {
      if (tid !== null) window.clearTimeout(tid);
      if (showLoading) setPostprocessFilesLoading(false);
    }
  }, []);

  useEffect(() => {
    if (page !== "processor") return;
    // Shallow only on entry: deep inspect opens every HDF5 and can monopolize the API thread pool for
    // minutes, starving preprocess state polling + kill (log/progress freeze, "Stopping…" stuck).
    void refreshDataInspect(true, true);
  }, [page, refreshDataInspect]);

  useEffect(() => {
    if (appView !== "data-existing-legacy") return;
    void refreshDataInspect(true, false);
    // Also load registry-enriched dataset status.
    setDsStatusLoading(true);
    getDatasetsStatus()
      .then((r) => setDsStatus(r.datasets))
      .catch(() => setDsStatus(null))
      .finally(() => setDsStatusLoading(false));
  }, [appView, refreshDataInspect]);

  useEffect(() => {
    if (appView !== "data-existing-legacy") return;
    if (existingDataTab !== "predicted") return;
    void refreshPostprocessingFiles(true);
  }, [appView, existingDataTab, refreshPostprocessingFiles]);

  const refreshMitoleInspect = useCallback(async (folder: string) => {
    setMitoleLoading(true);
    try {
      const inspect = await getMitoLeInspect(folder);
      setMitoleInspectRows(inspect.rows);
    } catch (e) {
      onNotifyRef.current("Refresh failed", e instanceof Error ? e.message : String(e), "err");
    } finally {
      setMitoleLoading(false);
    }
  }, []);

  const refreshMitoleCatalogue = useCallback(async (regenerate = false) => {
    setMitoleCatalogueLoading(true);
    try {
      const cat = await getMitoLeCatalogue(regenerate);
      setMitoleCatalogueRows(cat.rows);
      setMitoleCatalogueGenerated((cat.rows?.length ?? 0) > 0);
    } catch (e) {
      onNotifyRef.current("Catalogue refresh failed", e instanceof Error ? e.message : String(e), "err");
    } finally {
      setMitoleCatalogueLoading(false);
    }
  }, []);

  useEffect(() => {
    if (appView !== "data-hpc") return;
    setMitoleLoading(true);
    setMitoleStage1BootLoading(true);
    setMitoleCatalogueLoading(true);

    // Load lightweight Stage-1 config first; heavy scans are lazy-loaded when Stage 1 is opened.
    void getMitoLeConfig()
      .then((cfg) => {
        setMitoleBasePath(cfg.base_path);
        setMitoleFolders(cfg.folders);
        setMitoleSelectedSet(new Set(cfg.folders));
        setMitoleFolderPick("__all__");
      })
      .catch((e) => {
        onNotifyRef.current("MitoLE config load failed", e instanceof Error ? e.message : String(e), "err");
      })
      .finally(() => {
        setMitoleLoading(false);
      });

    // Stage 2 catalogue loads independently so Stage 2/3 don't wait on Stage 1 refresh.
    void getMitoLeCatalogue()
      .then((catalogue) => {
        setMitoleCatalogueRows(catalogue.rows);
        setMitoleCatalogueGenerated((catalogue.rows?.length ?? 0) > 0);
      })
      .catch((e) => {
        onNotifyRef.current("MitoLE stage 2 load failed", e instanceof Error ? e.message : String(e), "err");
      })
      .finally(() => {
        setMitoleCatalogueLoading(false);
      });
  }, [appView]);

  useEffect(() => {
    if (appView !== "data-hpc" || hpcPage !== "browser") return;
    setMitoleLoading(true);
    setMitoleStage1BootLoading(true);
    void Promise.all([getMitoLeSubfolders(), getMitoLeInspect(mitoleFolderPick || "__all__")])
      .then(([allFolders, inspect]) => {
        setMitoleAllSubfolders(allFolders.subfolders);
        setMitoleInspectRows(inspect.rows);
      })
      .catch((e) => {
        onNotifyRef.current("MitoLE stage 1 load failed", e instanceof Error ? e.message : String(e), "err");
      })
      .finally(() => {
        setMitoleLoading(false);
        setMitoleStage1BootLoading(false);
      });
  }, [appView, hpcPage, mitoleFolderPick]);

  // Intentionally do not auto-generate Stage 2 catalogue.
  // User explicitly controls regeneration via "Regenerate database/data catalogue".

  useEffect(() => {
    if (appView !== "model-postprocessing") return;
    void refreshPostprocessingFiles(true);
  }, [appView, refreshPostprocessingFiles]);

  const refreshInventoryCatalogue = useCallback(() => {
    setCatalogueLoading(true);
    void getInventoryCatalogue()
      .then(setCatalogue)
      .catch(() => setCatalogue(null))
      .finally(() => setCatalogueLoading(false));
  }, []);

  // Stage 0 Inventory: load catalogue when page is active.
  useEffect(() => {
    if (appView !== "data-scrape" || page !== "inventory") return;
    refreshInventoryCatalogue();
  }, [appView, page, refreshInventoryCatalogue]);

  const resetDownloadedTrainingAndHistory = useCallback(async () => {
    if (scrapeUiBusy || databaseUiBusy || dlUiBusy || preUiBusy || preKillBusy || trainingBusy) {
      onNotify("Cannot reset now", "A pipeline task is running. Stop or wait for completion first.", "err");
      return;
    }
    if (typeof window !== "undefined") {
      const ok = window.confirm(
        "Reset downloaded data + history?\n\nThis will delete Dataset001_mito2 train/test files and clear download/preprocess history.",
      );
      if (!ok) return;
    }
    setResetDownloadedBusy(true);
    try {
      const r = await postStudioResetDownloadedTraining(sessionId);
      // Also clear local Stage-0/1/2/3/4 UI state so the scrape pipeline looks untouched.
      setPendingDl(null);
      setPendingDlBusy(false);
      setDlPreview(null);
      setDlResult(null);
      setDlRunLog("");
      setDlRunProgress(null);
      setDatabaseBuildResult(null);
      setDatabaseRemoteLog("");
      setWsLast(null);
      setWsScrapeLog("");
      setPreRunLog("");
      setPreRunProgress(null);
      setPreResult(null);
      setTrainingResult(null);
      await Promise.all([
        getInventoryCatalogue().then(setCatalogue).catch(() => setCatalogue(null)),
        getStudioDataInspect({ shallow: true }).then(setDataInspect).catch(() => setDataInspect(null)),
      ]);
      onNotify(
        "Downloaded data reset",
        `Cleared Dataset001_mito2 train/test folders and history (downloads=${r.registry.downloads_deleted}, batches=${r.registry.download_batches_deleted}).`,
        "ok",
      );
    } catch (e) {
      onNotify("Reset failed", e instanceof Error ? e.message : String(e), "err");
    } finally {
      setResetDownloadedBusy(false);
    }
  }, [
    scrapeUiBusy,
    databaseUiBusy,
    dlUiBusy,
    preUiBusy,
    preKillBusy,
    trainingBusy,
    sessionId,
    onNotify,
  ]);

  const resetModelDownloadedDataAndHistory = useCallback(async () => {
    if (trainingBusy || inferenceBusy || postprocessBusy || evalBusy) {
      onNotify("Cannot reset now", "A model task is running. Stop or wait for completion first.", "err");
      return;
    }
    if (typeof window !== "undefined") {
      const ok = window.confirm(
        "Reset model outputs + history?\n\nThis will delete model outputs/cache and clear training/inference history.",
      );
      if (!ok) return;
    }
    setModelResetBusy(true);
    try {
      const r = await postStudioResetModelDownloadedDataHistory(sessionId);
      setPostprocessSelectedRowKeys(new Set());
      setPostprocessResult(null);
      setTrainingResult(null);
      setInferenceResult(null);
      setTrainingSelectedLogRoot("");
      setInferenceSelectedLogRoot("");
      void refreshPostprocessingFiles(true);
      onNotify(
        "Model outputs reset",
        `${r.message} Deleted ${r.deleted_files} file(s) and ${r.deleted_dirs} folder(s).`,
        "ok",
      );
    } catch (e) {
      onNotify("Reset model outputs failed", e instanceof Error ? e.message : String(e), "err");
    } finally {
      setModelResetBusy(false);
    }
  }, [
    trainingBusy,
    inferenceBusy,
    postprocessBusy,
    evalBusy,
    sessionId,
    refreshPostprocessingFiles,
    onNotify,
  ]);

  useEffect(() => {
    if (page !== "processor") return;
    if (sessionId !== "chat_bootstrap") return;
    let cancelled = false;
    let pollingDisabled = false;
    let warned404 = false;
    let timer: number | null = null;
    /** While preprocess is running, poll quickly so log/progress track ``[PROGRESS]`` without large lag. */
    const MS_RUNNING = 400;
    const MS_HIDDEN = 3000;
    const MS_IDLE_WITH_RESULT = 5000;
    const MS_IDLE = 3000;
    const schedule = (ms: number) => {
      if (cancelled || pollingDisabled) return;
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        void tick();
      }, ms);
    };
    const applyPreprocessState = (s: Awaited<ReturnType<typeof getStudioRunPreprocessSelectiveState>>) => {
      setPreSelectiveBusy(Boolean(s.running));
      setPreRemoteRunning(Boolean(s.running));
      if (typeof s.log === "string") {
        setPreRunLog(s.log);
      }
      if (s.progress) {
        setPreRunProgress({
          completed: s.progress.completed,
          total: s.progress.total,
          active: s.progress.current,
          dataset: s.progress.dataset,
        });
      } else if (!s.running) {
        setPreRunProgress(null);
      }
      if (s.result) {
        setPreResult(s.result);
      }
    };
    const tick = async () => {
      if (pollingDisabled) return;
      try {
        if (typeof document !== "undefined" && document.visibilityState === "hidden") {
          schedule(MS_HIDDEN);
          return;
        }
        const s = await getStudioRunPreprocessSelectiveState(sessionId);
        if (cancelled) return;
        applyPreprocessState(s);
        const prevRunning = prePrevRunningRef.current;
        const msgL = String(s.result?.message ?? "").toLowerCase();
        const killedStop =
          msgL.includes("stopped (killed)") || msgL.includes("kill requested") || Number(s.result?.returncode) === 130;
        if (prevRunning && !s.running && s.result && s.updated_at > preLastNotifiedDoneAtRef.current && !killedStop) {
          preLastNotifiedDoneAtRef.current = s.updated_at;
          onNotifyRef.current(
            "Selective preprocessor finished",
            s.result.message,
            s.result.ok ? "ok" : "err",
          );
          if (s.result.ok) {
            void (async () => {
              await refreshDataInspect(true, true);
              await refreshDataInspect(false, false);
            })();
          }
        }
        prePrevRunningRef.current = Boolean(s.running);
        if (s.running) {
          schedule(MS_RUNNING);
        } else if (s.result) {
          schedule(MS_IDLE_WITH_RESULT);
        } else {
          schedule(MS_IDLE);
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (!warned404 && msg.includes("404")) {
          warned404 = true;
          pollingDisabled = true;
          return;
        }
        schedule(5000);
      }
    };
    const onVisible = () => {
      if (cancelled || pollingDisabled || typeof document === "undefined") return;
      if (document.visibilityState === "visible") {
        schedule(0);
      }
    };
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", onVisible);
    }
    void tick();
    return () => {
      cancelled = true;
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", onVisible);
      }
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [page, sessionId, refreshDataInspect]);

  /** Every top-level name under ``data/raw`` that has at least one indexed file (includes ``preprocessed``, symlinks, etc.). */
  const preprocessDownloadRunOptions = useMemo(() => {
    if (!dataInspect) return [];
    const rawNorm = dataInspect.raw_base.replace(/\\/g, "/").replace(/\/+$/, "");
    const seen = new Set<string>();
    for (const d of dataInspect.raw_datasets) {
      const p = d.path.replace(/\\/g, "/");
      if (!p.startsWith(`${rawNorm}/`)) continue;
      const rel = p.slice(rawNorm.length + 1);
      const first = rel.split("/")[0];
      if (first) seen.add(first);
    }
    return Array.from(seen).sort(naturalCmp);
  }, [dataInspect]);

  /** Count of ``images/*_im.h5`` volumes from inspect index (fallback when server count not loaded). */
  const inspectEmH5Count = useMemo(
    () =>
      countPreprocessableEmStacksInRun(
        dataInspect,
        preprocessDownloadRun ?? "",
        preprocessDownloadRunOptions,
      ),
    [dataInspect, preprocessDownloadRun, preprocessDownloadRunOptions],
  );

  const preprocessImH5Count = serverEmH5Count !== null ? serverEmH5Count : inspectEmH5Count;
  const canRunPreprocessOnSelectedRun = preprocessImH5Count > 0;

  useEffect(() => {
    if (page !== "processor" || !preprocessDownloadRun || preprocessDownloadRunOptions.length === 0) {
      setServerEmH5Count(null);
      return;
    }
    if (!preprocessDownloadRunOptions.includes(preprocessDownloadRun)) {
      setServerEmH5Count(null);
      return;
    }
    const ac = new AbortController();
    let cancelled = false;
    void getStudioRawEmStacks(preprocessDownloadRun, ac.signal)
      .then((info) => {
        if (cancelled) return;
        setServerEmH5Count(typeof info.count === "number" ? info.count : 0);
      })
      .catch(() => {
        if (!cancelled) setServerEmH5Count(null);
      });
    return () => {
      cancelled = true;
      ac.abort();
    };
  }, [page, preprocessDownloadRun, preprocessDownloadRunOptions]);

  /** Selectable raw viewer folders exactly two levels under ``data/raw`` (e.g. ``preprocessed/labels``). */
  const rawViewerSelectionOptions = useMemo(() => {
    if (!dataInspect) return [];
    const rawNorm = dataInspect.raw_base.replace(/\\/g, "/").replace(/\/+$/, "");
    const seen = new Set<string>();
    for (const d of dataInspect.raw_datasets) {
      const p = d.path.replace(/\\/g, "/");
      if (!p.startsWith(`${rawNorm}/`)) continue;
      const rel = p.slice(rawNorm.length + 1);
      const parts = rel.split("/").filter(Boolean);
      if (parts.length < 2) continue;
      seen.add(`${parts[0]}/${parts[1]}`);
    }
    return Array.from(seen).sort(naturalCmp);
  }, [dataInspect]);

  /** Stage-4 progress is backend-owned state (worker + filesystem monitor). */
  const preProgress = preRunProgress;
  const preProgressPercent =
    preProgress && preProgress.total > 0 ? Math.round((preProgress.completed / preProgress.total) * 100) : 0;
  const preStateLabel = preUiBusy ? "running" : preResult?.ok ? "finished" : preResult ? "failed" : "idle";

  const rawViewerRows = useMemo<DatasetTableRow[]>(() => {
    if (!dataInspect) return [];
    const rawNorm = dataInspect.raw_base.replace(/\\/g, "/").replace(/\/+$/, "");
    const folderPick = rawViewerSelection ?? null;
    if (!folderPick) return [];
    const prefix = `${rawNorm}/${folderPick}/`;
    const wanted = dataInspect.raw_datasets.filter((d) => {
      const p = d.path.replace(/\\/g, "/");
      return p.startsWith(prefix);
    });
    const sorted = [...wanted].sort((a, b) => naturalCmp(a.path, b.path));
    return sorted.map((d, i) => {
      const base = normalizeDatasetInspectRow(d as DatasetInspectItem & Record<string, unknown>);
      return {
        ...base,
        groupKey: base.name,
        order: i + 1,
      };
    });
  }, [dataInspect, rawViewerSelection, rawViewerSelectionOptions]);

  const existingDataRows = useMemo<Array<DatasetTableRow & {
    source: "training" | "inference" | "instance";
    dataKind: "image" | "label" | "other";
    isInstanceLabel: boolean;
  }>>(() => {
    if (!dataInspect) return [];
    const training = dataInspect.training_datasets ?? dataInspect.preprocessed_datasets ?? [];
    const inference = dataInspect.inference_datasets ?? [];
    const instance = dataInspect.instance_datasets ?? [];
    const tagged = [
      ...training.map((d) => ({ ...d, source: "training" as const })),
      ...inference.map((d) => ({ ...d, source: "inference" as const })),
      ...instance.map((d) => ({ ...d, source: "instance" as const })),
    ];
    const sorted = [...tagged].sort((a, b) => naturalCmp(a.path, b.path));
    return sorted.map((d, i) => {
      const norm = normalizeDatasetInspectRow(d as DatasetInspectItem & Record<string, unknown>);
      const p = norm.path.replace(/\\/g, "/").toLowerCase();
      const isInstanceLabel = p.includes("/labelstr-instance/") || p.includes("/labelsts-instance/");
      const dataKind = p.includes("/labels/") || p.includes("/labelstr/") || p.includes("/labelsts/") || isInstanceLabel || p.endsWith("_seg.h5")
        ? "label"
        : p.includes("/images/") || p.endsWith("_im.h5")
          ? "image"
          : "other";
      return {
        ...norm,
        groupKey: norm.name,
        order: i + 1,
        source: d.source,
        dataKind,
        isInstanceLabel,
      };
    });
  }, [dataInspect]);

  const existingDataRowsFiltered = useMemo(
    () =>
      existingSourceFilter === "all"
        ? existingDataRows
        : existingDataRows.filter((r) => r.source === existingSourceFilter),
    [existingDataRows, existingSourceFilter],
  );

  const postprocessTableRows = useMemo(
    () =>
      [...postprocessFiles]
        .sort((a, b) => naturalCmp(a.path, b.path))
        .map((row, i) => ({
          ...row,
          order: i + 1,
        })),
    [postprocessFiles],
  );

  const postprocessTableRowsFiltered = useMemo(
    () =>
      predictedSourceFilter === "all"
        ? postprocessTableRows
        : postprocessTableRows.filter((r) => r.source === predictedSourceFilter),
    [postprocessTableRows, predictedSourceFilter],
  );

  const selectedTrainingDatasetNames = useMemo<string[]>(
    () => {
      const names = new Set<string>();
      for (const row of existingDataRowsFiltered) {
        const rowKey = `${row.source}:${row.name}:${row.path}`;
        if (row.source === "training" && dsSelectedRowKeys.has(rowKey)) names.add(toStableDatasetId(row.name));
      }
      return [...names];
    },
    [existingDataRowsFiltered, dsSelectedRowKeys],
  );

  const selectedInferenceImageRows = useMemo(
    () =>
      existingDataRows.filter(
        (row) =>
          row.source === "inference" &&
          row.dataKind === "image" &&
          selectedInferenceRowKeys.has(`${row.source}:${row.name}:${row.path}`),
      ),
    [existingDataRows, selectedInferenceRowKeys],
  );

  useEffect(() => {
    const inferenceRows = existingDataRows.filter((row) => row.source === "inference");
    if (inferenceRows.length === 0) {
      setSelectedInferenceRowKeys(new Set());
      return;
    }
    const next = new Set<string>();
    for (const row of inferenceRows) {
      const stableId = toStableDatasetId(row.name);
      const regInfo = dsStatus?.find((d) => d.stable_id === stableId || d.filename === row.name || d.filename.startsWith(stableId));
      if (!(regInfo?.hidden_from_inference ?? false)) {
        next.add(`${row.source}:${row.name}:${row.path}`);
      }
    }
    setSelectedInferenceRowKeys(next);
  }, [existingDataRows, dsStatus]);

  useEffect(() => {
    if (rawViewerSelectionOptions.length === 0) {
      setRawViewerSelection(null);
      return;
    }
    const valid = new Set(rawViewerSelectionOptions);
    setRawViewerSelection((prev) => (prev && valid.has(prev) ? prev : null));
  }, [rawViewerSelectionOptions]);

  /** Shallow global inspect skips HDF5 opens; deepen only the selected viewer folder so dimensions / label counts load. */
  useEffect(() => {
    if (page !== "processor") return;
    if (!rawViewerSelection || !dataInspect) return;
    if (dataInspect.inspect_shallow === false) return;
    if (dataInspect.inspect_deep_under === rawViewerSelection) return;
    void refreshDataInspect(true, true, rawViewerSelection);
  }, [page, rawViewerSelection, dataInspect, refreshDataInspect]);

  useEffect(() => {
    if (page !== "processor") {
      preprocessPrefsHydratedRef.current = false;
      preprocessCanPersistRef.current = false;
    }
  }, [page]);

  useEffect(() => {
    if (page !== "downloader" || appView !== "data-scrape") {
      downloaderPrefsHydratedRef.current = false;
      downloaderCanPersistRef.current = false;
    }
  }, [page, appView]);

  useEffect(() => {
    preprocessPrefsHydratedRef.current = false;
    preprocessCanPersistRef.current = false;
    downloaderPrefsHydratedRef.current = false;
    downloaderCanPersistRef.current = false;
  }, [sessionId]);

  useEffect(() => {
    if (page !== "processor") return;
    if (preprocessDownloadRunOptions.length === 0) {
      setPreprocessDownloadRun(null);
      preprocessCanPersistRef.current = true;
      return;
    }
    if (!preprocessPrefsHydratedRef.current) {
      preprocessPrefsHydratedRef.current = true;
      try {
        const raw = window.localStorage.getItem(studioPreprocessStorageKey(sessionId));
        if (raw) {
          const j = JSON.parse(raw) as {
            downloadRun?: string;
            splitLabelCc?: boolean;
          };
          if (j.downloadRun && preprocessDownloadRunOptions.includes(j.downloadRun)) {
            setPreprocessDownloadRun(j.downloadRun);
            if (typeof j.splitLabelCc === "boolean") {
              setPreprocessSplitLabelCc(j.splitLabelCc);
            }
            preprocessCanPersistRef.current = true;
            return;
          }
        }
      } catch {
        // ignore corrupt localStorage
      }
    }
    setPreprocessDownloadRun((prev) =>
      prev !== null && preprocessDownloadRunOptions.includes(prev)
        ? prev
        : null,
    );
    preprocessCanPersistRef.current = true;
  }, [page, sessionId, preprocessDownloadRunOptions]);

  useEffect(() => {
    if (!preprocessCanPersistRef.current || page !== "processor" || typeof window === "undefined") return;
    const payload = {
      downloadRun: preprocessDownloadRun,
      splitLabelCc: preprocessSplitLabelCc,
    };
    try {
      window.localStorage.setItem(studioPreprocessStorageKey(sessionId), JSON.stringify(payload));
    } catch {
      // ignore quota / private mode
    }
  }, [page, sessionId, preprocessDownloadRun, preprocessSplitLabelCc]);

  useEffect(() => {
    if (page !== "downloader" || appView !== "data-scrape") return;
    if (downloaderPrefsHydratedRef.current) return;
    downloaderPrefsHydratedRef.current = true;
    if (typeof window !== "undefined") {
      try {
        const rawV3 = window.localStorage.getItem(studioDownloaderStorageKey(sessionId));
        const raw = rawV3 ?? window.localStorage.getItem(studioDownloaderLegacyV2Key(sessionId));
        if (raw) {
          const j = JSON.parse(raw) as {
            datasetSplitsBySite?: Record<string, Record<string, DownloaderDatasetSplit>>;
            datasetTargetsBySite?: Record<string, Record<string, string>>;
          };
          if (j.datasetSplitsBySite && typeof j.datasetSplitsBySite === "object") {
            setDlSplitsBySite(j.datasetSplitsBySite);
          } else if (j.datasetTargetsBySite && typeof j.datasetTargetsBySite === "object") {
            const migrated: Record<string, Record<string, DownloaderDatasetSplit>> = {};
            for (const [site, rows] of Object.entries(j.datasetTargetsBySite)) {
              migrated[site] = {};
              for (const [dataset, target] of Object.entries(rows ?? {})) {
                const t = String(target || "").trim().toLowerCase();
                if (t === "inference") migrated[site][dataset] = { training: 0, inference: 1 };
                else if (t === "skip") migrated[site][dataset] = { training: 0, inference: 0 };
                else migrated[site][dataset] = { training: 1, inference: 0 };
              }
            }
            setDlSplitsBySite(migrated);
          }
        }
      } catch {
        /* ignore corrupt localStorage */
      }
    }
    downloaderCanPersistRef.current = true;
  }, [page, sessionId, appView]);

  useEffect(() => {
    if (!downloaderCanPersistRef.current || page !== "downloader" || appView !== "data-scrape" || typeof window === "undefined") return;
    try {
      window.localStorage.setItem(
        studioDownloaderStorageKey(sessionId),
        JSON.stringify({ datasetSplitsBySite: dlSplitsBySite }),
      );
    } catch {
      /* ignore quota / private mode */
    }
  }, [page, sessionId, appView, dlSplitsBySite]);

  // Fetch registry-aware pending download count when on Stage 3.
  useEffect(() => {
    if (page !== "downloader" || appView !== "data-scrape") return;
    const site = siteChoice.trim();
    if (!site) return;
    setPendingDl(null);
    setPendingDlBusy(true);
    getStudioPendingDownloads({
      site,
      n_crops: MAX_CROPS_PER_DATASET,
      chunk_zyx: STUDIO_DOWNLOADER_FIXED_CROP_VOXELS,
    })
      .then(setPendingDl)
      .catch(() => setPendingDl(null))
      .finally(() => setPendingDlBusy(false));
  }, [page, siteChoice, appView]);

  const rawDatasetInspectLoading = dataInspectLoading && !dataInspect;
  const downloadRunSelectDisabled =
    preUiBusy ||
    preKillBusy ||
    rawDatasetInspectLoading ||
    preprocessDownloadRunOptions.length === 0;
  const downloadRunSelectValue = rawDatasetInspectLoading
    ? STUDIO_SELECT_LOADING
    : (preprocessDownloadRun ?? "");
  const rawViewerScopeSelectValue = rawDatasetInspectLoading
    ? STUDIO_SELECT_LOADING
    : rawViewerSelection && rawViewerSelectionOptions.includes(rawViewerSelection)
        ? rawViewerSelection
        : "";
  const siteStemSelectValue = useMemo(() => {
    if (sitesLoading) return STUDIO_SELECT_LOADING;
    const cur = String(siteChoice || "").trim();
    if (!cur) return "";
    const curNorm = cur.toLowerCase();
    const matched = sites.find((s) => String(s || "").trim().toLowerCase() === curNorm);
    return matched ?? "";
  }, [sitesLoading, sites, siteChoice]);

  const dlPreviewSplits = useMemo<Record<string, DownloaderDatasetSplit>>(() => {
    const out: Record<string, DownloaderDatasetSplit> = {};
    for (const ds of dlPreview?.datasets ?? []) out[ds] = dlDatasetSplits[ds] ?? { training: 1, inference: 0 };
    return out;
  }, [dlDatasetSplits, dlPreview]);

  const dlFilteredPreviewDatasets = useMemo<string[]>(() => {
    const all = dlPreview?.datasets ?? [];
    if (all.length === 0) return [];
    const selected = new Set(dlSampleTypesSelected);
    return all.filter((ds) => selected.has(dlSampleTypeByDataset[ds] || "unknown"));
  }, [dlPreview, dlSampleTypesSelected, dlSampleTypeByDataset]);

  const showRawViewerTableBody = rawViewerRows.length > 0;
  const showRawViewerEmptyMessage =
    !dataInspectLoading && Boolean(dataInspect) && rawViewerRows.length === 0;

  const wsPickSelectValue = wsListLoading
    ? STUDIO_SELECT_LOADING
    : wsPickSlug && wsList.some((w) => w.slug === wsPickSlug)
      ? wsPickSlug
      : "";

  const scrapeTargetSelectValue = wsListLoading
    ? STUDIO_SELECT_LOADING
    : wsList.length === 0
      ? ""
      : scrapeTargetSlug && wsList.some((w) => w.slug === scrapeTargetSlug)
        ? scrapeTargetSlug
        : "";

  const deletePickSelectValue = wsListLoading
    ? STUDIO_SELECT_LOADING
    : deletePickSlug && wsListDeletable.some((w) => w.slug === deletePickSlug)
      ? deletePickSlug
      : "";

  const runTraining = async () => {
    setTrainingBusy(true);
    setTrainingResult(null);
    try {
      const r = await postStudioTraining(sessionId);
      setTrainingResult(r);
      try {
        const s = await getStudioTrainingState(sessionId, trainingSelectedLogRoot);
        setTrainingState(s);
      } catch {
        /* ignore state refresh errors */
      }
      onNotify(r.ok ? "Slurm training job submitted" : "Slurm submit failed", r.message, r.ok ? "ok" : "err");
      notifyDesktopPopup(r.ok ? "Slurm training job submitted" : "Slurm submit failed", r.message);
    } catch (e) {
      notifyJobOutcome("nnUNet training submit", false, e instanceof Error ? e.message : String(e));
    } finally {
      setTrainingBusy(false);
    }
  };

  const runInference = async () => {
    setInferenceBusy(true);
    setInferenceResult(null);
    try {
      const r = await postStudioInference(sessionId);
      setInferenceResult(r);
      try {
        const s = await getStudioInferenceState(sessionId, inferenceSelectedLogRoot);
        setInferenceState(s);
      } catch {
        /* ignore state refresh errors */
      }
      onNotify(
        r.ok ? "Slurm inference job submitted" : "Slurm submit failed",
        r.message,
        r.ok ? "ok" : "err",
      );
    } catch (e) {
      notifyJobOutcome("nnUNet inference submit", false, e instanceof Error ? e.message : String(e));
    } finally {
      setInferenceBusy(false);
    }
  };

  const runPostprocessing = async () => {
    setPostprocessBusy(true);
    setPostprocessResult(null);
    try {
      const r = await postStudioPostprocessing(sessionId, {
        input_dir: FIXED_POSTPROCESS_INPUT_DIR,
        output_dir: FIXED_POSTPROCESS_OUTPUT_DIR,
      });
      setPostprocessResult(r);
      notifyJobOutcome("Postprocessing", r.ok, r.message);
    } catch (e) {
      notifyJobOutcome("Postprocessing", false, e instanceof Error ? e.message : String(e));
    } finally {
      setPostprocessBusy(false);
    }
  };

  const runEvaluation = async () => {
    setEvalBusy(true);
    setEvalResult(null);
    try {
      const r = await postStudioEvaluation(sessionId, {
        pred_dir: FIXED_EVAL_PRED_DIR,
        gt_dir: FIXED_EVAL_GT_DIR,
      });
      setEvalResult(r);
      notifyJobOutcome("Evaluation", r.ok, r.message);
    } catch (e) {
      notifyJobOutcome("Evaluation", false, e instanceof Error ? e.message : String(e));
    } finally {
      setEvalBusy(false);
    }
  };

  const clearTrainingOutput = async () => {
    setTrainingClearBusy(true);
    try {
      await postStudioTrainingStateClear(sessionId);
      setTrainingState(null);
      setTrainingResult(null);
      setTrainingSelectedLogRoot("");
      onNotify("Training output cleared", "Cleared training out/err view for this session.", "ok");
    } catch (e) {
      onNotify("Clear training output failed", e instanceof Error ? e.message : String(e), "err");
    } finally {
      setTrainingClearBusy(false);
    }
  };

  const clearInferenceOutput = async () => {
    setInferenceClearBusy(true);
    try {
      await postStudioInferenceStateClear(sessionId);
      setInferenceState(null);
      setInferenceResult(null);
      setInferenceSelectedLogRoot("");
      onNotify("Inference output cleared", "Cleared inference out/err view for this session.", "ok");
    } catch (e) {
      onNotify("Clear inference output failed", e instanceof Error ? e.message : String(e), "err");
    } finally {
      setInferenceClearBusy(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const shouldPoll = appView === "model-training" || appView === "model-inference";
    if (!shouldPoll) return;

    const tick = async () => {
      try {
        const [ts, is] = await Promise.all([
          getStudioTrainingState(sessionId, trainingSelectedLogRoot),
          getStudioInferenceState(sessionId, inferenceSelectedLogRoot),
        ]);
        if (cancelled) return;
        setTrainingState(ts);
        setInferenceState(is);
      } catch {
        /* ignore polling errors */
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [
    appView,
    sessionId,
    trainingSelectedLogRoot,
    inferenceSelectedLogRoot,
  ]);

  const mitoleInspectRowsFiltered = useMemo(
    () => mitoleInspectRows.filter((r) => mitoleFolderPick === "__all__" || r.folder === mitoleFolderPick),
    [mitoleInspectRows, mitoleFolderPick],
  );
  const hpcBrowserRows = useMemo(() => {
    const rows = mitoleInspectRowsFiltered;
    const keyNorm = (name: string): string => {
      let s = String(name || "").toLowerCase();
      s = s.replace(/\.nii\.gz$/i, "").replace(/\.(h5|nii|tif|tiff)$/i, "");
      s = s.replace(/([._-])(im|image|img)$/i, "");
      s = s.replace(/([._-])(mito|seg|label|labels|mask|gt)$/i, "");
      s = s.replace(/[._-](v\d+|vol\d+|slice\d+|patch\d+)$/i, "");
      return s;
    };
    const isLabel = (r: MitoLeInspectRow): boolean => {
      const n = String(r.name || "").toLowerCase();
      const p = String(r.path || "").toLowerCase();
      return /(label|labels|mask|_seg|segmentation|_gt|_mito)/.test(n) || /(\/labels?\/|\/masks?\/|\/mito\/)/.test(p);
    };
    const bySource = new Map<string, { images: MitoLeInspectRow[]; labels: MitoLeInspectRow[] }>();
    for (const r of rows) {
      const src = String(r.folder || "");
      const bucket = bySource.get(src) ?? { images: [], labels: [] };
      if (isLabel(r)) bucket.labels.push(r);
      else bucket.images.push(r);
      bySource.set(src, bucket);
    }
    const out: Array<{
      order: number;
      source: string;
      pairName: string;
      image: MitoLeInspectRow;
      label: MitoLeInspectRow;
      dimensions: number[];
      spacing: number[];
      type: string;
    }> = [];
    for (const [source, bucket] of bySource.entries()) {
      const labelByKey = new Map<string, MitoLeInspectRow[]>();
      for (const l of bucket.labels) {
        const k = keyNorm(l.name);
        const arr = labelByKey.get(k) ?? [];
        arr.push(l);
        labelByKey.set(k, arr);
      }
      for (const im of bucket.images) {
        const k = keyNorm(im.name);
        const ls = labelByKey.get(k) ?? [];
        if (ls.length === 0) continue;
        const lb = ls.shift() as MitoLeInspectRow;
        labelByKey.set(k, ls);
        out.push({
          order: out.length + 1,
          source,
          pairName: k,
          image: im,
          label: lb,
          dimensions: (im.dimensions && im.dimensions.length > 0 ? im.dimensions : lb.dimensions) ?? [],
          spacing: (im.spacing && im.spacing.length > 0 ? im.spacing : lb.spacing) ?? [],
          type: String(im.type || lb.type || ""),
        });
      }
    }
    return out;
  }, [mitoleInspectRowsFiltered]);
  const mitoleToggleCatSort = (col: "dataset" | "source" | "organism" | "sample_type") => {
    if (mitoleCatSortCol === col) setMitoleCatSortAsc((v) => !v);
    else {
      setMitoleCatSortCol(col);
      setMitoleCatSortAsc(true);
    }
  };
  const mitoleCatSortIcon = (col: "dataset" | "source" | "organism" | "sample_type") =>
    mitoleCatSortCol === col ? (mitoleCatSortAsc ? " ↑" : " ↓") : "";
  const mitolePairCatalogueRows = useMemo(() => {
    return mitoleCatalogueRows.map((r, i) => {
      const dataset = String(r.dataset || "").trim();
      const source = String(r.source || r.folder || "").trim();
      return {
        key: `${source}:${dataset}:${i}`,
        order: i + 1,
        dataset,
        source,
        organism: String(r.organism || "unknown"),
        sample_type: String(r.sample_type || "unknown"),
        image_file: String(r.image_file || ""),
        label_file: String(r.label_file || ""),
        image_path: String(r.image_path || ""),
        label_path: String(r.label_path || ""),
        dimensions: Array.isArray(r.dimensions) ? r.dimensions : [],
        spacing: Array.isArray(r.spacing) ? r.spacing : [],
      };
    });
  }, [mitoleCatalogueRows]);
  const mitolePairCatalogueFilterOptions = useMemo(() => {
    return {
      sources: Array.from(new Set(mitolePairCatalogueRows.map((r) => String(r.source || "").trim()).filter(Boolean))).sort(),
      organisms: Array.from(new Set(mitolePairCatalogueRows.map((r) => String(r.organism || "").trim()).filter(Boolean))).sort(),
      sampleTypes: Array.from(new Set(mitolePairCatalogueRows.map((r) => String(r.sample_type || "").trim()).filter(Boolean))).sort(),
    };
  }, [mitolePairCatalogueRows]);
  const mitolePairCatalogueRowsFiltered = useMemo(() => {
    const q = mitoleCatQuery.trim().toLowerCase();
    const rows = mitolePairCatalogueRows.filter((r) => {
      if (mitoleCatSelectedSources.size > 0 && !mitoleCatSelectedSources.has(r.source)) return false;
      if (mitoleCatSelectedOrganisms.size > 0 && !mitoleCatSelectedOrganisms.has(r.organism)) return false;
      if (mitoleCatSelectedSampleTypes.size > 0 && !mitoleCatSelectedSampleTypes.has(r.sample_type)) return false;
      if (!q) return true;
      return [r.dataset, r.source, r.organism, r.sample_type, r.image_file, r.label_file].some((v) =>
        String(v || "").toLowerCase().includes(q),
      );
    });
    rows.sort((a, b) => {
      const av = String(a[mitoleCatSortCol] || "").toLowerCase();
      const bv = String(b[mitoleCatSortCol] || "").toLowerCase();
      const cmp = av.localeCompare(bv, undefined, { sensitivity: "base", numeric: true });
      return mitoleCatSortAsc ? cmp : -cmp;
    });
    return rows;
  }, [
    mitolePairCatalogueRows,
    mitoleCatQuery,
    mitoleCatSelectedSources,
    mitoleCatSelectedOrganisms,
    mitoleCatSelectedSampleTypes,
    mitoleCatSortCol,
    mitoleCatSortAsc,
  ]);

  const mitoleSelectedPair = useMemo(() => {
    if (!mitoleCatSelectedPairKey) return null;
    return mitolePairCatalogueRows.find((r) => r.key === mitoleCatSelectedPairKey) ?? null;
  }, [mitoleCatSelectedPairKey, mitolePairCatalogueRows]);
  const hpcDlSampleTypeByDataset = useMemo(() => {
    const m = new Map<string, string>();
    for (const r of mitoleCatalogueRows) {
      const ds = String(r.dataset || "").trim();
      if (!ds || m.has(ds)) continue;
      m.set(ds, String(r.sample_type || "unknown").trim() || "unknown");
    }
    return m;
  }, [mitoleCatalogueRows]);
  const hpcDlSourceByDataset = useMemo(() => {
    const m = new Map<string, string>();
    for (const r of mitoleCatalogueRows) {
      const ds = String(r.dataset || "").trim();
      if (!ds || m.has(ds)) continue;
      m.set(ds, String(r.source || r.folder || "").trim());
    }
    return m;
  }, [mitoleCatalogueRows]);
  const hpcDlSampleTypeOptions = useMemo(() => {
    const allDatasets = mitoleCatalogueRows
      .filter((r) => mitoleFolderPick === "__all__" || String(r.source || r.folder || "") === mitoleFolderPick)
      .map((r) => String(r.dataset || "").trim())
      .filter(Boolean);
    return Array.from(new Set(allDatasets.map((ds) => hpcDlSampleTypeByDataset.get(ds) || "unknown"))).sort((a, b) =>
      a.localeCompare(b),
    );
  }, [mitoleCatalogueRows, mitoleFolderPick, hpcDlSampleTypeByDataset]);
  const hpcDlFilteredPreviewDatasets = useMemo<string[]>(() => {
    const selectedTypes = new Set(hpcDlSampleTypesSelected);
    const allDatasets = Array.from(
      new Set(mitoleCatalogueRows.map((r) => String(r.dataset || "").trim()).filter(Boolean)),
    );
    return allDatasets.filter((ds) => {
      const srcOk = mitoleFolderPick === "__all__" || hpcDlSourceByDataset.get(ds) === mitoleFolderPick;
      if (!srcOk) return false;
      return selectedTypes.has(hpcDlSampleTypeByDataset.get(ds) || "unknown");
    });
  }, [mitoleCatalogueRows, mitoleFolderPick, hpcDlSourceByDataset, hpcDlSampleTypesSelected, hpcDlSampleTypeByDataset]);
  const activeDlFilteredPreviewDatasets = useMemo<string[]>(
    () => (appView === "data-hpc" && hpcPage === "downloader" ? hpcDlFilteredPreviewDatasets : dlFilteredPreviewDatasets),
    [appView, hpcPage, hpcDlFilteredPreviewDatasets, dlFilteredPreviewDatasets],
  );
  const mitoleStage1Busy = mitoleLoading || (appView === "data-hpc" && hpcPage === "browser" && mitoleStage1BootLoading);
  useEffect(() => {
    if (appView !== "data-hpc" || hpcPage !== "downloader") return;
    const all = hpcDlSampleTypeOptions;
    if (all.length === 0) {
      setHpcDlSampleTypesSelected((prev) => (prev.length === 0 ? prev : []));
      return;
    }
    setHpcDlSampleTypesSelected((prev) => {
      const kept = prev.filter((x) => all.includes(x));
      const next = kept.length > 0 ? kept : all;
      return sameStringSet(prev, next) ? prev : next;
    });
  }, [appView, hpcPage, hpcDlSampleTypeOptions]);
  const inDataSection = appView === "data-hpc" || appView === "data-scrape";
  const inModelSection =
    appView === "model-training" || appView === "model-inference" || appView === "model-postprocessing";
  const inHomeSection = appView === "home";
  const inLegacySection = appView === "data-existing-legacy";

  return (
    <div className={`pipeline-studio${chatPanelCollapsed ? " pipeline-studio-chat-collapsed" : ""}`}>
      {/* ── Breadcrumb ── */}
      <nav className="ia-breadcrumb" aria-label="Breadcrumb">
        <ol className="ia-breadcrumb-list">
          <li className="ia-breadcrumb-item" aria-current={inHomeSection ? "page" : undefined}>
            <button
              type="button"
              className={`ia-breadcrumb-link${inHomeSection ? " ia-breadcrumb-current" : ""}`}
              disabled={inHomeSection}
              onClick={() => setAppView("home")}
            >
              Home
            </button>
          </li>
          <li className="ia-breadcrumb-item" aria-current={inDataSection ? "page" : undefined}>
            <button
              type="button"
              className={`ia-breadcrumb-link${inDataSection ? " ia-breadcrumb-current" : ""}`}
              disabled={inDataSection}
              onClick={() => setAppView("data-hpc")}
            >
              Data
            </button>
          </li>
          <li className="ia-breadcrumb-item" aria-current={inModelSection ? "page" : undefined}>
            <button
              type="button"
              className={`ia-breadcrumb-link${inModelSection ? " ia-breadcrumb-current" : ""}`}
              disabled={inModelSection}
              onClick={() => setAppView("model-training")}
            >
              Model
            </button>
          </li>
          <li className="ia-breadcrumb-item" aria-current={inLegacySection ? "page" : undefined}>
            <button
              type="button"
              className={`ia-breadcrumb-link${inLegacySection ? " ia-breadcrumb-current" : ""}`}
              disabled={inLegacySection}
              onClick={() => setAppView("data-existing-legacy")}
            >
              Downloaded & Predicted Data
            </button>
          </li>
        </ol>
      </nav>

      {/* ── Data segment control ── */}
      {(appView === "data-hpc" || appView === "data-scrape") && (
        <div className="ia-segment-bar" role="tablist" aria-label="Data section">
          <button
            type="button"
            role="tab"
            className={`ia-segment-tab${appView === "data-scrape" && page === "inventory" ? " ia-segment-tab--active" : ""}`}
            aria-selected={appView === "data-scrape" && page === "inventory"}
            onClick={() => {
              setAppView("data-scrape");
              setPage("inventory");
            }}
          >
            Inventory
          </button>
          <button
            type="button"
            role="tab"
            className={`ia-segment-tab${appView === "data-hpc" ? " ia-segment-tab--active" : ""}`}
            aria-selected={appView === "data-hpc"}
            onClick={() => setAppView("data-hpc")}
          >
            Process Local HPC Data
          </button>
          <button
            type="button"
            role="tab"
            className={`ia-segment-tab${
              appView === "data-scrape" && page !== "inventory" ? " ia-segment-tab--active" : ""
            }`}
            aria-selected={appView === "data-scrape" && page !== "inventory"}
            onClick={() => {
              setAppView("data-scrape");
              if (page === "inventory") setPage("scraper");
            }}
          >
            Scrape Data From Websites
          </button>
        </div>
      )}

      {/* ── Model segment control ── */}
      {(appView === "model-training" || appView === "model-inference" || appView === "model-postprocessing") && (
        <div className="ia-model-top">
          <div className="ia-segment-bar ia-segment-bar--model" role="tablist" aria-label="Model section">
            <button
              type="button"
              role="tab"
              className={`ia-segment-tab${appView === "model-training" ? " ia-segment-tab--active" : ""}`}
              aria-selected={appView === "model-training"}
              onClick={() => setAppView("model-training")}
            >
              Training
            </button>
            <button
              type="button"
              role="tab"
              className={`ia-segment-tab${appView === "model-inference" ? " ia-segment-tab--active" : ""}`}
              aria-selected={appView === "model-inference"}
              onClick={() => setAppView("model-inference")}
            >
              Inference
            </button>
            <button
              type="button"
              role="tab"
              className={`ia-segment-tab${appView === "model-postprocessing" ? " ia-segment-tab--active" : ""}`}
              aria-selected={appView === "model-postprocessing"}
              onClick={() => setAppView("model-postprocessing")}
            >
              Postprocessing and evaluation
            </button>
          </div>
          <div className="ia-model-top-actions">
            <button
              type="button"
              className="danger"
              disabled={modelResetBusy || trainingBusy || inferenceBusy || postprocessBusy || evalBusy}
              onClick={() => void resetModelDownloadedDataAndHistory()}
              title="Clear outputs/bc, outputs/postprocessed, and train/infer slurm logs"
            >
              <SizeStableLabel
                label="Reset downloaded data + history"
                busyLabel="Resetting…"
                isBusy={modelResetBusy}
              />
            </button>
          </div>
        </div>
      )}

      {/* ── Scrape pipeline mini-stepper (circles + lines, only within data-scrape) ── */}
      {appView === "data-scrape" && page !== "inventory" && (
        <div className="studio-stepper-wrap">
          <nav className="studio-stepper" aria-label="Scrape pipeline steps">
            <div className="studio-stepper-sections-row" aria-hidden>
              <div style={{ flex: "1 1 0" }} />
              <div
                className="studio-stepper-section-badge studio-stepper-section-badge--data"
                style={{ flex: "4 4 0" }}
              >
                Scrape Pipeline
              </div>
              <div style={{ flex: "1 1 0" }} />
            </div>
            <ol className="studio-stepper-track">
              {SCRAPE_NAV.filter((t) => t.id !== "inventory").map((t, i, arr) => (
                <li
                  key={t.id}
                  className={[
                    "studio-step",
                    page === t.id ? "studio-step-current" : "",
                    scrapeNavRemoteWorking(t.id) ? "studio-step-working" : "",
                    "studio-step-data",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  <div className="studio-step-rail-row">
                    <span
                      className={`studio-step-seg studio-step-seg-left${i === 0 ? " studio-step-seg-empty" : ""}`}
                      aria-hidden
                    />
                    <button
                      type="button"
                      className="studio-step-node"
                      aria-current={page === t.id ? "step" : undefined}
                      aria-busy={scrapeNavRemoteWorking(t.id) || undefined}
                      aria-label={`${t.label}, stage ${t.stageNum}`}
                      onClick={() => setPage(t.id)}
                    >
                      <span className="studio-step-circle">{t.stageNum}</span>
                    </button>
                    <span
                      className={`studio-step-seg studio-step-seg-right${
                        i === arr.length - 1 ? " studio-step-seg-empty" : ""
                      }`}
                      aria-hidden
                    />
                  </div>
                  <span className="studio-step-caption">{t.label}</span>
                </li>
              ))}
            </ol>
          </nav>
        </div>
      )}

      {appView === "data-hpc" && (
        <div className="studio-stepper-wrap">
          <nav className="studio-stepper" aria-label="Local HPC data pipeline steps">
            <div className="studio-stepper-sections-row" aria-hidden>
              <div style={{ flex: "1 1 0" }} />
              <div
                className="studio-stepper-section-badge studio-stepper-section-badge--data"
                style={{ flex: "4 4 0" }}
              >
                Local HPC Pipeline
              </div>
              <div style={{ flex: "1 1 0" }} />
            </div>
            <ol className="studio-stepper-track">
              {HPC_NAV.map((t, i) => (
                <li
                  key={t.id}
                  className={[
                    "studio-step",
                    hpcPage === t.id ? "studio-step-current" : "",
                    hpcNavRemoteWorking(t.id) ? "studio-step-working" : "",
                    "studio-step-data",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  <div className="studio-step-rail-row">
                    <span className={`studio-step-seg studio-step-seg-left${i === 0 ? " studio-step-seg-empty" : ""}`} aria-hidden />
                    <button
                      type="button"
                      className="studio-step-node"
                      aria-current={hpcPage === t.id ? "step" : undefined}
                      aria-busy={hpcNavRemoteWorking(t.id) || undefined}
                      aria-label={`${t.label}, stage ${t.stageNum}`}
                      onClick={() => setHpcPage(t.id)}
                    >
                      <span className="studio-step-circle">{t.stageNum}</span>
                    </button>
                    <span className={`studio-step-seg studio-step-seg-right${i === HPC_NAV.length - 1 ? " studio-step-seg-empty" : ""}`} aria-hidden />
                  </div>
                  <span className="studio-step-caption">{t.label}</span>
                </li>
              ))}
            </ol>
          </nav>
        </div>
      )}

      <div className="studio-page">
        {/* ── Home ── */}
        {appView === "home" && (
          <div className="ia-home">
            <h1 className="ia-home-title">mitoFoundation2</h1>
            <p className="ia-home-lead muted-note">
              End-to-end pipeline for mitochondria segmentation with nnUNet.
            </p>
            <div className="ia-home-tiles">
              <button
                type="button"
                className="ia-home-tile ia-home-tile--data"
                onClick={() => setAppView("data-hpc")}
              >
                <div className="ia-home-tile-label">Data</div>
                <div className="ia-home-tile-desc">
                  Run local HPC data processing from selected MitoLE folders, or scrape new datasets from websites.
                </div>
              </button>
              <button
                type="button"
                className="ia-home-tile ia-home-tile--model"
                onClick={() => setAppView("model-training")}
              >
                <div className="ia-home-tile-label">Model</div>
                <div className="ia-home-tile-desc">
                  Training and inference workflows for Dataset001 with nnUNet.
                </div>
              </button>
            </div>
            <button
              type="button"
              className="ia-home-manage-tab"
              onClick={() => setAppView("data-existing-legacy")}
            >
              Manage Downloaded & Predicted Data
            </button>
            <p className="ia-home-summary-link muted-note">
              <button
                type="button"
                className="linkish"
                onClick={() => setAppView("pipeline-summary")}
              >
                View pipeline summary
              </button>
            </p>
          </div>
        )}

        {/* ── Local HPC Data Pipeline ── */}
        {appView === "data-hpc" && (
          <div className="ia-section">
            <h1 className="ia-section-title">Process Existing Local HPC Data</h1>
            <p className="muted-note">
              Pipeline root: <code>{mitoleBasePath}</code>. Stage 1 scans selected subfolders, Stage 2 builds a
              browseable catalogue view, and Stage 3 prepares downloader-ready dataset picks.
            </p>
            {hpcPage === "browser" && (
              <section className={`studio-dataset-viewer${mitoleStage1Busy ? " studio-surface-updating" : ""}`} aria-busy={mitoleStage1Busy}>
                <h2 className="studio-subhead">Stage 1</h2>
                <details open>
                  <summary className="studio-subhead">Dataset Source Selection</summary>
                  <p className="muted-note">Select which subfolders should be included in the browser table below.</p>
                  <div className="studio-dataset-table-wrap" style={{ maxHeight: "18rem" }}>
                    <table className="studio-dataset-table">
                      <thead>
                        <tr>
                          <th style={{ width: "2rem" }}>Use</th>
                          <th>Subfolder</th>
                        </tr>
                      </thead>
                      <tbody>
                        {mitoleAllSubfolders.map((folder) => (
                          <tr key={folder}>
                            <td>
                              <input
                                type="checkbox"
                                checked={mitoleSelectedSet.has(folder)}
                                onChange={(e) => {
                                  setMitoleSelectedSet((prev) => {
                                    const next = new Set(prev);
                                    if (e.target.checked) next.add(folder);
                                    else next.delete(folder);
                                    return next;
                                  });
                                }}
                              />
                            </td>
                            <td>{folder}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="studio-actions">
                    <button
                      type="button"
                      className="primary"
                      disabled={mitoleStage1Busy}
                      onClick={async () => {
                        setMitoleLoading(true);
                        try {
                          const nextFolders = [...mitoleSelectedSet].sort((a, b) => a.localeCompare(b));
                          const cfg = await postMitoLeConfig(nextFolders);
                          setMitoleFolders(cfg.folders);
                          const nextPick = "__all__";
                          setMitoleFolderPick(nextPick);
                          const inspect = await getMitoLeInspect(nextPick);
                          setMitoleInspectRows(inspect.rows);
                          setMitoleCatalogueGenerated(false);
                          onNotify("Source selection updated", `Using ${cfg.folders.length} subfolder(s).`, "ok");
                        } catch (e) {
                          onNotify("Update failed", e instanceof Error ? e.message : String(e), "err");
                        } finally {
                          setMitoleLoading(false);
                        }
                      }}
                    >
                      Apply selected sources
                    </button>
                    <button
                      type="button"
                      className="ghost"
                      disabled={mitoleStage1Busy}
                      onClick={async () => {
                        setMitoleLoading(true);
                        try {
                          const cfg = await postMitoLeConfig([]);
                          setMitoleFolders(cfg.folders);
                          setMitoleSelectedSet(new Set(cfg.folders));
                          const nextPick = "__all__";
                          setMitoleFolderPick(nextPick);
                          const inspect = await getMitoLeInspect(nextPick);
                          setMitoleInspectRows(inspect.rows);
                          setMitoleCatalogueGenerated(false);
                          onNotify("Default sources restored", `Using ${cfg.folders.length} default subfolder(s).`, "ok");
                        } catch (e) {
                          onNotify("Reset to defaults failed", e instanceof Error ? e.message : String(e), "err");
                        } finally {
                          setMitoleLoading(false);
                        }
                      }}
                    >
                      Back to default sources
                    </button>
                  </div>
                </details>
                <h3 className="studio-subhead studio-subhead-spaced">Folder dataset browser</h3>
              <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
                <span className="muted-note">Source filter</span>
                <select className="field-input studio-field" style={{ maxWidth: "20rem" }} value={mitoleFolderPick} onChange={(e) => setMitoleFolderPick(e.target.value)}>
                  <option value="__all__">All selected folders</option>
                  {mitoleFolders.map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
                <StudioUpdatingBadge active={mitoleStage1Busy} label="Refreshing…" />
                <button
                  type="button"
                  className="ghost"
                  disabled={mitoleStage1Busy}
                  onClick={async () => {
                    await refreshMitoleInspect(mitoleFolderPick);
                  }}
                >
                  Refresh
                </button>
              </div>
              <div className="studio-dataset-table-wrap">
                <table className="studio-dataset-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Pair</th>
                      <th>Source</th>
                      <th>Image file</th>
                      <th>Label file</th>
                      <th style={{ minWidth: "10rem" }}>Dimensions</th>
                      <th style={{ minWidth: "12rem" }}>Physical spacing</th>
                    </tr>
                  </thead>
                  <tbody>
                    {hpcBrowserRows.map((r) => {
                      const rowSelKey = `${r.source}:${r.image.path}:${r.label.path}`;
                      return (
                      <tr key={rowSelKey}>
                        <td>{r.order}</td>
                        <td>{r.pairName}</td>
                        <td>{r.source}</td>
                        <td>{r.image.name}</td>
                        <td>{r.label.name}</td>
                        <td style={{ minWidth: "10rem" }}>{asCellVector(r.dimensions)}</td>
                        <td style={{ minWidth: "12rem" }}>{asSpacingCell(r.spacing)}</td>
                      </tr>
                      );
                    })}
                    {hpcBrowserRows.length === 0 && (
                      <tr><td colSpan={7} className="studio-dataset-placeholder-cell">No image-label pairs found for current selection.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
              </section>
            )}

            {hpcPage === "catalog" && (
              <section className={`studio-dataset-viewer${mitoleCatalogueLoading ? " studio-surface-updating" : ""}`} aria-busy={mitoleCatalogueLoading}>
                <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
                  <h3 className="studio-subhead">Stage 2: Dataset catalog (browse & filter)</h3>
                  <div className="studio-actions" style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                    <button
                      type="button"
                      className="primary"
                      disabled={mitoleLoading || mitoleCatalogueLoading}
                      onClick={async () => {
                        // Rebuild persisted Stage 2 catalogue explicitly on user action.
                        await refreshMitoleCatalogue(true);
                        setMitoleCatSelectedPairKey(null);
                      }}
                      title="Rebuild the persisted Stage 2 database/data catalogue"
                    >
                      Regenerate database/data catalogue
                    </button>
                  </div>
                </div>
                <p className="muted-note" style={{ marginTop: "-0.2rem", marginBottom: "0.7rem" }}>
                  Pair-level browse UI from selected Stage 1 sources. Click a row to see paths and metadata.
                </p>

                {!mitoleCatalogueGenerated ? (
                  <div className="catalog-empty-cell" style={{ border: "1px dashed var(--border)", borderRadius: 8, background: "#fff" }}>
                    Stage 2 catalog is empty by default. Click <strong>Regenerate database/data catalogue</strong> to build it.
                  </div>
                ) : (
                <div style={{ display: "grid", gridTemplateColumns: "minmax(16rem, 18rem) 1fr", gap: "1rem", alignItems: "start" }}>
                  {/* Left: Filters */}
                  <div className="catalog-filter-panel" style={{ border: "1px solid var(--border)", borderRadius: "8px", padding: "12px 12px 10px", background: "var(--panel)" }}>
                    <div className="catalog-filter-header">
                      <span>Filters</span>
                      {(mitoleCatQuery.trim() ||
                        mitoleCatSelectedSources.size > 0 ||
                        mitoleCatSelectedOrganisms.size > 0 ||
                        mitoleCatSelectedSampleTypes.size > 0) && (
                        <button
                          type="button"
                          className="linkish catalog-clear-btn"
                          onClick={() => {
                            setMitoleCatQuery("");
                            setMitoleCatSelectedSources(new Set());
                            setMitoleCatSelectedOrganisms(new Set());
                            setMitoleCatSelectedSampleTypes(new Set());
                            setMitoleCatSelectedPairKey(null);
                          }}
                        >
                          Clear all
                        </button>
                      )}
                    </div>

                    <div className="catalog-filter-section">
                      <div className="catalog-filter-section-title">Search</div>
                      <input
                        className="field-input studio-field"
                        value={mitoleCatQuery}
                        onChange={(e) => setMitoleCatQuery(e.target.value)}
                        placeholder="dataset / source / organism / sample type"
                        disabled={mitoleCatalogueLoading}
                      />
                    </div>

                    <div className="catalog-filter-section">
                      <div className="catalog-filter-section-title">Source</div>
                      {mitolePairCatalogueFilterOptions.sources.map((s) => {
                        const checked = mitoleCatSelectedSources.has(s);
                        return (
                          <label key={s} className="catalog-filter-item">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => {
                                setMitoleCatSelectedSources((prev) => {
                                  const next = new Set(prev);
                                  if (next.has(s)) next.delete(s);
                                  else next.add(s);
                                  return next;
                                });
                              }}
                              disabled={mitoleCatalogueLoading}
                            />
                            <span>{s}</span>
                          </label>
                        );
                      })}
                      {mitolePairCatalogueFilterOptions.sources.length === 0 && (
                        <div className="muted-note" style={{ fontSize: "0.82rem" }}>
                          No sources yet.
                        </div>
                      )}
                    </div>

                    <div className="catalog-filter-section">
                      <div className="catalog-filter-section-title">Organism</div>
                      {mitolePairCatalogueFilterOptions.organisms.map((s) => {
                        const checked = mitoleCatSelectedOrganisms.has(s);
                        return (
                          <label key={s} className="catalog-filter-item">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => {
                                setMitoleCatSelectedOrganisms((prev) => {
                                  const next = new Set(prev);
                                  if (next.has(s)) next.delete(s);
                                  else next.add(s);
                                  return next;
                                });
                              }}
                              disabled={mitoleCatalogueLoading}
                            />
                            <span>{s}</span>
                          </label>
                        );
                      })}
                    </div>

                    <div className="catalog-filter-section">
                      <div className="catalog-filter-section-title">Sample type</div>
                      {mitolePairCatalogueFilterOptions.sampleTypes.map((s) => {
                        const checked = mitoleCatSelectedSampleTypes.has(s);
                        return (
                          <label key={s} className="catalog-filter-item">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => {
                                setMitoleCatSelectedSampleTypes((prev) => {
                                  const next = new Set(prev);
                                  if (next.has(s)) next.delete(s);
                                  else next.add(s);
                                  return next;
                                });
                              }}
                              disabled={mitoleCatalogueLoading}
                            />
                            <span>{s}</span>
                          </label>
                        );
                      })}
                    </div>
                  </div>

                  {/* Right: List + detail */}
                  <div className="catalog-main">
                    <div className="catalog-list-area" style={{ maxHeight: mitoleSelectedPair ? "42%" : undefined }}>
                      <div className="catalog-list-head">
                        <span className="catalog-result-count">
                          {mitoleCatalogueLoading ? "Updating ..." : `${mitolePairCatalogueRowsFiltered.length} pair(s)`}
                          {(mitoleCatQuery.trim() ||
                            mitoleCatSelectedSources.size > 0 ||
                            mitoleCatSelectedOrganisms.size > 0 ||
                            mitoleCatSelectedSampleTypes.size > 0) &&
                            !mitoleCatalogueLoading && <span className="catalog-filter-active-note"> (filtered)</span>}
                        </span>
                        <StudioUpdatingBadge active={mitoleCatalogueLoading} label="Updating ..." />
                      </div>

                      <div className="catalog-table-wrap">
                        <table className="catalog-table">
                          <thead>
                            <tr>
                              <th>
                                <button type="button" className="studio-sort-btn" onClick={() => mitoleToggleCatSort("dataset")}>
                                  Dataset{mitoleCatSortIcon("dataset")}
                                </button>
                              </th>
                              <th>
                                <button type="button" className="studio-sort-btn" onClick={() => mitoleToggleCatSort("source")}>
                                  Source{mitoleCatSortIcon("source")}
                                </button>
                              </th>
                              <th>
                                <button type="button" className="studio-sort-btn" onClick={() => mitoleToggleCatSort("organism")}>
                                  Organism{mitoleCatSortIcon("organism")}
                                </button>
                              </th>
                              <th>
                                <button type="button" className="studio-sort-btn" onClick={() => mitoleToggleCatSort("sample_type")}>
                                  Sample type{mitoleCatSortIcon("sample_type")}
                                </button>
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {mitolePairCatalogueRowsFiltered.map((r) => {
                              const selected = mitoleCatSelectedPairKey === r.key;
                              return (
                                <tr
                                  key={r.key}
                                  className={`catalog-table-row${selected ? " catalog-row-selected" : ""}`}
                                  onClick={() => setMitoleCatSelectedPairKey(r.key)}
                                >
                                  <td className="catalog-col-name">
                                    <span className="catalog-dataset-name">{r.dataset}</span>
                                  </td>
                                  <td>{r.source}</td>
                                  <td className="catalog-col-organism">{r.organism}</td>
                                  <td>{r.sample_type}</td>
                                </tr>
                              );
                            })}
                            {mitolePairCatalogueRowsFiltered.length === 0 && !mitoleCatalogueLoading && (
                              <tr>
                                <td colSpan={4} className="catalog-empty-cell">
                                  No pairs match current filters.
                                </td>
                              </tr>
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {mitoleSelectedPair && (
                      <div className="catalog-detail-area">
                        <div className="catalog-detail-panel">
                          <div className="catalog-detail-header">
                            <strong className="catalog-detail-name">{mitoleSelectedPair.dataset}</strong>
                            <span className="catalog-badge catalog-badge-layer">{mitoleSelectedPair.source}</span>
                            <button className="catalog-detail-close ghost" onClick={() => setMitoleCatSelectedPairKey(null)} aria-label="Close detail">
                              ×
                            </button>
                          </div>

                          <div className="catalog-detail-card">
                            <div className="catalog-detail-card-title">Pair metadata</div>
                            <dl className="catalog-detail-dl">
                              <div className="catalog-detail-dl-row">
                                <dt>Organism</dt>
                                <dd>{mitoleSelectedPair.organism}</dd>
                              </div>
                              <div className="catalog-detail-dl-row">
                                <dt>Sample type</dt>
                                <dd>{mitoleSelectedPair.sample_type}</dd>
                              </div>
                              <div className="catalog-detail-dl-row">
                                <dt>Dimensions</dt>
                                <dd>{asCellVector(mitoleSelectedPair.dimensions)}</dd>
                              </div>
                              <div className="catalog-detail-dl-row">
                                <dt>Physical spacing</dt>
                                <dd>{asSpacingCell(mitoleSelectedPair.spacing)}</dd>
                              </div>
                            </dl>
                          </div>

                          <div className="catalog-detail-card">
                            <div className="catalog-detail-card-title">File paths</div>
                            <dl className="catalog-detail-dl">
                              <div className="catalog-detail-dl-row">
                                <dt>Image</dt>
                                <dd><code className="catalog-path-code">{mitoleSelectedPair.image_path}</code></dd>
                              </div>
                              <div className="catalog-detail-dl-row">
                                <dt>Label</dt>
                                <dd><code className="catalog-path-code">{mitoleSelectedPair.label_path}</code></dd>
                              </div>
                            </dl>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                )}
              </section>
            )}

            {hpcPage === "downloader" && (
              <section className="studio-dataset-viewer">
                <h3 className="studio-subhead">Stage 3: Data Downloader</h3>
                <label className="studio-label" htmlFor="studio-hpc-source-filter">
                  Source filter
                </label>
                <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
                  <select
                    id="studio-hpc-source-filter"
                    className="field-input studio-field"
                    style={{ maxWidth: "20rem" }}
                    value={mitoleFolderPick}
                    onChange={(e) => setMitoleFolderPick(e.target.value)}
                    disabled={dlGenerateBusy}
                  >
                    <option value="__all__">All selected folders</option>
                    {mitoleFolders.map((f) => <option key={f} value={f}>{f}</option>)}
                  </select>
                </div>
                <p className="muted-note" style={{ marginTop: "0.5rem", marginBottom: "0.35rem" }}>
                  Fixed download settings: <strong>16 nm</strong> isotropic voxels and <strong>128³</strong> crop size.
                </p>
                <details className="studio-run-log" style={{ marginTop: "0.2rem" }}>
                  <summary>Sample type (default: all)</summary>
                  <div style={{ marginTop: "0.45rem", display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
                    <button
                      type="button"
                      className="ghost"
                      onClick={() => setHpcDlSampleTypesSelected([...hpcDlSampleTypeOptions])}
                      disabled={hpcDlSampleTypeOptions.length === 0}
                    >
                      Select all
                    </button>
                    <button
                      type="button"
                      className="ghost"
                      onClick={() => setHpcDlSampleTypesSelected([])}
                      disabled={hpcDlSampleTypeOptions.length === 0}
                    >
                      Clear all
                    </button>
                  </div>
                  <div className="studio-run-pre" style={{ marginTop: "0.5rem", padding: "0.5rem 0.75rem" }}>
                    {hpcDlSampleTypeOptions.length === 0 ? (
                      <p className="muted-note">No sample types found in the Stage 2 catalogue yet.</p>
                    ) : (
                      hpcDlSampleTypeOptions.map((t) => {
                        const checked = hpcDlSampleTypesSelected.includes(t);
                        return (
                          <label key={t} style={{ display: "block", margin: "0.15rem 0" }}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setHpcDlSampleTypesSelected((prev) => Array.from(new Set([...prev, t])));
                                } else {
                                  setHpcDlSampleTypesSelected((prev) => prev.filter((x) => x !== t));
                                }
                              }}
                            />{" "}
                            <span>{t}</span>
                          </label>
                        );
                      })
                    )}
                  </div>
                </details>
                <details className="studio-run-log" style={{ marginTop: "0.7rem" }}>
                  <summary>Choose how many crops to download</summary>
                  <div className="studio-dl-dataset-list">
                    <div className="studio-run-pre" style={{ padding: "0.5rem 0.75rem" }}>
                      {mitoleCatalogueLoading ? (
                        <p className="muted-note">Loading dataset list…</p>
                      ) : activeDlFilteredPreviewDatasets.length === 0 ? (
                        <p className="muted-note">No datasets match the current sample type filter.</p>
                      ) : (
                        activeDlFilteredPreviewDatasets.slice(0, 250).map((ds) => (
                          <div
                            key={ds}
                            style={{
                              display: "grid",
                              gridTemplateColumns: "1fr auto",
                              gap: "0.75rem",
                              alignItems: "start",
                              padding: "0.2rem 0",
                            }}
                          >
                            <code>{ds}</code>
                            <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "nowrap", justifyContent: "flex-end" }}>
                              <label className="muted-note" htmlFor={`hpc-dl-train-${ds}`}>Train</label>
                              <input
                                id={`hpc-dl-train-${ds}`}
                                type="number"
                                min={0}
                                max={MAX_CROPS_PER_DATASET}
                                step={1}
                                className="field-input studio-field"
                                style={{ width: "5rem" }}
                                value={dlDatasetSplits[ds]?.training ?? 1}
                                onChange={(e) => {
                                  const requested = Number(e.target.value) || 0;
                                  setDlDatasetSplits((prev) => {
                                    const current = prev[ds] ?? { training: 1, inference: 0 };
                                    const training = clampDownloaderSplit("training", requested, current.inference);
                                    const inference = clampDownloaderSplit("inference", current.inference, training);
                                    return { ...prev, [ds]: { training, inference } };
                                  });
                                }}
                                disabled={dlGenerateBusy}
                              />
                              <label className="muted-note" htmlFor={`hpc-dl-infer-${ds}`} style={{ marginLeft: "0.35rem" }}>Inference</label>
                              <input
                                id={`hpc-dl-infer-${ds}`}
                                type="number"
                                min={0}
                                max={MAX_CROPS_PER_DATASET}
                                step={1}
                                className="field-input studio-field"
                                style={{ width: "5rem" }}
                                value={dlDatasetSplits[ds]?.inference ?? 0}
                                onChange={(e) => {
                                  const requested = Number(e.target.value) || 0;
                                  setDlDatasetSplits((prev) => {
                                    const current = prev[ds] ?? { training: 1, inference: 0 };
                                    const inference = clampDownloaderSplit("inference", requested, current.training);
                                    const training = clampDownloaderSplit("training", current.training, inference);
                                    return { ...prev, [ds]: { training, inference } };
                                  });
                                }}
                                disabled={dlGenerateBusy}
                              />
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </details>
                <div className="studio-actions" style={{ marginTop: "0.8rem" }}>
                  <button type="button" className="primary" disabled={dlUiBusy} onClick={() => void runDownloaderGenerate()}>
                    <SizeStableLabel label="Download" busyLabel="Downloading…" isBusy={dlUiBusy} />
                  </button>
                  <button type="button" className="danger" disabled={!dlUiBusy} onClick={() => void killDownloaderRun()}>
                    Kill download
                  </button>
                </div>
                <h3 id="studio-hpc-dl-run-heading" className="studio-scrape-section-title" style={{ marginTop: "0.75rem" }}>
                  Download progress
                </h3>
                <div style={{ marginTop: "0.75rem", marginBottom: "0.75rem" }}>
                  <div className="muted-note" style={{ marginBottom: "0.25rem" }}>
                    {dlProgress
                      ? `Progress: ${dlProgress.completed}/${dlProgress.total} new crop pair(s)${
                          dlUiBusy ? ` (running pair ${dlProgress.active})` : ""
                        }`
                      : "Progress: waiting to start"}
                    {` — status: ${dlStateLabel}`}
                  </div>
                  <div
                    style={{
                      width: "100%",
                      height: "12px",
                      borderRadius: "999px",
                      background: "rgba(17, 24, 39, 0.16)",
                      border: "1px solid rgba(17, 24, 39, 0.25)",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${dlProgress ? Math.max(dlProgressPercent, dlUiBusy ? 3 : 0) : 0}%`,
                        height: "100%",
                        background: dlShowFailureStyle
                          ? "linear-gradient(90deg, #ff6b6b 0%, #ff9d6b 100%)"
                          : "linear-gradient(90deg, #2ec5ff 0%, #23d4a0 100%)",
                        transition: "width 180ms ease",
                      }}
                    />
                  </div>
                  <div className="muted-note" style={{ marginTop: "0.2rem" }}>
                    {dlProgress ? `${dlProgressPercent}%` : "0%"}
                  </div>
                </div>
                <div className="studio-scrape-live-log-wrap">
                  <div className="studio-scrape-live-log-label" style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}>
                    <span>Downloader output</span>
                    <button type="button" className="ghost" disabled={dlUiBusy} onClick={() => void clearDownloaderOutput()}>
                      Clear
                    </button>
                  </div>
                  <pre className="studio-scrape-live-log" aria-live="polite">
                    {dlRunLog || "— Click Download to see live progress here —"}
                  </pre>
                </div>
              </section>
            )}
          </div>
        )}

        {/* ── Existing Data Viewer (moved as legacy page) ── */}
        {appView === "data-existing-legacy" && (
          <div className="ia-section">
            <h1 className="ia-section-title">Downloaded & Predicted Data</h1>
            <p className="muted-note">
              Existing datasets from{" "}
              <code>{REL_NNUNET_DATASET}</code>{" "}
              (<code>imagesTr/labelsTr/labelsTr-instance</code> for training, <code>imagesTs/labelsTs/labelsTs-instance</code> for inference/testing).
              Select a dataset to use in Training or Inference.
            </p>
            {selectedDataset && (
              <div className="ia-selection-banner ia-selection-banner--ok">
                <span>
                  Selected for Model: <code>{selectedDataset}</code>
                </span>
                <div className="ia-selection-banner-actions">
                  <button type="button" className="primary" onClick={() => setAppView("model-training")}>
                    Go to Model
                  </button>
                  <button type="button" className="ghost" onClick={() => setSelectedDataset(null)}>
                    Clear selection
                  </button>
                </div>
              </div>
            )}
            <section
              className={`studio-dataset-viewer${dataInspectLoading ? " studio-surface-updating" : ""}`}
              aria-busy={dataInspectLoading}
            >
              <div className="ia-segment-bar ia-segment-bar--inner" role="tablist" aria-label="Existing data table">
                <button
                  type="button"
                  role="tab"
                  className={`ia-segment-tab${existingDataTab === "training" ? " ia-segment-tab--active" : ""}`}
                  aria-selected={existingDataTab === "training"}
                  onClick={() => setExistingDataTab("training")}
                >
                  Training
                </button>
                <button
                  type="button"
                  role="tab"
                  className={`ia-segment-tab${existingDataTab === "predicted" ? " ia-segment-tab--active" : ""}`}
                  aria-selected={existingDataTab === "predicted"}
                  onClick={() => setExistingDataTab("predicted")}
                >
                  Predicted
                </button>
              </div>
              <div className="studio-dataset-viewer-topbar">
                <div className="studio-dataset-summary">
                  <span>
                    {existingDataTab === "training" ? (
                      <>
                        Datasets shown:{" "}
                        <strong>{existingDataRowsFiltered.length}</strong>
                        <span className="muted-note" style={{ marginLeft: "0.4rem" }}>
                          (total: {existingDataRows.length})
                        </span>
                      </>
                    ) : (
                      <>
                        Predicted files shown: <strong>{postprocessTableRowsFiltered.length}</strong>
                        <span className="muted-note" style={{ marginLeft: "0.4rem" }}>
                          (total: {postprocessTableRows.length})
                        </span>
                      </>
                    )}
                  </span>
                  <StudioUpdatingBadge
                    active={existingDataTab === "training" ? (dataInspectLoading || dsStatusLoading) : postprocessFilesLoading}
                    label="Refreshing…"
                  />
                  <button
                    type="button"
                    className="ghost"
                    disabled={existingDataTab === "training" ? (dataInspectLoading || dsActionBusy) : postprocessFilesLoading}
                    onClick={() => {
                      if (existingDataTab === "training") {
                        void refreshDataInspect(true, false);
                        setDsStatusLoading(true);
                        getDatasetsStatus()
                          .then((r) => setDsStatus(r.datasets))
                          .catch(() => setDsStatus(null))
                          .finally(() => setDsStatusLoading(false));
                      } else {
                        void refreshPostprocessingFiles(true);
                      }
                    }}
                  >
                    Refresh
                  </button>
                </div>
              </div>
              {existingDataTab === "training" ? (
                <>
                  <div style={{ display: "flex", gap: "0.45rem", alignItems: "center", marginTop: "0.35rem", marginBottom: "0.35rem" }}>
                    <span className="muted-note">Source:</span>
                    <button
                      type="button"
                      className={existingSourceFilter === "all" ? "primary" : "ghost"}
                      onClick={() => setExistingSourceFilter("all")}
                    >
                      All
                    </button>
                    <button
                      type="button"
                      className={existingSourceFilter === "training" ? "primary" : "ghost"}
                      onClick={() => setExistingSourceFilter("training")}
                    >
                      Training
                    </button>
                    <button
                      type="button"
                      className={existingSourceFilter === "inference" ? "primary" : "ghost"}
                      onClick={() => setExistingSourceFilter("inference")}
                    >
                      Inference
                    </button>
                    <button
                      type="button"
                      className={existingSourceFilter === "instance" ? "primary" : "ghost"}
                      onClick={() => setExistingSourceFilter("instance")}
                    >
                      Instance
                    </button>
                  </div>
                  <p className="muted-note" style={{ marginTop: "0.4rem" }}>
                    Source roots:{" "}
                    <code>{REL_NNUNET_DATASET}/imagesTr</code>,{" "}
                    <code>labelsTr</code>, <code>labelsTr-instance</code>, <code>imagesTs</code>, <code>labelsTs</code>, and <code>labelsTs-instance</code>.
                    Training/Inference filters only show model-usable data; Instance shows <code>labelsTr-instance</code>/<code>labelsTs-instance</code> only.
                    Delete works for either source. Use in Model is left empty for Instance rows.
                  </p>
                  {dsSelectedRowKeys.size > 0 && (
                    <div className="studio-ds-action-bar" style={{ display: "flex", gap: "0.5rem", alignItems: "center", margin: "0.5rem 0", flexWrap: "wrap" }}>
                      <span className="muted-note">{dsSelectedRowKeys.size} selected</span>
                      <button
                        type="button"
                        className="danger"
                        disabled={dsActionBusy}
                        onClick={async () => {
                          setDsActionBusy(true);
                          try {
                            const paths = [...new Set(existingDataRowsFiltered
                              .filter((r) => dsSelectedRowKeys.has(`${r.source}:${r.name}:${r.path}`))
                              .map((r) => r.path)
                              .filter((p) => String(p || "").trim().length > 0))];
                            const r = await deleteDatasetFiles(paths);
                            onNotify(r.ok ? "Deleted" : "Delete errors", r.message, r.ok ? "ok" : "err");
                            setDsSelectedRowKeys(new Set());
                            void refreshDataInspect(true, true);
                            void refreshInventoryCatalogue();
                            setDsStatusLoading(true);
                            getDatasetsStatus().then((r2) => setDsStatus(r2.datasets)).catch(() => setDsStatus(null)).finally(() => setDsStatusLoading(false));
                          } catch (e) {
                            onNotify("Delete failed", e instanceof Error ? e.message : String(e), "err");
                          } finally {
                            setDsActionBusy(false);
                          }
                        }}
                      >
                        Delete selected
                      </button>
                      <button
                        type="button"
                        className="ghost"
                        disabled={dsActionBusy}
                        onClick={async () => {
                          if (selectedTrainingDatasetNames.length === 0) return;
                          setDsActionBusy(true);
                          try {
                            await unhideDatasets(selectedTrainingDatasetNames);
                            setDsStatusLoading(true);
                            getDatasetsStatus().then((r2) => setDsStatus(r2.datasets)).catch(() => setDsStatus(null)).finally(() => setDsStatusLoading(false));
                          } catch (err) {
                            onNotify("Use in model update failed", err instanceof Error ? err.message : String(err), "err");
                          } finally {
                            setDsActionBusy(false);
                          }
                        }}
                      >
                        Use in model: On
                      </button>
                      <button
                        type="button"
                        className="ghost"
                        disabled={dsActionBusy}
                        onClick={async () => {
                          if (selectedTrainingDatasetNames.length === 0) return;
                          setDsActionBusy(true);
                          try {
                            await hideDatasets(selectedTrainingDatasetNames);
                            setDsStatusLoading(true);
                            getDatasetsStatus().then((r2) => setDsStatus(r2.datasets)).catch(() => setDsStatus(null)).finally(() => setDsStatusLoading(false));
                          } catch (err) {
                            onNotify("Use in model update failed", err instanceof Error ? err.message : String(err), "err");
                          } finally {
                            setDsActionBusy(false);
                          }
                        }}
                      >
                        Use in model: Off
                      </button>
                    </div>
                  )}

                  <div className="studio-dataset-table-wrap">
                    <table className="studio-dataset-table">
                      <thead>
                        <tr>
                          <th style={{ width: "2rem" }}>
                            <input
                              type="checkbox"
                              title="Select all rows"
                              checked={existingDataRowsFiltered.length > 0 && dsSelectedRowKeys.size === existingDataRowsFiltered.length}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setDsSelectedRowKeys(new Set(existingDataRowsFiltered.map((r) => `${r.source}:${r.name}:${r.path}`)));
                                } else {
                                  setDsSelectedRowKeys(new Set());
                                }
                              }}
                            />
                          </th>
                          <th>Delete</th>
                          <th>#</th>
                          <th>Name</th>
                          <th>Source</th>
                          <th>Type</th>
                          <th>Dimensions</th>
                          <th>Physical spacing</th>
                          <th title="File role">Data</th>
                          <th title="Distinct nonzero label IDs (label files only)">Distinct Labels</th>
                          <th>Use in Model</th>
                        </tr>
                      </thead>
                      <tbody>
                        {existingDataRowsFiltered.length > 0 ? (
                          existingDataRowsFiltered.map((row) => {
                        const isTrainingRow = row.source === "training";
                        const rowSelKey = `${row.source}:${row.name}:${row.path}`;
                        const stableId = toStableDatasetId(row.name);
                        const regInfo = dsStatus?.find((d) => d.stable_id === stableId || d.filename === row.name || d.filename.startsWith(stableId));
                        const hidden = isTrainingRow
                          ? (regInfo?.hidden_from_training ?? false)
                          : (regInfo?.hidden_from_inference ?? false);
                        const canUseInModel = row.source === "instance" ? false : (isTrainingRow ? !row.isInstanceLabel : true);
                        return (
                          <tr
                            key={row.path}
                            className={[
                              selectedInferenceRowKeys.has(rowSelKey) ? "ia-dataset-row-selected" : "",
                              hidden ? "studio-ds-row-hidden" : "",
                            ].filter(Boolean).join(" ")}
                            style={hidden ? { opacity: 0.55 } : undefined}
                          >
                            <td>
                              <input
                                type="checkbox"
                                checked={dsSelectedRowKeys.has(rowSelKey)}
                                onChange={(e) => {
                                  setDsSelectedRowKeys((prev) => {
                                    const next = new Set(prev);
                                    if (e.target.checked) next.add(rowSelKey);
                                    else next.delete(rowSelKey);
                                    return next;
                                  });
                                }}
                              />
                            </td>
                            <td>
                              <button
                                type="button"
                                className="linkish"
                                style={{ color: "var(--color-danger, #b42318)" }}
                                disabled={dsActionBusy}
                                onClick={async () => {
                                  setDsActionBusy(true);
                                  try {
                                    const r = await deleteDatasetFiles([row.path]);
                                    onNotify(r.ok ? "Deleted" : "Delete errors", r.message, r.ok ? "ok" : "err");
                                    void refreshDataInspect(true, true);
                                    void refreshInventoryCatalogue();
                                    setDsStatusLoading(true);
                                    getDatasetsStatus()
                                      .then((r2) => setDsStatus(r2.datasets))
                                      .catch(() => setDsStatus(null))
                                      .finally(() => setDsStatusLoading(false));
                                  } catch (e) {
                                    onNotify("Delete failed", e instanceof Error ? e.message : String(e), "err");
                                  } finally {
                                    setDsActionBusy(false);
                                  }
                                }}
                              >
                                Delete
                              </button>
                            </td>
                            <td>{row.order}</td>
                            <td>{row.name}</td>
                            <td>{row.source}</td>
                            <td>{row.type || "—"}</td>
                            <td>{asCellVector(row.dimensions)}</td>
                            <td>{asSpacingCell(row.spacing)}</td>
                            <td>{row.dataKind}</td>
                            <td>{row.dataKind === "label" ? asLabelSummaryCell(row) : "—"}</td>
                            <td>
                              {row.source === "instance" ? null : (
                                isTrainingRow && !canUseInModel ? (
                                  <span className="muted-note">—</span>
                                ) : (
                                  <label
                                    className="studio-model-switch"
                                    title={isTrainingRow
                                      ? "Toggle dataset visibility for model training"
                                      : "Select this inference file for model inference"}
                                  >
                                    <input
                                      type="checkbox"
                                      checked={isTrainingRow ? !hidden : selectedInferenceRowKeys.has(rowSelKey)}
                                      disabled={dsActionBusy}
                                      onChange={async (e) => {
                                        const targetOn = e.target.checked;
                                        setDsActionBusy(true);
                                        try {
                                          if (!isTrainingRow) {
                                            await setDatasetUseInModel(stableId, "inference", targetOn);
                                          } else if (targetOn) {
                                            await unhideDatasets([stableId]);
                                          } else {
                                            await hideDatasets([stableId]);
                                          }
                                          setDsStatusLoading(true);
                                          getDatasetsStatus()
                                            .then((r2) => setDsStatus(r2.datasets))
                                            .catch(() => setDsStatus(null))
                                            .finally(() => setDsStatusLoading(false));
                                        } catch (err) {
                                          onNotify("Use in model update failed", err instanceof Error ? err.message : String(err), "err");
                                        } finally {
                                          setDsActionBusy(false);
                                        }
                                      }}
                                    />
                                    <span className="studio-model-switch__slider" aria-hidden="true" />
                                    <span className="studio-model-switch__label">
                                      {(isTrainingRow ? !hidden : selectedInferenceRowKeys.has(rowSelKey)) ? "On" : "Off"}
                                    </span>
                                  </label>
                                )
                              )}
                            </td>
                          </tr>
                        );
                          })
                        ) : (
                          <tr>
                            <td colSpan={11} className="studio-dataset-placeholder-cell">
                              {dataInspectLoading ? (
                                <>Updating ...</>
                              ) : dataInspect ? (
                                <>
                                  No datasets found under <code>Dataset001_mito2/imagesTr|labelsTr|imagesTs|labelsTs</code>.{" "}
                                  <button
                                    type="button"
                                    className="linkish"
                                    onClick={() => {
                                      setPage("downloader");
                                      setAppView("data-scrape");
                                    }}
                                  >
                                    Run Stage 3 Data Downloader
                                  </button>{" "}
                                  to generate data for training.
                                </>
                              ) : (
                                <>Dataset metadata not loaded — check backend connection.</>
                              )}
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <>
                  <div style={{ display: "flex", gap: "0.45rem", alignItems: "center", marginTop: "0.35rem", marginBottom: "0.35rem" }}>
                    <span className="muted-note">Source:</span>
                    <button
                      type="button"
                      className={predictedSourceFilter === "all" ? "primary" : "ghost"}
                      onClick={() => setPredictedSourceFilter("all")}
                    >
                      All
                    </button>
                    <button
                      type="button"
                      className={predictedSourceFilter === "input" ? "primary" : "ghost"}
                      onClick={() => setPredictedSourceFilter("input")}
                    >
                      Border-contour
                    </button>
                    <button
                      type="button"
                      className={predictedSourceFilter === "output" ? "primary" : "ghost"}
                      onClick={() => setPredictedSourceFilter("output")}
                    >
                      Postprocessed
                    </button>
                  </div>
                  <p className="muted-note" style={{ marginTop: "0.4rem" }}>
                    Predicted source roots: <code>{REL_OUTPUTS_BC}</code> and{" "}
                    <code>{REL_OUTPUTS_POSTPROCESSED}</code>.
                  </p>
                  {postprocessSelectedRowKeys.size > 0 && (
                    <div className="studio-ds-action-bar" style={{ display: "flex", gap: "0.5rem", alignItems: "center", margin: "0.5rem 0", flexWrap: "wrap" }}>
                      <span className="muted-note">{postprocessSelectedRowKeys.size} selected</span>
                      <button
                        type="button"
                        className="danger"
                        disabled={dsActionBusy}
                        onClick={async () => {
                          setDsActionBusy(true);
                          try {
                            const paths = [...new Set(postprocessTableRowsFiltered
                              .filter((r) => postprocessSelectedRowKeys.has(`${r.source}:${r.path}`))
                              .map((r) => r.path)
                              .filter((p) => String(p || "").trim().length > 0))];
                            const r = await deleteDatasetFiles(paths);
                            onNotify(r.ok ? "Deleted" : "Delete errors", r.message, r.ok ? "ok" : "err");
                            setPostprocessSelectedRowKeys(new Set());
                            void refreshPostprocessingFiles(true);
                          } catch (e) {
                            onNotify("Delete failed", e instanceof Error ? e.message : String(e), "err");
                          } finally {
                            setDsActionBusy(false);
                          }
                        }}
                      >
                        Delete selected
                      </button>
                    </div>
                  )}
                  <div className="studio-dataset-table-wrap">
                    <table className="studio-dataset-table">
                      <thead>
                        <tr>
                          <th style={{ width: "2rem" }}>
                            <input
                              type="checkbox"
                              title="Select all rows"
                              checked={postprocessTableRowsFiltered.length > 0 && postprocessSelectedRowKeys.size === postprocessTableRowsFiltered.length}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setPostprocessSelectedRowKeys(new Set(postprocessTableRowsFiltered.map((r) => `${r.source}:${r.path}`)));
                                } else {
                                  setPostprocessSelectedRowKeys(new Set());
                                }
                              }}
                            />
                          </th>
                          <th>Delete</th>
                          <th>#</th>
                          <th>Name</th>
                          <th>Source</th>
                          <th>Type</th>
                          <th>Dimensions</th>
                          <th>Physical spacing</th>
                          <th title="File role">Data</th>
                          <th>Distinct Labels</th>
                        </tr>
                      </thead>
                      <tbody>
                        {postprocessTableRowsFiltered.length > 0 ? (
                          postprocessTableRowsFiltered.map((row) => {
                            const rowKey = `${row.source}:${row.path}`;
                            const checked = postprocessSelectedRowKeys.has(rowKey);
                            return (
                              <tr key={rowKey}>
                                <td>
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={(e) => {
                                      setPostprocessSelectedRowKeys((prev) => {
                                        const next = new Set(prev);
                                        if (e.target.checked) next.add(rowKey);
                                        else next.delete(rowKey);
                                        return next;
                                      });
                                    }}
                                  />
                                </td>
                                <td>
                                  <button
                                    type="button"
                                    className="linkish"
                                    style={{ color: "var(--color-danger, #b42318)" }}
                                    disabled={dsActionBusy}
                                    onClick={async () => {
                                      setDsActionBusy(true);
                                      try {
                                        const r = await deleteDatasetFiles([row.path]);
                                        onNotify(r.ok ? "Deleted" : "Delete errors", r.message, r.ok ? "ok" : "err");
                                        setPostprocessSelectedRowKeys((prev) => {
                                          const next = new Set(prev);
                                          next.delete(rowKey);
                                          return next;
                                        });
                                        void refreshPostprocessingFiles(true);
                                      } catch (e) {
                                        onNotify("Delete failed", e instanceof Error ? e.message : String(e), "err");
                                      } finally {
                                        setDsActionBusy(false);
                                      }
                                    }}
                                  >
                                    Delete
                                  </button>
                                </td>
                                <td>{row.order}</td>
                                <td style={{ fontFamily: "monospace", fontSize: "0.85em" }}>{row.name}</td>
                                <td>{row.source === "input" ? "border-contour" : "postprocessed"}</td>
                                <td>{row.type || "—"}</td>
                                <td>{asCellVector(row.dimensions)}</td>
                                <td>{asSpacingCell(row.spacing)}</td>
                                <td>label</td>
                                <td>{asLabelSummaryCell(row)}</td>
                              </tr>
                            );
                          })
                        ) : (
                          <tr>
                            <td colSpan={10} className="studio-dataset-placeholder-cell">
                              {postprocessFilesLoading ? "Updating ..." : "No predicted NIfTI files found for the selected source filter."}
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </section>
          </div>
        )}

        {/* ── Pipeline Summary ── */}
        {appView === "pipeline-summary" && (
          <>
            <h1 className="ia-section-title">Pipeline Summary</h1>
            <p className="muted-note">Pipeline state and on-disk artifact checks.</p>
            <div className="studio-actions">
              <button type="button" className="ghost" disabled={summaryBusy} onClick={() => void refreshSummary()}>
                <SizeStableLabel label="Refresh" busyLabel="Refreshing…" isBusy={summaryBusy} />
              </button>
              <StudioUpdatingBadge active={summaryBusy} label="Updating ..." />
            </div>
            {summary && (
              <>
                <h3 className="studio-summary-section-head">Pipeline session</h3>
                <dl className="studio-summary-dl">
                  <dt>Current step</dt>
                  <dd>
                    <code>{summary.pipeline.step_label === "preprocess" ? "download_script" : summary.pipeline.step_label}</code>{" "}
                    (#{summary.pipeline.step_label === "preprocess" ? 3 : summary.pipeline.current_step})
                  </dd>
                  <dt>Last URL</dt>
                  <dd>{summary.pipeline.last_url || "—"}</dd>
                  <dt>Site stem</dt>
                  <dd>{summary.pipeline.last_site_stem || "—"}</dd>
                </dl>

                <h3 className="studio-summary-section-head">Stage outputs</h3>
                <dl className="studio-summary-dl">
                  <dt>Stage 1 — Probe Files</dt>
                  <dd>
                    {summary.probe_count} file(s)
                    {summary.latest_probe ? <> · latest: <code>{summary.latest_probe}</code></> : null}
                  </dd>
                  <dt>Stage 2 — Catalog DB</dt>
                  <dd>
                    {summary.catalog_db_labeled_ready > 0
                      ? <><strong>{summary.catalog_db_labeled_ready}</strong> labeled-ready dataset(s) in catalog DB</>
                      : "not built yet"}
                  </dd>
                  <dt>Stage 2 — Inventory Archive</dt>
                  <dd>
                    {summary.inventory_sqlite_exists
                      ? <><code>data/inventory.sqlite</code> — present</>
                      : "not created yet"}
                  </dd>
                  <dt>Stage 3 — Download Scripts</dt>
                  <dd>
                    {summary.generated_download_scripts} generated script(s) in{" "}
                    <code>3data_downloader/outputs/</code>
                  </dd>
                  <dt>Stage 3 — Preprocessed Data</dt>
                  <dd>{summary.preprocessed_dir_exists ? "present in Dataset001_mito2/" : "empty / not yet run"}</dd>
                  <dt>Stage 3 — Training Config</dt>
                  <dd>
                    {summary.training_config_exists
                      ? <><code>finetune_datalist.json</code> — ready for training configuration</>
                      : "not yet written — run stage 3 first"}
                  </dd>
                </dl>

                <h3 className="studio-summary-section-head">Registry <code>data/registry.sqlite</code></h3>
                <dl className="studio-summary-dl">
                  {summary.registry.exists ? (
                    summary.registry.error ? (
                      <>
                        <dt>Status</dt>
                        <dd className="studio-summary-warn">Error reading registry: {summary.registry.error}</dd>
                      </>
                    ) : (
                      <>
                        <dt>Providers</dt>
                        <dd>{summary.registry.providers}</dd>
                        <dt>Datasets</dt>
                        <dd>{summary.registry.datasets}</dd>
                        <dt>Assets</dt>
                        <dd>{summary.registry.assets}</dd>
                        <dt>Complete downloads</dt>
                        <dd>{summary.registry.complete_downloads}</dd>
                        <dt>Complete preprocess runs</dt>
                        <dd>{summary.registry.complete_preprocess_runs}</dd>
                      </>
                    )
                  ) : (
                    <>
                      <dt>Status</dt>
                      <dd className="studio-summary-warn">Not created — run stage 2 to populate</dd>
                    </>
                  )}
                </dl>

                <h3 className="studio-summary-section-head">Key output paths</h3>
                <table className="studio-intro-outputs-table">
                  <tbody>
                    <tr><td><code>1web_scraper_01/outputs/&lt;provider&gt;.probe.json</code></td><td>Per-dataset discovery registry</td></tr>
                    <tr><td><code>2database_builder/outputs/databases/&lt;provider&gt;.db</code></td><td>Catalog DB with resolved EM + mito seg paths</td></tr>
                    <tr><td><code>data/registry.sqlite</code></td><td>Incremental registry — providers, assets, download + preprocess state</td></tr>
                    <tr><td><code>data/inventory.sqlite</code></td><td>Probe archive (per-scrape raw mirror)</td></tr>
                    <tr><td><code>3data_downloader/outputs/download_&lt;provider&gt;_labeled.py</code></td><td>Generated download script (good mito labels only)</td></tr>
                    <tr><td><code>data/raw/&lt;provider&gt;_mito_*/</code></td><td>Raw HDF5 EM + label crops</td></tr>
                    <tr><td><code>data/nnUNet_raw/Dataset001_mito2/</code></td><td>nnUNet-ready NIfTI volumes (imagesTr/labelsTr/imagesTs/labelsTs)</td></tr>
                  </tbody>
                </table>
              </>
            )}
          </>
        )}

        {/* ── Stage 0: Inventory ── */}
        {appView === "data-scrape" && page === "inventory" && (
          <div className="studio-actions" style={{ justifyContent: "flex-end", marginBottom: "0.5rem" }}>
            <button
              type="button"
              className="danger"
              disabled={resetDownloadedBusy || catalogueLoading || scrapeUiBusy || databaseUiBusy || dlUiBusy || preUiBusy || preKillBusy || trainingBusy}
              onClick={() => void resetDownloadedTrainingAndHistory()}
              title="Delete Dataset001_mito2 train/test contents and clear registry download/preprocess history"
            >
              <SizeStableLabel
                label="Reset downloaded data + history"
                busyLabel="Resetting…"
                isBusy={resetDownloadedBusy}
              />
            </button>
          </div>
        )}
        {appView === "data-scrape" && page === "inventory" && (
          <InventoryPage
            catalogue={catalogue}
            catalogueLoading={catalogueLoading}
            catFilterStatus={catFilterStatus}
            setCatFilterStatus={setCatFilterStatus}
            catFilterBatch={catFilterBatch}
            setCatFilterBatch={setCatFilterBatch}
            catShowMissingOnly={catShowMissingOnly}
            setCatShowMissingOnly={setCatShowMissingOnly}
            catSortCol={catSortCol}
            setCatSortCol={setCatSortCol}
            catSortAsc={catSortAsc}
            setCatSortAsc={setCatSortAsc}
            onRefresh={refreshInventoryCatalogue}
          />
        )}

        {appView === "data-scrape" && page === "scraper" && (
          <>
            <h2 className="studio-page-title">Web Scraper (Stage 1)</h2>
            <p className="muted-note">
              Save a site profile, then run scrape to refresh probe files for Stage 2.
            </p>

            <section className="studio-scrape-section" aria-labelledby="studio-scrape-profile-heading">
              <div className="studio-scrape-section-card">
                <h3 id="studio-scrape-profile-heading" className="studio-scrape-section-title">
                  1. Site profile
                </h3>
                <p className="muted-note studio-scrape-section-lead">
                  Save profile before scraping. Use overwrite only when updating the currently loaded folder.
                </p>
                <label className="studio-label" htmlFor="ws-pick">
                  Load site to edit
                </label>
                <div className="studio-row-inline">
                  <select
                    id="ws-pick"
                    className="field-input studio-field studio-field-grow"
                    value={wsPickSelectValue}
                    onChange={(e) => {
                      const s = e.target.value;
                      if (s === STUDIO_SELECT_LOADING) return;
                      setWsPickSlug(s);
                      if (s) void loadWebsiteProfile(s);
                    }}
                    disabled={wsListLoading}
                  >
                    {wsListLoading ? (
                      <option value={STUDIO_SELECT_LOADING}>Updating ...</option>
                    ) : (
                      <>
                        <option value="">— New site —</option>
                        {wsList.map((w) => (
                          <option key={w.slug} value={w.slug}>
                            {w.display_name} ({w.slug})
                          </option>
                        ))}
                      </>
                    )}
                  </select>
                  <StudioUpdatingBadge active={wsListLoading} label="Updating ..." />
                  <button
                    type="button"
                    className="ghost"
                    disabled={!wsPickSlug || wsListLoading}
                    onClick={() => void loadWebsiteProfile(wsPickSlug)}
                  >
                    Reload
                  </button>
                </div>

                <label className="studio-label" htmlFor="ws-name">
                  Website name
                </label>
                <input
                  id="ws-name"
                  className="field-input studio-field"
                  value={wsDisplayName}
                  onChange={(e) => setWsDisplayName(e.target.value)}
                  placeholder="e.g. Open Organelle"
                  autoComplete="off"
                />

                <label className="studio-label" htmlFor="ws-url">
                  URL
                </label>
                <input
                  id="ws-url"
                  className="field-input studio-field"
                  value={wsUrl}
                  onChange={(e) => setWsUrl(e.target.value)}
                  placeholder="http://…"
                  autoComplete="off"
                />

                <MarkdownField
                  id="ws-desc"
                  label="Brief description"
                  value={wsDescription}
                  onChange={setWsDescription}
                  rows={5}
                  resetKey={wsPickSlug || "new"}
                  placeholder="What this portal is; optional context for future LLM triage."
                />

                <MarkdownField
                  id="ws-focus"
                  label="What to look for (datasets / data types)"
                  value={wsDataFocus}
                  onChange={setWsDataFocus}
                  rows={6}
                  resetKey={wsPickSlug || "new"}
                  placeholder="Generic: listings, APIs, file formats, collections. Do not limit to mitochondria here."
                />

                <label className="studio-label" htmlFor="ws-slug">
                  Folder slug (optional)
                </label>
                <input
                  id="ws-slug"
                  className="field-input studio-field"
                  value={wsSlugOverride}
                  onChange={(e) => setWsSlugOverride(e.target.value)}
                  placeholder="Optional base hint (e.g. mysite_01 → base mysite); Save picks next _NN"
                  autoComplete="off"
                />

                <div className="studio-overwrite-option">
                  <label htmlFor="ws-save-overwrite">
                    <input
                      id="ws-save-overwrite"
                      type="checkbox"
                      checked={wsSaveOverwrite}
                      onChange={(e) => setWsSaveOverwrite(e.target.checked)}
                      disabled={!wsPickSlug}
                    />
                    <span className="studio-overwrite-copy">
                      <span className="studio-overwrite-title">Overwrite loaded folder</span>
                      <span className="studio-overwrite-hint">
                        Leave unchecked to create the next versioned folder (<code>…_02</code>, <code>…_03</code>, …).
                        Check only to update the site selected under <strong>Load site to edit</strong> without creating a
                        new folder.
                      </span>
                    </span>
                  </label>
                </div>

                <div className="studio-actions">
                  <button type="button" className="primary" disabled={wsSaveBusy} onClick={() => void runWorkspaceSave()}>
                    <SizeStableLabel label="Save profile" busyLabel="Saving…" isBusy={wsSaveBusy} />
                  </button>
                </div>

                <div className="studio-delete-panel">
                  <label className="studio-label studio-label-spaced" htmlFor="ws-delete-pick">
                    Remove workspace
                  </label>
                  <p className="muted-note studio-scrape-section-lead studio-delete-hint">
                    Permanently deletes the folder under <code>1web_scraper_01/websites/</code> and matching{" "}
                    <code>outputs/&lt;slug&gt;.probe.json</code>. Built-in workspaces <code>bossdb_01</code> and{" "}
                    <code>openorganelle_01</code> cannot be removed here. Click <strong>Delete</strong> to arm, then{" "}
                    <strong>Confirm delete</strong> or <strong>Cancel</strong> (embedded browsers often block the old browser
                    confirm dialog).
                  </p>
                  {deletePanelArmed && deletePickSlug ? (
                    <p className="studio-delete-arm-hint" role="status">
                      Ready to remove <code>{deletePickSlug}</code>. Choose <strong>Confirm delete</strong> or{" "}
                      <strong>Cancel</strong>.
                    </p>
                  ) : null}
                  <div className="studio-row-inline studio-delete-actions">
                    <select
                      id="ws-delete-pick"
                      className="field-input studio-field studio-field-grow"
                      value={deletePickSelectValue}
                      onChange={(e) => {
                        const v = e.target.value;
                        if (v === STUDIO_SELECT_LOADING) return;
                        setDeletePickSlug(v);
                      }}
                      disabled={wsListDeletable.length === 0 || wsDeleteBusy || wsListLoading}
                    >
                      {wsListLoading ? (
                        <option value={STUDIO_SELECT_LOADING}>Updating ...</option>
                      ) : wsListDeletable.length === 0 ? (
                        <option value="">No deletable sites (built-in workspaces are protected)</option>
                      ) : (
                        <>
                          <option value="">— Select site to delete —</option>
                          {wsListDeletable.map((w) => (
                            <option key={w.slug} value={w.slug}>
                              {w.display_name} ({w.slug})
                            </option>
                          ))}
                        </>
                      )}
                    </select>
                    <button
                      type="button"
                      className={deletePanelArmed ? "danger studio-delete-confirm-btn" : "danger"}
                      disabled={wsDeleteBusy || !deletePickSlug}
                      onClick={() => void runWorkspaceDelete()}
                    >
                      {wsDeleteBusy ? "Deleting…" : deletePanelArmed ? "Confirm delete" : "Delete"}
                    </button>
                    {deletePanelArmed ? (
                      <button
                        type="button"
                        className="ghost"
                        disabled={wsDeleteBusy}
                        onClick={() => setDeletePanelArmed(false)}
                      >
                        Cancel
                      </button>
                    ) : null}
                  </div>
                </div>
              </div>
            </section>

            <section className="studio-scrape-section" aria-labelledby="studio-scrape-run-heading">
              <div className="studio-scrape-section-card">
                <h3 id="studio-scrape-run-heading" className="studio-scrape-section-title">
                  2. Run scrape
                </h3>
                <p className="muted-note studio-scrape-section-lead">
                  Uses the saved profile fields from <code>site.md</code>.
                </p>
                <label className="studio-label" htmlFor="ws-scrape-target">
                  Website to scrape
                </label>
                <div className="studio-field-row-updating">
                  <select
                    id="ws-scrape-target"
                    className="field-input studio-field"
                    value={scrapeTargetSelectValue}
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v === STUDIO_SELECT_LOADING) return;
                      setScrapeTargetSlug(v);
                    }}
                    disabled={scrapeUiBusy || wsList.length === 0 || wsListLoading}
                  >
                    {wsListLoading ? (
                      <option value={STUDIO_SELECT_LOADING}>Updating ...</option>
                    ) : wsList.length === 0 ? (
                      <option value="">No saved sites — complete part 1 first</option>
                    ) : (
                      <>
                        <option value="">— Select a site —</option>
                        {wsList.map((w) => (
                          <option key={w.slug} value={w.slug}>
                            {w.display_name} ({w.slug})
                          </option>
                        ))}
                      </>
                    )}
                  </select>
                  <StudioUpdatingBadge active={wsListLoading} label="Updating ..." />
                </div>

                <div className="studio-actions studio-scrape-run-row">
                  <button
                    type="button"
                    className="primary"
                    disabled={scrapeUiBusy || !scrapeTargetSlug}
                    onClick={() => void runWorkspaceScrape()}
                  >
                    <SizeStableLabel label="Scrape" busyLabel="Scraping…" isBusy={scrapeUiBusy} />
                  </button>
                  <button
                    type="button"
                    className="danger"
                    disabled={!scrapeUiBusy}
                    title="Kill the running scrape subprocess"
                    onClick={killWorkspaceScrape}
                  >
                    Kill scrape
                  </button>
                </div>
                <div className="studio-scrape-live-log-wrap">
                  <div className="studio-scrape-live-log-label" style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}>
                    <span>Scrape output</span>
                    <button type="button" className="ghost" disabled={scrapeUiBusy} onClick={clearScrapeOutput}>
                      Clear
                    </button>
                  </div>
                  <pre ref={scrapeLogRef} className="studio-scrape-live-log" aria-live="polite">
                    {wsScrapeLog || "— Run a scrape to see subprocess output here —"}
                  </pre>
                </div>
                <WorkspaceRunLog result={wsLast} />
              </div>
            </section>
          </>
        )}

        {appView === "data-scrape" && page === "database" && (
          <>
            <h2 className="studio-page-title">Database Builder (Stage 2)</h2>
            <p className="muted-note">
              Builds provider DB and syncs registry for downstream download/preprocess stages.
            </p>
            <label className="studio-label" htmlFor="studio-probe">
              Probe JSON
            </label>
            <div className="studio-field-row-updating">
              <select
                id="studio-probe"
                className="field-input studio-field"
                value={probesLoading ? "" : probeChoice}
                onChange={(e) => setProbeChoice(e.target.value)}
                disabled={databaseUiBusy || probesLoading}
              >
                {probesLoading ? (
                  <option value="">Updating ...</option>
                ) : (
                  <>
                    <option value="">-- Select --</option>
                    {probes.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </>
                )}
              </select>
              <StudioUpdatingBadge active={probesLoading} label="Updating ..." />
            </div>
            <div className="studio-actions">
              <button
                type="button"
                className="primary"
                disabled={databaseUiBusy || probesLoading}
                onClick={() => void runDatabaseBuild()}
              >
                <SizeStableLabel
                  label="Build database"
                  busyLabel="Running…"
                  isBusy={databaseUiBusy}
                />
              </button>
            </div>
            <div className="studio-scrape-live-log-wrap">
              <div className="studio-scrape-live-log-label" style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}>
                <span>Database build output</span>
                <button type="button" className="ghost" disabled={databaseUiBusy || databaseClearBusy} onClick={() => void clearDatabaseOutput()}>
                  Clear
                </button>
              </div>
              <pre ref={databaseBuildLogRef} className="studio-scrape-live-log" aria-live="polite">
                {databaseRemoteLog ||
                  "— Subprocess output from the database builder appears here while it runs (including chat/agent runs) —"}
              </pre>
            </div>
            <RunLog result={databaseBuildResult} />

            <section className="studio-scrape-section" aria-labelledby="studio-database-catalog-heading">
              <div className="studio-scrape-section-card">
                <h3 id="studio-database-catalog-heading" className="studio-scrape-section-title">
                  Dataset catalog (browse & filter)
                </h3>
                <p className="studio-scrape-section-lead muted-note">
                  Browse datasets, apply filters, and inspect EM/seg paths and layer tokens.
                </p>
                <div className="studio-database-catalog-embed">
                  <CatalogPage />
                </div>
              </div>
            </section>
          </>
        )}

        {appView === "data-scrape" && page === "downloader" && (
          <>
            <h2 className="studio-page-title">Data Downloader (Stage 3)</h2>

            <section className="studio-scrape-section" aria-labelledby="studio-dl-generate-heading">
              <div className="studio-scrape-section-card">
                <h3 id="studio-dl-generate-heading" className="studio-scrape-section-title">
                  Download
                </h3>
                <label className="studio-label" htmlFor="studio-site">
                  Site stem
                </label>
                <div className="studio-field-row-updating">
                  <select
                    id="studio-site"
                    className="field-input studio-field"
                    value={siteStemSelectValue}
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v === STUDIO_SELECT_LOADING) return;
                      setSiteChoice(v);
                    }}
                    disabled={dlGenerateBusy || sitesLoading || sites.length === 0}
                  >
                    {sitesLoading ? (
                      <option value={STUDIO_SELECT_LOADING}>Updating ...</option>
                    ) : (
                      <>
                        <option value="">-- Select site --</option>
                        {sites.map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))}
                      </>
                    )}
                  </select>
                  <StudioUpdatingBadge active={sitesLoading} label="Updating ..." />
                </div>
                {!sitesLoading && sites.length === 0 ? (
                  <>
                    <label className="studio-label" htmlFor="studio-site-custom">
                      Custom site stem
                    </label>
                    <input
                      id="studio-site-custom"
                      className="field-input studio-field"
                      value={siteChoice}
                      onChange={(e) => setSiteChoice(e.target.value)}
                      placeholder="e.g. mysite_01 (run scrape first)"
                      autoComplete="off"
                      disabled={dlGenerateBusy}
                    />
                  </>
                ) : null}
                <p className="muted-note" style={{ marginTop: "0.5rem", marginBottom: "0.35rem" }}>
                  Fixed download settings: <strong>16 nm</strong> isotropic voxels and <strong>128³</strong> crop size.
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "0.9rem", alignItems: "start" }}>
                  <details className="studio-run-log" style={{ marginTop: "0.2rem" }}>
                    <summary>Sample type (default: all)</summary>
                    <div style={{ marginTop: "0.45rem", display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
                      <button
                        type="button"
                        className="ghost"
                        onClick={() => setDlSampleTypesSelected([...dlSampleTypeOptions])}
                        disabled={dlSampleTypeOptions.length === 0}
                      >
                        Select all
                      </button>
                      <button
                        type="button"
                        className="ghost"
                        onClick={() => setDlSampleTypesSelected([])}
                        disabled={dlSampleTypeOptions.length === 0}
                      >
                        Clear all
                      </button>
                    </div>
                    <div className="studio-run-pre" style={{ marginTop: "0.5rem", padding: "0.5rem 0.75rem" }}>
                      {dlPreviewBusy ? (
                        <p className="muted-note">Updating ...</p>
                      ) : dlSampleTypeOptions.length === 0 ? (
                        <p className="muted-note">No sample types found for this site yet.</p>
                      ) : (
                        dlSampleTypeOptions.map((t) => {
                          const checked = dlSampleTypesSelected.includes(t);
                          return (
                            <label key={t} style={{ display: "block", margin: "0.15rem 0" }}>
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={(e) => {
                                  if (e.target.checked) {
                                    setDlSampleTypesSelected((prev) => Array.from(new Set([...prev, t])));
                                  } else {
                                    setDlSampleTypesSelected((prev) => prev.filter((x) => x !== t));
                                  }
                                }}
                              />{" "}
                              <span>{t}</span>
                            </label>
                          );
                        })
                      )}
                    </div>
                  </details>
                </div>
                <div className="studio-dl-preview">
                  <div className="studio-dl-preview-head">
                    <StudioUpdatingBadge active={dlPreviewBusy} label="Updating ..." />
                  </div>
                  {dlPreviewBusy ? (
                    <p className="muted-note">Loading dataset preview…</p>
                  ) : !siteChoice.trim() ? (
                    <p className="muted-note">Pick a site stem to load datasets.</p>
                  ) : dlPreview?.ok ? (
                    <p className="muted-note">
                      About to download <strong>{activeDlFilteredPreviewDatasets.length}</strong> dataset(s) for{" "}
                      <code>{dlPreview.site}</code> (labeled inventory: good mito masks + matching EM).
                    </p>
                  ) : dlPreview ? (
                    <p className="muted-note">Preview unavailable: {dlPreview.message}</p>
                  ) : null}
                  <details className="studio-run-log">
                    <summary>Choose how many crops to download</summary>
                    <div className="studio-dl-dataset-list">
                      <div className="studio-run-pre" style={{ padding: "0.5rem 0.75rem" }}>
                        {!siteChoice.trim() ? (
                          <p className="muted-note">Select a site stem first.</p>
                        ) : dlPreviewBusy ? (
                          <p className="muted-note">Loading dataset list…</p>
                        ) : !dlPreview?.ok ? (
                          <p className="muted-note">Dataset list unavailable for this site.</p>
                        ) : activeDlFilteredPreviewDatasets.length === 0 ? (
                          <p className="muted-note">No datasets match the current sample type filter.</p>
                        ) : (
                          activeDlFilteredPreviewDatasets.slice(0, 250).map((ds) => (
                            <div
                              key={ds}
                              style={{
                                display: "grid",
                                gridTemplateColumns: "1fr auto",
                                gap: "0.75rem",
                                alignItems: "start",
                                padding: "0.2rem 0",
                              }}
                            >
                              <code>{ds}</code>
                              <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "nowrap", justifyContent: "flex-end" }}>
                                <label className="muted-note" htmlFor={`dl-train-${ds}`}>Train</label>
                                <input
                                  id={`dl-train-${ds}`}
                                  type="number"
                                  min={0}
                                  max={MAX_CROPS_PER_DATASET}
                                  step={1}
                                  className="field-input studio-field"
                                  style={{ width: "5rem" }}
                                  value={dlPreviewSplits[ds]?.training ?? 1}
                                  onChange={(e) => {
                                    const requested = Number(e.target.value) || 0;
                                    setDlDatasetSplits((prev) => {
                                      const current = prev[ds] ?? { training: 1, inference: 0 };
                                      const training = clampDownloaderSplit("training", requested, current.inference);
                                      const inference = clampDownloaderSplit("inference", current.inference, training);
                                      return { ...prev, [ds]: { training, inference } };
                                    });
                                  }}
                                  disabled={dlGenerateBusy}
                                />
                                <label className="muted-note" htmlFor={`dl-infer-${ds}`} style={{ marginLeft: "0.35rem" }}>Inference</label>
                                <input
                                  id={`dl-infer-${ds}`}
                                  type="number"
                                  min={0}
                                  max={MAX_CROPS_PER_DATASET}
                                  step={1}
                                  className="field-input studio-field"
                                  style={{ width: "5rem" }}
                                  value={dlPreviewSplits[ds]?.inference ?? 0}
                                  onChange={(e) => {
                                    const requested = Number(e.target.value) || 0;
                                    setDlDatasetSplits((prev) => {
                                      const current = prev[ds] ?? { training: 1, inference: 0 };
                                      const inference = clampDownloaderSplit("inference", requested, current.training);
                                      const training = clampDownloaderSplit("training", current.training, inference);
                                      return { ...prev, [ds]: { training, inference } };
                                    });
                                  }}
                                  disabled={dlGenerateBusy}
                                />
                              </div>
                            </div>
                          ))
                        )}
                        {dlPreview?.ok && activeDlFilteredPreviewDatasets.length > 250 ? (
                          <p className="muted-note" style={{ marginTop: "0.4rem" }}>
                            … ({activeDlFilteredPreviewDatasets.length - 250} more)
                          </p>
                        ) : null}
                      </div>
                    </div>
                    <p className="muted-note" style={{ marginTop: "0.5rem" }}>
                      Per-dataset splits are embedded in the generated downloader script. Each dataset can assign
                      crops to <code>imagesTr/labelsTr/labelsTr-instance</code> and <code>imagesTs/labelsTs/labelsTs-instance</code>,
                      with <code>training + inference ≤ 16</code>.
                    </p>
                  </details>
                </div>
                <div className="studio-dl-pending-status" aria-live="polite">
                  {pendingDlBusy ? (
                    <p className="muted-note studio-dl-pending-checking">Checking registry for pending downloads…</p>
                  ) : pendingDl?.ok && pendingDl.pending_count === 0 ? (
                    <p className="muted-note studio-dl-pending-none">
                      ✓ Nothing new to download for the current profile.
                      Executing the script will be a no-op.
                    </p>
                  ) : pendingDl?.ok && pendingDl.pending_count != null ? (
                    <p className="muted-note studio-dl-pending-some">
                      <strong>{pendingDl.pending_count}</strong> dataset(s) pending
                      {" "}(registry profile <code>{pendingDl.profile_hash}</code>).
                    </p>
                  ) : pendingDl && !pendingDl.ok ? (
                    <p className="muted-note">Registry not available: {pendingDl.message ?? "run stage 2 first"}</p>
                  ) : (
                    /* Invisible placeholder reserves the same height so the button below doesn't jump */
                    <p className="muted-note" aria-hidden="true" style={{ visibility: "hidden" }}>
                      Checking registry for pending downloads…
                    </p>
                  )}
                </div>
                <div className="studio-actions">
                  <button type="button" className="primary" disabled={dlUiBusy} onClick={() => void runDownloaderGenerate()}>
                    <SizeStableLabel label="Download" busyLabel="Downloading…" isBusy={dlUiBusy} />
                  </button>
                  <button type="button" className="danger" disabled={!dlUiBusy} onClick={() => void killDownloaderRun()}>
                    Kill download
                  </button>
                </div>
                <h3 id="studio-dl-run-heading" className="studio-scrape-section-title" style={{ marginTop: "0.75rem" }}>
                  Download progress
                </h3>
                <div style={{ marginTop: "0.75rem", marginBottom: "0.75rem" }}>
                  <div className="muted-note" style={{ marginBottom: "0.25rem" }}>
                    {dlProgress
                      ? `Progress: ${dlProgress.completed}/${dlProgress.total} crop pair(s)${
                          dlUiBusy ? ` (running pair ${dlProgress.active})` : ""
                        }`
                      : "Progress: waiting to start"}
                    {` — status: ${dlStateLabel}`}
                  </div>
                  <div
                    style={{
                      width: "100%",
                      height: "12px",
                      borderRadius: "999px",
                      background: "rgba(17, 24, 39, 0.16)",
                      border: "1px solid rgba(17, 24, 39, 0.25)",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${dlProgress ? Math.max(dlProgressPercent, dlUiBusy ? 3 : 0) : 0}%`,
                        height: "100%",
                        background: dlShowFailureStyle
                          ? "linear-gradient(90deg, #ff6b6b 0%, #ff9d6b 100%)"
                          : "linear-gradient(90deg, #2ec5ff 0%, #23d4a0 100%)",
                        transition: "width 180ms ease",
                      }}
                    />
                  </div>
                  <div className="muted-note" style={{ marginTop: "0.2rem" }}>
                    {dlProgress ? `${dlProgressPercent}%` : "0%"}
                  </div>
                </div>
                <div className="studio-scrape-live-log-wrap">
                  <div className="studio-scrape-live-log-label" style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}>
                    <span>Downloader output</span>
                    <button type="button" className="ghost" disabled={dlUiBusy} onClick={() => void clearDownloaderOutput()}>
                      Clear
                    </button>
                  </div>
                  <pre className="studio-scrape-live-log" aria-live="polite">
                    {dlRunLog || "— Click Download to see live progress here —"}
                  </pre>
                </div>
              </div>
            </section>
            <RunLog result={dlResult} />
          </>
        )}

        {appView === "data-scrape" && page === "processor" && (
          <>
            <h2 className="studio-page-title">Data Preprocessor (Stage 3: Data Downloader)</h2>
            <p className="muted-note">
              Writes nnUNet-ready NIfTI under <code>data/nnUNet_raw/Dataset001_mito2</code>.
            </p>
            <section className="studio-scrape-section" aria-labelledby="studio-preprocess-selective-heading">
              <div className="studio-scrape-section-card">
                <h3 id="studio-preprocess-selective-heading" className="studio-scrape-section-title">
                  Preprocess selected datasets
                </h3>
                <p className="muted-note studio-scrape-section-lead">
                  Select a folder under <code>data/raw</code> with <code>images/*_im.h5</code> stacks, then run preprocessing.
                </p>
                <label className="studio-label" htmlFor="studio-preprocess-download-run">
                  Download run (folder under data/raw)
                </label>
                <div className="studio-field-row-updating">
                  <select
                    id="studio-preprocess-download-run"
                    className="field-input studio-field"
                    value={downloadRunSelectValue}
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v === STUDIO_SELECT_LOADING) return;
                      setPreprocessDownloadRun(v);
                    }}
                    disabled={downloadRunSelectDisabled}
                    aria-busy={dataInspectLoading}
                  >
                    {rawDatasetInspectLoading ? (
                      <option value={STUDIO_SELECT_LOADING}>Updating ...</option>
                    ) : (
                      <>
                        <option value="">-- Select --</option>
                        {preprocessDownloadRunOptions.map((name) => (
                          <option key={name} value={name}>
                            {name}
                          </option>
                        ))}
                      </>
                    )}
                  </select>
                  <StudioUpdatingBadge active={dataInspectLoading} label="Updating ..." />
                </div>
                {!rawDatasetInspectLoading &&
                preprocessDownloadRunOptions.length > 0 &&
                downloadRunSelectValue !== STUDIO_SELECT_LOADING &&
                !canRunPreprocessOnSelectedRun ? (
                  <p className="muted-note" style={{ marginTop: "0.5rem", marginBottom: 0 }}>
                    This folder has no <code>images/*_im.h5</code> stacks yet — preprocessing expects
                    EM files under <code>images/*_im.h5</code> (browse the tree in the panel below). The server rescans the folder when you pick a run.
                  </p>
                ) : null}
                <label className="studio-label" style={{ marginTop: "0.75rem" }} htmlFor="studio-preprocess-split-cc">
                  Label geometry
                </label>
                <label className="studio-checkbox-row" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <input
                    id="studio-preprocess-split-cc"
                    type="checkbox"
                    checked={preprocessSplitLabelCc}
                    onChange={(e) => setPreprocessSplitLabelCc(e.target.checked)}
                    disabled={preUiBusy || preKillBusy}
                  />
                  <span>Split disconnected components into separate label IDs (every label volume)</span>
                </label>
                <div className="studio-actions">
                  <button
                    type="button"
                    className="primary"
                    disabled={
                      preUiBusy ||
                      preKillBusy ||
                      rawDatasetInspectLoading ||
                      preprocessDownloadRunOptions.length === 0 ||
                      !preprocessDownloadRun ||
                      downloadRunSelectValue === STUDIO_SELECT_LOADING
                    }
                    onClick={() => void runPreSelected()}
                  >
                    <SizeStableLabel label="Run preprocessing" busyLabel="Running…" isBusy={preUiBusy} />
                  </button>
                  <button
                    type="button"
                    className="danger"
                    disabled={!preUiBusy || preKillBusy}
                    onClick={() => void killPreprocessRun()}
                  >
                    <SizeStableLabel label="Kill preprocess" busyLabel="Stopping…" isBusy={preKillBusy} />
                  </button>
                </div>
                <div style={{ marginTop: "0.75rem", marginBottom: "0.75rem" }}>
                  <div className="muted-note" style={{ marginBottom: "0.25rem" }}>
                    {preProgress
                      ? `Progress: ${preProgress.completed}/${preProgress.total} dataset(s)${
                          preUiBusy ? ` (running dataset ${preProgress.active})` : ""
                        }`
                      : preprocessImH5Count > 0
                        ? `Ready: ${preprocessImH5Count} EM volume(s) with paired labels in this run`
                        : "Progress: waiting to start"}
                    {` — status: ${preStateLabel}`}
                  </div>
                  <div
                    style={{
                      width: "100%",
                      height: "12px",
                      borderRadius: "999px",
                      background: "rgba(17, 24, 39, 0.16)",
                      border: "1px solid rgba(17, 24, 39, 0.25)",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${preProgress ? Math.max(preProgressPercent, preUiBusy ? 3 : 0) : 0}%`,
                        height: "100%",
                        background:
                          preResult && !preResult.ok
                            ? "linear-gradient(90deg, #ff6b6b 0%, #ff9d6b 100%)"
                            : "linear-gradient(90deg, #2ec5ff 0%, #23d4a0 100%)",
                        transition: "width 90ms ease-out",
                      }}
                    />
                  </div>
                  <div className="muted-note" style={{ marginTop: "0.2rem" }}>
                    {preProgress ? `${preProgressPercent}%` : "0%"}
                  </div>
                </div>
                <div className="studio-scrape-live-log-wrap">
                  <div
                    className="studio-scrape-live-log-label"
                    style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}
                  >
                    <span>Preprocessor output</span>
                    <button type="button" className="ghost" disabled={preUiBusy || preKillBusy} onClick={() => void clearPreprocessorOutput()}>
                      Clear
                    </button>
                  </div>
                  <pre ref={preprocessLogRef} className="studio-scrape-live-log" aria-live="polite">
                    {preRunLog.trim() || "— Run preprocessing to see live log here (safe to refresh while running) —"}
                  </pre>
                </div>
              </div>
            </section>
            <section
              className={`studio-dataset-viewer${dataInspectLoading ? " studio-surface-updating" : ""}`}
              aria-busy={dataInspectLoading}
            >
              <div className="studio-dataset-viewer-topbar">
                <div className="studio-dataset-summary">
                  <span>
                    Raw datasets (read-only): <strong>{rawViewerRows.length}</strong> file(s) in view
                  </span>
                  <StudioUpdatingBadge active={dataInspectLoading} label="Updating ..." />
                </div>
              </div>
              <label className="studio-label" htmlFor="studio-raw-view-scope">
                Raw scope (folder under data/raw)
              </label>
              <div className="studio-field-row-updating studio-dataset-viewer-run-select">
                <select
                  id="studio-raw-view-scope"
                  className="field-input studio-field"
                  value={rawViewerScopeSelectValue}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v === STUDIO_SELECT_LOADING) return;
                    if (rawViewerSelectionOptions.includes(v)) setRawViewerSelection(v);
                  }}
                  disabled={rawDatasetInspectLoading || rawViewerSelectionOptions.length === 0}
                  aria-busy={dataInspectLoading}
                >
                  {rawDatasetInspectLoading ? (
                    <option value={STUDIO_SELECT_LOADING}>Updating ...</option>
                  ) : (
                    <>
                      <option value="">-- Select --</option>
                      {rawViewerSelectionOptions.map((folder) => (
                        <option key={folder} value={folder}>
                          {rawViewerSourceRoot}/{folder}
                        </option>
                      ))}
                    </>
                  )}
                </select>
              </div>
              <p className="muted-note">
                Source root:{" "}
                <code>{rawViewerSourceRoot}</code>. This table is for inspection only; it does not start preprocessing.
              </p>
              <div className="studio-dataset-table-wrap">
                <table className="studio-dataset-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Name</th>
                      <th>Type</th>
                      <th>Dimensions</th>
                      <th>Physical spacing</th>
                      <th title="Distinct nonzero label IDs in the volume (segmentation stacks only)">Labels</th>
                    </tr>
                  </thead>
                  <tbody>
                    {showRawViewerTableBody ? (
                      rawViewerRows.map((row) => (
                        <tr key={row.path}>
                          <td>{row.order}</td>
                          <td>{row.name}</td>
                          <td>{row.type || "—"}</td>
                          <td>{asCellVector(row.dimensions)}</td>
                          <td>{asSpacingCell(row.spacing)}</td>
                          <td>{asLabelSummaryCell(row)}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={6} className="studio-dataset-placeholder-cell">
                          {dataInspectLoading ? (
                            <>Refreshing raw dataset rows from the server…</>
                          ) : showRawViewerEmptyMessage ? (
                            <>
                              No indexed raw files under{" "}
                              <code>
                                {rawViewerSourceRoot}/
                                {rawViewerScopeSelectValue !== STUDIO_SELECT_LOADING
                                  ? rawViewerScopeSelectValue
                                  : "<folder>/<subfolder>"}
                              </code>
                              .
                            </>
                          ) : !dataInspect ? (
                            <>Dataset metadata is not loaded. Re-open this step after fixing any API error.</>
                          ) : null}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
            <RunLog result={preResult} />
          </>
        )}

        {appView === "model-training" && (
          <>
            <h2 className="studio-page-title">Training</h2>
            {selectedDataset && (
              <div className="ia-selection-banner ia-selection-banner--ok">
                <span>
                  Data source: <code>{selectedDataset}</code>
                </span>
                <div className="ia-selection-banner-actions">
                  <button type="button" className="ghost" onClick={() => setAppView("data-existing-legacy")}>
                    Change dataset
                  </button>
                </div>
              </div>
            )}
            {!selectedDataset && (
              <p className="muted-note">
                No dataset selected.{" "}
                <button type="button" className="linkish" onClick={() => setAppView("data-existing-legacy")}>
                  Select a training dataset
                </button>{" "}
                from the Data section to specify the training data source.
              </p>
            )}
            <p className="muted-note">
              Submit nnUNet training for <code>Dataset001_mito2</code> using the cluster Slurm script.
            </p>
            <h3 className="studio-subhead studio-subhead-spaced">Run nnUNet training</h3>
            <div className="studio-actions">
              <button type="button" className="primary" disabled={trainingUiBusy} onClick={() => void runTraining()}>
                <SizeStableLabel label="Start training" busyLabel="Starting…" isBusy={trainingUiBusy} />
              </button>
            </div>
            <p className="muted-note" style={{ marginTop: "0.5rem" }}>
              Note: Slurm may queue this job while waiting for compute resources. The run log appears only after the
              job starts on a compute node and writes <code>.out</code>/<code>.err</code>.
            </p>
            <SlurmLogPanel
              title="Training output"
              outPath={trainingState?.out_path || ""}
              errPath={trainingState?.err_path || ""}
              outLog={trainingState?.out_log || ""}
              errLog={trainingState?.err_log || ""}
              activeTab={trainingLogTab}
              onTabChange={setTrainingLogTab}
              onClear={() => void clearTrainingOutput()}
              clearBusy={trainingClearBusy}
              clearDisabled={Boolean(trainingState?.running) || trainingBusy}
              logRoots={trainingState?.log_roots ?? []}
              selectedLogRoot={trainingSelectedLogRoot}
              onLogRootChange={setTrainingSelectedLogRoot}
              showSummary
              summary={trainingState?.summary ?? null}
            />
            <RunLog result={trainingResult} />
          </>
        )}

        {appView === "model-inference" && (
          <>
            <h2 className="studio-page-title">Inference</h2>
            {selectedInferenceImageRows.length > 0 && (
              <div className="ia-selection-banner ia-selection-banner--ok">
                <span>
                  Selected inference images: <strong>{selectedInferenceImageRows.length}</strong>
                </span>
                <div className="ia-selection-banner-actions">
                  <button type="button" className="ghost" onClick={() => setAppView("data-existing-legacy")}>
                    Change selection
                  </button>
                </div>
              </div>
            )}
            {selectedInferenceImageRows.length === 0 && (
              <p className="muted-note">
                No dataset selected.{" "}
                <button type="button" className="linkish" onClick={() => setAppView("data-existing-legacy")}>
                  Select inference image(s)
                </button>{" "}
                from the Data section to specify the inference input.
              </p>
            )}
            <p className="muted-note">
              Submit nnUNet inference for <code>Dataset001_mito2/imagesTs</code> and write outputs to <code>data/outputs/bc</code>.
            </p>
            <section className="studio-scrape-section" aria-labelledby="studio-inference-run-heading">
              <div className="studio-scrape-section-card">
                <h3 id="studio-inference-run-heading" className="studio-scrape-section-title">
                  Run nnUNet inference
                </h3>
                <div className="studio-actions">
                  <button type="button" className="primary" disabled={inferenceUiBusy} onClick={() => void runInference()}>
                    <SizeStableLabel label="Start inference" busyLabel="Starting…" isBusy={inferenceUiBusy} />
                  </button>
                </div>
                <p className="muted-note" style={{ marginTop: "0.5rem" }}>
                  Note: Slurm may queue this job while waiting for compute resources. The run log appears only after
                  the job starts on a compute node and writes <code>.out</code>/<code>.err</code>.
                </p>
              </div>
            </section>
            <SlurmLogPanel
              title="Inference output"
              outPath={inferenceState?.out_path || ""}
              errPath={inferenceState?.err_path || ""}
              outLog={inferenceState?.out_log || ""}
              errLog={inferenceState?.err_log || ""}
              activeTab={inferenceLogTab}
              onTabChange={setInferenceLogTab}
              onClear={() => void clearInferenceOutput()}
              clearBusy={inferenceClearBusy}
              clearDisabled={Boolean(inferenceState?.running) || inferenceBusy}
              logRoots={inferenceState?.log_roots ?? []}
              selectedLogRoot={inferenceSelectedLogRoot}
              onLogRootChange={setInferenceSelectedLogRoot}
            />
            <RunLog result={inferenceResult} />
          </>
        )}

        {appView === "model-postprocessing" && (
          <>
            <h2 className="studio-page-title">Postprocessing and evaluation</h2>
            <section className="studio-scrape-section" aria-labelledby="studio-postprocess-heading">
              <div className="studio-scrape-section-card">
                <h3 id="studio-postprocess-heading" className="studio-scrape-section-title">Postprocessing</h3>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.6rem" }}>
                  <button
                    type="button"
                    className="ghost"
                    disabled={postprocessFilesLoading}
                    onClick={() => void refreshPostprocessingFiles(true)}
                  >
                    <SizeStableLabel label="Refresh table" busyLabel="Refreshing…" isBusy={postprocessFilesLoading} />
                  </button>
                </div>
                <div className="studio-dataset-table-wrap" style={{ marginTop: "0.55rem" }}>
                  <table className="studio-dataset-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Name</th>
                        <th>Directory</th>
                        <th>Type</th>
                        <th>Dimensions</th>
                        <th>Physical spacing</th>
                        <th>Distinct Labels</th>
                      </tr>
                    </thead>
                    <tbody>
                      {postprocessTableRows.length > 0 ? (
                        postprocessTableRows.map((row) => {
                          const rowKey = `${row.source}:${row.path}`;
                          return (
                            <tr key={rowKey}>
                              <td>{row.order}</td>
                              <td style={{ fontFamily: "monospace", fontSize: "0.85em" }}>{row.name}</td>
                              <td>{row.source === "input" ? "Input border-contour directory" : "Output postprocessed directory"}</td>
                              <td>{row.type || "—"}</td>
                              <td>{asCellVector(row.dimensions)}</td>
                              <td>{asSpacingCell(row.spacing)}</td>
                              <td>{asLabelSummaryCell(row)}</td>
                            </tr>
                          );
                        })
                      ) : (
                        <tr>
                          <td colSpan={7} className="studio-dataset-placeholder-cell">
                            {postprocessFilesLoading ? "Updating ..." : "No NIfTI files found in input/output folders yet."}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
                <p className="muted-note" style={{ marginTop: "0.4rem" }}>
                  Input directory: <code>{FIXED_POSTPROCESS_INPUT_DIR}</code>
                  <br />
                  Output directory: <code>{FIXED_POSTPROCESS_OUTPUT_DIR}</code>
                </p>
                <div className="studio-actions">
                  <button type="button" className="primary" disabled={postprocessBusy} onClick={() => void runPostprocessing()}>
                    <SizeStableLabel label="Run postprocessing" busyLabel="Running…" isBusy={postprocessBusy} />
                  </button>
                </div>
                <RunLog result={postprocessResult} />
              </div>
            </section>
            <section className="studio-scrape-section" aria-labelledby="studio-eval-heading">
              <div className="studio-scrape-section-card">
                <h3 id="studio-eval-heading" className="studio-scrape-section-title">Evaluation</h3>
                <p className="muted-note">Prediction directory (after watershed)</p>
                <p><code>{FIXED_EVAL_PRED_DIR}</code></p>
                <p className="muted-note">Ground-truth directory</p>
                <p><code>{FIXED_EVAL_GT_DIR}</code></p>
                <div className="studio-actions">
                  <button type="button" className="primary" disabled={evalBusy} onClick={() => void runEvaluation()}>
                    <SizeStableLabel label="Run evaluation" busyLabel="Running…" isBusy={evalBusy} />
                  </button>
                </div>
                {evalResult?.evaluation_summary ? (
                  <div className="studio-summary-box" style={{ marginTop: "0.7rem" }}>
                    <h4 style={{ margin: 0, fontSize: "0.93em" }}>Evaluation results</h4>
                    <div style={{ marginTop: "0.35rem", display: "grid", gap: "0.2rem" }}>
                      <div><strong>Cases:</strong> {String(evalResult.evaluation_summary.n_cases ?? "")}</div>
                      <div><strong>Mean F1:</strong> {typeof evalResult.evaluation_summary.mean_binary_f1 === "number" ? evalResult.evaluation_summary.mean_binary_f1.toFixed(4) : ""}</div>
                      <div><strong>Mean Precision:</strong> {typeof evalResult.evaluation_summary.mean_binary_precision === "number" ? evalResult.evaluation_summary.mean_binary_precision.toFixed(4) : ""}</div>
                      <div><strong>Mean Recall:</strong> {typeof evalResult.evaluation_summary.mean_binary_recall === "number" ? evalResult.evaluation_summary.mean_binary_recall.toFixed(4) : ""}</div>
                      <div><strong>Mean IoU:</strong> {typeof evalResult.evaluation_summary.mean_binary_iou === "number" ? evalResult.evaluation_summary.mean_binary_iou.toFixed(4) : ""}</div>
                    </div>
                  </div>
                ) : null}
                <RunLog result={evalResult} />
              </div>
            </section>
          </>
        )}

      </div>
    </div>
  );
}

// ── Inventory page component ──────────────────────────────────────────────────

function statusBadge(status: CatalogueRow["status"], hidden: boolean): JSX.Element {
  const colors: Record<string, string> = {
    present: "var(--color-ok, #2d7a2d)",
    missing_or_deleted_local: "var(--color-danger, #c0392b)",
    pending: "var(--color-warn, #b08000)",
  };
  const labels: Record<string, string> = {
    present: "Present",
    missing_or_deleted_local: "Missing / Deleted",
    pending: "Pending",
  };
  return (
    <span style={{ display: "inline-flex", gap: "0.3em", alignItems: "center" }}>
      <span style={{ color: colors[status] ?? "inherit", fontWeight: 600 }}>
        {labels[status] ?? status}
      </span>
      {hidden && (
        <span
          style={{
            fontSize: "0.72em", color: "#888",
            border: "1px solid #bbb", borderRadius: "0.2em", padding: "0 0.3em",
          }}
          title="Hidden from training datalist"
        >
          hidden
        </span>
      )}
    </span>
  );
}

interface InventoryPageProps {
  catalogue: InventoryCatalogueResponse | null;
  catalogueLoading: boolean;
  catFilterStatus: string;
  setCatFilterStatus: (v: string) => void;
  catFilterBatch: string;
  setCatFilterBatch: (v: string) => void;
  catShowMissingOnly: boolean;
  setCatShowMissingOnly: (v: boolean) => void;
  catSortCol: keyof CatalogueRow;
  setCatSortCol: (v: keyof CatalogueRow) => void;
  catSortAsc: boolean;
  setCatSortAsc: (v: boolean) => void;
  onRefresh: () => void;
}

function InventoryPage({
  catalogue, catalogueLoading,
  catFilterStatus, setCatFilterStatus,
  catFilterBatch, setCatFilterBatch,
  catShowMissingOnly, setCatShowMissingOnly,
  catSortCol, setCatSortCol,
  catSortAsc, setCatSortAsc,
  onRefresh,
}: InventoryPageProps): JSX.Element {
  const [deleteHistoryDismissed, setDeleteHistoryDismissed] = useState(false);
  const [deleteHistoryLeaving, setDeleteHistoryLeaving] = useState(false);
  const deleteHistoryLenRef = useRef(0);
  const [batchesDismissed, setBatchesDismissed] = useState(false);
  const [batchesLeaving, setBatchesLeaving] = useState(false);
  const batchesLenRef = useRef(0);

  const s = catalogue?.summary;
  const rows = catalogue?.rows ?? [];
  const onDiskPairCount = s?.on_disk_pairs ?? 0;
  const batchCards = s?.batches ?? [];
  const deleteHistory = s?.delete_history ?? [];

  useEffect(() => {
    const n = deleteHistory.length;
    if (n === 0) {
      deleteHistoryLenRef.current = 0;
      setDeleteHistoryDismissed(false);
      setDeleteHistoryLeaving(false);
      return;
    }
    if (n > deleteHistoryLenRef.current) {
      setDeleteHistoryDismissed(false);
      setDeleteHistoryLeaving(false);
    }
    deleteHistoryLenRef.current = n;
  }, [deleteHistory.length]);

  useEffect(() => {
    const n = batchCards.length;
    if (n === 0) {
      batchesLenRef.current = 0;
      setBatchesDismissed(false);
      setBatchesLeaving(false);
      return;
    }
    if (n > batchesLenRef.current) {
      setBatchesDismissed(false);
      setBatchesLeaving(false);
    }
    batchesLenRef.current = n;
  }, [batchCards.length]);

  const batchLabelById = Object.fromEntries(
    batchCards.map((b) => [b.batch_id, b.display_title ?? b.batch_id]),
  );
  const tracked = s?.total_items ?? 0;
  const presentR = s?.present ?? 0;
  const missingR = s?.missing_or_deleted ?? 0;
  const pendingRegistryCount = s?.pending ?? 0;
  const logDownloadCompletions = s?.download_completions_total ?? 0;
  const registrySum = presentR + missingR + pendingRegistryCount;
  const registryBalanced = tracked > 0 && registrySum === tracked;

  const filtered = rows.filter((r) => {
    if (catShowMissingOnly && r.status !== "missing_or_deleted_local") return false;
    if (catFilterStatus && r.status !== catFilterStatus) return false;
    if (catFilterBatch) {
      const q = catFilterBatch.toLowerCase();
      if (!r.batch_id.toLowerCase().includes(q) && !r.stable_id.toLowerCase().includes(q)) return false;
    }
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    const va = String(a[catSortCol] ?? "");
    const vb = String(b[catSortCol] ?? "");
    const cmp = va.localeCompare(vb, undefined, { numeric: true });
    return catSortAsc ? cmp : -cmp;
  });

  const toggleSort = (col: keyof CatalogueRow) => {
    if (catSortCol === col) setCatSortAsc(!catSortAsc);
    else { setCatSortCol(col); setCatSortAsc(true); }
  };
  const sortIcon = (col: keyof CatalogueRow) =>
    catSortCol === col ? (catSortAsc ? " ↑" : " ↓") : "";

  const batchIds = [...new Set(rows.map((r) => r.batch_id))].sort();

  return (
    <>
      <h2 className="studio-page-title">Inventory (Stage 0)</h2>
      <p className="muted-note">
        Live view of registry <strong>batch_items</strong> with on-disk pair counts.
      </p>

      {/* Summary cards */}
      {s && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", margin: "1rem 0", alignItems: "stretch" }}>
          <div
            style={{
              display: "flex",
              flexWrap: "nowrap",
              gap: "0.35rem",
              alignItems: "stretch",
            }}
          >
            <div
              className="studio-scrape-section-card"
              style={{
                minWidth: "12rem",
                padding: "0.55rem 1rem 0.65rem",
                textAlign: "center",
              }}
              title={
                "Present = batch_items currently on disk. Log = sum over batches of "
                + "(new planned image/label pairs × 2) for EM + mito_seg units; 0 when a batch recorded no new pairs."
              }
            >
              <div style={{ display: "flex", justifyContent: "center", alignItems: "flex-end", gap: "1.25rem" }}>
                <div>
                  <div style={{ fontSize: "1.55em", fontWeight: 700, color: "var(--color-ok, #2d7a2d)" }}>
                    {presentR}
                  </div>
                  <div style={{ fontSize: "0.68em", color: "#666", marginTop: "0.12em", lineHeight: 1.25 }}>
                    Present files
                    <br />
                    (registry)
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: "1.55em", fontWeight: 700, color: "#888" }}>
                    {logDownloadCompletions}
                  </div>
                  <div style={{ fontSize: "0.68em", color: "#666", marginTop: "0.12em", lineHeight: 1.25 }}>
                    Log files
                    <br />
                    (downloads)
                  </div>
                </div>
              </div>
            </div>
            <div
              className="studio-scrape-section-card"
              style={{
                minWidth: "12rem",
                padding: "0.55rem 1rem 0.65rem",
                textAlign: "center",
              }}
              title="Registry rows marked missing/deleted vs. append-only file delete log"
            >
              <div style={{ display: "flex", justifyContent: "center", alignItems: "flex-end", gap: "1.25rem" }}>
                <div>
                  <div style={{ fontSize: "1.55em", fontWeight: 700, color: "var(--color-danger, #c0392b)" }}>
                    {missingR}
                  </div>
                  <div style={{ fontSize: "0.68em", color: "#666", marginTop: "0.12em", lineHeight: 1.25 }}>
                    Missing files
                    <br />
                    (registry)
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: "1.55em", fontWeight: 700, color: "#888" }}>
                    {s.deletion_events_count ?? 0}
                  </div>
                  <div style={{ fontSize: "0.68em", color: "#666", marginTop: "0.12em", lineHeight: 1.25 }}>
                    Deletes
                    <br />
                    (log)
                  </div>
                </div>
              </div>
            </div>
          </div>
          {(
            [
              { label: "On-disk training pairs",  value: Number(s.on_disk_pairs_training ?? 0), color: "inherit" },
              { label: "On-disk inference pairs", value: Number(s.on_disk_pairs_inference ?? 0), color: "inherit" },
            ] as { label: string; value: number; color: string }[]
          ).map(({ label, value, color }) => (
            <div key={label} className="studio-scrape-section-card"
              style={{ minWidth: "9rem", padding: "0.6rem 1rem", textAlign: "center" }}>
              <div style={{ fontSize: "1.6em", fontWeight: 700, color }}>{value}</div>
              <div style={{ fontSize: "0.78em", color: "#666", marginTop: "0.15em" }}>{label}</div>
            </div>
          ))}
          {(
            [
              { label: "Hidden from train", value: s.hidden_from_training, color: "#888" },
            ] as { label: string; value: number; color: string }[]
          ).map(({ label, value, color }) => (
            <div key={label} className="studio-scrape-section-card"
              style={{ minWidth: "9rem", padding: "0.6rem 1rem", textAlign: "center" }}>
              <div style={{ fontSize: "1.6em", fontWeight: 700, color }}>{value}</div>
              <div style={{ fontSize: "0.78em", color: "#666", marginTop: "0.15em" }}>{label}</div>
            </div>
          ))}
          {Object.entries(s.providers).map(([prov, cnt]) => (
            <div key={prov} className="studio-scrape-section-card"
              style={{ minWidth: "7rem", padding: "0.6rem 1rem", textAlign: "center" }}>
              <div style={{ fontSize: "1.6em", fontWeight: 700 }}>{cnt}</div>
              <div style={{ fontSize: "0.78em", color: "#666", marginTop: "0.15em" }}>{prov}</div>
            </div>
          ))}
        </div>
      )}
      {s && tracked > 0 && (
        <p className="muted-note" style={{ marginTop: "-0.35rem", fontSize: "0.82em" }}>
          Registry files: Present ({presentR}) + Missing ({missingR}) + Pending ({pendingRegistryCount}) = {registrySum}.
          {" "}
          {registryBalanced
            ? `Matches tracked total (${tracked}).`
            : `Expected ${registrySum} to equal tracked (${tracked}); try Refresh or inspect the registry.`}
        </p>
      )}

      {s && batchesDismissed && batchCards.length > 0 && (
        <p className="muted-note" style={{ margin: "0.25rem 0 0.5rem", fontSize: "0.82em" }}>
          <button
            type="button"
            className="ghost"
            onClick={() => {
              setBatchesDismissed(false);
              setBatchesLeaving(false);
            }}
          >
            Show batches ({batchCards.length})
          </button>
        </p>
      )}
      {/* Batches: collapsible like delete history; one wide card, stacked rows (newest first from API). */}
      {s && batchCards.length > 0 && !batchesDismissed && (
        <div
          className={`studio-inventory-delete-history-section${batchesLeaving ? " studio-inventory-delete-history-section--leave" : ""}`}
          style={{ margin: "0.75rem 0 1rem" }}
          onAnimationEnd={(e) => {
            if (!batchesLeaving) return;
            const name = String(e.animationName || "");
            if (!name.includes("studio-delete-history-pop-out")) return;
            setBatchesDismissed(true);
            setBatchesLeaving(false);
          }}
        >
          <div className="studio-inventory-delete-history-header">
            <h3 style={{ fontSize: "0.95em", fontWeight: 600, margin: 0 }}>Batches</h3>
            <button
              type="button"
              className="studio-inventory-delete-history-dismiss"
              title="Hide batches"
              aria-label="Hide batches"
              onClick={() => setBatchesLeaving(true)}
            >
              ×
            </button>
          </div>
          <div
            className="studio-scrape-section-card"
            style={{
              padding: "0.35rem 0.85rem",
              minWidth: "16rem",
              width: "100%",
              maxWidth: "56rem",
              boxSizing: "border-box",
            }}
          >
            {batchCards.map((b, idx) => {
              const last = idx === batchCards.length - 1;
              const trainingUnits = Number(
                b.training_units_this_run ?? (b.profile?.training_units_this_run as number | undefined) ?? 0,
              );
              const inferenceUnits = Number(
                b.inference_units_this_run ?? (b.profile?.inference_units_this_run as number | undefined) ?? 0,
              );
              return (
                <div
                  key={b.batch_id}
                  style={{
                    padding: "0.5rem 0",
                    borderBottom: last ? undefined : "1px solid var(--border, #ddd)",
                  }}
                >
                  <div style={{ fontSize: "0.88em", fontWeight: 600, lineHeight: 1.35 }}>
                    {b.display_title ?? b.batch_id}
                  </div>
                  <code style={{ fontSize: "0.72em", wordBreak: "break-all", color: "#888" }} title="batch_id">{b.batch_id}</code>
                  {b.profile_hash != null && String(b.profile_hash).length > 0 && (
                    <div style={{ fontSize: "0.7em", color: "#999", marginTop: "0.15rem", fontFamily: "monospace" }}>
                      profile <code>{String(b.profile_hash)}</code>
                    </div>
                  )}
                  <div style={{ marginTop: "0.25rem", fontSize: "0.82em", display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                    <span style={{ color: "var(--color-ok, #2d7a2d)" }}>{b.n_present} present</span>
                    {b.n_missing > 0 && <span style={{ color: "var(--color-danger, #c0392b)" }}>{b.n_missing} missing</span>}
                    <span style={{ color: "#888" }}>{b.n_items} total</span>
                  </div>
                  <div style={{ marginTop: "0.15rem", fontSize: "0.78em", color: "#666" }}>
                    This run (→ Log):{" "}
                    <strong style={{ color: "#555" }}>{b.download_asset_completions ?? 0}</strong>
                    {" "}
                    EM+mito_seg units (new pairs this run × 2)
                  </div>
                  <div style={{ marginTop: "0.1rem", fontSize: "0.76em", color: "#777" }}>
                    Training units: <strong style={{ color: "#555" }}>{trainingUnits}</strong>
                    {" "}· Inference units: <strong style={{ color: "#555" }}>{inferenceUnits}</strong>
                  </div>
                  {b.profile && Object.keys(b.profile).length > 0 && (
                    <div style={{ fontSize: "0.72em", color: "#888", marginTop: "0.2rem" }}>
                      {Object.entries(b.profile)
                        .filter(([k]) => !["foundation", "download_asset_completions", "planned_asset_downloads", "datasets_this_run", "planned_pairs", "n_windows", "training_units_this_run", "inference_units_this_run", "training_pairs_this_run", "inference_pairs_this_run", "dataset_totals"].includes(k))
                        .map(([k, v]) => {
                          let sv: string;
                          if (Array.isArray(v)) sv = v.join("×");
                          else if (v !== null && typeof v === "object") sv = JSON.stringify(v);
                          else sv = String(v);
                          return `${k}=${sv}`;
                        })
                        .join(" · ")}
                    </div>
                  )}
                  <div style={{ fontSize: "0.72em", color: "#aaa", marginTop: "0.15rem" }}>
                    {b.created_at ? toEasternString(b.created_at) : ""}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
      {s && deleteHistoryDismissed && deleteHistory.length > 0 && (
        <p className="muted-note" style={{ margin: "0.25rem 0 0.5rem", fontSize: "0.82em" }}>
          <button
            type="button"
            className="ghost"
            onClick={() => {
              setDeleteHistoryDismissed(false);
              setDeleteHistoryLeaving(false);
            }}
          >
            Show delete history ({deleteHistory.length})
          </button>
        </p>
      )}
      {s && deleteHistory.length > 0 && !deleteHistoryDismissed && (
        <div
          className={`studio-inventory-delete-history-section${deleteHistoryLeaving ? " studio-inventory-delete-history-section--leave" : ""}`}
          style={{ margin: "0.75rem 0 1rem" }}
          onAnimationEnd={(e) => {
            if (!deleteHistoryLeaving) return;
            const name = String(e.animationName || "");
            if (!name.includes("studio-delete-history-pop-out")) return;
            setDeleteHistoryDismissed(true);
            setDeleteHistoryLeaving(false);
          }}
        >
          <div className="studio-inventory-delete-history-header">
            <h3 style={{ fontSize: "0.95em", fontWeight: 600, margin: 0 }}>Delete history</h3>
            <button
              type="button"
              className="studio-inventory-delete-history-dismiss"
              title="Hide delete history"
              aria-label="Hide delete history"
              onClick={() => setDeleteHistoryLeaving(true)}
            >
              ×
            </button>
          </div>
          <div
            className="studio-scrape-section-card"
            style={{
              padding: "0.35rem 0.85rem",
              minWidth: "16rem",
              width: "100%",
              maxWidth: "56rem",
              boxSizing: "border-box",
            }}
          >
            {deleteHistory.slice(0, 20).map((d, idx) => {
              const slice = deleteHistory.slice(0, 20);
              const last = idx === slice.length - 1;
              return (
                <div
                  key={`${d.deleted_at}-${d.local_path}-${idx}`}
                  style={{
                    padding: "0.45rem 0",
                    borderBottom: last ? undefined : "1px solid var(--border, #ddd)",
                  }}
                >
                  <code style={{ fontSize: "0.78em", wordBreak: "break-all" }}>{d.stable_id}</code>
                  <div style={{ marginTop: "0.25rem", fontSize: "0.82em", display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                    {d.asset_type && (
                      <span style={{ color: "var(--color-warn, #b08000)" }}>{d.asset_type}</span>
                    )}
                    {d.provider && <span style={{ color: "#888" }}>{d.provider}</span>}
                  </div>
                  {d.local_path && (
                    <div style={{ fontSize: "0.72em", color: "#888", marginTop: "0.2rem" }} title={d.local_path}>
                      {d.local_path.length > 80 ? `…${d.local_path.slice(-80)}` : d.local_path}
                    </div>
                  )}
                  <div style={{ fontSize: "0.72em", color: "#aaa", marginTop: "0.15rem" }}>
                    {d.deleted_at ? toEasternString(d.deleted_at) : ""}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Filter + refresh */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "center", margin: "0.75rem 0" }}>
        <select className="studio-select" value={catFilterStatus}
          onChange={(e) => setCatFilterStatus(e.target.value)} aria-label="Filter by status"
          style={{ minWidth: "11rem" }}>
          <option value="">All statuses</option>
          <option value="present">Present</option>
          <option value="missing_or_deleted_local">Missing / Deleted</option>
          <option value="pending">Pending</option>
        </select>
        <select className="studio-select" value={catFilterBatch}
          onChange={(e) => setCatFilterBatch(e.target.value)} aria-label="Filter by batch"
          style={{ minWidth: "15rem" }}>
          <option value="">All batches</option>
          {batchIds.map((bid) => (
            <option key={bid} value={bid}>{batchLabelById[bid] ?? bid}</option>
          ))}
        </select>
        <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", cursor: "pointer", fontSize: "0.88em" }}>
          <input type="checkbox" checked={catShowMissingOnly} onChange={(e) => setCatShowMissingOnly(e.target.checked)} />
          Show missing only
        </label>
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <StudioUpdatingBadge active={catalogueLoading} label="Updating ..." />
          <button type="button" className="ghost" disabled={catalogueLoading} onClick={onRefresh}>Refresh</button>
        </span>
      </div>

      {!catalogueLoading && catalogue === null && (
        <p className="muted-note">Could not load inventory — check backend connection.</p>
      )}
      {!catalogueLoading && catalogue !== null && !catalogue.registry_exists && (
        <p className="muted-note">
          Registry not built yet — run <strong>Stage 2 Database Builder</strong> first to initialise the registry.
          Once you have downloaded datasets, they will appear here automatically.
        </p>
      )}

      {catalogue?.registry_exists && (
        <div className="studio-dataset-table-wrap">
          <table className="studio-dataset-table">
            <thead>
              <tr>
                <th>#</th>
                <th><button type="button" className="studio-sort-btn" onClick={() => toggleSort("stable_id")}>Dataset{sortIcon("stable_id")}</button></th>
                <th><button type="button" className="studio-sort-btn" onClick={() => toggleSort("provider")}>Provider{sortIcon("provider")}</button></th>
                <th><button type="button" className="studio-sort-btn" onClick={() => toggleSort("batch_id")}>Batch{sortIcon("batch_id")}</button></th>
                <th><button type="button" className="studio-sort-btn" onClick={() => toggleSort("data_source")}>Dataset split{sortIcon("data_source")}</button></th>
                <th>Asset type</th>
                <th><button type="button" className="studio-sort-btn" onClick={() => toggleSort("status")}>Status{sortIcon("status")}</button></th>
                <th title="Absolute path of the local file/directory">Local path</th>
                <th>Profile hash</th>
                <th><button type="button" className="studio-sort-btn" onClick={() => toggleSort("completed_at")}>Completed{sortIcon("completed_at")}</button></th>
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 ? (
                <tr>
                  <td colSpan={10} className="studio-dataset-placeholder-cell">
                    {catalogueLoading ? "Updating ..."
                      : rows.length === 0
                        ? (onDiskPairCount > 0
                          ? (
                            <>The summary sees {onDiskPairCount} on-disk pair{onDiskPairCount === 1 ? "" : "s"} but the table has no rows: restart the API (<code>./mito2</code> or <code>MITO2_RELOAD=1</code>), run <code>cd frontend &amp;&amp; npm run build</code>, then <strong>Refresh</strong>. With a current build, the first inventory load bootstraps paired training files into the registry so rows appear here.</>
                          ) : (
                            <>No tracked downloads yet. Run <strong>Stage 3 Data Downloader</strong> to populate the inventory.</>
                          ))
                        : "No rows match the current filters."}
                  </td>
                </tr>
              ) : sorted.map((r, i) => (
                <tr key={r.item_id}
                  style={r.status === "missing_or_deleted_local" ? { background: "rgba(192,57,43,0.04)" } : undefined}>
                  <td>{i + 1}</td>
                  <td style={{ fontFamily: "monospace", fontSize: "0.85em" }}>{r.stable_id}</td>
                  <td>{r.provider}</td>
                  <td style={{ fontFamily: "monospace", fontSize: "0.78em", maxWidth: "16rem",
                               overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                    title={batchLabelById[r.batch_id] ?? r.batch_id}>{r.batch_id}</td>
                  <td style={{ fontSize: "0.82em", color: "#666" }}>{r.data_source ?? "unknown"}</td>
                  <td style={{ fontSize: "0.82em", color: "#666" }}>{r.asset_type}</td>
                  <td>{statusBadge(r.status, r.hidden_from_training)}</td>
                  <td style={{ fontFamily: "monospace", fontSize: "0.72em", maxWidth: "18rem",
                               overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                               color: r.local_path ? "inherit" : "#bbb" }}
                    title={r.local_path ?? undefined}>{r.local_path ?? "—"}</td>
                  <td style={{ fontFamily: "monospace", fontSize: "0.78em", color: "#888" }}>{r.profile_hash ?? "—"}</td>
                  <td style={{ fontSize: "0.82em", whiteSpace: "nowrap", color: "#666" }}>
                    {r.completed_at ? toEasternString(r.completed_at) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {sorted.length > 0 && (
            <p className="muted-note" style={{ marginTop: "0.4rem", fontSize: "0.78em" }}>
              Showing {sorted.length} of {rows.length} table row(s).
              {s != null && (
                <> Tracked registry batch_items: {s.total_items}.</>
              )}
            </p>
          )}
        </div>
      )}
    </>
  );
}

