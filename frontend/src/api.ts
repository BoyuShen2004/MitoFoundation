const base = "";

/** FastAPI often returns `{ detail: string | object[] }` on errors. */
export async function parseApiError(res: Response): Promise<string> {
  const t = await res.text();
  try {
    const j = JSON.parse(t) as { detail?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) {
      return j.detail
        .map((d: { msg?: string; loc?: unknown }) => d.msg ?? JSON.stringify(d))
        .join("; ");
    }
    if (j.detail != null) return JSON.stringify(j.detail);
  } catch {
    /* ignore */
  }
  return t.trim() || res.statusText || `HTTP ${res.status}`;
}

export type ChatMode = "ask" | "plan" | "agent";

export type ChatMessage = { role: "user" | "assistant"; content: string; chat_mode?: ChatMode };
export type ChatThreadSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  preview: string;
};

export type PendingApproval = {
  id: string;
  title: string;
  command: string[];
  cwd: string;
  created_at: string;
  status: string;
  detail: Record<string, unknown>;
};

export type PipelinePlanRecord = {
  id: string;
  created_at: string;
  title: string;
  plan: Record<string, unknown>;
};

export type PlanningHint = {
  planning_ui_focus: string;
  plan_sites: string[];
  plan_stages: string[];
};

export type PostChatOptions = {
  chat_mode?: ChatMode;
  pipeline_action?: string | null;
  pipeline_plan?: Record<string, unknown> | null;
  signal?: AbortSignal;
  /** When true (default), use NDJSON stream so the UI can show LLM planning hints before the final reply. */
  stream_planning?: boolean;
  onPlanning?: (hint: PlanningHint) => void;
};

export type ChatApiResult = {
  reply: string;
  pending_approvals: PendingApproval[];
  pipeline: { current_step: number; step_label: string; last_url: string; last_site_stem: string };
  draft_pipeline_plan?: Record<string, unknown> | null;
  execution?: PipelineExecutionMeta | Record<string, unknown> | null;
  chat_mode?: ChatMode;
  planning_hint?: PlanningHint | null;
};

/** One row from `/api/chat` `execution.steps` (site + stage names come from the plan, not the UI). */
export type PipelineExecutionStep = { site: string; stage: string; ok: boolean };

/** Subset of server `execution` used to show pipeline progress in the chat chrome. */
export type PipelineExecutionMeta = {
  ok?: boolean;
  sites: string[];
  stages: string[];
  steps: PipelineExecutionStep[];
  n_crops?: number;
  n_crops_training?: number;
  n_crops_inference?: number;
};

/** Parse `execution` from chat responses; returns null if nothing to show. */
export function normalizePipelineExecution(raw: unknown): PipelineExecutionMeta | null {
  if (raw == null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const sites = Array.isArray(o.sites) ? o.sites.map((x) => String(x)) : [];
  const stages = Array.isArray(o.stages) ? o.stages.map((x) => String(x)) : [];
  const steps: PipelineExecutionStep[] = [];
  if (Array.isArray(o.steps)) {
    for (const row of o.steps) {
      if (row == null || typeof row !== "object") continue;
      const r = row as Record<string, unknown>;
      steps.push({
        site: String(r.site ?? ""),
        stage: String(r.stage ?? ""),
        ok: Boolean(r.ok),
      });
    }
  }
  if (steps.length === 0 && sites.length === 0 && stages.length === 0) return null;
  const hasDownload = stages.includes("download");
  return {
    ok: o.ok === undefined ? undefined : Boolean(o.ok),
    sites,
    stages,
    steps,
    n_crops: hasDownload && typeof o.n_crops === "number" ? o.n_crops : undefined,
    n_crops_training: hasDownload && typeof o.n_crops_training === "number" ? o.n_crops_training : undefined,
    n_crops_inference: hasDownload && typeof o.n_crops_inference === "number" ? o.n_crops_inference : undefined,
  };
}

function parseNdjsonChatBody(text: string, onPlanning?: (hint: PlanningHint) => void): ChatApiResult {
  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  let finalData: ChatApiResult | null = null;
  for (const line of lines) {
    let row: { event?: string; data?: unknown };
    try {
      row = JSON.parse(line) as { event?: string; data?: unknown };
    } catch {
      continue;
    }
    if (row.event === "planning" && row.data && typeof row.data === "object" && row.data !== null) {
      onPlanning?.(row.data as PlanningHint);
    } else if (row.event === "done" && row.data && typeof row.data === "object" && row.data !== null) {
      finalData = row.data as ChatApiResult;
    }
  }
  if (!finalData) throw new Error("Chat response: missing done event in NDJSON stream");
  return finalData;
}

async function readNdjsonChatResponse(res: Response, onPlanning?: (hint: PlanningHint) => void): Promise<ChatApiResult> {
  const reader = res.body?.getReader();
  if (!reader) {
    const t = await res.text();
    if (t.includes('"event"')) return parseNdjsonChatBody(t, onPlanning);
    return JSON.parse(t) as ChatApiResult;
  }
  const dec = new TextDecoder();
  let buf = "";
  let finalData: ChatApiResult | null = null;
  while (true) {
    const { done, value } = await reader.read();
    if (value) {
      buf += dec.decode(value, { stream: !done });
    }
    const lines = buf.split("\n");
    buf = done ? "" : (lines.pop() ?? "") || "";
    for (const line of lines) {
      const t = line.trim();
      if (!t) continue;
      let row: { event?: string; data?: unknown };
      try {
        row = JSON.parse(t) as { event?: string; data?: unknown };
      } catch {
        continue;
      }
      if (row.event === "planning" && row.data && typeof row.data === "object" && row.data !== null) {
        onPlanning?.(row.data as PlanningHint);
      } else if (row.event === "done" && row.data && typeof row.data === "object" && row.data !== null) {
        finalData = row.data as ChatApiResult;
      }
    }
    if (done) break;
  }
  if (buf.trim()) {
    try {
      const row = JSON.parse(buf) as { event?: string; data?: unknown };
      if (row.event === "planning" && row.data) onPlanning?.(row.data as PlanningHint);
      if (row.event === "done" && row.data) finalData = row.data as ChatApiResult;
    } catch {
      /* ignore */
    }
  }
  if (!finalData) throw new Error("Chat response: missing done event in NDJSON stream");
  return finalData;
}

export async function postChat(
  sessionId: string,
  message: string,
  history: ChatMessage[],
  opts?: PostChatOptions,
) {
  const useStream = opts?.stream_planning !== false;
  const body: Record<string, unknown> = {
    session_id: sessionId,
    message,
    history,
    chat_mode: opts?.chat_mode ?? "ask",
    stream_planning: useStream,
  };
  if (opts?.pipeline_action) body.pipeline_action = opts.pipeline_action;
  if (opts?.pipeline_plan != null) body.pipeline_plan = opts.pipeline_plan;
  const res = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: opts?.signal,
  });
  if (!res.ok) throw new Error(await res.text());
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (useStream) {
    if (ct.includes("ndjson") && res.body != null) {
      return readNdjsonChatResponse(res, opts?.onPlanning);
    }
    const raw = await res.text();
    try {
      return parseNdjsonChatBody(raw, opts?.onPlanning);
    } catch {
      return JSON.parse(raw) as ChatApiResult;
    }
  }
  return res.json() as Promise<ChatApiResult>;
}

export async function postChatStop(sessionId: string) {
  const res = await fetch(`${base}/api/chat/stop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; stop_requested: boolean; chat_turn_in_progress?: boolean }>;
}

export async function postChatEdit(
  sessionId: string,
  messageIndex: number,
  message: string,
  chatMode: ChatMode = "ask",
  signal?: AbortSignal,
) {
  const res = await fetch(`${base}/api/chat/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message_index: messageIndex,
      message,
      chat_mode: chatMode,
    }),
    signal,
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
    reply: string;
    messages: ChatMessage[];
    pending_approvals: PendingApproval[];
    pipeline: { current_step: number; step_label: string; last_url: string; last_site_stem: string };
    draft_pipeline_plan?: Record<string, unknown> | null;
    execution?: PipelineExecutionMeta | Record<string, unknown> | null;
    chat_mode?: ChatMode;
  }>;
}

export async function getChatState(sessionId: string) {
  const q = new URLSearchParams({ session_id: sessionId });
  const res = await fetch(`${base}/api/chat/state?${q.toString()}`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
    thread: { id: string; title: string; created_at: string; updated_at: string };
    messages: ChatMessage[];
    pending_approvals: PendingApproval[];
    pipeline: { current_step: number; step_label: string; last_url: string; last_site_stem: string };
    /** True while the server is still running a `/api/chat` or `/api/chat/edit` turn (e.g. long pipeline execute). */
    chat_turn_in_progress?: boolean;
    chat_turn_pending_user?: string;
    chat_turn_pending_mode?: string;
    chat_turn_pending_pipeline_plan?: Record<string, unknown> | null;
    /** Server ``time.time()`` (seconds) when the in-flight turn began — for stepper rehydration after refresh. */
    chat_turn_started_at?: number | null;
    last_pipeline_plan?: Record<string, unknown> | null;
    /** When true, the server decided an Agent Approve/Run plan is still pending (survives refresh; see routes). */
    chat_agent_plan_awaiting_approval?: boolean;
  }>;
}

/** Live session snapshot while `/api/chat` is blocked on a long pipeline execute. */
export async function getPipelineProgress(sessionId: string) {
  const q = new URLSearchParams({ session_id: sessionId });
  const res = await fetch(`${base}/api/pipeline?${q.toString()}`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
    current_step: number;
    step_label: string;
    last_url: string;
    last_site_stem: string;
    /** Present for ``chat_*`` sessions when an approved plan includes ``n_crops`` (agent pipeline mirror). */
    plan_n_crops?: number;
    plan_n_crops_training?: number;
    plan_n_crops_inference?: number;
  }>;
}

export async function listPipelinePlans() {
  const res = await fetch(`${base}/api/chat/plans`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ plans: PipelinePlanRecord[] }>;
}

export async function savePipelinePlan(plan: Record<string, unknown>, title = "") {
  const res = await fetch(`${base}/api/chat/plans`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan, title }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; entry: PipelinePlanRecord }>;
}

export async function deletePipelinePlan(planId: string) {
  const res = await fetch(`${base}/api/chat/plans/${encodeURIComponent(planId)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean }>;
}

export async function listChats(query = "") {
  const q = new URLSearchParams();
  if (query.trim()) q.set("query", query.trim());
  const res = await fetch(`${base}/api/chats?${q.toString()}`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ threads: ChatThreadSummary[] }>;
}

export async function createChat() {
  const res = await fetch(`${base}/api/chats/new`, { method: "POST" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ id: string }>;
}

export async function deleteChats(sessionIds: string[]) {
  const res = await fetch(`${base}/api/chats/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_ids: sessionIds }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; deleted: string[] }>;
}

export async function resolveApproval(sessionId: string, id: string, approved: boolean) {
  const res = await fetch(`${base}/api/approvals/${id}?session_id=${encodeURIComponent(sessionId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ status: string; result?: { returncode: number; stdout: string; stderr: string } }>;
}

export type AgentSkillMeta = {
  kind: "chat" | "orchestration";
  slug: string;
  id: string;
  title: string;
  label: string;
};

export async function getAgentSkillIndex(): Promise<{ chat: AgentSkillMeta[]; orchestration: AgentSkillMeta[] }> {
  const res = await fetch(`${base}/api/agent-skills`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ chat: AgentSkillMeta[]; orchestration: AgentSkillMeta[] }>;
}

export async function getAgentSkillDocument(
  kind: string,
  slug: string,
): Promise<{ kind: string; slug: string; document: string }> {
  const res = await fetch(
    `${base}/api/agent-skills/${encodeURIComponent(kind)}/${encodeURIComponent(slug)}`,
  );
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ kind: string; slug: string; document: string }>;
}

export async function putAgentSkillDocument(kind: string, slug: string, document: string): Promise<void> {
  const res = await fetch(
    `${base}/api/agent-skills/${encodeURIComponent(kind)}/${encodeURIComponent(slug)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document }),
    },
  );
  if (!res.ok) throw new Error(await parseApiError(res));
}

export async function postCreateAgentSkill(
  kind: string,
  payload: { slug: string; label: string; title?: string; id?: string; body?: string },
): Promise<{ kind: string; slug: string; status: string }> {
  const res = await fetch(`${base}/api/agent-skills/${encodeURIComponent(kind)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ kind: string; slug: string; status: string }>;
}

export async function patchAgentSkillSlug(kind: string, slug: string, newSlug: string): Promise<void> {
  const res = await fetch(
    `${base}/api/agent-skills/${encodeURIComponent(kind)}/${encodeURIComponent(slug)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_slug: newSlug }),
    },
  );
  if (!res.ok) throw new Error(await parseApiError(res));
}

export type LlmSettingsSaved = {
  openai_base_url: string;
  openai_model: string;
  openai_api_key: string;
  chatgpt_account_id: string;
  codex_auth_profile_id: string;
  codex_auth_json_path: string;
  /** Legacy; cleared on startup for ChatGPT Codex (model is fixed). */
  codex_agent_model_list?: string;
  /** chatgpt_codex | openai_api | "" (infer from saved URL) */
  llm_provider: string;
};

export type CodexProfileRow = {
  id: string;
  label: string;
  account_id_preview: string;
  has_access_token: boolean;
  /** Optional tooltip (e.g. email); primary label stays neutral. */
  detail?: string;
};

export type LlmSettingsResponse = {
  saved: LlmSettingsSaved;
  effective: {
    base_url: string;
    model: string;
    transport: string;
    api_key_configured: boolean;
    chatgpt_account_id_configured: boolean;
  };
  env_overrides: string[];
};

export async function getLlmSettings() {
  const res = await fetch(`${base}/api/llm/settings`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<LlmSettingsResponse>;
}

export async function getLlmStatus() {
  const res = await fetch(`${base}/api/llm/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{
    base_url: string;
    model: string;
    transport: string;
    api_key_configured: boolean;
    chatgpt_account_id_configured: boolean;
    hint_markdown: string;
  }>;
}

export async function postCodexLoginBrowser() {
  const q = new URLSearchParams({ wait_seconds: "90" });
  const res = await fetch(`${base}/api/llm/codex-login-browser?${q}`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{
    ok: boolean;
    generation: number;
    auth_url: string | null;
    device_code?: string | null;
    stale?: boolean;
    message: string;
  }>;
}

export async function postCodexLogout(authJsonPath?: string) {
  const q = authJsonPath?.trim()
    ? `?auth_json_path=${encodeURIComponent(authJsonPath.trim())}`
    : "";
  const res = await fetch(`${base}/api/llm/codex-logout${q}`, { method: "POST" });
  if (!res.ok) {
    const t = await res.text();
    let msg = t;
    try {
      const j = JSON.parse(t) as { detail?: unknown };
      if (typeof j.detail === "string") msg = j.detail;
    } catch {
      /* keep raw body */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<{ ok: boolean; message: string }>;
}

export async function getCodexProfiles(authJsonPath: string) {
  const q = authJsonPath.trim()
    ? `?path=${encodeURIComponent(authJsonPath.trim())}`
    : "";
  const res = await fetch(`${base}/api/llm/codex-profiles${q}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ auth_path: string; profiles: CodexProfileRow[] }>;
}

export async function saveLlmSettings(patch: {
  openai_base_url?: string;
  openai_model?: string;
  chatgpt_account_id?: string;
  openai_api_key?: string;
  codex_auth_profile_id?: string;
  codex_auth_json_path?: string;
  codex_agent_model_list?: string;
  llm_provider?: string;
}): Promise<LlmSettingsResponse> {
  const res = await fetch(`${base}/api/llm/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<LlmSettingsResponse>;
}

export type StudioRunResult = {
  ok: boolean;
  message: string;
  returncode: number;
  stdout: string;
  stderr: string;
  pipeline: { current_step: number; step_label: string; last_url: string; last_site_stem: string };
  /** Slurm stdout log path after ``POST /api/studio/run/training`` (same as ``slurm_out_path`` when set). */
  training_log_path?: string;
  training_pid?: number;
  slurm_job_id?: string;
  slurm_out_path?: string;
  slurm_err_path?: string;
  /** Present when the backend passes custom crop/voxel args into the provider-native script generator. */
  downloader_generations?: {
    mode: string;
    argv: string[];
    env?: Record<string, string>;
    method?: "in_process" | "subprocess";
  }[];
  /** @deprecated Use downloader_generations. Kept for backward compatibility. */
  openorganelle_generations?: {
    mode: string;
    argv: string[];
    env?: Record<string, string>;
    method?: "in_process" | "subprocess";
  }[];
  generated_scripts?: string[];
  evaluation_summary?: {
    n_cases?: number;
    mean_binary_f1?: number;
    mean_binary_precision?: number;
    mean_binary_recall?: number;
    mean_binary_iou?: number;
  } | null;
  evaluation_cases?: Array<Record<string, unknown>> | null;
  /** Local-HPC Stage 3: final console log (same as ``/run/downloader-script-state``). */
  downloader_log?: string;
  /** Local-HPC Stage 3: final progress snapshot ``{ completed, total, current, dataset }``. */
  downloader_progress?: { completed: number; total: number; current: number; dataset: string } | null;
};

export type StudioPendingDownloads = {
  ok: boolean;
  message?: string;
  pending_count: number | null;
  pending_datasets: string[];
  profile_hash: string | null;
  profile?: {
    n_crops: number;
    chunk_zyx: string;
    voxel_nm_zyx: string;
    mode: string;
    foundation: boolean;
  };
};

export async function getStudioPendingDownloads(params: {
  site?: string;
  n_crops: number;
  chunk_zyx?: string;
  voxel_nm_zyx?: string;
  mode?: string;
  foundation?: boolean;
}): Promise<StudioPendingDownloads> {
  const q = new URLSearchParams({
    site: params.site ?? "openorganelle",
    n_crops: String(params.n_crops),
    chunk_zyx: params.chunk_zyx ?? "128,128,128",
    voxel_nm_zyx: params.voxel_nm_zyx ?? "16,16,16",
    mode: params.mode ?? "labeled",
    foundation: String(params.foundation ?? true),
  });
  const res = await fetch(`${base}/api/studio/pending-downloads?${q.toString()}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<StudioPendingDownloads>;
}

export async function getStudioProbes() {
  const res = await fetch(`${base}/api/studio/probes`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ probes: string[] }>;
}

export async function getStudioSites() {
  const res = await fetch(`${base}/api/studio/sites`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ sites: string[] }>;
}

export async function getStudioSummary(sessionId: string) {
  const q = new URLSearchParams({ session_id: sessionId });
  const res = await fetch(`${base}/api/studio/summary?${q}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{
    pipeline: { current_step: number; step_label: string; last_url: string; last_site_stem: string };
    probe_count: number;
    latest_probe: string;
    inventory_sqlite_exists: boolean;
    catalog_db_labeled_ready: number;
    generated_download_scripts: number;
    preprocessed_dir_exists: boolean;
    training_config_exists: boolean;
    registry: {
      exists: boolean;
      providers?: number;
      datasets?: number;
      assets?: number;
      complete_downloads?: number;
      complete_preprocess_runs?: number;
      error?: string;
    };
  }>;
}

export async function postStudioDatabaseBuild(sessionId: string, probe: string) {
  const res = await fetch(`${base}/api/studio/run/database`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, probe }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<StudioRunResult>;
}

/** Matches server ``_STUDIO_RUN_DOWNLOADER_SYNC_TAG`` — not a real script path; filter from UI lists. */
export const STUDIO_PIPELINE_DOWNLOADER_SYNC_SCRIPT_PATH = "[pipeline] studio_run_downloader --execute";

export async function getStudioDatabaseBuildState(sessionId: string) {
  const q = new URLSearchParams({ session_id: sessionId });
  const res = await fetch(`${base}/api/studio/run/database-state?${q.toString()}`, {
    cache: "no-store",
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
    running: boolean;
    log: string;
    result: StudioRunResult | null;
  }>;
}

export async function postStudioDatabaseBuildStateClear(sessionId: string) {
  const res = await fetch(`${base}/api/studio/run/database-state/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; cleared: boolean }>;
}

export async function postStudioDownloader(
  sessionId: string,
  body: {
    site: string;
    n_crops: number;
    voxel_size_nm: string;
    crop_dimensions_voxels: string;
    /** Per-dataset split: number of training/inference crops (sum clamped per dataset). */
    dataset_splits?: Record<string, { training: number; inference: number }>;
    /** Stage 3 uses good-mito labeled inventory only. */
    data_scope: "labeled";
    execute: boolean;
  },
) {
  const res = await fetch(`${base}/api/studio/run/downloader`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...body }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<StudioRunResult>;
}

export async function postStudioMitoleDownloader(
  sessionId: string,
  body: {
    dataset_splits: Record<string, { training: number; inference: number }>;
    dataset_pairs: Array<{ dataset: string; source: string; image_path: string; label_path: string }>;
  },
) {
  const res = await fetch(`${base}/api/studio/mitole/run/downloader`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...body }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<StudioRunResult & { run_folder?: string; copied_pairs?: number }>;
}

export async function getStudioDownloaderPreview(site: string, dataScope: "labeled" = "labeled") {
  const q = new URLSearchParams({ site: site.trim(), data_scope: dataScope });
  const res = await fetch(`${base}/api/studio/downloader/preview?${q.toString()}`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
    ok: boolean;
    message: string;
    site: string;
    data_scope: string;
    db_path: string;
    count: number;
    datasets: string[];
    dataset_rows?: Array<{
      dataset_name: string;
      sample_type: string;
    }>;
  }>;
}

export async function getStudioDownloaderScripts(site: string, dataScope: "labeled" = "labeled") {
  const q = new URLSearchParams({ site: site.trim(), data_scope: dataScope });
  const res = await fetch(`${base}/api/studio/downloader/scripts?${q.toString()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
    ok: boolean;
    scripts: string[];
  }>;
}

export async function postStudioRunDownloaderScriptCancel(sessionId: string): Promise<{ ok: boolean; killed: boolean }> {
  const res = await fetch(`${base}/api/studio/run/downloader-script-cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; killed: boolean }>;
}

export async function postStudioRunDownloaderScriptStateClear(
  sessionId: string,
): Promise<{ ok: boolean; cleared: boolean }> {
  const res = await fetch(`${base}/api/studio/run/downloader-script-state/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; cleared: boolean }>;
}

export async function getStudioRunDownloaderScriptState(sessionId: string) {
  const q = new URLSearchParams({ session_id: sessionId });
  const res = await fetch(`${base}/api/studio/run/downloader-script-state?${q.toString()}`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
    ok: boolean;
    running: boolean;
    script_path: string;
    log: string;
    progress: { completed: number; total: number; current: number; dataset: string } | null;
    result: StudioRunResult | null;
    updated_at: number;
  }>;
}

export type StudioPreprocessSelectiveQueued = {
  ok: boolean;
  accepted: true;
  message: string;
};

export async function postStudioPreprocessSelective(
  sessionId: string,
  body: {
    dataset_paths: string[];
    task: "supervised";
    output_format?: "h5" | "nifti";
    split_label_cc?: boolean;
    raw_download_folder?: string;
  },
) {
  const res = await fetch(`${base}/api/studio/run/preprocess-selective`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...body }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<StudioPreprocessSelectiveQueued>;
}

export async function getStudioRunPreprocessSelectiveState(sessionId: string, opts?: { signal?: AbortSignal }) {
  const q = new URLSearchParams({ session_id: sessionId });
  const ctrl = new AbortController();
  const tid = typeof window !== "undefined" ? window.setTimeout(() => ctrl.abort(), 60_000) : 0;
  const outer = opts?.signal;
  if (outer) {
    if (outer.aborted) ctrl.abort();
    else outer.addEventListener("abort", () => ctrl.abort(), { once: true });
  }
  try {
    const res = await fetch(`${base}/api/studio/run/preprocess-selective-state?${q.toString()}`, {
      cache: "no-store",
      credentials: "same-origin",
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(await parseApiError(res));
    return (await res.json()) as {
      ok: boolean;
      running: boolean;
      log: string;
      progress: { completed: number; total: number; current: number; dataset: string } | null;
      result: StudioRunResult | null;
      updated_at: number;
    };
  } finally {
    if (tid) window.clearTimeout(tid);
  }
}

export async function postStudioRunPreprocessSelectiveStateClear(
  sessionId: string,
): Promise<{ ok: boolean; cleared: boolean }> {
  const res = await fetch(`${base}/api/studio/run/preprocess-selective-state/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; cleared: boolean }>;
}

export async function postStudioRunPreprocessSelectiveCancel(
  sessionId: string,
): Promise<{ ok: boolean; killed?: boolean; warning?: string; error?: string }> {
  const res = await fetch(`${base}/api/studio/run/preprocess-selective-cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
    cache: "no-store",
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return (await res.json()) as { ok: boolean; killed?: boolean; warning?: string; error?: string };
}

export async function getStudioDataInspect(opts?: {
  /** Default true: fast listing without opening every HDF5 (Studio UI). */
  shallow?: boolean;
  /**
   * When shallow is true, still open HDF5/NIfTI only under this path relative to ``data/raw``
   * (e.g. ``my_run/labels``) so the Stage-4 raw viewer can show dimensions and label counts.
   */
  deepUnder?: string | null;
  signal?: AbortSignal;
}) {
  const shallow = opts?.shallow !== false;
  const q = new URLSearchParams();
  q.set("cb", String(Date.now()));
  q.set("shallow", shallow ? "1" : "0");
  const du = typeof opts?.deepUnder === "string" ? opts.deepUnder.trim() : "";
  if (du) q.set("deep_under", du);
  const res = await fetch(`${base}/api/studio/data/inspect?${q.toString()}`, {
    cache: "no-store",
    credentials: "same-origin",
    signal: opts?.signal,
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
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
  }>;
}

export async function getStudioPostprocessingFiles(signal?: AbortSignal) {
  const q = new URLSearchParams();
  q.set("cb", String(Date.now()));
  const res = await fetch(`${base}/api/studio/postprocessing/files?${q.toString()}`, {
    cache: "no-store",
    credentials: "same-origin",
    signal,
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
    ok: boolean;
    input_dir: string;
    output_dir: string;
    files: {
      name: string;
      path: string;
      type: string;
      dimensions: number[];
      spacing: number[];
      label_summary?: string;
      source: "input" | "output";
    }[];
  }>;
}

/** Same filesystem rule as stage-4 preprocess discovery for selected runs. */
export async function getStudioRawEmStacks(run: string, signal?: AbortSignal) {
  const q = new URLSearchParams();
  q.set("run", run.trim());
  const res = await fetch(`${base}/api/studio/data/raw-em-stacks?${q.toString()}`, {
    cache: "no-store",
    credentials: "same-origin",
    signal,
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
    ok: boolean;
    run: string;
    count: number;
    images_dir?: string;
    sample_other_files?: string[];
    detail?: string;
    reason?: string;
  }>;
}

export async function postStudioTraining(sessionId: string) {
  const res = await fetch(`${base}/api/studio/run/training`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<StudioRunResult>;
}

export async function postStudioInference(sessionId: string) {
  const res = await fetch(`${base}/api/studio/run/inference`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<StudioRunResult>;
}

export async function postStudioPostprocessing(
  sessionId: string,
  body: { input_dir: string; output_dir: string },
) {
  const res = await fetch(`${base}/api/studio/run/postprocessing`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...body }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<StudioRunResult>;
}

export async function postStudioEvaluation(
  sessionId: string,
  body: { pred_dir: string; gt_dir: string },
) {
  const res = await fetch(`${base}/api/studio/run/evaluation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...body }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<StudioRunResult>;
}

export type StudioSlurmRunState = {
  ok: boolean;
  running?: boolean;
  selected_log_root?: string;
  log_roots?: string[];
  out_path: string;
  err_path: string;
  out_log: string;
  err_log: string;
  summary?: {
    complete: boolean;
    headline: string;
    runtime: string;
    ended_at: string;
    mean_validation_dice?: number | null;
    best_ema_pseudo_dice?: number | null;
    final_epoch?: number | null;
    final_train_loss?: number | null;
    final_val_loss?: number | null;
    final_pseudo_dice?: { values: number[]; mean: number } | null;
  } | null;
  result: StudioRunResult | null;
  updated_at: number;
};

export async function getStudioTrainingState(sessionId: string, logRoot = "") {
  const q = new URLSearchParams({ session_id: sessionId });
  if (logRoot.trim()) q.set("log_root", logRoot.trim());
  const res = await fetch(`${base}/api/studio/run/training-state?${q.toString()}`, {
    cache: "no-store",
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<StudioSlurmRunState>;
}

export async function postStudioTrainingStateClear(sessionId: string) {
  const res = await fetch(`${base}/api/studio/run/training-state/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; cleared: boolean }>;
}

export async function getStudioInferenceState(sessionId: string, logRoot = "") {
  const q = new URLSearchParams({ session_id: sessionId });
  if (logRoot.trim()) q.set("log_root", logRoot.trim());
  const res = await fetch(`${base}/api/studio/run/inference-state?${q.toString()}`, {
    cache: "no-store",
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<StudioSlurmRunState>;
}

export async function postStudioInferenceStateClear(sessionId: string) {
  const res = await fetch(`${base}/api/studio/run/inference-state/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; cleared: boolean }>;
}

export type StudioWebsiteSummary = {
  slug: string;
  display_name: string;
  url: string;
  data_focus: string;
  description: string;
  updated_at: string;
  description_preview: string;
  datasets_count: number;
};

export type StudioWebsiteScrapeResult = {
  ok: boolean;
  slug: string;
  folder: string;
  site_md: string;
  /** Unused; kept for older API clients (always empty). */
  datasets_json?: string;
  probe_path: string | null;
  fetch: Record<string, unknown>;
  mito_foundation_bridge?: Record<string, unknown>;
  /** Present when the user stopped the scrape (OpenOrganelle subprocess). */
  cancelled?: boolean;
  pipeline: { current_step: number; step_label: string; last_url: string; last_site_stem: string };
};

export async function getStudioWebsites() {
  const res = await fetch(`${base}/api/studio/websites`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ websites: StudioWebsiteSummary[] }>;
}

export async function getStudioWebsite(slug: string) {
  const res = await fetch(`${base}/api/studio/websites/${encodeURIComponent(slug)}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<StudioWebsiteSummary>;
}

export async function postStudioWebsiteSave(
  sessionId: string,
  body: {
    display_name: string;
    url: string;
    description: string;
    data_focus: string;
    slug?: string;
    /** Only with “overwrite loaded folder”: update that folder instead of allocating the next _NN */
    editing_slug?: string;
  },
) {
  const res = await fetch(`${base}/api/studio/websites/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...body }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ ok: boolean; slug: string; folder: string; site_md: string }>;
}

export async function deleteStudioWebsite(slug: string) {
  const res = await fetch(`${base}/api/studio/websites/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slug: slug.trim() }),
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{
    ok: boolean;
    slug: string;
    folder?: string;
    project_root?: string;
    removed_probe: boolean;
    note?: string;
  }>;
}

export async function postStudioScrapeCancel(sessionId: string): Promise<{ ok: boolean; killed: boolean }> {
  const res = await fetch(`${base}/api/studio/websites/scrape-cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; killed: boolean }>;
}

/** Live or last-completed scrape subprocess log (sync API, stream, or chat pipeline). */
export async function getStudioWebsiteScrapeState(sessionId: string): Promise<{ running: boolean; log: string }> {
  const q = new URLSearchParams({ session_id: sessionId });
  const res = await fetch(`${base}/api/studio/websites/scrape-state?${q.toString()}`, { credentials: "same-origin" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ running: boolean; log: string }>;
}

export async function postStudioWebsiteScrapeStateClear(
  sessionId: string,
): Promise<{ ok: boolean; cleared: boolean }> {
  const res = await fetch(`${base}/api/studio/websites/scrape-state/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<{ ok: boolean; cleared: boolean }>;
}

/** Stream workspace scrape logs (SSE) then final result — same inputs as ``postStudioWebsiteScrape``. */
export async function postStudioWebsiteScrapeStream(
  sessionId: string,
  body:
    | { slug: string }
    | {
        display_name: string;
        url: string;
        description: string;
        data_focus: string;
        slug?: string;
      },
  handlers: {
    onLog: (text: string) => void;
    onComplete: (result: StudioWebsiteScrapeResult) => void;
    onError: (message: string) => void;
  },
  options?: { signal?: AbortSignal },
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${base}/api/studio/websites/scrape-stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, ...body }),
      credentials: "same-origin",
      signal: options?.signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") return;
    throw e;
  }
  if (!res.ok) {
    handlers.onError(await parseApiError(res));
    return;
  }
  const reader = res.body?.getReader();
  if (!reader) {
    handlers.onError("No response body");
    return;
  }
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    let chunk: ReadableStreamReadResult<Uint8Array>;
    try {
      chunk = await reader.read();
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      throw e;
    }
    const { done, value } = chunk;
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let sep: number;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      let dataLine = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("data: ")) dataLine = line.slice(6).trim();
      }
      if (!dataLine) continue;
      let ev: unknown;
      try {
        ev = JSON.parse(dataLine);
      } catch {
        continue;
      }
      if (!ev || typeof ev !== "object") continue;
      const o = ev as { type?: string };
      if (o.type === "log" && "text" in o) {
        handlers.onLog(String((o as { text: string }).text));
      } else if (o.type === "error" && "message" in o) {
        handlers.onError(String((o as { message: string }).message));
        return;
      } else if (o.type === "done" && "payload" in o) {
        handlers.onComplete((o as { payload: StudioWebsiteScrapeResult }).payload);
        return;
      }
    }
  }
  handlers.onError("Stream ended without a final result");
}

// ── Database Catalog Viewer ────────────────────────────────────────────────────

export type CatalogDatabase = {
  stem: string;
  path: string;
  exists: boolean;
  size_bytes: number | null;
};

export type CatalogFilterSpec = {
  organisms: string[];
  sample_types: string[];
  sample_subtypes: string[];
  bio_targets: string[];
  bio_target_types: string[];
  stages: string[];
  organelles_present: string[];
  organelle_labels: Record<string, string>;
  content_types: string[];
  layer_roles: string[];
  mito_mask_qualities: (string | null)[];
  boolean_facets: Record<string, { label: string; count: number }>;
};

export type CatalogDatasetRow = {
  id: number;
  dataset_name: string;
  sample_organism: string | null;
  sample_type: string | null;
  sample_subtype: string | null;
  bio_target: string | null;
  bio_target_type: string | null;
  bio_target_source: string | null;
  bio_target_confidence: number | null;
  stage: string | null;
  description: string | null;
  mitochondria_in_layer_names: number;
  download_mito_mask_quality: string | null;
  s3_probe_img_path: string | null;
  s3_probe_error: string | null;
  ready_labeled: number | null;
  ready_em_only: number | null;
  path_source: string | null;
  resolved_voxel_nm: string | null;
  n_gt: number;
  n_pred: number;
  n_raw: number;
  n_em: number;
};

export type CatalogDatasetList = {
  total: number;
  offset: number;
  limit: number;
  datasets: CatalogDatasetRow[];
};

export type CatalogLayer = {
  id: number;
  dataset_id: number;
  layer_role: string;
  layer_kind: string;
  layer_name: string;
  layer_format: string | null;
  layer_stage: string | null;
  layer_url: string | null;
  layer_content_type: string | null;
  semantic: number;
  raw_token: string | null;
};

export type CatalogDatasetDetail = {
  dataset: Record<string, unknown>;
  layers: CatalogLayer[];
  resolved: {
    dataset_id: number;
    dataset_name: string;
    resolved_em_path: string | null;
    resolved_mito_seg_path: string | null;
    resolved_voxel_nm: string | null;
    path_source: string | null;
    ready_labeled: number;
    ready_em_only: number;
  } | null;
  sources: string[];
};

export async function getCatalogDatabases(): Promise<{ databases: CatalogDatabase[] }> {
  const res = await fetch(`${base}/api/studio/catalog/databases`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function getCatalogFilters(stem: string): Promise<CatalogFilterSpec> {
  const res = await fetch(`${base}/api/studio/catalog/${encodeURIComponent(stem)}/filters`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export type CatalogQueryFilters = {
  stage?: string[];
  organism?: string[];
  sample_type?: string[];
  sample_subtype?: string[];
  bio_target?: string[];
  bio_target_type?: string[];
  mito_mask_quality?: string[];
  content_type?: string[];
  layer_role?: string[];
  organelle?: string[];
  has_s3_probe?: boolean;
  ready_labeled?: boolean;
  ready_em?: boolean;
  has_mito_gt?: boolean;
  has_good_mito_mask?: boolean;
  has_em_layers?: boolean;
  has_predictions?: boolean;
  limit?: number;
  offset?: number;
};

export async function getCatalogDatasets(
  stem: string,
  filters: CatalogQueryFilters = {},
): Promise<CatalogDatasetList> {
  const q = new URLSearchParams();
  const appendList = (key: string, vals: string[] | undefined) =>
    vals?.forEach((v) => q.append(key, v));
  appendList("stage", filters.stage);
  appendList("organism", filters.organism);
  appendList("sample_type", filters.sample_type);
  appendList("sample_subtype", filters.sample_subtype);
  appendList("bio_target", filters.bio_target);
  appendList("bio_target_type", filters.bio_target_type);
  appendList("mito_mask_quality", filters.mito_mask_quality);
  appendList("content_type", filters.content_type);
  appendList("layer_role", filters.layer_role);
  appendList("organelle", filters.organelle);
  const setB = (k: string, v: boolean | undefined) => {
    if (v != null) q.set(k, v ? "true" : "false");
  };
  setB("has_s3_probe", filters.has_s3_probe);
  setB("ready_labeled", filters.ready_labeled);
  setB("ready_em", filters.ready_em);
  setB("has_mito_gt", filters.has_mito_gt);
  setB("has_good_mito_mask", filters.has_good_mito_mask);
  setB("has_em_layers", filters.has_em_layers);
  setB("has_predictions", filters.has_predictions);
  if (filters.limit != null) q.set("limit", String(filters.limit));
  if (filters.offset != null) q.set("offset", String(filters.offset));
  const res = await fetch(
    `${base}/api/studio/catalog/${encodeURIComponent(stem)}/datasets?${q.toString()}`,
  );
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function getCatalogDatasetDetail(
  stem: string,
  name: string,
): Promise<CatalogDatasetDetail> {
  const res = await fetch(
    `${base}/api/studio/catalog/${encodeURIComponent(stem)}/dataset/${encodeURIComponent(name)}`,
  );
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

// ── Batch provenance + dataset management ────────────────────────────────────

export interface BatchItem {
  id: number;
  stable_id: string;
  asset_type: string;
  local_path: string | null;
  status: "pending" | "present" | "missing_or_deleted_local" | "failed";
  completed_at: string | null;
}

export interface DownloadBatch {
  id: number;
  batch_id: string;
  provider: string;
  profile_hash: string | null;
  profile: Record<string, unknown>;
  run_folder: string | null;
  status: string;
  created_at: string;
  finished_at: string | null;
  n_items: number;
  n_present: number;
  n_missing: number;
  items: BatchItem[];
}

export interface DatasetStatusItem {
  stable_id: string;
  filename: string;
  path: string;
  size_bytes: number;
  hidden_from_training: boolean;
  hidden_from_inference: boolean;
  batch_ids: string[];
}

export async function getDatasetsStatus(provider?: string): Promise<{
  ok: boolean;
  provider: string;
  preprocessed_base: string;
  datasets: DatasetStatusItem[];
}> {
  const q = new URLSearchParams();
  if (provider) q.set("provider", provider);
  const res = await fetch(`${base}/api/studio/datasets/status?${q.toString()}`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function deleteDatasetFiles(filePaths: string[], provider = "OpenOrganelle"): Promise<{
  ok: boolean;
  deleted_files: string[];
  errors: string[];
  message: string;
}> {
  const res = await fetch(`${base}/api/studio/datasets/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_paths: filePaths, provider }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function hideDatasets(stableIds: string[], provider = "OpenOrganelle"): Promise<{
  ok: boolean;
  hidden: string[];
  not_found: string[];
  message: string;
}> {
  const res = await fetch(`${base}/api/studio/datasets/hide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stable_ids: stableIds, provider }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function unhideDatasets(stableIds: string[], provider = "OpenOrganelle"): Promise<{
  ok: boolean;
  unhidden: string[];
  not_found: string[];
  message: string;
}> {
  const res = await fetch(`${base}/api/studio/datasets/unhide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stable_ids: stableIds, provider }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function setDatasetUseInModel(
  stableId: string,
  source: "training" | "inference",
  useInModel: boolean,
  provider = "OpenOrganelle",
): Promise<{
  ok: boolean;
  stable_id: string;
  source: "training" | "inference";
  use_in_model: boolean;
  message: string;
}> {
  const res = await fetch(`${base}/api/studio/datasets/use-in-model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      stable_id: stableId,
      source,
      use_in_model: useInModel,
      provider,
    }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

// ── Inventory ────────────────────────────────────────────────────────────────

export interface CatalogueRow {
  item_id: number;
  stable_id: string;
  provider: string;
  batch_id: string;
  data_source: "training" | "inference" | "unknown";
  asset_type: string;
  status: "pending" | "present" | "missing_or_deleted_local" | "failed";
  local_path: string | null;
  hidden_from_training: boolean;
  completed_at: string | null;
  batch_created_at: string | null;
  profile_hash: string | null;
  profile: Record<string, unknown>;
}

export interface CatalogueSummary {
  total_items: number;
  /** Table row count (registry + optional synthetic on-disk rows). */
  inventory_row_count?: number;
  distinct_datasets: number;
  present: number;
  missing_or_deleted: number;
  /** File-level delete log size (may exceed missing batch_items). */
  deletion_events_count?: number;
  /** Sum of per-batch asset download counts (and legacy ``downloads`` fallback). */
  download_completions_total?: number;
  pending: number;
  failed: number;
  hidden_from_training: number;
  on_disk_pairs: number;
  on_disk_pairs_training?: number;
  on_disk_pairs_inference?: number;
  on_disk_images: number;
  on_disk_labels: number;
  delete_history: {
    stable_id: string;
    asset_type: string;
    local_path: string;
    deleted_at: string;
    provider: string;
  }[];
  providers: Record<string, number>;
  batches: {
    batch_id: string;
    display_title?: string;
    provider: string;
    profile: Record<string, unknown>;
    profile_hash?: string | null;
    created_at: string | null;
    n_items: number;
    n_present: number;
    n_missing: number;
    /** em + seg assets recorded for this batch run (sums into catalogue Log). */
    download_asset_completions?: number;
    training_units_this_run?: number;
    inference_units_this_run?: number;
  }[];
}

export interface InventoryCatalogueResponse {
  ok: boolean;
  registry_exists: boolean;
  summary: CatalogueSummary;
  rows: CatalogueRow[];
  error?: string;
}

export async function getInventoryCatalogue(provider?: string): Promise<InventoryCatalogueResponse> {
  const q = new URLSearchParams();
  if (provider) q.set("provider", provider);
  q.set("_ts", String(Date.now()));
  const res = await fetch(`${base}/api/studio/inventory/catalogue?${q.toString()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json() as Promise<InventoryCatalogueResponse>;
}

export async function postStudioResetDownloadedTraining(sessionId: string): Promise<{
  ok: boolean;
  message: string;
  training_root: string;
  inference_root?: string;
  deleted_files: number;
  deleted_dirs: number;
  deleted_paths: string[];
  delete_errors: string[];
  registry: {
    registry_exists: boolean;
    downloads_deleted: number;
    preprocess_runs_deleted: number;
    batch_items_deleted: number;
    download_batches_deleted: number;
    deletion_events_deleted?: number;
    datasets_unhidden: number;
  };
}> {
  const res = await fetch(`${base}/api/studio/inventory/reset-downloaded-training`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function postStudioResetModelDownloadedDataHistory(sessionId: string): Promise<{
  ok: boolean;
  message: string;
  targets: string[];
  deleted_files: number;
  deleted_dirs: number;
  deleted_paths: string[];
  delete_errors: string[];
}> {
  const res = await fetch(`${base}/api/studio/model/reset-downloaded-data-history`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export type MitoLeInspectRow = {
  name: string;
  path: string;
  type: string;
  dimensions: number[];
  spacing: number[];
  label_summary?: string;
  folder: string;
};

export async function getMitoLeConfig(): Promise<{
  ok: boolean;
  base_path: string;
  default_folders: string[];
  folders: string[];
}> {
  const res = await fetch(`${base}/api/studio/mitole/config`, { cache: "no-store" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function getMitoLeSubfolders(): Promise<{
  ok: boolean;
  base_path: string;
  subfolders: string[];
}> {
  const res = await fetch(`${base}/api/studio/mitole/subfolders`, { cache: "no-store" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function postMitoLeConfig(folders: string[]): Promise<{
  ok: boolean;
  base_path: string;
  folders: string[];
}> {
  const res = await fetch(`${base}/api/studio/mitole/config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folders }),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function getMitoLeInspect(folder = "__all__"): Promise<{
  ok: boolean;
  base_path: string;
  folders: string[];
  rows: MitoLeInspectRow[];
}> {
  const q = new URLSearchParams({ folder, _ts: String(Date.now()) });
  const res = await fetch(`${base}/api/studio/mitole/inspect?${q.toString()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export type MitoLeCatalogueRow = {
  dataset: string;
  folder: string;
  source: string;
  organism: string;
  sample_type: string;
  image_file?: string;
  label_file?: string;
  image_path?: string;
  label_path?: string;
  dimensions?: number[];
  spacing?: number[];
};

export async function getMitoLeCatalogue(regenerate = false): Promise<{
  ok: boolean;
  base_path: string;
  rows: MitoLeCatalogueRow[];
  filters: Record<string, string[]>;
}> {
  const q = new URLSearchParams({ _ts: String(Date.now()), regenerate: regenerate ? "1" : "0" });
  const res = await fetch(`${base}/api/studio/mitole/catalogue?${q.toString()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}
