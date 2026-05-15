import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import type {
  ChatMessage,
  ChatMode,
  ChatThreadSummary,
  CodexProfileRow,
  PendingApproval,
  PipelineExecutionMeta,
  PipelinePlanRecord,
  PlanningHint,
} from "./api";
import { AgentSkillsSettings } from "./AgentSkillsSettings";
import { MarkdownPreview } from "./MarkdownPreview";
import { PipelineStudio } from "./PipelineStudio";
import { STUDIO_SELECT_LOADING, StudioUpdatingBadge } from "./StudioUi";
import {
  createChat,
  deleteChats,
  deletePipelinePlan,
  getCodexProfiles,
  getChatState,
  getPipelineProgress,
  getLlmSettings,
  getLlmStatus,
  listChats,
  listPipelinePlans,
  normalizePipelineExecution,
  postChat,
  postChatStop,
  postChatEdit,
  postCodexLoginBrowser,
  postCodexLogout,
  resolveApproval,
  saveLlmSettings,
  savePipelinePlan,
} from "./api";

const CODEX_BASE = "https://chatgpt.com/backend-api/codex";
/** ChatGPT Codex HTTP uses this model only (no UI switching). */
const CODEX_FIXED_MODEL = "gpt-5.4";

const LS_CHAT_WIDTH = "mito2_chat_panel_width";
const LS_CHAT_COLLAPSED = "mito2_chat_panel_collapsed";
const LS_ACTIVE_CHAT_ID = "mito2_active_chat_id";
const LS_CHAT_HISTORY_COLLAPSED = "mito2_chat_history_collapsed";
const LS_CHAT_MODE = "mito2_chat_mode";
const LS_CHAT_INFLIGHT = "mito2_chat_inflight_v1";
/** Per-thread: user dismissed "Approve & run" for a plan signature (survives refresh; cleared when a new plan is offered). */
const SS_AGENT_PLAN_DISMISSED = "mito2_agent_plan_dismissed_v1";
/** Per-thread: user dismissed/saved Plan-mode draft card for a plan signature (survives refresh). */
const SS_DRAFT_PLAN_DISMISSED = "mito2_draft_plan_dismissed_v1";
const EXECUTE_APPROVED_PIPELINE_USER_LINE = "Execute approved pipeline plan.";
/** Minimum width of the main workspace when the chat panel is expanded (keep small so chat can widen symmetrically). */
const CHAT_RESIZE_WORKSPACE_MIN_PX = 160;
const CHAT_SPLIT_RAIL_PX = 11;
const CHAT_WIDTH_MIN_PX = 220;
const CHAT_WIDTH_DEFAULT = 400;
const CHAT_COLLAPSED_PX = 44;

type ChatRenderPhase = "idle" | "waiting" | "rendering";
type PipelineStage = "scrape" | "database" | "download" | "training";

function normalizePipelineStage(raw: unknown): PipelineStage | null {
  if (raw === "scrape" || raw === "database" || raw === "download" || raw === "training") return raw;
  if (raw === "train") return "training";
  if (raw === "schema") return "database";
  return null;
}
type PreparingContext = {
  label: string;
  stage: PipelineStage | null;
  /** From client-side draft / approved plan only. */
  provider: string;
  /** From LLM `ui_focus` + streamed planning hint (replaces keyword heuristics for the status line). */
  llmSourceLabel: string;
  planSites: string[];
  planStages: PipelineStage[];
  /** User message for this turn; used to infer named providers before streamed hints arrive (mirrors backend merge). */
  userPrompt: string;
  /** Long chat-side pipeline execute: poll `/api/pipeline` to refresh the waiting bubble and chrome. */
  pollExecutionProgress: boolean;
  /** Human line from last pipeline poll, e.g. "Stage 2 database build · BossDB". */
  executionLiveLine: string;
};

type InflightChatState = {
  sessionId: string;
  userText: string;
  chatMode: ChatMode;
  waitingCtx: PreparingContext;
  startedAt: number;
};

function splitReplyIntoChunks(text: string): string[] {
  const chunks = text.match(/(\s+|[^\s]+)/g);
  if (!chunks || chunks.length === 0) return [text];
  return chunks;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function toProviderLabel(raw: string): string {
  const token = (raw || "").trim();
  if (!token) return "";
  const low = token.toLowerCase();
  if (low === "openorganelle") return "OpenOrganelle";
  if (low === "bossdb") return "BossDB";
  return token.charAt(0).toUpperCase() + token.slice(1);
}

function siteStemToDisplayLabel(stem: string): string {
  const s = (stem || "").trim();
  if (!s) return "";
  const low = s.toLowerCase().replace(/[\s_]+/g, "");
  if (low.includes("openorganelle")) return "OpenOrganelle";
  if (low.includes("bossdb")) return "BossDB";
  const canon = toProviderLabel(s);
  return canon || s;
}

function formatExecutionProgressTitle(stepLabel: string, siteStem: string): string {
  const site = siteStemToDisplayLabel(siteStem);
  const suffix = site ? ` · ${site}` : "";
  if (stepLabel === "idle") return `Completing pipeline${suffix}`.trim();
  if (stepLabel === "scrape") return `Stage 1 scrape${suffix}`;
  if (stepLabel === "database_build") return `Stage 2 database build${suffix}`;
  if (stepLabel === "download_script") return `Stage 3 download${suffix}`;
  if (stepLabel === "preprocess") return `Stage 3 preprocess${suffix}`;
  if (stepLabel === "model_training") return `Stage 4 training${suffix}`;
  return `${stepLabel}${suffix}`;
}

/** Floor elapsed seconds for the planning stepper when ``chat_turn_started_at`` is absent (older server). */
function minWaitingElapsedFromPipelineStep(stepLabel: string | undefined): number {
  const s = (stepLabel || "").trim().toLowerCase();
  if (s === "scrape") return 30;
  if (s === "database_build") return 45;
  if (s === "download_script") return 60;
  if (s === "preprocess") return 75;
  return 15;
}

/** Epoch ms when the current waiting phase began (from server or inferred for UI continuity). */
function waitingEpochMsFromChatResume(st: {
  chat_turn_started_at?: number | null;
  pipeline?: { step_label?: string };
}): number {
  const raw = st.chat_turn_started_at;
  if (typeof raw === "number" && Number.isFinite(raw) && raw > 0) {
    if (raw >= 1e12) return Math.floor(raw);
    return Math.floor(raw * 1000);
  }
  const synth = minWaitingElapsedFromPipelineStep(st.pipeline?.step_label);
  return Date.now() - synth * 1000;
}

function agentPlanSignature(plan: Record<string, unknown>): string {
  try {
    return JSON.stringify({
      sites: Array.isArray(plan.sites) ? plan.sites : [],
      stages: Array.isArray(plan.stages) ? plan.stages : [],
      n_crops: plan.n_crops,
    });
  } catch {
    return "";
  }
}

function readAgentPlanDismissedMap(): Record<string, string> {
  if (typeof window === "undefined") return {};
  try {
    const raw = sessionStorage.getItem(SS_AGENT_PLAN_DISMISSED);
    const o = raw ? (JSON.parse(raw) as Record<string, string>) : {};
    return o && typeof o === "object" ? o : {};
  } catch {
    return {};
  }
}

function persistAgentPlanDismissed(chatId: string, plan: Record<string, unknown>): void {
  if (typeof window === "undefined" || !chatId) return;
  const sig = agentPlanSignature(plan);
  if (!sig) return;
  const m = readAgentPlanDismissedMap();
  m[chatId] = sig;
  sessionStorage.setItem(SS_AGENT_PLAN_DISMISSED, JSON.stringify(m));
}

function clearAgentPlanDismissed(chatId: string): void {
  if (typeof window === "undefined" || !chatId) return;
  const m = readAgentPlanDismissedMap();
  if (m[chatId] === undefined) return;
  delete m[chatId];
  sessionStorage.setItem(SS_AGENT_PLAN_DISMISSED, JSON.stringify(m));
}

function isAgentPlanDismissed(chatId: string, plan: Record<string, unknown>): boolean {
  return readAgentPlanDismissedMap()[chatId] === agentPlanSignature(plan);
}

function readDraftPlanDismissedMap(): Record<string, string> {
  if (typeof window === "undefined") return {};
  try {
    const raw = sessionStorage.getItem(SS_DRAFT_PLAN_DISMISSED);
    const o = raw ? (JSON.parse(raw) as Record<string, string>) : {};
    return o && typeof o === "object" ? o : {};
  } catch {
    return {};
  }
}

function persistDraftPlanDismissed(chatId: string, plan: Record<string, unknown>): void {
  if (typeof window === "undefined" || !chatId) return;
  const sig = agentPlanSignature(plan);
  if (!sig) return;
  const m = readDraftPlanDismissedMap();
  m[chatId] = sig;
  sessionStorage.setItem(SS_DRAFT_PLAN_DISMISSED, JSON.stringify(m));
}

function clearDraftPlanDismissed(chatId: string): void {
  if (typeof window === "undefined" || !chatId) return;
  const m = readDraftPlanDismissedMap();
  if (m[chatId] === undefined) return;
  delete m[chatId];
  sessionStorage.setItem(SS_DRAFT_PLAN_DISMISSED, JSON.stringify(m));
}

function isDraftPlanDismissed(chatId: string, plan: Record<string, unknown>): boolean {
  return readDraftPlanDismissedMap()[chatId] === agentPlanSignature(plan);
}

function indexOfLastAssistantAwaitingAgentApproval(messages: ChatMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== "assistant") continue;
    const c = m.content || "";
    if (
      c.includes("Use **Approve & run** to execute it.") ||
      c.includes("executable pipeline plan for approval") ||
      c.includes("Prepared an executable pipeline plan for approval") ||
      c.includes("Prepared a structured pipeline plan for approval.")
    ) {
      return i;
    }
  }
  return -1;
}

/** Rehydrate the "Approve & run" bar after refresh when the user has not executed or dismissed this plan. */
function shouldRestoreAgentPendingPlan(
  chatId: string,
  chatMode: ChatMode,
  messages: ChatMessage[],
  lastPlan: unknown,
  options?: { serverAgentPlanAwaiting?: boolean },
): Record<string, unknown> | null {
  const serverFlag = options?.serverAgentPlanAwaiting === true;
  if (!serverFlag && chatMode !== "agent") return null;
  if (!lastPlan || typeof lastPlan !== "object") return null;
  const plan = lastPlan as Record<string, unknown>;
  if (!Array.isArray(plan.sites) || plan.sites.length === 0) return null;
  if (!Array.isArray(plan.stages) || plan.stages.length === 0) return null;
  if (isAgentPlanDismissed(chatId, plan)) return null;

  if (serverFlag) {
    return plan;
  }
  const idx = indexOfLastAssistantAwaitingAgentApproval(messages);
  if (idx < 0) return null;

  for (let j = idx + 1; j < messages.length; j++) {
    const u = messages[j];
    if (u.role === "user" && (u.content || "").trim() === EXECUTE_APPROVED_PIPELINE_USER_LINE) {
      return null;
    }
  }
  return plan;
}

/** Rehydrate Plan-mode draft card after refresh (``last_pipeline_plan`` + last structured-plan assistant turn). */
function shouldRestoreDraftPipelinePlan(
  chatId: string,
  chatMode: ChatMode,
  messages: ChatMessage[],
  lastPlan: unknown,
): Record<string, unknown> | null {
  if (chatMode !== "plan" || !lastPlan || typeof lastPlan !== "object") return null;
  const plan = lastPlan as Record<string, unknown>;
  if (!Array.isArray(plan.sites) || plan.sites.length === 0) return null;
  if (!Array.isArray(plan.stages) || plan.stages.length === 0) return null;
  if (isDraftPlanDismissed(chatId, plan)) return null;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== "assistant") continue;
    const c = m.content || "";
    if (
      c.includes("Prepared a structured pipeline plan for approval.") ||
      c.includes("Prepared a draft pipeline plan.")
    ) {
      return plan;
    }
  }
  return null;
}

function stageToAction(stage: PipelineStage): string {
  if (stage === "scrape") return "Scraping Website";
  if (stage === "database") return "Building Database";
  if (stage === "download") return "Downloading Datasets";
  return "Starting Model Training";
}

function asPipelinePlanLike(plan: unknown): { sites: string[]; stages: string[] } {
  if (!plan || typeof plan !== "object") return { sites: [], stages: [] };
  const row = plan as { sites?: unknown; stages?: unknown };
  const sites = Array.isArray(row.sites) ? row.sites.filter((x): x is string => typeof x === "string") : [];
  const stagesRaw = Array.isArray(row.stages) ? row.stages.filter((x): x is string => typeof x === "string") : [];
  const stages = stagesRaw
    .map((x) => (x === "schema" ? "database" : x))
    .filter((x): x is PipelineStage => x === "scrape" || x === "database" || x === "download" || x === "training");
  return { sites, stages };
}

function mergeWaitingCtxFromPlanningHint(prev: PreparingContext, hint: PlanningHint): PreparingContext {
  const nextSitesRaw = (hint.plan_sites ?? [])
    .map((s) => toProviderLabel(s))
    .filter(Boolean);
  const mergedSites = nextSitesRaw.length > 0 ? nextSitesRaw : prev.planSites;
  const nextStages = (hint.plan_stages ?? [])
    .map((s) => normalizePipelineStage(s))
    .filter((s): s is PipelineStage => s !== null);
  const uif = (hint.planning_ui_focus || "").trim();
  // Match backend: structured sites list wins for multi-source display; don't let a one-line ui_focus drop a provider.
  const displaySource = mergedSites.length > 0 ? mergedSites.join(" → ") : uif;
  const nextStage: PipelineStage | null =
    nextStages.length > 0 ? (nextStages[0] as PipelineStage) : (prev.stage ?? null);
  return {
    ...prev,
    llmSourceLabel: displaySource,
    planSites: mergedSites.length > 0 ? mergedSites : prev.planSites,
    planStages: nextStages.length > 0 ? nextStages : prev.planStages,
    stage: nextStage,
  };
}

function buildPreparingResponseLabel(opts: {
  userText: string;
  pipelinePlan?: unknown;
}): PreparingContext {
  const userPrompt = (opts.userText || "").trim();
  const planLike = asPipelinePlanLike(opts.pipelinePlan);
  const planStages = planLike.stages.filter(
    (s): s is PipelineStage => s === "scrape" || s === "database" || s === "download" || s === "training",
  );
  const planStage = planStages[0] ?? null;
  const planSites = planLike.sites.map((s) => toProviderLabel(s)).filter(Boolean);
  const planProviderRaw = planSites[0] || planLike.sites[0] || "";
  const planProvider = toProviderLabel(planProviderRaw);
  /** All selected sources, for multi-site requests (never use first site only in user-visible copy). */
  const allSourcesLabel = planSites.length > 0 ? planSites.join(" → ") : planProvider;

  const msg = (opts.userText || "").toLowerCase();
  const pollExecutionProgress =
    /\bexecute\s+approved\s+pipeline\s+plan\b/.test(msg) || /\brun\s+plan\b/.test(msg);
  if (pollExecutionProgress) {
    if (planStage) {
      const stageIndex = planStage === "scrape" ? 1 : planStage === "database" ? 2 : 3;
      const stageIndexResolved = planStage === "training" ? 4 : stageIndex;
      const stageVerb = planStage === "scrape"
        ? "scrape"
        : planStage === "database"
          ? "database build"
          : planStage === "download"
            ? "download"
            : "training";
      const providerTxt = allSourcesLabel ? ` · ${allSourcesLabel}` : "";
      return {
        label: `Preparing Response: Stage ${stageIndexResolved}${providerTxt} ${stageVerb}`,
        stage: planStage,
        provider: planProvider,
        llmSourceLabel: "",
        planSites,
        planStages,
        userPrompt,
        pollExecutionProgress: true,
        executionLiveLine: "",
      };
    }
    return {
      label: "Preparing Response: Executing Approved Pipeline Plan",
      stage: null,
      provider: "",
      llmSourceLabel: "",
      planSites,
      planStages,
      userPrompt,
      pollExecutionProgress: true,
      executionLiveLine: "",
    };
  }

  const hasExplicitPlan = planStages.length > 0;
  const stage = hasExplicitPlan ? (planStage ?? null) : null;

  if (stage) {
    const action = stageToAction(stage);
    return {
      label: allSourcesLabel ? `Preparing Response: ${action} for ${allSourcesLabel}` : `Preparing Response: ${action}`,
      stage,
      provider: planProvider,
      llmSourceLabel: "",
      planSites,
      planStages,
      userPrompt,
      pollExecutionProgress: false,
      executionLiveLine: "",
    };
  }
  if (allSourcesLabel && !hasExplicitPlan) {
    return {
      label: `Preparing Response: Planning pipeline for ${allSourcesLabel}`,
      stage: null,
      provider: planProvider,
      llmSourceLabel: "",
      planSites,
      planStages,
      userPrompt,
      pollExecutionProgress: false,
      executionLiveLine: "",
    };
  }
  return {
    label: "Preparing response",
    stage: null,
    provider: "",
    llmSourceLabel: "",
    planSites,
    planStages,
    userPrompt,
    pollExecutionProgress: false,
    executionLiveLine: "",
  };
}

/** Shown in the waiting bubble: prefer full site list from streamed LLM/plan (multi-source) over the initial single-shot label. */
function resolveWaitingBubbleTitle(ctx: PreparingContext, initialLabel: string): string {
  const live = (ctx.executionLiveLine || "").trim();
  if (live) return `Preparing Response: ${live}`;
  const source = ctx.planSites.length > 0
    ? ctx.planSites.join(" → ")
    : (ctx.llmSourceLabel || "").trim();
  if (source) {
    if (source.toLowerCase() === "general" && ctx.planSites.length === 0) {
      return initialLabel;
    }
    if (ctx.stage) {
      return `Preparing Response: ${stageToAction(ctx.stage)} for ${source}`;
    }
    return `Preparing Response: Planning pipeline for ${source}`;
  }
  return initialLabel;
}

function waitingPhaseHint(mode: ChatMode, ctx: PreparingContext, elapsedSec: number): string {
  const live = (ctx.executionLiveLine || "").trim();
  if (live) return `Phase: ${live}`;
  if (elapsedSec <= 1) return "Phase: understanding request";
  if (elapsedSec <= 4) return "Phase: routing intent and checking context";
  if (ctx.planStages.length === 0) {
    return mode === "agent" ? "Phase: drafting execution plan" : "Phase: drafting response";
  }
  if (!ctx.stage) return "Phase: drafting response";
  if (ctx.stage === "scrape") {
    return mode === "agent" ? "Phase: Stage 1 scrape workflow in progress" : "Phase: preparing Stage 1 scrape plan";
  }
  if (ctx.stage === "database") {
    return mode === "agent"
      ? "Phase: Stage 2 database builder workflow in progress"
      : "Phase: preparing Stage 2 database plan";
  }
  if (ctx.stage === "download") {
    return mode === "agent" ? "Phase: Stage 3 download workflow in progress" : "Phase: preparing Stage 3 download plan";
  }
  return mode === "agent" ? "Phase: Stage 4 training workflow in progress" : "Phase: preparing Stage 4 training plan";
}

function waitingProgressTrace(mode: ChatMode, ctx: PreparingContext, elapsedSec: number): {
  items: string[];
  activeIndex: number;
} {
  const stageOrder: PipelineStage[] = ctx.planStages.length > 0 ? ctx.planStages : [];
  const siteLabel =
    ctx.planSites.length > 0
      ? ctx.planSites.join(" → ")
      : (ctx.llmSourceLabel || ctx.provider || "your request");
  const moduleForStage = (s: PipelineStage): string =>
    s === "scrape"
      ? "1web_scraper_01"
      : s === "database"
        ? "2database_builder"
        : s === "download"
          ? "3data_downloader"
          : "5model_training";
  const stageText = (s: PipelineStage): string =>
    s === "scrape"
      ? "Scraping website"
      : s === "database"
        ? "Building database/catalog"
        : s === "download"
          ? "Downloading + preprocess"
          : "Submitting training";
  const perStageSec = 14;
  const execIdx = stageOrder.length > 0 ? Math.min(stageOrder.length - 1, Math.floor(Math.max(0, elapsedSec - 8) / perStageSec)) : 0;
  const currentStage = stageOrder[execIdx] ?? "download";
  const stageLabel =
    currentStage === "scrape"
      ? "Stage 1 scrape"
      : currentStage === "database"
        ? "Stage 2 database"
        : currentStage === "download"
          ? "Stage 3 download"
          : "Stage 4 training";
  const providerSuffix = siteLabel ? ` · ${siteLabel}` : "";

  const live = (ctx.executionLiveLine || "").trim();
  const planningTail =
    live
      ? `Pipeline run in progress: ${live}`
      : mode === "agent"
      ? stageOrder.length > 0
        ? `Executing ${stageLabel}${providerSuffix} via ${moduleForStage(currentStage)}`
          : `Drafting executable pipeline plan${providerSuffix}`
        : `Drafting ${stageLabel} plan${providerSuffix}`;

  const items = [
    "Understand request",
    "Route intent and select workflow",
    "Validate dependencies and inputs",
    mode === "agent"
      ? stageOrder.length > 0
        ? `Prepare and confirm execution chain: ${stageOrder.map((s) => `${stageText(s)} · ${moduleForStage(s)}`).join(" → ")}`
        : planningTail
      : "Prepare execution chain",
  ];

  let activeIndex = 0;
  if (elapsedSec >= 2) activeIndex = 1;
  if (elapsedSec >= 5) activeIndex = 2;
  if (elapsedSec >= 7) activeIndex = 3;
  return { items, activeIndex };
}

function formatSavedAt(iso: string): string {
  const raw = (iso || "").trim();
  if (!raw) return "";
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return raw;
  return d.toLocaleString(undefined, { timeZone: "America/New_York" });
}

function getChatWidthBounds(): { min: number; max: number } {
  if (typeof window === "undefined") {
    return { min: CHAT_WIDTH_MIN_PX, max: 960 };
  }
  const inner = window.innerWidth;
  // Never force max above what fits: workspace + rail + chat ≤ inner (avoids broken flex / dead splitter).
  const maxPhys = Math.floor(inner - CHAT_RESIZE_WORKSPACE_MIN_PX - CHAT_SPLIT_RAIL_PX);
  const min = CHAT_WIDTH_MIN_PX;
  const max = Math.max(min, maxPhys);
  return { min, max };
}

function readStoredChatWidth(): number {
  if (typeof window === "undefined") return CHAT_WIDTH_DEFAULT;
  const raw = localStorage.getItem(LS_CHAT_WIDTH);
  const n = raw ? Number.parseInt(raw, 10) : NaN;
  const { min, max } = getChatWidthBounds();
  if (!Number.isFinite(n)) return Math.min(max, Math.max(min, CHAT_WIDTH_DEFAULT));
  return Math.min(max, Math.max(min, n));
}

function readStoredChatCollapsed(): boolean {
  if (typeof window === "undefined") return false;
  return localStorage.getItem(LS_CHAT_COLLAPSED) === "1";
}

function readStoredChatHistoryCollapsed(): boolean {
  if (typeof window === "undefined") return false;
  return localStorage.getItem(LS_CHAT_HISTORY_COLLAPSED) === "1";
}

function readStoredChatMode(): ChatMode {
  if (typeof window === "undefined") return "ask";
  const raw = (localStorage.getItem(LS_CHAT_MODE) || "ask").toLowerCase();
  if (raw === "plan" || raw === "agent") return raw;
  return "ask";
}

type SettingsBackendTab = "codex" | "openai";
type SettingsMainTab = "llm" | "skills";

type LlmEffective = {
  base_url: string;
  model: string;
  transport: string;
  api_key_configured: boolean;
  chatgpt_account_id_configured: boolean;
};

function inferBackendTab(saved: { llm_provider: string; openai_base_url: string }): SettingsBackendTab {
  const p = saved.llm_provider.trim().toLowerCase().replaceAll("-", "_");
  if (p === "openai_api") return "openai";
  if (p === "chatgpt_codex") return "codex";
  const u = saved.openai_base_url.toLowerCase();
  if (u.includes("chatgpt.com") && u.includes("codex")) return "codex";
  return "openai";
}

function agentReady(st: {
  transport: string;
  api_key_configured: boolean;
  chatgpt_account_id_configured: boolean;
}): boolean {
  if (!st.api_key_configured) return false;
  if (st.transport === "codex_responses") return st.chatgpt_account_id_configured;
  return true;
}

function firstTokenProfileId(rows: CodexProfileRow[]): string {
  const w = rows.filter((p) => p.has_access_token);
  return w.length ? w[0].id : "";
}

export function App() {
  const [chatThreads, setChatThreads] = useState<ChatThreadSummary[]>([]);
  const [chatSearch, setChatSearch] = useState("");
  const [chatHistoryCollapsed, setChatHistoryCollapsed] = useState(readStoredChatHistoryCollapsed);
  const [activeChatId, setActiveChatId] = useState(() =>
    typeof window === "undefined" ? "" : localStorage.getItem(LS_ACTIVE_CHAT_ID) ?? "",
  );
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [editingUserIdx, setEditingUserIdx] = useState<number | null>(null);
  const [editingUserText, setEditingUserText] = useState("");
  const [chatRenderPhase, setChatRenderPhase] = useState<ChatRenderPhase>("idle");
  const [waitingLabel, setWaitingLabel] = useState("Preparing response");
  const [waitingCtx, setWaitingCtx] = useState<PreparingContext>({
    label: "Preparing response",
    stage: null,
    provider: "",
    llmSourceLabel: "",
    planSites: [],
    planStages: [],
    userPrompt: "",
    pollExecutionProgress: false,
    executionLiveLine: "",
  });
  const [waitingElapsedSec, setWaitingElapsedSec] = useState(0);
  const [pending, setPending] = useState<PendingApproval[]>([]);
  const [pipelineLabel, setPipelineLabel] = useState("idle");
  const [pipelineExecutionMeta, setPipelineExecutionMeta] = useState<PipelineExecutionMeta | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsMainTab, setSettingsMainTab] = useState<SettingsMainTab>("llm");
  const [llmBase, setLlmBase] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [llmNewKey, setLlmNewKey] = useState("");
  const [llmEnvOverrides, setLlmEnvOverrides] = useState<string[]>([]);
  const [codexProfileId, setCodexProfileId] = useState("");
  const [codexProfileRows, setCodexProfileRows] = useState<CodexProfileRow[]>([]);
  const [llmEffective, setLlmEffective] = useState<LlmEffective | null>(null);
  const [codexFeedback, setCodexFeedback] = useState<string | null>(null);
  const [codexOpenLink, setCodexOpenLink] = useState<string | null>(null);
  const [codexDeviceCode, setCodexDeviceCode] = useState<string | null>(null);
  const [codexBusy, setCodexBusy] = useState(false);
  const [codexAccountListLoading, setCodexAccountListLoading] = useState(false);
  const [settingsBackendTab, setSettingsBackendTab] = useState<SettingsBackendTab>("codex");
  const [agentStrip, setAgentStrip] = useState<{
    model: string;
    transport: string;
    ready: boolean;
  } | null>(null);
  const [chatCollapsed, setChatCollapsed] = useState(readStoredChatCollapsed);
  const [chatWidth, setChatWidth] = useState(readStoredChatWidth);
  const [chatMode, setChatMode] = useState<ChatMode>(() => readStoredChatMode());
  /** Latest mode for ``loadChatStateFor`` (avoids stale closure and prevents init effect churn on mode sync). */
  const chatModeRef = useRef<ChatMode>(chatMode);
  useEffect(() => {
    chatModeRef.current = chatMode;
  }, [chatMode]);
  const [draftPipelinePlan, setDraftPipelinePlan] = useState<Record<string, unknown> | null>(null);
  const [draftPipelinePlanJsonOpen, setDraftPipelinePlanJsonOpen] = useState(false);
  const [agentPendingPlan, setAgentPendingPlan] = useState<Record<string, unknown> | null>(null);
  const [savedPlans, setSavedPlans] = useState<PipelinePlanRecord[]>([]);
  const [savedPlanPick, setSavedPlanPick] = useState("");
  const [studioToast, setStudioToast] = useState<{
    kind: "ok" | "err";
    title: string;
    detail?: string;
  } | null>(null);
  const workspaceRef = useRef<HTMLElement | null>(null);
  const [workspaceRect, setWorkspaceRect] = useState<{ left: number; width: number }>({ left: 0, width: 0 });
  const resizeDragRef = useRef<{ startX: number; startW: number } | null>(null);
  const resizeWindowCleanupRef = useRef<(() => void) | null>(null);
  const chatWidthDuringDragRef = useRef(readStoredChatWidth());
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const activeRequestAbortRef = useRef<AbortController | null>(null);
  const stopRequestedRef = useRef(false);
  /** True while waiting on a chat-issued `execute_plan` / saved plan run (poll `/api/pipeline`). */
  const executionPollDesiredRef = useRef(false);
  /** Wall-clock start of the current ``waiting`` phase (ms); keeps stepper elapsed correct after refresh. */
  const waitingPhaseEpochMsRef = useRef(0);

  const saveInflightState = useCallback((row: InflightChatState | null) => {
    if (typeof window === "undefined") return;
    if (!row) {
      localStorage.removeItem(LS_CHAT_INFLIGHT);
      return;
    }
    localStorage.setItem(LS_CHAT_INFLIGHT, JSON.stringify(row));
  }, []);

  const loadInflightState = useCallback((): InflightChatState | null => {
    if (typeof window === "undefined") return null;
    const raw = localStorage.getItem(LS_CHAT_INFLIGHT);
    if (!raw) return null;
    try {
      const row = JSON.parse(raw) as Partial<InflightChatState>;
      if (!row || typeof row !== "object") return null;
      if (typeof row.sessionId !== "string" || typeof row.userText !== "string") return null;
      const mode = row.chatMode === "ask" || row.chatMode === "plan" || row.chatMode === "agent" ? row.chatMode : "ask";
      const waitingCtx = row.waitingCtx && typeof row.waitingCtx === "object"
        ? {
            label: typeof row.waitingCtx.label === "string" ? row.waitingCtx.label : "Preparing response",
            stage: normalizePipelineStage(row.waitingCtx.stage),
            provider: typeof row.waitingCtx.provider === "string" ? row.waitingCtx.provider : "",
            llmSourceLabel: typeof (row.waitingCtx as { llmSourceLabel?: unknown }).llmSourceLabel === "string"
              ? (row.waitingCtx as { llmSourceLabel: string }).llmSourceLabel
              : "",
            planSites: Array.isArray(row.waitingCtx.planSites)
              ? row.waitingCtx.planSites.filter((x): x is string => typeof x === "string")
              : [],
            planStages: Array.isArray(row.waitingCtx.planStages)
              ? row.waitingCtx.planStages
                  .map((x) => normalizePipelineStage(x))
                  .filter((x): x is PipelineStage => x !== null)
              : [],
            userPrompt:
              typeof (row.waitingCtx as { userPrompt?: unknown }).userPrompt === "string"
                ? (row.waitingCtx as { userPrompt: string }).userPrompt
                : row.userText,
            pollExecutionProgress: Boolean((row.waitingCtx as { pollExecutionProgress?: unknown }).pollExecutionProgress),
            executionLiveLine:
              typeof (row.waitingCtx as { executionLiveLine?: unknown }).executionLiveLine === "string"
                ? (row.waitingCtx as { executionLiveLine: string }).executionLiveLine
                : "",
          }
        : {
            label: "Preparing response",
            stage: null,
            provider: "",
            llmSourceLabel: "",
            planSites: [],
            planStages: [],
            userPrompt: "",
            pollExecutionProgress: false,
            executionLiveLine: "",
          };
      const startedAt = typeof row.startedAt === "number" ? row.startedAt : Date.now();
      return { sessionId: row.sessionId, userText: row.userText, chatMode: mode, waitingCtx, startedAt };
    } catch {
      return null;
    }
  }, []);

  const clearInflightState = useCallback(() => {
    saveInflightState(null);
  }, [saveInflightState]);

  const resizeComposerTextarea = useCallback(() => {
    const ta = composerTextareaRef.current;
    if (!ta) return;
    const maxH = Math.min(440, Math.floor(window.innerHeight * 0.48));
    ta.style.height = "auto";
    const sh = ta.scrollHeight;
    const next = Math.min(Math.max(sh, 44), maxH);
    ta.style.height = `${next}px`;
    ta.style.overflowY = sh > maxH ? "auto" : "hidden";
  }, []);

  useEffect(() => {
    if (!activeChatId || typeof window === "undefined") return;
    localStorage.setItem(LS_ACTIVE_CHAT_ID, activeChatId);
  }, [activeChatId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(LS_CHAT_HISTORY_COLLAPSED, chatHistoryCollapsed ? "1" : "0");
  }, [chatHistoryCollapsed]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(LS_CHAT_MODE, chatMode);
  }, [chatMode]);

  const refreshAgentStrip = useCallback(async () => {
    try {
      const st = await getLlmStatus();
      setAgentStrip({
        model: st.model,
        transport: st.transport,
        ready: agentReady(st),
      });
    } catch {
      setAgentStrip(null);
    }
  }, []);

  useEffect(() => {
    void refreshAgentStrip();
  }, [refreshAgentStrip]);

  const refreshChatThreads = useCallback(async (query = "") => {
    const out = await listChats(query);
    setChatThreads(out.threads ?? []);
    return out.threads ?? [];
  }, []);

  const refreshSavedPlans = useCallback(async () => {
    try {
      const out = await listPipelinePlans();
      setSavedPlans(out.plans ?? []);
    } catch {
      setSavedPlans([]);
    }
  }, []);

  useEffect(() => {
    void refreshSavedPlans();
  }, [refreshSavedPlans]);

  useEffect(() => {
    if (draftPipelinePlan) setDraftPipelinePlanJsonOpen(false);
  }, [draftPipelinePlan]);

  useEffect(() => {
    const onResize = () => {
      const { min, max } = getChatWidthBounds();
      setChatWidth((w) => {
        const cur = Number(w);
        const base = Number.isFinite(cur) ? cur : CHAT_WIDTH_DEFAULT;
        return Math.min(max, Math.max(min, base));
      });
      queueMicrotask(() => resizeComposerTextarea());
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [resizeComposerTextarea]);

  useEffect(() => {
    return () => {
      resizeWindowCleanupRef.current?.();
    };
  }, []);

  useLayoutEffect(() => {
    resizeComposerTextarea();
  }, [input, activeChatId, chatCollapsed, resizeComposerTextarea]);

  const loadChatStateFor = useCallback(async (chatId: string) => {
    const st = await getChatState(chatId);
    let nextMessages = st.messages ?? [];

    const serverTurn = st.chat_turn_in_progress === true;
    const pendingUser = (st.chat_turn_pending_user || "").trim();
    const pendingModeRaw = (st.chat_turn_pending_mode || "").trim().toLowerCase();
    const pendingMode: ChatMode =
      pendingModeRaw === "plan" || pendingModeRaw === "agent" ? pendingModeRaw : "ask";
    const planForWait =
      (st.chat_turn_pending_pipeline_plan as unknown) ?? (st.last_pipeline_plan as unknown) ?? undefined;

    /** Mode used to restore plan/draft bars (server in-flight turn wins over stale localStorage mode). */
    let modeForRestores: ChatMode = chatModeRef.current;

    if (serverTurn && pendingUser) {
      clearInflightState();
      const last = nextMessages.length > 0 ? nextMessages[nextMessages.length - 1] : undefined;
      const alreadyPendingTail =
        last &&
        last.role === "user" &&
        last.content === pendingUser &&
        (last.chat_mode || "ask") === pendingMode;
      if (!alreadyPendingTail) {
        nextMessages = [...nextMessages, { role: "user", content: pendingUser, chat_mode: pendingMode }];
      }
      const ctx = buildPreparingResponseLabel({ userText: pendingUser, pipelinePlan: planForWait });
      const pl = st.pipeline;
      const line =
        pl && typeof pl.step_label === "string"
          ? formatExecutionProgressTitle(pl.step_label, typeof pl.last_site_stem === "string" ? pl.last_site_stem : "")
          : "";
      waitingPhaseEpochMsRef.current = waitingEpochMsFromChatResume(st);
      setWaitingCtx({ ...ctx, executionLiveLine: line });
      setWaitingLabel(ctx.label);
      setWaitingElapsedSec(Math.max(0, Math.floor((Date.now() - waitingPhaseEpochMsRef.current) / 1000)));
      setChatRenderPhase("waiting");
      executionPollDesiredRef.current = ctx.pollExecutionProgress;
      setBusy(true);
      if (pendingMode === "agent" || pendingMode === "plan") {
        modeForRestores = pendingMode;
        if (pendingMode !== chatModeRef.current) {
          setChatMode(pendingMode);
        }
      }
    } else {
      const inflight = loadInflightState();
      if (inflight && inflight.sessionId === chatId) {
        const seen = nextMessages.some((m) => m.role === "user" && m.content === inflight.userText);
        if (!seen) {
          nextMessages = [...nextMessages, { role: "user", content: inflight.userText, chat_mode: inflight.chatMode }];
          waitingPhaseEpochMsRef.current = inflight.startedAt;
          setWaitingCtx(inflight.waitingCtx);
          setWaitingLabel(inflight.waitingCtx.label);
          setWaitingElapsedSec(Math.max(0, Math.floor((Date.now() - inflight.startedAt) / 1000)));
          setChatRenderPhase("waiting");
          executionPollDesiredRef.current = inflight.waitingCtx.pollExecutionProgress;
          setBusy(true);
          modeForRestores = inflight.chatMode;
          if (inflight.chatMode !== chatModeRef.current) {
            setChatMode(inflight.chatMode);
          }
        } else {
          clearInflightState();
          setBusy(false);
        }
      } else {
        setBusy(false);
      }
    }
    setMessages(nextMessages);
    setPending(st.pending_approvals ?? []);
    setPipelineLabel(st.pipeline?.step_label ?? "idle");
    setPipelineExecutionMeta(null);
    const serverAgentAwaiting = st.chat_agent_plan_awaiting_approval === true;
    if (serverAgentAwaiting && modeForRestores !== "agent") {
      setChatMode("agent");
    }
    const modeForAgentBar: ChatMode = serverAgentAwaiting ? "agent" : modeForRestores;
    const modeForDraft: ChatMode = serverAgentAwaiting ? "agent" : modeForRestores;
    setAgentPendingPlan(
      serverTurn && pendingUser
        ? null
        : shouldRestoreAgentPendingPlan(chatId, modeForAgentBar, nextMessages, st.last_pipeline_plan, {
            serverAgentPlanAwaiting: serverAgentAwaiting,
          }),
    );
    setDraftPipelinePlan(
      serverTurn && pendingUser
        ? null
        : shouldRestoreDraftPipelinePlan(chatId, modeForDraft, nextMessages, st.last_pipeline_plan),
    );
  }, [clearInflightState, loadInflightState]);

  const loadChatStateForRef = useRef(loadChatStateFor);
  loadChatStateForRef.current = loadChatStateFor;

  const startNewChat = useCallback(async () => {
    const out = await createChat();
    setActiveChatId(out.id);
    await loadChatStateFor(out.id);
    await refreshChatThreads("");
  }, [loadChatStateFor, refreshChatThreads]);

  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      try {
        const threads = await refreshChatThreads("");
        if (cancelled) return;
        const stored =
          typeof window !== "undefined" ? (localStorage.getItem(LS_ACTIVE_CHAT_ID) || "").trim() : "";
        const candidate =
          stored && threads.some((t) => t.id === stored) ? stored : threads[0]?.id;
        if (candidate) {
          setActiveChatId(candidate);
          await loadChatStateForRef.current(candidate);
          return;
        }
        await startNewChat();
      } catch {
        // Keep empty defaults on initial load failures.
      }
    };
    void init();
    return () => {
      cancelled = true;
    };
    // Mount-only bootstrap: read active id from localStorage so we do not re-run when ``loadChatStateFor``/mode sync updates.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentional
  }, [refreshChatThreads, startNewChat]);

  useEffect(() => {
    if (!studioToast) return;
    const t = window.setTimeout(() => setStudioToast(null), 8000);
    return () => window.clearTimeout(t);
  }, [studioToast]);

  useLayoutEffect(() => {
    const update = () => {
      const el = workspaceRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      setWorkspaceRect({ left: r.left, width: r.width });
    };
    update();
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined" && workspaceRef.current) {
      ro = new ResizeObserver(() => update());
      ro.observe(workspaceRef.current);
    }
    window.addEventListener("resize", update);
    return () => {
      if (ro) ro.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [chatCollapsed, chatWidth]);


  useEffect(() => {
    void refreshChatThreads(chatSearch);
  }, [chatSearch, refreshChatThreads]);

  const loadCodexProfiles = useCallback(async (authJsonPath = "") => {
    try {
      const out = await getCodexProfiles(authJsonPath);
      setCodexProfileRows(out.profiles);
      return out;
    } catch (e) {
      console.error(e);
      setCodexProfileRows([]);
      return null;
    }
  }, []);

  const openSettings = useCallback(async () => {
    setSettingsMainTab("llm");
    setSettingsOpen(true);
    setCodexFeedback(null);
    try {
      const llm = await getLlmSettings();
      setSettingsBackendTab(inferBackendTab(llm.saved));
      setLlmBase(llm.saved.openai_base_url || "https://api.openai.com/v1");
      setLlmModel(llm.saved.openai_model);
      setCodexProfileId(llm.saved.codex_auth_profile_id);
      setLlmNewKey(llm.saved.openai_api_key || "");
      setLlmEnvOverrides(llm.env_overrides);
      setLlmEffective(llm.effective);
      setCodexAccountListLoading(true);
      try {
        const profs = await getCodexProfiles(llm.saved.codex_auth_json_path ?? "");
        setCodexProfileRows(profs.profiles);
      } catch {
        setCodexProfileRows([]);
      } finally {
        setCodexAccountListLoading(false);
      }
      void refreshAgentStrip();
    } catch (e) {
      console.error(e);
    }
  }, [refreshAgentStrip]);

  const runCodexBrowserLogin = useCallback(async () => {
    setCodexBusy(true);
    setCodexFeedback(null);
    setCodexOpenLink(null);
    setCodexDeviceCode(null);
    // Blank tab on click (user gesture). If pop-ups are blocked, tab is null — we still wait for the
    // server and show a clickable link (works in embedded browsers that swallow follow-up fetches).
    const tab = window.open("about:blank", "_blank");
    if (tab) {
      try {
        tab.opener = null;
      } catch {
        /* ignore */
      }
    }
    setCodexFeedback("Waiting for the Codex CLI (up to ~90s). Do not close this page…");
    try {
      const r = await postCodexLoginBrowser();
      if (typeof r.generation !== "number" || r.generation < 1) {
        throw new Error("Server did not return a login session id. Run `npm run build` in frontend/, restart ./mito2, hard-refresh this page.");
      }
      if (r.auth_url) {
        setCodexOpenLink(r.auth_url);
        const code = (r.device_code || "").trim().toUpperCase();
        setCodexDeviceCode(code || null);
        if (tab) {
          try {
            tab.location.href = r.auth_url;
          } catch {
            /* embedded browser may block navigation */
          }
        }
        setCodexFeedback(
          tab
            ? "Complete sign-in in the new tab (or use Open sign-in page below), then Refresh accounts and Save."
            : "Pop-ups blocked — use Open sign-in page below, then Refresh accounts and Save."
        );
      } else {
        if (tab) {
          try {
            tab.close();
          } catch {
            /* ignore */
          }
        }
        setCodexFeedback(r.message ?? "No sign-in URL from Codex.");
      }
    } catch (e) {
      if (tab) {
        try {
          tab.close();
        } catch {
          /* ignore */
        }
      }
      setCodexFeedback(e instanceof Error ? e.message : String(e));
    } finally {
      setCodexBusy(false);
    }
  }, []);

  const refreshCodexAccounts = useCallback(async () => {
    setCodexBusy(true);
    setCodexAccountListLoading(true);
    setCodexFeedback(null);
    try {
      const llm = await getLlmSettings();
      await loadCodexProfiles(llm.saved.codex_auth_json_path ?? "");
      await refreshAgentStrip();
      setCodexFeedback("Reloaded accounts from Codex auth data.");
    } catch (e) {
      setCodexFeedback(e instanceof Error ? e.message : String(e));
    } finally {
      setCodexBusy(false);
      setCodexAccountListLoading(false);
    }
  }, [loadCodexProfiles, refreshAgentStrip]);

  const runCodexLogout = useCallback(async () => {
    if (
      !window.confirm(
        "Sign out of Codex on this machine? You will need to sign in again to use ChatGPT Codex from this app.",
      )
    ) {
      return;
    }
    setCodexBusy(true);
    setCodexAccountListLoading(true);
    setCodexFeedback(null);
    setCodexOpenLink(null);
    try {
      const llm = await getLlmSettings();
      const path = llm.saved.codex_auth_json_path ?? "";
      const r = await postCodexLogout(path || undefined);
      setCodexFeedback(r.message || "Signed out of Codex.");
      setCodexProfileId("");
      await loadCodexProfiles(path);
      await refreshAgentStrip();
    } catch (e) {
      setCodexFeedback(e instanceof Error ? e.message : String(e));
    } finally {
      setCodexBusy(false);
      setCodexAccountListLoading(false);
    }
  }, [loadCodexProfiles, refreshAgentStrip]);

  const savePipelineDraftToLibrary = useCallback(async () => {
    const planToSave = draftPipelinePlan;
    const chatId = activeChatId;
    if (!planToSave || !chatId) return;
    // Match Dismiss UX: hide immediately, then persist.
    persistDraftPlanDismissed(chatId, planToSave);
    setDraftPipelinePlan(null);
    try {
      const title =
        typeof planToSave.rationale === "string"
          ? String(planToSave.rationale).slice(0, 120)
          : "";
      await savePipelinePlan(planToSave, title);
      await refreshSavedPlans();
    } catch (e) {
      // Restore draft so user can retry if save fails.
      clearDraftPlanDismissed(chatId);
      setDraftPipelinePlan(planToSave);
      setStudioToast({
        kind: "err",
        title: "Could not save plan",
        detail: e instanceof Error ? e.message : String(e),
      });
    }
  }, [activeChatId, draftPipelinePlan, refreshSavedPlans]);

  const deleteSavedPlanSelection = useCallback(async () => {
    const id = savedPlanPick.trim();
    if (!id) return;
    if (!window.confirm("Delete this saved plan?")) return;
    try {
      await deletePipelinePlan(id);
      setSavedPlanPick("");
      await refreshSavedPlans();
    } catch (e) {
      setStudioToast({
        kind: "err",
        title: "Delete failed",
        detail: e instanceof Error ? e.message : String(e),
      });
    }
  }, [refreshSavedPlans, savedPlanPick]);

  const streamAssistantReply = useCallback(async (fullReply: string, assistantMode?: ChatMode) => {
    const chunks = splitReplyIntoChunks(fullReply);
    const targetFrames = Math.min(90, Math.max(24, Math.ceil(chunks.length / 3)));
    const batchSize = Math.max(1, Math.ceil(chunks.length / targetFrames));

    setChatRenderPhase("rendering");
    setMessages((m) => [
      ...m,
      assistantMode
        ? { role: "assistant", content: "", chat_mode: assistantMode }
        : { role: "assistant", content: "" },
    ]);

    for (let i = 0; i < chunks.length; i += batchSize) {
      if (stopRequestedRef.current) break;
      const nextChunk = chunks.slice(i, i + batchSize).join("");
      setMessages((m) => {
        if (m.length === 0) return m;
        const lastIdx = m.length - 1;
        if (m[lastIdx]?.role !== "assistant") return m;
        const next = m.slice();
        const prev = next[lastIdx];
        next[lastIdx] = {
          ...prev,
          content: prev.content + nextChunk,
          ...(assistantMode ? { chat_mode: assistantMode } : {}),
        };
        return next;
      });
      await sleep(22);
    }
  }, []);

  const startCancelableRequest = useCallback((): AbortController => {
    activeRequestAbortRef.current?.abort();
    const ctl = new AbortController();
    activeRequestAbortRef.current = ctl;
    stopRequestedRef.current = false;
    return ctl;
  }, []);

  const stopOngoingResponse = useCallback(async () => {
    if (!activeChatId) return;
    stopRequestedRef.current = true;
    executionPollDesiredRef.current = false;
    activeRequestAbortRef.current?.abort();
    activeRequestAbortRef.current = null;
    try {
      await postChatStop(activeChatId);
    } catch {
      /* best effort: local abort still applies */
    }
    clearInflightState();
    setChatRenderPhase("idle");
    setBusy(false);
    setMessages((m) => [...m, { role: "assistant", content: "Stopped current response.", chat_mode: chatMode }]);
  }, [activeChatId, chatMode, clearInflightState]);

  const beginWaiting = useCallback((userText: string, pipelinePlan?: unknown) => {
    waitingPhaseEpochMsRef.current = Date.now();
    const ctx = buildPreparingResponseLabel({ userText, pipelinePlan });
    executionPollDesiredRef.current = ctx.pollExecutionProgress;
    setWaitingCtx(ctx);
    setWaitingLabel(ctx.label);
    setWaitingElapsedSec(0);
    setChatRenderPhase("waiting");
    if (activeChatId) {
      saveInflightState({
        sessionId: activeChatId,
        userText,
        chatMode,
        waitingCtx: ctx,
        startedAt: waitingPhaseEpochMsRef.current,
      });
    }
  }, [activeChatId, chatMode, saveInflightState]);

  const applyPlanningHint = useCallback((hint: PlanningHint) => {
    setWaitingCtx((prev) => mergeWaitingCtxFromPlanningHint(prev, hint));
  }, []);

  useEffect(() => {
    if (chatRenderPhase !== "waiting") {
      setWaitingElapsedSec(0);
      return;
    }
    const tick = () => {
      if (!waitingPhaseEpochMsRef.current) {
        waitingPhaseEpochMsRef.current = Date.now();
      }
      setWaitingElapsedSec(Math.max(0, Math.floor((Date.now() - waitingPhaseEpochMsRef.current) / 1000)));
    };
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, [chatRenderPhase]);

  useEffect(() => {
    if (chatRenderPhase !== "waiting" || !busy || !activeChatId) return;
    if (!executionPollDesiredRef.current) return;
    let cancelled = false;
    const tick = async () => {
      if (!executionPollDesiredRef.current || cancelled) return;
      try {
        const p = await getPipelineProgress(activeChatId);
        if (cancelled || !executionPollDesiredRef.current) return;
        const line = formatExecutionProgressTitle(p.step_label, p.last_site_stem);
        setWaitingCtx((prev) => ({ ...prev, executionLiveLine: line }));
        setPipelineLabel(line);
      } catch {
        /* ignore */
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 1300);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [chatRenderPhase, busy, activeChatId]);

  /** After a browser refresh, the POST /api/chat stream is gone but the server may still be running the turn. */
  useEffect(() => {
    if (!busy || chatRenderPhase !== "waiting" || !activeChatId) return;
    if (activeRequestAbortRef.current) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const st = await getChatState(activeChatId);
        if (cancelled || activeRequestAbortRef.current) return;
        if (st.chat_turn_in_progress) return;
        setMessages(st.messages ?? []);
        setPending(st.pending_approvals ?? []);
        setPipelineLabel(st.pipeline?.step_label ?? "idle");
        clearInflightState();
        setChatRenderPhase("idle");
        setBusy(false);
        void refreshChatThreads(chatSearch);
        void refreshAgentStrip();
      } catch {
        /* ignore */
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 1600);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [busy, chatRenderPhase, activeChatId, chatSearch, clearInflightState, refreshAgentStrip, refreshChatThreads]);

  const approveAgentPendingPlan = useCallback(async () => {
    if (!activeChatId || !agentPendingPlan || busy) return;
    clearAgentPlanDismissed(activeChatId);
    const userLine = EXECUTE_APPROVED_PIPELINE_USER_LINE;
    setBusy(true);
    const ctl = startCancelableRequest();
    beginWaiting(userLine, agentPendingPlan);
    const prior = messages;
    setMessages((m) => [...m, { role: "user", content: userLine, chat_mode: "agent" }]);
    setAgentPendingPlan(null);
    try {
      const out = await postChat(activeChatId, userLine, prior, {
        chat_mode: "agent",
        pipeline_action: "execute_plan",
        pipeline_plan: agentPendingPlan,
        signal: ctl.signal,
        onPlanning: applyPlanningHint,
      });
      executionPollDesiredRef.current = false;
      if (out.planning_hint) applyPlanningHint(out.planning_hint);
      setPending(out.pending_approvals);
      const execMeta = normalizePipelineExecution(out.execution);
      if (execMeta && execMeta.steps.length > 0) {
        setPipelineLabel(execMeta.ok === false ? "Pipeline failed" : "Pipeline run");
      } else {
        setPipelineLabel(out.pipeline.step_label);
      }
      setPipelineExecutionMeta(normalizePipelineExecution(out.execution));
      await streamAssistantReply(out.reply, "agent");
      await refreshChatThreads(chatSearch);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setPipelineExecutionMeta(null);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Pipeline run error: ${e instanceof Error ? e.message : String(e)}`,
          chat_mode: "agent",
        },
      ]);
    } finally {
      executionPollDesiredRef.current = false;
      activeRequestAbortRef.current = null;
      clearInflightState();
      setChatRenderPhase("idle");
      setBusy(false);
      void refreshAgentStrip();
    }
  }, [
    activeChatId,
    agentPendingPlan,
    applyPlanningHint,
    beginWaiting,
    busy,
    chatSearch,
    clearInflightState,
    messages,
    refreshAgentStrip,
    refreshChatThreads,
    startCancelableRequest,
    streamAssistantReply,
  ]);

  const runSavedPlanSelection = useCallback(async () => {
    const id = savedPlanPick.trim();
    if (!id || !activeChatId || busy) return;
    const row = savedPlans.find((p) => p.id === id);
    if (!row?.plan) return;
    const userLine = `Run plan: ${row.title || id}`;
    setAgentPendingPlan(null);
    setDraftPipelinePlan(row.plan as Record<string, unknown>);
    setBusy(true);
    const ctl = startCancelableRequest();
    beginWaiting(userLine, row.plan as Record<string, unknown>);
    const prior = messages;
    setMessages((m) => [...m, { role: "user", content: userLine, chat_mode: "agent" }]);
    try {
      const out = await postChat(activeChatId, userLine, prior, {
        chat_mode: "agent",
        pipeline_action: "execute_plan",
        pipeline_plan: row.plan as Record<string, unknown>,
        signal: ctl.signal,
        onPlanning: applyPlanningHint,
      });
      executionPollDesiredRef.current = false;
      if (out.planning_hint) applyPlanningHint(out.planning_hint);
      setPending(out.pending_approvals);
      const execMetaSaved = normalizePipelineExecution(out.execution);
      if (execMetaSaved && execMetaSaved.steps.length > 0) {
        setPipelineLabel(execMetaSaved.ok === false ? "Pipeline failed" : "Pipeline run");
      } else {
        setPipelineLabel(out.pipeline.step_label);
      }
      setPipelineExecutionMeta(normalizePipelineExecution(out.execution));
      setDraftPipelinePlan(null);
      await streamAssistantReply(out.reply, "agent");
      await refreshChatThreads(chatSearch);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setPipelineExecutionMeta(null);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Run saved plan error: ${e instanceof Error ? e.message : String(e)}`, chat_mode: "agent" },
      ]);
    } finally {
      executionPollDesiredRef.current = false;
      activeRequestAbortRef.current = null;
      clearInflightState();
      setChatRenderPhase("idle");
      setBusy(false);
      void refreshAgentStrip();
    }
  }, [
    activeChatId,
    applyPlanningHint,
    beginWaiting,
    busy,
    chatSearch,
    clearInflightState,
    messages,
    refreshAgentStrip,
    refreshChatThreads,
    startCancelableRequest,
    savedPlanPick,
    savedPlans,
    streamAssistantReply,
  ]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || busy || !activeChatId) return;
    setBusy(true);
    const ctl = startCancelableRequest();
    beginWaiting(text);
    setInput("");
    const prior = messages;
    setMessages((m) => [...m, { role: "user", content: text, chat_mode: chatMode }]);
    try {
      const out = await postChat(activeChatId, text, prior, {
        chat_mode: chatMode,
        signal: ctl.signal,
        onPlanning: applyPlanningHint,
      });
      if (out.planning_hint) applyPlanningHint(out.planning_hint);
      setPending(out.pending_approvals);
      setPipelineLabel(out.pipeline.step_label);
      setPipelineExecutionMeta(normalizePipelineExecution(out.execution));
      if (out.draft_pipeline_plan) {
        if (chatMode === "agent") {
          clearAgentPlanDismissed(activeChatId);
          setDraftPipelinePlan(null);
          setAgentPendingPlan(out.draft_pipeline_plan);
        } else {
          clearDraftPlanDismissed(activeChatId);
          setDraftPipelinePlan(out.draft_pipeline_plan);
        }
      } else {
        setDraftPipelinePlan(null);
        setAgentPendingPlan(null);
      }
      await streamAssistantReply(out.reply, chatMode);
      await refreshChatThreads(chatSearch);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setPipelineExecutionMeta(null);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Error: ${e instanceof Error ? e.message : String(e)}`, chat_mode: chatMode },
      ]);
    } finally {
      activeRequestAbortRef.current = null;
      clearInflightState();
      setChatRenderPhase("idle");
      setBusy(false);
      void refreshAgentStrip();
    }
  }, [activeChatId, applyPlanningHint, beginWaiting, busy, chatMode, chatSearch, clearInflightState, input, messages, refreshAgentStrip, refreshChatThreads, startCancelableRequest, streamAssistantReply]);

  const saveEditedUserMessage = useCallback(async () => {
    if (!activeChatId || editingUserIdx == null || busy) return;
    const nextText = editingUserText.trim();
    if (!nextText) return;
    const editIdx = editingUserIdx;
    setBusy(true);
    const ctl = startCancelableRequest();
    beginWaiting(nextText);
    // Immediate UX: close editor and branch conversation at edited turn right away.
    setEditingUserIdx(null);
    setEditingUserText("");
    setMessages((prev) => [...prev.slice(0, editIdx), { role: "user", content: nextText, chat_mode: chatMode }]);
    try {
      const out = await postChatEdit(activeChatId, editIdx, nextText, chatMode, ctl.signal);
      setPending(out.pending_approvals ?? []);
      setPipelineLabel(out.pipeline?.step_label ?? "idle");
      setPipelineExecutionMeta(normalizePipelineExecution(out.execution));
      if (out.draft_pipeline_plan) {
        if (chatMode === "agent") {
          clearAgentPlanDismissed(activeChatId);
          setDraftPipelinePlan(null);
          setAgentPendingPlan(out.draft_pipeline_plan);
        } else {
          clearDraftPlanDismissed(activeChatId);
          setDraftPipelinePlan(out.draft_pipeline_plan);
        }
      } else {
        setDraftPipelinePlan(null);
        setAgentPendingPlan(null);
      }
      await streamAssistantReply(out.reply, chatMode);
      await refreshChatThreads(chatSearch);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      await loadChatStateFor(activeChatId);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Edit/regenerate error: ${e instanceof Error ? e.message : String(e)}`, chat_mode: chatMode },
      ]);
    } finally {
      activeRequestAbortRef.current = null;
      clearInflightState();
      setChatRenderPhase("idle");
      setBusy(false);
      void refreshAgentStrip();
    }
  }, [
    activeChatId,
    beginWaiting,
    busy,
    chatMode,
    chatSearch,
    clearInflightState,
    editingUserIdx,
    editingUserText,
    loadChatStateFor,
    refreshAgentStrip,
    refreshChatThreads,
    startCancelableRequest,
    streamAssistantReply,
  ]);

  const openChat = useCallback(
    async (chatId: string) => {
      if (!chatId || busy) return;
      setActiveChatId(chatId);
      await loadChatStateFor(chatId);
    },
    [busy, loadChatStateFor],
  );

  const deleteChat = useCallback(
    async (chatId: string) => {
      if (!chatId || busy) return;
      if (!window.confirm("Delete this chat? This cannot be undone.")) return;
      const out = await deleteChats([chatId]);
      if (!out.deleted.includes(chatId)) return;
      const threads = await refreshChatThreads(chatSearch);
      if (activeChatId === chatId) {
        const next = threads[0]?.id;
        if (next) {
          setActiveChatId(next);
          await loadChatStateFor(next);
        } else {
          await startNewChat();
        }
      }
    },
    [activeChatId, busy, chatSearch, loadChatStateFor, refreshChatThreads, startNewChat],
  );

  const onApprove = useCallback(
    async (id: string, approved: boolean) => {
      try {
        if (!activeChatId) return;
        const out = await resolveApproval(activeChatId, id, approved);
        setPending((prev) => prev.filter((p) => p.id !== id));
        const note =
          out.status === "approved" && out.result
            ? `\n\n[exit ${out.result.returncode}]\n${out.result.stderr || out.result.stdout || ""}`.slice(
                0,
                4000,
              )
            : "";
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            content: `${approved ? "Approved and executed." : "Declined."}${note}`,
            chat_mode: chatMode,
          },
        ]);
        await refreshChatThreads(chatSearch);
      } catch (e) {
        setMessages((m) => [
          ...m,
          { role: "assistant", content: `Approval error: ${e instanceof Error ? e.message : String(e)}`, chat_mode: chatMode },
        ]);
      }
    },
    [activeChatId, chatMode, chatSearch, refreshChatThreads],
  );

  const saveSettings = useCallback(async () => {
    let llmPatch: Parameters<typeof saveLlmSettings>[0];
    if (settingsBackendTab === "codex") {
      let pid = codexProfileId.trim();
      if (!pid) pid = firstTokenProfileId(codexProfileRows);
      llmPatch = {
        llm_provider: "chatgpt_codex",
        openai_base_url: CODEX_BASE,
        openai_model: CODEX_FIXED_MODEL,
        codex_auth_profile_id: pid,
      };
    } else {
      llmPatch = {
        llm_provider: "openai_api",
        openai_base_url: llmBase.trim() || "https://api.openai.com/v1",
        openai_model: llmModel.trim(),
      };
      llmPatch.openai_api_key = llmNewKey.trim();
    }
    const llmOut = await saveLlmSettings(llmPatch);
    setLlmNewKey(llmOut.saved.openai_api_key || "");
    setLlmEnvOverrides(llmOut.env_overrides);
    setLlmEffective(llmOut.effective);
    setCodexProfileId(llmOut.saved.codex_auth_profile_id);
    void refreshAgentStrip();
    setSettingsOpen(false);
  }, [
    codexProfileId,
    codexProfileRows,
    llmBase,
    llmModel,
    llmNewKey,
    settingsBackendTab,
    refreshAgentStrip,
  ]);

  const transportLabel =
    agentStrip?.transport === "codex_responses"
      ? "OpenAI Codex"
      : agentStrip?.transport
        ? "api"
        : "";

  const setChatPanelCollapsed = useCallback((collapsed: boolean) => {
    setChatCollapsed(collapsed);
    if (typeof window !== "undefined") {
      localStorage.setItem(LS_CHAT_COLLAPSED, collapsed ? "1" : "0");
    }
  }, []);

  const onChatResizePointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (chatCollapsed) return;
      e.preventDefault();
      e.stopPropagation();
      resizeWindowCleanupRef.current?.();

      const sw = Number(chatWidth);
      const startW = Number.isFinite(sw) ? sw : CHAT_WIDTH_DEFAULT;
      resizeDragRef.current = { startX: e.clientX, startW };
      chatWidthDuringDragRef.current = startW;

      const move = (ev: PointerEvent) => {
        const d = resizeDragRef.current;
        if (!d) return;
        const { min, max } = getChatWidthBounds();
        const raw = d.startW + (d.startX - ev.clientX);
        let next = Math.min(max, Math.max(min, raw));
        if (!Number.isFinite(next)) next = d.startW;
        chatWidthDuringDragRef.current = next;
        setChatWidth(next);
      };
      const up = () => {
        window.removeEventListener("pointermove", move, true);
        window.removeEventListener("pointerup", up, true);
        window.removeEventListener("pointercancel", up, true);
        resizeWindowCleanupRef.current = null;
        resizeDragRef.current = null;
        const w = chatWidthDuringDragRef.current;
        if (typeof window !== "undefined" && Number.isFinite(w)) {
          localStorage.setItem(LS_CHAT_WIDTH, String(w));
        }
      };
      resizeWindowCleanupRef.current = () => {
        window.removeEventListener("pointermove", move, true);
        window.removeEventListener("pointerup", up, true);
        window.removeEventListener("pointercancel", up, true);
        resizeWindowCleanupRef.current = null;
      };
      window.addEventListener("pointermove", move, true);
      window.addEventListener("pointerup", up, true);
      window.addEventListener("pointercancel", up, true);
    },
    [chatCollapsed, chatWidth],
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages, chatRenderPhase]);

  return (
    <div className={`layout layout-with-chat-rail${chatCollapsed ? " layout-chat-collapsed" : ""}`}>
      <main className="workspace" ref={workspaceRef}>
        {studioToast && (
          <div
            className="studio-toast-host"
            style={{
              display: "contents",
            }}
          >
            <div
              className={`studio-toast studio-toast-${studioToast.kind}`}
              role="status"
              aria-live="polite"
              style={{
                position: "fixed",
                top: "12px",
                left:
                  workspaceRect.width > 240
                    ? `${workspaceRect.left + workspaceRect.width / 2}px`
                    : "50vw",
                transform: "translateX(-50%)",
                width:
                  workspaceRect.width > 240
                    ? `${Math.max(280, Math.min(760, workspaceRect.width - 24))}px`
                    : "min(760px, calc(100vw - 24px))",
                zIndex: 1200,
                pointerEvents: "auto",
              }}
            >
              <div className="studio-toast-inner">
                <strong className="studio-toast-title">{studioToast.title}</strong>
                {studioToast.detail ? <p className="studio-toast-body">{studioToast.detail}</p> : null}
              </div>
              <button
                type="button"
                className="studio-toast-close"
                aria-label="Dismiss"
                onClick={() => setStudioToast(null)}
              >
                ×
              </button>
            </div>
          </div>
        )}
        <PipelineStudio
          sessionId={activeChatId || "chat_bootstrap"}
          chatPanelCollapsed={chatCollapsed}
          onNotify={(title, detail, kind) => setStudioToast({ title, detail, kind: kind ?? "ok" })}
        />
      </main>
      {chatCollapsed ? (
        <div className="chat-split-rail" aria-hidden />
      ) : (
        <div
          className="chat-split-rail chat-split-rail--draggable"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize chat panel"
          onPointerDown={onChatResizePointerDown}
        />
      )}
      <aside
        className={`chat-column${chatCollapsed ? " chat-column-collapsed" : ""}`}
        style={{ width: chatCollapsed ? CHAT_COLLAPSED_PX : chatWidth }}
      >
        {chatCollapsed ? (
          <div className="chat-collapsed-bar">
            <button
              type="button"
              className="chat-panel-arrow"
              aria-label="Expand chat panel"
              title="Expand chat"
              onClick={() => setChatPanelCollapsed(false)}
            >
              ◀
            </button>
            <button
              type="button"
              className="chat-panel-arrow chat-collapsed-settings"
              aria-label="Settings"
              title="Settings"
              onClick={() => void openSettings()}
            >
              ⚙
            </button>
            <span className="chat-collapsed-vertical-label" aria-hidden>
              Agent
            </span>
          </div>
        ) : (
          <>
            <div className="chat-header-stack">
              <div className="chat-header">
                <div className="chat-header-leading">
                  <button
                    type="button"
                    className="chat-panel-arrow"
                    aria-label="Collapse chat panel"
                    title="Collapse chat"
                    onClick={() => setChatPanelCollapsed(true)}
                  >
                    ▶
                  </button>
                  <div className="chat-header-titles">
                    <div className="agent-title-row">
                      <h2>{chatThreads.find((t) => t.id === activeChatId)?.title || "Agent"}</h2>
                    </div>
                    {agentStrip && (
                      <div className={`agent-strip ${agentStrip.ready ? "agent-strip-ok" : "agent-strip-warn"}`}>
                        <span className="agent-strip-dot" aria-hidden />
                        <span className="agent-strip-text">
                          {agentStrip.ready ? (
                            <>
                              {transportLabel} · <code>{agentStrip.model}</code>
                            </>
                          ) : (
                            <>Not connected — open Settings</>
                          )}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
                <div className="chat-header-actions">
                  <span
                    className="pipeline-pill"
                    title={
                      pipelineExecutionMeta
                        ? [pipelineExecutionMeta.sites.join(", "), pipelineExecutionMeta.stages.join(", ")]
                            .filter(Boolean)
                            .join(" — ")
                        : undefined
                    }
                  >
                    {pipelineLabel}
                    {busy && chatRenderPhase === "waiting" ? " · …" : ""}
                  </span>
                  <button type="button" className="ghost" onClick={openSettings}>
                    Settings
                  </button>
                </div>
              </div>
              {pipelineExecutionMeta ? (
                <div className="chat-pipeline-exec-strip" aria-live="polite">
                  {pipelineExecutionMeta.sites.length > 0 || pipelineExecutionMeta.stages.length > 0 ? (
                    <div className="chat-pipeline-exec-plan">
                      {pipelineExecutionMeta.sites.length > 0 ? (
                        <span className="chat-pipeline-exec-sites">{pipelineExecutionMeta.sites.join(", ")}</span>
                      ) : null}
                      {pipelineExecutionMeta.sites.length > 0 && pipelineExecutionMeta.stages.length > 0 ? (
                        <span className="chat-pipeline-exec-sep"> · </span>
                      ) : null}
                      {pipelineExecutionMeta.stages.length > 0 ? (
                        <span className="chat-pipeline-exec-stages">{pipelineExecutionMeta.stages.join(" → ")}</span>
                      ) : null}
                      {typeof pipelineExecutionMeta.n_crops === "number" &&
                      pipelineExecutionMeta.stages.includes("download") ? (
                        <span className="chat-pipeline-exec-ncrops">{` · n_crops=${pipelineExecutionMeta.n_crops}`}</span>
                      ) : null}
                      {pipelineExecutionMeta.stages.includes("download") &&
                      (typeof pipelineExecutionMeta.n_crops_training === "number" ||
                        typeof pipelineExecutionMeta.n_crops_inference === "number") ? (
                        <span className="chat-pipeline-exec-ncrops">
                          {` · training=${pipelineExecutionMeta.n_crops_training ?? 0}, inference=${pipelineExecutionMeta.n_crops_inference ?? 0}`}
                        </span>
                      ) : null}
                    </div>
                  ) : null}
                  {pipelineExecutionMeta.steps.length > 0 ? (
                    <div className="chat-pipeline-exec-steps">
                      {pipelineExecutionMeta.steps.map((s, i) => (
                        <span
                          key={`${s.site}:${s.stage}:${i}`}
                          className={`chat-pipeline-exec-chip ${s.ok ? "chat-pipeline-exec-chip--ok" : "chat-pipeline-exec-chip--fail"}`}
                        >
                          {s.site}:{s.stage}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
            <div className={`chat-thread-nav${chatHistoryCollapsed ? " chat-thread-nav-collapsed" : ""}`}>
              <div className="chat-thread-search-row">
                <button type="button" className="ghost chat-thread-new-btn" disabled={busy} onClick={() => void startNewChat()}>
                  New chat
                </button>
                <input
                  className="chat-thread-search"
                  value={chatSearch}
                  onChange={(e) => setChatSearch(e.target.value)}
                  placeholder="Search chats"
                  disabled={chatHistoryCollapsed}
                />
                <button
                  type="button"
                  className="chat-thread-collapse-btn"
                  onClick={() => setChatHistoryCollapsed((v) => !v)}
                  title={chatHistoryCollapsed ? "Expand chat history" : "Collapse chat history"}
                >
                  {chatHistoryCollapsed ? `Open (${chatThreads.length})` : "Collapse"}
                </button>
              </div>
              <div className={`chat-thread-list${chatHistoryCollapsed ? " chat-thread-list-hidden" : ""}`}>
                {chatThreads.map((t) => (
                  <div key={t.id} className={`chat-thread-item${t.id === activeChatId ? " chat-thread-item-active" : ""}`}>
                    <button type="button" className="chat-thread-open" onClick={() => void openChat(t.id)}>
                      <span className="chat-thread-title">{t.title}</span>
                      <span className="chat-thread-preview">{t.preview || "No messages yet"}</span>
                    </button>
                    <button type="button" className="chat-thread-delete" onClick={() => void deleteChat(t.id)} title="Delete chat">
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>
            <div className="messages">
              {messages.map((b, i) => (
                <div
                  key={i}
                  className={`bubble ${b.role}${
                    chatRenderPhase === "rendering" && i === messages.length - 1 && b.role === "assistant"
                      ? " bubble-streaming"
                      : ""
                  }`}
                >
                  {b.role === "user" ? (
                    editingUserIdx === i ? (
                      <div className="chat-edit-wrap">
                        <textarea
                          className="chat-edit-textarea"
                          value={editingUserText}
                          onChange={(e) => setEditingUserText(e.target.value)}
                          disabled={busy}
                        />
                        <div className="chat-edit-actions">
                          <button type="button" className="primary" disabled={busy} onClick={() => void saveEditedUserMessage()}>
                            Save & regenerate
                          </button>
                          <button
                            type="button"
                            className="ghost"
                            onClick={() => {
                              setEditingUserIdx(null);
                              setEditingUserText("");
                            }}
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="chat-user-message-wrap">
                        {b.chat_mode ? (
                          <span className="chat-msg-mode-pill" title="Mode used for this message">
                            {b.chat_mode}
                          </span>
                        ) : null}
                        <button
                          type="button"
                          className="chat-user-edit-btn"
                          title="Edit and regenerate from this turn"
                          disabled={busy || pending.length > 0}
                          onClick={() => {
                            setEditingUserIdx(i);
                            setEditingUserText(b.content);
                          }}
                        >
                          Edit
                        </button>
                        <button
                          type="button"
                          className="chat-user-delete-btn"
                          title="Stop ongoing response"
                          disabled={!activeChatId || (chatRenderPhase === "idle" && !busy)}
                          onClick={() => void stopOngoingResponse()}
                        >
                          Stop
                        </button>
                        <MarkdownPreview markdown={b.content} className="markdown-in-bubble" hideWhenEmpty />
                      </div>
                    )
                  ) : (
                    <>
                      {b.chat_mode ? (
                        <span className="chat-msg-mode-pill" title="Mode used for this reply">
                          {b.chat_mode}
                        </span>
                      ) : null}
                      <MarkdownPreview markdown={b.content} className="markdown-in-bubble" hideWhenEmpty />
                    </>
                  )}
                  {chatRenderPhase === "rendering" && i === messages.length - 1 && b.role === "assistant" && (
                    <span className="chat-stream-cursor" aria-hidden />
                  )}
                </div>
              ))}
              {chatRenderPhase === "waiting" && (
                <div className="bubble assistant bubble-waiting" role="status" aria-live="polite">
                  <span className="chat-waiting-title">{resolveWaitingBubbleTitle(waitingCtx, waitingLabel)}</span>
                  <span className="chat-waiting-phase">
                    {waitingPhaseHint(chatMode, waitingCtx, waitingElapsedSec)}
                  </span>
                  {(() => {
                    const trace = waitingProgressTrace(chatMode, waitingCtx, waitingElapsedSec);
                    return (
                      <ul className="chat-waiting-trace">
                        {trace.items.map((step, idx) => {
                          const active = idx === trace.activeIndex;
                          const done = idx < trace.activeIndex;
                          return (
                            <li
                              key={`${step}:${idx}`}
                              className={`chat-waiting-trace-item${done ? " is-done" : ""}${active ? " is-active" : ""}`}
                            >
                              {done ? "✓" : active ? "→" : "·"} {step}
                            </li>
                          );
                        })}
                      </ul>
                    );
                  })()}
                  <span className="chat-waiting-dots" aria-hidden>
                    <span />
                    <span />
                    <span />
                  </span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            {chatMode === "agent" && agentPendingPlan ? (
              <div className="agent-plan-approval-bar" role="region" aria-label="Pipeline plan approval">
                <span className="muted-note">Execute this routed plan now?</span>
                <div className="agent-plan-approval-actions">
                  <button type="button" className="primary" disabled={busy} onClick={() => void approveAgentPendingPlan()}>
                    Approve &amp; run
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    disabled={busy}
                    onClick={() => {
                      if (activeChatId && agentPendingPlan) persistAgentPlanDismissed(activeChatId, agentPendingPlan);
                      setAgentPendingPlan(null);
                    }}
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            ) : null}
            {pending.length > 0 && (
              <div className="approvals">
                <h3>Pending approval</h3>
                {pending.map((p) => (
                  <div key={p.id} className="card">
                    <div className="card-title">{p.title}</div>
                    <div className="cmd">
                      $ {p.command.join(" ")}
                      <br />
                      <span style={{ color: "var(--muted)" }}>cwd: {p.cwd}</span>
                    </div>
                    <div className="row">
                      <button type="button" className="primary" onClick={() => onApprove(p.id, true)}>
                        Approve
                      </button>
                      <button type="button" className="danger" onClick={() => onApprove(p.id, false)}>
                        Decline
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {draftPipelinePlan && chatMode === "plan" ? (
              <div className="pipeline-draft-card" role="region" aria-label="Draft pipeline plan">
                <div className="pipeline-draft-card-head pipeline-draft-card-head--row">
                  <button
                    type="button"
                    className="pipeline-draft-toggle"
                    aria-expanded={draftPipelinePlanJsonOpen}
                    onClick={() => setDraftPipelinePlanJsonOpen((o) => !o)}
                  >
                    <span className="pipeline-draft-chevron" aria-hidden>
                      {draftPipelinePlanJsonOpen ? "▼" : "▶"}
                    </span>
                    <strong>Draft pipeline plan</strong>
                  </button>
                  <span className="muted-note pipeline-draft-card-note">Stages 1–4 (scrape / database / download / training)</span>
                </div>
                {draftPipelinePlanJsonOpen ? (
                  <pre className="pipeline-draft-pre">{JSON.stringify(draftPipelinePlan, null, 2)}</pre>
                ) : (
                  <p className="muted-note pipeline-draft-collapsed-hint">JSON hidden — use the arrow to expand.</p>
                )}
                <div className="pipeline-draft-actions">
                  <button type="button" className="primary" disabled={busy} onClick={() => void savePipelineDraftToLibrary()}>
                    Save plan
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    disabled={busy}
                    onClick={() => {
                      if (activeChatId && draftPipelinePlan) persistDraftPlanDismissed(activeChatId, draftPipelinePlan);
                      setDraftPipelinePlan(null);
                    }}
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            ) : null}
            {chatMode === "plan" || chatMode === "agent" ? (
              <div className="chat-saved-plans-bar">
                <div className="chat-saved-plans-left">
                  <label className="sr-only" htmlFor="saved-plan-select">
                    Saved pipeline plans
                  </label>
                  <select
                    id="saved-plan-select"
                    className="chat-saved-plans-select"
                    value={savedPlanPick}
                    onChange={(e) => setSavedPlanPick(e.target.value)}
                    disabled={busy}
                    aria-label="Saved pipeline plans"
                  >
                    <option value="">Saved plans…</option>
                    {savedPlans.map((p) => (
                      <option key={p.id} value={p.id}>
                        {(p.title || p.id) + (p.created_at ? ` · ${formatSavedAt(p.created_at)}` : "")}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    className="ghost chat-saved-plans-run"
                    disabled={busy || !savedPlanPick}
                    onClick={() => void runSavedPlanSelection()}
                  >
                    Run plan
                  </button>
                  {chatMode === "agent" && savedPlanPick ? (
                    <span className="muted-note">Use Saved plans to run the latest routed plan.</span>
                  ) : null}
                </div>
                <button
                  type="button"
                  className="danger chat-saved-plans-delete"
                  disabled={busy || !savedPlanPick}
                  onClick={() => void deleteSavedPlanSelection()}
                >
                  Delete
                </button>
              </div>
            ) : null}
            <div className="composer">
              <select
                className="chat-mode-select"
                value={chatMode}
                onChange={(e) => {
                  const v = e.target.value as ChatMode;
                  setChatMode(v);
                  if (v === "ask") setDraftPipelinePlan(null);
                }}
                disabled={busy}
                title="Chat mode"
                aria-label="Chat mode"
              >
                <option value="ask">Ask</option>
                <option value="plan">Plan</option>
                <option value="agent">Agent</option>
              </select>
              <textarea
                ref={composerTextareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Message…"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void send();
                  }
                }}
              />
              <button type="button" className="primary" disabled={busy} onClick={() => void send()}>
                Send
              </button>
            </div>
          </>
        )}
      </aside>
      {settingsOpen && (
        <div className="settings" role="dialog" aria-modal="true">
          <div className="settings-panel">
            <h2>Settings</h2>
            <div className="settings-tabs settings-tabs-main" role="tablist" aria-label="Settings section">
              <button
                type="button"
                role="tab"
                className={`settings-tab ${settingsMainTab === "llm" ? "active" : ""}`}
                aria-selected={settingsMainTab === "llm"}
                onClick={() => setSettingsMainTab("llm")}
              >
                LLM backend
              </button>
              <button
                type="button"
                role="tab"
                className={`settings-tab ${settingsMainTab === "skills" ? "active" : ""}`}
                aria-selected={settingsMainTab === "skills"}
                onClick={() => setSettingsMainTab("skills")}
              >
                Agent skills
              </button>
            </div>

            {settingsMainTab === "skills" ? (
              <div className="settings-section" role="tabpanel">
                <AgentSkillsSettings />
                <div className="settings-actions">
                  <button type="button" className="ghost" onClick={() => setSettingsOpen(false)}>
                    Close
                  </button>
                </div>
              </div>
            ) : (
              <>
            {llmEffective && (
              <p className="settings-live-line">
                Chat uses <code>{llmEffective.model}</code> ·{" "}
                {llmEffective.transport === "codex_responses" ? "OpenAI Codex" : "api"}
                {llmEffective.api_key_configured ? "" : " · no credentials"}
              </p>
            )}
            <div className="settings-tabs" role="tablist" aria-label="Backend">
              <button
                type="button"
                role="tab"
                className={`settings-tab ${settingsBackendTab === "codex" ? "active" : ""}`}
                aria-selected={settingsBackendTab === "codex"}
                onClick={() => setSettingsBackendTab("codex")}
              >
                OpenAI Codex
              </button>
              <button
                type="button"
                role="tab"
                className={`settings-tab ${settingsBackendTab === "openai" ? "active" : ""}`}
                aria-selected={settingsBackendTab === "openai"}
                onClick={() => setSettingsBackendTab("openai")}
              >
                api
              </button>
            </div>

            {settingsBackendTab === "codex" && (
              <div className="settings-section settings-section-tight" role="tabpanel">
                <div className="codex-actions-row codex-actions-row-top">
                  <button
                    type="button"
                    className="primary"
                    disabled={codexBusy}
                    onClick={() => void runCodexBrowserLogin()}
                  >
                    Sign in with Codex (open browser)
                  </button>
                  <button
                    type="button"
                    className="danger"
                    disabled={codexBusy}
                    onClick={() => void runCodexLogout()}
                  >
                    Sign out of Codex
                  </button>
                </div>
                <div className="codex-actions-row codex-actions-row-below-signin">
                  <button type="button" disabled={codexBusy} onClick={() => void refreshCodexAccounts()}>
                    Refresh accounts
                  </button>
                </div>
                <p className="muted-note codex-signout-hint">
                  Sign out runs <code>codex logout</code> on the <strong>same computer as this app</strong> (default{" "}
                  <code>~/.codex/</code> unless you use a custom auth path).
                </p>
                <p className="muted-note">
                  Runs the Codex CLI on the <strong>same computer as this app</strong>. A new tab opens from this page
                  when the sign-in URL is ready (or use the link below). Then <strong>Refresh accounts</strong>, pick an
                  account, and <strong>Save</strong>. Chat uses model <code>{CODEX_FIXED_MODEL}</code>.
                </p>
                {codexFeedback && (
                  <p className="codex-setup-feedback" role="status">
                    {codexFeedback}
                  </p>
                )}
                {codexOpenLink && (
                  <p className="codex-open-link-wrap">
                    <a className="codex-open-link" href={codexOpenLink} target="_blank" rel="noopener noreferrer">
                      Open sign-in page
                    </a>
                    <span className="muted-note codex-open-link-url">{codexOpenLink}</span>
                  </p>
                )}
                {codexDeviceCode && (
                  <p className="codex-setup-feedback" role="status">
                    Device code: <code>{codexDeviceCode}</code>
                  </p>
                )}
                <label htmlFor="codex-profile">Account</label>
                <div className="settings-field-row-updating">
                  <select
                    id="codex-profile"
                    className="field-input"
                    value={codexAccountListLoading ? STUDIO_SELECT_LOADING : codexProfileId}
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v === STUDIO_SELECT_LOADING) return;
                      setCodexProfileId(v);
                    }}
                    disabled={codexBusy || codexAccountListLoading}
                    aria-busy={codexAccountListLoading}
                  >
                    {codexAccountListLoading ? (
                      <option value={STUDIO_SELECT_LOADING}>Updating ...</option>
                    ) : (
                      <>
                        <option value="">— Use refresh after sign-in —</option>
                        {codexProfileRows.map((r) => (
                          <option key={r.id} value={r.id} title={r.detail ? `Also: ${r.detail}` : r.label}>
                            {r.label} ({r.id})
                            {r.has_access_token ? "" : " · no token"}
                          </option>
                        ))}
                      </>
                    )}
                  </select>
                  <StudioUpdatingBadge active={codexAccountListLoading} label="Updating ..." />
                </div>
                <p className="muted-note">
                  Model for ChatGPT Codex: <code>{CODEX_FIXED_MODEL}</code> (fixed).
                </p>
              </div>
            )}

            {settingsBackendTab === "openai" && (
              <div className="settings-section settings-section-tight" role="tabpanel">
                <p className="muted-note">
                  OpenAI-compatible HTTP API (<code>/v1/chat/completions</code> or your provider’s equivalent). Point base
                  URL at any vendor that speaks the same protocol.
                </p>
                <label htmlFor="llm-base">Base URL</label>
                <input
                  id="llm-base"
                  className="field-input"
                  value={llmBase}
                  onChange={(e) => setLlmBase(e.target.value)}
                  placeholder="https://api.openai.com/v1"
                  autoComplete="off"
                />
                <label htmlFor="llm-model-openai">Model</label>
                <input
                  id="llm-model-openai"
                  className="field-input"
                  value={llmModel}
                  onChange={(e) => setLlmModel(e.target.value)}
                  placeholder="provider-specific model id"
                  autoComplete="off"
                />
                <label htmlFor="llm-key">API key</label>
                <input
                  id="llm-key"
                  className="field-input"
                  type="text"
                  value={llmNewKey}
                  onChange={(e) => setLlmNewKey(e.target.value)}
                  placeholder="key…"
                  autoComplete="off"
                />
              </div>
            )}

            {llmEnvOverrides.length > 0 && (
              <p className="muted-note">Env overrides file: {llmEnvOverrides.join(", ")}</p>
            )}

            <div className="settings-actions">
              <button type="button" className="ghost" onClick={() => setSettingsOpen(false)}>
                Cancel
              </button>
              <button type="button" className="primary" onClick={() => void saveSettings()}>
                Save
              </button>
            </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
