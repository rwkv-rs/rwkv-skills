// Typed client for the FastAPI dashboard API. All calls are same-origin in
// production; in dev Vite proxies /api -> FastAPI (see vite.config.ts).

export interface TableViewOption {
  key: string;
  label: string;
}

export interface DomainGroup {
  key: string;
  label: string;
  title: string;
}

export interface MetaResponse {
  auto_label: string;
  default_view: string;
  table_views: TableViewOption[];
  domain_groups: DomainGroup[];
  models: string[];
  model_choices: string[];
  entry_count: number;
  errors: string[];
}

export interface CellMeta {
  cell_id: string;
  task_id: number | null;
  benchmark_name: string;
  eval_method: string;
  k_metric: string;
  column_label: string;
  model: string | null;
  tooltip: string | null;
  clickable: boolean;
}

export interface DetailCell {
  percent: number | null;
  meta: CellMeta | null;
}

export interface DeltaCell {
  prev: number | null;
  latest: number | null;
  delta: number | null;
  prev_meta: CellMeta | null;
  latest_meta: CellMeta | null;
}

export interface LeaderboardRow {
  benchmark_name: string;
  num_samples: number | null;
  eval_method: string;
  k_metric: string;
  cells: (DetailCell | DeltaCell)[];
}

export interface DomainTable {
  key: string;
  title: string;
  label: string;
  rows: LeaderboardRow[];
}

export interface ParamColumn {
  param: string;
  param_label: string;
  latest_model: string;
  latest_label: string;
  prev_model: string | null;
  prev_label: string | null;
}

export interface OverviewRow {
  domain_key: string;
  domain_title: string;
  cells: ({ percent: number | null } | DeltaCell)[];
}

export interface SelectionInfo {
  dropdown_value: string;
  selected_label: string;
  auto_selected: boolean;
  model_sequence: string[];
  skipped_small_params: number;
  auto_label: string;
}

export interface ChartSeriesPoint {
  k: number;
  acc: number;
}
export interface ChartPayload {
  knowledge: { subjects: string[]; models: string[]; data: { model: string; subject: string; score: number }[] } | null;
  math: { ks: number[]; series: { name: string; points: ChartSeriesPoint[] }[] } | null;
  instruction_following: { domains: string[]; models: string[]; data: { domain: string; model: string; score: number }[] } | null;
  coding: { datasets: string[]; models: string[]; data: { dataset: string; model: string; score: number; metric: string }[] } | null;
}

export interface LeaderboardResponse {
  view: string;
  view_label: string;
  is_delta: boolean;
  is_field_avg: boolean;
  param_columns: ParamColumn[];
  interaction_meta: Record<string, CellMeta>;
  domains: DomainTable[];
  overview?: OverviewRow[];
  selection: SelectionInfo;
  charts: ChartPayload;
  errors: string[];
}

export interface EvalRecord {
  sample_index: number;
  repeat_index: number;
  pass_index: number;
  is_passed: boolean;
  answer: string;
  ref_answer: string;
  fail_reason: string;
  context_preview?: string;
}

export interface EvalRecordsPage {
  task_id: number;
  records: EvalRecord[];
  offset: number;
  limit: number;
  next_offset: number;
  has_more: boolean;
}

export interface StopToken {
  id: number;
  token: string;
}

export interface EvalContext {
  view: "text" | "structured";
  raw_text: string;
  context: Record<string, unknown> | null;
  stop_tokens: Record<string, StopToken[]>;
  errors: string[];
}

// ---- Admin (scheduler control) types ----------------------------------

export interface AdminOptions {
  jobs: { name: string; domain: string }[];
  domains: string[];
  model_select: string[];
  worker_profile: string[];
  protocol: string[];
  run_mode: string[];
}

/** Mirrors build_status_response() in src/eval/scheduler/admin.py. */
export interface AdminStatus {
  status: string; // idle | starting | running | paused | cancelled | completed | failed
  desired_state: string | null;
  run_id: string | null;
  error: string | null;
  started_at_unix_ms: number | null;
  updated_at_unix_ms: number | null;
  finished_at_unix_ms: number | null;
  pending_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  tasks_total: number;
  progress_percent: number;
  queue_head: string[];
  active_jobs: string[];
  available_gpus: string[];
  request: Record<string, unknown> | null;
}

export interface AdminHealth {
  status: string;
  active: boolean;
  auth_required: boolean;
}

export interface BackpressureModel {
  model: string;
  model_slug: string;
  status: string;
  route_count: number;
  ok_route_count: number;
  pending_queue: number;
  max_batch_size: number | null;
  failed_batches: number;
  last_total_tok_s: number | null;
  error: string | null;
}

export interface AdminBackpressure {
  infer_base_url: string;
  available_gpus: string[];
  models: BackpressureModel[];
  error: string | null;
}

// Admin bearer token (optional; only needed when RWKV_ADMIN_API_KEY is set
// server-side). Persisted in localStorage so it survives reloads.
const ADMIN_TOKEN_KEY = "rwkv_admin_token";
export function getAdminToken(): string {
  try {
    return localStorage.getItem(ADMIN_TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}
export function setAdminToken(token: string): void {
  try {
    if (token) localStorage.setItem(ADMIN_TOKEN_KEY, token);
    else localStorage.removeItem(ADMIN_TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

function adminHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { Accept: "application/json", ...extra };
  const token = getAdminToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

async function getJsonAuth<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: adminHeaders() });
  if (!res.ok) throw new Error(`${res.status}: ${await errText(res)}`);
  return res.json() as Promise<T>;
}

async function postJsonAuth<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: adminHeaders({ "Content-Type": "application/json" }),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status}: ${await errText(res)}`);
  return res.json() as Promise<T>;
}

async function errText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return typeof data?.detail === "string" ? data.detail : JSON.stringify(data);
  } catch {
    return res.statusText;
  }
}

export const api = {
  meta: () => getJson<MetaResponse>("/api/meta"),
  refresh: () =>
    fetch("/api/refresh", { method: "POST" }).then((r) => r.json() as Promise<{ entry_count: number; errors: string[] }>),
  leaderboard: (model: string | null, view: string) => {
    const params = new URLSearchParams({ view });
    if (model) params.set("model", model);
    return getJson<LeaderboardResponse>(`/api/leaderboard?${params.toString()}`);
  },
  evalRecords: (taskId: number, onlyWrong: boolean, limit: number, offset: number) => {
    const params = new URLSearchParams({
      task_id: String(taskId),
      only_wrong: String(onlyWrong),
      limit: String(limit),
      offset: String(offset),
    });
    return getJson<EvalRecordsPage>(`/api/eval-records?${params.toString()}`);
  },
  evalContext: (taskId: number, sampleIndex: number, repeatIndex: number, passIndex: number) => {
    const params = new URLSearchParams({
      task_id: String(taskId),
      sample_index: String(sampleIndex),
      repeat_index: String(repeatIndex),
      pass_index: String(passIndex),
    });
    return getJson<EvalContext>(`/api/eval-context?${params.toString()}`);
  },

  // ---- Admin (scheduler control) ----
  adminHealth: () => getJsonAuth<AdminHealth>("/api/admin/health"),
  adminOptions: () => getJsonAuth<AdminOptions>("/api/admin/eval/options"),
  adminDraft: () => getJsonAuth<Record<string, unknown>>("/api/admin/eval/draft"),
  adminStatus: () => getJsonAuth<AdminStatus>("/api/admin/eval/status"),
  adminStart: (payload: Record<string, unknown>) =>
    postJsonAuth<AdminStatus>("/api/admin/eval/start", payload),
  adminPause: () => postJsonAuth<AdminStatus>("/api/admin/eval/pause"),
  adminResume: () => postJsonAuth<AdminStatus>("/api/admin/eval/resume"),
  adminCancel: () => postJsonAuth<AdminStatus>("/api/admin/eval/cancel"),
  adminBackpressure: (inferBaseUrl?: string) => {
    const params = new URLSearchParams();
    if (inferBaseUrl) params.set("infer_base_url", inferBaseUrl);
    const qs = params.toString();
    return getJsonAuth<AdminBackpressure>(`/api/admin/backpressure${qs ? `?${qs}` : ""}`);
  },
};
