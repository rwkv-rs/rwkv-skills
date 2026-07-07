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

export interface ParamColumn {
  param: string;
  param_label: string;
  latest_model: string;
  latest_label: string;
  prev_model: string | null;
  prev_label: string | null;
}

export interface DomainTable {
  key: string;
  title: string;
  label: string;
  param_columns: ParamColumn[];
  rows: LeaderboardRow[];
}

export interface NaiveBoard {
  key: string;
  title: string;
  label: string;
  is_delta: boolean;
  param_columns: ParamColumn[];
  rows: LeaderboardRow[];
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
  naive_board?: NaiveBoard;
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

export interface AdminOptions {
  jobs: { name: string; domain: string }[];
  domains: string[];
  model_select: string[];
  worker_profile: string[];
  protocol: string[];
  run_mode: string[];
}

export interface AdminStatus {
  status: string;
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

export interface ScoreHistoryPoint {
  score_id: number;
  task_id: number;
  cot_mode: string;
  evaluator: string | null;
  board: "normal" | "naive";
  percent: number | null;
  metric: string | null;
  created_at: string | null;
  sampling_summary: string;
  model: string | null;
  benchmark: string | null;
}

export interface ScoreHistoryGroup {
  cot_mode: string;
  points: ScoreHistoryPoint[];
}

export interface ScoreHistoryResponse {
  model: string;
  benchmark: string;
  total: number;
  groups: ScoreHistoryGroup[];
}

export interface ScoreHistoryOptions {
  models: string[];
  benchmarks: string[];
  pairs: { model: string; dataset: string }[];
}

export interface StopTokenRow {
  id: number;
  token: string;
}

export interface StageSampling {
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  max_tokens: number | null;
  stop_tokens: StopTokenRow[];
  penalties: {
    presence_penalty: number | null;
    repetition_penalty: number | null;
    penalty_decay: number | null;
  };
}

export interface ScoreHistoryDetail {
  found: boolean;
  task_id: number;
  model: string | null;
  benchmark: string | null;
  cot_mode: string | null;
  evaluator: string | null;
  board: string;
  metric: string | null;
  percent: number | null;
  metrics: Record<string, unknown>;
  sampling: {
    stages: Record<string, StageSampling>;
    effective_sample_count: number | null;
    avg_k: unknown;
    pass_ks: unknown;
    n_shot: unknown;
    sample_limit: unknown;
    prompt_profile: string | null;
  };
  stages: { prompt: string; completion: string; stop_reason: unknown }[];
}

export interface CapturePageRequest {
  url: string;
  width?: number;
  height?: number;
}

export interface CapturePageResponse {
  path: string;
  url: string;
  width: number;
  height: number;
  page_height: number | null;
  full_page: boolean;
}
