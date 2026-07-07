import type {
  AdminBackpressure,
  AdminHealth,
  AdminOptions,
  AdminStatus,
  CapturePageRequest,
  CapturePageResponse,
  EvalContext,
  EvalRecordsPage,
  LeaderboardResponse,
  MetaResponse,
  ScoreHistoryDetail,
  ScoreHistoryOptions,
  ScoreHistoryResponse,
} from "../../dtos/api";
import { getJson, getJsonAuth, postJson, postJsonAuth } from "../http";

export const api = {
  meta: () => getJson<MetaResponse>("/api/meta"),
  refresh: () => postJson<{ entry_count: number; errors: string[] }>("/api/refresh"),
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
  scoreHistoryOptions: () => getJson<ScoreHistoryOptions>("/api/score-history/options"),
  scoreHistory: (model: string, benchmark: string) =>
    getJson<ScoreHistoryResponse>(
      `/api/score-history?model=${encodeURIComponent(model)}&benchmark=${encodeURIComponent(benchmark)}`
    ),
  scoreHistoryDetail: (taskId: number) =>
    getJson<ScoreHistoryDetail>(`/api/score-history/detail?task_id=${taskId}`),
  capturePage: (payload: CapturePageRequest) => postJson<CapturePageResponse>("/api/capture-page", payload),
  evalContext: (taskId: number, sampleIndex: number, repeatIndex: number, passIndex: number) => {
    const params = new URLSearchParams({
      task_id: String(taskId),
      sample_index: String(sampleIndex),
      repeat_index: String(repeatIndex),
      pass_index: String(passIndex),
    });
    return getJson<EvalContext>(`/api/eval-context?${params.toString()}`);
  },

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
