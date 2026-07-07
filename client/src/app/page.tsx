import { AdminPage } from "../components/Admin";
import { DashboardPage } from "../components/DashboardPage";
import { ScoreHistory } from "../components/ScoreHistory";
import { api } from "../lib/api";
import type { LeaderboardResponse, MetaResponse } from "../dtos/api";

export const dynamic = "force-dynamic";

type SearchParams = Promise<Record<string, string | string[] | undefined>>;
const BASE_PATH = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH);

function value(params: Record<string, string | string[] | undefined>, key: string, fallback: string): string {
  const raw = params[key];
  if (Array.isArray(raw)) return raw[0] || fallback;
  return raw || fallback;
}

function normalizeBasePath(value: string | undefined): string {
  const trimmed = (value || "").trim();
  if (!trimmed) return "";
  const withLeadingSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return withLeadingSlash.replace(/\/+$/, "");
}

function appHref(path: string): string {
  return `${BASE_PATH}${path}`;
}

export default async function Home({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const page = value(params, "page", "dashboard");
  const view = value(params, "view", "benchmark_detail_delta");
  const model = value(params, "model", "");
  const tab = value(params, "tab", "knowledge");

  let meta: MetaResponse | null = null;
  let leaderboard: LeaderboardResponse | null = null;
  let loadError: string | null = null;
  const isDashboard = page !== "history" && page !== "admin";

  if (isDashboard) {
    try {
      meta = await api.meta();
      const selectedModel = model || meta.auto_label;
      leaderboard = await api.leaderboard(selectedModel, view);
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    }
  }

  const selectedModel = meta ? model || meta.auto_label : model;
  const subtitle =
    page === "history"
      ? "分数历史"
      : page === "admin"
        ? "调度器管理"
        : `评测看板 · ${leaderboard?.view_label ?? view}`;

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1>
            <span className="brand-dot">⦿</span> RWKV Skills
          </h1>
          <div className="subtitle">{subtitle}</div>
        </div>
        <nav className="page-nav">
          <a
            className={page === "dashboard" ? "active" : ""}
            href={appHref(`/?page=dashboard&view=${view}&model=${encodeURIComponent(selectedModel)}&tab=${tab}`)}
          >
            评测看板
          </a>
          <a className={page === "history" ? "active" : ""} href={appHref("/?page=history")}>
            分数历史
          </a>
          <a className={page === "admin" ? "active" : ""} href={appHref("/?page=admin")}>
            管理面板
          </a>
        </nav>
      </header>

      {loadError ? <div className="error-bar">加载评测看板失败：{loadError}</div> : null}
      {page === "history" ? (
        <ScoreHistory />
      ) : page === "admin" ? (
        <AdminPage />
      ) : meta && leaderboard ? (
        <DashboardPage
          initialMeta={meta}
          initialLeaderboard={leaderboard}
          initialModel={selectedModel}
          initialView={view}
          initialTab={tab}
        />
      ) : (
        <div className="empty">暂无数据。</div>
      )}
    </main>
  );
}
