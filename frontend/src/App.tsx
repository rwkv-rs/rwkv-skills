import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type CellMeta, type LeaderboardResponse } from "./api";
import { LeaderboardTable } from "./components/LeaderboardTable";
import { OverviewTable } from "./components/OverviewTable";
import { DomainCharts } from "./components/DomainCharts";
import { EvalRecordsPanel } from "./components/EvalRecords";
import { AdminPage } from "./components/Admin";

type Page = "dashboard" | "admin";

export function App() {
  const qc = useQueryClient();

  // ---- top-level page ----
  const [page, setPage] = useState<Page>("dashboard");

  // ---- meta (model list, domains, view labels) ----
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta, staleTime: 300_000 });

  // ---- control state ----
  const [model, setModel] = useState<string>("");
  const [view, setView] = useState<string>("benchmark_detail_latest");
  const [activeTab, setActiveTab] = useState<string>("knowledge");
  const [clickedMeta, setClickedMeta] = useState<CellMeta | null>(null);

  // Once meta loads, set defaults
  useMemo(() => {
    if (meta.data && !model) setModel(meta.data.auto_label);
    if (meta.data && view === "benchmark_detail_latest") setView(meta.data.default_view);
  }, [meta.data]);

  // ---- leaderboard ----
  const lb = useQuery({
    queryKey: ["leaderboard", model, view],
    queryFn: () => api.leaderboard(model || meta.data!.auto_label, view),
    enabled: !!meta.data,
    staleTime: 120_000,
  });

  const leaderboard: LeaderboardResponse | undefined = lb.data;

  // ---- helpers ----
  const refresh = () => {
    api.refresh().then(() => {
      qc.invalidateQueries({ queryKey: ["leaderboard"] });
      qc.invalidateQueries({ queryKey: ["meta"] });
    });
  };

  const viewLabel = meta.data?.table_views.find((v) => v.key === view)?.label ?? view;

  return (
    <div className="app-shell">
      {/* ---- Header ---- */}
      <header className="app-header">
        <div>
          <h1>
            <span className="brand-dot">⦿</span> RWKV Skills
          </h1>
          <div className="subtitle">
            {page === "dashboard" ? `评测看板 · ${viewLabel}` : "调度器管理"}
          </div>
        </div>
        <nav className="page-nav">
          <button
            className={`page-nav-btn${page === "dashboard" ? " active" : ""}`}
            onClick={() => setPage("dashboard")}
          >
            评测看板
          </button>
          <button
            className={`page-nav-btn${page === "admin" ? " active" : ""}`}
            onClick={() => setPage("admin")}
          >
            管理面板
          </button>
        </nav>
      </header>

      {page === "admin" ? (
        <AdminPage />
      ) : (
        <>
      {/* ---- Errors ---- */}
      {meta.isError && <div className="error-bar">元数据加载失败：{String(meta.error)}</div>}
      {lb.isError && <div className="error-bar">排行榜加载失败：{String(lb.error)}</div>}
      {leaderboard?.errors?.length ? (
        <div className="error-bar">{leaderboard.errors.join("; ")}</div>
      ) : null}

      {/* ---- Controls ---- */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="controls">
          <div className="control-group">
            <label>模型选择</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {meta.data?.model_choices.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
          <div className="control-group">
            <label>视图模式</label>
            <select value={view} onChange={(e) => setView(e.target.value)}>
              {meta.data?.table_views.map((v) => (
                <option key={v.key} value={v.key}>
                  {v.label}
                </option>
              ))}
            </select>
          </div>
          <button className="btn btn-primary" onClick={refresh}>
            🔄 刷新数据
          </button>
          {leaderboard?.selection && (
            <span className="muted" style={{ fontSize: 12 }}>
              {leaderboard.selection.model_sequence.length} 个模型 ·
              {leaderboard.selection.skipped_small_params > 0
                ? ` 跳过 ${leaderboard.selection.skipped_small_params} 个小参数档位`
                : ""}
            </span>
          )}
          {lb.isFetching && <span className="muted" style={{ fontSize: 12 }}>加载中…</span>}
        </div>
      </div>

      {/* ---- Tabs ---- */}
      <nav className="tabs">
        {(meta.data?.domain_groups ?? []).map((g) => (
          <button
            key={g.key}
            className={`tab${activeTab === g.key ? " active" : ""}`}
            onClick={() => setActiveTab(g.key)}
          >
            {g.label}
          </button>
        ))}
      </nav>

      {/* ---- Tab content ---- */}
      {leaderboard && (() => {
        if (leaderboard.is_field_avg && leaderboard.overview) {
          return (
            <div className="card">
              <div className="card-title">领域均分</div>
              <OverviewTable rows={leaderboard.overview} columns={leaderboard.param_columns} isDelta={leaderboard.is_delta} />
            </div>
          );
        }
        const domain = leaderboard.domains.find((d) => d.key === activeTab);
        if (!domain) return <div className="empty">暂无数据。</div>;
        return (
          <>
            <div className="card">
              <div className="card-title">
                {domain.title} · {leaderboard.view_label}
              </div>
              <LeaderboardTable
                data={leaderboard}
                domain={domain}
                onCellClick={(m) => setClickedMeta(m)}
              />
            </div>
            <div className="card">
              <div className="card-title">图表</div>
              <DomainCharts chart={leaderboard.charts[activeTab as keyof typeof leaderboard.charts]} />
            </div>
          </>
        );
      })()}
      {lb.isFetching && !leaderboard && <div className="spinner">加载排行榜…</div>}

      {/* ---- Eval records ---- */}
      <div className="card" style={{ marginTop: 20 }}>
        <EvalRecordsPanel meta={clickedMeta} />
      </div>
        </>
      )}
    </div>
  );
}
