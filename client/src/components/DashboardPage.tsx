"use client";

import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { api, type CellMeta, type LeaderboardResponse, type MetaResponse } from "../api";
import { DomainCharts } from "./DomainCharts";
import { EvalRecordsPanel } from "./EvalRecords";
import { LeaderboardTable } from "./LeaderboardTable";
import { OverviewTable } from "./OverviewTable";

type CaptureState =
  | { status: "idle" }
  | { status: "saving" }
  | { status: "saved"; path: string }
  | { status: "error"; message: string };

function queryValue(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const value = new URLSearchParams(window.location.search).get(name);
  return value && value.trim() ? value : fallback;
}

function hasQueryValue(name: string): boolean {
  if (typeof window === "undefined") return false;
  const value = new URLSearchParams(window.location.search).get(name);
  return Boolean(value && value.trim());
}

export function DashboardPage({
  initialMeta,
  initialLeaderboard,
  initialModel,
  initialView,
  initialTab,
}: {
  initialMeta?: MetaResponse;
  initialLeaderboard?: LeaderboardResponse;
  initialModel?: string;
  initialView?: string;
  initialTab?: string;
}) {
  const qc = useQueryClient();
  const meta = useQuery({
    queryKey: ["meta"],
    queryFn: api.meta,
    initialData: initialMeta,
    staleTime: 300_000,
  });

  const [model, setModel] = useState<string>(() => initialModel ?? queryValue("model", ""));
  const [view, setView] = useState<string>(() => initialView ?? queryValue("view", "benchmark_detail_delta"));
  const [activeTab, setActiveTab] = useState<string>(() => initialTab ?? queryValue("tab", "knowledge"));
  const [tabPinned, setTabPinned] = useState<boolean>(() => Boolean(initialTab) || hasQueryValue("tab"));
  const [clickedMeta, setClickedMeta] = useState<CellMeta | null>(null);
  const [capture, setCapture] = useState<CaptureState>({ status: "idle" });

  useEffect(() => {
    if (!meta.data) return;
    if (!model) setModel(meta.data.auto_label);
    if (!meta.data.table_views.some((v) => v.key === view)) setView(meta.data.default_view);
    if (!meta.data.domain_groups.some((g) => g.key === activeTab)) {
      setActiveTab(meta.data.domain_groups[0]?.key ?? "knowledge");
    }
  }, [activeTab, meta.data, model, view]);

  const lb = useQuery({
    queryKey: ["leaderboard", model, view],
    queryFn: () => api.leaderboard(model || meta.data!.auto_label, view),
    initialData: initialLeaderboard,
    enabled: !!meta.data,
    staleTime: 120_000,
  });

  const leaderboard: LeaderboardResponse | undefined = lb.data;

  useEffect(() => {
    if (!leaderboard || tabPinned || activeTab === "naive") return;
    const current = leaderboard.domains.find((d) => d.key === activeTab);
    if (current?.rows.length) return;
    const firstPopulated = leaderboard.domains.find((d) => d.rows.length)?.key;
    if (firstPopulated && firstPopulated !== activeTab) setActiveTab(firstPopulated);
  }, [activeTab, leaderboard, tabPinned]);

  const refresh = () => {
    api.refresh().then(() => {
      qc.invalidateQueries({ queryKey: ["leaderboard"] });
      qc.invalidateQueries({ queryKey: ["meta"] });
    });
  };

  const capturePage = async () => {
    if (typeof window === "undefined") return;
    const target = new URL(window.location.href);
    target.searchParams.set("page", "dashboard");
    target.searchParams.set("tab", activeTab);
    target.searchParams.set("view", view);
    if (model) target.searchParams.set("model", model);

    setCapture({ status: "saving" });
    try {
      const result = await api.capturePage({
        url: target.toString(),
        width: Math.max(window.innerWidth, 1440),
        height: Math.max(window.innerHeight, 1000),
      });
      setCapture({ status: "saved", path: result.path });
    } catch (err) {
      setCapture({ status: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  return (
    <>
      {meta.isError && <div className="error-bar">元数据加载失败：{String(meta.error)}</div>}
      {lb.isError && <div className="error-bar">排行榜加载失败：{String(lb.error)}</div>}
      {leaderboard?.errors?.length ? <div className="error-bar">{leaderboard.errors.join("; ")}</div> : null}

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
            刷新数据
          </button>
          <button className="btn" onClick={capturePage} disabled={capture.status === "saving"}>
            {capture.status === "saving" ? "截图中..." : "长截图"}
          </button>
          {leaderboard?.selection && (
            <span className="muted" style={{ fontSize: 12 }}>
              {leaderboard.selection.model_sequence.length} 个模型
              {leaderboard.selection.skipped_small_params > 0
                ? ` 跳过 ${leaderboard.selection.skipped_small_params} 个小参数档位`
                : ""}
            </span>
          )}
          {lb.isFetching && <span className="muted" style={{ fontSize: 12 }}>加载中...</span>}
          {capture.status === "saved" && <span className="capture-status">已保存：{capture.path}</span>}
          {capture.status === "error" && <span className="capture-status error">截图失败：{capture.message}</span>}
        </div>
      </div>

      <nav className="tabs">
        {(meta.data?.domain_groups ?? []).map((g) => (
          <button
            key={g.key}
            className={`tab${activeTab === g.key ? " active" : ""}`}
            onClick={() => {
              setActiveTab(g.key);
              setTabPinned(true);
            }}
          >
            {g.label}
          </button>
        ))}
      </nav>

      {leaderboard ? renderLeaderboard(leaderboard, activeTab, setClickedMeta) : null}
      {lb.isFetching && !leaderboard && <div className="spinner">加载排行榜...</div>}

      {clickedMeta ? (
        <div className="card" style={{ marginTop: 20 }}>
          <EvalRecordsPanel meta={clickedMeta} />
        </div>
      ) : null}
    </>
  );
}

function renderLeaderboard(
  leaderboard: LeaderboardResponse,
  activeTab: string,
  setClickedMeta: (meta: CellMeta) => void,
) {
  if (activeTab === "naive") {
    const nb = leaderboard.naive_board;
    if (!nb || !nb.rows.length) return <div className="empty">朴素榜暂无数据。</div>;
    return (
      <div className="card">
        <div className="card-title">朴素榜 · {leaderboard.view_label}</div>
        <LeaderboardTable
          paramColumns={nb.param_columns}
          isDelta={nb.is_delta}
          rows={nb.rows}
          onCellClick={setClickedMeta}
        />
      </div>
    );
  }
  if (leaderboard.is_field_avg && leaderboard.overview) {
    return (
      <div className="card">
        <div className="card-title">领域均分 · {leaderboard.view_label}</div>
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
          paramColumns={domain.param_columns}
          isDelta={leaderboard.is_delta}
          rows={domain.rows}
          onCellClick={setClickedMeta}
        />
      </div>
      {activeTab === "coding" ? (
        <div className="card">
          <div className="card-title">图表</div>
          <DomainCharts chart={leaderboard.charts.coding} />
        </div>
      ) : null}
    </>
  );
}
