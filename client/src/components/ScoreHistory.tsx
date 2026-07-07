"use client";

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type ScoreHistoryDetail, type ScoreHistoryGroup, type ScoreHistoryPoint } from "../api";

const NAIVE_COLOR = "#f5b14c"; // amber — 朴素榜
const NORMAL_COLOR = "#5b8cff"; // blue — 正式榜
const PER_BAR = 140; // fixed bar slot width; chart scrolls horizontally when many

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function pctText(v: number | null): string {
  return v == null ? "—" : `${v.toFixed(1)}%`;
}

/** One cot_mode chart: fixed bar width, horizontal scroll, full timestamp labels. */
function HistoryChart({
  group,
  selectedTask,
  onSelect,
}: {
  group: ScoreHistoryGroup;
  selectedTask: number | null;
  onSelect: (p: ScoreHistoryPoint) => void;
}) {
  const data = group.points.map((p, i) => ({ ...p, _label: fmtTime(p.created_at), _idx: i }));
  const width = Math.max(560, data.length * PER_BAR + 80);

  return (
    <div className="card" style={{ marginBottom: 18 }}>
      <div className="card-title">
        {group.cot_mode} · {group.points.length} 条分数
      </div>
      <div style={{ overflowX: "auto" }}>
        <BarChart
          width={width}
          height={360}
          data={data}
          margin={{ top: 8, right: 16, bottom: 44, left: 0 }}
          barCategoryGap="20%"
        >
          <CartesianGrid stroke="#252834" vertical={false} />
          <XAxis
            dataKey="_label"
            tick={{ fill: "#9aa0b0", fontSize: 11 }}
            textAnchor="middle"
            interval={0}
            height={42}
          />
          <YAxis
            tick={{ fill: "#646b7e", fontSize: 11 }}
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            cursor={{ fill: "rgba(91,140,255,0.08)" }}
            content={<HistoryTooltip />}
          />
          <Bar dataKey="percent" radius={[3, 3, 0, 0]} onClick={(d: any) => onSelect(d.payload as ScoreHistoryPoint)}>
            {data.map((p) => (
              <Cell
                key={p.score_id}
                cursor="pointer"
                fill={p.board === "naive" ? NAIVE_COLOR : NORMAL_COLOR}
                opacity={selectedTask == null || selectedTask === p.task_id ? 1 : 0.45}
              />
            ))}
          </Bar>
        </BarChart>
      </div>
    </div>
  );
}

function HistoryTooltip({ active, payload }: { active?: boolean; payload?: { payload: ScoreHistoryPoint }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="tooltip-pop" style={{ position: "static", maxWidth: 320 }}>
      {`时间: ${fmtTime(p.created_at)}
score: ${pctText(p.percent)}
metric: ${p.metric ?? "—"}
evaluator: ${p.evaluator ?? "—"}
board: ${p.board === "naive" ? "朴素榜" : "正式榜"}
task_id: ${p.task_id}`}
    </div>
  );
}

/** Right sticky detail panel for a clicked bar. */
function DetailPanel({ taskId }: { taskId: number | null }) {
  const { data, isPending, isError, error } = useQuery({
    queryKey: ["score-history-detail", taskId],
    queryFn: () => api.scoreHistoryDetail(taskId!),
    enabled: taskId != null,
    staleTime: 300_000,
  });

  if (taskId == null) return <div className="empty muted">点击任意柱子查看该分数来源。</div>;
  if (isPending) return <div className="spinner">加载中…</div>;
  if (isError) return <div className="error-bar">加载失败：{String(error)}</div>;
  if (!data || !data.found) return <div className="empty muted">未找到该 task 的详情。</div>;
  return <DetailBody d={data} />;
}

function DetailBody({ d }: { d: ScoreHistoryDetail }) {
  const stageEntries = Object.entries(d.sampling.stages);
  return (
    <div>
      <div className="sh-detail-head">
        <span className="stat-pill stat-good">{pctText(d.percent)}</span>
        <span className={`badge ${d.board === "naive" ? "fail" : "pass"}`}>
          {d.board === "naive" ? "朴素榜" : "正式榜"}
        </span>
        <span className="muted mono-sm">{d.cot_mode}</span>
        <span className="muted mono-sm">task #{d.task_id}</span>
      </div>
      <div className="muted mono-sm" style={{ marginBottom: 12 }}>
        {d.model} · {d.benchmark} · metric={d.metric ?? "—"} · {d.evaluator ?? "—"}
      </div>

      <div className="card-title">采样参数</div>
      <div className="kv" style={{ marginBottom: 12 }}>
        {`effective_sample_count: ${d.sampling.effective_sample_count ?? "—"}
avg_k: ${String(d.sampling.avg_k ?? "—")}   pass_ks: ${JSON.stringify(d.sampling.pass_ks ?? null)}
n_shot: ${String(d.sampling.n_shot ?? "—")}   sample_limit: ${String(d.sampling.sample_limit ?? "—")}
prompt_profile: ${d.sampling.prompt_profile ?? "—"}`}
      </div>
      {stageEntries.length === 0 ? (
        <div className="muted mono-sm" style={{ marginBottom: 12 }}>（该评测无生成采样参数，如 logits 直评）</div>
      ) : (
        stageEntries.map(([name, s]) => (
          <div className="stage" key={name} style={{ marginBottom: 10 }}>
            <div className="stage-label">{name}</div>
            <pre className="kv" style={{ padding: 10 }}>
              {`temperature: ${s.temperature ?? "—"}   top_k: ${s.top_k ?? "—"}   top_p: ${s.top_p ?? "—"}
max_tokens: ${s.max_tokens ?? "—"}
penalties: presence=${s.penalties.presence_penalty ?? "—"} repetition=${s.penalties.repetition_penalty ?? "—"} decay=${s.penalties.penalty_decay ?? "—"}
stop_tokens: ${s.stop_tokens.map((t) => `${t.id}(${t.token})`).join(" ") || "—"}`}
            </pre>
          </div>
        ))
      )}

      <div className="card-title" style={{ marginTop: 14 }}>Prompt（代表样本）</div>
      {d.stages.length === 0 ? (
        <div className="muted mono-sm">无 prompt context。</div>
      ) : (
        d.stages.map((s, i) => (
          <div className="stage" key={i} style={{ marginBottom: 10 }}>
            <div className="stage-label">stage {i + 1} · stop_reason={String(s.stop_reason ?? "—")}</div>
            <pre>{s.prompt || ""}</pre>
            {s.completion ? <pre style={{ borderTop: "1px solid var(--border)" }}>{s.completion}</pre> : null}
          </div>
        ))
      )}
    </div>
  );
}

export function ScoreHistory() {
  const opts = useQuery({ queryKey: ["score-history-options"], queryFn: api.scoreHistoryOptions, staleTime: 300_000 });
  const [model, setModel] = useState("");
  const [benchmark, setBenchmark] = useState("");
  const [selectedTask, setSelectedTask] = useState<number | null>(null);

  // Benchmarks available for the chosen model (from the pairs list).
  const benchmarks = useMemo(() => {
    if (!opts.data) return [];
    if (!model) return opts.data.benchmarks;
    return [...new Set(opts.data.pairs.filter((p) => p.model === model).map((p) => p.dataset))].sort();
  }, [opts.data, model]);

  // Pick sensible defaults once options load.
  useEffect(() => {
    if (opts.data && !model && opts.data.models.length) setModel(opts.data.models[0]);
  }, [opts.data]);
  useEffect(() => {
    if (benchmarks.length && !benchmarks.includes(benchmark)) setBenchmark(benchmarks[0]);
  }, [benchmarks]);

  const history = useQuery({
    queryKey: ["score-history", model, benchmark],
    queryFn: () => api.scoreHistory(model, benchmark),
    enabled: !!model && !!benchmark,
  });

  // Reset the selected bar when the (model, benchmark) changes.
  useEffect(() => {
    setSelectedTask(null);
  }, [model, benchmark]);

  return (
    <div>
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="controls">
          <div className="control-group">
            <label>模型权重</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {opts.data?.models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          <div className="control-group">
            <label>Benchmark</label>
            <select value={benchmark} onChange={(e) => setBenchmark(e.target.value)}>
              {benchmarks.map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </div>
          <button className="btn btn-primary" onClick={() => history.refetch()}>🔄 刷新</button>
          {history.data && (
            <span className="muted" style={{ fontSize: 12 }}>
              共 {history.data.total} 条分数 · {history.data.groups.length} 张图
            </span>
          )}
          {history.isFetching && <span className="muted" style={{ fontSize: 12 }}>加载中…</span>}
        </div>
      </div>

      {opts.isError && <div className="error-bar">选项加载失败：{String(opts.error)}</div>}
      {history.isError && <div className="error-bar">分数历史加载失败：{String(history.error)}</div>}

      <div className="sh-layout">
        <div className="sh-charts">
          {history.data && history.data.groups.length === 0 && (
            <div className="empty">该组合下暂无正式分数。</div>
          )}
          {history.data?.groups.map((g) => (
            <HistoryChart
              key={g.cot_mode}
              group={g}
              selectedTask={selectedTask}
              onSelect={(p) => setSelectedTask(p.task_id)}
            />
          ))}
        </div>
        <div className="sh-detail card">
          <div className="card-title">分数来源</div>
          <DetailPanel taskId={selectedTask} />
        </div>
      </div>
    </div>
  );
}
