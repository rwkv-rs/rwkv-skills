"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  getAdminToken,
  setAdminToken,
  type AdminStatus,
  type BackpressureModel,
} from "../api";

const TERMINAL = new Set(["idle", "completed", "cancelled", "failed"]);

function statusClass(status: string): string {
  if (status === "running" || status === "starting") return "stat-run";
  if (status === "paused" || status === "pausing") return "stat-warn";
  if (status === "completed") return "stat-good";
  if (status === "failed") return "stat-bad";
  return "stat-idle";
}

function fmtTime(ms: number | null): string {
  if (!ms) return "—";
  try {
    return new Date(ms).toLocaleString();
  } catch {
    return String(ms);
  }
}

/** Status banner + progress + lifecycle controls. */
function StatusPanel({ status }: { status: AdminStatus }) {
  const qc = useQueryClient();
  const isTerminal = TERMINAL.has(status.status);
  const canControl = !isTerminal && status.status !== "idle";

  const mut = (fn: () => Promise<AdminStatus>) =>
    useMutation({
      mutationFn: fn,
      onSuccess: (data) => qc.setQueryData(["admin", "status"], data),
    });

  const pause = mut(api.adminPause);
  const resume = mut(api.adminResume);
  const cancel = mut(api.adminCancel);

  const pct = Math.round((status.progress_percent ?? 0) * 100);

  return (
    <div className="card">
      <div className="admin-status-head">
        <span className={`stat-pill ${statusClass(status.status)}`}>{status.status}</span>
        {status.run_id && <span className="muted mono-sm">{status.run_id}</span>}
        {status.desired_state && status.desired_state !== status.status && (
          <span className="muted mono-sm">→ {status.desired_state}</span>
        )}
        <div className="admin-controls">
          <button className="btn" disabled={!canControl || pause.isPending} onClick={() => pause.mutate()}>
            暂停
          </button>
          <button className="btn" disabled={!canControl || resume.isPending} onClick={() => resume.mutate()}>
            恢复
          </button>
          <button className="btn btn-danger" disabled={!canControl || cancel.isPending} onClick={() => cancel.mutate()}>
            取消
          </button>
        </div>
      </div>

      {status.error && <div className="error-bar" style={{ marginTop: 12 }}>{status.error}</div>}

      <div className="progress-bar" title={`${pct}%`}>
        <div className={`progress-fill ${statusClass(status.status)}`} style={{ width: `${pct}%` }} />
      </div>

      <div className="stat-grid">
        <Stat label="总任务" value={status.tasks_total} />
        <Stat label="待处理" value={status.pending_jobs} />
        <Stat label="运行中" value={status.running_jobs} />
        <Stat label="已完成" value={status.completed_jobs} good />
        <Stat label="失败" value={status.failed_jobs} bad={status.failed_jobs > 0} />
        <Stat label="进度" value={`${pct}%`} />
      </div>

      <div className="muted mono-sm" style={{ marginTop: 10 }}>
        开始 {fmtTime(status.started_at_unix_ms)} · 更新 {fmtTime(status.updated_at_unix_ms)}
        {status.finished_at_unix_ms ? ` · 结束 ${fmtTime(status.finished_at_unix_ms)}` : ""}
      </div>
    </div>
  );
}

function Stat({ label, value, good, bad }: { label: string; value: number | string; good?: boolean; bad?: boolean }) {
  return (
    <div className="stat-cell">
      <div className="stat-num" style={{ color: bad ? "var(--bad)" : good ? "var(--good)" : undefined }}>
        {value}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

/** Live job queue + active jobs. */
function QueuePanel({ status }: { status: AdminStatus }) {
  return (
    <div className="card">
      <div className="card-title">任务队列</div>
      <div className="queue-cols">
        <div>
          <div className="queue-head-label">运行中 ({status.active_jobs.length})</div>
          {status.active_jobs.length === 0 ? (
            <div className="muted">无</div>
          ) : (
            <ul className="job-list">
              {status.active_jobs.map((j) => (
                <li key={j} className="job-item active">{j}</li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <div className="queue-head-label">队列头部 ({status.queue_head.length})</div>
          {status.queue_head.length === 0 ? (
            <div className="muted">空</div>
          ) : (
            <ul className="job-list">
              {status.queue_head.map((j) => (
                <li key={j} className="job-item">{j}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

/** GPU + remote-worker telemetry. */
function TelemetryPanel({ status }: { status: AdminStatus }) {
  const inferUrl = (status.request?.infer_base_url as string) || "";
  const [override, setOverride] = useState("");
  const url = override || inferUrl;

  const bp = useQuery({
    queryKey: ["admin", "backpressure", url],
    queryFn: () => api.adminBackpressure(url || undefined),
    refetchInterval: 5000,
  });

  return (
    <div className="card">
      <div className="card-title">GPU / 推理 worker 遥测</div>

      <div style={{ marginBottom: 12 }}>
        <span className="muted mono-sm">本地 GPU：</span>
        {status.available_gpus.length === 0 ? (
          <span className="muted"> 无上报</span>
        ) : (
          status.available_gpus.map((g) => (
            <span key={g} className="chip">{g}</span>
          ))
        )}
      </div>

      <div className="controls" style={{ marginBottom: 12 }}>
        <div className="control-group" style={{ flex: 1 }}>
          <label>infer_base_url（留空用当前任务配置）</label>
          <input
            type="text"
            value={override}
            placeholder={inferUrl || "http://127.0.0.1:8000/v1"}
            onChange={(e) => setOverride(e.target.value)}
            style={{ minWidth: 320 }}
          />
        </div>
        <button className="btn" onClick={() => bp.refetch()}>刷新</button>
      </div>

      {bp.data?.error && <div className="error-bar">{bp.data.error}</div>}
      {!url && <div className="muted">未配置远端推理服务，无 worker 遥测。</div>}

      {bp.data && bp.data.models.length > 0 && (
        <div className="pivot-wrap">
          <table className="pivot">
            <thead>
              <tr>
                <th>模型</th>
                <th>状态</th>
                <th>路由 (健康/总)</th>
                <th>排队</th>
                <th>批大小</th>
                <th>失败批</th>
                <th>tok/s</th>
              </tr>
            </thead>
            <tbody>
              {bp.data.models.map((m: BackpressureModel) => (
                <tr key={m.model_slug}>
                  <td className="bench">{m.model}</td>
                  <td className={m.status === "ok" ? "delta up" : "delta down"}>{m.status}</td>
                  <td className="score">{m.ok_route_count}/{m.route_count}</td>
                  <td className="score">{m.pending_queue}</td>
                  <td className="score">{m.max_batch_size ?? "—"}</td>
                  <td className="score">{m.failed_batches}</td>
                  <td className="score">{m.last_total_tok_s != null ? m.last_total_tok_s.toFixed(1) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/** Config editor + Start. The JSON textarea is the source of truth (46 fields). */
function ConfigPanel({ disabled }: { disabled: boolean }) {
  const qc = useQueryClient();
  const draft = useQuery({ queryKey: ["admin", "draft"], queryFn: api.adminDraft, staleTime: Infinity });
  const options = useQuery({ queryKey: ["admin", "options"], queryFn: api.adminOptions, staleTime: Infinity });

  const [text, setText] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);

  useEffect(() => {
    if (draft.data && !text) setText(JSON.stringify(draft.data, null, 2));
  }, [draft.data]);

  const start = useMutation({
    mutationFn: (payload: Record<string, unknown>) => api.adminStart(payload),
    onSuccess: (data) => qc.setQueryData(["admin", "status"], data),
  });

  const patch = (key: string, value: unknown) => {
    try {
      const obj = JSON.parse(text || "{}");
      obj[key] = value;
      setText(JSON.stringify(obj, null, 2));
      setParseError(null);
    } catch {
      setParseError("当前 JSON 无法解析，无法快捷修改。");
    }
  };

  const onStart = () => {
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(text || "{}");
      setParseError(null);
    } catch (e) {
      setParseError(`JSON 解析失败：${String(e)}`);
      return;
    }
    start.mutate(payload);
  };

  const opts = options.data;

  return (
    <div className="card">
      <div className="card-title">评测配置 · 启动</div>

      {opts && (
        <div className="controls" style={{ marginBottom: 14 }}>
          <QuickSelect label="model_select" options={opts.model_select} onPick={(v) => patch("model_select", v)} />
          <QuickSelect label="run_mode" options={opts.run_mode} onPick={(v) => patch("run_mode", v)} />
          <QuickSelect label="domains（追加）" options={opts.domains} onPick={(v) => patch("domains", [v])} />
          <QuickSelect label="protocol" options={opts.protocol} onPick={(v) => patch("infer_protocol", v)} />
        </div>
      )}

      <textarea
        className="config-editor"
        value={text}
        spellCheck={false}
        onChange={(e) => setText(e.target.value)}
      />

      {parseError && <div className="error-bar" style={{ marginTop: 10 }}>{parseError}</div>}
      {start.isError && <div className="error-bar" style={{ marginTop: 10 }}>{String(start.error)}</div>}

      <div style={{ marginTop: 12, display: "flex", gap: 10, alignItems: "center" }}>
        <button className="btn btn-primary" disabled={disabled || start.isPending} onClick={onStart}>
          {start.isPending ? "启动中…" : "启动评测"}
        </button>
        {disabled && <span className="muted mono-sm">已有任务在运行，无法启动新任务。</span>}
      </div>
    </div>
  );
}

function QuickSelect({ label, options, onPick }: { label: string; options: string[]; onPick: (v: string) => void }) {
  return (
    <div className="control-group">
      <label>{label}</label>
      <select
        defaultValue=""
        onChange={(e) => {
          if (e.target.value) onPick(e.target.value);
          e.target.value = "";
        }}
        style={{ minWidth: 160 }}
      >
        <option value="" disabled>
          选择…
        </option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  );
}

/** Optional bearer-token field (only relevant when RWKV_ADMIN_API_KEY is set). */
function TokenField({ onChange }: { onChange: () => void }) {
  const [val, setVal] = useState(getAdminToken());
  return (
    <div className="control-group">
      <label>Admin Token（可选）</label>
      <input
        type="password"
        value={val}
        placeholder="Bearer token，可留空"
        onChange={(e) => {
          setVal(e.target.value);
          setAdminToken(e.target.value.trim());
          onChange();
        }}
        style={{ minWidth: 220 }}
      />
    </div>
  );
}

export function AdminPage() {
  const qc = useQueryClient();
  const health = useQuery({ queryKey: ["admin", "health"], queryFn: api.adminHealth, retry: 0 });
  const status = useQuery({
    queryKey: ["admin", "status"],
    queryFn: api.adminStatus,
    refetchInterval: 3000,
    retry: 0,
  });

  const disabled = useMemo(() => {
    const s = status.data?.status;
    return !!s && !TERMINAL.has(s);
  }, [status.data?.status]);

  return (
    <div>
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="controls">
          <TokenField onChange={() => qc.invalidateQueries({ queryKey: ["admin"] })} />
          {health.data?.auth_required && (
            <span className="muted mono-sm">服务端已开启鉴权，请填写 Token。</span>
          )}
          {health.isError && <span className="error-bar" style={{ margin: 0 }}>无法连接 admin 接口（鉴权或服务未就绪）。</span>}
        </div>
      </div>

      {status.isError ? (
        <div className="error-bar">读取调度器状态失败：{String(status.error)}</div>
      ) : status.data ? (
        <>
          <StatusPanel status={status.data} />
          <div className="admin-grid">
            <ConfigPanel disabled={disabled} />
            <div>
              <QueuePanel status={status.data} />
              <TelemetryPanel status={status.data} />
            </div>
          </div>
        </>
      ) : (
        <div className="spinner">加载调度器状态…</div>
      )}
    </div>
  );
}
