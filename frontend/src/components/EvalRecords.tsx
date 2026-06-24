import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type CellMeta, type EvalRecord } from "../api";

interface Props {
  meta: CellMeta | null;
}

export function EvalRecordsPanel({ meta }: Props) {
  const [onlyWrong, setOnlyWrong] = useState(false);
  const [page, setPage] = useState(0);
  const limit = 15;

  const taskId = meta?.task_id ?? null;
  const {
    data,
    isPending,
    isError,
    error,
  } = useQuery({
    queryKey: ["eval-records", taskId, onlyWrong, page],
    queryFn: () => api.evalRecords(taskId!, onlyWrong, limit, page * limit),
    enabled: taskId !== null,
    placeholderData: (prev) => prev, // keep previous page while loading
  });

  if (!meta) {
    return <div className="empty muted">点击上方的分数单元格，即可查看评测明细。</div>;
  }
  if (!taskId) {
    return <div className="empty muted">该单元格没有可查询的 task ID。</div>;
  }

  return (
    <div>
      <div className="card-title" style={{ marginBottom: 10 }}>
        评测明细 · {meta.benchmark_name} · {meta.eval_method} · {meta.k_metric}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <label className="toggle">
          <input type="checkbox" checked={onlyWrong} onChange={(e) => { setOnlyWrong(e.target.checked); setPage(0); }} />
          仅展示错题
        </label>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
            ← 上一页
          </button>
          <button className="btn" disabled={!data?.has_more} onClick={() => setPage((p) => p + 1)}>
            下一页 →
          </button>
        </div>
      </div>
      {isError && <div className="error-bar">加载失败：{String(error)}</div>}
      {isPending && !data ? (
        <div className="spinner">加载中…</div>
      ) : (
        <EvalTable records={data?.records ?? []} taskId={taskId} />
      )}
    </div>
  );
}

function EvalTable({ records, taskId }: { records: EvalRecord[]; taskId: number }) {
  const [contextRec, setContextRec] = useState<EvalRecord | null>(null);
  if (!records.length) return <div className="empty">暂无数据。</div>;
  return (
    <>
      <table className="eval-table">
        <thead>
          <tr>
            <th>#</th>
            <th>r</th>
            <th>p</th>
            <th>output</th>
            <th>pass</th>
            <th>fail reason</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {records.map((r) => (
            <tr key={`${r.sample_index}-${r.repeat_index}-${r.pass_index}`}>
              <td>{r.sample_index}</td>
              <td>{r.repeat_index}</td>
              <td>{r.pass_index}</td>
              <td className="pre">{r.answer?.slice(0, 140) || "—"}</td>
              <td>
                <span className={`badge ${r.is_passed ? "pass" : "fail"}`}>{r.is_passed ? "✓ pass" : "✗ fail"}</span>
              </td>
              <td className="dim">{r.fail_reason?.slice(0, 80) || "—"}</td>
              <td>
                <button className="btn" onClick={() => setContextRec(r)}>
                  context
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {contextRec && (
        <ContextModal
          taskId={taskId}
          sampleIndex={contextRec.sample_index}
          repeatIndex={contextRec.repeat_index}
          passIndex={contextRec.pass_index}
          onClose={() => setContextRec(null)}
        />
      )}
    </>
  );
}

/** Private: context modal with lazy API fetch. */
function ContextModal({
  taskId,
  sampleIndex,
  repeatIndex,
  passIndex,
  onClose,
}: {
  taskId: number;
  sampleIndex: number;
  repeatIndex: number;
  passIndex: number;
  onClose: () => void;
}) {
  const { data, isPending, isError, error } = useQuery({
    queryKey: ["eval-context", taskId, sampleIndex, repeatIndex, passIndex],
    queryFn: () => api.evalContext(taskId, sampleIndex, repeatIndex, passIndex),
    staleTime: 300_000,
  });
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span className="card-title" style={{ margin: 0 }}>
            context · sample={sampleIndex} repeat={repeatIndex} pass={passIndex}
          </span>
          <button className="btn" onClick={onClose}>✕</button>
        </div>
        <div style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
          {isPending && <div className="spinner">加载中…</div>}
          {isError && <div className="error-bar">加载失败：{String(error)}</div>}
          {data?.view === "structured" && data.context && (
            <StructuredContext context={data.context} stopTokens={data.stop_tokens} errors={data.errors} />
          )}
          {data?.view === "text" && (
            <pre className="kv" style={{ padding: 20 }}>
              {data.raw_text}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

type StopTokens = Record<string, { id: number; token: string }[]>;

function StructuredContext({
  context,
  stopTokens,
  errors,
}: {
  context: Record<string, unknown>;
  stopTokens: StopTokens;
  errors: string[];
}) {
  const stages: unknown[] = Array.isArray((context as Record<string, unknown>).stages)
    ? ((context as Record<string, unknown>).stages as unknown[])
    : [];
  const samplingConfig: Record<string, unknown> =
    ((context as Record<string, unknown>).sampling_config as Record<string, unknown>) ?? {};
  return (
    <div className="modal-body">
      <div className="modal-col">
        {stages.length ? (
          stages.map((s, i) => {
            const stage = s as Record<string, unknown>;
            return (
              <div className="stage" key={i}>
                <div className="stage-label">
                  stage {i + 1} · stop_reason={String(stage.stop_reason ?? "—")}
                </div>
                <pre>{String(stage.prompt || "")}</pre>
                <pre>{String(stage.completion || "")}</pre>
              </div>
            );
          })
        ) : (
          <pre className="kv">{JSON.stringify(context, null, 2)}</pre>
        )}
        {errors.length > 0 && <div className="error-bar" style={{ marginTop: 12 }}>{errors.join("; ")}</div>}
      </div>
      <div className="modal-col right">
        <div className="card-title">sampling config</div>
        <pre className="kv">{JSON.stringify(samplingConfig, null, 2)}</pre>
        {Object.keys(stopTokens).length > 0 && (
          <>
            <div className="card-title" style={{ marginTop: 16 }}>stop tokens</div>
            {Object.entries(stopTokens).map(([stageName, tokens]) => (
              <div key={stageName}>
                <div className="muted" style={{ fontSize: 11 }}>{stageName}</div>
                <pre className="kv">{tokens.map((t) => `${t.id}\t${t.token}`).join("\n")}</pre>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
