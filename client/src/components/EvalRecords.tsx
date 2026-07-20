"use client";

import { useEffect, useState } from "react";
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

  // Reset pagination when the selected cell/task changes, so switching tasks
  // never requests a stale offset or briefly shows the previous task's rows.
  useEffect(() => {
    setPage(0);
  }, [taskId]);

  const {
    data,
    isPending,
    isError,
    error,
  } = useQuery({
    queryKey: ["eval-records", taskId, onlyWrong, page],
    queryFn: () => api.evalRecords(taskId!, onlyWrong, limit, page * limit),
    enabled: taskId !== null,
    // Keep the previous page only while paging within the SAME task; on task
    // switch, drop it so we never flash the previous task's rows.
    placeholderData: (prev, prevQuery) =>
      prevQuery && prevQuery.queryKey[1] === taskId ? prev : undefined,
  });

  if (!meta) {
    return <div className="empty muted">点击上方的分数单元格，即可查看评测明细。</div>;
  }
  if (!taskId) {
    return <div className="empty muted">该单元格没有可查询的 task ID。</div>;
  }

  const filterTag = onlyWrong ? "仅错题" : "全部";

  return (
    <div>
      <div className="card-title" style={{ marginBottom: 10 }}>
        评测明细 · {meta.benchmark_name} · {meta.eval_method} · {meta.model ?? "—"} · {filterTag}
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
            <th>sample</th>
            <th>repeat</th>
            <th>pass_idx</th>
            <th>抽取答案（evaluator）</th>
            <th>参考答案</th>
            <th>评测结果</th>
            <th>失败原因</th>
            <th>prompt 预览 / stages</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r) => (
            <tr key={`${r.sample_index}-${r.repeat_index}-${r.pass_index}`}>
              <td>{r.sample_index}</td>
              <td>{r.repeat_index}</td>
              <td>{r.pass_index}</td>
              <td className="pre" title={r.answer || "—"}>{r.answer?.slice(0, 140) || "—"}</td>
              <td className="pre" title={r.ref_answer || "—"}>{r.ref_answer?.slice(0, 140) || "—"}</td>
              <td>
                <span className={`badge ${r.is_passed ? "pass" : "fail"}`}>{r.is_passed ? "✓ pass" : "✗ fail"}</span>
              </td>
              <td className="dim" title={r.fail_reason || "—"}>{r.fail_reason?.slice(0, 80) || "—"}</td>
              <td>
                <button
                  className="context-preview-btn"
                  title="点击查看原始 prompt、各阶段 completion 与辅助 context"
                  onClick={() => setContextRec(r)}
                >
                  {r.context_preview ? `prompt: ${r.context_preview}` : "查看 prompt / stages"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {contextRec && (
        <ContextModal
          record={contextRec}
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
  record,
  taskId,
  sampleIndex,
  repeatIndex,
  passIndex,
  onClose,
}: {
  record: EvalRecord;
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
          <div>
            <span className="card-title" style={{ margin: 0 }}>
              context · sample={sampleIndex} repeat={repeatIndex} pass={passIndex}
            </span>
            <div className="context-outcome">
              <span>answer={record.answer || "—"}</span>
              <span>ref={record.ref_answer || "—"}</span>
              <span className={record.is_passed ? "outcome-pass" : "outcome-fail"}>
                {record.is_passed ? "passed" : "failed"}
              </span>
              {record.fail_reason ? <span>{record.fail_reason}</span> : null}
            </div>
          </div>
          <button className="btn" onClick={onClose}>✕</button>
        </div>
        <div style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
          {isPending && <div className="spinner">加载中…</div>}
          {isError && <div className="error-bar">加载失败：{String(error)}</div>}
          {data?.view === "structured" && data.context && (
            <StructuredContext
              context={data.context}
              stopTokens={data.stop_tokens}
              errors={data.errors}
              answer={record.answer}
              refAnswer={record.ref_answer}
            />
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
  answer,
  refAnswer,
}: {
  context: Record<string, unknown>;
  stopTokens: StopTokens;
  errors: string[];
  answer: string;
  refAnswer: string;
}) {
  const stages: unknown[] = Array.isArray((context as Record<string, unknown>).stages)
    ? ((context as Record<string, unknown>).stages as unknown[])
    : [];
  const samplingConfig: Record<string, unknown> =
    ((context as Record<string, unknown>).sampling_config as Record<string, unknown>) ?? {};
  const lastCompletion = [...stages]
    .reverse()
    .map((stage) => String((stage as Record<string, unknown>).completion ?? ""))
    .find((completion) => completion.length > 0) ?? "";
  const extraContext = Object.entries(context).filter(
    ([key]) => key !== "stages" && key !== "sampling_config",
  );

  const displayValue = (value: unknown): string => {
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value, null, 2) ?? String(value);
    } catch {
      return String(value);
    }
  };

  return (
    <div className="modal-body">
      <div className="modal-col">
        <div className="context-audit">
          <div><span className="muted">evaluator 抽取答案</span><code>{answer || "—"}</code></div>
          <div><span className="muted">参考答案</span><code>{refAnswer || "—"}</code></div>
          <div><span className="muted">最后非空 completion（模型原文）</span><code>{lastCompletion || "—"}</code></div>
        </div>
        {stages.length ? (
          stages.map((s, i) => {
            const stage = s as Record<string, unknown>;
            return (
              <div className="stage" key={i}>
                <div className="stage-label">
                  stage {i + 1} · stop_reason={String(stage.stop_reason ?? "—")}
                </div>
                <div className="stage-part">
                  <div className="stage-part-label">prompt（输入）</div>
                  <pre>{String(stage.prompt || "—")}</pre>
                </div>
                <div className="stage-part">
                  <div className="stage-part-label">completion（模型原文）</div>
                  <pre>{String(stage.completion || "—")}</pre>
                </div>
              </div>
            );
          })
        ) : (
          <pre className="kv">{JSON.stringify(context, null, 2)}</pre>
        )}
        {extraContext.length > 0 && (
          <div className="extra-context">
            <div className="card-title">辅助 context（原始结构）</div>
            {extraContext.map(([key, value]) => (
              <div className="extra-context-item" key={key}>
                <div className="stage-part-label">{key}</div>
                <pre className="kv">{displayValue(value)}</pre>
              </div>
            ))}
          </div>
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
