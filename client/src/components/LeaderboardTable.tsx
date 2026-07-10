"use client";

import { Fragment, useState } from "react";
import type {
  CellMeta,
  DeltaCell,
  DetailCell,
  LeaderboardRow,
  ParamColumn,
} from "../api";
import { deltaClass, pct, signedPct } from "./format";

interface Props {
  /** Param columns + delta flag come from the leaderboard payload. */
  paramColumns: ParamColumn[];
  isDelta: boolean;
  rows: LeaderboardRow[];
  onCellClick: (meta: CellMeta) => void;
}

const ROW_AXIS_COLUMNS = [
  { className: "axis-benchmark", label: "benchmark" },
  { className: "axis-samples", label: "samples" },
  { className: "axis-method", label: "eval_method" },
  { className: "axis-kmetric", label: "k_metric" },
] as const;
const ROW_AXIS_WIDTH = 484;
const SCREEN_GROUP_CAPACITY = 6;
const OVERFLOW_SCORE_COL_WIDTH = 56;

function Tooltip({ text }: { text: string }) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  return (
    <>
      <span
        onMouseMove={(e) => setPos({ x: e.clientX + 14, y: e.clientY + 14 })}
        onMouseLeave={() => setPos(null)}
        className="cell-info"
      >
        ⓘ
      </span>
      {pos && (
        <div className="tooltip-pop" style={{ left: pos.x, top: pos.y }}>
          {text}
        </div>
      )}
    </>
  );
}

function ScoreCell({
  cell,
  onClick,
  className = "",
}: {
  cell: DetailCell;
  onClick: (m: CellMeta) => void;
  className?: string;
}) {
  const meta = cell.meta;
  const clickable = meta?.clickable ?? false;
  return (
    <td
      className={`score ${className}${clickable ? " clickable" : ""}`}
      onClick={clickable && meta ? () => onClick(meta) : undefined}
    >
      <span className="score-cell">{pct(cell.percent)}</span>
      {meta?.tooltip ? <Tooltip text={meta.tooltip} /> : null}
    </td>
  );
}

/** One param column in delta view renders as three <td>: prev / latest / delta. */
function DeltaCells({ cell, onClick }: { cell: DeltaCell; onClick: (m: CellMeta) => void }) {
  return (
    <>
      <td
        className={`score group-start subcol-prev${cell.prev_meta?.clickable ? " clickable" : ""}`}
        onClick={cell.prev_meta?.clickable ? () => onClick(cell.prev_meta!) : undefined}
      >
        {pct(cell.prev)}
        {cell.prev_meta?.tooltip ? <Tooltip text={cell.prev_meta.tooltip} /> : null}
      </td>
      <td
        className={`score subcol-latest${cell.latest_meta?.clickable ? " clickable" : ""}`}
        onClick={cell.latest_meta?.clickable ? () => onClick(cell.latest_meta!) : undefined}
      >
        {pct(cell.latest)}
        {cell.latest_meta?.tooltip ? <Tooltip text={cell.latest_meta.tooltip} /> : null}
      </td>
      <td className={`score delta subcol-delta ${deltaClass(cell.delta)}`}>{signedPct(cell.delta)}</td>
    </>
  );
}

/**
 * Faithful reproduction of the old Gradio `_render_pivot_html` two-row header:
 * Two-dimensional benchmark matrix:
 *  - row axis: benchmark / samples / eval_method / k_metric.
 *  - column axis: every available parameter size, each split into
 *    prev/latest/delta sub-columns in delta view.
 */
export function LeaderboardTable({ paramColumns, isDelta, rows, onCellClick }: Props) {
  if (!rows.length) {
    return <div className="empty">该领域暂无满足展示阈值的数据。</div>;
  }
  const span = isDelta ? 3 : 1;
  const scoreColumnCount = paramColumns.length * span;
  const shouldFitViewport = paramColumns.length <= SCREEN_GROUP_CAPACITY;
  const minTableWidth = shouldFitViewport
    ? "100%"
    : `${ROW_AXIS_WIDTH + scoreColumnCount * OVERFLOW_SCORE_COL_WIDTH}px`;
  const scoreColumnWidth = shouldFitViewport
    ? `calc((100% - ${ROW_AXIS_WIDTH}px) / ${scoreColumnCount})`
    : `${OVERFLOW_SCORE_COL_WIDTH}px`;

  return (
    <div className="pivot-wrap">
      <table className="pivot bench-table" style={{ minWidth: minTableWidth }}>
        <colgroup>
          {ROW_AXIS_COLUMNS.map((col) => (
            <col key={col.className} className={`col-${col.className}`} />
          ))}
          {paramColumns.map((c) =>
            isDelta ? (
              <Fragment key={c.param}>
                <col className="col-score" style={{ width: scoreColumnWidth }} />
                <col className="col-score" style={{ width: scoreColumnWidth }} />
                <col className="col-score col-delta-width" style={{ width: scoreColumnWidth }} />
              </Fragment>
            ) : (
              <col key={c.param} className="col-score" style={{ width: scoreColumnWidth }} />
            )
          )}
        </colgroup>
        <thead>
          <tr className="group-row">
            {ROW_AXIS_COLUMNS.map((col) => (
              <th
                key={col.className}
                className={`col-meta axis-header ${col.className}`}
                rowSpan={2}
              >
                {col.label}
              </th>
            ))}
            {paramColumns.map((c, i) => (
              <th key={i} className="param-group" colSpan={span}>
                {c.param_label.toUpperCase()}
              </th>
            ))}
          </tr>
          <tr className="subhead-row">
            {paramColumns.map((c, i) =>
              isDelta ? (
                <Fragment3 key={i} prev={c.prev_label ?? "—"} latest={c.latest_label} />
              ) : (
                <th key={i} className="col-arch group-start">{c.latest_label}</th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              <td className="bench axis-benchmark">{row.benchmark_name}</td>
              <td className="dim axis-samples">{row.num_samples ?? "—"}</td>
              <td className="dim axis-method">{row.eval_method}</td>
              <td className="dim axis-kmetric">{row.k_metric}</td>
              {row.cells.map((cell, ci) =>
                isDelta ? (
                  <DeltaCells key={ci} cell={cell as DeltaCell} onClick={onCellClick} />
                ) : (
                  <ScoreCell key={ci} cell={cell as DetailCell} onClick={onCellClick} className="group-start" />
                )
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Three architecture sub-headers for a delta param column. */
function Fragment3({ prev, latest }: { prev: string; latest: string }) {
  return (
    <>
      <th className="col-arch group-start subcol-prev">{prev}</th>
      <th className="col-arch subcol-latest">{latest}</th>
      <th className="col-arch col-delta subcol-delta">delta</th>
    </>
  );
}
