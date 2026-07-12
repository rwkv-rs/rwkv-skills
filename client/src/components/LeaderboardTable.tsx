"use client";

import { Fragment, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TooltipContentProps, TooltipValueType } from "recharts";
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
const CURVE_POPOVER_WIDTH = 440;
const CURVE_POPOVER_HEIGHT = 360;
const CHART_AXIS = { fill: "#646b7e", fontSize: 11 };
const CHART_GRID = "#252834";
const PREV_LINE = "#f5b14c";
const LATEST_LINE = "#5b8cff";

function InfoTooltip({ text }: { text: string }) {
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
      {meta?.tooltip ? <InfoTooltip text={meta.tooltip} /> : null}
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
        {cell.prev_meta?.tooltip ? <InfoTooltip text={cell.prev_meta.tooltip} /> : null}
      </td>
      <td
        className={`score subcol-latest${cell.latest_meta?.clickable ? " clickable" : ""}`}
        onClick={cell.latest_meta?.clickable ? () => onClick(cell.latest_meta!) : undefined}
      >
        {pct(cell.latest)}
        {cell.latest_meta?.tooltip ? <InfoTooltip text={cell.latest_meta.tooltip} /> : null}
      </td>
      <td className={`score delta subcol-delta ${deltaClass(cell.delta)}`}>{signedPct(cell.delta)}</td>
    </>
  );
}

interface CurvePoint {
  axisLabel: string;
  paramLabel: string;
  prev: number | null;
  latest: number | null;
  delta: number | null;
  prevModelLabel: string;
  latestModelLabel: string;
}

function finiteScore(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function chartTick(value: number) {
  return `${Math.round(value)}%`;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(value, max));
}

function curvePopoverPosition(target: HTMLElement) {
  const rect = target.getBoundingClientRect();
  const margin = 12;
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  let left = rect.right + margin;
  if (left + CURVE_POPOVER_WIDTH > viewportWidth - margin) {
    left = rect.left - CURVE_POPOVER_WIDTH - margin;
  }
  left = clamp(left, margin, Math.max(margin, viewportWidth - CURVE_POPOVER_WIDTH - margin));
  const top = clamp(
    rect.top - margin,
    margin,
    Math.max(margin, viewportHeight - CURVE_POPOVER_HEIGHT - margin)
  );
  return { left, top };
}

function buildCurvePoints(row: LeaderboardRow, paramColumns: ParamColumn[], isDelta: boolean): CurvePoint[] {
  return paramColumns.map((column, index) => {
    const cell = row.cells[index];
    if (isDelta) {
      const deltaCell = cell as DeltaCell | undefined;
      return {
        axisLabel: column.param_label.toUpperCase(),
        paramLabel: column.param,
        prev: finiteScore(deltaCell?.prev),
        latest: finiteScore(deltaCell?.latest),
        delta: finiteScore(deltaCell?.delta),
        prevModelLabel: column.prev_label ?? "previous",
        latestModelLabel: column.latest_label,
      };
    }
    const detailCell = cell as DetailCell | undefined;
    return {
      axisLabel: column.param_label.toUpperCase(),
      paramLabel: column.param,
      prev: null,
      latest: finiteScore(detailCell?.percent),
      delta: null,
      prevModelLabel: "previous",
      latestModelLabel: column.latest_label,
    };
  });
}

function CurveTooltipContent({
  active,
  payload,
  isDelta,
}: TooltipContentProps<TooltipValueType, string | number> & { isDelta: boolean }) {
  if (!active || !payload?.length) return null;
  const point = payload.find((entry) => entry.payload)?.payload as CurvePoint | undefined;
  if (!point) return null;
  return (
    <div className="benchmark-curve-tooltip">
      <div className="benchmark-curve-tooltip-title">{point.axisLabel}</div>
      {isDelta ? (
        <>
          <div>{`previous (${point.prevModelLabel}): ${pct(point.prev)}`}</div>
          <div>{`latest (${point.latestModelLabel}): ${pct(point.latest)}`}</div>
          <div className={`delta ${deltaClass(point.delta)}`}>{`delta: ${signedPct(point.delta)}`}</div>
        </>
      ) : (
        <div>{`${point.latestModelLabel}: ${pct(point.latest)}`}</div>
      )}
    </div>
  );
}

function BenchmarkCurvePopover({
  row,
  paramColumns,
  isDelta,
  style,
  onMouseEnter,
  onMouseLeave,
}: {
  row: LeaderboardRow;
  paramColumns: ParamColumn[];
  isDelta: boolean;
  style: React.CSSProperties;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}) {
  const points = buildCurvePoints(row, paramColumns, isDelta);
  const prevCount = points.filter((point) => point.prev !== null).length;
  const latestCount = points.filter((point) => point.latest !== null).length;
  const hasCurve = isDelta ? prevCount >= 2 || latestCount >= 2 : latestCount >= 2;
  const chartWidth = Math.max(360, points.length * 62);
  const subtitle = `${row.eval_method} / ${row.k_metric} / samples=${row.num_samples ?? "—"}`;

  return (
    <div
      className="benchmark-curve-popover"
      style={style}
      role="dialog"
      aria-label={`${row.benchmark_name} score curve`}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="benchmark-curve-head">
        <div className="benchmark-curve-title">{row.benchmark_name}</div>
        <div className="benchmark-curve-subtitle">{subtitle}</div>
      </div>
      {hasCurve ? (
        <div className="benchmark-curve-chart-scroll">
          <LineChart
            width={chartWidth}
            height={190}
            data={points}
            margin={{ top: 8, right: 18, bottom: 22, left: -12 }}
          >
            <CartesianGrid stroke={CHART_GRID} vertical={false} />
            <XAxis dataKey="axisLabel" tick={CHART_AXIS} interval={0} height={34} />
            <YAxis tick={CHART_AXIS} tickFormatter={chartTick} domain={[0, 100]} width={42} />
            <ChartTooltip
              cursor={{ stroke: "#646b7e", strokeDasharray: "3 3" }}
              content={(props) => (
                <CurveTooltipContent {...props} isDelta={isDelta} />
              )}
            />
            {isDelta && prevCount >= 2 ? (
              <Line
                type="linear"
                dataKey="prev"
                name="previous"
                stroke={PREV_LINE}
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 4 }}
                connectNulls={false}
                isAnimationActive={false}
              />
            ) : null}
            {latestCount >= 2 ? (
              <Line
                type="linear"
                dataKey="latest"
                name={isDelta ? "latest" : "score"}
                stroke={LATEST_LINE}
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 4 }}
                connectNulls={false}
                isAnimationActive={false}
              />
            ) : null}
          </LineChart>
        </div>
      ) : (
        <div className="benchmark-curve-empty">曲线数据不足</div>
      )}
      <div className="benchmark-curve-values">
        {points.map((point) => (
          <div className="benchmark-curve-value-row" key={point.paramLabel}>
            <span>{point.axisLabel}</span>
            {isDelta ? (
              <span>
                {`${pct(point.prev)} -> ${pct(point.latest)} `}
                <span className={`delta ${deltaClass(point.delta)}`}>{signedPct(point.delta)}</span>
              </span>
            ) : (
              <span>{pct(point.latest)}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function BenchmarkNameCell({
  row,
  paramColumns,
  isDelta,
}: {
  row: LeaderboardRow;
  paramColumns: ParamColumn[];
  isDelta: boolean;
}) {
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null);

  function clearCloseTimer() {
    if (closeTimerRef.current !== null) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }

  function openPopover() {
    clearCloseTimer();
    if (triggerRef.current) {
      setPosition(curvePopoverPosition(triggerRef.current));
    }
  }

  function closePopover() {
    clearCloseTimer();
    setPosition(null);
  }

  function scheduleClose() {
    clearCloseTimer();
    closeTimerRef.current = setTimeout(closePopover, 120);
  }

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="benchmark-name-trigger"
        onMouseEnter={openPopover}
        onMouseLeave={scheduleClose}
        onFocus={openPopover}
        onBlur={scheduleClose}
        aria-haspopup="dialog"
        aria-expanded={position !== null}
      >
        {row.benchmark_name}
      </button>
      {position && typeof document !== "undefined"
        ? createPortal(
            <BenchmarkCurvePopover
              row={row}
              paramColumns={paramColumns}
              isDelta={isDelta}
              style={{ left: position.left, top: position.top }}
              onMouseEnter={clearCloseTimer}
              onMouseLeave={closePopover}
            />,
            document.body
          )
        : null}
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
              <td className="bench axis-benchmark">
                <BenchmarkNameCell row={row} paramColumns={paramColumns} isDelta={isDelta} />
              </td>
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
