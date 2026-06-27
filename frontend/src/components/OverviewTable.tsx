import { Fragment } from "react";
import type { OverviewRow, ParamColumn, DeltaCell } from "../api";
import { deltaClass, pct, signedPct } from "./format";

interface Props {
  rows: OverviewRow[];
  columns: ParamColumn[];
  isDelta: boolean;
}

const ROW_AXIS_WIDTH = 200;
const SCREEN_GROUP_CAPACITY = 6;
const OVERFLOW_SCORE_COL_WIDTH = 56;

/** 领域均分 (field average) with the same two-row param/architecture header. */
export function OverviewTable({ rows, columns, isDelta }: Props) {
  if (!rows.length) return <div className="empty">暂无领域均分数据。</div>;
  const span = isDelta ? 3 : 1;
  const scoreColumnCount = columns.length * span;
  const shouldFitViewport = columns.length <= SCREEN_GROUP_CAPACITY;
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
          <col className="col-axis-benchmark" />
          {columns.map((c) =>
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
            <th className="col-name axis-header axis-benchmark" rowSpan={2}>field_name</th>
            {columns.map((c, i) => (
              <th key={i} className="param-group" colSpan={span}>
                {c.param_label.toUpperCase()}
              </th>
            ))}
          </tr>
          <tr className="subhead-row">
            {columns.map((c, i) =>
              isDelta ? (
                <>
                  <th key={`${i}-p`} className="col-arch group-start subcol-prev">{c.prev_label ?? "—"}</th>
                  <th key={`${i}-l`} className="col-arch subcol-latest">{c.latest_label}</th>
                  <th key={`${i}-d`} className="col-arch col-delta subcol-delta">delta</th>
                </>
              ) : (
                <th key={i} className="col-arch group-start">{c.latest_label}</th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              <td className="bench axis-benchmark">{row.domain_title}</td>
              {row.cells.map((cell, ci) =>
                isDelta ? (
                  <DeltaTriple key={ci} cell={cell as DeltaCell} />
                ) : (
                  <td key={ci} className="score">
                    {pct((cell as { percent: number | null }).percent)}
                  </td>
                )
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DeltaTriple({ cell }: { cell: DeltaCell }) {
  return (
    <>
      <td className="score group-start subcol-prev">{pct(cell.prev)}</td>
      <td className="score subcol-latest">{pct(cell.latest)}</td>
      <td className={`score delta subcol-delta ${deltaClass(cell.delta)}`}>{signedPct(cell.delta)}</td>
    </>
  );
}
