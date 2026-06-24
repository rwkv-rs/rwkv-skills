import { useState } from "react";
import type {
  CellMeta,
  DeltaCell,
  DetailCell,
  DomainTable,
  LeaderboardResponse,
  ParamColumn,
} from "../api";
import { deltaClass, pct, signedPct } from "./format";

interface Props {
  data: LeaderboardResponse;
  domain: DomainTable;
  onCellClick: (meta: CellMeta) => void;
}

function Tooltip({ text }: { text: string }) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  return (
    <>
      <span
        onMouseMove={(e) => setPos({ x: e.clientX + 14, y: e.clientY + 14 })}
        onMouseLeave={() => setPos(null)}
        style={{ borderBottom: "1px dotted var(--text-muted)" }}
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

function ScoreCell({ cell, onClick }: { cell: DetailCell; onClick: (m: CellMeta) => void }) {
  const meta = cell.meta;
  const clickable = meta?.clickable ?? false;
  return (
    <td
      className={`score${clickable ? " clickable" : ""}`}
      onClick={clickable && meta ? () => onClick(meta) : undefined}
    >
      <span className="score-cell">{pct(cell.percent)}</span>
      {meta?.tooltip ? <Tooltip text={meta.tooltip} /> : null}
    </td>
  );
}

function DeltaCells({ cell, onClick }: { cell: DeltaCell; onClick: (m: CellMeta) => void }) {
  return (
    <>
      <td
        className={`score${cell.prev_meta?.clickable ? " clickable" : ""}`}
        onClick={cell.prev_meta?.clickable ? () => onClick(cell.prev_meta!) : undefined}
      >
        {pct(cell.prev)}
      </td>
      <td
        className={`score${cell.latest_meta?.clickable ? " clickable" : ""}`}
        onClick={cell.latest_meta?.clickable ? () => onClick(cell.latest_meta!) : undefined}
      >
        {pct(cell.latest)}
        {cell.latest_meta?.tooltip ? <Tooltip text={cell.latest_meta.tooltip} /> : null}
      </td>
      <td className={`score delta ${deltaClass(cell.delta)}`}>{signedPct(cell.delta)}</td>
    </>
  );
}

function headerCols(cols: ParamColumn[], isDelta: boolean): string[] {
  if (!isDelta) return cols.map((c) => `${c.param_label} · ${c.latest_label}`);
  const out: string[] = [];
  for (const c of cols) {
    out.push(`${c.param_label} prev (${c.prev_label ?? "—"})`);
    out.push(`${c.param_label} latest (${c.latest_label})`);
    out.push(`${c.param_label} Δ`);
  }
  return out;
}

export function LeaderboardTable({ data, domain, onCellClick }: Props) {
  if (!domain.rows.length) {
    return <div className="empty">该领域暂无满足展示阈值的数据。</div>;
  }
  const isDelta = data.is_delta;
  const cols = headerCols(data.param_columns, isDelta);

  return (
    <div className="pivot-wrap">
      <table className="pivot">
        <thead>
          <tr>
            <th>benchmark</th>
            <th>n</th>
            <th>method</th>
            <th>k</th>
            {cols.map((c, i) => (
              <th key={i}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {domain.rows.map((row, ri) => (
            <tr key={ri}>
              <td className="bench">{row.benchmark_name}</td>
              <td className="dim">{row.num_samples ?? "—"}</td>
              <td className="dim">{row.eval_method}</td>
              <td className="dim">{row.k_metric}</td>
              {row.cells.map((cell, ci) =>
                isDelta ? (
                  <DeltaCells key={ci} cell={cell as DeltaCell} onClick={onCellClick} />
                ) : (
                  <ScoreCell key={ci} cell={cell as DetailCell} onClick={onCellClick} />
                )
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
