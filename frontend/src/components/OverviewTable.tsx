import type { OverviewRow, ParamColumn, DeltaCell } from "../api";
import { deltaClass, pct, signedPct } from "./format";

interface Props {
  rows: OverviewRow[];
  columns: ParamColumn[];
  isDelta: boolean;
}

export function OverviewTable({ rows, columns, isDelta }: Props) {
  if (!rows.length) return <div className="empty">暂无领域均分数据。</div>;
  return (
    <div className="pivot-wrap">
      <table className="pivot">
        <thead>
          <tr>
            <th>领域</th>
            {columns.map((c, i) =>
              isDelta ? (
                <th key={i} colSpan={3}>
                  {c.param_label} · {c.latest_label}
                </th>
              ) : (
                <th key={i}>
                  {c.param_label} · {c.latest_label}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              <td className="bench">{row.domain_title}</td>
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
      <td className="score">{pct(cell.prev)}</td>
      <td className="score">{pct(cell.latest)}</td>
      <td className={`score delta ${deltaClass(cell.delta)}`}>{signedPct(cell.delta)}</td>
    </>
  );
}
