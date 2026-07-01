import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartPayload } from "../api";

const PALETTE = ["#5b8cff", "#3ecf8e", "#f5b14c", "#f0637a", "#a78bfa", "#22d3ee", "#fb923c"];
const AXIS = { stroke: "#646b7e", fontSize: 11 };
const GRID = "#252834";

function pctTick(v: number) {
  return `${Math.round(v * 100)}%`;
}

function tooltipFmt(value: unknown): string {
  return typeof value === "number" ? pctTick(value) : String(value ?? "");
}

/** Pivot flat [{model, <dim>, score}] rows into recharts-friendly grouped rows. */
function pivot(data: { model: string; score: number }[], dimKey: string, models: string[]) {
  const byDim = new Map<string, Record<string, number | string>>();
  for (const d of data as Record<string, string | number>[]) {
    const dim = String(d[dimKey]);
    const row = byDim.get(dim) ?? { [dimKey]: dim };
    row[String(d.model)] = d.score as number;
    byDim.set(dim, row);
  }
  return { rows: [...byDim.values()], models };
}

function GroupedBar({
  data,
  dimKey,
  models,
  height,
}: {
  data: { model: string; score: number }[];
  dimKey: string;
  models: string[];
  height: number;
}) {
  const { rows } = pivot(data, dimKey, models);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey={dimKey} tick={AXIS} angle={-30} textAnchor="end" height={70} interval={0} />
        <YAxis tick={AXIS} tickFormatter={pctTick} domain={[0, 1]} />
        <Tooltip
          contentStyle={{ background: "#12141a", border: "1px solid #333748", borderRadius: 8, fontSize: 12 }}
          formatter={tooltipFmt}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {models.map((m, i) => (
          <Bar key={m} dataKey={m} fill={PALETTE[i % PALETTE.length]} radius={[3, 3, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DomainCharts({ chart }: { chart: ChartPayload[keyof ChartPayload] | null | undefined }) {
  if (!chart) return null;

  if ("subjects" in chart) {
    return <GroupedBar data={chart.data} dimKey="subject" models={chart.models} height={Math.max(360, chart.subjects.length * 26)} />;
  }
  if ("datasets" in chart) {
    return <GroupedBar data={chart.data} dimKey="dataset" models={chart.models} height={380} />;
  }
  if ("domains" in chart) {
    return <GroupedBar data={chart.data} dimKey="domain" models={chart.models} height={400} />;
  }
  if ("ks" in chart) {
    // AIME pass@k curves: merge series by k.
    const byK = new Map<number, Record<string, number>>();
    for (const s of chart.series) {
      for (const p of s.points) {
        const row = byK.get(p.k) ?? { k: p.k };
        row[s.name] = p.acc;
        byK.set(p.k, row);
      }
    }
    const rows = [...byK.values()].sort((a, b) => a.k - b.k);
    return (
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} />
          <XAxis dataKey="k" tick={AXIS} />
          <YAxis tick={AXIS} tickFormatter={pctTick} domain={[0, "auto"]} />
          <Tooltip
            contentStyle={{ background: "#12141a", border: "1px solid #333748", borderRadius: 8, fontSize: 12 }}
            formatter={tooltipFmt}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {chart.series.map((s, i) => (
            <Line key={s.name} dataKey={s.name} stroke={PALETTE[i % PALETTE.length]} dot strokeWidth={2} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  }
  return null;
}
