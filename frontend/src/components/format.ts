export function pct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(1)}%`;
}

export function signedPct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const s = value >= 0 ? "+" : "";
  return `${s}${value.toFixed(1)}`;
}

export function deltaClass(value: number | null | undefined): string {
  if (value === null || value === undefined) return "flat";
  if (value > 0.05) return "up";
  if (value < -0.05) return "down";
  return "flat";
}
