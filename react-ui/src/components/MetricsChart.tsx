import { humanizeKey } from "../format";
import { SEQ_BLUE, DIVERGE_RED, GRID, MUTED, INK, SURFACE } from "../palette";
import type { DebugState } from "../types";

// An SVG port of gradio_app.py's _build_metrics_plot (matplotlib isn't
// available in a browser bundle, so this redraws the same chart --
// horizontal bars, blue for gains / red for losses, same diverging pair
// -- directly in SVG). The padding math is carried over verbatim,
// including the single-bar case fix (a lone "day change" value whose
// own span is too small to leave room for its label without this).
export default function MetricsChart({ debugState }: { debugState: DebugState }) {
  const metrics = new Map<string, number>();
  for (const tr of Object.values(debugState.tool_results ?? {})) {
    if (!tr.ok || !tr.data) continue;
    for (const [k, v] of Object.entries(tr.data)) {
      if (typeof v === "number" && k.toLowerCase().endsWith("_pct")) {
        const label = humanizeKey(k);
        if (!metrics.has(label)) metrics.set(label, v);
      }
    }
  }
  if (metrics.size === 0) return null;

  const entries = Array.from(metrics.entries()).slice(0, 8);
  const labels = entries.map(([l]) => l);
  const values = entries.map(([, v]) => v);

  const vmax = Math.max(...values);
  const vmin = Math.min(...values);
  const span = Math.max(vmax - vmin, 1.0);
  const offset = span * 0.04;
  const pad = span * 0.18;
  const domainMin = Math.min(0, vmin) - pad;
  const domainMax = Math.max(0, vmax) + pad;
  const domainSpan = domainMax - domainMin;

  const W = 440;
  const labelColW = 108;
  const plotW = W - labelColW - 56; // right margin for value labels
  const rowH = 30;
  const topPad = 36;
  const H = topPad + rowH * labels.length + 28;

  const xScale = (v: number) => ((v - domainMin) / domainSpan) * plotW;
  const zeroX = xScale(0);

  return (
    <div style={{ marginTop: 10 }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} style={{ background: SURFACE }}>
        <text x={labelColW} y={18} fontSize={11} fontWeight={700} fill={INK}>
          Key percentage metrics
        </text>
        {/* gridlines + zero line */}
        <line
          x1={labelColW}
          x2={labelColW + plotW}
          y1={topPad - 10}
          y2={topPad - 10}
          stroke={GRID}
        />
        {labels.map((label, i) => {
          const v = values[i];
          const y = topPad + i * rowH;
          const barX = xScale(v);
          const barLeft = Math.min(zeroX, barX);
          const barWidth = Math.abs(barX - zeroX);
          const color = v >= 0 ? SEQ_BLUE : DIVERGE_RED;
          const labelX = labelColW + barX + (v >= 0 ? offset : -offset) * (plotW / domainSpan);
          return (
            <g key={label}>
              <text x={labelColW - 8} y={y + 14} fontSize={10} fill={INK} textAnchor="end">
                {label}
              </text>
              <rect
                x={labelColW + barLeft}
                y={y}
                width={Math.max(barWidth, 1)}
                height={14}
                fill={color}
                rx={2}
              />
              <text
                x={labelX}
                y={y + 11}
                fontSize={9.5}
                fill={INK}
                textAnchor={v >= 0 ? "start" : "end"}
              >
                {`${v >= 0 ? "+" : ""}${v.toFixed(1)}%`}
              </text>
            </g>
          );
        })}
        <line
          x1={labelColW + zeroX}
          x2={labelColW + zeroX}
          y1={topPad - 10}
          y2={topPad + rowH * labels.length - 6}
          stroke={MUTED}
          strokeWidth={0.8}
        />
        <text x={labelColW + plotW} y={topPad + rowH * labels.length + 14} fontSize={9} fill={MUTED} textAnchor="end">
          Percent (%)
        </text>
      </svg>
    </div>
  );
}
