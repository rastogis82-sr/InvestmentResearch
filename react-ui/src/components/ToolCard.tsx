import Badge from "./Badge";
import { formatValue, humanizeKey, humanizeTool } from "../format";
import { GRID, MUTED, INK } from "../palette";
import type { ToolResult } from "../types";

// One card per tool call. The status badge is always a short fixed
// label ("ok"/"failed") -- the real error text (which can be a full
// sentence, e.g. TwelveData's "plan_restricted:/profile is available
// exclusively with growth plans and above") is rendered as its own
// wrapped line below the header, never inside the pill. This mirrors
// the exact fix applied to gradio_app.py's _build_summary_html after
// the same overflow bug showed up there.
export default function ToolCard({ name, result }: { name: string; result: ToolResult }) {
  const ok = Boolean(result.ok);
  const data = ok ? result.data : undefined;
  const rows = data
    ? Object.entries(data).filter(([k]) => k !== "source" && k !== "note")
    : [];
  const note = data && typeof data.note === "string" ? data.note : undefined;

  return (
    <div
      style={{
        border: `1px solid ${GRID}`,
        borderRadius: 8,
        padding: "8px 10px",
        marginBottom: 6,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>{humanizeTool(name)}</span>
        <Badge text={ok ? "ok" : "failed"} status={ok ? "good" : "serious"} />
      </div>
      {!ok && (
        <div style={{ fontSize: 12, color: "#7a3b1f", marginTop: 6, overflowWrap: "break-word" }}>
          {String(result.error ?? "Unknown error")}
        </div>
      )}
      {rows.length > 0 && (
        <div style={{ marginTop: 6 }}>
          {rows.map(([k, v]) => (
            <div
              key={k}
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 12,
                padding: "2px 0",
                color: MUTED,
                gap: 12,
              }}
            >
              <span>{humanizeKey(k)}</span>
              <span style={{ fontVariantNumeric: "tabular-nums", color: INK, fontWeight: 500 }}>
                {formatValue(k, v)}
              </span>
            </div>
          ))}
        </div>
      )}
      {note && (
        <div style={{ fontSize: 11, color: MUTED, marginTop: 4, fontStyle: "italic" }}>{note}</div>
      )}
    </div>
  );
}
