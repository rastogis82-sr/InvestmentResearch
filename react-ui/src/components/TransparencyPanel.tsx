import { useState } from "react";
import Badge from "./Badge";
import ToolCard from "./ToolCard";
import MetricsChart from "./MetricsChart";
import { flagStatus, humanizeFlag } from "../format";
import { GRID, MUTED, INK, SURFACE } from "../palette";
import type { ChatResult, DebugState } from "../types";

export default function TransparencyPanel({
  result,
  debugState,
}: {
  result: ChatResult | null;
  debugState: DebugState | null;
}) {
  const [showRaw, setShowRaw] = useState(false);

  if (!result) {
    return (
      <div
        style={{
          fontFamily: "system-ui, -apple-system, 'Segoe UI', sans-serif",
          background: SURFACE,
          border: `1px solid ${GRID}`,
          borderRadius: 10,
          padding: "14px 16px",
          color: MUTED,
          fontSize: 13,
        }}
      >
        Ask a question to see guardrail flags, tool calls, and retrieved sources here.
      </div>
    );
  }

  const flags = result.flags ?? debugState?.flags ?? [];
  const resolvedCompany = debugState?.resolved_company;
  const latencyMs = result.latency_ms;
  const toolResults = debugState?.tool_results ?? {};
  const retrievedDocs = debugState?.retrieved_docs ?? [];

  const transparency = {
    flags,
    latency_ms: latencyMs ?? null,
    resolved_company: resolvedCompany ?? null,
    tool_results: toolResults,
    retrieved_docs: retrievedDocs.map((d) => d.source),
  };

  return (
    <div
      style={{
        fontFamily: "system-ui, -apple-system, 'Segoe UI', sans-serif",
        background: SURFACE,
        border: `1px solid ${GRID}`,
        borderRadius: 10,
        padding: "14px 16px",
        color: INK,
        overflowWrap: "break-word",
        wordBreak: "break-word",
      }}
    >
      {(resolvedCompany || latencyMs !== undefined) && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
          {resolvedCompany && (
            <span
              style={{
                display: "inline-block",
                background: "#e8eef8",
                border: "1px solid #2a78d6",
                color: "#1c4a80",
                borderRadius: 6,
                padding: "2px 8px",
                fontWeight: 700,
                fontSize: 13,
              }}
            >
              {resolvedCompany}
            </span>
          )}
          {latencyMs !== undefined && latencyMs !== null && (
            <span style={{ color: MUTED, fontSize: 12 }}>{(latencyMs / 1000).toFixed(1)}s response time</span>
          )}
        </div>
      )}

      <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.04em", color: MUTED, marginBottom: 4 }}>
        Guardrail flags
      </div>
      <div style={{ marginBottom: 12 }}>
        {flags.length > 0 ? (
          flags.map((f, i) => <Badge key={`${f}-${i}`} text={humanizeFlag(f)} status={flagStatus(f)} />)
        ) : (
          <Badge text="No issues flagged" status="good" />
        )}
      </div>

      {Object.keys(toolResults).length > 0 && (
        <>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.04em", color: MUTED, marginBottom: 4 }}>
            Tool calls
          </div>
          {Object.entries(toolResults).map(([name, tr]) => (
            <ToolCard key={name} name={name} result={tr} />
          ))}
        </>
      )}

      <MetricsChart debugState={debugState ?? {}} />

      {retrievedDocs.length > 0 && (
        <>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.04em", color: MUTED, margin: "10px 0 4px" }}>
            Retrieved sources
          </div>
          <div>
            {retrievedDocs.map((d, i) => (
              <span
                key={i}
                style={{
                  display: "inline-block",
                  background: "#f0efec",
                  border: "1px solid #c3c2b7",
                  color: MUTED,
                  borderRadius: 6,
                  padding: "2px 8px",
                  fontSize: 11,
                  margin: "2px 4px 2px 0",
                }}
              >
                {d.source ?? String(d)}
              </span>
            ))}
          </div>
        </>
      )}

      <details style={{ marginTop: 12 }} open={showRaw} onToggle={(e) => setShowRaw((e.target as HTMLDetailsElement).open)}>
        <summary style={{ cursor: "pointer", fontSize: 12, color: MUTED }}>Raw debug JSON</summary>
        <pre
          style={{
            fontSize: 11,
            background: "#1a1a19",
            color: "#e1e0d9",
            borderRadius: 6,
            padding: 10,
            overflowX: "auto",
            marginTop: 6,
          }}
        >
          {JSON.stringify(transparency, null, 2)}
        </pre>
      </details>
    </div>
  );
}
