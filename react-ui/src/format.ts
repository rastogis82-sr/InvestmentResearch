// Direct TypeScript ports of the same-named helpers in gradio_app.py.
// Keeping the logic identical (not just visually similar) is what keeps
// the two UIs in sync as the API's flag/tool/field vocabulary grows.
import type { Status } from "./types";

export function flagStatus(flag: string): Status {
  if (flag.startsWith("injection")) return "critical";
  if (flag.startsWith("tool_failure")) return "serious";
  if (flag.startsWith("rag_failure") || flag.startsWith("conflicting_data")) return "warning";
  return "warning";
}

export function humanizeFlag(flag: string): string {
  if (flag.startsWith("tool_failure:")) {
    return `Tool failed: ${humanizeTool(flag.split(":", 2)[1] ?? "")}`;
  }
  if (flag.startsWith("rag_failure")) return "No matching document found";
  if (flag.startsWith("conflicting_data")) return "Conflicting data between sources";
  if (flag.startsWith("injection")) return "Prompt injection detected";
  return flag.replace(/_/g, " ").replace(/:/g, " — ");
}

export function humanizeKey(key: string): string {
  let base = key;
  for (const suffix of ["_usd_m", "_usd", "_pct"]) {
    if (base.endsWith(suffix)) {
      base = base.slice(0, -suffix.length);
      break;
    }
  }
  const label = titleCase(base.replace(/_/g, " ").trim());
  const fixed = label.replace(/Yoy/g, "YoY");
  return fixed || key;
}

export function humanizeTool(name: string): string {
  const label = name.startsWith("get_") ? name.slice(4) : name;
  const titled = titleCase(label.replace(/_/g, " ").trim());
  return titled || name;
}

function titleCase(s: string): string {
  return s
    .split(" ")
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

export function formatValue(key: string, value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    const keyL = key.toLowerCase();
    if (keyL.endsWith("_pct")) {
      const sign = value >= 0 ? "+" : "";
      return `${sign}${value.toFixed(2)}%`;
    }
    if (keyL.includes("usd") || keyL.endsWith("_low") || keyL.endsWith("_high")) {
      if (keyL.endsWith("_usd_m")) {
        return `$${value.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}M`;
      }
      return `$${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    if (Number.isInteger(value)) {
      return value.toLocaleString("en-US");
    }
    return value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return String(value);
}

// --- Research-brief parsing -------------------------------------------
// A brief's first line is always "Research brief — <COMPANY>" and every
// section after it is "<Label>: item; item; item" (see node_logic.py /
// investment_research_api.py's synthesize_brief_node). This parses that
// shape into structured data so <ChatMessage> can render real headings
// and bullet lists instead of dumping one dense paragraph -- the same
// display-only reformatting gradio_app.py's _format_chat_response does,
// ported here so it renders as JSX instead of Markdown. Never touches
// the underlying API response string itself.
const BRIEF_TITLE_RE = /^Research brief\s*[—-]\s*(.+)$/;
const BRIEF_SECTION_RE = /^([A-Za-z][A-Za-z /()]{2,40}):\s(.*)$/;

export interface BriefSection {
  label: string;
  items: string[];
}

export interface ParsedBrief {
  company: string;
  sections: BriefSection[];
  trailing: string[]; // any line that didn't match the "Label: ..." shape
}

export function parseBrief(text: string): ParsedBrief | null {
  if (!text) return null;
  const lines = text.split("\n");
  const titleMatch = BRIEF_TITLE_RE.exec(lines[0].trim());
  if (!titleMatch) return null;

  const sections: BriefSection[] = [];
  const trailing: string[] = [];
  for (const raw of lines.slice(1)) {
    const line = raw.trim();
    if (!line) continue;
    const secMatch = BRIEF_SECTION_RE.exec(line);
    if (!secMatch) {
      trailing.push(line);
      continue;
    }
    const label = secMatch[1].trim();
    const body = secMatch[2].trim();
    const items = body
      .split("; ")
      .map((i) => i.trim())
      .filter(Boolean);
    sections.push({ label, items });
  }
  return { company: titleMatch[1].trim(), sections, trailing };
}
