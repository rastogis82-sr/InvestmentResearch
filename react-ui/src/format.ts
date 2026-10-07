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

// Small set of acronyms/abbreviations title-casing mangles (e.g. "Ceo"
// instead of "CEO") -- applied after title-casing. Kept identical to
// gradio_app.py's _KEY_LABEL_FIXUPS so both UIs render the same labels
// for fields contributed by yahoo_finance_market_data.py (pe_ratio, hq).
// "Ceo" is kept in this map even though no current provider returns a
// ceo field -- harmless if unused, and free to reactivate if a future
// provider adds one back.
const KEY_LABEL_FIXUPS: Record<string, string> = {
  Yoy: "YoY",
  Ceo: "CEO",
  "Pe Ratio": "P/E Ratio",
  Hq: "HQ",
};

export function humanizeKey(key: string): string {
  let base = key;
  // Longer/more specific suffixes checked before shorter ones, same
  // ordering discipline as gradio_app.py's _humanize_key.
  for (const suffix of ["_usd_b", "_usd_m", "_usd", "_pct"]) {
    if (base.endsWith(suffix)) {
      base = base.slice(0, -suffix.length);
      break;
    }
  }
  let label = titleCase(base.replace(/_/g, " ").trim());
  for (const [wrong, right] of Object.entries(KEY_LABEL_FIXUPS)) {
    label = label.split(wrong).join(right);
  }
  return label || key;
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
    // market_cap_usd_b (from Yahoo Finance, see yahoo_finance_market_data.py)
    // is already in BILLIONS -- auto-scale to trillions above 1000, same as
    // gradio_app.py's _format_value, rather than printing an unwieldy
    // "$3,806.3B" for a company the size of Apple.
    if (keyL.endsWith("_usd_b")) {
      if (Math.abs(value) >= 1000) {
        return `$${(value / 1000).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}T`;
      }
      return `$${value.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}B`;
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
