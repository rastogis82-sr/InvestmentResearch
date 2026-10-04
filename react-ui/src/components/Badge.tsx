import { STATUS_STYLE } from "../palette";
import type { Status } from "../types";

// Icon + label pairing is required by the palette's accessibility rule --
// never color alone. overflow-wrap + normal white-space (not nowrap) is
// deliberate: a flag or error string of unknown length must never force
// a horizontal scrollbar -- this was a real bug fixed in the Gradio UI
// (a long TwelveData error overflowed a `nowrap` pill) and the fix is
// ported here from the start.
export default function Badge({ text, status }: { text: string; status: Status }) {
  const s = STATUS_STYLE[status];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        maxWidth: "100%",
        background: s.bg,
        border: `1px solid ${s.border}`,
        color: s.fg,
        borderRadius: 999,
        padding: "2px 10px",
        fontSize: 12,
        fontWeight: 600,
        margin: "2px 4px 2px 0",
        whiteSpace: "normal",
        overflowWrap: "break-word",
      }}
    >
      {s.icon} {text}
    </span>
  );
}
