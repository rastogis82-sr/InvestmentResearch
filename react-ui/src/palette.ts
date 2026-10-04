// The project's validated data-viz palette (see architecture.md /
// dataviz skill references/palette.md), kept identical to the hex
// values already shipped and screenshot-verified in gradio_app.py, so
// the Gradio UI and this React UI read as the same product.
import type { Status } from "./types";

export const STATUS_STYLE: Record<
  Status,
  { bg: string; border: string; fg: string; icon: string }
> = {
  critical: { bg: "#fbe2e2", border: "#d03b3b", fg: "#7a1f1f", icon: "✕" },
  serious: { bg: "#fbe9e2", border: "#ec835a", fg: "#7a3b1f", icon: "⚠" },
  warning: { bg: "#fdf1d6", border: "#fab219", fg: "#6b4e05", icon: "⚠" },
  good: { bg: "#e3f6e3", border: "#0ca30c", fg: "#0a4d0a", icon: "✓" },
  info: { bg: "#e8eef8", border: "#2a78d6", fg: "#1c4a80", icon: "ℹ" },
};

export const SURFACE = "#fcfcfb";
export const GRID = "#e1e0d9";
export const MUTED = "#898781";
export const INK = "#0b0b0b";
export const SEQ_BLUE = "#2a78d6";
export const DIVERGE_RED = "#e34948";
