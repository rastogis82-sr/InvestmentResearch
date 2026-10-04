// Mirrors the JSON shapes returned by investment_research_api.py's
// /chat and /debug/{session_id} endpoints (see gradio_app.py's _call_chat
// / _call_debug for the Python-side equivalent of these two calls).

export interface ChatResult {
  ok: boolean;
  response?: string;
  flags?: string[];
  latency_ms?: number;
  error?: string;
}

export interface ToolResult {
  ok: boolean;
  tool?: string;
  error?: string;
  data?: Record<string, unknown>;
}

export interface RetrievedDoc {
  source?: string;
  score?: number;
}

export interface DebugState {
  resolved_company?: string;
  flags?: string[];
  tool_results?: Record<string, ToolResult>;
  retrieved_docs?: RetrievedDoc[];
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export type Status = "critical" | "serious" | "warning" | "good" | "info";
