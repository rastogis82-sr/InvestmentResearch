import type { ChatResult, DebugState } from "./types";

// Baked in at `npm run build` time by Vite -- see .env.example and
// RENDER_REACT_DEPLOY.md. Trimmed and stripped of a trailing slash the
// same way gradio_app.py normalizes API_BASE_URL.
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").trim().replace(/\/+$/, "");

const API_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS ?? 60_000);

async function fetchWithTimeout(url: string, init: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// POSTs to the deployed API's /chat endpoint. Never throws -- on any
// failure (missing config, network error, timeout, non-2xx) it returns
// an {ok: false, error} result so the UI can show a clean message
// instead of crashing, same contract as gradio_app.py's _call_chat.
export async function callChat(sessionId: string, message: string): Promise<ChatResult> {
  if (!API_BASE_URL) {
    return { ok: false, error: "VITE_API_BASE_URL is not configured on this deployment." };
  }
  try {
    const resp = await fetchWithTimeout(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
    });
    if (!resp.ok) {
      return { ok: false, error: `The API returned an error (HTTP ${resp.status}).` };
    }
    const data = await resp.json();
    return { ok: true, ...data };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      return {
        ok: false,
        error:
          "Request timed out. If the service has been idle, Render's free tier can take 30-60s to wake up -- try again.",
      };
    }
    return { ok: false, error: `Request to the API failed: ${String(err)}` };
  }
}

// GETs the deployed API's /debug/{session_id} endpoint for the
// transparency panel. Returns {} on any failure, same contract as
// gradio_app.py's _call_debug.
export async function callDebug(sessionId: string): Promise<DebugState> {
  if (!API_BASE_URL) return {};
  try {
    const resp = await fetchWithTimeout(`${API_BASE_URL}/debug/${sessionId}`, { method: "GET" });
    if (!resp.ok) return {};
    return (await resp.json()) as DebugState;
  } catch {
    return {};
  }
}
