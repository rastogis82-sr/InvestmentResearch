import { useEffect, useState } from "react";
import ChatPanel from "./components/ChatPanel";
import TransparencyPanel from "./components/TransparencyPanel";
import { API_BASE_URL, callChat, callDebug } from "./api";
import type { ChatResult, ChatTurn, DebugState } from "./types";

const SESSION_STORAGE_KEY = "ira_session_id";

function newSessionId(): string {
  return crypto.randomUUID();
}

export default function App() {
  // A session id persists per-browser-tab (sessionStorage, not
  // localStorage) so a page refresh keeps the same LangGraph
  // checkpointer thread on the API side, but a brand-new tab -- or the
  // "New session" button -- always starts fresh. Mirrors gradio_app.py's
  // _on_load, which mints one fresh id per page load.
  const [sessionId, setSessionId] = useState<string>(() => {
    const existing = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existing) return existing;
    const fresh = newSessionId();
    sessionStorage.setItem(SESSION_STORAGE_KEY, fresh);
    return fresh;
  });
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [busy, setBusy] = useState(false);
  const [lastResult, setLastResult] = useState<ChatResult | null>(null);
  const [debugState, setDebugState] = useState<DebugState | null>(null);

  useEffect(() => {
    sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }, [sessionId]);

  async function handleSend(message: string) {
    setBusy(true);
    setHistory((h) => [...h, { role: "user", content: message }]);
    const result = await callChat(sessionId, message);
    const responseText = result.ok ? result.response ?? "" : `⚠️ ${result.error ?? "Unknown error."}`;
    setHistory((h) => [...h, { role: "assistant", content: responseText }]);
    setLastResult(result);
    const debug = await callDebug(sessionId);
    setDebugState(debug);
    setBusy(false);
  }

  function handleNewSession() {
    const fresh = newSessionId();
    setSessionId(fresh);
    setHistory([]);
    setLastResult(null);
    setDebugState(null);
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Investment Research Assistant</h1>
        <p className="app-subtitle">
          Talking to: <code>{API_BASE_URL || "(VITE_API_BASE_URL not set -- see deploy docs)"}</code>
        </p>
      </header>

      <main className="app-grid">
        <section className="app-col app-col-chat">
          <ChatPanel history={history} busy={busy} onSend={handleSend} onNewSession={handleNewSession} />
        </section>
        <section className="app-col app-col-panel">
          <h2 className="panel-heading">Transparency panel</h2>
          <TransparencyPanel result={lastResult} debugState={debugState} />
        </section>
      </main>
    </div>
  );
}
