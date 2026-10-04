import { useRef, useState, type FormEvent } from "react";
import ChatMessage from "./ChatMessage";
import { GRID, MUTED, SURFACE } from "../palette";
import type { ChatTurn } from "../types";

export default function ChatPanel({
  history,
  busy,
  onSend,
  onNewSession,
}: {
  history: ChatTurn[];
  busy: boolean;
  onSend: (message: string) => void;
  onNewSession: () => void;
}) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  function submit(e: FormEvent) {
    e.preventDefault();
    const message = draft.trim();
    if (!message || busy) return;
    onSend(message);
    setDraft("");
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          minHeight: 420,
          maxHeight: 520,
          overflowY: "auto",
          background: SURFACE,
          border: `1px solid ${GRID}`,
          borderRadius: 10,
          padding: 14,
        }}
      >
        {history.length === 0 ? (
          <div style={{ color: MUTED, fontSize: 13 }}>
            Ask about a company &mdash; e.g. &ldquo;Research ABC Technologies: revenue growth and
            profitability.&rdquo;
          </div>
        ) : (
          history.map((turn, i) => <ChatMessage key={i} turn={turn} />)
        )}
        {busy && <div style={{ color: MUTED, fontSize: 13, fontStyle: "italic" }}>Thinking&hellip;</div>}
      </div>

      <form onSubmit={submit} style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Research ABC Technologies: revenue growth and profitability."
          disabled={busy}
          style={{
            flex: 1,
            padding: "10px 12px",
            borderRadius: 8,
            border: `1px solid ${GRID}`,
            fontSize: 14,
          }}
        />
        <button
          type="submit"
          disabled={busy || !draft.trim()}
          style={{
            padding: "10px 18px",
            borderRadius: 8,
            border: "none",
            background: "#2a78d6",
            color: "#fff",
            fontWeight: 600,
            fontSize: 14,
            cursor: busy ? "default" : "pointer",
            opacity: busy || !draft.trim() ? 0.6 : 1,
          }}
        >
          Send
        </button>
        <button
          type="button"
          onClick={onNewSession}
          disabled={busy}
          style={{
            padding: "10px 14px",
            borderRadius: 8,
            border: `1px solid ${GRID}`,
            background: "#fff",
            color: "#333",
            fontSize: 14,
            cursor: busy ? "default" : "pointer",
          }}
        >
          New session
        </button>
      </form>
    </div>
  );
}
