import { parseBrief } from "../format";
import { GRID, MUTED } from "../palette";
import type { ChatTurn } from "../types";

// Renders an assistant turn's text as real headings + bullet lists when
// it's shaped like a research brief ("Research brief — X" followed by
// "Label: item; item" lines) -- the same display-only reformatting
// gradio_app.py's _format_chat_response does for gr.Chatbot's Markdown
// renderer, done here as JSX instead. Anything that doesn't match that
// shape (a guardrail refusal, a clarification question, an error) passes
// through as plain text, completely unchanged.
export default function ChatMessage({ turn }: { turn: ChatTurn }) {
  const isUser = turn.role === "user";
  const brief = !isUser ? parseBrief(turn.content) : null;

  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", marginBottom: 10 }}>
      <div
        style={{
          maxWidth: "88%",
          background: isUser ? "#2a78d6" : "#ffffff",
          color: isUser ? "#ffffff" : "#0b0b0b",
          border: isUser ? "none" : `1px solid ${GRID}`,
          borderRadius: 12,
          padding: "10px 14px",
          fontSize: 14,
          lineHeight: 1.5,
          overflowWrap: "break-word",
        }}
      >
        {brief ? (
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 8 }}>
              Research Brief &mdash; {brief.company}
            </div>
            {brief.sections.map((sec, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>{sec.label}</div>
                {sec.items.length > 1 ? (
                  <ul style={{ margin: "4px 0 0 0", paddingLeft: 18 }}>
                    {sec.items.map((item, j) => (
                      <li key={j} style={{ marginBottom: 2 }}>
                        {item}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div>{sec.items[0] ?? ""}</div>
                )}
              </div>
            ))}
            {brief.trailing.map((line, i) => (
              <div key={i} style={{ color: MUTED, fontSize: 12 }}>
                {line}
              </div>
            ))}
          </div>
        ) : (
          <span style={{ whiteSpace: "pre-wrap" }}>{turn.content}</span>
        )}
      </div>
    </div>
  );
}
