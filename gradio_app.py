"""
Investment Research Assistant — Gradio frontend
=================================================

A standalone chat UI that sits in front of the *deployed* FastAPI service
(`investment_research_api.py`) over plain HTTP, rather than driving the
LangGraph agent in-process the way the notebook's Section 11 demo does.

Same two-column layout as that notebook demo (chat on the left,
transparency panel on the right showing flags / resolved company / tool
results / retrieved docs) — but this one talks to a live `/chat` and
`/debug/{session_id}` endpoint, so it works against any environment
that's serving the API, including the deployed Render instance, and can
itself be deployed as its own small service (Render, Hugging Face
Spaces, or just run locally with `python gradio_app.py`).

All configuration is read from environment variables — nothing is
hardcoded here, in `render.yaml`, or anywhere else. See
`.env.gradio.example` for the template.

Environment variables:
    API_BASE_URL          Required. Base URL of the deployed FastAPI
                           service, e.g.
                           https://investment-research-api-xxxx.onrender.com
                           (no trailing slash). Until this is set, the UI
                           loads but every request returns a clear error
                           instead of crashing.
    API_TIMEOUT_SECONDS    Optional, default 60. HTTP timeout per request
                           — generous by default to tolerate a Render
                           free-tier cold start (30-60s) on the first
                           request after idle.
    GRADIO_SHARE           Optional, default "false". Set "true" to also
                           get a temporary public gradio.live tunnel —
                           handy for a quick local demo; not needed when
                           this app is itself deployed with a stable URL.
    PORT                   Optional, default 7860. Render injects its own
                           PORT; Gradio must bind to it and to 0.0.0.0,
                           not localhost, to be reachable.
"""

import html
import json
import os
import re
import uuid

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # headless -- no display available on a server
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

load_dotenv()  # no-op if no .env file exists; never overrides a real env var

# --- Color palette for the transparency panel -------------------------
# Status colors (fixed, never themed) and chart chrome from the project's
# standard data-viz palette. Every status badge pairs an icon with the
# label text, never color alone, per that palette's accessibility rule.
_STATUS_STYLE = {
    "critical": {"bg": "#fbe2e2", "border": "#d03b3b", "fg": "#7a1f1f", "icon": "✕"},
    "serious":  {"bg": "#fbe9e2", "border": "#ec835a", "fg": "#7a3b1f", "icon": "⚠"},
    "warning":  {"bg": "#fdf1d6", "border": "#fab219", "fg": "#6b4e05", "icon": "⚠"},
    "good":     {"bg": "#e3f6e3", "border": "#0ca30c", "fg": "#0a4d0a", "icon": "✓"},
    "info":     {"bg": "#e8eef8", "border": "#2a78d6", "fg": "#1c4a80", "icon": "ℹ"},
}
_SURFACE, _GRID, _MUTED, _INK = "#fcfcfb", "#e1e0d9", "#898781", "#0b0b0b"
_SEQ_BLUE, _DIVERGE_RED = "#2a78d6", "#e34948"  # the palette's diverging pair

API_BASE_URL = os.environ.get("API_BASE_URL", "").strip().rstrip("/")
API_TIMEOUT_SECONDS = float(os.environ.get("API_TIMEOUT_SECONDS", "60"))
GRADIO_SHARE = os.environ.get("GRADIO_SHARE", "false").strip().lower() == "true"
PORT = int(os.environ.get("PORT", "7860"))

if not API_BASE_URL:
    print(
        "WARNING: API_BASE_URL is not set. Set it to your deployed "
        "investment_research_api.py URL (e.g. "
        "https://investment-research-api-xxxx.onrender.com) before using "
        "the chat UI -- every request will fail with a clear in-UI error "
        "until then."
    )


def _call_chat(session_id: str, message: str) -> dict:
    """POSTs to the deployed API's /chat endpoint. Never raises -- on any
    failure (missing config, network error, timeout, non-2xx), returns an
    {"ok": False, "error": ...} dict so the UI can show a clean message
    instead of crashing."""
    if not API_BASE_URL:
        return {"ok": False, "error": "API_BASE_URL is not configured on this deployment."}
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={"session_id": session_id, "message": message},
            timeout=API_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return {"ok": True, **resp.json()}
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "error": (
                "Request timed out. If the service has been idle, Render's "
                "free tier can take 30-60s to wake up -- try again."
            ),
        }
    except requests.exceptions.RequestException as exc:
        return {"ok": False, "error": f"Request to the API failed: {exc}"}


def _call_debug(session_id: str) -> dict:
    """GETs the deployed API's /debug/{session_id} endpoint for the
    transparency panel. Returns {} on any failure -- the panel just stays
    empty/stale rather than breaking the chat."""
    if not API_BASE_URL:
        return {}
    try:
        resp = requests.get(f"{API_BASE_URL}/debug/{session_id}", timeout=API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return {}


def _new_session_id() -> str:
    return str(uuid.uuid4())


# A research brief's first line is always "Research brief — <COMPANY>" (see
# node_logic.py / investment_research_api.py's synthesize_brief_node) and
# every section after it is "<Label>: item; item; item". Reformatting this
# into Markdown is display-only: it happens here in the Gradio layer, never
# touches the API's actual response string, so the test harness, the
# notebook, and the raw /chat JSON all keep seeing the original plain text.
_BRIEF_TITLE_RE = re.compile(r"^Research brief\s*[—-]\s*(.+)$")
_BRIEF_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z /()]{2,40}):\s(.*)$")


def _format_chat_response(text: str) -> str:
    """Turns the brief's labeled, semicolon-joined lines into Markdown
    (a heading plus a bulleted list per section) for gr.Chatbot, which
    renders Markdown natively. Any response that isn't shaped like a brief
    -- a guardrail's plain refusal, clarification request, or error message
    -- doesn't match the title line and passes through completely
    unchanged."""
    if not text:
        return text

    lines = text.split("\n")
    title_match = _BRIEF_TITLE_RE.match(lines[0].strip())
    if not title_match:
        return text

    out = [f"### Research Brief — {title_match.group(1).strip()}"]
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        sec_match = _BRIEF_SECTION_RE.match(line)
        if not sec_match:
            out.append(line)
            continue
        label, body = sec_match.group(1).strip(), sec_match.group(2).strip()
        items = [i.strip() for i in body.split("; ") if i.strip()]
        if len(items) > 1:
            # A blank line MUST separate the header paragraph from the list
            # that follows -- without it, Markdown treats "- item" lines as
            # a continuation of the same paragraph (literal hyphens in
            # running text) rather than as list items.
            out.append(f"\n**{label}**\n")
            out.extend(f"- {i}" for i in items)
        else:
            out.append(f"\n**{label}**\n{body}")
    return "\n".join(out)


def _flag_status(flag: str) -> str:
    """Maps a flag string's prefix to a status severity. Unknown flags
    default to 'warning' -- better to over-flag than silently drop a
    guardrail hit the UI doesn't recognize yet."""
    if flag.startswith("injection"):
        return "critical"
    if flag.startswith("tool_failure"):
        return "serious"
    if flag.startswith("rag_failure") or flag.startswith("conflicting_data"):
        return "warning"
    return "warning"


def _badge(text: str, status: str) -> str:
    s = _STATUS_STYLE[status]
    # white-space:normal + overflow-wrap (not nowrap) is deliberate: a flag
    # or error string of unknown length must never force the panel to
    # scroll horizontally. Worst case a long label wraps to a second line
    # inside the pill -- still bounded, never overflowing.
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;max-width:100%;'
        f'background:{s["bg"]};border:1px solid {s["border"]};color:{s["fg"]};'
        f'border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600;'
        f'margin:2px 4px 2px 0;white-space:normal;overflow-wrap:break-word;">'
        f'{s["icon"]} {html.escape(text)}</span>'
    )


def _humanize_flag(flag: str) -> str:
    """Turns a raw flag string (e.g. 'tool_failure:get_company_financials')
    into a short human label for the badge -- the raw flag can be long
    (an injection flag embeds the matched text via repr()) and a pill
    badge should never carry a full sentence."""
    if flag.startswith("tool_failure:"):
        return f"Tool failed: {_humanize_tool(flag.split(':', 1)[1])}"
    if flag.startswith("rag_failure"):
        return "No matching document found"
    if flag.startswith("conflicting_data"):
        return "Conflicting data between sources"
    if flag.startswith("injection"):
        return "Prompt injection detected"
    return flag.replace("_", " ").replace(":", " — ")


# Small set of acronyms/abbreviations that Python's str.title() mangles
# (e.g. "Ceo" instead of "CEO") -- applied after title-casing, not instead
# of it, so "pe_ratio" still goes through the normal " "-join + title()
# path and only the final token gets corrected.
_KEY_LABEL_FIXUPS = {"Yoy": "YoY", "Ceo": "CEO", "Pe Ratio": "P/E Ratio", "Hq": "HQ"}


def _humanize_key(key: str) -> str:
    base = key
    # Longer/more specific suffixes must be checked before shorter ones
    # that could also match (e.g. "market_cap_usd_b" would otherwise never
    # reach "_usd_b" if "_usd" were checked -- and matched -- first; in
    # practice it isn't, since "_usd_b" doesn't end with "_usd", but the
    # ordering is kept deliberate rather than relying on that).
    for suffix in ("_usd_b", "_usd_m", "_usd", "_pct"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    label = base.replace("_", " ").strip().title()
    for wrong, right in _KEY_LABEL_FIXUPS.items():
        label = label.replace(wrong, right)
    return label or key


def _humanize_tool(name: str) -> str:
    label = name[4:] if name.startswith("get_") else name
    return label.replace("_", " ").strip().title() or name


def _format_value(key: str, value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        key_l = key.lower()
        if key_l.endswith("_pct"):
            # Always 2 decimal places -- a real TwelveData percent_change
            # (e.g. 0.922395) must not render as "+0.922395%".
            return f"{value:+.2f}%"
        # market_cap_usd_b (from FCS API, see fcs_market_data.py) is
        # already in BILLIONS -- auto-scale to trillions above 1000 rather
        # than printing an unwieldy "$3,806.3B" for a company like Apple.
        if key_l.endswith("_usd_b"):
            if abs(value) >= 1000:
                return f"${value / 1000:,.2f}T"
            return f"${value:,.1f}B"
        # "usd" covers revenue_usd_m/last_price_usd; the _low/_high suffix
        # covers fifty_two_week_low/fifty_two_week_high, which are also
        # dollar prices but don't carry "usd" in the key name.
        if "usd" in key_l or key_l.endswith("_low") or key_l.endswith("_high"):
            if key_l.endswith("_usd_m"):
                return f"${value:,.1f}M"
            return f"${value:,.2f}"
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


def _build_summary_html(result: dict, debug_state: dict) -> str:
    """Renders the transparency panel as styled HTML: colored flag badges,
    a tool-call card per tool with its fields formatted, and retrieved-doc
    chips -- instead of a raw JSON dump. Never raises: any missing/odd
    field just gets skipped rather than crashing the UI."""
    flags = result.get("flags") or debug_state.get("flags") or []
    resolved_company = debug_state.get("resolved_company")
    latency_ms = result.get("latency_ms")
    tool_results = debug_state.get("tool_results") or {}
    retrieved_docs = debug_state.get("retrieved_docs") or []

    parts = [
        '<div style="font-family:system-ui,-apple-system,\'Segoe UI\',sans-serif;'
        f'background:{_SURFACE};border:1px solid {_GRID};border-radius:10px;'
        f'padding:14px 16px;color:{_INK};overflow-wrap:break-word;word-break:break-word;">'
    ]

    header_bits = []
    if resolved_company:
        header_bits.append(
            '<span style="display:inline-block;background:#e8eef8;border:1px solid #2a78d6;'
            f'color:#1c4a80;border-radius:6px;padding:2px 8px;font-weight:700;font-size:13px;">'
            f'{html.escape(str(resolved_company))}</span>'
        )
    if latency_ms is not None:
        header_bits.append(
            f'<span style="color:{_MUTED};font-size:12px;">{latency_ms / 1000:.1f}s response time</span>'
        )
    if header_bits:
        parts.append(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
            + "".join(header_bits)
            + "</div>"
        )

    parts.append(
        f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.04em;'
        f'color:{_MUTED};margin-bottom:4px;">Guardrail flags</div>'
    )
    parts.append('<div style="margin-bottom:12px;">')
    if flags:
        parts.append("".join(_badge(_humanize_flag(f), _flag_status(f)) for f in flags))
    else:
        parts.append(_badge("No issues flagged", "good"))
    parts.append("</div>")

    if tool_results:
        parts.append(
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.04em;'
            f'color:{_MUTED};margin-bottom:4px;">Tool calls</div>'
        )
        for tool_name, tr in tool_results.items():
            ok = bool(tr.get("ok"))
            # The status badge is always a short fixed label ("ok"/"failed")
            # -- the actual error can be a full sentence (e.g. TwelveData's
            # "plan_restricted:/profile is available exclusively with
            # growth plans and above"), which must never go inside a pill;
            # it's rendered as its own wrapped line below instead.
            status_badge = _badge("ok", "good") if ok else _badge("failed", "serious")
            parts.append(
                f'<div style="border:1px solid {_GRID};border-radius:8px;padding:8px 10px;margin-bottom:6px;">'
                '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">'
                f'<span style="font-weight:600;font-size:13px;">{html.escape(_humanize_tool(tool_name))}</span>'
                f"{status_badge}</div>"
            )
            if not ok:
                error_text = str(tr.get("error", "Unknown error"))
                parts.append(
                    f'<div style="font-size:12px;color:#7a3b1f;margin-top:6px;">{html.escape(error_text)}</div>'
                )
            data = tr.get("data") if ok else None
            if isinstance(data, dict) and data:
                rows = []
                for k, v in data.items():
                    if k in ("source", "note"):
                        continue
                    rows.append(
                        '<div style="display:flex;justify-content:space-between;font-size:12px;'
                        f'padding:2px 0;color:{_MUTED};">'
                        f"<span>{html.escape(_humanize_key(k))}</span>"
                        f'<span style="font-variant-numeric:tabular-nums;color:{_INK};font-weight:500;">'
                        f"{html.escape(_format_value(k, v))}</span></div>"
                    )
                if rows:
                    parts.append(f'<div style="margin-top:6px;">{"".join(rows)}</div>')
                note = data.get("note")
                if note:
                    parts.append(
                        f'<div style="font-size:11px;color:{_MUTED};margin-top:4px;font-style:italic;">'
                        f"{html.escape(str(note))}</div>"
                    )
            parts.append("</div>")

    if retrieved_docs:
        parts.append(
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.04em;'
            f'color:{_MUTED};margin:10px 0 4px;">Retrieved sources</div><div>'
        )
        for d in retrieved_docs:
            src = d.get("source") if isinstance(d, dict) else str(d)
            parts.append(
                '<span style="display:inline-block;background:#f0efec;border:1px solid #c3c2b7;'
                f'color:{_MUTED};border-radius:6px;padding:2px 8px;font-size:11px;margin:2px 4px 2px 0;">'
                f"{html.escape(str(src))}</span>"
            )
        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


def _build_metrics_plot(debug_state: dict):
    """Builds a horizontal bar chart of every *_pct field returned by any
    successful tool call this turn (revenue growth, margins, day change,
    etc. -- works unchanged for synthetic ABC/XYZ/DEF and real-company
    data since both share the same _pct-suffixed field convention). Blue
    for positive, red for negative -- the palette's diverging pair, which
    fits naturally since these are gain/loss values straddling zero.
    Returns None (clearing the plot) when there's nothing numeric to show,
    e.g. a RAG-failure or tool-failure turn."""
    tool_results = debug_state.get("tool_results") or {}
    metrics = {}
    for tr in tool_results.values():
        if not tr.get("ok"):
            continue
        data = tr.get("data")
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k.lower().endswith("_pct"):
                metrics.setdefault(_humanize_key(k), float(v))

    if not metrics:
        return None

    labels = list(metrics.keys())[:8]
    values = [metrics[l] for l in labels]

    fig, ax = plt.subplots(figsize=(5, max(2.2, 0.45 * len(labels) + 0.8)))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=[_SEQ_BLUE if v >= 0 else _DIVERGE_RED for v in values], height=0.55, zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9, color=_INK)
    ax.invert_yaxis()
    ax.axvline(0, color=_MUTED, linewidth=0.8, zorder=2)
    ax.set_xlabel("Percent (%)", fontsize=9, color=_MUTED)
    ax.tick_params(axis="x", labelsize=8, colors=_MUTED)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.grid(axis="x", color=_GRID, linewidth=0.7, zorder=1)
    ax.set_title("Key percentage metrics", fontsize=10, color=_INK, loc="left", fontweight="bold")

    # Pad the axis so value labels never clip off the edge, including the
    # single-bar case (e.g. only a "day change" value) where the data span
    # alone would be too small to leave room for the label.
    vmax, vmin = max(values), min(values)
    span = max(vmax - vmin, 1.0)
    offset, pad = span * 0.04, span * 0.18
    ax.set_xlim(min(0, vmin) - pad, max(0, vmax) + pad)
    for i, v in enumerate(values):
        ax.text(
            v + (offset if v >= 0 else -offset), i, f"{v:+.1f}%",
            va="center", ha="left" if v >= 0 else "right", fontsize=8, color=_INK,
        )
    fig.tight_layout()
    return fig


def _on_submit(message: str, history: list, session_id: str):
    result = _call_chat(session_id, message)
    if result.get("ok"):
        response_text = _format_chat_response(result.get("response", ""))
    else:
        response_text = f"⚠️ {result.get('error', 'Unknown error.')}"

    # gr.Chatbot expects a flat list of {"role", "content"} dicts
    # (OpenAI-style), not the old [[user, bot], ...] tuple-pairs format.
    # In gradio==6.29.1 (pinned in requirements-gradio.txt) this is the
    # *only* supported format -- Chatbot.__init__ doesn't even accept a
    # `type=` kwarg anymore (that only existed in versions that still
    # supported both formats), so there's nothing to select explicitly.
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_text},
    ]

    debug_state = _call_debug(session_id)
    summary_html = _build_summary_html(result, debug_state)
    metrics_plot = _build_metrics_plot(debug_state)

    # Raw JSON stays available too (collapsed, under "Raw debug JSON") for
    # anyone who wants the exact payload rather than the formatted summary.
    transparency = {
        "flags": result.get("flags", debug_state.get("flags", [])),
        "latency_ms": result.get("latency_ms"),
        "resolved_company": debug_state.get("resolved_company"),
        "tool_results": debug_state.get("tool_results", {}),
        "retrieved_docs": [d.get("source") for d in debug_state.get("retrieved_docs", [])],
    }
    raw_json = json.dumps(transparency, indent=2, default=str)
    return history, "", summary_html, metrics_plot, raw_json, session_id


def _on_new_session():
    return [], "", "", None, "", _new_session_id()


def _on_load():
    # Runs once per browser page load, so each visitor gets their own
    # session_id (and hence their own LangGraph checkpointer thread on the
    # API side) rather than sharing one across every concurrent user of
    # this deployment.
    return _new_session_id()


with gr.Blocks(title="Investment Research Assistant") as demo:
    gr.Markdown("# Investment Research Assistant")
    gr.Markdown(f"Talking to: `{API_BASE_URL or '(API_BASE_URL not set -- see server logs)'}`")

    session_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=450)
            msg = gr.Textbox(
                label="Your question",
                placeholder="Research ABC Technologies: revenue growth and profitability.",
            )
            new_session_btn = gr.Button("New session")
        with gr.Column(scale=1):
            gr.Markdown("### Transparency panel")
            summary_html = gr.HTML()
            metrics_plot = gr.Plot(label=None, show_label=False)
            with gr.Accordion("Raw debug JSON", open=False):
                transparency_box = gr.Code(
                    label="flags / resolved_company / tool_results / retrieved_docs",
                    language="json",
                )

    demo.load(_on_load, outputs=[session_state])

    msg.submit(
        _on_submit,
        [msg, chatbot, session_state],
        [chatbot, msg, summary_html, metrics_plot, transparency_box, session_state],
    )
    new_session_btn.click(
        _on_new_session,
        outputs=[chatbot, msg, summary_html, metrics_plot, transparency_box, session_state],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=GRADIO_SHARE)
