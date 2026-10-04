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

import json
import os
import uuid

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()  # no-op if no .env file exists; never overrides a real env var

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


def _on_submit(message: str, history: list, session_id: str):
    result = _call_chat(session_id, message)
    if result.get("ok"):
        response_text = result.get("response", "")
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
    transparency = {
        "flags": result.get("flags", debug_state.get("flags", [])),
        "latency_ms": result.get("latency_ms"),
        "resolved_company": debug_state.get("resolved_company"),
        "tool_results": debug_state.get("tool_results", {}),
        "retrieved_docs": [d.get("source") for d in debug_state.get("retrieved_docs", [])],
    }
    return history, "", json.dumps(transparency, indent=2, default=str), session_id


def _on_new_session():
    return [], "", "", _new_session_id()


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
            transparency_box = gr.Code(
                label="flags / resolved_company / tool_results / retrieved_docs",
                language="json",
            )

    demo.load(_on_load, outputs=[session_state])

    msg.submit(
        _on_submit,
        [msg, chatbot, session_state],
        [chatbot, msg, transparency_box, session_state],
    )
    new_session_btn.click(
        _on_new_session,
        outputs=[chatbot, msg, transparency_box, session_state],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=GRADIO_SHARE)
