# Deploying the Gradio UI to Render.com

A second, independent Render Web Service that puts a chat UI in front of
the already-deployed API (`investment-research-api`, at
`https://investmentresearch.onrender.com`). This service talks to that
API over plain HTTP — it never touches LangGraph, the vector store, or
any LLM/API keys directly, so it's a much lighter deploy than the API
itself (no OpenAI/Anthropic/TwelveData keys needed at all).

All configuration is read from environment variables — nothing is
hardcoded in `gradio_app.py`, `render.yaml`, or anywhere else. Locally
you'd use a `.env` file (see `.env.gradio.example`); on Render you set
real environment variables in the dashboard instead.

## Files in this deployment bundle

| File | Purpose |
|---|---|
| `gradio_app.py` | The Gradio app — two-column chat + transparency panel, calling the deployed API's `/chat` and `/debug/{session_id}` endpoints over HTTP. The panel renders flags as colored status badges, formats each tool call as a labeled card, and charts any `*_pct` fields (revenue growth, margins, day change) as a horizontal bar chart — blue for gains, red for losses. The exact JSON payload is still available underneath, collapsed in a "Raw debug JSON" accordion. |
| `requirements-gradio.txt` | `gradio` (pinned, see below), `requests`, `python-dotenv`, `matplotlib` for the metrics chart — deliberately separate from `requirements.txt` so this service's build doesn't pull in LangChain/LangGraph/FAISS at all. |
| `.env.gradio.example` | Template for local testing — copy to `.env` and fill in `API_BASE_URL`. |

## Step 1 — Push these files to your GitHub repo

Same repo as the API is fine — Render builds each service from whatever
start command you give it, so extra files in the repo don't interfere.

```bash
git add gradio_app.py requirements-gradio.txt .env.gradio.example
git commit -m "Add Gradio frontend for the Investment Research Assistant"
git push
```

## Step 2 — Create a second Web Service on Render

- render.com → **New +** → **Web Service**
- Connect the **same GitHub repo**
- Fill in manually (Render's manual flow doesn't auto-read `render.yaml`,
  same as when you set up the API service):

| Field | Value |
|---|---|
| Name | `investment-research-gradio` (or anything you like) |
| Runtime | `Python` |
| Branch | `main` |
| Build Command | `pip install -r requirements-gradio.txt` |
| Start Command | `python gradio_app.py` |
| Health Check Path | `/` |

## Step 3 — Set environment variables

| Key | Required? | Value |
|---|---|---|
| `API_BASE_URL` | **Required** | `https://investmentresearch.onrender.com` (your deployed API's URL — no trailing slash) |
| `API_TIMEOUT_SECONDS` | Optional | `60` (default) — generous to tolerate the API's own free-tier cold start |
| `GRADIO_SHARE` | Optional | `false` (default) — leave off; this service already gets its own stable Render URL |

Nothing secret here. `PORT` is injected automatically by Render — don't
set it yourself.

## Step 4 — Deploy and verify

- Select the **Free** plan
- Click **Create Web Service** and watch the build logs (should be fast —
  three small packages, no torch/faiss/langchain)
- When you see `==> Your service is live`, open the URL in a browser —
  you should see the chat UI, with "Talking to: `https://investmentresearch.onrender.com`"
  printed under the title
- Send a test message (e.g. "Research ABC Technologies: revenue growth
  and profitability.") and confirm the transparency panel on the right
  fills in with flags / resolved company / tool results / retrieved docs

## Honest notes

- **Two free-tier services means two cold starts.** If both this UI and
  the API have been idle 15+ minutes, your first message after opening
  the UI may take up to a minute or two: the UI service itself has to
  wake up, then its first request wakes up the API. `API_TIMEOUT_SECONDS=60`
  is already generous for this; if you still see timeouts, send one
  `/health` warm-up request to the API first (or just retry once).
- **This UI shares nothing with the notebook's Section 11 Gradio demo**
  beyond layout — that one drives the LangGraph `app` in-process inside
  Colab; this one is a standalone deployment that only ever talks to the
  API over HTTP. You can run either, both, or neither independently.
- **No authentication** on this UI, same as the API's `/debug/{session_id}` —
  fine for a demo URL, not for anything public-facing with real users.
