# Deploying the Streamlit UI to Render.com

A third, independent Render Web Service — the dedicated guardrail-layer
control surface (`architecture.md` Section 9g). Unlike `gradio_app.py`
and `react-ui/` (both thin HTTP clients against the already-deployed API),
`streamlit_app.py` runs the LangGraph-equivalent pipeline and the
guardrail layer **in-process**, inside this service itself. That means
this is a heavier deploy than the Gradio one — it needs the same
LLM/RAG/guardrail stack as `investment_research_api.py`, not just a URL
to call.

All configuration is read from environment variables — nothing is
hardcoded in `streamlit_app.py`, `render.yaml`, or anywhere else. Locally
you'd use a `.env` file (see `.env.example`); on Render you set real
environment variables in the dashboard instead.

## Files in this deployment bundle

| File | Purpose |
|---|---|
| `streamlit_app.py` | The Streamlit app — Assistant / Review queue / Safety lab / Governance tabs. See its module docstring and `architecture.md` Section 9g for the full guardrail-to-tab mapping. |
| `requirements-streamlit.txt` | The same agent-side stack as `requirements.txt` (LangChain/LangGraph/FAISS/yfinance), minus `fastapi`/`uvicorn`, plus `streamlit` itself — this service runs the agent directly, so it needs everything the API needs. |
| `safety_layer.py`, `node_logic.py`, `graph_sim.py`, `mock_llm.py`, `local_retriever.py`, `corpus.py` | Shared local modules this app imports directly (no HTTP calls to the other services). |

## Step 1 — Push these files to your GitHub repo

Same repo as the API and Gradio UI is fine.

```bash
git add streamlit_app.py requirements-streamlit.txt
git commit -m "Add Streamlit guardrail-layer UI for the Investment Research Assistant"
git push
```

## Step 2 — Create a third Web Service on Render

- render.com → **New +** → **Web Service**
- Connect the **same GitHub repo**
- Fill in manually (Render's manual flow doesn't auto-read `render.yaml`,
  same as the API and Gradio services):

| Field | Value |
|---|---|
| Name | `investment-research-streamlit` (or anything you like) |
| Runtime | `Python` |
| Branch | `main` |
| Build Command | `pip install -r requirements-streamlit.txt` |
| Start Command | `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false` |
| Health Check Path | `/_stcore/health` |

`$PORT` is injected automatically by Render — don't hardcode a port
anywhere; `--server.address 0.0.0.0` is required so Render can route
external traffic in (Streamlit's default, `localhost`, is not reachable
from outside the container). `--server.headless true` skips the
first-run "would you like to share usage stats" prompt, which would
otherwise block startup on a non-interactive host. The two `--server.*`
CORS/XSRF flags are the standard fix for Streamlit's own cross-origin
checks tripping over Render's reverse proxy — if you ever see "Please
enter the CORS token" or the page loads but never connects, those two
flags are the first thing to check.

## Step 3 — Set environment variables

Same full set `investment_research_api.py` needs — this service runs the
agent itself, it doesn't call the API over HTTP:

| Key | Required? | Value |
|---|---|---|
| `MOCK_LLM` | Optional | `"true"` for a **free, zero-API-key** deploy — the offline heuristic engine is enough to demo every guardrail layer (PII masking, injection detection, the review queue, action dry-runs, the audit trail) without an LLM at all. Set `"false"` only if you want real LLM-written research briefs on this service too. |
| `LLM_PROVIDER` | Optional | e.g. `"openai:gpt-4o-mini"` — only used when `MOCK_LLM` is false. |
| `OPENAI_API_KEY` | Required if `MOCK_LLM=false` | RAG embeddings always call OpenAI's API regardless of `LLM_PROVIDER` — see `RENDER_DEPLOY.md`'s identical note for the API service. |
| `ANTHROPIC_API_KEY` | Optional | Only if `LLM_PROVIDER` starts with `anthropic:`. |
| `OPENAI_BASE_URL` | Optional | A Vocareum-style proxy URL, if applicable. |
| `IRA_SCORE_THRESHOLD` | Optional | `"0.35"` (default) — RAG similarity threshold, real embeddings only. |
| `SAFETY_AUDIT_DB_PATH` | Optional | `"data/safety_audit.db"` (default) — see below for why this service's audit trail is independent of the API's. |

**Recommended for a quick public demo of just the guardrails:** set only
`MOCK_LLM=true` and leave everything else unset. No OpenAI/Anthropic/
TwelveData key needed, build is fast, and every guardrail tab works
identically to the local `MOCK_LLM=true` testing this app was built and
verified against.

## Step 4 — Deploy and verify

- Select the **Free** plan
- Click **Create Web Service** and watch the build logs — this one pulls
  in LangChain/LangGraph/FAISS (same as the API service's build), so
  expect it to take noticeably longer than the Gradio build
- When you see `==> Your service is live`, open the URL — you should land
  on the Assistant tab
- Run through the test prompts in `architecture.md` Section 9g / the
  guardrail test guide (PII, injection, "Add AAPL to my watchlist", the
  Review queue tab's approve/reject buttons, the Safety lab's fairness
  tool, the Governance tab's audit export)

## Honest notes

- **This service's audit trail is its own, separate SQLite file** —
  `data/safety_audit.db` on *this* service's local disk, independent from
  the API service's own `data/safety_audit.db`. The two services don't
  share a filesystem, so an action previewed here won't show up in the
  API's `/review/{session_id}`, and vice versa. If you want one unified
  audit trail across every deployed surface, `Store`'s SQLite backend
  would need to move to a shared database (e.g. a managed Postgres/SQLite-
  over-network service) — out of scope here, and not something this
  project's `Store` implementation supports today.
- **Free-tier ephemeral disk, same as the API**: that audit trail (and
  the review queue in it) is wiped on every redeploy/restart, same
  limitation `RENDER_DEPLOY.md` already documents for the API service.
- **This is the heaviest of the three UI deploys** — it carries the full
  LangChain/LangGraph/FAISS stack in its own process, where Gradio/React
  carry none. If `MOCK_LLM=false`, expect a RAM profile closer to the API
  service than to `gradio_app.py`; the same OpenAI-embeddings fix already
  applied to `investment_research_api.py` (to dodge the free tier's
  512MB cap — see `RENDER_DEPLOY.md`) is used here too, for the same
  reason.
- **No authentication**, same as every other UI/endpoint in this project.
  The Review queue and Safety lab tabs are read/write controls with no
  login — fine for a demo URL you control, not for a public link with
  real users.
- **Not built or run against a live Render instance from this sandbox**
  (same network restriction noted throughout this project for other
  packages/services) — the Start Command's flags follow Streamlit's own
  documented cloud-deployment guidance, but weren't exercised against an
  actual Render container here. If the health check fails on first
  deploy, check the build logs for the exact Streamlit startup line
  before assuming the flags are wrong.
