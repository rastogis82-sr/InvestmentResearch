# Deploying the Investment Research Assistant to Render.com

Following the same pattern as the Week 16 notebook's Section 5
("Deployment 2: Render.com"), applied to `investment_research_api.py`
instead of the course's ShopAssist demo. You already have a GitHub repo
and a Render account, so this skips account setup and goes straight to
pushing code and configuring the service.

All configuration is read from environment variables — nothing is
hardcoded in the app, in `render.yaml`, or anywhere else in this repo.
Locally you'd use a `.env` file (see `.env.example`); on Render you set
real environment variables in the dashboard instead. `.env` is excluded
in `.gitignore` on purpose — it must never be pushed.

## Files in this deployment bundle

| File | Purpose |
|---|---|
| `investment_research_api.py` | The FastAPI service itself (`/health`, `/chat`, `/debug/{session_id}`) — unchanged behavior, now also calls `load_dotenv()` at startup so a local `.env` file (if present) fills in any config that isn't already set by the real environment. |
| `requirements.txt` | Every package the service imports, pinned to a minimum version. |
| `render.yaml` | Render's infrastructure-as-code file — build command, start command, health check path, and the full list of environment variables the app reads (secrets marked `sync: false` so Render prompts for them in the dashboard rather than storing them in this file). |
| `.env.example` | Template for local/Vocareum testing — copy to `.env` and fill in real values. Never commit the real `.env`. |
| `.gitignore` | Excludes `.env`, `__pycache__/`, and other local-only files from git. |

## Step 1 — Push these files to your GitHub repo

From this project's working directory, run the four commands the notebook
teaches, staging only the five files above (not the notebook, not the
`.py` source files that get embedded into `investment_research_api.py`,
not `architecture_diagram.png`, etc. — Render only needs the standalone
service):

```bash
git init                       # skip if this repo already exists
git add investment_research_api.py requirements.txt render.yaml .env.example .gitignore
git commit -m "Investment Research Assistant — Render deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git   # skip if already set
git push -u origin main
```

If `git remote add origin ...` fails because a remote already exists,
use `git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git`
instead.

## Step 2 — Create the Web Service on Render

- render.com → **New +** → **Web Service**
- Connect the GitHub repo you just pushed to
- Render should auto-detect `render.yaml`. Verify these fields manually
  either way (Render sometimes leaves the start command blank on the free
  tier):

| Field | Value |
|---|---|
| Name | `investment-research-api` |
| Runtime | `Python` |
| Branch | `main` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn investment_research_api:app --host 0.0.0.0 --port 10000` |
| Health Check Path | `/health` |

## Step 3 — Set environment variables

`render.yaml` already declares the non-secret config inline (`MOCK_LLM`,
`LLM_PROVIDER`, `IRA_SCORE_THRESHOLD`, `TWELVEDATA_CACHE_TTL_SECONDS`).
In the **Environment** tab, set the secrets Render prompts for:

| Key | Required? | Value |
|---|---|---|
| `OPENAI_API_KEY` | Required (unless using Anthropic) | Your OpenAI key |
| `ANTHROPIC_API_KEY` | Only if `LLM_PROVIDER` starts with `anthropic:` | Your Anthropic key |
| `OPENAI_BASE_URL` | Optional | A Vocareum-style proxy URL, if you're using one instead of a personal OpenAI key |
| `TWELVEDATA_API_KEY` | Optional | Enables real-company (AAPL/MSFT/GOOGL/AMZN/TSLA/NVDA) live data — free signup at twelvedata.com/pricing. Without it, real-company questions get a clean tool failure; ABC/XYZ/DEF are unaffected either way. |

## Step 4 — Deploy and verify

- Select the **Free** plan (or paid, if you want to avoid cold starts)
- Click **Create Web Service** and watch the build logs
- When you see `==> Your service is live`, copy the URL — it'll look like
  `https://investment-research-api-xxxx.onrender.com`

Then verify with the same two-call pattern the notebook uses:

```python
import requests

RENDER_URL = "https://investment-research-api-xxxx.onrender.com"  # replace with your URL

health = requests.get(f"{RENDER_URL}/health", timeout=60)
print("Health:", health.json())

result = requests.post(
    f"{RENDER_URL}/chat",
    json={"session_id": "render_test_1",
          "message": "Research ABC Technologies: revenue growth and profitability."},
    timeout=60,
)
print("Response:", result.json())
```

## Honest notes before you deploy

- **Free-tier cold starts**: Render's free tier spins down after 15
  minutes of inactivity. The first request after that takes 30-60s. Send
  one warm-up request before a demo.
- **Free-tier memory**: this service loads a local HuggingFace embeddings
  model (`sentence-transformers/all-MiniLM-L6-v2`) and builds a FAISS
  index at startup — heavier than the course's ShopAssist demo, which has
  no RAG component. Render's free instance has 512MB RAM; if the build
  succeeds but the service crashes or never passes the health check,
  check the runtime logs for an out-of-memory kill and consider a paid
  instance tier with more RAM.
- **`MemorySaver` is in-process**: exactly the limitation the notebook's
  Discussion section (Question 3) calls out — a Render restart (deploy,
  crash-recovery, or free-tier spin-down) loses all session history. Fine
  for this assignment's demo scope; a production deployment would swap in
  a persisted LangGraph checkpointer (Redis/Postgres).
- **`/debug/{session_id}` has no authentication** — same note as in
  `architecture.md`'s Deployment section. Acceptable for a demo URL only
  you and your grader know, not for a public production service.
