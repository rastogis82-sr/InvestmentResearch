# Deploying the Investment Research Assistant with Docker

Following the Week 16 notebook's Section 7 pattern (Deployment 3: Docker),
applied to both services you've already built and deployed to Render:
`investment_research_api.py` and `gradio_app.py`. Same code, different
envelope — nothing in either app file changes for Docker.

Unlike ngrok and Render, with Docker **you** control the exact Python
version, the exact dependency versions, and the runtime environment —
and the resulting image runs identically on your laptop, a teammate's
machine, or any cloud (Cloud Run, ECS, Kubernetes).

All configuration is still environment-only: secrets are injected at
**container runtime** (`--env-file` / docker-compose's `env_file`), never
baked into the image. `.dockerignore` excludes `.env` for exactly the
reason the notebook calls out — a leaked key in a pushed image has cost
companies millions.

## Files in this deployment bundle

| File | Purpose |
|---|---|
| `Dockerfile.api` | Builds the API image from `investment_research_api.py` + `requirements.txt`. |
| `Dockerfile.gradio` | Builds the Gradio UI image from `gradio_app.py` + `requirements-gradio.txt`. |
| `docker-compose.yml` | Runs both containers together, networked so Gradio reaches the API by Compose service name (`http://api:8000`) — no public URL needed to develop or demo locally. |
| `.dockerignore` | Excludes `.env`, notebook/build/test files, and anything else not needed inside either image — both `investment_research_api.py` and `gradio_app.py` are fully self-contained single files with no sibling-module imports, so nothing else in this repo is needed at runtime. |

## Quick start: docker-compose (recommended)

```bash
cp .env.example .env   # fill in real values — OPENAI_API_KEY at minimum
docker compose up --build
```

- API: http://localhost:8000 (`/docs`, `/health`, `/chat`)
- Gradio UI: http://localhost:7860

Compose builds both images, starts the API first, waits for its
healthcheck to pass, then starts Gradio already pointed at
`http://api:8000` — no manual networking needed.

Stop everything with `docker compose down` (add `-v` to also drop the
build cache volumes).

## Manual build/run (matching the notebook's commands exactly)

If you'd rather run each container by hand instead of Compose (useful
for understanding exactly what Compose automates, or to match the
notebook's Section 7 commands 1:1):

**Build:**
```bash
docker build -t investment-research-api:v1 -f Dockerfile.api .
docker build -t investment-research-gradio:v1 -f Dockerfile.gradio .
```

**Run the API:**
```bash
docker run -d \
  --name investment-research-api \
  -p 8000:8000 \
  --env-file .env \
  investment-research-api:v1
```

**Run the Gradio UI, pointed at the API container via Docker's built-in
host alias** (works without a Compose network, since both containers are
on the host's default bridge and `host.docker.internal` resolves to the
host from inside a container):
```bash
docker run -d \
  --name investment-research-gradio \
  -p 7860:7860 \
  -e API_BASE_URL=http://host.docker.internal:8000 \
  investment-research-gradio:v1
```

**Verify:**
```bash
docker ps
docker logs investment-research-api
docker logs investment-research-gradio

curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "docker_test_1", "message": "Research ABC Technologies: revenue growth and profitability."}'
```

Open http://localhost:7860 in a browser — same chat UI as the Render
deployment, same transparency panel, just talking to your local
container instead of the hosted one.

**Stop and remove:**
```bash
docker stop investment-research-api investment-research-gradio
docker rm investment-research-api investment-research-gradio
```

## Optional: push to a registry and deploy to Google Cloud Run

Same pattern as the notebook's Section 8, done twice (once per image).
Replace `YOUR_PROJECT_ID`.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create investment-research-repo \
  --repository-format=docker --location=us-central1
gcloud auth configure-docker us-central1-docker.pkg.dev

# API
docker tag investment-research-api:v1 \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/investment-research-repo/investment-research-api:v1
docker push \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/investment-research-repo/investment-research-api:v1
gcloud run deploy investment-research-api \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/investment-research-repo/investment-research-api:v1 \
  --platform managed --region us-central1 --allow-unauthenticated \
  --set-env-vars MOCK_LLM=false,LLM_PROVIDER=openai:gpt-4o-mini,IRA_SCORE_THRESHOLD=0.35,TWELVEDATA_CACHE_TTL_SECONDS=30 \
  --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest
# (or --set-env-vars OPENAI_API_KEY=... directly if you haven't set up Secret Manager)

# Gradio — API_BASE_URL points at the Cloud Run URL printed by the deploy above
docker tag investment-research-gradio:v1 \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/investment-research-repo/investment-research-gradio:v1
docker push \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/investment-research-repo/investment-research-gradio:v1
gcloud run deploy investment-research-gradio \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/investment-research-repo/investment-research-gradio:v1 \
  --platform managed --region us-central1 --allow-unauthenticated \
  --set-env-vars API_BASE_URL=https://investment-research-api-xxxx-uc.a.run.app
```

Cloud Run injects its own `PORT` (defaults to 8080 there, not 8000/7860)
— both `investment_research_api.py` (via its own `--port` in the
Dockerfile `CMD`) and `gradio_app.py` (via the `PORT` env var it already
reads) would need that adjusted for Cloud Run specifically; Render and
local Docker don't have this constraint, which is why the Dockerfiles
above hardcode 8000/7860 in `CMD`/`EXPOSE` rather than reading `PORT`.
If you do deploy to Cloud Run, override the start command's `--port` (API)
or set `PORT=8080` (Gradio) to match.

## Honest notes

- **Built and validated, not run end-to-end, in this environment.**
  I wrote and syntax/structure-validated `Dockerfile.api`,
  `Dockerfile.gradio`, and `docker-compose.yml` (confirmed valid with
  `docker compose config`, and confirmed both Dockerfiles parse correctly
  up through the base-image pull). I could not actually pull
  `python:3.11-slim` or run a full build in this sandbox — outbound
  access to Docker Hub (`registry-1.docker.io`) is blocked by this
  session's network policy. This is a sandbox restriction, not a problem
  with the files themselves; `docker build`/`docker compose up` should
  work normally on your own machine or in Vocareum's terminal, both of
  which have unrestricted Docker Hub access.
- **Image size**: `python:3.11-slim` + the full LangChain/LangGraph/FAISS
  stack is a few hundred MB — nowhere near the torch/transformers bloat
  that caused the Render OOM crash (that dependency was removed from
  `requirements.txt` entirely, see `architecture.md`'s Troubleshooting
  section), but still noticeably larger than the Gradio image, which has
  no ML dependencies at all.
- **`.env` must exist before `docker compose up`** — copy `.env.example`
  first. Without it, Compose fails fast with a clear "env file not
  found" error rather than silently starting with missing config.
- **`MemorySaver` is still in-process**, same limitation as every other
  deployment envelope in this project: a container restart loses session
  history.
