# Deploying the React UI to Render (Static Site)

A third frontend for the same deployed API — alongside the Gradio UI, not
replacing it. This one is a modern chat interface built with React + Vite
+ TypeScript: message bubbles, a colored-badge/card/chart transparency
panel, a gradient header — the same data (flags, tool results, retrieved
sources) as the Gradio UI's panel, same validated color palette, just a
different look and feel. It's a pure static bundle (HTML/CSS/JS) that
calls the deployed API's `/chat` and `/debug/{session_id}` endpoints
directly from the browser — no server process of its own, so it deploys
on Render as a **Static Site**, not a Web Service.

## Files in this deployment bundle

All under `react-ui/`:

| File / folder | Purpose |
|---|---|
| `src/App.tsx` | Top-level layout: gradient header, two-column grid (chat + transparency panel), session-id lifecycle. |
| `src/components/ChatPanel.tsx`, `ChatMessage.tsx` | Chat log and input. A research-brief response ("Research brief — X" + labeled lines) is parsed into a real heading and bullet lists; anything else (a guardrail refusal, a clarification question) renders as plain text, unchanged. |
| `src/components/TransparencyPanel.tsx`, `Badge.tsx`, `ToolCard.tsx`, `MetricsChart.tsx` | Colored flag badges, one card per tool call (short "ok"/"failed" badge, full error text wrapped below it, never inside the pill), and an SVG bar chart of any `*_pct` fields — ported 1:1 from the same logic in `gradio_app.py`. |
| `src/api.ts` | `callChat` / `callDebug` — thin fetch wrappers around the API's `/chat` and `/debug/{session_id}`, same never-throw contract as the Python `_call_chat`/`_call_debug`. |
| `src/format.ts`, `palette.ts` | Flag/tool/field humanization, number formatting, and the exact palette hex values already shipped in `gradio_app.py`. |
| `package.json`, `vite.config.ts`, `tsconfig*.json`, `index.html` | Standard Vite + React + TypeScript project scaffold. |
| `.env.example` | Template for `VITE_API_BASE_URL` — see the build-time note below. |

## Step 1 — Install and build locally first

This sandbox can't run `npm install` (its registry is network-restricted
here), so this hasn't been built end-to-end in CI the way the Python
services were — it's been verified by rendering every component's output
with `react-dom/server` against realistic sample data (the same MSFT /
TwelveData-error scenario used to verify the Gradio panel) and
screenshotting the result, but **please do run this step yourself**
before trusting a Render build:

```bash
cd react-ui
npm install
cp .env.example .env        # then edit VITE_API_BASE_URL if needed
npm run dev                 # sanity check in a browser at localhost:5173
npm run build                # produces dist/ -- this is what Render runs
```

## Step 2 — Push these files to your GitHub repo

Same repo as the API and Gradio UI — Render builds a Static Site from a
subdirectory via the Root Directory setting below, so this doesn't
interfere with the other two services.

```bash
git add react-ui/
git commit -m "Add React frontend for the Investment Research Assistant"
git push
```

## Step 3 — Create a Static Site on Render

- render.com → **New +** → **Static Site**
- Connect the **same GitHub repo**
- Fill in:

| Field | Value |
|---|---|
| Name | `investment-research-react` (or anything you like) |
| Branch | `main` |
| Root Directory | `react-ui` |
| Build Command | `npm install && npm run build` |
| Publish Directory | `dist` |

## Step 4 — Set the build-time environment variable

This is the one genuinely different thing about a static site versus the
two Python services: **there is no running server here to read an
environment variable at request time.** Vite only exposes variables
prefixed `VITE_` to client code, and it inlines their value into the
JavaScript bundle at `npm run build` time. So:

| Key | Required? | Value |
|---|---|---|
| `VITE_API_BASE_URL` | **Required** | `https://investmentresearch.onrender.com` (your deployed API's URL — no trailing slash) |

Set this under the Static Site's **Environment** tab (Render calls these
"Environment Variables" there too, but on a Static Site they only affect
the *build*, not anything at runtime). If you change this value later,
you must trigger a new deploy (Render does this automatically when the
env var changes, or you can click **Manual Deploy**) — just saving it
doesn't update an already-built bundle sitting in `dist/`.

## Step 5 — Deploy and verify

- Click **Create Static Site** and watch the build logs
- When the deploy finishes, open the URL — you should see the gradient
  header, "Talking to: `https://investmentresearch.onrender.com`" under
  it, and the two-column chat + transparency-panel layout
- Send a test message and confirm: the chat bubble renders the research
  brief as a heading + bullet sections (not one dense paragraph), and the
  transparency panel fills in with colored flag badges, tool-call cards,
  and (when the response includes any `*_pct` field) a bar chart

## Honest notes

- **This UI and the Gradio UI are independent and both stay live** — per
  your choice, this doesn't replace `investment-research-gradio`. Both
  talk to the same API over HTTP and will show the same underlying data,
  just with different UI treatments.
- **No build verification from this environment.** Everything here was
  checked by rendering each component to static HTML with
  `react-dom/server` (using the pre-installed `react`/`react-dom`
  packages) against the exact sample payloads used to verify the Gradio
  panel, then screenshotting the result with Playwright — logic and
  layout are verified, but an actual `npm run build` has not run in this
  sandbox (its npm registry access is blocked). Do Step 1 before
  deploying.
- **Cold starts work differently here than for the Gradio UI.** A static
  site has no server to sleep — the page itself loads instantly. The
  *first* request to the API after it's been idle can still take
  30-60s on Render's free tier; `api.ts`'s `callChat`/`callDebug` use a
  60s timeout (`VITE_API_TIMEOUT_MS`, optional, default `60000`) for the
  same reason `API_TIMEOUT_SECONDS=60` is generous in the Gradio UI.
- **No authentication**, same as the other two frontends — fine for a
  demo URL, not for anything public-facing with real users.
- **Session persistence**: the session id lives in `sessionStorage` (not
  `localStorage`), so a page refresh keeps your conversation thread but a
  new tab, or the "New session" button, always starts fresh — intentional,
  so multiple visitors to the same deployed URL don't share one
  LangGraph checkpointer thread by accident.
