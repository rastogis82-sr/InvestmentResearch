# Investment Research Assistant — Architecture & Design

## 0. Summary

An agentic AI assistant that answers investment-research questions about a
small synthetic universe of companies, grounded in retrieved documents
(RAG) and structured tool calls, with explicit guardrails against
fabrication, prompt injection, blank/gibberish input, and conflicting
numeric data. Implemented in Python, runnable in Google Colab, and
deployable as a public FastAPI service via ngrok.

**Framework choice: LangGraph.** LangChain's higher-level agent
abstractions, CrewAI, and Google ADK were all considered. LangGraph was
chosen because this assignment's requirements — explicit branching on
input validity, a parallel RAG+tools fan-out, a strict "never fabricate"
contract enforced structurally rather than only by prompting, and durable
per-session memory — map directly onto a `StateGraph` of small, auditable,
independently-testable node functions with explicit conditional edges. A
single-LLM-agent-with-tools framework (plain LangChain agents, CrewAI)
would hide this control flow inside the model's own tool-selection
reasoning, which is harder to guarantee and harder to unit-test
deterministically. Google ADK was not used, as it adds Google-specific
scaffolding this assignment does not need.

**RAG: yes**, via a small synthetic in-memory corpus + FAISS, not a live
web-search API (Serper/SearchApi.io were considered and rejected — the
assignment's test cases, including the deliberate RAG-failure and
poisoned-document scenarios, need a fixed, reproducible knowledge base
rather than live, changing search results).

**UI: Gradio**, not Streamlit — chosen for lower-friction Colab embedding
(`demo.launch(share=True)` produces a public URL immediately, no separate
hosting step) and a natural two-column chat-plus-transparency-panel layout.

---

## 1. Architecture diagram

![Architecture](architecture_diagram.png)

The diagram shows the compiled `StateGraph`: color-coded by node type
(orange = deterministic guardrail, gray = routing/control, blue =
tools/RAG data retrieval, green = LLM synthesis, purple = session memory,
red = terminal template responses, black = graph START/END).

---

## 2. Node-by-node walkthrough

| Node | Type | Responsibility |
|---|---|---|
| `input_guard` | deterministic | Blank check, injection regex, gibberish heuristic. Runs first on *every* turn and is the designated "clean slate" point — see Section 3. |
| `blank_node` / `gibberish_node` / `injection_node` | deterministic template | Terminal responses for the three failure modes `input_guard` can catch outright. |
| `intent_router` | LLM (`IntentResult`) | Resolves the target company (falling back to session memory for pronoun/follow-up references), detects fabrication requests, and makes the *authoritative* semantic judgment on whether input is gibberish/unrelated (see Section 7's two-tier design). |
| `refusal_node` | deterministic template | Fires when `is_fabrication_request` is true — refuses rather than inventing a plausible-sounding number. |
| `clarification_node` | deterministic template | Fires when no company can be resolved — asks the LLM-generated clarifying question rather than guessing. |
| `fan_out` | no-op | Trivial node with two unconditional outgoing edges, used only to fan out to `rag_node` and `tools_node` in parallel using the most basic, version-stable LangGraph primitives (see Section 3). |
| `rag_node` | data | FAISS similarity search (HuggingFace local embeddings), company-filtered in Python, with a similarity threshold below which "no relevant document" is reported rather than returning a weak match. Also applies the injection-detection regex to each retrieved chunk. |
| `tools_node` | data | Deterministic, orchestrated calls to the four mock tools (financials, price, profile, sector) — not left to the LLM's own tool-choice, for auditability. |
| `reconcile_node` | control | The single fan-in point after the parallel `rag_node`/`tools_node` branch. Combines `rag_flags` + `tool_flags` and adds `conflicting_data` flags by comparing numbers mentioned in retrieved text against tool-returned numbers (tolerance-based). |
| `synthesize_brief_node` | LLM (`ResearchBrief`) | Produces the final structured brief: `retrieved_facts` / `tool_data` / `calculations` / `assumptions` / `limitations` / `summary` — the system prompt instructs the model never to state a number outside one of these buckets. |
| `update_memory_node` | memory | Updates `session_entities` (remembered company/focus) and appends the assistant's reply to `messages`, both persisted via the LangGraph checkpointer keyed by `thread_id`/`session_id`. |

---

## 3. State design rationale

- **`messages`** uses the standard `Annotated[List[BaseMessage], add_messages]`
  reducer so chat history appends correctly turn over turn.
- **`rag_flags` / `tool_flags` as separate, disjoint keys** (rather than one
  shared `flags` key with a custom additive reducer): `rag_node` and
  `tools_node` run concurrently in the fan-out, so a shared key would need
  a custom reducer. Worse, because the checkpointer persists state *across
  turns* as well as within one, a naive additive reducer would accumulate
  flags from every past turn forever with no reset mechanism. Giving each
  concurrent node its own output key avoids any custom reducer; a single
  sequential node (`reconcile_node`) is the only writer of the final
  `flags`.
- **`input_guard` as "the clean slate point"**: because state persists
  across turns via the checkpointer, every transient per-turn field
  (`retrieved_docs`, `tool_results`, `flags`, `brief`, `response`, etc.)
  must be explicitly reset at the start of each new turn, or a prior turn's
  values could silently leak into the current one. `input_guard` runs
  first on every path and returns a full reset dict (`TRANSIENT_DEFAULTS`)
  merged with its own classification result, making it the one place this
  reset is guaranteed to happen.
- **Defensive `.get()` on `session_entities`**: on a brand-new thread's
  first turn, LangGraph has not yet materialized any state channel beyond
  what was passed into `.invoke()` — unlike the dependency-free local-dict
  simulation used during development, which always pre-populated every
  key. `intent_router` and `update_memory_node` both read
  `state.get("session_entities", {"company": None, "focus": None})` rather
  than `state["session_entities"]` to avoid a `KeyError` on that first
  turn.
- **Fan-out via a no-op `fan_out` node** with two unconditional
  `add_edge` calls, rather than a conditional-edge function returning a
  list of destination node names. Both are valid LangGraph patterns; this
  implementation deliberately uses only the single-string-return form of
  `add_conditional_edges` (with an explicit path map) everywhere, for
  version stability.

---

## 4. RAG design

- Corpus: five short synthetic documents — two ABC Technologies analyst
  notes (one of which, `sector_note_injection.txt`, is a **deliberately
  poisoned document** containing an embedded prompt-injection string), one
  XYZ Corp analyst note, and one company-less general sector outlook.
  **DEF Industries** has full tool data but *no* corpus document — the
  fixture for the RAG-failure test case.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` via
  `HuggingFaceEmbeddings` — free, local, no API key required regardless of
  which paid chat LLM is configured.
- Retrieval: FAISS `similarity_search_with_score`, company-filtered in
  **Python** (not a specific LangChain/FAISS version's native metadata
  filter argument, which varies across versions) and thresholded
  (`SCORE_THRESHOLD = 0.35`) so that a weak match is treated as "no
  relevant document found" rather than silently returned anyway.
- Every retrieved chunk is run through the same injection-detection regex
  used on user input, so a poisoned document is caught the same way a
  poisoned user message would be.

---

## 5. Tools design

Four mock tools (`get_company_financials`, `get_stock_price`,
`get_company_profile`, `get_sector_data`) living in `tools_mock.py`, each
returning a structured `{"ok": True/False, ...}` result — never raising,
never silently returning a fabricated placeholder. Tool invocation is
**deterministic and orchestrated** by `tools_node`, not left to the LLM's
own tool-selection judgment, so exactly which tools ran (and whether they
succeeded) is always known and testable.

---

## 5b. Real-company market data (Twelve Data)

The synthetic ABC/XYZ/DEF universe exists specifically because the
assignment's required test scenarios (forced tool failure, guaranteed RAG
failure, conflicting data, a poisoned document) need to be deterministic
and reproducible on demand — a live data source can never guarantee that.
Real companies were added **alongside** that universe, not instead of it,
to answer the genuinely useful "what's AAPL's price right now" kind of
question, via the Twelve Data REST API (https://twelvedata.com/stocks),
using a free, no-credit-card `TWELVEDATA_API_KEY`.

- **`real_market_data.py`** curates a small, explicit `REAL_COMPANIES`
  dict (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`, `NVDA`) — kept small and
  closed-set, the same design reasoning as the three synthetic companies,
  rather than open-ended symbol search against Twelve Data's full listing.
  It exposes `get_real_stock_price` / `get_real_company_profile` /
  `get_real_company_financials`, each returning the *exact same*
  `{"ok": True/False, "error":..., "tool":..., "data":...}` contract
  `tools_mock.py` already uses.
- **`tools_node` (in `node_logic.py`) dispatches** between the mock and
  real tool functions based on whether the resolved ticker is in
  `REAL_COMPANIES`:
  ```python
  is_real_company = ticker in REAL_COMPANIES
  financials_fn = get_real_company_financials if is_real_company else get_company_financials
  profile_fn = get_real_company_profile if is_real_company else get_company_profile
  price_fn = get_real_stock_price if is_real_company else get_stock_price
  ```
  No other node changes shape to support this — `reconcile_node`,
  `synthesize_brief_node`, and every guardrail treat a real-company tool
  result identically to a mock one, since the contract is identical.
  `intent_router` resolves a `company_reference` against the **union** of
  both registries (`tools_mock.known_companies()` merged with
  `real_market_data.known_real_companies()`), so "What's Tesla's stock
  price?" and "Compare ABC to Apple" both resolve correctly.
- **Honest free-tier limitation:** Twelve Data's free ("Basic") plan only
  includes the `/quote` endpoint (live price, day change, volume, 52-week
  range) — `get_real_stock_price` works on a free key for every ticker
  above. The `/profile` endpoint (sector, industry, HQ, description)
  requires a paid Grow-tier-or-higher plan, and there is no free
  income-statement/fundamentals endpoint at all. On a free key,
  `get_real_company_profile` and `get_real_company_financials` (which also
  reads `/profile`) both legitimately return
  `{"ok": False, "error": "plan_restricted:..."}`, surfaced as an ordinary
  `tool_failure` flag — consistent with this project's "never fabricate,
  just say so" contract, now holding up against a real API's real plan
  restrictions instead of a simulated failure, rather than being treated
  as a bug to work around.
- **Real companies will always show `rag_failure`:** the synthetic corpus
  only contains documents about ABC/XYZ/DEF, so any real-company question
  correctly flags no matching document found — this is expected, not an
  error, and is called out in the notebook's live-query demo cell (Section
  10b) rather than silently hidden.
- **Scope limitation — offline mock-LLM mode:** `mock_llm.py`'s
  `_guess_companies()` heuristic (used only when `MOCK_LLM = True`, the
  free/offline demo mode) still reads company names from
  `tools_mock.known_companies()` only, so it will not recognize a real
  company name like "Apple" by name (a bare ticker like "AAPL" still
  routes through `_resolve_any_ticker`, since router-level resolution is
  shared code). Real-company name resolution is fully correct only on the
  production path (`MOCK_LLM = False`, a real LLM via `intent_router`).
  This was a deliberate scope decision, not a bug — extending
  `_guess_companies()` to also check `real_market_data.known_real_companies()`
  would close this gap if the offline demo needs it later.
- Verified with 27 unit tests in `test_real_market_data.py` (mocked
  `requests.get`, since this is a genuinely live third-party API) covering
  ticker/name resolution, a missing API key, a successful `/quote` parse,
  a plan-restricted `/profile` response, a successful `/profile` parse, an
  unknown symbol, and a network timeout — all **27/27 pass** — plus the
  existing 23-check `test_harness.py` suite, confirmed unchanged at
  **23/23** after this integration (the synthetic path makes zero network
  calls, verified explicitly via `mock_get.call_count == 0` for an ABC
  query).

---

## 6. Guardrails and security

| Failure mode | Detection | Where |
|---|---|---|
| Blank input | `is_blank()` | `input_guard` (deterministic, pre-LLM) |
| Prompt injection (user input or retrieved doc) | regex `_INJECTION_RE` | `input_guard` (user input) and `rag_node` (retrieved chunks) |
| Gibberish / nonsensical input | two-tier: fast heuristic + authoritative LLM judgment | `input_guard` (fast path) and `intent_router` (`IntentResult.is_gibberish_or_unrelated`) |
| Fabrication request | keyword/intent detection | `intent_router` (`IntentResult.is_fabrication_request`) → `refusal_node` |
| Missing company/context | no ticker resolved, no session fallback | `intent_router` → `clarification_node` |
| Tool failure | `{"ok": False, ...}` | `tools_node` → flagged in `tool_flags` |
| No relevant document found | below similarity threshold | `rag_node` → flagged in `rag_flags` |
| Conflicting numeric data (doc vs. tool) | number extraction + tolerance comparison | `reconcile_node` |

**Design evolution worth calling out explicitly:** the assignment's
required gibberish test case, `"banana banana 999"`, is lexically valid
(real English words, a plausible vowel ratio), so the pure
character-heuristic `gibberish_score` alone scores it as *not* gibberish
(≈0.40 against a 0.6 threshold) — a genuine limitation of any
heuristic-only approach. This was solved by adding
`is_gibberish_or_unrelated` to the `IntentResult` schema so `intent_router`
makes the authoritative semantic judgment via the LLM, while the
heuristic in `input_guard` remains in place only as a free, fast pre-filter
that catches obvious character-salad input before it would otherwise incur
an LLM call.

**Honest limitation:** prompt-injection defense here is first-layer only —
a regex prefilter plus a system-prompt trust hierarchy in the synthesis
call. It is not a guarantee against every possible injection phrasing, and
a production system would want additional layers (e.g. a dedicated
classifier, output-side verification).

---

## 7. Session memory

`MemorySaver` (LangGraph's in-process checkpointer) persists `messages`
and `session_entities` (remembered company/focus) keyed by `thread_id` (in
the FastAPI deployment, `session_id` is passed through directly as
`thread_id`). This allows follow-up turns like *"Now compare its
profitability with XYZ"* to resolve "its" back to a company named in an
earlier turn. A pronoun+comparison heuristic
(`\bits\b|\bit\b` combined with a `"compare"` cue) ensures a single
company name in such a follow-up is captured as the **comparison target**
rather than overwriting the remembered primary company.

---

## 8. Testing

All business logic was developed and verified in a **dependency-free local
simulation** (`node_logic.py` + `graph_sim.py`, using a heuristic
`MockChatModel` and a keyword-overlap `LocalKeywordRetriever` standing in
for the LLM and vector store), because this development sandbox could not
install `langgraph`/`langchain`/`faiss`/`sentence-transformers` (outbound
package installation was blocked). This produced a 23-check automated test
suite (`test_harness.py`) covering every required scenario — positive,
missing information, fabrication refusal, tool failure, RAG failure,
conflicting data, gibberish (both the heuristic fast-path and the
semantic/LLM-judged case), blank input, injection via user input,
injection via a retrieved document, and two-turn session memory — which
passes **23/23**.

The identical node functions (with the two adaptations noted in Section 3
and the notebook) were then ported into the real LangGraph `StateGraph` in
`Investment_Research_Assistant.ipynb`, which includes an equivalent
test-harness cell run against the real compiled graph (free and offline
with `MOCK_LLM = True`, or against a real provider with `MOCK_LLM =
False`).

---

## 9. Deployment (FastAPI + ngrok)

The notebook's Section 12 wraps the same compiled LangGraph agent
(`research_graph` in the standalone service, `app` inside the notebook
itself) in a FastAPI service and exposes it on a public URL via ngrok,
following the background-thread-Uvicorn-plus-pyngrok-tunnel pattern from
the course's reference deployment notebook.

### Request path

![Deployment](deployment_diagram.png)

```
External Client → Ngrok Cloud (HTTPS, TLS termination, reverse proxy)
  → Colab VM → ngrok agent (pyngrok) → Uvicorn (0.0.0.0:8000, background thread)
  → FastAPI app → research_graph.invoke(state, config={"thread_id": session_id})
```

### File layout

- **`investment_research_api.py`** — a standalone, importable FastAPI
  service containing the embedded, tested logic from `tools_mock.py`,
  `guardrails.py`, `schemas.py`, `corpus.py`, `mock_llm.py`, and
  `node_logic.py`, plus the FAISS vector store setup, the compiled
  LangGraph graph (named `research_graph` to avoid colliding with the
  `app = FastAPI(...)` instance), and the FastAPI app itself.
- The notebook writes this file to disk from a **base64-encoded** string
  literal (decoded and `ast.parse`-verified before being written) —
  avoiding the string-escaping pitfalls of pasting ~1,500 lines containing
  quotes, f-strings, and triple-quoted docstrings directly into a notebook
  cell.

### Configuration: a `.env` file instead of repeated `getpass()` prompts

The notebook's Section 0 writes a `.env` file to the Colab VM's local disk
(`%%writefile .env`) with placeholder values, which you edit in place with
your real keys before running the load cell that follows it
(`python-dotenv`'s `load_dotenv()`). Any field left as a placeholder (i.e.
still starting with `your_`) falls back to an interactive `getpass()`
prompt for the keys that are actually required — `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` and `NGROK_AUTH_TOKEN` — while the LangSmith fields
stay genuinely optional.

```
OPENAI_API_KEY=your_openai_key_here
OPENAI_BASE_URL=https://openai.vocareum.com/v1
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=investment-research-assistant
LANGCHAIN_TRACING_V2=true
NGROK_AUTH_TOKEN=your_ngrok_token_here
```

- **`OPENAI_BASE_URL`** routes every LLM call through a Vocareum
  OpenAI-compatible proxy endpoint instead of `api.openai.com` — needed
  for a course-issued Vocareum key rather than a personal OpenAI account.
  It's pre-filled by default; clear it to call OpenAI directly. Both the
  notebook's own LLM-setup cell and `investment_research_api.py` read this
  (mirrored to `OPENAI_API_BASE` too, since library versions vary in which
  name they check) and pass it through to `init_chat_model(...,
  base_url=...)` whenever the provider is `"openai:..."`.
- **`LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` / `LANGCHAIN_TRACING_V2`**
  enable full LangSmith tracing of every LLM call and graph run when a
  real key is supplied — useful for inspecting exactly what
  `intent_router` / `synthesize_brief_node` produced on a given turn.
  Tracing is explicitly disabled (`LANGCHAIN_TRACING_V2=false`) when the
  key is left as a placeholder, so there's no silent partial
  configuration.
- `.env` lives only on the ephemeral Colab VM's disk (not Google Drive)
  and disappears when the runtime recycles — but treat a filled-in copy of
  this notebook as containing secrets, and don't share or commit it as-is.

### Configuration (environment variables read by `investment_research_api.py`)

| Variable | Default | Purpose |
|---|---|---|
| `MOCK_LLM` | `"false"` | `"true"` uses the offline heuristic `MockChatModel` — no API key needed, for free smoke-testing the deployment wiring itself. |
| `LLM_PROVIDER` | `"openai:gpt-4o-mini"` | Passed to `langchain.chat_models.init_chat_model` when `MOCK_LLM` is false. |
| `OPENAI_BASE_URL` / `OPENAI_API_BASE` | unset | Optional Vocareum (or other OpenAI-compatible) proxy URL — see above. |
| `IRA_SCORE_THRESHOLD` | `"0.35"` | RAG similarity threshold. |
| `IRA_CORPUS_DIR` | `/tmp/investment_research_kb` | Where the synthetic knowledge base is written before indexing. |
| `TWELVEDATA_API_KEY` | unset | Optional. Enables real-company (AAPL/MSFT/GOOGL/AMZN/TSLA/NVDA) market data via Twelve Data — see Section 5b. Unset or missing just makes real-company tool calls fail cleanly (`missing_twelvedata_api_key`); the synthetic ABC/XYZ/DEF universe is unaffected either way. |
| `TWELVEDATA_CACHE_TTL_SECONDS` | `"30"` | How long a Twelve Data response is cached in memory before a repeat call re-hits the network — see Section 9b. Set to `"0"` to disable caching entirely. |

### Endpoints

- `GET /health` — status, uptime, and current LLM configuration; used by
  the notebook's own startup polling loop and suitable for an external
  uptime check.
- `POST /chat` — `{"session_id": str, "message": str}` →
  `{"session_id", "response", "latency_ms", "flags"}`. Reusing the same
  `session_id` across calls gets session memory, since it is passed
  straight through as the LangGraph `thread_id`. Note `message` has **no
  `min_length` constraint** — an empty string is valid input that should
  reach the graph's own blank-input guardrail rather than being rejected
  by Pydantic first.
- `GET /debug/{session_id}` — returns the resolved company, flags, tool
  results, and retrieved chunks behind a session's current state — the API
  equivalent of the Gradio UI's transparency panel. **Deliberately left
  unauthenticated**, which is acceptable only because it sits behind a
  private, short-lived ngrok URL for a classroom demo (see Limitations).

### Operational details

- **Structured JSON logging** (`JSONFormatter`, never `print()`) so
  request/response/error events are machine-parseable (Datadog, CloudWatch,
  GCP Logging, etc. all ingest JSON lines directly).
- **CORS** is wide open (`allow_origins=["*"]`) — appropriate for a demo
  reachable only via an ngrok URL you control, not for a production
  deployment serving real users.
- Uvicorn runs in a **background daemon thread** inside the notebook
  process so the cell returns immediately; a health-check polling loop
  (60s timeout) confirms the server is actually serving before the
  notebook proceeds to open the ngrok tunnel.
- `ngrok.kill()` is called before opening a new tunnel, to clear a stale
  tunnel from a previous run and avoid the free tier's `ERR_NGROK_334`
  ("too many simultaneous tunnels") error.
- The free ngrok tier shows a one-time browser interstitial to human
  visitors; the `ngrok-skip-browser-warning: true` header bypasses it for
  API clients (used throughout the notebook's own live-endpoint tests).
- `GET /` redirects to `/docs`, since the bare host/tunnel URL has no
  content of its own — otherwise the first thing a human visitor sees is a
  bare `{"detail":"Not Found"}`.
- **Customizing the ngrok URL:** ngrok's free tier does not allow choosing
  a custom subdomain name — that requires a paid plan. It does give every
  account one free, fixed **dev domain** (e.g.
  `your-assigned-name.ngrok-free.app`) that stays the same across runs,
  instead of a new random URL every time the tunnel reconnects. Claim it
  once at https://dashboard.ngrok.com/domains ("+ Create Domain" — the
  name itself is auto-assigned, not typed in), put it in `.env`'s
  `NGROK_DOMAIN` field, and Section 12.3 passes it to
  `ngrok.connect(8000, "http", domain=NGROK_DOMAIN)` instead of letting
  ngrok assign a random one. Leave that field as its placeholder to keep
  the default random-URL behavior.

### Troubleshooting

- **`TypeError: Type is not msgpack serializable: numpy.float32`** when
  calling `chat(...)` against the real (non-mock) graph: this was a real
  bug in an earlier version of the `FAISSRetriever.retrieve()` method —
  FAISS's `similarity_search_with_score` returns `distance` as
  `numpy.float32`, and the derived similarity score inherited that numpy
  dtype all the way into `retrieved_docs`, which `MemorySaver`'s
  msgpack-based checkpoint serializer cannot encode. Fixed by casting to a
  plain Python `float` at the point the score is computed
  (`similarity = float(1.0 / (1.0 + distance))`) in both the notebook's
  Section 5 and `investment_research_api.py`. If you're on an older copy
  of either file, re-download the current version.
- **`langsmith.utils.LangSmithError: ... 403 Client Error: Forbidden`**
  repeated on every chat call: this is the optional LangSmith tracing
  integration (Section 0's `.env` fields), and was never fatal — LangChain
  logs the warning and the agent call itself completes normally — but an
  unauthorized key used to cause this to print on *every single* graph
  run, which is its own problem even though nothing is actually broken.
  Section 0 now validates the key with one cheap API call before turning
  tracing on at all: a bad/unauthorized `LANGSMITH_API_KEY` is caught once
  up front (printing a single clear "tracing disabled" message) rather
  than spamming a 403 on every call, and `langsmith.client`'s logger is
  also set to `ERROR` as a second layer in case a transient failure slips
  through after tracing was validated. If you want tracing to actually
  work, fix the key/project at smith.langchain.com; otherwise leave
  `LANGSMITH_API_KEY` as its `.env` placeholder.
- **`[FAIL] rag_failure: flagged`** in the Section 10 test harness, only
  when running against the real FAISS + embeddings retriever (the offline
  `MOCK_LLM = True` / `LocalKeywordRetriever` path always passed this
  check): the original `rag_node` flagged `rag_failure` only when *zero*
  chunks came back at all. Retrieval deliberately lets company-less,
  general-sector documents through regardless of `company_filter` (so a
  general outlook can support any company's question) — and with a real
  semantic embeddings model, that general document can score as
  "relevant" to almost any company query, including DEF Industries
  (which has tool data but, by design, zero documents in the corpus — the
  fixture for this exact test case). So a generic hit was silently masking
  the fact that nothing *specific to DEF* was ever found. Fixed by
  changing the check to "was any chunk's `company` equal to the resolved
  company" rather than "were any chunks returned at all" — a
  company-less chunk can still be included in `retrieved_docs` for
  context, but no longer prevents the `rag_failure` flag on its own.
- **`KeyError: 'brief'`** (or any other `Optional[...]` field) when
  reading `get_state(thread_id).values` after a real graph run, on a path
  that never set that field to anything but its `input_guard`-reset
  `None` (e.g. `brief` on the injection/refusal/clarification paths, which
  never reach `synthesize_brief_node`): unlike the dependency-free local
  simulation (where `new_state()` always pre-populates every key), the
  real compiled LangGraph app does not guarantee a channel whose value is
  still exactly `None` will appear in `get_state().values()` at all — only
  `input_classification`, `flags`, and similar fields that get set to a
  real (non-`None`) value are guaranteed present. Section 10's test
  harness and `investment_research_api.py`'s `/debug` endpoint and Gradio
  transparency panel all read these fields with `.get(key, default)`
  rather than `state[key]` for exactly this reason — if you add your own
  state-inspection code, do the same for any `Optional[...]` field.

### Honest limitations

- The free-tier ngrok URL is **ephemeral** — it changes every time the
  tunnel cell is re-run, and both the tunnel and the server end when the
  Colab runtime disconnects. This is a demo deployment, not a durable one.
- `MemorySaver` is **in-process and in-memory**: session state is lost on
  restart and would not be shared across multiple worker processes — a
  production deployment needing durable or horizontally-scaled session
  memory would need a persisted checkpointer (e.g. backed by Redis/Postgres,
  which LangGraph supports via other checkpointer implementations).
- `/debug/{session_id}` has no authentication.
- CORS is wide open.
- This is a single Colab-hosted process behind a tunnel, not a scaled,
  production-grade deployment — appropriate for this assignment's demo
  requirements, not as-is for serving real users.

---

## 9b. Performance and token-usage optimizations

A later pass over the whole solution, specifically targeting runtime
latency, external-API efficiency, LLM token usage, and FastAPI service
overhead — each verified against the existing test suites rather than
assumed safe.

**External API efficiency (`real_market_data.py`).** `tools_node` was
issuing three sequential Twelve Data calls per real-company turn, but two
of them — `get_company_profile` and `get_company_financials` — hit the
exact same `/profile` endpoint for the same ticker (Twelve Data has no
separate free fundamentals endpoint; `get_company_financials` reads
`/profile` too — see Section 5b). That duplicate call is now eliminated by
a short-TTL (`TWELVEDATA_CACHE_TTL_SECONDS`, default 30s) in-memory cache
inside `_twelvedata_get`, keyed by `(endpoint_path, ticker)`: whichever
call runs first populates the cache, and the second is served from memory
with its `"tool"` field relabeled to match the caller. Only deterministic
outcomes are cached — a successful response and a deterministic API error
like `plan_restricted` or "symbol not found" — never a transient network
error, so a momentary blip isn't "remembered" as a sustained outage for
the rest of the TTL window. A repeated question about the same ticker
within that window (e.g. "what's AAPL's price?" asked twice) is also now a
cache hit rather than a fresh call, which matters on Twelve Data's free
tier's limited per-minute request quota. A module-level `requests.Session()`
(`_SESSION`) replaces ad hoc `requests.get()` calls, reusing one HTTP
connection (keep-alive) across every call instead of a new TCP/TLS
handshake each time. Verified with 8 new checks in
`test_real_market_data.py` (now 35/35) that explicitly assert the mocked
network call count — e.g. `get_company_profile` + `get_company_financials`
for the same ticker make exactly **one** mocked network call, not two;
`clear_cache()` forces a fresh call; a mocked timeout is never cached.

**Latency (`node_logic.py`'s `tools_node`).** For a real company, `/quote`
(price) and `/profile` (profile) are independent Twelve Data endpoints
with no data dependency between them, so they're now fetched concurrently
via a small `concurrent.futures.ThreadPoolExecutor(max_workers=2)` instead
of waiting for one blocking network call before starting the other.
`get_company_financials` is called only after that pool resolves — not
submitted into it alongside `get_company_profile` — so it deterministically
lands on the cache entry profile just populated instead of racing it (two
truly-concurrent calls would both miss an empty cache and both hit the
network, defeating the dedup above). Net effect for a real-company turn:
two real network calls instead of three, and the two that remain run in
parallel rather than in series. Verified end-to-end (not just at the
`real_market_data.py` unit level) with a direct call to `tools_node` under
mocked HTTP, asserting exactly 2 network calls and a correctly-relabeled
`get_company_financials` result. The synthetic ABC/XYZ/DEF branch is
untouched — those are in-memory dict lookups with no I/O to parallelize,
and leaving that code path exactly as it was keeps the existing 23-check
`test_harness.py` suite (which only exercises synthetic companies) a
no-risk change rather than something that needed re-verifying.

**LLM token usage (`synthesize_brief_node`).** `_render_tool_line` used to
interpolate `result["data"]`'s raw Python dict repr into the prompt sent
to the chat LLM. That repr included two kinds of pure token cost with zero
decision-relevant information: a `"source"` field whose value is always
the same static string, repeated once per tool result every single turn,
and `None`-valued fields (common on a real company's profile, where
Twelve Data doesn't return every field). It now renders compact
`key=value` pairs with both dropped, while keeping every field the LLM
actually needs — including Twelve Data's free-tier "note" on
`get_company_financials`, since the brief's "limitations" bucket depends
on the model seeing it. `mock_llm.py`'s `_mock_brief` only does
header-based line-splitting (it never parses specific field names out of
`TOOL_RESULTS:`), so this formatting change doesn't touch the 23-check
suite's assertions — confirmed by re-running `test_harness.py` (still
23/23) after the change.

**FastAPI service efficiency.** Added `GZipMiddleware` (500-byte floor) to
`investment_research_api.py`'s app, alongside the existing CORS middleware
— compresses `/chat` and especially `/debug/{session_id}` responses
(which can include several retrieved-document chunks plus full tool
results) before they cross the ngrok tunnel. `/health` and other tiny
responses are left uncompressed by the size floor, since compression
overhead there would cost more than it saves.

**Considered and intentionally left alone:**
- *FAISS/embeddings as singletons*: already built once at module import
  time (`_embeddings`, `_vectorstore`, `retriever` in Section 5 /
  `build_api_app.py`), not per-request — there was nothing to fix here;
  this was confirmed, not changed.
- *Caching LLM responses themselves* (not just tool/API calls): considered
  and rejected. A cached answer to "what's AAPL's price" would go stale
  the moment the real price moves, which directly conflicts with this
  project's foundational "never present fabricated or stale data as
  current fact" guardrail philosophy (Section 6) — the token savings
  weren't worth reintroducing that risk.
- *Making `/chat` an `async def` endpoint*: FastAPI already runs a
  sync `def` route in a worker thread pool automatically, so concurrent
  requests don't block the event loop as-is; converting to `async def`
  would need an async LLM client and an async Twelve Data client to
  actually gain anything, which is more surface area than this demo-scale
  deployment needs.

---

## 9c. Deployment 2: Render.com (alongside ngrok)

Section 9's ngrok tunnel and this Render deployment serve two different
needs with the exact same application code — nothing about
`investment_research_api.py` changes between them, only the envelope
around it:

| | ngrok (Section 9) | Render.com |
|---|---|---|
| URL | Random, changes every restart (or a claimed dev domain) | Permanent, same URL across restarts |
| Lifetime | Dies when the Colab runtime disconnects | Persists; auto-restarts on crash |
| Best for | A time-boxed demo, iterating in Colab | A URL a grader/teammate can hit any time, without you keeping a notebook running |

**Config is environment-only, same rule as everywhere else in this
project**: `investment_research_api.py` now also calls `load_dotenv()` at
startup (it's a no-op if no `.env` file exists), so locally or in
Vocareum you can copy `.env.example` to `.env` and fill in real values
instead of exporting shell variables by hand — `load_dotenv()` only fills
in a key that ISN'T already set, so it never overrides whatever a real
host (Render, Docker, Cloud Run) injects directly, and nothing is ever
hardcoded in the app, in `render.yaml`, or anywhere else in the repo.
`.env` itself is excluded via `.gitignore` and must never be committed.

**Deployment bundle** (five files, all at the repo root, all pushed to
GitHub — see `RENDER_DEPLOY.md` for the exact commands and Render
dashboard steps):

- `investment_research_api.py` — the service itself, unchanged behavior.
- `requirements.txt` — every package the service actually imports
  (FastAPI/Uvicorn, LangGraph/LangChain + both `langchain-openai` and
  `langchain-anthropic` since `LLM_PROVIDER` supports either, FAISS,
  HuggingFace embeddings/sentence-transformers, `python-dotenv`,
  `requests`), each pinned to a minimum version.
- `render.yaml` — build command (`pip install -r requirements.txt`),
  start command (`uvicorn investment_research_api:app --host 0.0.0.0
  --port 10000`), `healthCheckPath: /health`, and the full set of
  environment variables the app reads. Non-secret config (`MOCK_LLM`,
  `LLM_PROVIDER`, `IRA_SCORE_THRESHOLD`, `TWELVEDATA_CACHE_TTL_SECONDS`)
  is inlined directly in the file since there's nothing sensitive about
  it; every actual secret (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `OPENAI_BASE_URL`, `TWELVEDATA_API_KEY`) is declared with `sync: false`,
  which makes Render prompt for the value in its dashboard rather than
  storing it in this version-controlled file.
- `.env.example` — a template (placeholder values only) documenting every
  config key the app reads, for local/Vocareum testing.
- `.gitignore` — excludes `.env`, `__pycache__/`, and other local-only
  files, so a real `.env` can never be pushed by accident.

**Honest limitations specific to Render, beyond what Section 9 already
covers for ngrok:**
- **Free-tier cold starts**: the service spins down after 15 minutes of
  inactivity; the first request after that takes 30-60s. Send a warm-up
  request before a demo.
- **Free-tier memory**: this service loads a local HuggingFace embeddings
  model and builds a FAISS index at startup — meaningfully heavier than a
  tool-only agent with no RAG component. Render's free instance has
  512MB RAM; if the build succeeds but the service never passes its
  health check, check the runtime logs for an out-of-memory kill before
  assuming the code is broken.
- **`MemorySaver` is still in-process** (Section 9's limitation holds
  here too, and is the sharper edge on Render specifically): any
  restart — a redeploy, a crash recovery, or a free-tier spin-down — loses
  every session's history. A production deployment needing durable
  memory across restarts would swap in a persisted LangGraph checkpointer
  (Redis/Postgres), same as noted in Section 9.
- **`/debug/{session_id}` is still unauthenticated** — fine for a demo URL
  only you and a grader know, not for a public production service.

---

## 10. Running instructions

1. Open `Investment_Research_Assistant.ipynb` in Google Colab.
2. Run Section 0's cells in order: install dependencies, run the
   `%%writefile .env` cell once, **edit that cell with your real keys**
   (OpenAI/Vocareum key, optionally a LangSmith key, your ngrok auth token,
   and optionally a free `TWELVEDATA_API_KEY` for real-company data — see
   Section 5b) and re-run it, then run the `load_dotenv()` cell — any
   placeholder left in `.env` falls back to a `getpass()` prompt for the
   keys that are required (`TWELVEDATA_API_KEY` is optional and has no
   `getpass()` fallback; leave it as its placeholder to skip real-company
   data entirely).
3. Run Sections 1–9 to build the agent (tools, guardrails, schemas, corpus,
   vector store, LLM, graph) — this includes Section 1b, which loads the
   real-company tools from `real_market_data.py`.
4. Run Section 10 to execute the full automated test suite against the
   real compiled graph. Section 10b is an optional, ungraded demo cell
   that asks a live real-company question (e.g. "What's Tesla's current
   stock price?") when `TWELVEDATA_API_KEY` is set and `MOCK_LLM = False`;
   it self-skips otherwise.
5. Optionally run Section 11 (`demo.launch(share=True)`) for an in-notebook
   Gradio chat UI.
6. Run Section 12 to deploy the agent as a public FastAPI service via
   ngrok, and exercise the live HTTP endpoint with the provided test calls.

---

## 11. Rubric mapping (test-case coverage)

| Required scenario | Covered by |
|---|---|
| Positive case | `test_harness.py` check 1 / notebook test `t1` / API demo 1 |
| Missing information → clarify | check 2 / `t2` |
| Fabrication request → refuse | check 3 / `t3` |
| Tool failure → flagged, not fabricated | check 4 / `t4` |
| RAG failure (no matching document) → flagged | check 5 / `t5` |
| Conflicting data (doc vs. tool) → flagged | check 6 / unit test on `reconcile_node` |
| Gibberish/unrelated input | checks 7a/7b / `t7`, `t7b` / API demo 2 |
| Blank input | check 8 / `t8` |
| Prompt injection via user input | check 9 / `t9` / API demo 3 |
| Prompt injection via retrieved document | check 10 / `t10` |
| Session memory across turns | check 11 / `t11` / API demo 4 |
