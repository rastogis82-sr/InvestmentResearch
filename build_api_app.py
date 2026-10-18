"""
build_api_app.py
-----------------
Assembles investment_research_api.py: a standalone FastAPI service wrapping
the same tested LangGraph agent (tools_mock.py, guardrails.py, schemas.py,
corpus.py, node_logic.py) used in Investment_Research_Assistant.ipynb.

Run with: python3 build_api_app.py
Produces: investment_research_api.py (validated with ast.parse)
"""

import ast
import re

LOCAL_MODULES = ["guardrails", "schemas", "tools_mock", "corpus", "node_logic", "mock_llm", "local_retriever", "real_market_data", "yahoo_finance_market_data", "merged_market_data", "safety_layer"]


def load_and_clean(path: str) -> str:
    with open(path) as f:
        src = f.read()
    src = re.split(r"\nif __name__ == .__main__.:\n", src)[0]
    lines = []
    skipping_multiline_import = False
    for line in src.splitlines():
        stripped = line.strip()
        if skipping_multiline_import:
            if ")" in stripped:
                skipping_multiline_import = False
            continue
        if stripped.startswith("from ") and any(f"from {m} import" in stripped for m in LOCAL_MODULES):
            if stripped.endswith("("):
                skipping_multiline_import = True
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


node_logic_src = load_and_clean("node_logic.py").replace(
    '''# ---------------------------------------------------------------------------
# Node: update_memory_node  (runs at the end of every path)
#
# NOTE: this version manually appends a plain dict to a plain Python list,
# which is correct for THIS file's dependency-free local simulation. The
# notebook's real LangGraph version instead returns
# `{"messages": [AIMessage(content=state["response"])], ...}` and lets the
# `add_messages` reducer append it — see architecture.md Section 4 — rather
# than reconstructing the whole list by hand, which is what a reducer is for.
# ---------------------------------------------------------------------------
def update_memory_node(state: dict, session_id: str = "default-session", safety: Optional[SafetyRuntime] = None) -> dict:
    new_entities = dict(state.get("session_entities", {"company": None, "focus": None}))
    if state["resolved_company"]:
        new_entities["company"] = state["resolved_company"]
    if state["resolved_focus"]:
        new_entities["focus"] = state["resolved_focus"]

    messages = state["messages"] + [{"role": "assistant", "content": state["response"]}]

    # L_audit — one HMAC-chained, metadata-only record per turn (see
    # safety_layer.Store). Never the raw question/answer text or a PII
    # value: only classification, flags, PII *type* counts, and any
    # review id. Runs on EVERY path (this node is the universal sink —
    # see this function's position in graph_sim.run_turn), so a
    # blank/gibberish/injection/action-request turn is audited exactly
    # like a normal research turn. Same "bookkeeping, not a decision"
    # exception as action_preview_node / human_review_gate_node above:
    # this node's actual job (updating session_entities/messages) must
    # still complete and return even if the audit-trail write itself
    # fails — losing one audit record is a far smaller problem than
    # losing session memory / the turn's reply on every guardrail-
    # flagged turn.
    safety = safety or SafetyRuntime.default()
    try:
        safety.store.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "owner": safety.store.owner(session_id),
            "input_classification": state.get("input_classification"),
            "resolved_company": state.get("resolved_company"),
            "pii_types": sorted((state.get("pii_audit") or {}).keys()),
            "flags": state.get("flags", []),
            "action": (state.get("action_request") or {}).get("action"),
            "review_id": state.get("review_id"),
        })
    except Exception:
        pass
    return {"session_entities": new_entities, "messages": messages}''',
    '''# ---------------------------------------------------------------------------
# Node: update_memory_node  (runs at the end of every path)
#
# This version differs from node_logic.py's in two ways, both required by
# the REAL LangGraph app rather than node_logic.py's dependency-free local
# simulation: (1) it returns a single `AIMessage` and lets the
# `add_messages` reducer append it — see architecture.md Section 4 —
# rather than rebuilding the whole plain-dict list by hand; (2) it takes
# `(state, config)`, LangGraph's documented two-argument node signature
# <https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes>,
# and reads `session_id` from `config["configurable"]["thread_id"]` --
# the SAME id already used to key session memory, so one id now also
# owns this turn's audit record and any review-queue entry. (Unverified
# LIVE in this sandbox, which cannot install langgraph -- see this
# project's established pattern of disclosing what's verified against
# docs/stable behavior vs. actually run, e.g. yahoo_finance_market_data.py.)
# ---------------------------------------------------------------------------
def update_memory_node(state: dict, config: dict) -> dict:
    new_entities = dict(state.get("session_entities", {"company": None, "focus": None}))
    if state["resolved_company"]:
        new_entities["company"] = state["resolved_company"]
    if state["resolved_focus"]:
        new_entities["focus"] = state["resolved_focus"]

    session_id = config["configurable"]["thread_id"]
    try:
        safety.store.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "owner": safety.store.owner(session_id),
            "input_classification": state.get("input_classification"),
            "resolved_company": state.get("resolved_company"),
            "pii_types": sorted((state.get("pii_audit") or {}).keys()),
            "flags": state.get("flags", []),
            "action": (state.get("action_request") or {}).get("action"),
            "review_id": state.get("review_id"),
        })
    except Exception as exc:
        logger.warning("Audit-trail write failed (turn still answered)", extra={
            "session_id": session_id, "error": str(exc),
        })

    # `add_messages` (declared on ResearchState.messages) appends this single
    # new message to the persisted history — no need to rebuild the list.
    return {"session_entities": new_entities, "messages": [AIMessage(content=state["response"])]}'''
)

APP_SOURCE = f'''"""
investment_research_api.py
---------------------------
Standalone FastAPI service wrapping the Investment Research Assistant
LangGraph agent — the same tested logic as Investment_Research_Assistant.ipynb
(tools_mock.py, guardrails.py, schemas.py, corpus.py, node_logic.py), packaged
as an importable ASGI app so uvicorn can serve it directly.

Run standalone:
    uvicorn investment_research_api:app --host 0.0.0.0 --port 8000

Or let the notebook's deployment section (Section 13) start it in a
background thread and expose it via an ngrok public URL — see
architecture.md, "Deployment" section, for the full request path and the
honest limitations of this setup (single-process in-memory checkpointer,
ephemeral free-tier ngrok URL, no auth on the debug endpoint).

Configuration (environment variables, all optional with sane defaults):
    MOCK_LLM            "true"/"false" (default "false") — offline heuristic
                        LLM stand-in AND a dependency-free keyword-overlap
                        retriever (skips embeddings/FAISS entirely), no API
                        key or network call needed, for free smoke tests.
    LLM_PROVIDER        e.g. "openai:gpt-4o-mini" (default) or
                        "anthropic:claude-3-5-haiku-latest" — only used
                        when MOCK_LLM is false, for the CHAT model. RAG
                        embeddings always use OpenAI's API regardless of
                        this setting (Anthropic has no public embeddings
                        endpoint), so OPENAI_API_KEY is required whenever
                        MOCK_LLM is false, even with an Anthropic chat
                        model — ANTHROPIC_API_KEY is only needed in
                        addition, for the chat calls themselves.
    IRA_SCORE_THRESHOLD RAG similarity threshold (default 0.35).
    IRA_CORPUS_DIR      Where the synthetic knowledge base is written
                        (default /tmp/investment_research_kb).
    TWELVEDATA_API_KEY  Optional. Enables live price data (and, on a paid
                        plan, profile) for a curated set of REAL companies
                        (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA — see
                        real_market_data.REAL_COMPANIES) alongside the
                        synthetic ABC/XYZ/DEF universe. Free signup at
                        https://twelvedata.com/pricing. Without it, its
                        calls fail cleanly (missing_twelvedata_api_key)
                        rather than fabricating data.
    (Yahoo Finance, the second real-company market-data provider, needs
    NO API key at all -- see yahoo_finance_market_data.py. It fills in
    company profile and real financial-statement data that Twelve Data's
    free plan can't provide at all, independent of whether
    TWELVEDATA_API_KEY is set; ABC/XYZ/DEF are unaffected either way.)
    SAFETY_AUDIT_DB_PATH Optional, default "data/safety_audit.db". The
                        guardrail layer's tamper-evident audit log +
                        human-review queue (see safety_layer.py) -- a
                        local SQLite file; never raw question/answer
                        text or a PII value, only metadata.

All of the above are read from the process environment — never
hardcoded in this file. Locally (or in Vocareum), copy `.env.example` to
`.env` and fill in real values; `load_dotenv()` below picks it up. On a
real host (Render, Cloud Run, Docker, ...), set them as actual platform
environment variables instead — `load_dotenv()` only fills in a value
that ISN'T already set, so it never overrides what the platform injects,
and does nothing at all if no `.env` file is present (e.g. the deployed
container, where `.env` is intentionally never committed — see
.gitignore).
"""

import os
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Annotated, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field as PydanticField

load_dotenv()  # no-op if no .env file is present (e.g. on Render) — see
                # the module docstring's Configuration section above.

# =============================================================================
# Structured JSON logging — a deployed service should never rely on print().
# JSON lines are parseable by Datadog, CloudWatch, GCP Logging, etc.
# =============================================================================
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {{
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }}
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in log_entry:
                log_entry[key] = value
        # record.exc_info (when logger.error(..., exc_info=True) is used)
        # is a raw (type, value, traceback) tuple -- the generic loop above
        # would otherwise dump it as an unreadable object repr via
        # json.dumps(default=str). Render it as actual traceback TEXT
        # instead, so a Render Logs search for an error message actually
        # shows what broke, not just that something did.
        if record.exc_info:
            log_entry["traceback"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = get_logger("investment_research_assistant")

# =============================================================================
# Mock tools — identical to tools_mock.py (see that file / architecture.md
# Section 6 for the full rationale; four required mock tools with realistic,
# structured failure modes instead of hard-coded final answers).
# =============================================================================
{load_and_clean("tools_mock.py")}

# =============================================================================
# Real-company market data, provider 1 (Twelve Data) — identical to
# real_market_data.py. Sits alongside the synthetic tools above. Set
# TWELVEDATA_API_KEY to enable it (free signup at
# https://twelvedata.com/pricing); without it, its calls fail cleanly
# rather than fabricating data. Reliable for live price; its free plan
# can't serve company profile or any real financial-statement data at
# all -- see provider 2 below.
# =============================================================================
{load_and_clean("real_market_data.py")}

# =============================================================================
# Real-company market data, provider 2 (Yahoo Finance, via the yfinance
# package) — identical to yahoo_finance_market_data.py. Closes the exact
# gap provider 1's free plan leaves: company profile and real
# financial-statement data (revenue, margins, YoY growth). No API key, no
# signup, no published rate limit -- unofficial/undocumented endpoints
# though, see the module's own docstring for the honest trade-off.
# Always available regardless of whether TWELVEDATA_API_KEY is set.
# =============================================================================
{load_and_clean("yahoo_finance_market_data.py")}

# =============================================================================
# Combines both providers above, field by field — identical to
# merged_market_data.py. `tools_node` below calls THESE functions
# (get_merged_stock_price / get_merged_company_profile /
# get_merged_company_financials), not either single-provider module's
# functions directly. See merged_market_data.py's module docstring for
# the full merge strategy.
# =============================================================================
{load_and_clean("merged_market_data.py")}

# =============================================================================
# Guardrails — identical to guardrails.py (deterministic blank / injection /
# gibberish-heuristic checks; see architecture.md Section 7).
# =============================================================================
{load_and_clean("guardrails.py")}

# =============================================================================
# Guardrail layer ported from the Week 18 Ethics/Safety/Governance notebook
# (as hardened in the SafeBank AI capstone) — identical to safety_layer.py.
# node_logic.py below imports from this module, so it must be defined
# first. See safety_layer.py's module docstring and architecture.md's
# guardrail section for the full mapping onto this agent's pipeline.
# =============================================================================
{load_and_clean("safety_layer.py")}

SAFETY_AUDIT_DB_PATH = os.environ.get("SAFETY_AUDIT_DB_PATH", "data/safety_audit.db")
safety = SafetyRuntime(store=Store(SAFETY_AUDIT_DB_PATH))

# =============================================================================
# Structured-output schemas — identical to schemas.py.
# =============================================================================
{load_and_clean("schemas.py")}

# =============================================================================
# Synthetic knowledge base — identical to corpus.py, including the
# deliberately poisoned document used to test retrieved-content injection
# defense (see architecture.md Section 5).
# =============================================================================
{load_and_clean("corpus.py")}

# =============================================================================
# MOCK_LLM / LLM_PROVIDER env vars — read early (before the vector store
# section below) because retrieval strategy now depends on MOCK_LLM too,
# not just which chat LLM gets used.
# =============================================================================
{load_and_clean("mock_llm.py")}

MOCK_LLM = os.environ.get("MOCK_LLM", "false").strip().lower() == "true"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai:gpt-4o-mini")
# Optional custom OpenAI-compatible endpoint — e.g. a Vocareum proxy URL
# instead of api.openai.com, for courses that issue Vocareum-hosted keys
# rather than a personal OpenAI key. Used below by BOTH the embeddings
# client and the chat LLM client.
_openai_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

# =============================================================================
# Vector store.
#
# Embeddings: OpenAI's API (`text-embedding-3-small`), not a local
# HuggingFace/sentence-transformers model. This was a deliberate change —
# the original design used a free local model specifically so RAG needed
# no API key. In practice, loading sentence-transformers pulls in
# `torch`+`transformers` (plus, on a default pip install, a full set of
# unused CUDA packages), which alone exceeds Render's free-tier 512MB RAM
# cap at the "build the vector store" step (see architecture.md's
# Troubleshooting section for the exact OOM this caused). OpenAI embeddings
# trade "needs an API key + network" for "needs ~0 extra RAM and ~0 extra
# install size" — worth it for a service meant to run on a memory-capped
# host. This does mean OPENAI_API_KEY (and OPENAI_BASE_URL, if you're on a
# Vocareum-style proxy) is now required even if LLM_PROVIDER is set to an
# Anthropic model — Anthropic has no public embeddings endpoint, so RAG
# always calls OpenAI's regardless of which model answers the chat turn.
#
# MOCK_LLM=True skips this branch entirely and uses a dependency-free
# keyword-overlap retriever instead (same one test_harness.py's local
# simulation uses) — zero network calls, zero API key, for a genuinely
# free/offline smoke test of the graph's control flow.
# =============================================================================
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CORPUS_DIR = os.environ.get("IRA_CORPUS_DIR", "/tmp/investment_research_kb")
write_corpus_to_disk(CORPUS_DIR)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    company: Optional[str]
    score: float


def _load_documents_as_langchain_docs():
    docs = []
    for meta in DOCUMENTS:
        docs.append(Document(
            page_content=meta["text"],
            metadata={{"source": meta["filename"], "company": meta["company"], "doc_type": meta["doc_type"]}},
        ))
    return docs


class _LocalKeywordRetriever:
    """Dependency-free stand-in used only when MOCK_LLM=True — scores by
    keyword overlap instead of vector similarity. Same
    `.retrieve(query, company_filter, k)` interface as `FAISSRetriever`
    below, so `rag_node` doesn't need to know which one it's calling."""

    def __init__(self, score_threshold: float = 0.12):
        self.score_threshold = score_threshold
        self.documents = DOCUMENTS

    def retrieve(self, query: str, company_filter: Optional[str] = None, k: int = 4) -> List[RetrievedChunk]:
        q_words = set(w.lower() for w in query.split() if len(w) > 2)
        out = []
        for doc in self.documents:
            if company_filter and doc["company"] not in (company_filter, None):
                continue
            d_words = set(w.lower().strip(".,:;()") for w in doc["text"].split() if len(w) > 2)
            score = (len(q_words & d_words) / len(q_words)) if q_words else 0.0
            if score >= self.score_threshold:
                out.append(RetrievedChunk(text=doc["text"], source=doc["filename"], company=doc["company"], score=score))
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:k]


class FAISSRetriever:
    """Same `.retrieve(query, company_filter, k)` interface used throughout
    — filters by company in Python rather than relying on a specific
    LangChain version's native FAISS metadata-filter argument.
    """

    def __init__(self, vectorstore: FAISS, score_threshold: float):
        self.vectorstore = vectorstore
        self.score_threshold = score_threshold

    def retrieve(self, query: str, company_filter: Optional[str] = None, k: int = 4) -> List[RetrievedChunk]:
        raw_hits = self.vectorstore.similarity_search_with_score(query, k=max(k * 4, 12))
        out = []
        for doc, distance in raw_hits:
            doc_company = doc.metadata.get("company")
            if company_filter and doc_company not in (company_filter, None):
                continue
            # Cast to a plain Python float: FAISS returns `distance` as
            # numpy.float32, which otherwise flows into retrieved_docs and
            # breaks LangGraph's msgpack-based checkpoint serializer
            # (`TypeError: Type is not msgpack serializable: numpy.float32`).
            similarity = float(1.0 / (1.0 + distance))
            if similarity < self.score_threshold:
                continue
            out.append(RetrievedChunk(
                text=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                company=doc_company,
                score=similarity,
            ))
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:k]


if MOCK_LLM:
    retriever = _LocalKeywordRetriever()
    logger.info("Retrieval mode: MOCK (keyword-overlap, no embeddings/FAISS — offline, zero API key)")
else:
    logger.info("Building vector store (OpenAI embeddings)...")
    from langchain_openai import OpenAIEmbeddings

    _embedding_kwargs = {{}}
    if _openai_base_url:
        _embedding_kwargs["base_url"] = _openai_base_url
    _embeddings = OpenAIEmbeddings(model="text-embedding-3-small", **_embedding_kwargs)
    _splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    _chunked_docs = _splitter.split_documents(_load_documents_as_langchain_docs())
    _vectorstore = FAISS.from_documents(_chunked_docs, _embeddings)

    SCORE_THRESHOLD = float(os.environ.get("IRA_SCORE_THRESHOLD", "0.35"))
    retriever = FAISSRetriever(_vectorstore, SCORE_THRESHOLD)
    logger.info("Vector store ready", extra={{"chunks": len(_chunked_docs)}})

# =============================================================================
# Chat LLM.
# =============================================================================
if MOCK_LLM:
    llm = MockChatModel()
    logger.info("LLM mode: MOCK (offline wiring test, no API key used)")
else:
    from langchain.chat_models import init_chat_model

    _llm_kwargs = {{}}
    if _openai_base_url and LLM_PROVIDER.startswith("openai"):
        _llm_kwargs["base_url"] = _openai_base_url

    llm = init_chat_model(LLM_PROVIDER, temperature=0, **_llm_kwargs)
    logger.info("LLM mode: real", extra={{"provider": LLM_PROVIDER, "custom_base_url": bool(_openai_base_url)}})

# =============================================================================
# LangGraph state schema and node functions — identical to node_logic.py,
# adapted the same way as the notebook (Section 8): `update_memory_node`
# returns a single new AIMessage for the `add_messages` reducer to append.
# =============================================================================
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage


class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    session_entities: Dict[str, Any]

    user_input: str
    input_classification: Optional[str]
    resolved_company: Optional[str]
    resolved_focus: Optional[str]
    is_comparison: bool
    compare_to: Optional[str]
    is_fabrication_request: bool
    need_clarification: bool
    clarifying_question: Optional[str]
    retrieved_docs: List[dict]
    tool_results: Dict[str, Any]
    rag_flags: List[str]
    tool_flags: List[str]
    flags: List[str]
    brief: Optional[dict]
    response: Optional[str]
    # -- guardrail-layer fields (see safety_layer.py) --
    pii_audit: Dict[str, int]
    action_request: Optional[dict]
    action_preview: Optional[dict]
    review_id: Optional[str]


{node_logic_src}

# =============================================================================
# Build and compile the graph — identical wiring to the notebook's Section 9.
# =============================================================================
_builder = StateGraph(ResearchState)

_builder.add_node("input_guard", input_guard)
_builder.add_node("blank_node", blank_node)
_builder.add_node("gibberish_node", gibberish_node)
_builder.add_node("injection_node", injection_node)
_builder.add_node("intent_router", lambda state: intent_router(state, llm))
_builder.add_node("refusal_node", refusal_node)
_builder.add_node("clarification_node", clarification_node)
_builder.add_node("fan_out", lambda state: {{}})
_builder.add_node("rag_node", lambda state: rag_node(state, retriever))
_builder.add_node("tools_node", tools_node)
_builder.add_node("reconcile_node", reconcile_node)
_builder.add_node("synthesize_brief_node", lambda state: synthesize_brief_node(state, llm))
# L8 / L5 — guardrail-layer nodes (see safety_layer.py and node_logic.py's
# module docstring). Wired via a lambda, same pattern as intent_router/
# rag_node/synthesize_brief_node above, to supply the module-level
# `safety` runtime; `session_id` comes from `config["configurable"]
# ["thread_id"]` — the SAME id MemorySaver already keys session memory
# by, per LangGraph's documented (state, config) node signature
# <https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes>
# (unverified LIVE in this sandbox — see update_memory_node's comment
# above for why).
_builder.add_node("action_preview_node", lambda state, config: action_preview_node(
    state, config["configurable"]["thread_id"], safety))
_builder.add_node("human_review_gate_node", lambda state, config: human_review_gate_node(
    state, config["configurable"]["thread_id"], safety))
# update_memory_node is registered through a lambda too, NOT passed directly
# like `_builder.add_node("update_memory_node", update_memory_node)` --
# confirmed via a real production traceback (LangGraph 1.2.16) that passing
# it directly raises `TypeError: update_memory_node() missing 1 required
# positional argument: 'config'` on every invocation. Root cause: LangGraph
# 1.x's add_node decides whether to inject `config` by inspecting the
# second parameter's TYPE ANNOTATION, not just its arity -- a plain `def`
# with `config: dict` (as this file types it, for readability) isn't
# recognized as wanting RunnableConfig, so LangGraph calls it with only
# `state`. The untyped lambdas above sidestep this entirely: LangGraph
# falls back to counting positional parameters for an unannotated
# callable, which correctly detects 2 and passes config -- same escape
# hatch used here, rather than importing and annotating with the exact
# RunnableConfig type LangGraph 1.x expects (unverified which import path/
# shape that needs in this sandbox, which cannot install langgraph at all).
_builder.add_node("update_memory_node", lambda state, config: update_memory_node(state, config))

_builder.add_edge(START, "input_guard")
_builder.add_conditional_edges("input_guard", route_after_input_guard, {{
    "blank_node": "blank_node",
    "gibberish_node": "gibberish_node",
    "injection_node": "injection_node",
    "action_preview_node": "action_preview_node",
    "intent_router": "intent_router",
}})
_builder.add_conditional_edges("intent_router", route_after_intent, {{
    "gibberish_node": "gibberish_node",
    "refusal_node": "refusal_node",
    "clarification_node": "clarification_node",
    "proceed": "fan_out",
}})
_builder.add_edge("fan_out", "rag_node")
_builder.add_edge("fan_out", "tools_node")
_builder.add_edge("rag_node", "reconcile_node")
_builder.add_edge("tools_node", "reconcile_node")
_builder.add_edge("reconcile_node", "synthesize_brief_node")
# L5 — every research-path answer passes through the human-review gate
# (a no-op unless this turn's flags warrant a review — see
# safety_layer.flags_warrant_review) before memory/audit update.
_builder.add_edge("synthesize_brief_node", "human_review_gate_node")

for _terminal in ["blank_node", "gibberish_node", "injection_node", "refusal_node",
                   "clarification_node", "action_preview_node", "human_review_gate_node"]:
    _builder.add_edge(_terminal, "update_memory_node")

_builder.add_edge("update_memory_node", END)

research_graph = _builder.compile(checkpointer=MemorySaver())
logger.info("LangGraph compiled", extra={{"nodes": len(research_graph.get_graph().nodes)}})

# =============================================================================
# FastAPI application
# =============================================================================
app = FastAPI(
    title="Investment Research Assistant API",
    description="LangGraph-powered RAG + mock-tool investment research agent.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Performance: compress JSON responses over the wire (meaningful for
# /debug/{{session_id}}, which can return several retrieved_docs chunks
# plus full tool_results) — cheap CPU cost on a tiny payload, real
# bandwidth/latency win over a public ngrok tunnel. 500-byte floor so
# tiny responses (like /health) aren't compressed for no benefit.
app.add_middleware(GZipMiddleware, minimum_size=500)


class ChatRequest(BaseModel):
    session_id: str = PydanticField(
        ..., description="Stable id per conversation; reuse it across calls for session memory.",
        json_schema_extra={{"example": "investor_123"}},
    )
    message: str = PydanticField(
        ..., description="The research question. An empty string is valid input — the agent's own guardrail asks for a request rather than erroring.",
        json_schema_extra={{"example": "Research ABC Technologies: revenue growth and profitability."}},
    )
    consent: bool = PydanticField(
        default=True,
        description="L2 guardrail — must be true to proceed (see safety_layer.SafetyGate). Defaults to true "
                     "so existing clients (gradio_app.py, react-ui/) built before this guardrail layer keep "
                     "working unchanged; a UI that surfaces its own consent checkbox (streamlit_app.py) sets it explicitly.",
    )
    stopped: bool = PydanticField(
        default=False,
        description="L2 guardrail — this session's kill switch. true blocks the turn before the graph runs at all.",
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str
    latency_ms: float
    flags: List[str] = []
    blocked: bool = False
    review_id: Optional[str] = None
    action_preview: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    mock_llm: bool
    llm_provider: str


class DebugStateResponse(BaseModel):
    session_id: str
    resolved_company: Optional[str] = None
    flags: List[str] = []
    tool_results: Dict[str, Any] = {{}}
    retrieved_docs: List[dict] = []
    pii_audit: Dict[str, int] = {{}}
    action_preview: Optional[dict] = None
    review_id: Optional[str] = None


class ReviewItem(BaseModel):
    id: str
    reason: str
    status: str
    created: float


class ReviewDecisionRequest(BaseModel):
    outcome: str = PydanticField(..., description="SIMULATED_APPROVED or SIMULATED_REJECTED — see safety_layer.Store.review.")


class AuditIntegrityResponse(BaseModel):
    verified: bool


SERVER_START_TIME = time.time()


@app.get("/", include_in_schema=False)
def root():
    """The bare host/tunnel URL has no content of its own — redirect human
    visitors to the interactive API docs instead of returning a bare 404,
    which is otherwise the first (confusing) thing anyone sees after
    opening the ngrok URL in a browser."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health_check():
    """Used by uptime checks and, if you later move this off ngrok, by a
    load balancer / Render / Cloud Run health probe."""
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - SERVER_START_TIME, 2),
        mock_llm=MOCK_LLM,
        llm_provider=LLM_PROVIDER,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(request: ChatRequest):
    """Main agent endpoint. Pass the same `session_id` across calls to get
    session memory (the graph's `session_entities` + message history are
    keyed by this id via the LangGraph checkpointer's thread_id).

    L2 guardrail (safety_layer.SafetyGate) runs BEFORE the graph: a
    blocked request (no consent, kill switch on, blank, oversized, or
    rate-limited) never reaches the LangGraph app at all, and is still
    audited — see safety.store.append below.
    """
    logger.info("Incoming request", extra={{"session_id": request.session_id, "message_length": len(request.message)}})
    start_time = time.time()

    block = safety.gate.check(request.message, request.session_id, request.consent, request.stopped)
    if block is not None:
        # The block decision itself (SafetyGate.check, pure in-memory) has
        # already happened -- logging it is bookkeeping, so a Store/SQLite
        # failure here must not turn an already-decided block into a 500.
        try:
            safety.store.append({{
                "time": datetime.now(timezone.utc).isoformat(),
                "owner": safety.store.owner(request.session_id),
                "input_classification": f"blocked:{{block['reason']}}",
                "flags": [f"safety_gate_blocked:{{block['reason']}}"],
            }})
        except Exception as audit_exc:
            logger.warning("Audit-trail write failed (block still returned)", extra={{
                "session_id": request.session_id, "error": str(audit_exc),
            }})
        return ChatResponse(
            session_id=request.session_id,
            response=block["message"],
            latency_ms=round((time.time() - start_time) * 1000, 2),
            flags=[f"safety_gate_blocked:{{block['reason']}}"],
            blocked=True,
        )

    try:
        # L1 — privacy tokenization happens HERE, at the API's entry point,
        # rather than inside a graph node: `research_graph.invoke`'s
        # initial state (including the HumanMessage the `add_messages`
        # reducer will persist into session memory) is built from the
        # MASKED text, so no raw PII ever enters the graph, an LLM call,
        # or the checkpointer's stored history. node_logic.py's own
        # `start_turn` does the equivalent masking for the dependency-free
        # local simulation (graph_sim.run_turn) — see that function's
        # docstring; this is the same control, applied at the real API's
        # actual entry point instead, since the real LangGraph app never
        # calls `start_turn` (LangGraph builds its initial state directly
        # from this endpoint's `.invoke()` call, not from that helper).
        masked_message, _pii_map = safety.masker.mask(request.message)
        config = {{"configurable": {{"thread_id": request.session_id}}}}
        result = research_graph.invoke(
            {{
                "messages": [HumanMessage(content=masked_message)],
                "user_input": masked_message,
                "pii_audit": safety.masker.audit_summary(_pii_map),
            }},
            config=config,
        )
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info("Request completed", extra={{
            "session_id": request.session_id, "latency_ms": latency_ms, "flags": result.get("flags", []),
        }})
        return ChatResponse(
            session_id=request.session_id,
            response=result["response"],
            latency_ms=latency_ms,
            flags=result.get("flags", []),
            blocked=result.get("input_classification") == "injection",
            review_id=result.get("review_id"),
            action_preview=result.get("action_preview"),
        )
    except Exception as e:
        # exc_info=True so the FULL traceback lands in Render's logs, not
        # just str(e) -- this is what you should grep your deploy's Logs
        # tab for ("Agent invocation failed") if you ever see a 500 here.
        logger.error(
            "Agent invocation failed",
            extra={{"session_id": request.session_id, "error": str(e)}},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Investment Research Assistant is temporarily unavailable. Please try again.",
        )


@app.get("/review/{{session_id}}", response_model=List[ReviewItem], tags=["Operations"])
def list_review_queue(session_id: str):
    """L5 guardrail — this session's human-review queue (see
    safety_layer.Store.queue). NOT authenticated, same caveat as
    /debug/{{session_id}} above."""
    return [ReviewItem(**item) for item in safety.store.queue(session_id)]


@app.post("/review/{{session_id}}/{{review_id}}", tags=["Operations"])
def decide_review(session_id: str, review_id: str, request: ReviewDecisionRequest):
    """Records a SIMULATED review outcome only — see
    safety_layer.Store.review. Never changes a watchlist, alert, or
    order; this sandbox has no code path that executes anything."""
    try:
        safety.store.review(session_id, review_id, request.outcome)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {{"status": "ok"}}


@app.get("/audit/verify", response_model=AuditIntegrityResponse, tags=["Operations"])
def audit_verify():
    """L_audit — verifies the HMAC-chained audit log hasn't been edited
    since it was written (see safety_layer.Store.verify)."""
    return AuditIntegrityResponse(verified=safety.store.verify())


@app.get("/debug/{{session_id}}", response_model=DebugStateResponse, tags=["Operations"])
def debug_state(session_id: str):
    """Inspect the flags / tool results / retrieved chunks behind a
    session's most recent reply — the API equivalent of the notebook's
    Gradio transparency panel. NOT authenticated: fine for a classroom demo
    behind an ngrok URL only you have, but see architecture.md's
    "Deployment" section before exposing this more broadly."""
    config = {{"configurable": {{"thread_id": session_id}}}}
    snapshot = research_graph.get_state(config).values
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"No session found for session_id={{session_id!r}}")
    return DebugStateResponse(
        session_id=session_id,
        resolved_company=snapshot.get("resolved_company"),
        flags=snapshot.get("flags", []),
        tool_results=snapshot.get("tool_results", {{}}),
        retrieved_docs=snapshot.get("retrieved_docs", []),
        pii_audit=snapshot.get("pii_audit", {{}}),
        action_preview=snapshot.get("action_preview"),
        review_id=snapshot.get("review_id"),
    )
'''

with open("investment_research_api.py", "w") as f:
    f.write(APP_SOURCE)

try:
    # ast.parse() alone has a blind spot: it does NOT enforce that
    # `from __future__ import ...` appears only at the top of the file
    # (that rule lives in the bytecode compiler, not the parser) -- a real
    # bug this project hit once a second module (safety_layer.py) got
    # inlined mid-file. compile() with mode="exec" runs the same checks
    # python itself runs on import, so it catches that class of error too.
    ast.parse(APP_SOURCE)
    compile(APP_SOURCE, "investment_research_api.py", "exec")
    print("investment_research_api.py written and syntax-verified.")
    print(f"Lines: {len(APP_SOURCE.splitlines())}")
except SyntaxError as e:
    raise RuntimeError(f"investment_research_api.py has a syntax error at line {e.lineno}: {e.msg}")
