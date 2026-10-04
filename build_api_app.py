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

LOCAL_MODULES = ["guardrails", "schemas", "tools_mock", "corpus", "node_logic", "mock_llm", "local_retriever", "real_market_data"]


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
def update_memory_node(state: dict) -> dict:
    new_entities = dict(state.get("session_entities", {"company": None, "focus": None}))
    if state["resolved_company"]:
        new_entities["company"] = state["resolved_company"]
    if state["resolved_focus"]:
        new_entities["focus"] = state["resolved_focus"]

    messages = state["messages"] + [{"role": "assistant", "content": state["response"]}]
    return {"session_entities": new_entities, "messages": messages}''',
    '''# ---------------------------------------------------------------------------
# Node: update_memory_node  (runs at the end of every path)
# ---------------------------------------------------------------------------
def update_memory_node(state: dict) -> dict:
    new_entities = dict(state.get("session_entities", {"company": None, "focus": None}))
    if state["resolved_company"]:
        new_entities["company"] = state["resolved_company"]
    if state["resolved_focus"]:
        new_entities["focus"] = state["resolved_focus"]

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
                        LLM stand-in, no API key needed, for free smoke tests.
    LLM_PROVIDER        e.g. "openai:gpt-4o-mini" (default) or
                        "anthropic:claude-3-5-haiku-latest". Only used when
                        MOCK_LLM is false, and only then does an API key
                        (OPENAI_API_KEY / ANTHROPIC_API_KEY) need to be set.
    IRA_SCORE_THRESHOLD RAG similarity threshold (default 0.35).
    IRA_CORPUS_DIR      Where the synthetic knowledge base is written
                        (default /tmp/investment_research_kb).
    TWELVEDATA_API_KEY  Optional. Enables live data for a curated set of
                        REAL companies (AAPL, MSFT, GOOGL, AMZN, TSLA,
                        NVDA — see real_market_data.REAL_COMPANIES)
                        alongside the synthetic ABC/XYZ/DEF universe. Free
                        signup at https://twelvedata.com/pricing. Without
                        it, real-company tool calls fail cleanly
                        (missing_twelvedata_api_key) rather than
                        fabricating data — ABC/XYZ/DEF are unaffected
                        either way.

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
# Real-company market data (Twelve Data) — identical to real_market_data.py.
# Sits alongside the synthetic tools above; `tools_node` dispatches to
# whichever module matches the resolved ticker. Set TWELVEDATA_API_KEY to
# enable it (free signup at https://twelvedata.com/pricing); without it,
# real-company tool calls fail cleanly rather than fabricating data.
# =============================================================================
{load_and_clean("real_market_data.py")}

# =============================================================================
# Guardrails — identical to guardrails.py (deterministic blank / injection /
# gibberish-heuristic checks; see architecture.md Section 7).
# =============================================================================
{load_and_clean("guardrails.py")}

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
# Vector store (FAISS + free local HuggingFace embeddings — no API key
# needed for RAG regardless of which chat LLM is configured below).
# =============================================================================
from langchain_huggingface import HuggingFaceEmbeddings
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


logger.info("Building vector store...")
_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
_chunked_docs = _splitter.split_documents(_load_documents_as_langchain_docs())
_vectorstore = FAISS.from_documents(_chunked_docs, _embeddings)

SCORE_THRESHOLD = float(os.environ.get("IRA_SCORE_THRESHOLD", "0.35"))


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


retriever = FAISSRetriever(_vectorstore, SCORE_THRESHOLD)
logger.info("Vector store ready", extra={{"chunks": len(_chunked_docs)}})

# =============================================================================
# LLM — MOCK_LLM / LLM_PROVIDER env vars, same toggle as the notebook.
# =============================================================================
{load_and_clean("mock_llm.py")}

MOCK_LLM = os.environ.get("MOCK_LLM", "false").strip().lower() == "true"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai:gpt-4o-mini")

if MOCK_LLM:
    llm = MockChatModel()
    logger.info("LLM mode: MOCK (offline wiring test, no API key used)")
else:
    from langchain.chat_models import init_chat_model

    # Optional custom OpenAI-compatible endpoint — e.g. a Vocareum proxy URL
    # instead of api.openai.com, for courses that issue Vocareum-hosted keys
    # rather than a personal OpenAI key. Only applied for openai-family
    # providers, and only when actually set via .env / the environment.
    _openai_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
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
_builder.add_node("update_memory_node", update_memory_node)

_builder.add_edge(START, "input_guard")
_builder.add_conditional_edges("input_guard", route_after_input_guard, {{
    "blank_node": "blank_node",
    "gibberish_node": "gibberish_node",
    "injection_node": "injection_node",
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

for _terminal in ["blank_node", "gibberish_node", "injection_node", "refusal_node",
                   "clarification_node", "synthesize_brief_node"]:
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


class ChatResponse(BaseModel):
    session_id: str
    response: str
    latency_ms: float
    flags: List[str] = []


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
    keyed by this id via the LangGraph checkpointer's thread_id)."""
    logger.info("Incoming request", extra={{"session_id": request.session_id, "message_length": len(request.message)}})
    start_time = time.time()
    try:
        config = {{"configurable": {{"thread_id": request.session_id}}}}
        result = research_graph.invoke(
            {{"messages": [HumanMessage(content=request.message)], "user_input": request.message}},
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
        )
    except Exception as e:
        logger.error("Agent invocation failed", extra={{"session_id": request.session_id, "error": str(e)}})
        raise HTTPException(
            status_code=500,
            detail="Investment Research Assistant is temporarily unavailable. Please try again.",
        )


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
    )
'''

with open("investment_research_api.py", "w") as f:
    f.write(APP_SOURCE)

try:
    ast.parse(APP_SOURCE)
    print("investment_research_api.py written and syntax-verified.")
    print(f"Lines: {len(APP_SOURCE.splitlines())}")
except SyntaxError as e:
    raise RuntimeError(f"investment_research_api.py has a syntax error at line {e.lineno}: {e.msg}")
