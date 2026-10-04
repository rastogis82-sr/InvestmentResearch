"""
tools_mock.py
-------------
Standalone smoke-test version of the four mock financial-data tools used by
the Investment Research Assistant. This file has ZERO third-party
dependencies on purpose, so it can be executed directly in this sandbox
(which has no outbound package installation) to verify the tool logic,
the mock dataset, and the simulated failure modes before they are wired
into LangChain @tool wrappers inside the Colab notebook.

The versions of these functions that ship in the notebook are identical
in behaviour; the notebook versions are simply decorated with LangChain's
`@tool` so the LLM can call them, and return JSON-serializable dicts.
"""

import random

# ---------------------------------------------------------------------------
# Mock dataset. In a real system this would be a call to Bloomberg/Refinitiv/
# a market-data vendor. Here it is a small static lookup table so the whole
# assignment is reproducible and gradeable without any external API keys.
# ---------------------------------------------------------------------------
_COMPANY_DB = {
    "ABC": {
        "name": "ABC Technologies",
        "sector": "Enterprise Software",
        "industry": "Cloud Applications",
        "hq": "Bengaluru, India",
        "description": (
            "ABC Technologies builds cloud-based CRM and workflow-automation "
            "software for mid-market enterprises."
        ),
        "financials": {
            "fiscal_year": "FY2026",
            "revenue_usd_m": 842.0,
            "revenue_growth_yoy_pct": 18.4,
            "net_income_usd_m": 96.0,
            "net_margin_pct": 11.4,
            "gross_margin_pct": 71.2,
        },
        "price": {"last_price_usd": 128.40, "day_change_pct": 1.35},
    },
    "XYZ": {
        "name": "XYZ Corp",
        "sector": "Enterprise Software",
        "industry": "Cloud Applications",
        "hq": "Austin, USA",
        "description": (
            "XYZ Corp provides collaboration and project-management SaaS "
            "tools for distributed teams."
        ),
        "financials": {
            "fiscal_year": "FY2026",
            "revenue_usd_m": 1210.0,
            "revenue_growth_yoy_pct": 9.7,
            "net_income_usd_m": 68.0,
            "net_margin_pct": 5.6,
            "gross_margin_pct": 68.9,
        },
        "price": {"last_price_usd": 74.10, "day_change_pct": -0.62},
    },
    # DEF Industries deliberately has tool data but NO document in corpus.py's
    # DOCUMENTS list — this is what makes the RAG-failure test case
    # ("no relevant internal research is retrieved") realistic and
    # reproducible rather than simulated by disabling the retriever.
    "DEF": {
        "name": "DEF Industries",
        "sector": "Industrial Manufacturing",
        "industry": "Precision Components",
        "hq": "Pune, India",
        "description": (
            "DEF Industries manufactures precision components for the "
            "automotive and industrial-automation sectors."
        ),
        "financials": {
            "fiscal_year": "FY2026",
            "revenue_usd_m": 305.0,
            "revenue_growth_yoy_pct": 4.1,
            "net_income_usd_m": 18.0,
            "net_margin_pct": 5.9,
            "gross_margin_pct": 32.4,
        },
        "price": {"last_price_usd": 41.20, "day_change_pct": 0.15},
    },
}

_SECTOR_DB = {
    "Enterprise Software": {
        "avg_revenue_growth_pct": 14.2,
        "avg_net_margin_pct": 9.8,
        "avg_ev_to_revenue": 6.1,
        "outlook": (
            "Steady demand for automation and AI-assisted tooling; "
            "pricing pressure from open-source alternatives is a watch item."
        ),
    },
    "Industrial Manufacturing": {
        "avg_revenue_growth_pct": 5.0,
        "avg_net_margin_pct": 7.2,
        "avg_ev_to_revenue": 1.8,
        "outlook": (
            "Demand tracks industrial capex cycles; margin performance "
            "depends heavily on input-cost pass-through."
        ),
    },
}

# Simulated failure rate for each tool, used to exercise the fallback /
# error-handling path deterministically during testing (seeded) and
# realistically during live Colab demos (unseeded).
_FAILURE_RATE = 0.0  # overridden per-call in tests via `force_failure`


def _lookup(ticker: str):
    return _COMPANY_DB.get(ticker.upper().strip())


def resolve_ticker_or_none(name_or_ticker: str):
    """Best-effort resolution of a free-text company reference (as an LLM
    might extract it from a user message, e.g. "ABC Technologies", "abc",
    "ABC") to a canonical ticker key in `_COMPANY_DB`. Returns None if no
    confident match is found — callers must treat that as "missing
    information" and ask the user, never guess.

    This is deliberately a deterministic, dependency-free helper (no LLM
    call) so ticker resolution is reproducible for grading, with the LLM
    (in the notebook's `intent_router` node) used only to pull the raw
    company reference out of the sentence, not to guess the ticker.
    """
    if not name_or_ticker:
        return None
    needle = name_or_ticker.strip().upper()
    if needle in _COMPANY_DB:
        return needle
    for ticker, record in _COMPANY_DB.items():
        if needle in record["name"].upper() or record["name"].upper() in needle:
            return ticker
    return None


def known_companies() -> dict:
    """Read-only view of {ticker: display_name} for prompting/UI use."""
    return {t: r["name"] for t, r in _COMPANY_DB.items()}


def get_company_financials(ticker: str, force_failure: bool = False) -> dict:
    """Mock tool: returns headline financials for a ticker.

    Returns a structured error object (never raises, never fabricates)
    when the ticker is unknown or a failure is forced/simulated, so the
    calling graph node can distinguish "legitimately no data" from a
    transient failure without ever inventing numbers.
    """
    if force_failure or random.random() < _FAILURE_RATE:
        return {"ok": False, "error": "data_provider_timeout", "tool": "get_company_financials"}

    record = _lookup(ticker)
    if record is None:
        return {"ok": False, "error": f"ticker_not_found:{ticker}", "tool": "get_company_financials"}

    return {"ok": True, "tool": "get_company_financials", "data": record["financials"]}


def get_stock_price(ticker: str, force_failure: bool = False) -> dict:
    """Mock tool: returns last price + day change for a ticker."""
    if force_failure or random.random() < _FAILURE_RATE:
        return {"ok": False, "error": "rate_limited", "tool": "get_stock_price"}

    record = _lookup(ticker)
    if record is None:
        return {"ok": False, "error": f"ticker_not_found:{ticker}", "tool": "get_stock_price"}

    return {"ok": True, "tool": "get_stock_price", "data": record["price"]}


def get_company_profile(ticker: str, force_failure: bool = False) -> dict:
    """Mock tool: returns qualitative company profile info."""
    if force_failure or random.random() < _FAILURE_RATE:
        return {"ok": False, "error": "profile_service_unavailable", "tool": "get_company_profile"}

    record = _lookup(ticker)
    if record is None:
        return {"ok": False, "error": f"ticker_not_found:{ticker}", "tool": "get_company_profile"}

    return {
        "ok": True,
        "tool": "get_company_profile",
        "data": {
            "name": record["name"],
            "sector": record["sector"],
            "industry": record["industry"],
            "hq": record["hq"],
            "description": record["description"],
        },
    }


def get_sector_data(sector: str, force_failure: bool = False) -> dict:
    """Mock tool: returns sector-level averages/benchmarks."""
    if force_failure or random.random() < _FAILURE_RATE:
        return {"ok": False, "error": "sector_service_timeout", "tool": "get_sector_data"}

    record = _SECTOR_DB.get(sector)
    if record is None:
        return {"ok": False, "error": f"sector_not_found:{sector}", "tool": "get_sector_data"}

    return {"ok": True, "tool": "get_sector_data", "data": record}


if __name__ == "__main__":
    # Quick manual smoke test — run with: python3 tools_mock.py
    print("Known ticker  :", get_company_financials("ABC"))
    print("Unknown ticker:", get_company_financials("QQQ"))
    print("Forced failure:", get_stock_price("ABC", force_failure=True))
    print("Sector lookup :", get_sector_data("Enterprise Software"))
    print("Bad sector    :", get_sector_data("Widgets"))
    print("Profile       :", get_company_profile("XYZ"))
