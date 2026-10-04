"""
real_market_data.py
--------------------
Live market-data tools for a small curated set of REAL public companies,
backed by the Twelve Data REST API (https://twelvedata.com/docs,
https://twelvedata.com/stocks). This sits ALONGSIDE tools_mock.py's
synthetic ABC/XYZ/DEF universe, not instead of it: the synthetic companies
exist specifically because the assignment needs deterministic, reproducible
test scenarios (a forced tool failure, a guaranteed RAG failure, a
conflicting-data case, a poisoned document) that a live data source can
never give you on demand. Real companies add the genuinely useful "ask
about an actual stock" capability on top of that, using the exact same
`{"ok": True/False, ...}` contract tools_mock.py already uses, so none of
the graph/node logic has to change shape to support both universes side by
side — see node_logic.py's `tools_node`, which simply dispatches to this
module instead of tools_mock.py when the resolved ticker is a real one.

Requires a Twelve Data API key (free signup, no credit card):
https://twelvedata.com/pricing — set via the TWELVEDATA_API_KEY
environment variable (collected in the notebook's Section 0 `.env` file,
or as an env var for the FastAPI deployment). If it's missing, every call
below returns a clean {"ok": False, "error": "missing_twelvedata_api_key"}
result rather than raising — the rest of the graph already knows how to
handle a failed tool call without crashing or fabricating an answer.

Honest limitation — read before being surprised by a `tool_failure` flag
on a real company: Twelve Data's free ("Basic") plan only includes the
`/quote` endpoint (live price, day change, volume, 52-week range). The
`/profile` endpoint (sector, industry, HQ, description) requires a
Grow-tier-or-higher paid plan. On a free key, `get_real_company_profile`
and `get_real_company_financials` (which also reads `/profile`, since
Twelve Data has no free income-statement/fundamentals endpoint at all)
will both legitimately report `{"ok": False, "error": "plan_restricted:..."}`.
This is not a bug — it's the same "never fabricate, just say so" contract
the synthetic tools already enforce, now holding up against a real API's
real plan restrictions instead of a simulated failure. `get_real_stock_price`
works on the free plan and should succeed for any of the tickers below.

Performance notes (added alongside the real-company integration, not a
separate feature):
  - A module-level `requests.Session()` (`_SESSION`) is reused across every
    call instead of opening a fresh connection each time, so repeated calls
    within a process benefit from HTTP keep-alive (no new TCP/TLS handshake
    per request).
  - A short-TTL in-memory cache inside `_twelvedata_get`, keyed by
    `(endpoint_path, ticker)`, means `get_real_company_profile` and
    `get_real_company_financials` — which both read Twelve Data's `/profile`
    endpoint for a real company, since there is no separate free
    fundamentals endpoint — issue at most ONE network call between them
    instead of two identical ones, and a repeated question about the same
    ticker within the TTL window (default 30s, `TWELVEDATA_CACHE_TTL_SECONDS`)
    is served from memory instead of spending another call against the free
    tier's limited per-minute request quota. Only genuinely deterministic
    responses are cached (a successful quote/profile, or a deterministic API
    error like `plan_restricted` or a bad symbol) — a transient network
    error is never cached, so a real outage isn't "remembered" as broken for
    the whole TTL window. Set `TWELVEDATA_CACHE_TTL_SECONDS=0` to disable
    caching entirely (e.g. while debugging a live data issue).
"""

import os
import time
from typing import Optional

import requests

TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
_SESSION = requests.Session()
_CACHE_TTL_SECONDS = float(os.environ.get("TWELVEDATA_CACHE_TTL_SECONDS", "30"))
_CACHE: dict = {}


def clear_cache() -> None:
    """Drop every cached Twelve Data response immediately. Exposed mainly
    for tests (each needs a fresh network call regardless of what an
    earlier test already cached for the same ticker/endpoint) and for
    anyone debugging a live data issue who wants to bypass the cache
    without restarting the process."""
    _CACHE.clear()


def _cache_get(key) -> Optional[dict]:
    hit = _CACHE.get(key)
    if hit is None:
        return None
    cached_at, value = hit
    if time.time() - cached_at > _CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key, value: dict) -> None:
    if _CACHE_TTL_SECONDS > 0:
        _CACHE[key] = (time.time(), value)

# A small, curated set of well-known real companies — kept deliberately
# small (rather than open-ended symbol search against Twelve Data's full
# listing at https://twelvedata.com/stocks) so company resolution stays
# deterministic and side effect free, the same design reasoning as
# tools_mock.py's three synthetic companies. Add more tickers here if you
# want broader real-company coverage; no other code needs to change.
REAL_COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "TSLA": "Tesla, Inc.",
    "NVDA": "NVIDIA Corporation",
}


def known_real_companies() -> dict:
    """Read-only view of {ticker: display_name} — same shape as
    tools_mock.known_companies(), so the two can be merged for prompting."""
    return dict(REAL_COMPANIES)


def resolve_real_ticker_or_none(name_or_ticker: str) -> Optional[str]:
    """Same resolution contract as tools_mock.resolve_ticker_or_none:
    returns a canonical ticker from REAL_COMPANIES, or None if there's no
    confident match. Never guesses, never calls the network."""
    if not name_or_ticker:
        return None
    needle = name_or_ticker.strip().upper()
    if needle in REAL_COMPANIES:
        return needle
    for ticker, name in REAL_COMPANIES.items():
        if needle in name.upper() or name.upper() in needle:
            return ticker
    return None


def _get_api_key() -> str:
    return os.environ.get("TWELVEDATA_API_KEY", "").strip()


def _twelvedata_get(path: str, params: dict, tool_name: str) -> dict:
    """Shared request helper used by every tool below. Never raises —
    whatever goes wrong (missing key, network error, bad symbol, rate
    limit, plan restriction) comes back as the same
    {"ok": False, "error": ..., "tool": ...} shape tools_mock.py uses, so
    callers (tools_node, reconcile_node) don't need to know which module a
    result came from.

    Checks the response cache first (see module docstring's Performance
    notes): a cache hit is relabeled with THIS call's `tool_name` before
    being returned, since the exact same cached `/profile` response is
    legitimately reused by both `get_company_profile` and
    `get_company_financials` — each should report itself as the tool that
    answered, not whichever one happened to populate the cache first.
    """
    cache_key = (path, params.get("symbol"))
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "tool": tool_name}

    api_key = _get_api_key()
    if not api_key:
        return {"ok": False, "error": "missing_twelvedata_api_key", "tool": tool_name}

    try:
        resp = _SESSION.get(
            f"{TWELVEDATA_BASE_URL}{path}",
            params={**params, "apikey": api_key},
            timeout=10,
        )
    except requests.exceptions.RequestException as e:
        # Never cached: a network blip is transient, not a deterministic
        # fact about this ticker/endpoint — remembering it as "broken" for
        # the whole TTL window would make a momentary hiccup look like a
        # sustained outage to every caller during that window.
        return {"ok": False, "error": f"network_error:{type(e).__name__}", "tool": tool_name}

    try:
        payload = resp.json()
    except ValueError:
        return {"ok": False, "error": f"non_json_response:status_{resp.status_code}", "tool": tool_name}

    # Twelve Data typically returns HTTP 200 with a JSON
    # {"code": ..., "message": ..., "status": "error"} body for API-level
    # failures (bad symbol, plan restriction, rate limit) rather than a
    # non-200 HTTP status — check the payload shape, not just status_code.
    if isinstance(payload, dict) and payload.get("status") == "error":
        code = payload.get("code")
        message = str(payload.get("message", "unknown_error"))
        if code in (403, 429) or "plan" in message.lower() or "upgrade" in message.lower():
            result = {"ok": False, "error": f"plan_restricted:{message}", "tool": tool_name}
        else:
            result = {"ok": False, "error": f"api_error_{code}:{message}", "tool": tool_name}
        # These ARE cached: for a given free key, "this endpoint needs a
        # paid plan" or "that symbol doesn't exist" is a deterministic fact
        # for the TTL window, not a transient failure — caching it avoids
        # spending more of the free tier's limited per-minute quota asking
        # an already-answered question again.
        _cache_set(cache_key, result)
        return result

    if resp.status_code != 200:
        return {"ok": False, "error": f"http_{resp.status_code}", "tool": tool_name}

    result = {"ok": True, "tool": tool_name, "data": payload}
    _cache_set(cache_key, result)
    return result


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_real_stock_price(ticker: str) -> dict:
    """Live price snapshot via Twelve Data's `/quote` endpoint — works on
    the free plan."""
    result = _twelvedata_get("/quote", {"symbol": ticker}, "get_stock_price")
    if not result["ok"]:
        return result
    q = result["data"]
    fifty_two_week = q.get("fifty_two_week") or {}
    return {
        "ok": True,
        "tool": "get_stock_price",
        "data": {
            "last_price_usd": _to_float(q.get("close")),
            "day_change_pct": _to_float(q.get("percent_change")),
            "volume": _to_float(q.get("volume")),
            "fifty_two_week_low": _to_float(fifty_two_week.get("low")),
            "fifty_two_week_high": _to_float(fifty_two_week.get("high")),
            "as_of": q.get("datetime"),
            "source": "twelvedata.com (live)",
        },
    }


def get_real_company_profile(ticker: str) -> dict:
    """Company profile via Twelve Data's `/profile` endpoint — requires a
    Grow-tier-or-higher plan. Reports `plan_restricted` honestly on a free
    key rather than inventing a sector, industry, or description."""
    result = _twelvedata_get("/profile", {"symbol": ticker}, "get_company_profile")
    if not result["ok"]:
        return result
    p = result["data"]
    hq = ", ".join(x for x in [p.get("city"), p.get("country")] if x)
    return {
        "ok": True,
        "tool": "get_company_profile",
        "data": {
            "name": p.get("name", REAL_COMPANIES.get(ticker, ticker)),
            "sector": p.get("sector"),
            "industry": p.get("industry"),
            "hq": hq or None,
            "description": p.get("description"),
            "source": "twelvedata.com (live)",
        },
    }


def get_real_company_financials(ticker: str) -> dict:
    """Twelve Data's free plan has no income-statement/fundamentals
    endpoint (no free /statistics, /income_statement, /earnings) — rather
    than inventing a revenue-growth or margin figure that was never
    actually returned, this reuses `/profile`'s company-level fields and
    says explicitly what it does and doesn't contain. On a free key this
    legitimately reports plan_restricted, same as get_real_company_profile.
    """
    result = _twelvedata_get("/profile", {"symbol": ticker}, "get_company_financials")
    if not result["ok"]:
        return result
    p = result["data"]
    return {
        "ok": True,
        "tool": "get_company_financials",
        "data": {
            "name": p.get("name", REAL_COMPANIES.get(ticker, ticker)),
            "sector": p.get("sector"),
            "industry": p.get("industry"),
            "employees": p.get("employees"),
            "note": (
                "Twelve Data's free plan has no income-statement "
                "fundamentals endpoint (revenue growth, margins); this is "
                "company-profile data only, not financial-statement data."
            ),
            "source": "twelvedata.com (live)",
        },
    }


if __name__ == "__main__":
    # Manual smoke test — requires TWELVEDATA_API_KEY set in the
    # environment AND outbound internet access to api.twelvedata.com,
    # neither of which is available in the sandbox this file was authored
    # in (no network egress to third-party APIs). Run this in Colab, where
    # both are available, to see it hit the real API:
    #     TWELVEDATA_API_KEY=... python3 real_market_data.py
    print("Resolve 'Apple'          ->", resolve_real_ticker_or_none("Apple"))
    print("Resolve 'apple inc'      ->", resolve_real_ticker_or_none("apple inc"))
    print("Resolve 'AAPL'           ->", resolve_real_ticker_or_none("AAPL"))
    print("Resolve 'Netflix' (n/a)  ->", resolve_real_ticker_or_none("Netflix"))
    print("Known real companies    :", known_real_companies())
    print()
    print("Live quote for AAPL     :", get_real_stock_price("AAPL"))
    print("Live profile for AAPL   :", get_real_company_profile("AAPL"))
    print("Live financials for AAPL:", get_real_company_financials("AAPL"))
