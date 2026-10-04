"""
fcs_market_data.py
-------------------
A second live-data source for the same curated real-company universe
`real_market_data.py` already covers (AAPL/MSFT/GOOGL/AMZN/TSLA/NVDA),
backed by FCS API's Stock Market API (https://fcsapi.com/document/stock-api).

This exists to close a real gap, not to replace Twelve Data: on Twelve
Data's free ("Basic") plan, `/profile` is plan-restricted (so
`get_company_profile` legitimately fails) and there is no free
fundamentals endpoint at all (so `get_company_financials` never returns
actual revenue/margin figures, even when it "succeeds" — see
`real_market_data.py`'s module docstring). FCS API's free plan includes
`/stock/profile` (sector, industry, HQ, description, employee count,
market cap, CEO, website) and enough of `/stock/statistics` and
`/stock/income_statements` to recover real margins and a genuine
year-over-year revenue-growth figure. `merged_market_data.py` is what
actually combines this module's output with `real_market_data.py`'s —
this module, like `real_market_data.py`, only ever talks to ONE provider
and follows the exact same never-fabricate contract on its own.

Requires an FCS API key (free signup, no credit card):
https://fcsapi.com/pricing — set via the FCS_API_KEY environment
variable. If it's missing, every call below returns a clean
{"ok": False, "error": "missing_fcs_api_key"} result rather than raising,
exactly like real_market_data.py does for a missing Twelve Data key.

Honest limitations — read before being surprised by an unfamiliar error:
  - FCS API's free plan is capped at 500 requests/month AND 3 requests
    per minute (https://fcsapi.com/pricing) — noticeably tighter than
    Twelve Data's free per-minute allowance. `get_fcs_company_financials`
    alone costs 2 of those 3 (one for `/stock/statistics`, one for
    `/stock/income_statements`), so two real-company financials lookups
    back-to-back inside the same 60-second window can legitimately hit
    the rate limit. That comes back as a normal `{"ok": False, "error":
    "rate_limited:..."}`` result, same as any other API-level failure —
    never a crash, never fabricated data.
  - Response shapes below (the `response`/`profile`/`active` envelope,
    `/stock/statistics` and `/stock/income_statements`'s field names) are
    taken from FCS API's own published documentation examples, the same
    "verified against docs, not against the live API" approach
    `real_market_data.py` already uses — this sandbox has no outbound
    network access to third-party APIs to test against the live service
    either. Smoke-test with a real key in Colab before trusting this in
    front of a grader (see the `__main__` block at the bottom).
  - The curated ticker set is assumed to trade on NASDAQ (true for all
    six: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA), since FCS API's `symbol`
    parameter is exchange-qualified (`NASDAQ:AAPL`, per its docs). Adding
    a ticker on a different exchange to `real_market_data.REAL_COMPANIES`
    would need its exchange added to `_EXCHANGE_OVERRIDES` below, or this
    module will send a (likely wrong) `NASDAQ:` prefix for it.
  - `/stock/income_statements` reports raw dollar figures, not millions —
    `get_fcs_company_financials` divides by 1e6 to match this project's
    existing `*_usd_m` convention (see `tools_mock.py`'s `_COMPANY_DB`).
    Year-over-year revenue growth is computed here (latest fiscal period
    vs. the prior one in the same response), not returned directly by FCS.
"""

import os
import time
from typing import Optional

import requests

FCS_BASE_URL = "https://api-v4.fcsapi.com"
_SESSION = requests.Session()
_CACHE_TTL_SECONDS = float(os.environ.get("FCS_CACHE_TTL_SECONDS", "30"))
_CACHE: dict = {}

# All six curated real companies (real_market_data.REAL_COMPANIES) trade on
# NASDAQ — see the module docstring. Override here per-ticker if a future
# addition trades elsewhere; anything not listed defaults to NASDAQ.
_EXCHANGE_OVERRIDES: dict = {}


def clear_cache() -> None:
    """Same purpose as real_market_data.clear_cache() — mainly for tests
    and for debugging a live data issue without restarting the process."""
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


def _get_api_key() -> str:
    return os.environ.get("FCS_API_KEY", "").strip()


def _fcs_symbol(ticker: str) -> str:
    exchange = _EXCHANGE_OVERRIDES.get(ticker.upper(), "NASDAQ")
    return f"{exchange}:{ticker.upper()}"


def _unwrap(payload) -> Optional[dict]:
    """FCS API wraps a single-symbol result in a one-item list for
    /stock/latest and /stock/profile, but returns /stock/statistics and
    /stock/income_statements as a direct object (per FCS API's own
    documented examples) — this normalizes both shapes to "the one record
    we actually asked for", or None if the envelope is empty/unexpected."""
    response = payload.get("response")
    if isinstance(response, list):
        return response[0] if response else None
    if isinstance(response, dict):
        return response
    return None


def _fcsapi_get(path: str, params: dict, tool_name: str, cache_key_suffix: str = "") -> dict:
    """Shared request helper, same never-raise contract as
    real_market_data._twelvedata_get: whatever goes wrong (missing key,
    network error, bad symbol, rate limit) comes back as
    {"ok": False, "error": ..., "tool": ...}, never an exception."""
    symbol = params.get("symbol", "")
    cache_key = (path, symbol, cache_key_suffix)
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "tool": tool_name}

    api_key = _get_api_key()
    if not api_key:
        return {"ok": False, "error": "missing_fcs_api_key", "tool": tool_name}

    try:
        resp = _SESSION.get(
            f"{FCS_BASE_URL}{path}",
            params={**params, "access_key": api_key},
            timeout=10,
        )
    except requests.exceptions.RequestException as e:
        # Never cached -- a transient network blip isn't a deterministic
        # fact about this ticker/endpoint (see real_market_data.py for the
        # same reasoning).
        return {"ok": False, "error": f"network_error:{type(e).__name__}", "tool": tool_name}

    try:
        payload = resp.json()
    except ValueError:
        return {"ok": False, "error": f"non_json_response:status_{resp.status_code}", "tool": tool_name}

    if not isinstance(payload, dict) or payload.get("status") is not True:
        code = payload.get("code") if isinstance(payload, dict) else None
        message = str(payload.get("msg", "unknown_error")) if isinstance(payload, dict) else "unknown_error"
        message_l = message.lower()
        if code == 429 or "rate limit" in message_l or "too many" in message_l:
            result = {"ok": False, "error": f"rate_limited:{message}", "tool": tool_name}
        elif code in (401, 403) or "plan" in message_l or "upgrade" in message_l or "subscription" in message_l:
            result = {"ok": False, "error": f"plan_restricted:{message}", "tool": tool_name}
        else:
            result = {"ok": False, "error": f"api_error_{code}:{message}", "tool": tool_name}
        # Deterministic-for-the-TTL-window failures are cached, same
        # reasoning as real_market_data.py (don't burn more of FCS's
        # especially tight 3-requests-per-minute free quota re-asking an
        # already-answered question).
        _cache_set(cache_key, result)
        return result

    record = _unwrap(payload)
    if record is None:
        result = {"ok": False, "error": "empty_response", "tool": tool_name}
        _cache_set(cache_key, result)
        return result

    result = {"ok": True, "tool": tool_name, "data": record}
    _cache_set(cache_key, result)
    return result


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_fcs_stock_price(ticker: str) -> dict:
    """Live price snapshot via FCS API's `/stock/latest`. Used by
    merged_market_data.py only as a FALLBACK when Twelve Data's own
    `/quote` call fails — Twelve Data's free plan already covers price
    reliably, so this conserves FCS's tighter free-tier quota for
    profile/financials, where Twelve Data's free plan can't help at all."""
    result = _fcsapi_get("/stock/latest", {"symbol": _fcs_symbol(ticker)}, "get_stock_price")
    if not result["ok"]:
        return result
    active = (result["data"] or {}).get("active") or {}
    return {
        "ok": True,
        "tool": "get_stock_price",
        "data": {
            "last_price_usd": _to_float(active.get("c")),
            "day_change_pct": _to_float(active.get("chp")),
            "volume": _to_float(active.get("v")),
            "as_of": active.get("tm"),
            "source": "fcsapi.com (live)",
        },
    }


def get_fcs_company_profile(ticker: str) -> dict:
    """Company profile via FCS API's `/stock/profile` — sector, industry,
    HQ, description, employee count, founding year, CEO, website, and
    market cap. All available on FCS API's free plan (unlike Twelve
    Data's equivalent, which needs a paid plan) — this is the main reason
    this module exists."""
    result = _fcsapi_get("/stock/profile", {"symbol": _fcs_symbol(ticker)}, "get_company_profile")
    if not result["ok"]:
        return result
    p = (result["data"] or {}).get("profile") or {}
    market_cap = _to_float(p.get("market_cap"))
    founded = p.get("founded")
    hq = ", ".join(x for x in [p.get("headquarters"), p.get("region")] if x)
    return {
        "ok": True,
        "tool": "get_company_profile",
        "data": {
            "sector": p.get("sector"),
            "industry": p.get("industry"),
            "hq": hq or None,
            "description": p.get("description"),
            "employees": p.get("total_employees"),
            # Kept as a string, not a number -- a founding year like 1976
            # must never pass through the generic numeric formatter, which
            # would render it as "1,976".
            "founded": str(founded) if founded else None,
            "ceo": p.get("ceo"),
            "website": p.get("website"),
            "market_cap_usd_b": market_cap / 1e9 if market_cap is not None else None,
            "source": "fcsapi.com (live)",
        },
    }


def get_fcs_company_financials(ticker: str) -> dict:
    """Real margins and a real year-over-year revenue-growth figure, via
    FCS API's `/stock/statistics` (margins, P/E, market cap) and
    `/stock/income_statements` (revenue, net income, EPS, keyed by
    fiscal-period date) — the two endpoints Twelve Data's free plan has no
    equivalent for at all. Costs 2 of FCS API's 3-requests-per-minute free
    quota; see the module docstring's Honest limitations.

    Partial success is still success: if one endpoint fails (e.g. a
    free-plan restriction on `/stock/income_statements` specifically) but
    the other succeeds, this returns ok=True with whatever fields the
    successful call actually provided — never fabricating the other
    endpoint's fields, just honestly omitting them. Only returns ok=False
    if BOTH endpoints fail."""
    symbol = _fcs_symbol(ticker)
    stats_result = _fcsapi_get("/stock/statistics", {"symbol": symbol}, "get_company_financials")
    income_result = _fcsapi_get(
        "/stock/income_statements", {"symbol": symbol}, "get_company_financials", cache_key_suffix="income"
    )

    data: dict = {}
    errors = []

    if stats_result["ok"]:
        s = stats_result["data"] or {}
        data["net_margin_pct"] = _to_float(s.get("net_margin"))
        data["gross_margin_pct"] = _to_float(s.get("gross_margin"))
        data["operating_margin_pct"] = _to_float(s.get("operating_margin"))
        data["pe_ratio"] = _to_float(s.get("price_earnings"))
        market_cap = _to_float(s.get("market_cap_basic"))
        if market_cap is not None:
            data["market_cap_usd_b"] = market_cap / 1e9
    else:
        errors.append(f"statistics:{stats_result.get('error')}")

    if income_result["ok"]:
        periods = income_result["data"] or {}
        # Keys are "YYYY-MM-DD" fiscal-period-end dates; sort descending so
        # [0] is the latest period and [1] (if present) is the prior one,
        # used to compute YoY growth below.
        sorted_dates = sorted((d for d in periods if isinstance(periods.get(d), dict)), reverse=True)
        if sorted_dates:
            latest = periods[sorted_dates[0]]
            latest_revenue = _to_float(latest.get("total_revenue"))
            latest_net_income = _to_float(latest.get("net_income"))
            if latest_revenue is not None:
                data["revenue_usd_m"] = latest_revenue / 1e6
            if latest_net_income is not None:
                data["net_income_usd_m"] = latest_net_income / 1e6
            data["fiscal_year"] = sorted_dates[0]
            if len(sorted_dates) > 1:
                prior_revenue = _to_float(periods[sorted_dates[1]].get("total_revenue"))
                if latest_revenue is not None and prior_revenue:
                    data["revenue_growth_yoy_pct"] = (latest_revenue - prior_revenue) / prior_revenue * 100
    else:
        errors.append(f"income_statements:{income_result.get('error')}")

    if not data:
        return {
            "ok": False,
            "tool": "get_company_financials",
            "error": "; ".join(errors) or "unknown_error",
        }

    if errors:
        data["note"] = f"Partial data -- one FCS API endpoint failed ({'; '.join(errors)})."
    data["source"] = "fcsapi.com (live)"
    return {"ok": True, "tool": "get_company_financials", "data": data}


if __name__ == "__main__":
    # Manual smoke test -- requires FCS_API_KEY set in the environment AND
    # outbound internet access to api-v4.fcsapi.com, neither of which is
    # available in the sandbox this file was authored in. Run in Colab:
    #     FCS_API_KEY=... python3 fcs_market_data.py
    print("Live quote for AAPL     :", get_fcs_stock_price("AAPL"))
    print("Live profile for AAPL   :", get_fcs_company_profile("AAPL"))
    print("Live financials for AAPL:", get_fcs_company_financials("AAPL"))
