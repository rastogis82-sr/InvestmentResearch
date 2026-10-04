"""
alpha_vantage_market_data.py
------------------------------
A second live-data source for the same curated real-company universe
`real_market_data.py` already covers (AAPL/MSFT/GOOGL/AMZN/TSLA/NVDA),
backed by Alpha Vantage (https://www.alphavantage.co/documentation/).

This REPLACES fcs_market_data.py as the secondary provider. FCS API's
documentation implied its `/stock/profile` and `/stock/statistics` /
`/stock/income_statements` endpoints were free-plan accessible, but a live
test against the user's real FCS API key showed the opposite: FCS API's
free plan rejects EVERY stock endpoint outright ("A Free API Key user
cannot be used with stock/index endpoint. Please upgrade your plan.") —
not a documentation gap on our side, a documentation-vs-reality mismatch
on FCS API's own site. Alpha Vantage's free tier, by contrast, has no
endpoint-level gating: any function works on a free key, the constraint
is purely a request-count ceiling (25 requests/day, 5 requests/minute —
see https://www.alphavantage.co/premium/).

Why this module exists at all (same gap as fcs_market_data.py was closing):
Twelve Data's free plan can't provide `get_company_profile` (`/profile`
is plan-restricted) or real `get_company_financials` figures (no free
fundamentals endpoint exists at all — see real_market_data.py's module
docstring). Alpha Vantage's `OVERVIEW` function covers both gaps in a
SINGLE call: it returns company profile fields (sector, industry,
description, address, website) AND trailing-twelve-month financial
fields (RevenueTTM, GrossProfitTTM, ProfitMargin, OperatingMarginTTM,
QuarterlyRevenueGrowthYOY, EBITDA, MarketCapitalization, PERatio) in one
response. That one-call-covers-both design is deliberate here: given
Alpha Vantage's tight 25-requests/day free ceiling, calling `OVERVIEW`
once per ticker and reusing it for both `get_alpha_vantage_company_profile`
and `get_alpha_vantage_company_financials` (via the module-level cache
below) costs half what two separate endpoints would.

Requires an Alpha Vantage API key (free signup, no credit card):
https://www.alphavantage.co/support/#api-key — set via the
ALPHA_VANTAGE_API_KEY environment variable. If it's missing, every call
below returns a clean {"ok": False, "error": "missing_alpha_vantage_api_key"}
result rather than raising, exactly like real_market_data.py does for a
missing Twelve Data key.

Honest limitations — read before being surprised by an unfamiliar error:
  - Free tier is 25 requests/DAY (not per-minute) and 5 requests/minute
    (https://www.alphavantage.co/premium/). That daily cap is tight
    enough that `_CACHE_TTL_SECONDS` defaults higher here than the other
    two market-data modules' 30-second default — see below — but a cache
    TTL only helps within one process's lifetime; it cannot stretch a
    hard daily quota across a classroom's worth of demo runs. Budget
    accordingly when demoing live.
  - Alpha Vantage signals rate-limiting and plan/key problems INSIDE a
    200 OK response body, not via HTTP status codes: a `"Note"` key means
    the per-minute/per-day limit was hit, an `"Information"` key means a
    bad/demo API key or an unrecognized function, and an `"Error Message"`
    key means a bad symbol or malformed parameter. `_alpha_vantage_get`
    below checks for all three before trusting the payload as real data.
  - `OVERVIEW` does not include employee count, CEO name, or founding
    year (the fields FCS API's `/stock/profile` used to supply) — Alpha
    Vantage simply doesn't have a field for these. `merged_market_data.py`
    is built to never fabricate a field that genuinely isn't available
    from either provider, so those three fields are just honestly absent
    now rather than back-filled with a guess.
  - `OVERVIEW`'s numeric fields come back as JSON STRINGS (e.g.
    `"ProfitMargin": "0.2431"`, a fraction, not a percentage — multiply by
    100 before treating it as one), and a field Alpha Vantage doesn't have
    data for is the literal string `"None"`, not JSON null. `_to_float`
    below handles both.
  - Response field names below are taken from Alpha Vantage's own
    published documentation and the GAAP fundamentals field reference
    (https://www.alphavantage.co/documentation/,
    https://documentation.alphavantage.co/FundamentalDataDocs/gaap_documentation.html),
    the same "verified against docs, not against the live API" approach
    the other two market-data modules use — this sandbox has no outbound
    network access to third-party APIs to test against the live service
    either. Smoke-test with a real key in Colab before trusting this in
    front of a grader (see the `__main__` block at the bottom).
"""

import os
import time
from typing import Optional

import requests

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
_SESSION = requests.Session()
# Higher default than real_market_data.py / fcs_market_data.py's 30s: Alpha
# Vantage's quota is per-DAY (25/day), not just per-minute, so it's worth
# holding a cached answer longer within one process's lifetime.
_CACHE_TTL_SECONDS = float(os.environ.get("ALPHA_VANTAGE_CACHE_TTL_SECONDS", "120"))
_CACHE: dict = {}


def clear_cache() -> None:
    """Same purpose as real_market_data.clear_cache() / fcs_market_data's
    equivalent — mainly for tests and for debugging a live data issue
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


def _get_api_key() -> str:
    return os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()


def _clean_str(value) -> Optional[str]:
    """Alpha Vantage uses the literal string "None" for a text field it has
    no data for too (not just numeric fields) -- same normalization as
    _to_float, just returning the original string instead of a float."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", ""):
        return None
    return value


def _to_float(value) -> Optional[float]:
    """Alpha Vantage returns numeric fields as JSON strings, and uses the
    literal string "None" (not JSON null) for a field it has no data for.
    Handles both, same contract as the other two market-data modules'
    _to_float helpers."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _alpha_vantage_get(params: dict, tool_name: str, cache_key_suffix: str = "") -> dict:
    """Shared request helper, same never-raise contract as
    real_market_data._twelvedata_get / fcs_market_data._fcsapi_get:
    whatever goes wrong (missing key, network error, bad symbol, rate
    limit) comes back as {"ok": False, "error": ..., "tool": ...}, never
    an exception. Returns {"ok": True, "tool": ..., "data": <raw payload>}
    on success -- callers map the raw Alpha Vantage field names to this
    project's own field names."""
    symbol = params.get("symbol", "")
    function = params.get("function", "")
    cache_key = (function, symbol, cache_key_suffix)
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "tool": tool_name}

    api_key = _get_api_key()
    if not api_key:
        return {"ok": False, "error": "missing_alpha_vantage_api_key", "tool": tool_name}

    try:
        resp = _SESSION.get(
            ALPHA_VANTAGE_BASE_URL,
            params={**params, "apikey": api_key},
            timeout=10,
        )
    except requests.exceptions.RequestException as e:
        # Never cached -- a transient network blip isn't a deterministic
        # fact about this ticker/endpoint (see real_market_data.py for the
        # same reasoning).
        return {"ok": False, "error": f"network_error:{type(e).__name__}", "tool": tool_name}

    if resp.status_code != 200:
        return {"ok": False, "error": f"api_error_{resp.status_code}:http_status", "tool": tool_name}

    try:
        payload = resp.json()
    except ValueError:
        return {"ok": False, "error": f"non_json_response:status_{resp.status_code}", "tool": tool_name}

    if not isinstance(payload, dict):
        result = {"ok": False, "error": "unexpected_response_shape", "tool": tool_name}
        return result

    # Alpha Vantage signals problems INSIDE a 200 OK body -- see the module
    # docstring's Honest limitations.
    if "Note" in payload:
        result = {"ok": False, "error": f"rate_limited:{payload['Note']}", "tool": tool_name}
        # Deterministic for the rest of this quota window -- don't keep
        # re-asking an already-rate-limited question and burning more of
        # the 25-requests/day free quota.
        _cache_set(cache_key, result)
        return result
    if "Information" in payload:
        message = str(payload["Information"])
        message_l = message.lower()
        if "api key" in message_l or "apikey" in message_l or "demo" in message_l:
            result = {"ok": False, "error": f"invalid_api_key:{message}", "tool": tool_name}
        else:
            result = {"ok": False, "error": f"api_error:{message}", "tool": tool_name}
        _cache_set(cache_key, result)
        return result
    if "Error Message" in payload:
        result = {"ok": False, "error": f"api_error:{payload['Error Message']}", "tool": tool_name}
        _cache_set(cache_key, result)
        return result

    if not payload:
        # Alpha Vantage's OVERVIEW returns an empty {} for an unrecognized
        # symbol, with none of the three error keys above set.
        result = {"ok": False, "error": "empty_response", "tool": tool_name}
        _cache_set(cache_key, result)
        return result

    result = {"ok": True, "tool": tool_name, "data": payload}
    _cache_set(cache_key, result)
    return result


def _get_overview(ticker: str, tool_name: str) -> dict:
    """OVERVIEW is cached per-ticker and shared by BOTH
    get_alpha_vantage_company_profile and get_alpha_vantage_company_financials
    below -- see the module docstring for why one call covering both is
    deliberate given the 25-requests/day free ceiling. `tool_name` is only
    used to label the result if this particular call fails; the cache key
    itself is keyed on (function, symbol) so profile and financials share
    the same cached entry regardless of which one asked first."""
    return _alpha_vantage_get({"function": "OVERVIEW", "symbol": ticker.upper()}, tool_name)


def get_alpha_vantage_stock_price(ticker: str) -> dict:
    """Live price snapshot via Alpha Vantage's `GLOBAL_QUOTE` function.
    Used by merged_market_data.py only as a FALLBACK when Twelve Data's
    own `/quote` call fails -- Twelve Data's free plan already covers
    price reliably, so this conserves Alpha Vantage's especially tight
    daily quota for profile/financials, where Twelve Data's free plan
    can't help at all."""
    result = _alpha_vantage_get({"function": "GLOBAL_QUOTE", "symbol": ticker.upper()}, "get_stock_price")
    if not result["ok"]:
        return result
    quote = (result["data"] or {}).get("Global Quote") or {}
    if not quote:
        return {"ok": False, "error": "empty_response", "tool": "get_stock_price"}
    change_pct_raw = quote.get("10. change percent", "")
    change_pct = _to_float(str(change_pct_raw).rstrip("%")) if change_pct_raw else None
    return {
        "ok": True,
        "tool": "get_stock_price",
        "data": {
            "last_price_usd": _to_float(quote.get("05. price")),
            "day_change_pct": change_pct,
            "volume": _to_float(quote.get("06. volume")),
            "as_of": quote.get("07. latest trading day"),
            "source": "alphavantage.co (live)",
        },
    }


def get_alpha_vantage_company_profile(ticker: str) -> dict:
    """Company profile via Alpha Vantage's `OVERVIEW` function -- sector,
    industry, HQ, description, website, and market cap. All available on
    Alpha Vantage's free plan (unlike Twelve Data's equivalent, which
    needs a paid plan). Note: OVERVIEW has no employee count, CEO name, or
    founding year fields at all -- those simply aren't included here (see
    the module docstring's Honest limitations), not silently dropped."""
    result = _get_overview(ticker, "get_company_profile")
    if not result["ok"]:
        return result
    o = result["data"] or {}
    market_cap = _to_float(o.get("MarketCapitalization"))
    address = _clean_str(o.get("Address"))
    country = _clean_str(o.get("Country"))
    hq = ", ".join(x for x in [address, country] if x)
    return {
        "ok": True,
        "tool": "get_company_profile",
        "data": {
            "name": _clean_str(o.get("Name")),
            "sector": _clean_str(o.get("Sector")),
            "industry": _clean_str(o.get("Industry")),
            "hq": hq or None,
            "description": _clean_str(o.get("Description")),
            "website": _clean_str(o.get("OfficialSite")),
            "market_cap_usd_b": market_cap / 1e9 if market_cap is not None else None,
            "source": "alphavantage.co (live)",
        },
    }


def get_alpha_vantage_company_financials(ticker: str) -> dict:
    """Real margins, revenue, and a real year-over-year revenue-growth
    figure, all from the SAME `OVERVIEW` call `get_alpha_vantage_company_profile`
    already makes (and shares via the module-level cache) -- the one
    endpoint Twelve Data's free plan has no equivalent for at all.
    `net_income_usd_m` is calculated (RevenueTTM x ProfitMargin), not a
    field Alpha Vantage returns directly -- labeled as such in `note` so
    the transparency panel never implies it's a raw reported figure."""
    result = _get_overview(ticker, "get_company_financials")
    if not result["ok"]:
        return result
    o = result["data"] or {}

    revenue_ttm = _to_float(o.get("RevenueTTM"))
    profit_margin = _to_float(o.get("ProfitMargin"))
    gross_profit_ttm = _to_float(o.get("GrossProfitTTM"))
    operating_margin = _to_float(o.get("OperatingMarginTTM"))
    revenue_growth = _to_float(o.get("QuarterlyRevenueGrowthYOY"))
    market_cap = _to_float(o.get("MarketCapitalization"))
    pe_ratio = _to_float(o.get("PERatio"))
    ebitda = _to_float(o.get("EBITDA"))

    data: dict = {}
    if revenue_ttm is not None:
        data["revenue_usd_m"] = revenue_ttm / 1e6
    if profit_margin is not None:
        data["net_margin_pct"] = profit_margin * 100
    if revenue_ttm is not None and gross_profit_ttm is not None:
        data["gross_margin_pct"] = gross_profit_ttm / revenue_ttm * 100 if revenue_ttm else None
    if operating_margin is not None:
        data["operating_margin_pct"] = operating_margin * 100
    if revenue_growth is not None:
        data["revenue_growth_yoy_pct"] = revenue_growth * 100
    if pe_ratio is not None:
        data["pe_ratio"] = pe_ratio
    if market_cap is not None:
        data["market_cap_usd_b"] = market_cap / 1e9
    if ebitda is not None:
        data["ebitda_usd_m"] = ebitda / 1e6
    if revenue_ttm is not None and profit_margin is not None:
        data["net_income_usd_m"] = revenue_ttm * profit_margin / 1e6
    fiscal_quarter = _clean_str(o.get("LatestQuarter"))
    if fiscal_quarter:
        data["fiscal_year"] = fiscal_quarter

    if not data:
        return {"ok": False, "tool": "get_company_financials", "error": "empty_response"}

    notes = []
    if "net_income_usd_m" in data:
        notes.append("net_income_usd_m is calculated (RevenueTTM x ProfitMargin), not a figure Alpha Vantage reports directly.")
    if notes:
        data["note"] = " ".join(notes)
    data["source"] = "alphavantage.co (live)"
    return {"ok": True, "tool": "get_company_financials", "data": data}


if __name__ == "__main__":
    # Manual smoke test -- requires ALPHA_VANTAGE_API_KEY set in the
    # environment AND outbound internet access to www.alphavantage.co,
    # neither of which is available in the sandbox this file was authored
    # in. Run in Colab:
    #     ALPHA_VANTAGE_API_KEY=... python3 alpha_vantage_market_data.py
    print("Live quote for AAPL     :", get_alpha_vantage_stock_price("AAPL"))
    print("Live profile for AAPL   :", get_alpha_vantage_company_profile("AAPL"))
    print("Live financials for AAPL:", get_alpha_vantage_company_financials("AAPL"))
