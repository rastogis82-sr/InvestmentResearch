"""
yahoo_finance_market_data.py
------------------------------
A second live-data source for the same curated real-company universe
`real_market_data.py` already covers (AAPL/MSFT/GOOGL/AMZN/TSLA/NVDA), via
Yahoo Finance's data, accessed through the `yfinance` package
(https://github.com/ranaroussi/yfinance) rather than hand-rolled HTTP calls.

This REPLACES alpha_vantage_market_data.py as the secondary provider, at
the user's explicit request to keep only Twelve Data and Yahoo Finance.
(Earlier still, FCS API was the secondary provider and had to be dropped
because its free plan rejected every stock endpoint outright — see
merged_market_data.py's module docstring for that history. Alpha Vantage
worked but came with its own tight 25-requests/day free-tier ceiling.)

Why this module exists at all (same gap every secondary provider here has
closed in turn): Twelve Data's free plan can't provide `get_company_profile`
(`/profile` is plan-restricted) or real `get_company_financials` figures
(no free fundamentals endpoint exists at all — see real_market_data.py's
module docstring). A single `yfinance.Ticker(ticker).info` call covers
both gaps at once: it returns company profile fields (sector, industry,
description, address, website, employee count) AND financial fields
(trailing P/E, profit/gross/operating margins, total revenue, EBITDA,
revenue growth, market cap) in one dict. That one-call-covers-both design
mirrors the earlier Alpha Vantage module's `OVERVIEW` call and is kept
here too: `_get_yahoo_info` is cached per-ticker and shared by
`get_yahoo_company_profile`, `get_yahoo_company_financials`, AND
`get_yahoo_stock_price` below, so a real company looked up for all three
things in the same turn costs exactly one underlying Yahoo request, not
three.

No API key is required — `yfinance` is a thin wrapper around Yahoo
Finance's own (undocumented, unofficial) web endpoints, the same way a
browser visiting finance.yahoo.com gets this data, not a published,
versioned, rate-limit-documented public API the way Twelve Data and Alpha
Vantage are. That is a real, honest trade-off, not a detail to gloss
over — see Honest limitations below.

Requires the `yfinance` package (`pip install yfinance`; already in
requirements.txt). If it isn't importable for any reason, every call
below returns a clean {"ok": False, "error": "missing_yfinance_package"}
result rather than raising, exactly like real_market_data.py does for a
missing Twelve Data key — no API key is needed, so there is no
"missing_..._api_key" case here the way there is for Twelve Data / the
earlier Alpha Vantage module.

Honest limitations — read before being surprised by an unfamiliar error:
  - Yahoo Finance has no official, documented, supported public API.
    `yfinance` (and every library like it) works by calling the same
    internal endpoints Yahoo's own website uses, which Yahoo can change,
    rate-limit, or block at any time without notice and without a
    published SLA — unlike Twelve Data's and Alpha Vantage's versioned,
    documented REST APIs. It has worked reliably for a long time and is
    very widely used, but "widely used and currently working" is a
    different guarantee than "documented and supported." Treat any
    failure here (and especially a sudden block from a specific hosting
    provider's IP range, which Yahoo has been known to do to deployed
    cloud services more aggressively than to home/residential IPs) as
    expected and survivable, not as a bug to chase — this is exactly why
    `get_merged_*` in merged_market_data.py never depends on Yahoo
    succeeding, same as every other secondary-provider integration here.
  - `yfinance` raises real Python exceptions on request failure (HTTP
    errors, rate limiting, JSON decode failures) rather than returning a
    structured error object the way Twelve Data's and Alpha Vantage's
    REST responses do. `_get_yahoo_info` below wraps every call in a
    broad `try/except Exception`, classifying by the exception's own
    type name and message, so this module's own contract — never raise,
    always return `{"ok": False, "error": ..., "tool": ...}` — holds
    regardless of what `yfinance` itself does internally.
  - An unrecognized ticker does not raise in `yfinance` — it returns an
    `info` dict with only a couple of placeholder keys (observed in
    practice as something like `{"trailingPegRatio": None}` with
    everything else absent). `_get_yahoo_info` treats "fewer than a
    handful of real keys" as an effectively empty response, the same
    honest-failure outcome as Alpha Vantage's bad-symbol `{}` or FCS
    API's `empty_response`.
  - `yfinance`'s `.info` property does not carry a CEO name or founding
    year (the fields FCS API's `/stock/profile` used to supply, back when
    FCS API was usable at all) — but unlike Alpha Vantage's `OVERVIEW`, it
    DOES include employee count (`fullTimeEmployees`), so that field is
    back after being absent in the Alpha Vantage version of this
    integration.
  - Response field names below are taken from `yfinance`'s long-stable
    `Ticker.info` dict shape, the same "verified against well-established
    behavior, not against a live call" approach the other market-data
    modules use — this sandbox has no outbound network access to
    third-party services (including PyPI itself, confirmed while building
    this module) to test against the live service or even install
    `yfinance` to inspect it directly. Smoke-test with a real deployment
    in Colab/Render before trusting this in front of a grader (see the
    `__main__` block at the bottom), and don't be surprised if Yahoo's
    exact field set has shifted slightly by the time you do — `yfinance`
    itself ships frequent point releases specifically to track Yahoo's
    endpoint changes, which is also why it's left unpinned in
    requirements.txt (same reasoning as faiss-cpu/sentence-transformers/
    gradio in the notebook's install cell).
"""

import os
import time
from typing import Optional

try:
    import yfinance as yf
    _YFINANCE_IMPORT_ERROR: Optional[str] = None
except Exception as e:  # pragma: no cover -- exercised only if yfinance truly isn't installed
    yf = None
    _YFINANCE_IMPORT_ERROR = type(e).__name__

_CACHE_TTL_SECONDS = float(os.environ.get("YAHOO_FINANCE_CACHE_TTL_SECONDS", "30"))
_CACHE: dict = {}

# A real yfinance .info response for a known ticker has dozens of keys.
# An unrecognized/delisted ticker still returns a dict, but with only a
# couple of placeholder keys (e.g. just "trailingPegRatio": None) and no
# actual company data -- this threshold is what distinguishes "real data"
# from "Yahoo silently found nothing", the same role an empty {} response
# played for Alpha Vantage.
_MIN_INFO_KEYS_FOR_REAL_TICKER = 5


def clear_cache() -> None:
    """Same purpose as real_market_data.clear_cache() / the earlier Alpha
    Vantage module's equivalent — mainly for tests and for debugging a
    live data issue without restarting the process."""
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


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_str(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _get_yahoo_info(ticker: str, tool_name: str) -> dict:
    """Shared fetch+cache helper, same never-raise contract as
    real_market_data._twelvedata_get / the earlier Alpha Vantage module's
    _alpha_vantage_get: whatever goes wrong (yfinance not installed,
    network error, rate limiting, an unrecognized ticker) comes back as
    {"ok": False, "error": ..., "tool": ...}, never an exception. Cached
    per-ticker and shared by ALL THREE get_yahoo_* functions below -- see
    the module docstring for why one call covering price+profile+
    financials is deliberate."""
    cache_key = ticker.upper()
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "tool": tool_name}

    if yf is None:
        return {
            "ok": False,
            "error": f"missing_yfinance_package:{_YFINANCE_IMPORT_ERROR}",
            "tool": tool_name,
        }

    try:
        info = yf.Ticker(ticker.upper()).info
    except Exception as e:
        # Never cached -- a transient network/rate-limit blip isn't a
        # deterministic fact about this ticker (see real_market_data.py
        # for the same reasoning). yfinance doesn't give us a structured
        # error the way a REST JSON body does, so classify by exception
        # type/message instead.
        error_text = f"{type(e).__name__}:{e}".lower()
        if "429" in error_text or "rate limit" in error_text or "too many requests" in error_text:
            return {"ok": False, "error": f"rate_limited:{type(e).__name__}", "tool": tool_name}
        return {"ok": False, "error": f"network_error:{type(e).__name__}", "tool": tool_name}

    if not isinstance(info, dict) or len(info) < _MIN_INFO_KEYS_FOR_REAL_TICKER:
        result = {"ok": False, "error": "empty_response", "tool": tool_name}
        _cache_set(cache_key, result)
        return result

    result = {"ok": True, "tool": tool_name, "data": info}
    _cache_set(cache_key, result)
    return result


def get_yahoo_stock_price(ticker: str) -> dict:
    """Live-ish price snapshot via yfinance's `Ticker.info`. Used by
    merged_market_data.py only as a FALLBACK when Twelve Data's own
    `/quote` call fails -- Twelve Data's free plan already covers price
    reliably, so this conserves Yahoo Finance's unofficial, undocumented
    endpoints for the cases that actually need them: profile and
    financials, where Twelve Data's free plan can't help at all."""
    result = _get_yahoo_info(ticker, "get_stock_price")
    if not result["ok"]:
        return result
    info = result["data"] or {}
    price = _to_float(info.get("currentPrice")) or _to_float(info.get("regularMarketPrice"))
    # NOTE: yfinance's price-change fields (regularMarketChangePercent)
    # come back already in percent units (e.g. -0.21 means -0.21%), NOT a
    # 0-1 fraction -- unlike its margin/growth-ratio fields below
    # (profitMargins, grossMargins, revenueGrowth, ...), which ARE
    # fractions and do need *100. Mixing these two conventions up was a
    # real risk carried over from the Alpha Vantage version of this
    # module, where every percent-shaped field was a fraction -- confirmed
    # against yfinance's well-established field conventions, not a live
    # call (see the module docstring).
    change_pct = _to_float(info.get("regularMarketChangePercent"))
    return {
        "ok": True,
        "tool": "get_stock_price",
        "data": {
            "last_price_usd": price,
            "day_change_pct": change_pct,
            "volume": _to_float(info.get("regularMarketVolume")),
            "source": "finance.yahoo.com (live, via yfinance)",
        },
    }


def get_yahoo_company_profile(ticker: str) -> dict:
    """Company profile via yfinance's `Ticker.info` -- sector, industry,
    HQ, description, employee count, website, and market cap. All
    available with no API key or paid plan, unlike Twelve Data's
    equivalent. Unlike the earlier Alpha Vantage version of this
    integration, employee count (`fullTimeEmployees`) IS available here;
    CEO name and founding year still aren't (yfinance's `.info` has
    neither), so those stay honestly absent rather than guessed."""
    result = _get_yahoo_info(ticker, "get_company_profile")
    if not result["ok"]:
        return result
    info = result["data"] or {}
    market_cap = _to_float(info.get("marketCap"))
    city = _clean_str(info.get("city"))
    state = _clean_str(info.get("state"))
    country = _clean_str(info.get("country"))
    hq = ", ".join(x for x in [city, state, country] if x)
    return {
        "ok": True,
        "tool": "get_company_profile",
        "data": {
            "name": _clean_str(info.get("longName")) or _clean_str(info.get("shortName")),
            "sector": _clean_str(info.get("sector")),
            "industry": _clean_str(info.get("industry")),
            "hq": hq or None,
            "description": _clean_str(info.get("longBusinessSummary")),
            "employees": info.get("fullTimeEmployees"),
            "website": _clean_str(info.get("website")),
            "market_cap_usd_b": market_cap / 1e9 if market_cap is not None else None,
            "source": "finance.yahoo.com (live, via yfinance)",
        },
    }


def get_yahoo_company_financials(ticker: str) -> dict:
    """Real margins, revenue, and a real year-over-year revenue-growth
    figure, all from the SAME `Ticker.info` call `get_yahoo_company_profile`
    already makes (and shares via the module-level cache) -- the one
    endpoint Twelve Data's free plan has no equivalent for at all.
    `net_income_usd_m` comes directly from yfinance's `netIncomeToCommon`
    when present; unlike the Alpha Vantage version of this integration,
    nothing here needs to be calculated from other fields."""
    result = _get_yahoo_info(ticker, "get_company_financials")
    if not result["ok"]:
        return result
    info = result["data"] or {}

    revenue = _to_float(info.get("totalRevenue"))
    net_margin = _to_float(info.get("profitMargins"))
    gross_margin = _to_float(info.get("grossMargins"))
    operating_margin = _to_float(info.get("operatingMargins"))
    revenue_growth = _to_float(info.get("revenueGrowth"))
    market_cap = _to_float(info.get("marketCap"))
    pe_ratio = _to_float(info.get("trailingPE"))
    ebitda = _to_float(info.get("ebitda"))
    net_income = _to_float(info.get("netIncomeToCommon"))

    data: dict = {}
    if revenue is not None:
        data["revenue_usd_m"] = revenue / 1e6
    if net_margin is not None:
        data["net_margin_pct"] = net_margin * 100
    if gross_margin is not None:
        data["gross_margin_pct"] = gross_margin * 100
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
    if net_income is not None:
        data["net_income_usd_m"] = net_income / 1e6

    if not data:
        return {"ok": False, "tool": "get_company_financials", "error": "empty_response"}

    data["source"] = "finance.yahoo.com (live, via yfinance)"
    return {"ok": True, "tool": "get_company_financials", "data": data}


if __name__ == "__main__":
    # Manual smoke test -- requires the yfinance package AND outbound
    # internet access to Yahoo Finance's endpoints, neither fully
    # verifiable in the sandbox this file was authored in (no network
    # access to even install/import yfinance to confirm its current field
    # names). Run in Colab:
    #     pip install yfinance && python3 yahoo_finance_market_data.py
    print("Live quote for AAPL     :", get_yahoo_stock_price("AAPL"))
    print("Live profile for AAPL   :", get_yahoo_company_profile("AAPL"))
    print("Live financials for AAPL:", get_yahoo_company_financials("AAPL"))
