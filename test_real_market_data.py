"""
test_real_market_data.py
--------------------------
Unit tests for real_market_data.py's request/response handling, using
mocked `requests.get` calls (this sandbox has no outbound network access
to api.twelvedata.com, so these verify the PARSING and ERROR-HANDLING
logic against realistic response shapes taken from Twelve Data's own
published docs — not the live API itself, which needs a real key and
should be smoke-tested once in Colab via `real_market_data.py`'s own
`__main__` block).

Run with: python3 test_real_market_data.py
"""

import os
from unittest.mock import patch, MagicMock

import real_market_data as rmd

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, condition: bool, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status))
    print(f"[{status}] {name}" + (f" — {detail}" if detail and status == FAIL else ""))


# ---------------------------------------------------------------------------
# Resolution (no network involved)
# ---------------------------------------------------------------------------
check("resolve: ticker direct", rmd.resolve_real_ticker_or_none("AAPL") == "AAPL")
check("resolve: ticker lowercase", rmd.resolve_real_ticker_or_none("aapl") == "AAPL")
check("resolve: by name", rmd.resolve_real_ticker_or_none("Apple") == "AAPL")
check("resolve: by full legal name", rmd.resolve_real_ticker_or_none("Apple Inc.") == "AAPL")
check("resolve: unknown real-world company returns None", rmd.resolve_real_ticker_or_none("Netflix") is None)
check("resolve: empty string returns None", rmd.resolve_real_ticker_or_none("") is None)
check("known_real_companies: includes curated tickers",
      set(rmd.known_real_companies()) == set(rmd.REAL_COMPANIES))


def _mock_response(json_body, status_code=200):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_body
    return m


# NOTE on patch target: real_market_data now issues requests through a
# shared module-level `requests.Session()` (`_SESSION`) rather than calling
# `requests.get` fresh each time (see module docstring's Performance
# notes — this reuses the HTTP connection instead of paying a new TCP/TLS
# handshake on every call), so tests patch `real_market_data._SESSION.get`
# instead of `real_market_data.requests.get`.
#
# NOTE on cache isolation: `_twelvedata_get` now caches responses keyed by
# (endpoint_path, ticker) — see module docstring. That's the whole point of
# several tests below, but it also means tests that reuse the same
# ticker+endpoint combination with a DIFFERENT mocked response (e.g. a
# later "network error" test re-using AAPL/"/quote" after an earlier
# "quote success" test already cached a result for it) would silently get
# a stale cached answer instead of exercising the new mock. Every `with
# patch(...)` block below starts with `rmd.clear_cache()` for exactly this
# reason — it's test isolation, not a behavior being tested, except where a
# block's own point IS the cache (clearly marked).

# ---------------------------------------------------------------------------
# Missing API key — must fail cleanly, no network call attempted
# ---------------------------------------------------------------------------
rmd.clear_cache()
with patch.dict(os.environ, {"TWELVEDATA_API_KEY": ""}, clear=False):
    with patch("real_market_data._SESSION.get") as mock_get:
        r = rmd.get_real_stock_price("AAPL")
        check("missing key: reports ok=False", r["ok"] is False)
        check("missing key: specific error code", r.get("error") == "missing_twelvedata_api_key")
        check("missing key: no network call attempted", mock_get.call_count == 0)

# ---------------------------------------------------------------------------
# Successful /quote response (shape taken from Twelve Data's documented example)
# ---------------------------------------------------------------------------
QUOTE_SUCCESS = {
    "symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "currency": "USD",
    "datetime": "2026-10-01", "open": "148.44", "high": "148.96", "low": "147.22",
    "close": "148.85", "volume": "67903927", "previous_close": "149.09",
    "change": "-0.23", "percent_change": "-0.16",
    "fifty_two_week": {"low": "103.10", "high": "157.25"},
}
rmd.clear_cache()
with patch.dict(os.environ, {"TWELVEDATA_API_KEY": "fake-key-for-test"}, clear=False):
    with patch("real_market_data._SESSION.get", return_value=_mock_response(QUOTE_SUCCESS)) as mock_get:
        r = rmd.get_real_stock_price("AAPL")
        check("quote success: ok=True", r["ok"] is True)
        check("quote success: price parsed as float", r["data"]["last_price_usd"] == 148.85)
        check("quote success: pct change parsed", r["data"]["day_change_pct"] == -0.16)
        check("quote success: 52wk range parsed", r["data"]["fifty_two_week_low"] == 103.10
              and r["data"]["fifty_two_week_high"] == 157.25)
        check("quote success: apikey param sent", mock_get.call_args.kwargs["params"]["apikey"] == "fake-key-for-test")
        check("quote success: correct endpoint path", mock_get.call_args.args[0].endswith("/quote"))

        # Cache behavior: a second call for the SAME ticker within the TTL
        # window must be served from memory, not a second network call.
        r_again = rmd.get_real_stock_price("AAPL")
        check("cache: repeated quote call is a cache hit (no 2nd network call)", mock_get.call_count == 1)
        check("cache: cached result still correct", r_again["data"]["last_price_usd"] == 148.85)

        # clear_cache() must actually force a fresh network call again.
        rmd.clear_cache()
        rmd.get_real_stock_price("AAPL")
        check("cache: clear_cache() forces a fresh network call", mock_get.call_count == 2)

# ---------------------------------------------------------------------------
# Plan-restricted /profile response (free-tier key hitting a paid-only
# endpoint) — this is the EXPECTED, common case for anyone using a free
# Twelve Data key, and must be reported as a clean tool failure, never as
# a fabricated profile.
# ---------------------------------------------------------------------------
PROFILE_PLAN_RESTRICTED = {
    "code": 403,
    "message": "This endpoint is not available under your current subscription plan. Please upgrade.",
    "status": "error",
}
rmd.clear_cache()
with patch.dict(os.environ, {"TWELVEDATA_API_KEY": "fake-key-for-test"}, clear=False):
    with patch("real_market_data._SESSION.get", return_value=_mock_response(PROFILE_PLAN_RESTRICTED)) as mock_get:
        r = rmd.get_real_company_profile("AAPL")
        check("profile plan-restricted: ok=False", r["ok"] is False)
        check("profile plan-restricted: error labeled plan_restricted",
              str(r.get("error", "")).startswith("plan_restricted"))
        check("profile plan-restricted: tool field is get_company_profile", r.get("tool") == "get_company_profile")

        r2 = rmd.get_real_company_financials("AAPL")
        check("financials plan-restricted: ok=False (reuses /profile)", r2["ok"] is False)
        check("financials plan-restricted: error labeled plan_restricted",
              str(r2.get("error", "")).startswith("plan_restricted"))
        check("financials plan-restricted: tool field relabeled to get_company_financials",
              r2.get("tool") == "get_company_financials")

        # Dedup check: get_real_company_profile and get_real_company_financials
        # both read Twelve Data's /profile endpoint for the same ticker — the
        # cache should mean only ONE of those two calls actually hit the
        # (mocked) network, not two identical requests.
        check("cache: profile+financials share one /profile call, not two", mock_get.call_count == 1)

# ---------------------------------------------------------------------------
# Successful /profile response
# ---------------------------------------------------------------------------
PROFILE_SUCCESS = {
    "symbol": "AAPL", "name": "Apple Inc", "sector": "Technology",
    "industry": "Consumer Electronics",
    "description": "Apple Inc. designs, manufactures, and markets smartphones...",
    "city": "Cupertino", "country": "United States", "employees": 164000,
}
rmd.clear_cache()
with patch.dict(os.environ, {"TWELVEDATA_API_KEY": "fake-key-for-test"}, clear=False):
    with patch("real_market_data._SESSION.get", return_value=_mock_response(PROFILE_SUCCESS)):
        r = rmd.get_real_company_profile("AAPL")
        check("profile success: ok=True", r["ok"] is True)
        check("profile success: sector parsed", r["data"]["sector"] == "Technology")
        check("profile success: hq combines city+country", r["data"]["hq"] == "Cupertino, United States")

# ---------------------------------------------------------------------------
# Unknown ticker / bad symbol
# ---------------------------------------------------------------------------
BAD_SYMBOL = {"code": 400, "message": "**symbol** not found: ZZZZ", "status": "error"}
rmd.clear_cache()
with patch.dict(os.environ, {"TWELVEDATA_API_KEY": "fake-key-for-test"}, clear=False):
    with patch("real_market_data._SESSION.get", return_value=_mock_response(BAD_SYMBOL)):
        r = rmd.get_real_stock_price("ZZZZ")
        check("bad symbol: ok=False", r["ok"] is False)
        check("bad symbol: not misclassified as plan_restricted",
              not str(r.get("error", "")).startswith("plan_restricted"))

# ---------------------------------------------------------------------------
# Network error (timeout, DNS failure, connection reset, ...)
# ---------------------------------------------------------------------------
import requests as _requests  # noqa: E402  (imported late on purpose, mirrors real_market_data's own import)

rmd.clear_cache()
with patch.dict(os.environ, {"TWELVEDATA_API_KEY": "fake-key-for-test"}, clear=False):
    with patch("real_market_data._SESSION.get", side_effect=_requests.exceptions.Timeout("timed out")) as mock_get:
        r = rmd.get_real_stock_price("AAPL")
        check("network error: ok=False, never raises", r["ok"] is False)
        check("network error: labeled as network_error", str(r.get("error", "")).startswith("network_error"))

        # A network error must NEVER be cached — a transient blip shouldn't
        # make every call for the rest of the TTL window fail too.
        rmd.get_real_stock_price("AAPL")
        check("cache: network errors are never cached (2nd call retries)", mock_get.call_count == 2)

# ---------------------------------------------------------------------------
# Session reuse — a single shared connection pool, not a fresh one per call.
# ---------------------------------------------------------------------------
check("session: _SESSION is a reused requests.Session instance",
      isinstance(rmd._SESSION, _requests.Session))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_pass = sum(1 for _, s in results if s == PASS)
n_fail = len(results) - n_pass
print(f"\n{'='*60}\n{n_pass} passed, {n_fail} failed out of {len(results)} checks\n{'='*60}")
if n_fail:
    raise SystemExit(1)
