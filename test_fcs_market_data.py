"""
test_fcs_market_data.py
-------------------------
Unit tests for fcs_market_data.py's request/response handling, using
mocked `requests.get` calls (same reasoning as test_real_market_data.py:
this sandbox has no outbound network access to api-v4.fcsapi.com, so
these verify PARSING and ERROR-HANDLING against response shapes taken
from FCS API's own published documentation examples).

Run with: python3 test_fcs_market_data.py
"""

import os
from unittest.mock import patch, MagicMock

import fcs_market_data as fcs

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, condition: bool, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status))
    print(f"[{status}] {name}" + (f" — {detail}" if detail and status == FAIL else ""))


def _mock_response(json_body, status_code=200):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_body
    return m


# ---------------------------------------------------------------------------
# Missing API key -- must fail cleanly, no network call attempted
# ---------------------------------------------------------------------------
fcs.clear_cache()
with patch.dict(os.environ, {"FCS_API_KEY": ""}, clear=False):
    with patch("fcs_market_data._SESSION.get") as mock_get:
        r = fcs.get_fcs_stock_price("AAPL")
        check("missing key: reports ok=False", r["ok"] is False)
        check("missing key: specific error code", r.get("error") == "missing_fcs_api_key")
        check("missing key: no network call attempted", mock_get.call_count == 0)

# ---------------------------------------------------------------------------
# Successful /stock/latest response (shape taken from FCS API's documented
# example -- a one-item list under "response", nested active/previous)
# ---------------------------------------------------------------------------
LATEST_SUCCESS = {
    "status": True, "code": 200, "msg": "Successfully",
    "response": [{
        "ticker": "NASDAQ:AAPL",
        "update": 1759881596,
        "updateTime": "2025-10-07 23:59:56",
        "active": {
            "o": 256.805, "h": 257.4, "l": 255.43, "c": 256.48,
            "v": 31955689, "t": 1759843800, "vw": 256.437,
            "tm": "2025-10-07 13:30:00", "ch": -0.325, "chp": -0.127,
        },
        "previous": {"c": 256.69, "chp": -0.504},
    }],
    "info": {"server_time": "2025-10-08 13:01:45 UTC", "credit_count": 1},
}
fcs.clear_cache()
with patch.dict(os.environ, {"FCS_API_KEY": "fake-key"}, clear=False):
    with patch("fcs_market_data._SESSION.get", return_value=_mock_response(LATEST_SUCCESS)) as mock_get:
        r = fcs.get_fcs_stock_price("AAPL")
        check("quote success: ok=True", r["ok"] is True)
        check("quote success: price parsed", r["data"]["last_price_usd"] == 256.48)
        check("quote success: pct change parsed (already a percent, not a fraction)",
              r["data"]["day_change_pct"] == -0.127)
        check("quote success: volume parsed", r["data"]["volume"] == 31955689.0)
        check("quote success: access_key param sent", mock_get.call_args.kwargs["params"]["access_key"] == "fake-key")
        check("quote success: exchange-qualified symbol sent",
              mock_get.call_args.kwargs["params"]["symbol"] == "NASDAQ:AAPL")

        # cache behavior
        fcs.get_fcs_stock_price("AAPL")
        check("cache: repeated call is a cache hit", mock_get.call_count == 1)
        fcs.clear_cache()
        fcs.get_fcs_stock_price("AAPL")
        check("cache: clear_cache() forces a fresh network call", mock_get.call_count == 2)

# ---------------------------------------------------------------------------
# Successful /stock/profile response (shape taken from FCS API's documented
# example)
# ---------------------------------------------------------------------------
PROFILE_SUCCESS = {
    "status": True, "code": 200, "msg": "Successfully",
    "response": [{
        "ticker": "NASDAQ:AAPL",
        "profile": {
            "primary_name": "NASDAQ:AAPL", "symbol": "NASDAQ:AAPL",
            "description": "Apple, Inc. engages in the design, manufacture, and sale...",
            "market_cap": 3806265792000, "isin": "US0378331005",
            "timezone": "America/New_York", "ceo": "Timothy Donald Cook",
            "total_employees": 164000, "founded": 1976,
            "website": "http://www.apple.com", "headquarters": "Cupertino",
            "region": "CA", "sector": "Electronic Technology",
            "industry": "Telecommunications Equipment",
        },
    }],
    "info": {"server_time": "2025-10-01 12:15:11 UTC", "credit_count": 1},
}
fcs.clear_cache()
with patch.dict(os.environ, {"FCS_API_KEY": "fake-key"}, clear=False):
    with patch("fcs_market_data._SESSION.get", return_value=_mock_response(PROFILE_SUCCESS)):
        r = fcs.get_fcs_company_profile("AAPL")
        check("profile success: ok=True", r["ok"] is True)
        check("profile success: sector parsed", r["data"]["sector"] == "Electronic Technology")
        check("profile success: hq combines headquarters+region", r["data"]["hq"] == "Cupertino, CA")
        check("profile success: employees parsed", r["data"]["employees"] == 164000)
        check("profile success: founded kept as string (never comma-formatted)", r["data"]["founded"] == "1976")
        check("profile success: ceo parsed", r["data"]["ceo"] == "Timothy Donald Cook")
        check("profile success: market cap converted to billions",
              abs(r["data"]["market_cap_usd_b"] - 3806.265792) < 0.001)

# ---------------------------------------------------------------------------
# Successful /stock/statistics + /stock/income_statements (direct-object
# "response" shape, per FCS API's documented examples for these two)
# ---------------------------------------------------------------------------
STATS_SUCCESS = {
    "status": True, "code": 200, "msg": "Successfully",
    "response": {
        "market_cap_basic": 1024135840000, "total_shares_outstanding": 3224000000,
        "price_earnings": 183.958767662729, "price_sales": 12.0561425798102,
        "gross_margin": 17.2386201991465, "operating_margin": 4.10295163584637,
        "ebitda_margin": 10.472972972973, "net_margin": 5.20981507823613,
    },
}
INCOME_SUCCESS = {
    "status": True, "code": 200, "msg": "Successfully",
    "response": {
        "2024-09-30": {
            "total_revenue": 391035000000,
            "net_income": 93736000000,
            "earnings_per_share_basic": 6.109,
        },
        "2023-09-30": {
            "total_revenue": 383285000000,
            "net_income": 96995000000,
        },
    },
}
fcs.clear_cache()
with patch.dict(os.environ, {"FCS_API_KEY": "fake-key"}, clear=False):
    def _route(url, params, timeout):
        if url.endswith("/stock/statistics"):
            return _mock_response(STATS_SUCCESS)
        if url.endswith("/stock/income_statements"):
            return _mock_response(INCOME_SUCCESS)
        raise AssertionError(f"unexpected path called: {url}")

    with patch("fcs_market_data._SESSION.get", side_effect=lambda url, params, timeout: _route(url, params, timeout)):
        r = fcs.get_fcs_company_financials("AAPL")
        check("financials success: ok=True", r["ok"] is True)
        check("financials: net margin parsed", abs(r["data"]["net_margin_pct"] - 5.20981507823613) < 1e-6)
        check("financials: gross margin parsed", abs(r["data"]["gross_margin_pct"] - 17.2386201991465) < 1e-6)
        check("financials: pe ratio parsed", abs(r["data"]["pe_ratio"] - 183.958767662729) < 1e-6)
        check("financials: revenue converted to millions", abs(r["data"]["revenue_usd_m"] - 391035.0) < 0.01)
        check("financials: net income converted to millions", abs(r["data"]["net_income_usd_m"] - 93736.0) < 0.01)
        check("financials: fiscal_year is the latest period", r["data"]["fiscal_year"] == "2024-09-30")
        # YoY: (391035000000 - 383285000000) / 383285000000 * 100
        expected_growth = (391035000000 - 383285000000) / 383285000000 * 100
        check("financials: YoY revenue growth computed from two periods",
              abs(r["data"]["revenue_growth_yoy_pct"] - expected_growth) < 1e-6)

# ---------------------------------------------------------------------------
# Partial success: statistics fails (plan-restricted), income_statements
# succeeds -- must still return ok=True with whatever it has, plus a note.
# ---------------------------------------------------------------------------
STATS_PLAN_RESTRICTED = {"status": False, "code": 403, "msg": "Please upgrade your subscription plan."}
fcs.clear_cache()
with patch.dict(os.environ, {"FCS_API_KEY": "fake-key"}, clear=False):
    def _route2(url, params, timeout):
        if url.endswith("/stock/statistics"):
            return _mock_response(STATS_PLAN_RESTRICTED)
        if url.endswith("/stock/income_statements"):
            return _mock_response(INCOME_SUCCESS)
        raise AssertionError(f"unexpected path called: {url}")

    with patch("fcs_market_data._SESSION.get", side_effect=lambda url, params, timeout: _route2(url, params, timeout)):
        r = fcs.get_fcs_company_financials("AAPL")
        check("partial financials: still ok=True (income succeeded)", r["ok"] is True)
        check("partial financials: has revenue (from income_statements)", "revenue_usd_m" in r["data"])
        check("partial financials: no net_margin_pct (statistics failed)", "net_margin_pct" not in r["data"])
        check("partial financials: note mentions the partial failure", "note" in r["data"])

# ---------------------------------------------------------------------------
# Rate limiting -- FCS API's free plan is 3 requests/minute, must be
# reported distinctly from a generic API error, not misclassified.
# ---------------------------------------------------------------------------
RATE_LIMITED = {"status": False, "code": 429, "msg": "Too Many Requests. Rate limit exceeded."}
fcs.clear_cache()
with patch.dict(os.environ, {"FCS_API_KEY": "fake-key"}, clear=False):
    with patch("fcs_market_data._SESSION.get", return_value=_mock_response(RATE_LIMITED)):
        r = fcs.get_fcs_stock_price("AAPL")
        check("rate limited: ok=False", r["ok"] is False)
        check("rate limited: labeled as rate_limited", str(r.get("error", "")).startswith("rate_limited"))

# ---------------------------------------------------------------------------
# Auth error (bad/expired key) -- the documented FCS API error envelope
# shape differs from a plan-restriction body, both must parse without
# raising.
# ---------------------------------------------------------------------------
AUTH_ERROR = {
    "status": False, "code": 401, "msg": "Invalid API key",
    "error": {"type": "AUTHENTICATION_ERROR", "details": "The provided API key is invalid or expired"},
}
fcs.clear_cache()
with patch.dict(os.environ, {"FCS_API_KEY": "bad-key"}, clear=False):
    with patch("fcs_market_data._SESSION.get", return_value=_mock_response(AUTH_ERROR)):
        r = fcs.get_fcs_stock_price("AAPL")
        check("auth error: ok=False, never raises", r["ok"] is False)
        check("auth error: labeled as plan_restricted (401/403 bucket)",
              str(r.get("error", "")).startswith("plan_restricted"))

# ---------------------------------------------------------------------------
# Network error
# ---------------------------------------------------------------------------
import requests as _requests  # noqa: E402

fcs.clear_cache()
with patch.dict(os.environ, {"FCS_API_KEY": "fake-key"}, clear=False):
    with patch("fcs_market_data._SESSION.get", side_effect=_requests.exceptions.Timeout("timed out")) as mock_get:
        r = fcs.get_fcs_stock_price("AAPL")
        check("network error: ok=False, never raises", r["ok"] is False)
        check("network error: labeled as network_error", str(r.get("error", "")).startswith("network_error"))
        fcs.get_fcs_stock_price("AAPL")
        check("cache: network errors are never cached (2nd call retries)", mock_get.call_count == 2)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_pass = sum(1 for _, s in results if s == PASS)
n_fail = len(results) - n_pass
print(f"\n{'='*60}\n{n_pass} passed, {n_fail} failed out of {len(results)} checks\n{'='*60}")
if n_fail:
    raise SystemExit(1)
