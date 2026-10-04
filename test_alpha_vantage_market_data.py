"""
test_alpha_vantage_market_data.py
-----------------------------------
Unit tests for alpha_vantage_market_data.py's request/response handling,
using mocked `requests.get` calls (same reasoning as
test_real_market_data.py / test_fcs_market_data.py: this sandbox has no
outbound network access to www.alphavantage.co, so these verify PARSING
and ERROR-HANDLING against response shapes taken from Alpha Vantage's own
published documentation and GAAP fundamentals field reference).

Run with: python3 test_alpha_vantage_market_data.py
"""

import os
from unittest.mock import patch, MagicMock

import alpha_vantage_market_data as av

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
av.clear_cache()
with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": ""}, clear=False):
    with patch("alpha_vantage_market_data._SESSION.get") as mock_get:
        r = av.get_alpha_vantage_stock_price("AAPL")
        check("missing key: reports ok=False", r["ok"] is False)
        check("missing key: specific error code", r.get("error") == "missing_alpha_vantage_api_key")
        check("missing key: no network call attempted", mock_get.call_count == 0)

# ---------------------------------------------------------------------------
# Successful GLOBAL_QUOTE response (shape taken from Alpha Vantage's
# documented example -- numbered string keys under "Global Quote")
# ---------------------------------------------------------------------------
QUOTE_SUCCESS = {
    "Global Quote": {
        "01. symbol": "AAPL",
        "02. open": "255.95",
        "03. high": "257.40",
        "04. low": "255.43",
        "05. price": "256.48",
        "06. volume": "31955689",
        "07. latest trading day": "2025-10-07",
        "08. previous close": "256.69",
        "09. change": "-0.21",
        "10. change percent": "-0.0818%",
    }
}
av.clear_cache()
with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "fake-key"}, clear=False):
    with patch("alpha_vantage_market_data._SESSION.get", return_value=_mock_response(QUOTE_SUCCESS)) as mock_get:
        r = av.get_alpha_vantage_stock_price("AAPL")
        check("quote success: ok=True", r["ok"] is True)
        check("quote success: price parsed", r["data"]["last_price_usd"] == 256.48)
        check("quote success: pct change parsed (% suffix stripped)", r["data"]["day_change_pct"] == -0.0818)
        check("quote success: volume parsed", r["data"]["volume"] == 31955689.0)
        check("quote success: as_of parsed", r["data"]["as_of"] == "2025-10-07")
        check("quote success: apikey param sent", mock_get.call_args.kwargs["params"]["apikey"] == "fake-key")
        check("quote success: function param sent", mock_get.call_args.kwargs["params"]["function"] == "GLOBAL_QUOTE")

        # cache behavior
        av.get_alpha_vantage_stock_price("AAPL")
        check("cache: repeated call is a cache hit", mock_get.call_count == 1)
        av.clear_cache()
        av.get_alpha_vantage_stock_price("AAPL")
        check("cache: clear_cache() forces a fresh network call", mock_get.call_count == 2)

# ---------------------------------------------------------------------------
# Successful OVERVIEW response (shape taken from Alpha Vantage's documented
# example + GAAP fundamentals field reference) -- shared by profile AND
# financials, each reading different fields out of the same payload.
# ---------------------------------------------------------------------------
OVERVIEW_SUCCESS = {
    "Symbol": "AAPL", "AssetType": "Common Stock", "Name": "Apple Inc",
    "Description": "Apple Inc. designs, manufactures, and markets smartphones...",
    "Exchange": "NASDAQ", "Currency": "USD", "Country": "USA",
    "Sector": "TECHNOLOGY", "Industry": "ELECTRONIC COMPUTERS",
    "Address": "ONE APPLE PARK WAY, CUPERTINO, CA, US",
    "OfficialSite": "https://www.apple.com",
    "FiscalYearEnd": "September", "LatestQuarter": "2024-09-30",
    "MarketCapitalization": "3806265792000", "EBITDA": "134661001000",
    "PERatio": "183.96", "ProfitMargin": "0.2402",
    "RevenueTTM": "391035000000", "GrossProfitTTM": "180683000000",
    "OperatingMarginTTM": "0.3151", "QuarterlyRevenueGrowthYOY": "0.0202",
    "QuarterlyEarningsGrowthYOY": "0.121",
    "AnalystTargetPrice": "260.0",
}
av.clear_cache()
with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "fake-key"}, clear=False):
    with patch("alpha_vantage_market_data._SESSION.get", return_value=_mock_response(OVERVIEW_SUCCESS)) as mock_get:
        r = av.get_alpha_vantage_company_profile("AAPL")
        check("profile success: ok=True", r["ok"] is True)
        check("profile success: name parsed", r["data"]["name"] == "Apple Inc")
        check("profile success: sector parsed", r["data"]["sector"] == "TECHNOLOGY")
        check("profile success: hq combines Address+Country",
              r["data"]["hq"] == "ONE APPLE PARK WAY, CUPERTINO, CA, US, USA")
        check("profile success: website parsed", r["data"]["website"] == "https://www.apple.com")
        check("profile success: market cap converted to billions",
              abs(r["data"]["market_cap_usd_b"] - 3806.265792) < 0.001)
        check("profile success: no employees field (Alpha Vantage doesn't have one)",
              "employees" not in r["data"])
        check("function param sent for OVERVIEW", mock_get.call_args.kwargs["params"]["function"] == "OVERVIEW")

        # financials reuses the SAME cached OVERVIEW call -- no second network hit
        r2 = av.get_alpha_vantage_company_financials("AAPL")
        check("financials success: ok=True", r2["ok"] is True)
        check("financials: reuses cached OVERVIEW (no extra network call)", mock_get.call_count == 1)
        check("financials: revenue converted to millions", abs(r2["data"]["revenue_usd_m"] - 391035.0) < 0.01)
        check("financials: net margin converted from fraction to percent",
              abs(r2["data"]["net_margin_pct"] - 24.02) < 0.001)
        check("financials: gross margin computed from GrossProfitTTM/RevenueTTM",
              abs(r2["data"]["gross_margin_pct"] - (180683000000 / 391035000000 * 100)) < 1e-6)
        check("financials: operating margin converted from fraction to percent",
              abs(r2["data"]["operating_margin_pct"] - 31.51) < 0.001)
        check("financials: revenue growth converted from fraction to percent",
              abs(r2["data"]["revenue_growth_yoy_pct"] - 2.02) < 0.001)
        check("financials: pe ratio parsed", abs(r2["data"]["pe_ratio"] - 183.96) < 1e-6)
        check("financials: ebitda converted to millions", abs(r2["data"]["ebitda_usd_m"] - 134661.001) < 0.01)
        check("financials: net income calculated (RevenueTTM x ProfitMargin)",
              abs(r2["data"]["net_income_usd_m"] - (391035000000 * 0.2402 / 1e6)) < 0.01)
        check("financials: fiscal_year parsed from LatestQuarter", r2["data"]["fiscal_year"] == "2024-09-30")
        check("financials: note labels net_income_usd_m as calculated", "calculated" in r2["data"]["note"])

# ---------------------------------------------------------------------------
# "None" string fields -- Alpha Vantage uses the literal string "None" for
# a field it has no data for, not JSON null.
# ---------------------------------------------------------------------------
OVERVIEW_PARTIAL = {
    "Symbol": "XYZ", "Name": "Some Company", "Sector": "None", "Industry": "None",
    "Address": "None", "Country": "None", "OfficialSite": "None",
    "MarketCapitalization": "None", "EBITDA": "None", "PERatio": "None",
    "ProfitMargin": "None", "RevenueTTM": "None", "GrossProfitTTM": "None",
    "OperatingMarginTTM": "None", "QuarterlyRevenueGrowthYOY": "None",
    "LatestQuarter": "None",
}
av.clear_cache()
with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "fake-key"}, clear=False):
    with patch("alpha_vantage_market_data._SESSION.get", return_value=_mock_response(OVERVIEW_PARTIAL)):
        r = av.get_alpha_vantage_company_profile("XYZ")
        check("partial overview: ok=True (name still present)", r["ok"] is True)
        check("partial overview: 'None' string parsed as missing, not literal", r["data"]["sector"] is None)
        check("partial overview: hq is None when both Address/Country are 'None'", r["data"]["hq"] is None)

        r2 = av.get_alpha_vantage_company_financials("XYZ")
        check("partial financials: ok=False when every numeric field is 'None'", r2["ok"] is False)

# ---------------------------------------------------------------------------
# Rate limiting -- Alpha Vantage signals this with a "Note" key inside a
# 200 OK body, not an HTTP status code.
# ---------------------------------------------------------------------------
RATE_LIMITED = {
    "Note": "Thank you for using Alpha Vantage! Our standard API rate limit is "
    "25 requests per day. Please visit https://www.alphavantage.co/premium/ ..."
}
av.clear_cache()
with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "fake-key"}, clear=False):
    with patch("alpha_vantage_market_data._SESSION.get", return_value=_mock_response(RATE_LIMITED)):
        r = av.get_alpha_vantage_stock_price("AAPL")
        check("rate limited: ok=False", r["ok"] is False)
        check("rate limited: labeled as rate_limited", str(r.get("error", "")).startswith("rate_limited"))

# ---------------------------------------------------------------------------
# Invalid / demo API key -- Alpha Vantage signals this with an
# "Information" key inside a 200 OK body.
# ---------------------------------------------------------------------------
DEMO_KEY_REJECTED = {
    "Information": "The **demo** API key is for demo purposes only. "
    "Please claim your own API key at https://www.alphavantage.co/support/#api-key"
}
av.clear_cache()
with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "demo"}, clear=False):
    with patch("alpha_vantage_market_data._SESSION.get", return_value=_mock_response(DEMO_KEY_REJECTED)):
        r = av.get_alpha_vantage_company_profile("IBM")
        check("demo key: ok=False, never raises", r["ok"] is False)
        check("demo key: labeled as invalid_api_key", str(r.get("error", "")).startswith("invalid_api_key"))

# ---------------------------------------------------------------------------
# Bad symbol -- Alpha Vantage's OVERVIEW returns an empty {} for an
# unrecognized symbol (no error keys at all).
# ---------------------------------------------------------------------------
av.clear_cache()
with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "fake-key"}, clear=False):
    with patch("alpha_vantage_market_data._SESSION.get", return_value=_mock_response({})):
        r = av.get_alpha_vantage_company_profile("NOTAREALTICKER")
        check("bad symbol: ok=False, never raises", r["ok"] is False)
        check("bad symbol: labeled as empty_response", r.get("error") == "empty_response")

# ---------------------------------------------------------------------------
# Network error
# ---------------------------------------------------------------------------
import requests as _requests  # noqa: E402

av.clear_cache()
with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "fake-key"}, clear=False):
    with patch("alpha_vantage_market_data._SESSION.get", side_effect=_requests.exceptions.Timeout("timed out")) as mock_get:
        r = av.get_alpha_vantage_stock_price("AAPL")
        check("network error: ok=False, never raises", r["ok"] is False)
        check("network error: labeled as network_error", str(r.get("error", "")).startswith("network_error"))
        av.get_alpha_vantage_stock_price("AAPL")
        check("cache: network errors are never cached (2nd call retries)", mock_get.call_count == 2)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_pass = sum(1 for _, s in results if s == PASS)
n_fail = len(results) - n_pass
print(f"\n{'='*60}\n{n_pass} passed, {n_fail} failed out of {len(results)} checks\n{'='*60}")
if n_fail:
    raise SystemExit(1)
