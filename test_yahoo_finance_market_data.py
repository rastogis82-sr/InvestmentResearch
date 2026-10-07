"""
test_yahoo_finance_market_data.py
------------------------------------
Unit tests for yahoo_finance_market_data.py's request/response handling,
using a mocked `yfinance.Ticker` (same reasoning as the other market-data
modules' tests: this sandbox has no outbound network access to Yahoo
Finance's endpoints -- and, unlike Twelve Data/Alpha Vantage/FCS API,
couldn't even install the `yfinance` package itself from PyPI to inspect
it directly while building this module, confirmed while building it). The
module-level `yf` name is patched wholesale (`patch("yahoo_finance_market_data.yf")`)
rather than `yf.Ticker`, since this sandbox's own `yf` is actually None
(the real import failed here) -- patching the whole name works regardless
of whether a real yfinance package is present in whatever environment
runs this file.

Run with: python3 test_yahoo_finance_market_data.py
"""

import os
from unittest.mock import patch, MagicMock

import yahoo_finance_market_data as yfmd

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, condition: bool, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status))
    print(f"[{status}] {name}" + (f" — {detail}" if detail and status == FAIL else ""))


def _mock_ticker(info: dict):
    """Builds a MagicMock standing in for yfinance's `yf` module, whose
    `Ticker(symbol).info` returns the given dict -- the same shape a real
    yfinance.Ticker instance's `.info` property returns."""
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value.info = info
    return mock_yf


# ---------------------------------------------------------------------------
# yfinance not installed/importable -- must fail cleanly, never raise
# ---------------------------------------------------------------------------
yfmd.clear_cache()
with patch("yahoo_finance_market_data.yf", None):
    r = yfmd.get_yahoo_stock_price("AAPL")
    check("missing yfinance: reports ok=False", r["ok"] is False)
    check("missing yfinance: specific error code", str(r.get("error", "")).startswith("missing_yfinance_package"))

# ---------------------------------------------------------------------------
# Successful info response -- shape taken from yfinance's long-stable
# Ticker.info dict (price fields already in percent units; margin/growth
# fields as 0-1 fractions -- see the module docstring on this distinction).
# ---------------------------------------------------------------------------
AAPL_INFO = {
    "symbol": "AAPL", "longName": "Apple Inc.", "shortName": "Apple Inc.",
    "sector": "Technology", "industry": "Consumer Electronics",
    "longBusinessSummary": "Apple Inc. designs, manufactures, and markets smartphones...",
    "fullTimeEmployees": 164000, "website": "https://www.apple.com",
    "city": "Cupertino", "state": "CA", "country": "United States",
    "currentPrice": 256.48, "regularMarketPrice": 256.48,
    "regularMarketChangePercent": -0.127, "regularMarketVolume": 31955689,
    "marketCap": 3806265792000, "trailingPE": 183.96,
    "profitMargins": 0.2402, "grossMargins": 0.4622, "operatingMargins": 0.3151,
    "totalRevenue": 391035000000, "revenueGrowth": 0.0202,
    "ebitda": 134661001000, "netIncomeToCommon": 93736000000,
}
yfmd.clear_cache()
with patch("yahoo_finance_market_data.yf", _mock_ticker(AAPL_INFO)) as mock_yf:
    r = yfmd.get_yahoo_stock_price("AAPL")
    check("price success: ok=True", r["ok"] is True)
    check("price success: price parsed", r["data"]["last_price_usd"] == 256.48)
    check("price success: pct change NOT multiplied by 100 (already percent-shaped)",
          r["data"]["day_change_pct"] == -0.127)
    check("price success: volume parsed", r["data"]["volume"] == 31955689.0)
    check("price success: Ticker() called with uppercased symbol",
          mock_yf.Ticker.call_args.args[0] == "AAPL")

    # cache behavior -- shared across price/profile/financials for the
    # SAME ticker, see the module docstring.
    yfmd.get_yahoo_stock_price("AAPL")
    check("cache: repeated call is a cache hit", mock_yf.Ticker.call_count == 1)

    r2 = yfmd.get_yahoo_company_profile("AAPL")
    check("profile success: ok=True", r2["ok"] is True)
    check("profile success: reuses cached info (no extra Ticker() call)", mock_yf.Ticker.call_count == 1)
    check("profile success: name parsed", r2["data"]["name"] == "Apple Inc.")
    check("profile success: sector parsed", r2["data"]["sector"] == "Technology")
    check("profile success: hq combines city+state+country", r2["data"]["hq"] == "Cupertino, CA, United States")
    check("profile success: employees parsed (available here, unlike Alpha Vantage)",
          r2["data"]["employees"] == 164000)
    check("profile success: website parsed", r2["data"]["website"] == "https://www.apple.com")
    check("profile success: market cap converted to billions",
          abs(r2["data"]["market_cap_usd_b"] - 3806.265792) < 0.001)

    r3 = yfmd.get_yahoo_company_financials("AAPL")
    check("financials success: ok=True", r3["ok"] is True)
    check("financials success: reuses cached info (still no extra Ticker() call)", mock_yf.Ticker.call_count == 1)
    check("financials: revenue converted to millions", abs(r3["data"]["revenue_usd_m"] - 391035.0) < 0.01)
    check("financials: net margin converted from fraction to percent",
          abs(r3["data"]["net_margin_pct"] - 24.02) < 0.001)
    check("financials: gross margin converted from fraction to percent",
          abs(r3["data"]["gross_margin_pct"] - 46.22) < 0.001)
    check("financials: operating margin converted from fraction to percent",
          abs(r3["data"]["operating_margin_pct"] - 31.51) < 0.001)
    check("financials: revenue growth converted from fraction to percent",
          abs(r3["data"]["revenue_growth_yoy_pct"] - 2.02) < 0.001)
    check("financials: pe ratio parsed", abs(r3["data"]["pe_ratio"] - 183.96) < 1e-6)
    check("financials: ebitda converted to millions", abs(r3["data"]["ebitda_usd_m"] - 134661.001) < 0.01)
    check("financials: net income taken directly, not calculated",
          abs(r3["data"]["net_income_usd_m"] - 93736.0) < 0.01)

    yfmd.clear_cache()
    yfmd.get_yahoo_stock_price("AAPL")
    check("cache: clear_cache() forces a fresh Ticker() call", mock_yf.Ticker.call_count == 2)

# ---------------------------------------------------------------------------
# Bad/unrecognized ticker -- yfinance doesn't raise, it returns an info
# dict with only a couple of placeholder keys and no real company data.
# ---------------------------------------------------------------------------
yfmd.clear_cache()
with patch("yahoo_finance_market_data.yf", _mock_ticker({"trailingPegRatio": None})):
    r = yfmd.get_yahoo_company_profile("NOTAREALTICKER")
    check("bad ticker: ok=False, never raises", r["ok"] is False)
    check("bad ticker: labeled as empty_response", r.get("error") == "empty_response")

# ---------------------------------------------------------------------------
# Rate limiting -- yfinance raises a real exception rather than returning
# a structured error; classify by exception type/message.
# ---------------------------------------------------------------------------
class _FakeRateLimitError(Exception):
    pass


yfmd.clear_cache()
mock_yf_rate_limited = MagicMock()
mock_yf_rate_limited.Ticker.side_effect = _FakeRateLimitError("429 Client Error: Too Many Requests")
with patch("yahoo_finance_market_data.yf", mock_yf_rate_limited):
    r = yfmd.get_yahoo_stock_price("AAPL")
    check("rate limited: ok=False, never raises", r["ok"] is False)
    check("rate limited: labeled as rate_limited", str(r.get("error", "")).startswith("rate_limited"))

# ---------------------------------------------------------------------------
# Network error -- a generic exception not matching the rate-limit
# heuristic falls back to network_error, not an unhandled crash.
# ---------------------------------------------------------------------------
yfmd.clear_cache()
mock_yf_network_error = MagicMock()
mock_yf_network_error.Ticker.side_effect = ConnectionError("Connection refused")
with patch("yahoo_finance_market_data.yf", mock_yf_network_error) as mock_yf:
    r = yfmd.get_yahoo_stock_price("AAPL")
    check("network error: ok=False, never raises", r["ok"] is False)
    check("network error: labeled as network_error", str(r.get("error", "")).startswith("network_error"))
    yfmd.get_yahoo_stock_price("AAPL")
    check("cache: network errors are never cached (2nd call retries)", mock_yf.Ticker.call_count == 2)

# ---------------------------------------------------------------------------
# Partial data -- some fields present, others missing (None) -- never
# fabricated, just honestly absent from the result dict.
# ---------------------------------------------------------------------------
PARTIAL_INFO = {
    "symbol": "XYZ", "longName": "Some Company", "sector": "Industrials",
    "industry": None, "city": None, "state": None, "country": None,
    "longBusinessSummary": None, "fullTimeEmployees": None, "website": None,
    "currentPrice": 42.0, "marketCap": None, "totalRevenue": None,
    "profitMargins": None, "trailingPE": None,
}
yfmd.clear_cache()
with patch("yahoo_finance_market_data.yf", _mock_ticker(PARTIAL_INFO)):
    r = yfmd.get_yahoo_company_profile("XYZ")
    check("partial profile: ok=True (name/sector still present)", r["ok"] is True)
    check("partial profile: industry absent, not fabricated", r["data"]["industry"] is None)
    check("partial profile: hq is None when city/state/country all missing", r["data"]["hq"] is None)

    r2 = yfmd.get_yahoo_company_financials("XYZ")
    check("partial financials: ok=False when every numeric field is missing", r2["ok"] is False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_pass = sum(1 for _, s in results if s == PASS)
n_fail = len(results) - n_pass
print(f"\n{'='*60}\n{n_pass} passed, {n_fail} failed out of {len(results)} checks\n{'='*60}")
if n_fail:
    raise SystemExit(1)
