"""
test_merged_market_data.py
----------------------------
Unit tests for merged_market_data.py's field-level merge logic, using
hand-built {"ok": ..., "data": {...}} results (the SAME shape
real_market_data.py / yahoo_finance_market_data.py actually return) rather
than mocking HTTP/yfinance -- this is testing the merge function itself,
not either provider's request/response handling (that's what
test_real_market_data.py and test_yahoo_finance_market_data.py already
cover).

Run with: python3 test_merged_market_data.py
"""

from unittest.mock import patch

import merged_market_data as mmd

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, condition: bool, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status))
    print(f"[{status}] {name}" + (f" — {detail}" if detail and status == FAIL else ""))


# ---------------------------------------------------------------------------
# get_merged_stock_price: Twelve Data succeeds -> Yahoo Finance never called
# ---------------------------------------------------------------------------
TD_PRICE_OK = {"ok": True, "tool": "get_stock_price", "data": {"last_price_usd": 256.48, "source": "twelvedata.com (live)"}}
with patch("merged_market_data.get_real_stock_price", return_value=TD_PRICE_OK) as td_mock, \
     patch("merged_market_data.get_yahoo_stock_price") as yahoo_mock:
    r = mmd.get_merged_stock_price("AAPL")
    check("price: Twelve Data success returned as-is", r == TD_PRICE_OK)
    check("price: Yahoo Finance not called when Twelve Data succeeds", yahoo_mock.call_count == 0)

# ---------------------------------------------------------------------------
# get_merged_stock_price: Twelve Data fails -> Yahoo Finance used as fallback
# ---------------------------------------------------------------------------
TD_PRICE_FAIL = {"ok": False, "tool": "get_stock_price", "error": "network_error:ConnectionError"}
YAHOO_PRICE_OK = {"ok": True, "tool": "get_stock_price", "data": {"last_price_usd": 255.90, "source": "finance.yahoo.com (live, via yfinance)"}}
with patch("merged_market_data.get_real_stock_price", return_value=TD_PRICE_FAIL), \
     patch("merged_market_data.get_yahoo_stock_price", return_value=YAHOO_PRICE_OK) as yahoo_mock:
    r = mmd.get_merged_stock_price("AAPL")
    check("price fallback: ok=True from Yahoo Finance", r["ok"] is True)
    check("price fallback: Yahoo Finance was actually called", yahoo_mock.call_count == 1)
    check("price fallback: price value is Yahoo Finance's", r["data"]["last_price_usd"] == 255.90)
    check("price fallback: note explains why", "Twelve Data unavailable" in r["data"].get("note", ""))

# ---------------------------------------------------------------------------
# get_merged_stock_price: both fail -> ok=False, both errors visible
# ---------------------------------------------------------------------------
YAHOO_PRICE_FAIL = {"ok": False, "tool": "get_stock_price", "error": "rate_limited:_FakeRateLimitError"}
with patch("merged_market_data.get_real_stock_price", return_value=TD_PRICE_FAIL), \
     patch("merged_market_data.get_yahoo_stock_price", return_value=YAHOO_PRICE_FAIL):
    r = mmd.get_merged_stock_price("AAPL")
    check("price both fail: ok=False", r["ok"] is False)
    check("price both fail: error names both providers",
          "twelvedata" in r["error"] and "finance.yahoo.com" in r["error"])

# ---------------------------------------------------------------------------
# get_merged_company_profile: Twelve Data plan-restricted (free tier, the
# common case), Yahoo Finance succeeds -> field-level fill-in, not a
# replacement
# ---------------------------------------------------------------------------
TD_PROFILE_RESTRICTED = {"ok": False, "tool": "get_company_profile", "error": "plan_restricted:upgrade required"}
YAHOO_PROFILE_OK = {
    "ok": True, "tool": "get_company_profile",
    "data": {
        "name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
        "hq": "Cupertino, CA, United States",
        "description": "Apple Inc. designs, manufactures, and markets smartphones...",
        "employees": 164000, "website": "https://www.apple.com", "market_cap_usd_b": 3806.27,
        "source": "finance.yahoo.com (live, via yfinance)",
    },
}
with patch("merged_market_data.get_real_company_profile", return_value=TD_PROFILE_RESTRICTED), \
     patch("merged_market_data.get_yahoo_company_profile", return_value=YAHOO_PROFILE_OK):
    r = mmd.get_merged_company_profile("AAPL")
    check("profile: TD restricted, Yahoo Finance fills entirely -> ok=True", r["ok"] is True)
    check("profile: sector came through from Yahoo Finance", r["data"]["sector"] == "Technology")
    check("profile: employees (Yahoo-only field here) present", r["data"]["employees"] == 164000)
    check("profile: source names only finance.yahoo.com", r["data"]["source"] == "finance.yahoo.com (live)")

# ---------------------------------------------------------------------------
# get_merged_company_profile: BOTH succeed -> Twelve Data's fields win,
# Yahoo Finance only fills what Twelve Data doesn't have (employees,
# website, market cap -- fields real_market_data.get_real_company_profile
# never surfaces at all)
# ---------------------------------------------------------------------------
TD_PROFILE_OK = {
    "ok": True, "tool": "get_company_profile",
    "data": {
        "name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
        "hq": "Cupertino, United States", "description": "Twelve Data's own description.",
        "source": "twelvedata.com (live)",
    },
}
with patch("merged_market_data.get_real_company_profile", return_value=TD_PROFILE_OK), \
     patch("merged_market_data.get_yahoo_company_profile", return_value=YAHOO_PROFILE_OK):
    r = mmd.get_merged_company_profile("AAPL")
    check("profile both ok: Twelve Data's sector wins (not overwritten)", r["data"]["sector"] == "Technology")
    check("profile both ok: Twelve Data's description wins", r["data"]["description"] == "Twelve Data's own description.")
    check("profile both ok: Yahoo-only field (employees) still filled in", r["data"]["employees"] == 164000)
    check("profile both ok: Yahoo-only field (market_cap_usd_b) still filled in", r["data"]["market_cap_usd_b"] == 3806.27)
    check("profile both ok: source names both providers", r["data"]["source"] == "twelvedata.com + finance.yahoo.com (live, merged)")
    check("profile both ok: note lists which fields were filled in from Yahoo Finance",
          "employees" in r["data"]["note"] and "market_cap_usd_b" in r["data"]["note"])

# ---------------------------------------------------------------------------
# get_merged_company_financials: Twelve Data "succeeds" (profile-only
# fields, NO real financials -- see real_market_data.py), Yahoo Finance
# supplies the actual revenue/margin numbers -- this is the exact gap the
# user asked to close.
# ---------------------------------------------------------------------------
TD_FINANCIALS_OK_BUT_THIN = {
    "ok": True, "tool": "get_company_financials",
    "data": {
        "name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
        "employees": None,
        "note": "Twelve Data's free plan has no income-statement fundamentals endpoint...",
        "source": "twelvedata.com (live)",
    },
}
YAHOO_FINANCIALS_OK = {
    "ok": True, "tool": "get_company_financials",
    "data": {
        "net_margin_pct": 24.02, "gross_margin_pct": 46.22, "operating_margin_pct": 31.51,
        "pe_ratio": 183.96, "market_cap_usd_b": 3806.27, "ebitda_usd_m": 134661.0,
        "revenue_usd_m": 391035.0, "net_income_usd_m": 93736.0,
        "revenue_growth_yoy_pct": 2.02,
        "source": "finance.yahoo.com (live, via yfinance)",
    },
}
with patch("merged_market_data.get_real_company_financials", return_value=TD_FINANCIALS_OK_BUT_THIN), \
     patch("merged_market_data.get_yahoo_company_financials", return_value=YAHOO_FINANCIALS_OK):
    r = mmd.get_merged_company_financials("AAPL")
    check("financials: ok=True", r["ok"] is True)
    check("financials: revenue present (only Yahoo Finance has this)", r["data"]["revenue_usd_m"] == 391035.0)
    check("financials: revenue growth present (only Yahoo Finance has this)", r["data"]["revenue_growth_yoy_pct"] == 2.02)
    check("financials: employees=None from TD doesn't block Yahoo Finance fields", "revenue_usd_m" in r["data"])
    check("financials: TD's sector still present (TD field, not overwritten)", r["data"]["sector"] == "Technology")
    check("financials: combined note mentions both the TD limitation and the Yahoo Finance fill-in",
          "income-statement" in r["data"]["note"] and "revenue_usd_m" in r["data"]["note"])

# ---------------------------------------------------------------------------
# get_merged_company_financials: BOTH providers fail -> ok=False, both
# errors visible, never silently empty
# ---------------------------------------------------------------------------
TD_FINANCIALS_FAIL = {"ok": False, "tool": "get_company_financials", "error": "plan_restricted:upgrade"}
YAHOO_FINANCIALS_FAIL = {"ok": False, "tool": "get_company_financials", "error": "empty_response"}
with patch("merged_market_data.get_real_company_financials", return_value=TD_FINANCIALS_FAIL), \
     patch("merged_market_data.get_yahoo_company_financials", return_value=YAHOO_FINANCIALS_FAIL):
    r = mmd.get_merged_company_financials("AAPL")
    check("financials both fail: ok=False", r["ok"] is False)
    check("financials both fail: error names both providers",
          "twelvedata" in r["error"] and "finance.yahoo.com" in r["error"])

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_pass = sum(1 for _, s in results if s == PASS)
n_fail = len(results) - n_pass
print(f"\n{'='*60}\n{n_pass} passed, {n_fail} failed out of {len(results)} checks\n{'='*60}")
if n_fail:
    raise SystemExit(1)
