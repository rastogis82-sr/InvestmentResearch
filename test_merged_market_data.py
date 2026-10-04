"""
test_merged_market_data.py
----------------------------
Unit tests for merged_market_data.py's field-level merge logic, using
hand-built {"ok": ..., "data": {...}} results (the SAME shape
real_market_data.py / fcs_market_data.py actually return) rather than
mocking HTTP -- this is testing the merge function itself, not either
provider's request/response handling (that's what test_real_market_data.py
and test_fcs_market_data.py already cover).

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
# get_merged_stock_price: Twelve Data succeeds -> FCS API never called
# ---------------------------------------------------------------------------
TD_PRICE_OK = {"ok": True, "tool": "get_stock_price", "data": {"last_price_usd": 256.48, "source": "twelvedata.com (live)"}}
with patch("merged_market_data.get_real_stock_price", return_value=TD_PRICE_OK) as td_mock, \
     patch("merged_market_data.get_fcs_stock_price") as fcs_mock:
    r = mmd.get_merged_stock_price("AAPL")
    check("price: Twelve Data success returned as-is", r == TD_PRICE_OK)
    check("price: FCS API not called when Twelve Data succeeds", fcs_mock.call_count == 0)

# ---------------------------------------------------------------------------
# get_merged_stock_price: Twelve Data fails -> FCS API used as fallback
# ---------------------------------------------------------------------------
TD_PRICE_FAIL = {"ok": False, "tool": "get_stock_price", "error": "network_error:ConnectionError"}
FCS_PRICE_OK = {"ok": True, "tool": "get_stock_price", "data": {"last_price_usd": 255.90, "source": "fcsapi.com (live)"}}
with patch("merged_market_data.get_real_stock_price", return_value=TD_PRICE_FAIL), \
     patch("merged_market_data.get_fcs_stock_price", return_value=FCS_PRICE_OK) as fcs_mock:
    r = mmd.get_merged_stock_price("AAPL")
    check("price fallback: ok=True from FCS API", r["ok"] is True)
    check("price fallback: FCS API was actually called", fcs_mock.call_count == 1)
    check("price fallback: price value is FCS API's", r["data"]["last_price_usd"] == 255.90)
    check("price fallback: note explains why", "Twelve Data unavailable" in r["data"].get("note", ""))

# ---------------------------------------------------------------------------
# get_merged_stock_price: both fail -> ok=False, both errors visible
# ---------------------------------------------------------------------------
FCS_PRICE_FAIL = {"ok": False, "tool": "get_stock_price", "error": "rate_limited:Too Many Requests"}
with patch("merged_market_data.get_real_stock_price", return_value=TD_PRICE_FAIL), \
     patch("merged_market_data.get_fcs_stock_price", return_value=FCS_PRICE_FAIL):
    r = mmd.get_merged_stock_price("AAPL")
    check("price both fail: ok=False", r["ok"] is False)
    check("price both fail: error names both providers",
          "twelvedata" in r["error"] and "fcsapi.com" in r["error"])

# ---------------------------------------------------------------------------
# get_merged_company_profile: Twelve Data plan-restricted (free tier, the
# common case), FCS API succeeds -> field-level fill-in, not a replacement
# ---------------------------------------------------------------------------
TD_PROFILE_RESTRICTED = {"ok": False, "tool": "get_company_profile", "error": "plan_restricted:upgrade required"}
FCS_PROFILE_OK = {
    "ok": True, "tool": "get_company_profile",
    "data": {
        "sector": "Electronic Technology", "industry": "Telecommunications Equipment",
        "hq": "Cupertino, CA", "description": "Apple, Inc. engages in...",
        "employees": 164000, "founded": "1976", "ceo": "Timothy Donald Cook",
        "website": "http://www.apple.com", "market_cap_usd_b": 3806.27,
        "source": "fcsapi.com (live)",
    },
}
with patch("merged_market_data.get_real_company_profile", return_value=TD_PROFILE_RESTRICTED), \
     patch("merged_market_data.get_fcs_company_profile", return_value=FCS_PROFILE_OK):
    r = mmd.get_merged_company_profile("AAPL")
    check("profile: TD restricted, FCS fills entirely -> ok=True", r["ok"] is True)
    check("profile: sector came through from FCS", r["data"]["sector"] == "Electronic Technology")
    check("profile: ceo (FCS-only field) present", r["data"]["ceo"] == "Timothy Donald Cook")
    check("profile: source names only fcsapi.com", r["data"]["source"] == "fcsapi.com (live)")

# ---------------------------------------------------------------------------
# get_merged_company_profile: BOTH succeed -> Twelve Data's fields win,
# FCS API only fills what Twelve Data doesn't have (ceo, website, employees,
# founded, market cap -- fields real_market_data.get_real_company_profile
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
     patch("merged_market_data.get_fcs_company_profile", return_value=FCS_PROFILE_OK):
    r = mmd.get_merged_company_profile("AAPL")
    check("profile both ok: Twelve Data's sector wins (not overwritten)", r["data"]["sector"] == "Technology")
    check("profile both ok: Twelve Data's description wins", r["data"]["description"] == "Twelve Data's own description.")
    check("profile both ok: FCS-only field (ceo) still filled in", r["data"]["ceo"] == "Timothy Donald Cook")
    check("profile both ok: FCS-only field (market_cap_usd_b) still filled in", r["data"]["market_cap_usd_b"] == 3806.27)
    check("profile both ok: source names both providers", r["data"]["source"] == "twelvedata.com + fcsapi.com (live, merged)")
    check("profile both ok: note lists which fields were filled in from FCS",
          "ceo" in r["data"]["note"] and "market_cap_usd_b" in r["data"]["note"])

# ---------------------------------------------------------------------------
# get_merged_company_financials: Twelve Data "succeeds" (profile-only
# fields, NO real financials -- see real_market_data.py), FCS API supplies
# the actual revenue/margin numbers -- this is the exact gap the user
# asked to close.
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
FCS_FINANCIALS_OK = {
    "ok": True, "tool": "get_company_financials",
    "data": {
        "net_margin_pct": 5.21, "gross_margin_pct": 17.24, "operating_margin_pct": 4.10,
        "pe_ratio": 183.96, "market_cap_usd_b": 1024.14,
        "revenue_usd_m": 391035.0, "net_income_usd_m": 93736.0,
        "revenue_growth_yoy_pct": 2.02, "fiscal_year": "2024-09-30",
        "source": "fcsapi.com (live)",
    },
}
with patch("merged_market_data.get_real_company_financials", return_value=TD_FINANCIALS_OK_BUT_THIN), \
     patch("merged_market_data.get_fcs_company_financials", return_value=FCS_FINANCIALS_OK):
    r = mmd.get_merged_company_financials("AAPL")
    check("financials: ok=True", r["ok"] is True)
    check("financials: revenue present (only FCS has this)", r["data"]["revenue_usd_m"] == 391035.0)
    check("financials: revenue growth present (only FCS has this)", r["data"]["revenue_growth_yoy_pct"] == 2.02)
    check("financials: employees=None from TD doesn't block FCS fields", "revenue_usd_m" in r["data"])
    check("financials: TD's sector still present (TD field, not overwritten)", r["data"]["sector"] == "Technology")
    check("financials: combined note mentions both the TD limitation and the FCS fill-in",
          "income-statement" in r["data"]["note"] and "revenue_usd_m" in r["data"]["note"])

# ---------------------------------------------------------------------------
# get_merged_company_financials: BOTH providers fail -> ok=False, both
# errors visible, never silently empty
# ---------------------------------------------------------------------------
TD_FINANCIALS_FAIL = {"ok": False, "tool": "get_company_financials", "error": "plan_restricted:upgrade"}
FCS_FINANCIALS_FAIL = {"ok": False, "tool": "get_company_financials", "error": "statistics:rate_limited:...; income_statements:rate_limited:..."}
with patch("merged_market_data.get_real_company_financials", return_value=TD_FINANCIALS_FAIL), \
     patch("merged_market_data.get_fcs_company_financials", return_value=FCS_FINANCIALS_FAIL):
    r = mmd.get_merged_company_financials("AAPL")
    check("financials both fail: ok=False", r["ok"] is False)
    check("financials both fail: error names both providers",
          "twelvedata" in r["error"] and "fcsapi.com" in r["error"])

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_pass = sum(1 for _, s in results if s == PASS)
n_fail = len(results) - n_pass
print(f"\n{'='*60}\n{n_pass} passed, {n_fail} failed out of {len(results)} checks\n{'='*60}")
if n_fail:
    raise SystemExit(1)
