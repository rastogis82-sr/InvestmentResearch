"""
merged_market_data.py
----------------------
Combines real_market_data.py (Twelve Data) and yahoo_finance_market_data.py
(Yahoo Finance, via the `yfinance` package) into the three
provider-agnostic functions node_logic.py's `tools_node` actually calls
for a real company — `get_merged_stock_price`, `get_merged_company_profile`,
`get_merged_company_financials`. Same `{"ok": True/False, ...}` contract
either single-provider module already uses, so nothing downstream has to
know two providers were ever involved.

Why two providers at all: Twelve Data's free plan is reliable for price
but can't help with profile (`/profile` is paid-plan-only) or financials
(no free fundamentals endpoint exists at all — see
real_market_data.py's module docstring). Yahoo Finance (via `yfinance`)
covers exactly that gap with no API key and no published rate limit at
all, at the cost of being an unofficial, undocumented interface rather
than a supported REST API — see yahoo_finance_market_data.py's module
docstring for that honest trade-off. Neither provider is "better" —
they're combined because each covers a hole the other has.

Provider history — this module has used three different secondary
providers, in order, each dropped or replaced for a concrete reason:
  1. **FCS API** (fcsapi.com) — its documentation did not flag
     `/stock/profile` as paid-plan-only, but a live test against a real
     free-tier key showed FCS API rejects EVERY stock endpoint on its
     free plan ("A Free API Key user cannot be used with stock/index
     endpoint. Please upgrade your plan.") — a documentation-vs-reality
     mismatch on FCS API's side. Dropped entirely; `fcs_market_data.py`
     was deleted rather than kept around unusable.
  2. **Alpha Vantage** — free tier worked with no endpoint-level gating,
     but is capped at a tight 25 requests/day (not just per-minute),
     which a single classroom demo session could exhaust. A real bug was
     also caught and fixed in that version: Alpha Vantage's own
     daily-quota-exceeded message is delivered under an `"Information"`
     body key containing the substring "API key" ("We have detected your
     API key as ... and our standard API rate limit is 25 requests per
     day..."), which an earlier classifier mistakenly read as an invalid
     key rather than an exhausted quota. Replaced at the user's explicit
     request to standardize on Twelve Data + Yahoo Finance only;
     `alpha_vantage_market_data.py` was deleted.
  3. **Yahoo Finance** (current) — no API key, no published daily/monthly
     cap, via the widely-used `yfinance` package. The trade-off is that
     it's an unofticial interface to Yahoo's own internal endpoints, not
     a documented, versioned, SLA-backed API the way Twelve Data and
     Alpha Vantage both are — see yahoo_finance_market_data.py's module
     docstring for the full honest-limitations discussion.

Merge strategy, field by field, not provider by provider:
  - **Price**: Twelve Data is the primary and (on its free plan) already
    reliable source. Yahoo Finance's price fields are only read as a
    FALLBACK, when Twelve Data's call itself fails -- deliberately, so
    the common case (Twelve Data succeeding) never depends on Yahoo
    Finance's unofficial endpoints at all.
  - **Profile and financials**: both providers are queried, and the
    result is a FIELD-LEVEL merge, not a pick-one-provider merge. Twelve
    Data's fields win where it actually has them (it's generally the
    more complete source on a paid plan); any field Twelve Data doesn't
    have — either because its call failed outright (free-plan
    restriction) or because it simply doesn't return that field at all
    (Twelve Data's "financials" never includes real revenue/margin
    figures, paid plan or not) — is filled in from Yahoo Finance instead.
    A field is never silently dropped and never fabricated: it's either a
    real number from one of the two providers, or absent.
  - The merged result's `source` field names every provider that actually
    contributed at least one field this turn (e.g. "twelvedata.com +
    finance.yahoo.com (live, merged)"), and a `note` is appended naming
    which specific fields came from the secondary provider -- so the
    transparency panel stays honest about provenance, not just about
    success/failure.
"""

import concurrent.futures

# Deliberately "from X import name1, name2" rather than "import X as td" /
# "import X as yf_md": build_api_app.py and assemble_notebook.py inline
# this file's source directly into investment_research_api.py / the
# notebook (stripping only "from <local module> import ..." lines, since
# the functions end up sharing one flat namespace there, not a real
# importable package) -- the same convention node_logic.py's own
# local-module imports already follow. An "import X as td" alias would
# survive that stripping unchanged and then fail at runtime wherever
# real_market_data.py / yahoo_finance_market_data.py aren't ALSO deployed
# as sibling files next to investment_research_api.py.
from real_market_data import get_real_stock_price, get_real_company_profile, get_real_company_financials
from yahoo_finance_market_data import (
    get_yahoo_stock_price,
    get_yahoo_company_profile,
    get_yahoo_company_financials,
)

# Fields that describe the RESPONSE itself, not actual company data -- never
# treated as a "field to merge", just regenerated fresh by _merge() itself.
_META_FIELDS = ("source", "note")


def _merge(primary: dict, secondary: dict, tool_name: str, secondary_label: str = "finance.yahoo.com") -> dict:
    """Field-level merge of two {"ok": ..., "data": {...}} results for the
    SAME tool call. `primary`'s fields always win when present; anything
    `primary` is missing (its call failed entirely, or it succeeded but
    simply never returns that field) is filled in from `secondary`."""
    primary_ok = bool(primary.get("ok"))
    secondary_ok = bool(secondary.get("ok"))

    if not primary_ok and not secondary_ok:
        return {
            "ok": False,
            "tool": tool_name,
            "error": f"twelvedata:{primary.get('error', 'unknown')}; {secondary_label}:{secondary.get('error', 'unknown')}",
        }

    primary_data = dict(primary.get("data") or {}) if primary_ok else {}
    secondary_data = dict(secondary.get("data") or {}) if secondary_ok else {}

    merged = {k: v for k, v in primary_data.items() if k not in _META_FIELDS}
    filled_from_secondary = []
    for key, value in secondary_data.items():
        if key in _META_FIELDS:
            continue
        if merged.get(key) is None and value is not None:
            merged[key] = value
            filled_from_secondary.append(key)

    contributors = []
    if primary_ok and any(k not in _META_FIELDS for k in primary_data):
        contributors.append("twelvedata.com")
    if filled_from_secondary:
        contributors.append(secondary_label)
    if len(contributors) > 1:
        merged["source"] = " + ".join(contributors) + " (live, merged)"
    elif contributors:
        merged["source"] = f"{contributors[0]} (live)"
    else:
        merged["source"] = secondary_label + " (live)"

    notes = [n for n in (primary_data.get("note"), secondary_data.get("note")) if n]
    if filled_from_secondary:
        humanized = ", ".join(sorted(filled_from_secondary))
        notes.append(f"Filled in from {secondary_label} (not available from Twelve Data here): {humanized}.")
    if notes:
        merged["note"] = " ".join(notes)

    return {"ok": True, "tool": tool_name, "data": merged}


def get_merged_stock_price(ticker: str) -> dict:
    """Twelve Data's /quote first; Yahoo Finance's price fields ONLY as a
    fallback if that fails. See module docstring for why this one isn't a
    field-level merge like profile/financials are."""
    primary = get_real_stock_price(ticker)
    if primary.get("ok"):
        return primary
    fallback = get_yahoo_stock_price(ticker)
    if fallback.get("ok"):
        fallback["data"]["note"] = f"Twelve Data unavailable this turn ({primary.get('error')}); using Yahoo Finance instead."
        return fallback
    return {
        "ok": False,
        "tool": "get_stock_price",
        "error": f"twelvedata:{primary.get('error')}; finance.yahoo.com:{fallback.get('error')}",
    }


def get_merged_company_profile(ticker: str) -> dict:
    """Twelve Data + Yahoo Finance, run concurrently (two different
    providers, no shared cache or rate limit to serialize for) and merged
    field by field."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        td_future = pool.submit(get_real_company_profile, ticker)
        yahoo_future = pool.submit(get_yahoo_company_profile, ticker)
        td_result = td_future.result()
        yahoo_result = yahoo_future.result()
    return _merge(td_result, yahoo_result, "get_company_profile")


def get_merged_company_financials(ticker: str) -> dict:
    """Twelve Data's financials (company-profile fields only, reused from
    its cached /profile response -- see real_market_data.py) + Yahoo
    Finance's real revenue/margin figures (from the same `Ticker.info`
    call get_merged_company_profile uses, shared via
    yahoo_finance_market_data's own cache), merged field by field. Yahoo
    Finance is the only source of actual financial-statement data here;
    Twelve Data's free plan has none at all, paid plan or not."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        td_future = pool.submit(get_real_company_financials, ticker)
        yahoo_future = pool.submit(get_yahoo_company_financials, ticker)
        td_result = td_future.result()
        yahoo_result = yahoo_future.result()
    return _merge(td_result, yahoo_result, "get_company_financials")


if __name__ == "__main__":
    # Manual smoke test -- requires TWELVEDATA_API_KEY set, plus the
    # yfinance package installed, plus outbound internet access neither
    # available in this sandbox. Run in Colab:
    #     TWELVEDATA_API_KEY=... python3 merged_market_data.py
    print("Merged quote for AAPL     :", get_merged_stock_price("AAPL"))
    print("Merged profile for AAPL   :", get_merged_company_profile("AAPL"))
    print("Merged financials for AAPL:", get_merged_company_financials("AAPL"))
