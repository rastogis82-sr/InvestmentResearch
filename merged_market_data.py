"""
merged_market_data.py
----------------------
Combines real_market_data.py (Twelve Data) and fcs_market_data.py (FCS
API) into the three provider-agnostic functions node_logic.py's
`tools_node` actually calls for a real company — `get_merged_stock_price`,
`get_merged_company_profile`, `get_merged_company_financials`. Same
`{"ok": True/False, ...}` contract either single-provider module already
uses, so nothing downstream has to know two providers were ever involved.

Why two providers at all: Twelve Data's free plan is reliable for price
but can't help with profile (`/profile` is paid-plan-only) or financials
(no free fundamentals endpoint exists at all — see
real_market_data.py's module docstring). FCS API's free plan covers
exactly that gap (`/stock/profile`, `/stock/statistics`,
`/stock/income_statements`), at the cost of a much tighter per-minute
request quota. Neither provider is "better" — they're combined because
each covers a hole the other has.

Merge strategy, field by field, not provider by provider:
  - **Price**: Twelve Data is the primary and (on its free plan) already
    reliable source. FCS API's `/stock/latest` is only called as a
    FALLBACK, when Twelve Data's call itself fails -- deliberately, to
    conserve FCS API's especially tight 3-requests-per-minute free quota
    for profile/financials, where Twelve Data's free plan can't help at
    all and FCS API is actually needed every time.
  - **Profile and financials**: both providers are queried, and the
    result is a FIELD-LEVEL merge, not a pick-one-provider merge. Twelve
    Data's fields win where it actually has them (it's generally the
    more complete source on a paid plan); any field Twelve Data doesn't
    have — either because its call failed outright (free-plan
    restriction) or because it simply doesn't return that field at all
    (Twelve Data's "financials" never includes real revenue/margin
    figures, paid plan or not) — is filled in from FCS API instead. A
    field is never silently dropped and never fabricated: it's either a
    real number from one of the two providers, or absent.
  - The merged result's `source` field names every provider that actually
    contributed at least one field this turn (e.g. "twelvedata.com +
    fcsapi.com (merged)"), and a `note` is appended naming which specific
    fields came from the secondary provider -- so the transparency panel
    stays honest about provenance, not just about success/failure.
"""

import concurrent.futures

# Deliberately "from X import name1, name2" rather than "import X as td" /
# "import X as fcs": build_api_app.py and assemble_notebook.py inline this
# file's source directly into investment_research_api.py / the notebook
# (stripping only "from <local module> import ..." lines, since the
# functions end up sharing one flat namespace there, not a real importable
# package) -- the same convention node_logic.py's own local-module imports
# already follow. An "import X as td" alias would survive that stripping
# unchanged and then fail at runtime wherever real_market_data.py /
# fcs_market_data.py aren't ALSO deployed as sibling files next to
# investment_research_api.py.
from real_market_data import get_real_stock_price, get_real_company_profile, get_real_company_financials
from fcs_market_data import get_fcs_stock_price, get_fcs_company_profile, get_fcs_company_financials

# Fields that describe the RESPONSE itself, not actual company data -- never
# treated as a "field to merge", just regenerated fresh by _merge() itself.
_META_FIELDS = ("source", "note")


def _merge(primary: dict, secondary: dict, tool_name: str, secondary_label: str = "fcsapi.com") -> dict:
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
    """Twelve Data's /quote first; FCS API's /stock/latest ONLY as a
    fallback if that fails. See module docstring for why this one isn't a
    field-level merge like profile/financials are."""
    primary = get_real_stock_price(ticker)
    if primary.get("ok"):
        return primary
    fallback = get_fcs_stock_price(ticker)
    if fallback.get("ok"):
        fallback["data"]["note"] = f"Twelve Data unavailable this turn ({primary.get('error')}); using FCS API instead."
        return fallback
    return {
        "ok": False,
        "tool": "get_stock_price",
        "error": f"twelvedata:{primary.get('error')}; fcsapi.com:{fallback.get('error')}",
    }


def get_merged_company_profile(ticker: str) -> dict:
    """Twelve Data + FCS API's /stock/profile, run concurrently (two
    different providers, no shared cache or rate limit to serialize for)
    and merged field by field."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        td_future = pool.submit(get_real_company_profile, ticker)
        fcs_future = pool.submit(get_fcs_company_profile, ticker)
        td_result = td_future.result()
        fcs_result = fcs_future.result()
    return _merge(td_result, fcs_result, "get_company_profile")


def get_merged_company_financials(ticker: str) -> dict:
    """Twelve Data's financials (company-profile fields only, reused from
    its cached /profile response -- see real_market_data.py) + FCS API's
    real revenue/margin figures (/stock/statistics +
    /stock/income_statements), merged field by field. FCS API is the only
    source of actual financial-statement data here; Twelve Data's free
    plan has none at all, paid plan or not."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        td_future = pool.submit(get_real_company_financials, ticker)
        fcs_future = pool.submit(get_fcs_company_financials, ticker)
        td_result = td_future.result()
        fcs_result = fcs_future.result()
    return _merge(td_result, fcs_result, "get_company_financials")


if __name__ == "__main__":
    # Manual smoke test -- requires BOTH TWELVEDATA_API_KEY and
    # FCS_API_KEY set, plus outbound internet access neither available in
    # this sandbox. Run in Colab:
    #     TWELVEDATA_API_KEY=... FCS_API_KEY=... python3 merged_market_data.py
    print("Merged quote for AAPL     :", get_merged_stock_price("AAPL"))
    print("Merged profile for AAPL   :", get_merged_company_profile("AAPL"))
    print("Merged financials for AAPL:", get_merged_company_financials("AAPL"))
