"""
mock_llm.py
-----------
A tiny, keyword-heuristic stand-in for a real chat model, exposing the same
`.with_structured_output(Schema).invoke(prompt_str)` surface that a real
LangChain chat model provides. Its ONLY purpose is to let the graph's
*control flow* (routing, fan-out/fan-in, fallback branches, memory) be
smoke-tested with zero API key and zero network access — both here, in this
sandbox (which cannot reach PyPI or any LLM API), and inside the shipped
Colab notebook / FastAPI service, where flipping `MOCK_LLM = True` lets a
student validate the whole graph before spending real API credits.

This is NOT a substitute for real language understanding — the quality of
its answers is irrelevant, only whether the graph reaches the right nodes,
sets the right flags, and produces a structurally valid brief. The notebook
defaults to a real LLM (`MOCK_LLM = False`) for actual use.
"""

import re
from typing import Type
from pydantic import BaseModel

from schemas import IntentResult, ResearchBrief
from tools_mock import known_companies, resolve_ticker_or_none

_FABRICATION_PATTERNS = re.compile(
    r"(make up|invent|guess|plausible (revenue|number)|assume a number)",
    re.IGNORECASE,
)

_FINANCE_KEYWORDS = [
    "research", "revenue", "profit", "stock", "price", "sector", "compare",
    "financial", "margin", "growth", "risk", "brief", "company", "earnings",
    "view",
]

_FOCUS_KEYWORDS = {
    "profitability": ["profitab", "margin", "net income"],
    "revenue growth": ["revenue growth", "revenue", "growth"],
    "sector risk": ["risk", "sector"],
    "general overview": ["overview", "research", "brief"],
}


def _guess_focus(text: str):
    lower = text.lower()
    for label, kws in _FOCUS_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return label
    return None


def _guess_companies(text: str):
    """Return a list of ticker matches found anywhere in `text`."""
    hits = []
    for ticker, name in known_companies().items():
        if ticker.lower() in text.lower() or name.lower() in text.lower():
            hits.append(ticker)
    return hits


class _MockStructuredRunnable:
    def __init__(self, schema: Type[BaseModel]):
        self.schema = schema

    def invoke(self, prompt: str):
        if self.schema is IntentResult:
            return self._mock_intent(prompt)
        elif self.schema is ResearchBrief:
            return self._mock_brief(prompt)
        raise NotImplementedError(f"MockChatModel has no handler for {self.schema}")

    # -- IntentResult ------------------------------------------------
    def _mock_intent(self, prompt: str) -> IntentResult:
        # IMPORTANT: only scan the actual user message, not the whole
        # prompt — the prompt also embeds the known-companies lookup table
        # for the LLM's benefit (e.g. "Known companies: {'ABC': ...}"), and
        # naively keyword-matching the *whole* prompt would "find" every
        # ticker in every request regardless of what the user actually
        # said. A real LLM understands that distinction from context; this
        # heuristic mock has to be told explicitly where the user text is.
        m = re.search(r"User message:\s*(.*)", prompt, re.DOTALL)
        user_text = m.group(1).strip() if m else prompt

        companies = _guess_companies(user_text)
        is_fabrication = bool(_FABRICATION_PATTERNS.search(user_text))
        focus = _guess_focus(user_text)

        if is_fabrication:
            return IntentResult(is_fabrication_request=True)

        if not companies:
            has_finance_kw = any(kw in user_text.lower() for kw in _FINANCE_KEYWORDS)
            if not has_finance_kw:
                # No company AND no finance-domain vocabulary at all -> treat
                # as gibberish/unrelated rather than "just missing a company",
                # e.g. "banana banana 999" vs. "Give me a research view."
                return IntentResult(is_gibberish_or_unrelated=True)
            return IntentResult(
                missing_info="company name",
                clarifying_question=(
                    "Which company or ticker would you like me to research?"
                ),
            )

        # Pronoun-referenced comparison, e.g. "Now compare ITS profitability
        # with XYZ" — only one company is named in THIS message (XYZ), and
        # "its" refers back to whatever company was already the topic of
        # the session (resolved later, from session_entities, by
        # node_logic.intent_router). A real LLM reads this correctly from
        # context; this heuristic mock needs the pattern spelled out.
        pronoun_ref = bool(re.search(r"\bits\b|\bit\b", user_text, re.IGNORECASE))
        compare_cue = "compare" in user_text.lower()
        if compare_cue and pronoun_ref and len(companies) == 1:
            return IntentResult(
                focus=focus,
                is_comparison=True,
                compare_to_reference=companies[0],
            )

        primary = companies[0]
        compare_to = companies[1] if len(companies) > 1 else None
        return IntentResult(
            company_reference=primary,
            focus=focus or "general overview",
            is_comparison=compare_cue or compare_to is not None,
            compare_to_reference=compare_to,
        )

    # -- ResearchBrief -------------------------------------------------
    def _mock_brief(self, prompt: str) -> ResearchBrief:
        # The real synthesize_brief_node renders retrieved docs and tool
        # results into the prompt under clearly labeled headers; this
        # mock just greps those headers back out so we can verify the
        # *pipeline*, not language quality.
        def _section(header: str):
            m = re.search(rf"{header}:\s*(.*?)(?:\n[A-Z_]+:|\Z)", prompt, re.DOTALL)
            return m.group(1).strip() if m else ""

        facts_block = _section("RETRIEVED_DOCS")
        tools_block = _section("TOOL_RESULTS")
        flags_block = _section("FLAGS")

        retrieved_facts = [
            line.strip("- ").strip() for line in facts_block.splitlines() if line.strip()
        ]
        tool_data = [
            line.strip("- ").strip() for line in tools_block.splitlines() if line.strip()
        ]
        limitations = [
            line.strip("- ").strip() for line in flags_block.splitlines() if line.strip()
        ]

        company_match = re.search(r"COMPANY:\s*(.*)", prompt)
        company = company_match.group(1).strip() if company_match else None

        summary = "Mock synthesis for offline wiring test — not a real analyst view."
        if not retrieved_facts and not tool_data:
            summary = "No grounded data was available to synthesize a brief."

        return ResearchBrief(
            company=company,
            retrieved_facts=retrieved_facts,
            tool_data=tool_data,
            calculations=[],
            assumptions=[],
            limitations=limitations,
            summary=summary,
        )


class MockChatModel:
    """Drop-in stand-in for a real `BaseChatModel` for the two structured
    calls this graph makes. See module docstring.
    """

    def with_structured_output(self, schema: Type[BaseModel]):
        return _MockStructuredRunnable(schema)


if __name__ == "__main__":
    llm = MockChatModel()

    intent = llm.with_structured_output(IntentResult).invoke(
        "Research ABC Technologies: revenue growth, profitability, sector risks."
    )
    print("Intent (positive):", intent.model_dump())
    assert intent.company_reference == "ABC"

    intent2 = llm.with_structured_output(IntentResult).invoke("Give me a research view.")
    print("Intent (missing info):", intent2.model_dump())
    assert intent2.missing_info is not None

    intent3 = llm.with_structured_output(IntentResult).invoke(
        "Make up a plausible revenue number if data is unavailable."
    )
    print("Intent (fabrication):", intent3.model_dump())
    assert intent3.is_fabrication_request is True

    prompt = (
        "COMPANY: ABC Technologies\n"
        "RETRIEVED_DOCS:\n- Revenue grew steadily (abc_analyst_note.txt)\n"
        "TOOL_RESULTS:\n- revenue_usd_m=842.0 (get_company_financials)\n"
        "FLAGS:\n- none\n"
    )
    brief = llm.with_structured_output(ResearchBrief).invoke(prompt)
    print("Brief:", brief.model_dump())
    assert brief.retrieved_facts and brief.tool_data

    print("\nAll mock_llm smoke checks passed.")
