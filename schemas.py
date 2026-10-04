"""
schemas.py
----------
Pydantic schemas used for LLM structured output. Using `.with_structured_
output(Schema)` (a standard LangChain chat-model method) instead of free-form
text is what makes it possible to *enforce*, not just prompt for, the
assignment's requirement to "distinguish retrieved facts, calculations and
assumptions" — the separation is a schema field, not a hope.

Only two LLM calls in the whole graph need structured output:
  1. `IntentResult`      — used by the `intent_router` node.
  2. `ResearchBrief`      — used by the `synthesize_brief_node` node.
Every other guardrail path (blank / gibberish / injection / fabrication
refusal) is template-based and deliberately does NOT call the LLM, so that
security-critical behaviour is 100% deterministic (see guardrails.py and
architecture.md, Section 5, for the reasoning).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """Structured extraction of what the user is asking for."""

    is_gibberish_or_unrelated: bool = Field(
        default=False,
        description=(
            "True if the message is not a coherent investment-research request at "
            "all — nonsense, random words, or entirely unrelated to company/market "
            "research — even if it contains real dictionary words (e.g. 'banana "
            "banana 999'). False for a coherent but incomplete request such as "
            "'Give me a research view.', which should set missing_info instead."
        ),
    )
    company_reference: Optional[str] = Field(
        default=None,
        description=(
            "The company name or ticker the user mentioned in THIS message, "
            "verbatim or close to it (e.g. 'ABC Technologies', 'XYZ', 'ABC'). "
            "None if no company is mentioned in this message at all."
        ),
    )
    focus: Optional[str] = Field(
        default=None,
        description=(
            "Short label for what aspect the user wants (e.g. 'profitability', "
            "'revenue growth', 'sector risk', 'general overview', 'comparison'). "
            "None if unclear."
        ),
    )
    is_comparison: bool = Field(
        default=False,
        description="True if the user is asking to compare two companies.",
    )
    compare_to_reference: Optional[str] = Field(
        default=None,
        description="The second company mentioned for a comparison request, if any.",
    )
    is_fabrication_request: bool = Field(
        default=False,
        description=(
            "True ONLY if the user explicitly asks the assistant to invent, "
            "guess, or make up a plausible number when real data is unavailable."
        ),
    )
    missing_info: Optional[str] = Field(
        default=None,
        description=(
            "What required piece of information is missing to proceed (e.g. "
            "'company name'). None if nothing required is missing."
        ),
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="A short, specific question to ask the user if missing_info is set.",
    )


class ResearchBrief(BaseModel):
    """Structured analyst brief. Every field must be traceable to a RAG
    chunk or a tool result passed into the prompt — the system prompt used
    with this schema instructs the model never to add outside numbers.
    """

    company: Optional[str] = None
    retrieved_facts: List[str] = Field(
        default_factory=list,
        description="Qualitative statements grounded in retrieved documents, each ending with its source filename in parentheses.",
    )
    tool_data: List[str] = Field(
        default_factory=list,
        description="Quantitative facts taken directly from tool results, each naming the tool that produced it.",
    )
    calculations: List[str] = Field(
        default_factory=list,
        description="Any derived figure with the explicit formula shown, e.g. 'Net margin = 96/842 = 11.4% (from get_company_financials)'.",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Explicitly flagged assumptions made in the absence of direct data. Must be clearly labeled as assumptions, not facts.",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Data gaps, tool failures, RAG failures, or conflicting figures the user should be aware of.",
    )
    summary: str = Field(
        default="",
        description="A short analyst-style narrative summary synthesizing the above, with no new facts not already listed above.",
    )
