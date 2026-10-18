"""
graph_sim.py
------------
A minimal hand-rolled executor that runs `node_logic.py`'s node functions in
exactly the order/conditions the real LangGraph `StateGraph` (built in the
Colab notebook) would, including the rag_node/tools_node fan-out and their
join at reconcile_node. This exists ONLY because this sandbox cannot
install the `langgraph` package to run the real graph — it lets the control
flow be verified end-to-end before the identical node functions are wired
into an actual `StateGraph`.

This is not shipped as part of the notebook; the notebook builds and
compiles a real `langgraph.graph.StateGraph` using the same node functions
from `node_logic.py` (copied into notebook cells) plus `MemorySaver` for
checkpointing, which is the actual persisted-session-memory mechanism. The
sequential loop below simply substitutes for that persistence by keeping
`state` in a local Python variable across calls within one test.
"""

from datetime import datetime, timezone

from node_logic import (
    new_state,
    start_turn,
    input_guard,
    route_after_input_guard,
    intent_router,
    route_after_intent,
    rag_node,
    tools_node,
    reconcile_node,
    synthesize_brief_node,
    blank_node,
    gibberish_node,
    injection_node,
    refusal_node,
    clarification_node,
    action_preview_node,
    human_review_gate_node,
    update_memory_node,
)
from safety_layer import SafetyRuntime


def run_turn(
    state: dict,
    user_input: str,
    llm,
    retriever,
    session_id: str = "default-session",
    consent: bool = True,
    stopped: bool = False,
    safety: SafetyRuntime = None,
) -> dict:
    """Runs one full turn through the graph, returning the NEW state.
    Mirrors invoking the compiled LangGraph app with a given thread_id.

    `session_id`/`consent`/`stopped`/`safety` are the guardrail-layer
    additions (see safety_layer.py and node_logic.py's module docstring);
    all four default to values that reproduce this function's EXACT prior
    behaviour (consent granted, kill switch off, a shared lazily-created
    default SafetyRuntime) so every pre-existing caller/test is unaffected.

    L2 (consent / kill switch / rate limit / size bound) is the very first
    check, BEFORE `start_turn` even resets state for this turn — exactly
    like SafeBank's `SafeBank.run()`, a blocked turn never reaches the
    graph at all, and is still audited (see the final `safety.store.append`
    call below).
    """
    safety = safety or SafetyRuntime.default()
    block = safety.gate.check(user_input, session_id, consent, stopped)
    if block is not None:
        # L2's block decision itself already happened (SafetyGate.check,
        # pure in-memory, no IO) -- logging it is bookkeeping, so a
        # Store/SQLite failure here must not swallow the block response
        # the caller is about to receive. See node_logic.py's
        # update_memory_node for the identical reasoning.
        try:
            safety.store.append({
                "time": datetime.now(timezone.utc).isoformat(),
                "owner": safety.store.owner(session_id),
                "input_classification": f"blocked:{block['reason']}",
                "flags": [f"safety_gate_blocked:{block['reason']}"],
            })
        except Exception:
            pass
        return {**state, "response": block["message"], "input_classification": f"blocked:{block['reason']}"}

    state = start_turn(state, user_input)

    state = {**state, **input_guard(state)}
    dest = route_after_input_guard(state)

    if dest == "blank_node":
        state = {**state, **blank_node(state)}
    elif dest == "gibberish_node":
        state = {**state, **gibberish_node(state)}
    elif dest == "injection_node":
        state = {**state, **injection_node(state)}
    elif dest == "action_preview_node":
        state = {**state, **action_preview_node(state, session_id, safety)}
    else:  # intent_router
        state = {**state, **intent_router(state, llm)}
        dest2 = route_after_intent(state)

        if dest2 == "gibberish_node":
            state = {**state, **gibberish_node(state)}
        elif dest2 == "refusal_node":
            state = {**state, **refusal_node(state)}
        elif dest2 == "clarification_node":
            state = {**state, **clarification_node(state)}
        else:  # "proceed" -> fan_out (rag_node + tools_node), mirrors the
               # notebook's no-op fan_out node with two unconditional edges
            # Fan-out: rag_node and tools_node both read the SAME
            # pre-fan-out state and each write to disjoint keys
            # (retrieved_docs/rag_flags vs. tool_results/tool_flags), so
            # there is no concurrent-write conflict to resolve — see
            # node_logic.py's TRANSIENT_DEFAULTS comment. reconcile_node
            # is the single place that combines both into `flags`.
            state = {**state, **rag_node(state, retriever)}
            state = {**state, **tools_node(state)}
            state = {**state, **reconcile_node(state)}
            state = {**state, **synthesize_brief_node(state, llm)}
            # L5 — human review queue, triggered by the SAME flags the
            # pipeline already raises for other reasons (see
            # safety_layer.flags_warrant_review). Only on the research
            # path: action_preview_node above already unconditionally
            # queues a review for every action request.
            state = {**state, **human_review_gate_node(state, session_id, safety)}

    state = {**state, **update_memory_node(state, session_id, safety)}
    return state
