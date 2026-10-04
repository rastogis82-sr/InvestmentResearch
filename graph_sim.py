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
    update_memory_node,
)


def run_turn(state: dict, user_input: str, llm, retriever) -> dict:
    """Runs one full turn through the graph, returning the NEW state.
    Mirrors invoking the compiled LangGraph app with a given thread_id.
    """
    state = start_turn(state, user_input)

    state = {**state, **input_guard(state)}
    dest = route_after_input_guard(state)

    if dest == "blank_node":
        state = {**state, **blank_node(state)}
    elif dest == "gibberish_node":
        state = {**state, **gibberish_node(state)}
    elif dest == "injection_node":
        state = {**state, **injection_node(state)}
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

    state = {**state, **update_memory_node(state)}
    return state
