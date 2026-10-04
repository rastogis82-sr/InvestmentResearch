"""
build_diagram.py
-----------------
Renders the LangGraph architecture as a PNG flowchart using matplotlib only
(no graphviz / mermaid CLI available in this environment). Run once to
produce architecture_diagram.png, which is embedded in architecture.md and
sent to the user alongside the notebook.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COLORS = {
    "guardrail": "#F2994A",
    "routing": "#828282",
    "data": "#2F80ED",
    "synthesis": "#27AE60",
    "memory": "#9B51E0",
    "terminal": "#EB5757",
    "endpoint": "#111111",
}

fig, ax = plt.subplots(figsize=(16, 15))
ax.set_xlim(0, 100)
ax.set_ylim(0, 108)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, text, color, fontsize=9.3, text_color="white"):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.35,rounding_size=2",
        linewidth=1.2,
        edgecolor="#2b2b2b",
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color, wrap=True, zorder=3)
    return (x, y, w, h)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def bottom_at(b, frac):
    x, y, w, h = b
    return (x + w * frac, y)


def top_at(b, frac):
    x, y, w, h = b
    return (x + w * frac, y + h)


def arrow(p1, p2, label=None, color="#333333", style="-", connection="arc3,rad=0.0", lw=1.3, label_offset=(0, 0)):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=color, linestyle=style, connectionstyle=connection, shrinkA=3, shrinkB=3,
        zorder=1,
    )
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + label_offset[0], (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=7.8, color=color, ha="center", va="center",
                backgroundcolor="white", zorder=4)


# ---------------------------------------------------------------------------
# Row 0: START
# ---------------------------------------------------------------------------
start = box(43, 101, 14, 4.5, "START\n(new user turn)", COLORS["endpoint"], fontsize=9)

# ---------------------------------------------------------------------------
# Row 1: input_guard
# ---------------------------------------------------------------------------
input_guard = box(33, 91, 34, 6.5,
                   "input_guard\nblank check  •  injection regex  •  gibberish heuristic\n(deterministic — no LLM call)",
                   COLORS["guardrail"], fontsize=9.5)

# ---------------------------------------------------------------------------
# Row 2: terminal templates (left) + intent_router (right)
# ---------------------------------------------------------------------------
blank_n = box(1, 78, 15, 6.5, "blank_node\n(template response)", COLORS["terminal"], fontsize=8.7)
gibberish_n = box(18, 78, 16, 6.5, "gibberish_node\n(template response)", COLORS["terminal"], fontsize=8.7)
injection_n = box(36, 78, 16.5, 6.5, "injection_node\n(template response)", COLORS["terminal"], fontsize=8.7)

intent_router = box(59, 77.5, 26, 8,
                     "intent_router  (LLM → IntentResult)\nresolves company (falls back to\nsession memory) • detects fabrication\nrequests • also judges incoherent/\nunrelated input → gibberish_node",
                     COLORS["routing"], fontsize=8.6)

# ---------------------------------------------------------------------------
# Row 3: refusal_node / clarification_node (right column, stacked)
# ---------------------------------------------------------------------------
refusal_n = box(88, 82, 11, 6, "refusal_node\n(fabrication\nrequest)", COLORS["terminal"], fontsize=8.2)
clarification_n = box(88, 72.5, 11, 7, "clarification_node\n(missing info —\nasks LLM's\nquestion)", COLORS["routing"], fontsize=8.2)

# ---------------------------------------------------------------------------
# Row 4: rag_node / tools_node (fan-out, parallel)
# ---------------------------------------------------------------------------
rag_n = box(28, 60, 21, 8.5,
            "rag_node\nFAISS + HuggingFace embeddings\nsimilarity threshold\ninjection prefilter on chunks",
            COLORS["data"], fontsize=8.6)
tools_n = box(53, 60, 21, 8.5,
              "tools_node\n4 mock tools:\nfinancials, price,\nprofile, sector",
              COLORS["data"], fontsize=8.8)

# ---------------------------------------------------------------------------
# Row 5: reconcile_node
# ---------------------------------------------------------------------------
reconcile_n = box(35, 49, 30, 6.5,
                   "reconcile_node  (fan-in / join)\nflags: rag_failure • tool_failure • conflicting_data",
                   COLORS["routing"], fontsize=8.8)

# ---------------------------------------------------------------------------
# Row 6: synthesize_brief_node
# ---------------------------------------------------------------------------
synth_n = box(30, 37, 40, 8,
              "synthesize_brief_node  (LLM → ResearchBrief)\nretrieved_facts | tool_data | calculations |\nassumptions | limitations  — no numbers outside these",
              COLORS["synthesis"], fontsize=8.8)

# ---------------------------------------------------------------------------
# Row 7: update_memory_node  (wide — everything converges here)
# ---------------------------------------------------------------------------
memory_n = box(22, 24, 56, 7.5,
               "update_memory_node\nappend response to messages  •  update session_entities\n(both persisted across turns via the LangGraph checkpointer / thread_id)",
               COLORS["memory"], fontsize=9)

# ---------------------------------------------------------------------------
# Row 8: END
# ---------------------------------------------------------------------------
end_n = box(43, 13, 14, 5, "END\n(response returned,\nstate checkpointed)", COLORS["endpoint"], fontsize=8.7)

# ---------------------------------------------------------------------------
# Edges — input_guard fan-out
# ---------------------------------------------------------------------------
arrow(bottom(start), top(input_guard))

arrow(left(input_guard), top_at(blank_n, 0.75), "blank", connection="arc3,rad=-0.15")
arrow(bottom_at(input_guard, 0.35), top(gibberish_n), "gibberish\n(heuristic)", connection="arc3,rad=-0.05")
arrow(bottom_at(input_guard, 0.55), top(injection_n), "injection\n(regex hit)", connection="arc3,rad=0.05")
arrow(right(input_guard), top_at(intent_router, 0.3), "valid", connection="arc3,rad=0.15")

# intent_router branches
arrow(right(intent_router), left(refusal_n), "fabrication\nrequest")
arrow(bottom_at(intent_router, 0.9), top(clarification_n), "missing\ncompany", connection="arc3,rad=-0.25")
arrow(bottom_at(intent_router, 0.15), top(tools_n), "proceed\n(fan-out)", connection="arc3,rad=0.15")
arrow(bottom_at(intent_router, 0.05), top(rag_n), "proceed\n(fan-out)", connection="arc3,rad=0.3")

# fan-in to reconcile
arrow(bottom(rag_n), top_at(reconcile_n, 0.2), connection="arc3,rad=0.08")
arrow(bottom(tools_n), top_at(reconcile_n, 0.8), connection="arc3,rad=-0.08")

arrow(bottom(reconcile_n), top(synth_n))

# Convergence on update_memory_node
arrow(bottom(blank_n), top_at(memory_n, 0.03), color="#999999", connection="arc3,rad=-0.15")
arrow(bottom(gibberish_n), top_at(memory_n, 0.12), color="#999999", connection="arc3,rad=-0.1")
arrow(bottom(injection_n), top_at(memory_n, 0.22), color="#999999", connection="arc3,rad=-0.05")
arrow(bottom(refusal_n), top_at(memory_n, 0.95), color="#999999", connection="arc3,rad=0.2")
arrow(bottom(clarification_n), top_at(memory_n, 0.85), color="#999999", connection="arc3,rad=0.12")
arrow(bottom(synth_n), top_at(memory_n, 0.5))

arrow(bottom(memory_n), top(end_n))

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_items = [
    ("Guardrail (deterministic)", COLORS["guardrail"]),
    ("Routing / control", COLORS["routing"]),
    ("Tools / RAG (data)", COLORS["data"]),
    ("LLM synthesis", COLORS["synthesis"]),
    ("Session memory", COLORS["memory"]),
    ("Terminal template response", COLORS["terminal"]),
]
ly = 2
for i, (label, color) in enumerate(legend_items):
    lx = 2 + i * 16.3
    ax.add_patch(FancyBboxPatch((lx, ly), 2, 2, boxstyle="round,pad=0.1", facecolor=color, edgecolor="none"))
    ax.text(lx + 2.8, ly + 1, label, fontsize=8, va="center", ha="left")

ax.text(50, 106, "Investment Research Assistant — LangGraph State Machine",
        ha="center", fontsize=15, fontweight="bold")

plt.tight_layout()
plt.savefig("/home/claude/ira_project/architecture_diagram.png", dpi=170, bbox_inches="tight", facecolor="white")
print("Saved architecture_diagram.png")
