"""
build_deployment_diagram.py
----------------------------
Renders the FastAPI + ngrok deployment topology as a PNG (matplotlib only,
same constraint as build_diagram.py: no graphviz/mermaid CLI available).
Shows the full request path from an external client, through ngrok's cloud
reverse proxy, into the Colab VM running Uvicorn/FastAPI, and down into the
compiled LangGraph agent. Saved as deployment_diagram.png.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COLORS = {
    "external": "#333333",
    "ngrok": "#1A73E8",
    "colab": "#F2994A",
    "fastapi": "#27AE60",
    "agent": "#9B51E0",
    "store": "#EB5757",
}

fig, ax = plt.subplots(figsize=(13, 11))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, text, color, fontsize=10, text_color="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2.2",
                        linewidth=1.3, edgecolor="#2b2b2b", facecolor=color, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=3)
    return (x, y, w, h)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def arrow(p1, p2, label=None, color="#333333", connection="arc3,rad=0.0", lw=1.5):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15, linewidth=lw,
                         color=color, connectionstyle=connection, shrinkA=4, shrinkB=4, zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + 3, my, label, fontsize=8.3, color=color, ha="left", va="center",
                backgroundcolor="white", zorder=4)


ax.text(50, 96.5, "Investment Research Assistant — FastAPI + ngrok Deployment",
        ha="center", fontsize=14.5, fontweight="bold")

client = box(35, 85, 30, 7, "External Client\n(browser / curl / requests)\nsends HTTPS POST /chat", COLORS["external"], fontsize=9.5)

ngrok_cloud = box(32, 72, 36, 8,
                   "Ngrok Cloud\npublic HTTPS URL (https://<id>.ngrok-free.app)\nreverse-proxy gateway, TLS termination", COLORS["ngrok"], fontsize=9)

colab_box = FancyBboxPatch((10, 20), 80, 48, boxstyle="round,pad=0.6,rounding_size=2",
                            linewidth=1.6, edgecolor="#F2994A", facecolor="#FFF6EC", zorder=0)
ax.add_patch(colab_box)
ax.text(50, 65.5, "Google Colab VM  (ngrok agent / tunnel runs here)", ha="center",
        fontsize=9.8, color="#B35B00", fontweight="bold")

ngrok_agent = box(37, 57, 26, 6, "ngrok agent (pyngrok)\nlocal tunnel endpoint", COLORS["ngrok"], fontsize=8.8)

uvicorn_box = box(32, 47, 36, 6.5, "Uvicorn ASGI server\n0.0.0.0:8000  (background thread)", COLORS["colab"], fontsize=9, text_color="#222222")

fastapi_box = box(22, 36, 56, 8,
                   "FastAPI app\nPOST /chat · GET /health · GET /debug/{session_id}\nPydantic request/response validation · CORS · structured JSON logs",
                   COLORS["fastapi"], fontsize=9)

agent_box = box(22, 24, 56, 8.5,
                 "Compiled LangGraph agent  (research_graph)\ninput_guard → intent_router → rag_node/tools_node →\nreconcile_node → synthesize_brief_node → update_memory_node",
                 COLORS["agent"], fontsize=8.8)

checkpointer = box(10, 11, 34, 6, "MemorySaver checkpointer\n(in-process, keyed by session_id\n→ thread_id)", COLORS["store"], fontsize=8.3)
corpus_store = box(56, 11, 34, 6, "FAISS vector store +\nmock tool/company DB\n(in-memory, Colab filesystem)", COLORS["store"], fontsize=8.3)

arrow(bottom(client), top(ngrok_cloud), "HTTPS")
arrow(bottom(ngrok_cloud), top(ngrok_agent), "tunnel")
arrow(bottom(ngrok_agent), top(uvicorn_box))
arrow(bottom(uvicorn_box), top(fastapi_box), "ASGI")
arrow(bottom(fastapi_box), top(agent_box), ".invoke(state, config)")
arrow(bottom(agent_box), top(checkpointer), connection="arc3,rad=-0.15")
arrow(bottom(agent_box), top(corpus_store), connection="arc3,rad=0.15")

legend_items = [
    ("External", COLORS["external"]),
    ("Ngrok", COLORS["ngrok"]),
    ("Colab host process", COLORS["colab"]),
    ("FastAPI", COLORS["fastapi"]),
    ("LangGraph agent", COLORS["agent"]),
    ("In-memory state", COLORS["store"]),
]
ly = 2.5
for i, (label, color) in enumerate(legend_items):
    lx = 2 + i * 16
    ax.add_patch(FancyBboxPatch((lx, ly), 1.8, 1.8, boxstyle="round,pad=0.08", facecolor=color, edgecolor="none"))
    ax.text(lx + 2.4, ly + 0.9, label, fontsize=7.6, va="center", ha="left")

plt.tight_layout()
plt.savefig("/home/claude/ira_project/deployment_diagram.png", dpi=170, bbox_inches="tight", facecolor="white")
print("Saved deployment_diagram.png")
