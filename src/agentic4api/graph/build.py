"""
Assemble le StateGraph, compile, et EXPOSE `graph`.

C'est LE point d'import unique : `chat/app.py` ET `batch/run_batch.py` importent
tous deux `graph` d'ici. Un seul cerveau, deux bouches → l'éval teste exactement
la même orchestration que la prod (pas de confound harness vs prod).

Graphe Phase 1 (équivalent n8n, mono-outil) :
    START → guard ─┬─(corrompue)→ corrupted_answer → END
                   └─(ok)────────→ retrieve → answer → END

Les nœuds decompose / threshold (Phase 3) viendront s'insérer ici, un par un,
chacun mesuré séparément.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentic4api.graph.nodes import answer, corrupted_answer, guard, retrieve
from agentic4api.graph.state import AgentState


def _route_after_guard(state: AgentState) -> str:
    return "corrupted" if state.get("is_corrupted") else "ok"


def build_graph(checkpointer: MemorySaver | None = None, *, use_memory: bool = True):
    """
    use_memory=True  (défaut, pour le CHAT) : compile avec un MemorySaver — mémoire
        conversationnelle par thread_id.
    use_memory=False (pour le BATCH) : compile SANS checkpointer — chaque question est
        indépendante, pas besoin de mémoire (évite d'allouer un MemorySaver pour rien
        sur 464 questions).

    NB Cloud Run : MemorySaver est EN MÉMOIRE VIVE. Sur un service qui scale/redémarre,
    la mémoire conversationnelle n'est pas partagée entre conteneurs ni persistée. OK
    pour des questions one-shot ; pour une vraie mémoire persistante, brancher un
    checkpointer externe (Postgres/Redis).
    """
    g = StateGraph(AgentState)

    g.add_node("guard", guard)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("corrupted_answer", corrupted_answer)

    g.add_edge(START, "guard")
    g.add_conditional_edges(
        "guard",
        _route_after_guard,
        {"ok": "retrieve", "corrupted": "corrupted_answer"},
    )
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    g.add_edge("corrupted_answer", END)

    # Chat : MemorySaver (mémoire par thread_id). Batch : aucun checkpointer.
    if not use_memory:
        return g.compile()
    return g.compile(checkpointer=checkpointer or MemorySaver())


# Point d'import unique.
graph = build_graph()
