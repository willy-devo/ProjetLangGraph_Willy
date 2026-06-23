"""
Assemble le graphe et expose `build_graph`.

Deux modes contrôlés par RETRIEVAL_MODE dans le .env :

  "rag" (défaut — déterministe)
    START → guard ─┬─(corrompue)→ corrupted_answer → END
                   └─(ok)────────→ retrieve → answer → END
    Pinecone est appelé UNE seule fois, systématiquement, avant le LLM.
    Avantage : prévisible, facilement debuggable, reproductible.

  "agentic" (fidèle au mode N8N — le LLM pilote)
    Le LLM reçoit `search_apis_tool` comme outil et décide lui-même
    quand / combien de fois appeler Pinecone (ReAct loop, max 5 itérations).
    Avantage : le LLM peut reformuler sa requête si le 1er résultat est pauvre.
    Inconvénient : moins prévisible, tokens plus élevés, latence variable.

`build_graph` renvoie dans les deux cas un graphe dont l'interface d'invoke
est normalisée : entrée {"question": str}, sortie avec answer_text / final_apis.
Pour le mode agentic, run_batch.py détecte settings.retrieval_mode et extrait
la réponse depuis les messages (format create_react_agent).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentic4api.config.settings import settings
from agentic4api.graph.nodes import answer, corrupted_answer, guard, retrieve
from agentic4api.graph.state import AgentState


def _route_after_guard(state: AgentState) -> str:
    return "corrupted" if state.get("is_corrupted") else "ok"


def _build_rag(use_memory: bool = False):
    """Graphe RAG fixe : retrieve toujours appelé avant le LLM."""
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

    if not use_memory:
        return g.compile()
    return g.compile(checkpointer=MemorySaver())


def _build_agentic(use_memory: bool = False):
    """Graphe agentic : le LLM appelle Pinecone comme outil (ReAct loop)."""
    from langgraph.prebuilt import create_react_agent

    from agentic4api.graph.nodes import _llm, search_apis_tool
    from agentic4api.graph.prompts import SYSTEM_PROMPT

    checkpointer = MemorySaver() if use_memory else None
    return create_react_agent(
        _llm(),
        tools=[search_apis_tool],
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def build_graph(checkpointer: MemorySaver | None = None, *, use_memory: bool = True):
    """
    use_memory=True  (chat) : compile avec MemorySaver — mémoire par thread_id.
    use_memory=False (batch): sans checkpointer — chaque question est indépendante.

    Le mode (rag / agentic) est lu depuis settings.retrieval_mode.
    """
    if settings.retrieval_mode == "agentic":
        return _build_agentic(use_memory=use_memory)
    return _build_rag(use_memory=use_memory)


# Point d'import unique.
graph = build_graph()
