"""
Assemble le graphe et expose `build_graph`.

Mode principal : "agentic" (défaut)
─────────────────────────────────────
  Le LLM reçoit `search_apis_tool` comme outil et pilote lui-même les appels
  Pinecone (ReAct loop). Fidèle au comportement N8N d'origine.
  Avantage : le LLM peut reformuler sa requête si le 1er résultat est pauvre.
  Inconvénient : tokens plus élevés, latence variable.

Mode optionnel : "rag"
──────────────────────
  START → retrieve → answer → END
  Pinecone est appelé UNE seule fois avant le LLM (pipeline fixe).
  Avantage : prévisible, debuggable, reproductible, tokens stables.
  Utile pour : comparer les deux architectures sur le golden dataset.

Changer de mode : RETRIEVAL_MODE=rag dans le .env (défaut : agentic).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentic4api.config.settings import settings
from agentic4api.graph.nodes import answer, retrieve
from agentic4api.graph.state import AgentState


def _build_rag(use_memory: bool = False):
    """Graphe RAG fixe : START → retrieve → answer → END."""
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)

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

    Mode par défaut : "agentic". Passer RETRIEVAL_MODE=rag dans le .env pour le mode fixe.
    """
    if settings.retrieval_mode == "rag":
        return _build_rag(use_memory=use_memory)
    return _build_agentic(use_memory=use_memory)


# Point d'import unique.
graph = build_graph()
