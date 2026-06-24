"""
Assemble le graphe et expose `build_graph`.

Les deux modes utilisent StateGraph(AgentState) — contrôle total sur le State.

Mode principal : "agentic" (défaut)
─────────────────────────────────────
  START → agent ─┬─(tool call)→ tools → agent → ...
                 └─(réponse finale)────────────→ END
  Le LLM décide lui-même quand/combien de fois appeler Pinecone (ReAct loop).
  Tokens capturés à chaque appel LLM et accumulés dans AgentState.

Mode optionnel : "rag"
──────────────────────
  START → retrieve → answer → END
  Pinecone appelé une seule fois avant le LLM (pipeline fixe).
  Utile pour comparer les deux architectures sur le golden dataset.

Changer de mode : RETRIEVAL_MODE=rag dans le .env (défaut : agentic).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentic4api.config.settings import settings
from agentic4api.graph.nodes import agent_node, answer, retrieve, should_continue, tools_node
from agentic4api.graph.state import AgentState


def _build_agentic(use_memory: bool = False):
    """ReAct loop custom : AgentState + contrôle total sur tokens et observabilité."""
    g = StateGraph(AgentState)

    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    checkpointer = MemorySaver() if use_memory else None
    return g.compile(checkpointer=checkpointer)


def _build_rag(use_memory: bool = False):
    """Pipeline fixe : START → retrieve → answer → END."""
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)

    checkpointer = MemorySaver() if use_memory else None
    return g.compile(checkpointer=checkpointer)


def build_graph(*, use_memory: bool = True):
    """
    use_memory=False (batch) : chaque question indépendante, pas de MemorySaver.
    use_memory=True  (chat)  : mémoire par thread_id (non utilisé — stateless).

    Mode par défaut : "agentic". Passer RETRIEVAL_MODE=rag dans le .env pour le mode fixe.
    """
    if settings.retrieval_mode == "rag":
        return _build_rag(use_memory=use_memory)
    return _build_agentic(use_memory=use_memory)


# Point d'import unique — stateless : chaque question repart de zéro.
graph = build_graph(use_memory=False)
