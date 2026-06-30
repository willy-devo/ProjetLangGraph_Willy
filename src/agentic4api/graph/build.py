"""
Assemble le graphe et expose `build_graph`.

  START → agent ─┬─(outil demandé)→ tools → agent → ...
                 └─(réponse finale)────────────────→ END

  Le LLM décide lui-même quand/combien de fois appeler Pinecone (ReAct loop).

  Deux variantes selon TOOL_MODE dans le .env :
    "text"       (défaut) : le LLM écrit SEARCH: <requête> dans son texte
                            → compatible Kong (pas de thought_signature)
    "bind_tools"          : OpenAI structured tool calls (bind_tools)
                            → nécessite que Kong transmette la thought_signature Gemini
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentic4api.config.settings import settings
from agentic4api.graph.nodes import (
    agent_node, should_continue, tools_node,
    agent_node_bt, should_continue_bt, tools_node_bt,
)
from agentic4api.graph.state import AgentState


def build_graph(*, use_memory: bool = True):
    """
    use_memory=False (batch) : chaque question indépendante, pas de MemorySaver.
    use_memory=True  (chat)  : mémoire par thread_id (non utilisé — stateless).
    """
    if settings.tool_mode == "bind_tools":
        _agent, _tools, _route = agent_node_bt, tools_node_bt, should_continue_bt
    else:
        _agent, _tools, _route = agent_node, tools_node, should_continue

    g = StateGraph(AgentState)
    g.add_node("agent", _agent)
    g.add_node("tools", _tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    checkpointer = MemorySaver() if use_memory else None
    return g.compile(checkpointer=checkpointer)


# Point d'import unique — stateless : chaque question repart de zéro.
graph = build_graph(use_memory=False)
