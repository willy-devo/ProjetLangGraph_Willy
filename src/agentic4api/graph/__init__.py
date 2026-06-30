"""
Le graphe LangGraph — le cœur partagé entre le chat et le batch.

Sous-modules :
  - state.py     : AgentState (TypedDict) porté à travers les nœuds.
  - retriever.py : wrapper Pinecone (embed + query + scores bruts).
  - prompts.py   : system prompt (noms d'API génériques, format RECOMMANDED_APIS).
  - nodes.py     : agent_node / tools_node / should_continue (+ capture tokens & latence).
  - build.py     : assemble, compile, et expose `graph` (point d'import unique).

Imports pratiques (chargés à la demande pour ne pas tirer langgraph inutilement) :
    from agentic4api.graph import graph, build_graph, AgentState
"""

__all__ = ["graph", "build_graph", "AgentState"]


def __getattr__(name: str):
    if name in ("graph", "build_graph"):
        from agentic4api.graph import build as _build
        return getattr(_build, name)
    if name == "AgentState":
        from agentic4api.graph.state import AgentState
        return AgentState
    raise AttributeError(f"module 'agentic4api.graph' has no attribute '{name}'")
