"""
Le State unique partagé par les deux modes (agentic et RAG).

Reducers :
  messages        → add_messages      : append (historique du ReAct loop)
  tokens_*        → operator.add      : accumule sur chaque appel LLM
  tool_call_inputs→ operator.add      : accumule les queries Pinecone successives
  tool_call_count → operator.add      : compte les appels Pinecone
  llm_call_count  → operator.add      : compte les appels LLM de raisonnement
  retrieved_slugs → _merge_slug_counts: fusionne les dicts slug→count
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def _merge_slug_counts(a: dict | None, b: dict | None) -> dict:
    """Reducer : fusionne deux dicts slug→count en additionnant les comptes."""
    merged = dict(a or {})
    for slug, count in (b or {}).items():
        merged[slug] = merged.get(slug, 0) + count
    return merged


class AgentState(TypedDict, total=False):
    # --- ReAct loop (mode agentic) ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- Entrée RAG ---
    question: str

    # --- Retrieval RAG ---
    candidates: list[dict]
    scores: list[float]

    # --- Sortie commune ---
    final_apis: list[str]    # slugs recommandés extraits de la réponse finale
    answer_text: str         # texte brut complet (contient "RECOMMANDED_APIS: [...]")

    # --- Observabilité / debug ---
    retrieved_slugs: Annotated[dict[str, int], _merge_slug_counts]
    # slug → nb de fois retrievé (un même slug peut revenir sur plusieurs tool calls)

    tool_call_inputs: Annotated[list[str], operator.add]
    # queries envoyées à Pinecone dans l'ordre (une par tool call)

    tool_call_count: Annotated[int, operator.add]
    # nb d'appels à search_apis_tool (= nb de fois que Pinecone a été interrogé)

    llm_call_count: Annotated[int, operator.add]
    # nb d'appels LLM de raisonnement (toujours >= tool_call_count)

    # --- Tokens (operator.add = accumulation sur le ReAct loop) ---
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    tokens_think: Annotated[int, operator.add]
    tokens_total: Annotated[int, operator.add]
