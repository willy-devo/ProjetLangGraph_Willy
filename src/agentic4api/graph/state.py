"""
Le State du graphe — porté à travers tous les nœuds.

C'est ici que vivent les compteurs (retries, tokens) qui s'ACCUMULENT au fil des
nœuds : un agent fait plusieurs appels LLM par question (raisonnement + tool calls),
donc les tokens doivent s'additionner, pas s'écraser. Le State est exactement
l'endroit prévu pour ça en LangGraph.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import operator


class AgentState(TypedDict, total=False):
    # --- Entrée ---
    question: str

    # --- Retrieval ---
    candidates: list[dict]   # [{"slug": "...", "score": 0.87, "text": "..."}]
    scores: list[float]      # scores bruts Pinecone (indispensables au threshold déterministe)

    # --- Sortie ---
    final_apis: list[str]    # slugs recommandés (vide = négatif assumé)
    answer_text: str         # texte brut de l'agent (contient "RECOMMANDED_APIS: [...]")

    # --- Garde / contrôle ---
    is_corrupted: bool       # question vide / illisible → court-circuit

    # --- Mesures par question (Annotated[..., operator.add] = accumulation) ---
    retries: int
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    tokens_think: Annotated[int, operator.add]   # calculé : total - in - out (raisonnement interne Gemini)
    tokens_total: Annotated[int, operator.add]
    # latency_s n'est PAS ici : mesurée bout-en-bout autour de l'invoke dans run_batch.
