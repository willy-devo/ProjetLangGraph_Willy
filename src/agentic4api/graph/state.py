"""
Le State du graphe — porté à travers tous les nœuds (mode RAG uniquement).

En mode agentic (défaut), create_react_agent gère son propre State interne
basé sur messages — AgentState n'est utilisé qu'en mode RAG optionnel.

Les tokens utilisent Annotated[int, operator.add] pour s'accumuler sur
plusieurs appels LLM sans s'écraser.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import operator


class AgentState(TypedDict, total=False):
    # --- Entrée ---
    question: str

    # --- Retrieval ---
    candidates: list[dict]   # [{"slug": "...", "score": 0.87, "text": "..."}]
    scores: list[float]      # scores bruts Pinecone

    # --- Sortie ---
    final_apis: list[str]    # slugs recommandés (vide = négatif assumé)
    answer_text: str         # texte brut de l'agent (contient "RECOMMANDED_APIS: [...]")

    # --- Mesures par question (Annotated[..., operator.add] = accumulation) ---
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    tokens_think: Annotated[int, operator.add]   # calculé : total - in - out (raisonnement interne Gemini)
    tokens_total: Annotated[int, operator.add]
    # latency_s n'est PAS ici : mesurée bout-en-bout autour de l'invoke dans run_batch.
