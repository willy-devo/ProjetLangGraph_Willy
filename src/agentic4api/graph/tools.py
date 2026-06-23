"""
Outils (tools) disponibles pour l'agent agentic.

Chaque fonction décorée @lc_tool devient un outil que le LLM peut appeler
librement pendant le ReAct loop. La docstring EST le message envoyé au LLM
pour qu'il comprenne quand et comment utiliser l'outil.
"""

from __future__ import annotations

from langchain_core.tools import tool as lc_tool

from agentic4api.config.settings import settings
from agentic4api.graph.retriever import search


def _format_candidate(c: dict, text_limit: int = 300) -> str:
    return (
        f"- name: {c['slug']} | title: {c.get('title', '')} "
        f"| statut: {c.get('status', 'unknown')} | score: {c['score']:.3f}\n"
        f"  description: {c['text'][:text_limit]}"
    )


@lc_tool
def search_apis_tool(query: str) -> str:
    """
    Catalogue de recherche sémantique des APIs internes de Devoteam nexDigital,
    couvrant les domaines métier de l'entreprise (e-commerce, paiement, logistique,
    RH, analytics, etc.).

    À utiliser dès qu'un développeur cherche une API, un endpoint ou une capacité
    technique existante — qu'il décrive une action métier ("créer une commande"),
    une fonctionnalité produit ("une barre de recherche avec autocomplete"), un
    déclencheur ("quand l'utilisateur ajoute au panier") ou un simple besoin vague
    ("gérer les stocks").

    La recherche se fait par SENS et non par mot-clé : une question formulée de
    façon familière, urgente, abrégée ou approximative reste valide. Appelle
    l'outil avec le besoin fonctionnel exprimé.

    Si aucun résultat ne correspond réellement au besoin, considère que la
    capacité n'existe pas au catalogue plutôt que de forcer une correspondance.
    """
    results = search(query, top_k=settings.top_k)
    if not results:
        return "Aucun résultat trouvé pour cette requête."
    return "\n".join(_format_candidate(r) for r in results)
