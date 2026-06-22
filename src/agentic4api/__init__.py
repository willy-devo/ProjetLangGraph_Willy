# ─────────────────────────────────────────────────────────────────────────────
# EMPLACEMENT DE CE FICHIER : src/agentic4api/__init__.py
# (à la racine du package, juste sous src/agentic4api/)
# ─────────────────────────────────────────────────────────────────────────────
"""
agentic4api — découverte sémantique d'API.

Un agent LangGraph (Pinecone + Gemini) partagé entre deux points d'entrée :
  - chat/   : interface conversationnelle Chainlit (prod).
  - batch/  : génération des réponses du golden dataset → écriture Google Sheet (éval).

L'évaluation des métriques (MRR, nDCG, Recall…) reste dans le notebook Colab,
qui lit le Sheet rempli par le batch.

Import du graphe (chargé seulement quand on y accède, pas à l'import du package) :
    from agentic4api import graph
ou, plus explicite :
    from agentic4api.graph.build import graph, build_graph
"""

__version__ = "0.1.0"

__all__ = ["graph", "__version__"]


def __getattr__(name: str):
    """
    Import PARESSEUX : accéder à `agentic4api.graph` ne charge langgraph/gemini QUE
    si on accède réellement à `graph`. Ça évite que des imports légers (ex.
    `from agentic4api.batch.golden import load_golden`) tirent toute la stack LLM.
    """
    if name == "graph":
        from agentic4api.graph.build import graph
        return graph
    raise AttributeError(f"module 'agentic4api' has no attribute '{name}'")
