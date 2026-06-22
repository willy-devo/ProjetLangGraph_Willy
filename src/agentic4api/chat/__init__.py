# ─────────────────────────────────────────────────────────────────────────────
# EMPLACEMENT DE CE FICHIER : src/agentic4api/chat/__init__.py
# (dans le sous-dossier chat/, à côté de app.py)
# ⚠ Ce fichier contenait par erreur le contenu du __init__ RACINE — remplace-le.
# ─────────────────────────────────────────────────────────────────────────────
"""
chat — interface conversationnelle Chainlit (prod).

Importe le graphe compilé (`agentic4api.graph.build.graph`) et l'expose en UI de
chat streaming. C'est le point d'entrée déployé sur Cloud Run (service) pour que
les devs / Marc puissent interroger l'agent de découverte d'API.

Sous-modules :
  - app.py : application Chainlit (graph.astream_events → streaming des tokens).
"""
