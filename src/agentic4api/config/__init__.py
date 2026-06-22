"""
Configuration centralisée du projet.

Expose l'instance unique `settings` — UN seul endroit pour tous les hyperparamètres
(model string Gemini, modèle d'embedding, topK Pinecone, seuils, secrets via .env).
Centraliser ici sert le principe d'isolation des variables : pour comparer deux
configs proprement, on change un paramètre à un seul endroit.

Usage :
    from agentic4api.config import settings
    print(settings.gemini_model, settings.top_k)
"""

from agentic4api.config.settings import Settings, settings

__all__ = ["settings", "Settings"]
