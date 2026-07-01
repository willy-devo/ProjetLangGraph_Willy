"""
Helper LangFuse — retourne un CallbackHandler prêt à l'emploi.

Si LANGFUSE_ENABLED=false (ou si le package n'est pas installé),
toutes les fonctions retournent None sans lever d'erreur.
"""

from __future__ import annotations

from agentic4api.config.settings import settings


def get_handler(session_id: str = "", trace_name: str = ""):
    """Retourne un LangFuse CallbackHandler ou None si désactivé."""
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            public_key  = settings.langfuse_public_key,
            secret_key  = settings.langfuse_secret_key,
            host        = settings.langfuse_host,
            session_id  = session_id or None,
            trace_name  = trace_name or None,
        )
    except ImportError:
        return None


def callbacks(session_id: str = "", trace_name: str = "") -> list:
    """Retourne [handler] ou [] — utilisable directement dans config={'callbacks': ...}"""
    h = get_handler(session_id=session_id, trace_name=trace_name)
    return [h] if h else []
