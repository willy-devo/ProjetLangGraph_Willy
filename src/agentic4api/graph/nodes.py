"""
Les nœuds du graphe.

- guard    : court-circuite les questions vides/corrompues.
- retrieve : appelle Pinecone, stocke candidats + scores dans le State.
- answer   : Gemini décide/formate `RECOMMANDED_APIS: [...]`, et on CAPTURE les tokens.

Pourquoi un transport httpx custom :
  Kong expose le chat sur /ai-api/v1/chat/gemini, pas sur /chat/completions (standard
  OpenAI). ChatOpenAI ajoute toujours /chat/completions à la base URL — on ne peut pas
  l'en empêcher. Le transport redirige silencieusement la requête vers l'URL Kong exacte,
  sans changer le reste de la chaîne LangChain (streaming, usage_metadata, etc.).

Sur la capture des tokens (OpenAI-compat, validé sur réponse Kong) :
  LangChain normalise prompt_tokens/completion_tokens → input_tokens/output_tokens dans
  usage_metadata. On utilise donc les mêmes clés qu'avec l'API Google native.
  tokens_think sera toujours 0 : Kong ne remonte pas les reasoning tokens séparément.
"""

from __future__ import annotations

import re
from functools import lru_cache

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

from agentic4api.config.settings import settings
from agentic4api.graph.prompts import SYSTEM_PROMPT
from agentic4api.graph.retriever import search
from agentic4api.graph.state import AgentState
from agentic4api.graph.transports import AsyncKongChatTransport, KongChatTransport

_RECO_RE = re.compile(r"RECOMMAN?DED_APIS\s*:\s*\[([^\]]*)\]", re.IGNORECASE)


@lru_cache(maxsize=1)
def _llm() -> ChatOpenAI:
    """Construit le client ChatOpenAI câblé sur la route Kong."""
    verify = settings.kong_verify_ssl
    return ChatOpenAI(
        base_url="http://kong-placeholder/v1",  # ignoré par le transport
        api_key=settings.kong_api_key,
        model=settings.gemini_model,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        http_client=httpx.Client(
            transport=KongChatTransport(settings.kong_chat_url, verify=verify)
        ),
        async_client=httpx.AsyncClient(
            transport=AsyncKongChatTransport(settings.kong_chat_url, verify=verify)
        ),
    )


def _usage_delta(response) -> dict:
    """Extrait les tokens d'UNE réponse pour accumulation dans le State.

    LangChain normalise prompt_tokens/completion_tokens → input_tokens/output_tokens.
    """
    u = getattr(response, "usage_metadata", None) or {}
    t_in    = u.get("input_tokens", 0)
    t_out   = u.get("output_tokens", 0)
    t_total = u.get("total_tokens", 0)
    return {
        "tokens_in":    t_in,
        "tokens_out":   t_out,
        "tokens_think": max(0, t_total - t_in - t_out),  # raisonnement interne Gemini
        "tokens_total": t_total,
    }


def _parse_apis(text: str) -> list[str]:
    """Extrait les slugs de la ligne RECOMMANDED_APIS: [...]."""
    m = _RECO_RE.search(text or "")
    if not m:
        return []
    inner = m.group(1).strip()
    if not inner:
        return []
    return [s.strip().strip("`*\"' ") for s in inner.split(",") if s.strip()]


# ── Outil Pinecone (mode agentic) ──────────────────────────────────────────

def _format_candidate(c: dict, text_limit: int = 300) -> str:
    """Formate un candidat Pinecone pour le prompt — même format en RAG et agentic."""
    return (
        f"- name: {c['slug']} | title: {c.get('title', '')} "
        f"| statut: {c.get('status', 'unknown')} | score: {c['score']:.3f}\n"
        f"  description: {c['text'][:text_limit]}"
    )


@lc_tool
def search_apis_tool(query: str) -> str:
    """Recherche sémantique d'APIs internes selon un besoin fonctionnel.
    Renvoie les candidats avec name, title, statut et description."""
    results = search(query, top_k=settings.top_k)
    if not results:
        return "Aucun résultat trouvé pour cette requête."
    return "\n".join(_format_candidate(r) for r in results)


# ── Nœuds ──────────────────────────────────────────────────────────────────

def guard(state: AgentState) -> dict:
    q = (state.get("question") or "").strip()
    return {"is_corrupted": len(q) < 3, "retries": 0}


def retrieve(state: AgentState) -> dict:
    candidates = search(state["question"])
    return {
        "candidates": candidates,
        "scores": [c["score"] for c in candidates],
    }


def answer(state: AgentState) -> dict:
    candidates = state.get("candidates", [])
    context = "\n".join(_format_candidate(c) for c in candidates)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Demande : {state['question']}\n\nCandidats Pinecone :\n{context}"),
    ]
    response = _llm().invoke(messages)
    text = response.content if isinstance(response.content, str) else str(response.content)

    out = {"answer_text": text, "final_apis": _parse_apis(text)}
    out.update(_usage_delta(response))
    return out


def corrupted_answer(state: AgentState) -> dict:
    """Réponse de court-circuit pour une question corrompue."""
    return {"answer_text": "RECOMMANDED_APIS: []", "final_apis": []}
