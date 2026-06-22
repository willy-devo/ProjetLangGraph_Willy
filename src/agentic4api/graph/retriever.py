"""
Wrapper Pinecone : embed la question (via Kong) → query l'index → renvoie
les candidats AVEC leurs scores bruts.

Pourquoi on n'utilise pas OpenAIEmbeddings :
  OpenAIEmbeddings (langchain-openai) tokenise le texte avec tiktoken ou
  transformers avant d'envoyer la requête. Kong/Gemini attend du texte brut
  ({"input": "ma question"}) — les token IDs font planter le backend. On appelle
  donc Kong directement avec httpx, ce qui est plus simple et évite la dépendance
  à un tokeniseur externe.

Les scores doivent remonter jusqu'au State pour le futur nœud threshold déterministe.
"""

from __future__ import annotations

from functools import lru_cache

import httpx
from pinecone import Pinecone

from agentic4api.config.settings import settings
from agentic4api.graph.transports import KongEmbedTransport


@lru_cache(maxsize=1)
def _http_client() -> httpx.Client:
    return httpx.Client(
        transport=KongEmbedTransport(verify=settings.kong_verify_ssl),
        timeout=30.0,
    )


def _embed(text: str) -> list[float]:
    """Appel direct à Kong embeddings — renvoie le vecteur brut."""
    resp = _http_client().post(
        settings.kong_embed_url,
        json={"input": text, "model": settings.gemini_embed_model},
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


@lru_cache(maxsize=1)
def _index():
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_host:
        return pc.Index(host=settings.pinecone_host)
    return pc.Index(settings.pinecone_index)


def search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Renvoie une liste de candidats triés par score décroissant :
        [{"slug": str, "score": float, "text": str}, ...]

    ⚠ VÉRIFIER : les clés de metadata (`slug`, `text`) doivent correspondre à ce
    qui a réellement été indexé dans Pinecone par scripts/index_pinecone.py.
    """
    top_k = top_k or settings.top_k
    vector = _embed(query)
    res = _index().query(vector=vector, top_k=top_k, include_metadata=True)

    out: list[dict] = []
    for match in res.matches:                    # Pinecone v5 : attribut, pas dict
        md = match.metadata or {}
        out.append(
            {
                "slug": md.get("slug") or md.get("api_name") or md.get("name", ""),
                "score": float(match.score),
                "text": md.get("text") or md.get("content", ""),
            }
        )
    return out
