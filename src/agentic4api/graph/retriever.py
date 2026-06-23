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

import json
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


def _parse_metadata(md: dict) -> dict:
    """Parse le champ 'text' des métadonnées Pinecone.

    Le champ text est un JSON imbriqué :
      {"id": "api-catalogue-500/telemedicine-api",
       "text": "Telemedicine API. Description...",
       "content_brut": "{\"openapi\":\"3.0.0\", \"info\": {\"title\": ..., \"x-status\": ...}}"}

    On en extrait :
      - slug   : dernière partie de id  (ex. "telemedicine-api")
      - title  : info.title du spec OpenAPI
      - status : info.x-status du spec OpenAPI ("active" | "deprecated")
      - text   : description lisible (champ text interne)
    """
    raw = md.get("text", "")
    slug = ""
    title = ""
    status = "unknown"
    description = raw

    try:
        data = json.loads(raw)
        api_id = data.get("id", "")
        slug = api_id.split("/")[-1] if api_id else ""
        description = data.get("text", raw)

        content_brut = data.get("content_brut", "")
        if content_brut:
            try:
                openapi = json.loads(content_brut)
                info = openapi.get("info", {})
                title = info.get("title", "")
                status = info.get("x-status", "unknown")
                if not slug:
                    slug = info.get("x-api-id", "")
            except (json.JSONDecodeError, AttributeError):
                pass
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback si le JSON n'a pas pu être parsé
    if not slug:
        slug = md.get("slug") or md.get("api_name") or md.get("name", "")

    return {"slug": slug, "title": title, "status": status, "text": description}


def search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Renvoie une liste de candidats triés par score décroissant :
        [{"slug": str, "title": str, "status": str, "score": float, "text": str}, ...]
    """
    top_k = top_k or settings.top_k
    vector = _embed(query)
    res = _index().query(vector=vector, top_k=top_k, include_metadata=True)

    out: list[dict] = []
    for match in res.matches:
        parsed = _parse_metadata(match.metadata or {})
        out.append({**parsed, "score": float(match.score)})
    return out
