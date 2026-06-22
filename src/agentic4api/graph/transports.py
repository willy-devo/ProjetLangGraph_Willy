"""
Transports httpx pour les routes Kong.

Deux problèmes résolus ici :
  1. Le chemin /chat/completions (standard OpenAI) ≠ /chat/gemini (route Kong) →
     KongChatTransport réécrit l'URL à chaque requête.
  2. Kong gère sa propre auth vers Google en interne. Si on lui envoie
     "Authorization: Bearer dummy", il le forward à Google qui refuse (401).
     Les deux transports suppriment donc ce header avant d'envoyer.
"""

from __future__ import annotations

import httpx


def _strip_auth(headers: httpx.Headers) -> dict[str, str]:
    """Copie les headers en supprimant Authorization."""
    return {k: v for k, v in headers.items() if k.lower() != "authorization"}


class KongChatTransport(httpx.BaseTransport):
    """Redirige vers l'URL Kong exacte + supprime Authorization."""

    def __init__(self, target_url: str, verify: bool = True):
        self._target = httpx.URL(target_url)
        self._inner = httpx.HTTPTransport(verify=verify)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        new_req = httpx.Request(
            method=request.method,
            url=self._target,
            headers=_strip_auth(request.headers),
            content=request.content,
        )
        return self._inner.handle_request(new_req)


class AsyncKongChatTransport(httpx.AsyncBaseTransport):
    """Version async de KongChatTransport (nécessaire pour astream_events)."""

    def __init__(self, target_url: str, verify: bool = True):
        self._target = httpx.URL(target_url)
        self._inner = httpx.AsyncHTTPTransport(verify=verify)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        new_req = httpx.Request(
            method=request.method,
            url=self._target,
            headers=_strip_auth(request.headers),
            content=request.content,
        )
        return await self._inner.handle_async_request(new_req)


class KongEmbedTransport(httpx.BaseTransport):
    """URL inchangée (chemin /embeddings = standard OpenAI) + supprime Authorization."""

    def __init__(self, verify: bool = True):
        self._inner = httpx.HTTPTransport(verify=verify)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        new_req = httpx.Request(
            method=request.method,
            url=request.url,
            headers=_strip_auth(request.headers),
            content=request.content,
        )
        return self._inner.handle_request(new_req)
