"""
Serveur MCP — expose search_apis comme outil MCP (pour Claude Desktop, etc.)
et comme endpoint REST /search (utilisé en interne par nodes.py quand USE_MCP=true).

Lancer en local :
    python -m agentic4api.mcp.server

Dans Docker :
    docker compose up mcp
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from agentic4api.config.settings import settings
from agentic4api.graph.tools import _format_candidate
from agentic4api.graph.retriever import search

# ── MCP (Claude Desktop et autres clients MCP) ────────────────────────────
_mcp = FastMCP("agentic4api-search")


@_mcp.tool()
def search_apis(query: str) -> str:
    """
    Cherche des APIs dans le catalogue Pinecone pour une requête donnée.
    Retourne les APIs les plus pertinentes avec leur nom, statut et description.
    """
    results = search(query, top_k=settings.top_k)
    if not results:
        return "Aucun résultat trouvé pour cette requête."
    return "\n".join(_format_candidate(r) for r in results)


# ── REST (nodes.py interne quand USE_MCP=true) ────────────────────────────
app = FastAPI(title="agentic4api MCP Server", version="1.0.0")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


@app.post("/search")
def search_endpoint(req: SearchRequest):
    """Endpoint REST interne — retourne les candidats Pinecone bruts."""
    results = search(req.query, top_k=req.top_k)
    return {"results": results}


@app.get("/health")
def health():
    return {"status": "ok"}


# Monte le SSE MCP sous /mcp (ex: http://mcp:8090/mcp/sse pour Claude Desktop)
app.mount("/mcp", _mcp.sse_app())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
