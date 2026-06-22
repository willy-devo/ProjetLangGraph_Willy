"""
(Ré)indexation du catalogue OpenAPI dans Pinecone.

SQUELETTE à compléter avec ta logique de chunking (slug depuis le filename,
stableStringify, content_sha, etc.). Ce qui compte ici : les clés de metadata
que tu écris DOIVENT correspondre à ce que retriever.py lit (slug, text).

⚠ VÉRIFIER : aligne les noms de champs metadata avec graph/retriever.py.
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

from agentic4api.config.settings import settings


def index_catalog(chunks: list[dict]) -> None:
    """
    `chunks` : [{"slug": str, "text": str, ...metadata}, ...]
    """
    embedder = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embed_model,
        google_api_key=settings.google_api_key,
    )
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index)

    vectors = []
    for i, c in enumerate(chunks):
        vec = embedder.embed_query(c["text"])
        vectors.append({
            "id": c.get("slug", f"chunk-{i}"),
            "values": vec,
            "metadata": {"slug": c["slug"], "text": c["text"]},  # ⚠ doit matcher retriever.py
        })

    # upsert par batch de 100 (limite Pinecone)
    for start in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[start:start + 100])

    print(f"✅ {len(vectors)} vecteurs indexés dans « {settings.pinecone_index} ».")


if __name__ == "__main__":
    # TODO : charger tes specs OpenAPI, chunker, puis index_catalog(chunks)
    raise SystemExit("Complète le chargement/chunking de tes specs avant de lancer.")
