#!/usr/bin/env python3
"""
normalize_kong_response.py

Démontre, sans dépendances, ce que fait la classe `ChatOpenAI` de
`langchain-openai` quand elle reçoit une réponse au format OpenAI (ici renvoyée
par Kong, qui a lui-même traduit le natif Gemini → format OpenAI).

But pédagogique : montrer le passage
    JSON brut (prompt_tokens / completion_tokens)
            │  ← ce que fait ChatOpenAI
            ▼
    objet normalisé (usage_metadata : input_tokens / output_tokens)

Usage :
    python normalize_kong_response.py                 # utilise l'exemple intégré
    python normalize_kong_response.py reponse.json    # lit un fichier JSON
    echo '{...}' | python normalize_kong_response.py - # lit l'entrée standard
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────
# Exemple intégré (la réponse brute que tu as fournie)
# ──────────────────────────────────────────────────────────────────────────
EXEMPLE = {
    "model": "gemini-3.5-flash",
    "object": "chat.completion",
    "id": "KRk9arLeGZj8vdIPkvuv6Ak",
    "usage": {
        "prompt_tokens": 23,
        "completion_tokens": 35,
        "total_tokens": 919,
    },
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    '{\n  "name": "web_search",\n  "arguments": {\n'
                    '    "query": "API pour créer une commande client"\n  }\n}'
                ),
            },
            "finish_reason": "stop",
        }
    ],
    "created": 0,
}


# ──────────────────────────────────────────────────────────────────────────
# Ce que ChatOpenAI produit : un AIMessage normalisé
# (version simplifiée, sans LangChain, juste pour montrer la structure)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class AIMessageLike:
    """Imite l'objet renvoyé par ChatOpenAI.invoke()."""
    content: str
    usage_metadata: dict = field(default_factory=dict)        # format LangChain
    response_metadata: dict = field(default_factory=dict)     # format brut conservé
    tool_calls: list = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            "AIMessage(\n"
            f"    content={self.content!r},\n"
            f"    usage_metadata={self.usage_metadata},\n"
            f"    tool_calls={self.tool_calls},\n"
            f"    response_metadata={{'token_usage': "
            f"{self.response_metadata.get('token_usage')}}},\n"
            ")"
        )


def normalize_like_chatopenai(raw: dict) -> AIMessageLike:
    """Reproduit la normalisation faite par langchain-openai (ChatOpenAI).

    Points clés de ce que fait réellement la classe :
      1. extrait le message du 1er choice
      2. renomme les compteurs de tokens :
           prompt_tokens     -> input_tokens
           completion_tokens -> output_tokens
           total_tokens      -> total_tokens (inchangé)
         et range ça dans `usage_metadata` (format unifié, identique quel que
         soit le fournisseur).
      3. CONSERVE les champs bruts dans `response_metadata['token_usage']`
         (donc rien n'est recalculé : un total incohérent reste incohérent).
    """
    choice = raw["choices"][0]
    message = choice["message"]

    usage = raw.get("usage", {}) or {}
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", 0)

    # (2) le renommage — c'est LE cœur de ce que fait ChatOpenAI
    usage_metadata = {
        "input_tokens": prompt,         # ← prompt_tokens
        "output_tokens": completion,    # ← completion_tokens
        "total_tokens": total,          # ← inchangé, recopié tel quel
    }

    # tool_calls éventuels (si finish_reason == "tool_calls")
    tool_calls = []
    for tc in message.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            pass
        tool_calls.append({"name": fn.get("name"), "args": args, "id": tc.get("id")})

    return AIMessageLike(
        content=message.get("content") or "",
        usage_metadata=usage_metadata,
        response_metadata={"token_usage": usage},   # (3) brut conservé
        tool_calls=tool_calls,
    )


def usage_delta(response: AIMessageLike) -> dict:
    """Ta fonction _usage_delta() : reconstruit le thinking caché.

    Le total inclut le raisonnement interne du modèle, mais output_tokens ne
    compte que la sortie visible → think = total - in - out.
    """
    u = response.usage_metadata or {}
    t_in = u.get("input_tokens", 0)
    t_out = u.get("output_tokens", 0)
    t_total = u.get("total_tokens", 0)
    t_think = max(0, t_total - t_in - t_out)
    return {
        "tokens_in": t_in,
        "tokens_out": t_out,
        "tokens_think": t_think,
        "tokens_total": t_total,
    }


def _load_input(argv: list[str]) -> dict:
    """Récupère le JSON : argument fichier, stdin (-), ou exemple intégré."""
    if len(argv) >= 2 and argv[1] not in ("-", ""):
        with open(argv[1], "r", encoding="utf-8") as f:
            return json.load(f)
    if len(argv) >= 2 and argv[1] == "-":
        return json.load(sys.stdin)
    return EXEMPLE


def main() -> None:
    raw = _load_input(sys.argv)

    print("=" * 70)
    print("1) RÉPONSE BRUTE (format OpenAI renvoyé par Kong)")
    print("=" * 70)
    print(f"  usage.prompt_tokens     = {raw['usage']['prompt_tokens']}")
    print(f"  usage.completion_tokens = {raw['usage']['completion_tokens']}")
    print(f"  usage.total_tokens      = {raw['usage']['total_tokens']}")

    response = normalize_like_chatopenai(raw)

    print()
    print("=" * 70)
    print("2) APRÈS ChatOpenAI (objet normalisé renvoyé par .invoke())")
    print("=" * 70)
    print(response)

    print()
    print("=" * 70)
    print("3) TES MÉTRIQUES (via _usage_delta — reconstruit le thinking)")
    print("=" * 70)
    delta = usage_delta(response)
    for k, v in delta.items():
        print(f"  {k:14s} = {v}")

    # contrôle de cohérence
    incoherence = delta["tokens_total"] - delta["tokens_in"] - delta["tokens_out"]
    print()
    print("  Vérif : total - in - out =", incoherence,
          "→ ce sont les tokens de raisonnement (thinking) cachés,")
    print("          présents dans total_tokens mais absents de completion_tokens.")


if __name__ == "__main__":
    main()