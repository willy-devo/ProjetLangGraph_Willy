"""
Tests du graphe.

Deux niveaux, tous SANS réseau :
  1. Parsing (purement local) : _parse_apis.
  2. Smoke test d'invocation : graph.invoke() de bout en bout en mode agentic,
     avec _llm_with_tools et search MOCKÉS. Vérifie le câblage des nœuds,
     l'accumulation des tokens et le format de sortie.

Pour un vrai test d'intégration (vrai Gemini + vrai Pinecone), lance plutôt :
    agentic4api-batch --limit 2
"""

from __future__ import annotations

import pytest

from agentic4api.graph.nodes import _parse_apis


# ── 1. Parsing (sans dépendances) ───────────────────────────────────────────

def test_parse_apis_basic():
    assert _parse_apis("RECOMMANDED_APIS: [order-api-v4]") == ["order-api-v4"]


def test_parse_apis_multi():
    assert _parse_apis("bla\nRECOMMENDED_APIS: [auth-api, mfa-api]") == ["auth-api", "mfa-api"]


def test_parse_apis_empty():
    assert _parse_apis("RECOMMANDED_APIS: []") == []


def test_parse_apis_single_m_tolerated():
    assert _parse_apis("RECOMMANDED_APIS: [foo-bar-api]") == ["foo-bar-api"]


def test_parse_apis_strips_quotes_and_stars():
    assert _parse_apis('RECOMMANDED_APIS: [`foo-bar-api`, "baz-api"]') == ["foo-bar-api", "baz-api"]


# ── 2. Smoke test : invocation complète du graphe, mocks sans réseau ────────

class _FakeLLMResponse:
    """Imite la réponse d'un ChatOpenAI en mode agentic (contenu + tokens, pas de tool_call)."""
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []  # liste vide → réponse finale, pas d'appel outil
        self.usage_metadata = {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
        }


class _FakeBoundLLM:
    def invoke(self, messages):
        return _FakeLLMResponse("Voici l'API.\nRECOMMENDED_APIS: [order-api-v4]")


@pytest.fixture
def patched_graph(monkeypatch, mock_candidates):
    """
    Graphe agentic dont _llm_with_tools et search sont mockés :
      - _llm_with_tools → retourne directement une réponse finale (pas de tool call)
      - search          → retourne mock_candidates (pas de Pinecone)
    """
    import agentic4api.graph.nodes as nodes

    monkeypatch.setattr(nodes, "_llm_with_tools", lambda: _FakeBoundLLM())
    monkeypatch.setattr(nodes, "search", lambda q, top_k=None: mock_candidates)

    from agentic4api.graph.build import build_graph
    return build_graph(use_memory=False)


def test_graph_invoke_normal_question(patched_graph):
    state = patched_graph.invoke({"messages": [("human", "je veux créer une commande client")]})
    assert state["final_apis"] == ["order-api-v4"]
    assert state["tokens_in"] == 120
    assert state["tokens_total"] == 150
    assert state["llm_call_count"] == 1
