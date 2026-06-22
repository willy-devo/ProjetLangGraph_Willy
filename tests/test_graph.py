"""
Tests du graphe.

Deux niveaux, tous SANS réseau :
  1. Parsing & routing (purement local) : _parse_apis, guard.
  2. Smoke test d'invocation : graph.invoke() de bout en bout, avec le retriever
     Pinecone et le LLM Gemini MOCKÉS (monkeypatch). Vérifie que le câblage des
     nœuds, l'accumulation des tokens et le format de sortie tiennent ensemble.

Pour un vrai test d'intégration (vrai Gemini + vrai Pinecone), lance plutôt :
    agentic4api-batch --limit 2
"""

from __future__ import annotations

import pytest

from agentic4api.graph.nodes import _parse_apis, guard


# ── 1. Parsing & routing (sans dépendances) ─────────────────────────────────

def test_parse_apis_basic():
    assert _parse_apis("RECOMMANDED_APIS: [order-api-v4]") == ["order-api-v4"]


def test_parse_apis_multi():
    assert _parse_apis("bla\nRECOMMANDED_APIS: [auth-api, mfa-api]") == ["auth-api", "mfa-api"]


def test_parse_apis_empty():
    assert _parse_apis("RECOMMANDED_APIS: []") == []


def test_parse_apis_single_m_tolerated():
    # la regex tolère 1 ou 2 M : RECOMMANDED / RECOMMANDED
    assert _parse_apis("RECOMMANDED_APIS: [foo-bar-api]") == ["foo-bar-api"]


def test_parse_apis_strips_quotes_and_stars():
    assert _parse_apis('RECOMMANDED_APIS: [`foo-bar-api`, "baz-api"]') == ["foo-bar-api", "baz-api"]


def test_guard_flags_empty():
    assert guard({"question": ""})["is_corrupted"] is True


def test_guard_passes_normal():
    out = guard({"question": "créer une commande"})
    assert out["is_corrupted"] is False
    assert out["retries"] == 0


# ── 2. Smoke test : invocation complète du graphe, mocks sans réseau ─────────

class _FakeLLMResponse:
    """Imite la réponse d'un ChatGoogleGenerativeAI (content + usage_metadata)."""
    def __init__(self, content: str):
        self.content = content
        self.usage_metadata = {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "output_token_details": {"reasoning": 12},
        }


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages):
        return _FakeLLMResponse(self._content)


@pytest.fixture
def patched_graph(monkeypatch, mock_candidates):
    """
    Construit un graphe dont le retrieve et le answer sont mockés :
      - retriever.search → renvoie mock_candidates (pas de Pinecone)
      - nodes._llm()     → renvoie un faux LLM (pas de Gemini)
    """
    import agentic4api.graph.nodes as nodes

    monkeypatch.setattr(nodes, "search", lambda q, top_k=None: mock_candidates)
    monkeypatch.setattr(
        nodes, "_llm",
        lambda: _FakeLLM("Voici l'API.\nRECOMMANDED_APIS: [order-api-v4]"),
    )

    from agentic4api.graph.build import build_graph
    return build_graph(use_memory=False)


def test_graph_invoke_normal_question(patched_graph):
    state = patched_graph.invoke({"question": "je veux créer une commande client"})
    # le format de sortie est correctement parsé
    assert state["final_apis"] == ["order-api-v4"]
    # les candidats et scores ont bien transité par le State
    assert state["scores"] == [0.91, 0.62]
    # les tokens ont été capturés et accumulés
    assert state["tokens_in"] == 120
    assert state["tokens_total"] == 150


def test_graph_invoke_corrupted_question_short_circuits(patched_graph):
    state = patched_graph.invoke({"question": ""})
    # question vide → court-circuit, pas d'appel LLM, liste vide
    assert state["final_apis"] == []
    assert state.get("is_corrupted") is True
