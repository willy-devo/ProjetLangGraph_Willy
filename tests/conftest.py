"""Fixtures de test : golden échantillon + retriever mocké (pas d'appel réseau)."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_golden():
    return [
        {"id": "T1", "question": "je veux créer une commande client",
         "expected_apis": ["order-api-v4"], "category": "simple"},
        {"id": "T2", "question": "", "expected_apis": [], "category": "simple"},
    ]


@pytest.fixture
def mock_candidates():
    return [
        {"slug": "order-api-v4", "score": 0.91, "text": "Gestion des commandes."},
        {"slug": "cart-api", "score": 0.62, "text": "Gestion du panier."},
    ]
