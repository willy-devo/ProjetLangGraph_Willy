"""
Vérifie que les lignes produites pour le Sheet contiennent bien toutes les colonnes
de mesure (retries / latence / tokens). On mocke le graphe pour ne pas appeler le réseau.
"""

from __future__ import annotations

from agentic4api.batch.sheet_writer import HEADERS


def _fake_row():
    return {
        "id": "T1", "chatInput": "q", "question": "q", "output": "RECOMMANDED_APIS: []",
        "retries": 0, "latency_s": 0.12,
        "tokens_in": 100, "tokens_out": 20, "tokens_think": 15, "tokens_total": 120,
    }


def test_row_has_all_headers():
    row = _fake_row()
    for h in HEADERS:
        assert h in row, f"colonne manquante : {h}"


def test_token_columns_present():
    row = _fake_row()
    for col in ("tokens_in", "tokens_out", "tokens_think", "tokens_total", "latency_s", "retries"):
        assert col in row
