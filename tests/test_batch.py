"""
Vérifie que les lignes produites pour le Sheet contiennent bien toutes les colonnes
de mesure. On mocke le graphe pour ne pas appeler le réseau.
"""

from __future__ import annotations

from agentic4api.batch.sheet_writer import HEADERS


def _fake_row():
    return {
        "id": "T1",
        "question": "q",
        "output": "RECOMMANDED_APIS: []",
        "final_apis": "[]",
        "expected_apis": '["order-api-v4"]',
        "latency_s": 0.12,
        "tokens_in": 100,
        "tokens_out": 20,
        "tokens_think": 15,
        "tokens_total": 120,
        "tokens_detail": '{"tokens_in": [100], "tokens_out": [20], "tokens_think": [15]}',
        "llm_call_count": 1,
        "tool_call_count": 2,
        "tool_call_inputs": "[]",
        "nb_embedded_tokens": 0,
        "retrieved_slugs": "{}",
        "history_summary": "",
    }


def test_row_has_all_headers():
    row = _fake_row()
    for h in HEADERS:
        assert h in row, f"colonne manquante : {h}"


def test_token_columns_present():
    row = _fake_row()
    for col in ("tokens_in", "tokens_out", "tokens_think", "tokens_total", "latency_s",
                "llm_call_count", "tool_call_count"):
        assert col in row
