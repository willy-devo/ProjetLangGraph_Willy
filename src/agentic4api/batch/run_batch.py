"""
CLI : exécute le graphe sur le golden dataset, mesure latence/tokens PAR question,
écrit le tout dans le Google Sheet. Remplace le webhook n8n.

Important — pourquoi invoke séquentiel et pas graph.batch() :
  La latence par question doit être propre. graph.batch() parallélise les exécutions,
  qui se chevauchent → impossible de chronométrer une question isolément depuis
  l'extérieur. On boucle donc en invoke séquentiel (plus lent, mais mesure exacte).
  Si la latence ne t'intéressait pas, graph.batch() serait plus rapide.

Lancer :
    agentic4api-batch                 # tout le golden
    agentic4api-batch --limit 5       # smoke test sur 5 questions
    agentic4api-batch --worksheet run_gemini35flash
"""

from __future__ import annotations

import argparse
import time

from agentic4api.batch.golden import load_golden
from agentic4api.batch.sheet_writer import write_results
from agentic4api.config.settings import settings
from agentic4api.graph.build import build_graph
from agentic4api.graph.nodes import _parse_apis


def _extract_result(state: dict) -> tuple[str, list[str], dict]:
    """Normalise la sortie du graphe selon le mode (agentic ou rag).

    Retourne (answer_text, final_apis, tokens_dict).
    Mode agentic (défaut) : sortie dans state["messages"] (format create_react_agent).
    Mode rag (optionnel)  : sortie dans state["answer_text"] / state["final_apis"].
    """
    if settings.retrieval_mode == "rag":
        text = state.get("answer_text", "")
        return text, state.get("final_apis", []), {
            "tokens_in":    state.get("tokens_in", 0),
            "tokens_out":   state.get("tokens_out", 0),
            "tokens_think": state.get("tokens_think", 0),
            "tokens_total": state.get("tokens_total", 0),
        }

    # Mode agentic — create_react_agent retourne les messages
    messages = state.get("messages", [])
    text = messages[-1].content if messages else ""
    if not isinstance(text, str):
        text = str(text)
    return text, _parse_apis(text), {"tokens_in": 0, "tokens_out": 0, "tokens_think": 0, "tokens_total": 0}


def _graph_input(question: str) -> dict:
    """Adapte l'entrée selon le mode (agentic par défaut, rag optionnel)."""
    if settings.retrieval_mode == "rag":
        return {"question": question}
    return {"messages": [("human", question)]}


def run(limit: int | None = None, worksheet: str | None = None) -> list[dict]:
    worksheet = worksheet or settings.sheet_worksheet
    golden = load_golden()
    if limit:
        golden = golden[:limit]

    # Graphe sans mémoire : chaque question = exécution indépendante (pas de MemorySaver).
    graph = build_graph(use_memory=False)

    rows: list[dict] = []
    for i, item in enumerate(golden, 1):
        q = item["question"]

        t0 = time.perf_counter()  # monotone : robuste aux ajustements d'horloge
        state = graph.invoke(_graph_input(q))
        latency_s = round(time.perf_counter() - t0, 3)

        answer_text, final_apis, tokens = _extract_result(state)

        rows.append({
            "id": item.get("id", ""),
            "chatInput": q,
            "question": q,
            "output": answer_text,
            "retries": state.get("retries", 0),
            "latency_s": latency_s,
            **tokens,
        })
        print(f"[{i}/{len(golden)}] {item.get('id','')} "
              f"-> {final_apis}  ({latency_s}s)")

        # Rate limiting : pause entre chaque question (identique au Wait N8N, configurable via BATCH_WAIT_S)
        if i < len(golden):
            time.sleep(settings.batch_wait_s)

    write_results(rows, worksheet_name=worksheet)
    print(f"\n[OK] {len(rows)} lignes ecrites dans l'onglet '{worksheet}'.")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Batch d'evaluation -> Google Sheet")
    p.add_argument("--limit", type=int, default=None, help="N premieres questions (smoke test)")
    p.add_argument("--worksheet", type=str, default=None, help="Nom de l'onglet cible (defaut : SHEET_WORKSHEET dans .env)")
    args = p.parse_args()
    run(limit=args.limit, worksheet=args.worksheet)


if __name__ == "__main__":
    main()
