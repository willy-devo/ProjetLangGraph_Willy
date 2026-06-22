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
    agentic4api-batch --worksheet run_gemini25flash
"""

from __future__ import annotations

import argparse
import time

from agentic4api.batch.golden import load_golden
from agentic4api.batch.sheet_writer import write_results
from agentic4api.config.settings import settings
from agentic4api.graph.build import build_graph


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
        state = graph.invoke({"question": q})
        latency_s = round(time.perf_counter() - t0, 3)

        rows.append({
            "id": item.get("id", ""),
            "chatInput": q,
            "question": q,
            "output": state.get("answer_text", ""),
            "retries": state.get("retries", 0),
            "latency_s": latency_s,
            "tokens_in": state.get("tokens_in", 0),
            "tokens_out": state.get("tokens_out", 0),
            "tokens_total": state.get("tokens_total", 0),
        })
        print(f"[{i}/{len(golden)}] {item.get('id','')} "
              f"-> {state.get('final_apis', [])}  ({latency_s}s)")

    write_results(rows, worksheet_name=worksheet)
    print(f"\n[OK] {len(rows)} lignes ecrites dans l'onglet '{worksheet}'.")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Batch d'évaluation → Google Sheet")
    p.add_argument("--limit", type=int, default=None, help="N premières questions (smoke test)")
    p.add_argument("--worksheet", type=str, default=None, help="Nom de l'onglet cible (défaut : SHEET_WORKSHEET dans .env)")
    args = p.parse_args()
    run(limit=args.limit, worksheet=args.worksheet)


if __name__ == "__main__":
    main()
