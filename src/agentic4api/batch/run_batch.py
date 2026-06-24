"""
CLI : exécute le graphe sur le golden dataset, écrit le tout dans le Google Sheet.

Deux modes d'exécution :

  séquentiel (défaut)
    Une question à la fois → latence exacte par question mesurable.
    Lancer : agentic4api-batch

  parallèle  (--parallel)
    Les questions d'un même chunk tournent en parallèle via graph.batch().
    Plus rapide, mais latency_s = durée du chunk entier (pas par question isolée).
    Lancer : agentic4api-batch --parallel --batch-size 3

Exemples :
    agentic4api-batch                              # séquentiel, tout le golden
    agentic4api-batch --limit 5                    # smoke test 5 questions
    agentic4api-batch --parallel --batch-size 3    # 3 questions en parallèle
    agentic4api-batch --worksheet run_gemini35flash
"""

from __future__ import annotations

import argparse
import json
import time
from math import ceil

from agentic4api.batch.golden import load_golden
from agentic4api.batch.sheet_writer import write_results
from agentic4api.config.settings import settings
from agentic4api.graph.build import build_graph


def _extract_result(state: dict) -> tuple[str, list[str], dict]:
    """Extrait tous les champs mesurables depuis AgentState."""
    text = state.get("answer_text", "")
    return text, state.get("final_apis", []), {
        "tokens_in":       state.get("tokens_in", 0),
        "tokens_out":      state.get("tokens_out", 0),
        "tokens_think":    state.get("tokens_think", 0),
        "tokens_total":    state.get("tokens_total", 0),
        "llm_call_count":  state.get("llm_call_count", 0),
        "tool_call_count": state.get("tool_call_count", 0),
        # Listes sérialisées en JSON pour tenir dans une cellule Sheet
        "final_apis":       json.dumps(state.get("final_apis", []),        ensure_ascii=False),
        "tool_call_inputs": json.dumps(state.get("tool_call_inputs", []), ensure_ascii=False),
        "retrieved_slugs":  json.dumps(state.get("retrieved_slugs", {}),  ensure_ascii=False),
    }


def _graph_input(question: str) -> dict:
    """Adapte l'entrée selon le mode."""
    if settings.retrieval_mode == "rag":
        return {"question": question}
    return {"messages": [("human", question)]}


def _run_sequential(golden: list[dict], graph) -> list[dict]:
    """Une question à la fois — latence exacte mesurable par question."""
    rows: list[dict] = []
    total = len(golden)

    for i, item in enumerate(golden, 1):
        q = item["question"]

        t0 = time.perf_counter()
        state = graph.invoke(_graph_input(q))
        latency_s = round(time.perf_counter() - t0, 3)

        answer_text, final_apis, tokens = _extract_result(state)
        rows.append({
            "id":        item.get("id", ""),
            "chatInput": q,
            "question":  q,
            "output":    answer_text,
            "latency_s": latency_s,
            **tokens,
        })
        print(f"[{i}/{total}] {item.get('id','')} -> {final_apis}  ({latency_s}s)")

        if i < total:
            time.sleep(settings.batch_wait_s)

    return rows


def _run_parallel(golden: list[dict], graph, batch_size: int) -> list[dict]:
    """Questions par chunks parallèles — latency_s = durée du chunk, pas par question."""
    rows: list[dict] = []
    chunks = [golden[i : i + batch_size] for i in range(0, len(golden), batch_size)]
    total_chunks = len(chunks)

    for ci, chunk in enumerate(chunks, 1):
        inputs = [_graph_input(item["question"]) for item in chunk]

        t0 = time.perf_counter()
        # graph.batch() exécute les inputs en parallèle (threads LangGraph)
        states = graph.batch(inputs)
        chunk_latency = round(time.perf_counter() - t0, 3)

        for item, state in zip(chunk, states):
            answer_text, final_apis, tokens = _extract_result(state)
            rows.append({
                "id":        item.get("id", ""),
                "chatInput": item["question"],
                "question":  item["question"],
                "output":    answer_text,
                "latency_s": chunk_latency,   # durée du chunk entier
                **tokens,
            })
            print(f"[chunk {ci}/{total_chunks}] {item.get('id','')} -> {final_apis}")

        print(f"  chunk {ci} terminé en {chunk_latency}s ({len(chunk)} questions)")

        if ci < total_chunks:
            time.sleep(settings.batch_wait_s)

    return rows


def run(
    limit: int | None = None,
    worksheet: str | None = None,
    parallel: bool = False,
    batch_size: int = 5,
) -> list[dict]:
    worksheet = worksheet or settings.sheet_worksheet
    golden = load_golden()
    if limit:
        golden = golden[:limit]

    graph = build_graph(use_memory=False)

    mode_label = f"parallèle (batch_size={batch_size})" if parallel else "séquentiel"
    print(f"Mode : {mode_label} | {len(golden)} questions | retrieval: {settings.retrieval_mode}")

    if parallel:
        rows = _run_parallel(golden, graph, batch_size)
    else:
        rows = _run_sequential(golden, graph)

    write_results(rows, worksheet_name=worksheet)
    print(f"\n[OK] {len(rows)} lignes ecrites dans l'onglet '{worksheet}'.")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Batch d'evaluation -> Google Sheet")
    p.add_argument("--limit",      type=int,  default=None,  help="N premieres questions (smoke test)")
    p.add_argument("--worksheet",  type=str,  default=None,  help="Nom de l'onglet cible")
    p.add_argument("--parallel",   action="store_true",      help="Exécution parallèle par chunks")
    p.add_argument("--batch-size", type=int,  default=5,     help="Taille des chunks en mode parallèle (défaut: 5)")
    args = p.parse_args()
    run(
        limit=args.limit,
        worksheet=args.worksheet,
        parallel=args.parallel,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
