"""
CLI : exécute le graphe sur le golden dataset, écrit le tout dans le Google Sheet
et sauvegarde un log JSONL complet dans logs/.

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

from agentic4api.batch.golden import load_golden
from agentic4api.batch.logger import log_path, serialize_messages, write_jsonl
from agentic4api.batch.sheet_writer import write_results
from agentic4api.config.settings import settings
from agentic4api.graph.build import build_graph


def _history_summary(messages: list) -> str:
    """10 premiers mots par message, un par ligne.

    Format :
        Human : Je veux envoyer une commande...
        AI→tool : search("commande client")
        Tool : - name: order-api-v4 | title:...
        AI : Voici les APIs recommandées pour...
    """
    lines = []
    for msg in messages or []:
        mtype = type(msg).__name__

        if mtype == "HumanMessage":
            words = " ".join((msg.content or "").split()[:10])
            lines.append(f"Human : {words}")

        elif mtype == "AIMessage":
            if getattr(msg, "tool_calls", None):
                query = msg.tool_calls[0]["args"].get("query", "")
                lines.append(f'AI->tool : search("{query}")')
            else:
                words = " ".join((msg.content or "").split()[:10])
                lines.append(f"AI : {words}")

        elif mtype == "ToolMessage":
            words = " ".join((msg.content or "").split()[:10])
            lines.append(f"Tool : {words}")

    return "\n".join(lines)


def _extract_result(state: dict) -> tuple[str, list[str], dict]:
    """Extrait tous les champs mesurables depuis AgentState."""
    text = state.get("answer_text", "")
    return text, state.get("final_apis", []), {
        "tokens_in":        state.get("tokens_in", 0),
        "tokens_out":       state.get("tokens_out", 0),
        "tokens_think":     state.get("tokens_think", 0),
        "tokens_total":     state.get("tokens_total", 0),
        "llm_call_count":   state.get("llm_call_count", 0),
        "tool_call_count":  state.get("tool_call_count", 0),
        "history_summary":  _history_summary(state.get("messages", [])),
        # Listes/dicts sérialisés en JSON pour tenir dans une cellule Sheet
        "final_apis":       json.dumps(state.get("final_apis", []),       ensure_ascii=False),
        "tool_call_inputs": json.dumps(state.get("tool_call_inputs", []), ensure_ascii=False),
        "retrieved_slugs":  json.dumps(state.get("retrieved_slugs", {}),  ensure_ascii=False),
    }


def _to_log_row(item: dict, state: dict, latency_s: float) -> dict:
    """Ligne complète pour le fichier JSONL (messages sérialisés en entier)."""
    return {
        "id":               item.get("id", ""),
        "question":         item["question"],
        "final_apis":       state.get("final_apis", []),
        "answer_text":      state.get("answer_text", ""),
        "latency_s":        latency_s,
        "llm_call_count":   state.get("llm_call_count", 0),
        "tool_call_count":  state.get("tool_call_count", 0),
        "tool_call_inputs": state.get("tool_call_inputs", []),
        "retrieved_slugs":  state.get("retrieved_slugs", {}),
        "tokens_in":        state.get("tokens_in", 0),
        "tokens_out":       state.get("tokens_out", 0),
        "tokens_think":     state.get("tokens_think", 0),
        "tokens_total":     state.get("tokens_total", 0),
        "messages":         serialize_messages(state.get("messages", [])),
    }


def _graph_input(question: str) -> dict:
    if settings.retrieval_mode == "rag":
        return {"question": question}
    return {"messages": [("human", question)]}


def _run_sequential(golden: list[dict], graph) -> tuple[list[dict], list[dict]]:
    """Une question à la fois — latence exacte mesurable par question."""
    rows, log_rows = [], []
    total = len(golden)

    for i, item in enumerate(golden, 1):
        q = item["question"]

        t0 = time.perf_counter()
        state = graph.invoke(_graph_input(q))
        latency_s = round(time.perf_counter() - t0, 3)

        answer_text, final_apis, metrics = _extract_result(state)
        rows.append({
            "id":        item.get("id", ""),
            "chatInput": q,
            "question":  q,
            "output":    answer_text,
            "latency_s": latency_s,
            **metrics,
        })
        log_rows.append(_to_log_row(item, state, latency_s))
        print(f"[{i}/{total}] {item.get('id','')} -> {final_apis}  ({latency_s}s)")

        if i < total:
            time.sleep(settings.batch_wait_s)

    return rows, log_rows


def _run_parallel(golden: list[dict], graph, batch_size: int) -> tuple[list[dict], list[dict]]:
    """Questions par chunks parallèles."""
    rows, log_rows = [], []
    chunks       = [golden[i : i + batch_size] for i in range(0, len(golden), batch_size)]
    total_chunks = len(chunks)

    for ci, chunk in enumerate(chunks, 1):
        inputs = [_graph_input(item["question"]) for item in chunk]

        t0 = time.perf_counter()
        states = graph.batch(inputs)
        chunk_latency = round(time.perf_counter() - t0, 3)

        for item, state in zip(chunk, states):
            answer_text, final_apis, metrics = _extract_result(state)
            rows.append({
                "id":        item.get("id", ""),
                "chatInput": item["question"],
                "question":  item["question"],
                "output":    answer_text,
                "latency_s": chunk_latency,
                **metrics,
            })
            log_rows.append(_to_log_row(item, state, chunk_latency))
            print(f"[chunk {ci}/{total_chunks}] {item.get('id','')} -> {final_apis}")

        print(f"  chunk {ci} termine en {chunk_latency}s ({len(chunk)} questions)")

        if ci < total_chunks:
            time.sleep(settings.batch_wait_s)

    return rows, log_rows


def run(
    limit: int | None = None,
    worksheet: str | None = None,
    parallel: bool = False,
    batch_size: int = 5,
) -> list[dict]:
    worksheet = worksheet or settings.sheet_worksheet
    golden    = load_golden()
    if limit:
        golden = golden[:limit]

    graph = build_graph(use_memory=False)

    mode_label = f"parallele (batch_size={batch_size})" if parallel else "sequentiel"
    print(f"Mode : {mode_label} | {len(golden)} questions | retrieval: {settings.retrieval_mode}")

    if parallel:
        rows, log_rows = _run_parallel(golden, graph, batch_size)
    else:
        rows, log_rows = _run_sequential(golden, graph)

    # Google Sheet
    write_results(rows, worksheet_name=worksheet)
    print(f"\n[OK] {len(rows)} lignes ecrites dans l'onglet '{worksheet}'.")

    # Log JSONL
    path = log_path()
    write_jsonl(log_rows, path)
    print(f"[OK] Log complet : {path}")

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Batch d'evaluation -> Google Sheet + log JSONL")
    p.add_argument("--limit",      type=int,  default=None,  help="N premieres questions (smoke test)")
    p.add_argument("--worksheet",  type=str,  default=None,  help="Nom de l'onglet cible")
    p.add_argument("--parallel",   action="store_true",      help="Execution parallele par chunks")
    p.add_argument("--batch-size", type=int,  default=5,     help="Taille des chunks en mode parallele (defaut: 5)")
    args = p.parse_args()
    run(
        limit=args.limit,
        worksheet=args.worksheet,
        parallel=args.parallel,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
