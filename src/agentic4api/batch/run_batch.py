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

Robustesse :
    - Retry automatique sur 429 (rate limit) : backoff 60s / 120s / 300s / 600s
    - JSONL écrit après CHAQUE question → résultats partiels préservés si crash
    - Sheet mis à jour en temps réel après chaque question
    - --skip N : reprend depuis la question N sans effacer le Sheet existant
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import openai

import gspread

from agentic4api.batch.golden import load_golden
from agentic4api.batch.logger import append_jsonl, log_path, serialize_messages
from agentic4api.batch.sheet_writer import append_row, get_worksheet, init_sheet
from agentic4api.config.settings import settings
from agentic4api.graph.build import build_graph

_RATE_LIMIT_BACKOFFS = [60, 120, 300, 600]  # secondes d'attente entre les retries


def _history_summary(messages: list) -> str:
    """10 premiers mots par message, un par ligne."""
    lines = []
    for msg in messages or []:
        mtype = type(msg).__name__

        if mtype == "HumanMessage":
            content = msg.content or ""
            if content.startswith("[Résultats Pinecone"):
                end = content.find("]")
                words = content[1:end] if end > 0 else content[:40]
                lines.append(f"Tool : {words}")
            else:
                words = " ".join(content.split()[:10])
                lines.append(f"Human : {words}")

        elif mtype == "AIMessage":
            content = msg.content or ""
            m = re.search(r"SEARCH:\s*(.+)", content)
            if m:
                lines.append(f'AI->tool : search("{m.group(1).strip()}")')
            else:
                words = " ".join(content.split()[:10])
                lines.append(f"AI : {words}")

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


def _invoke_with_retry(graph, inp: dict) -> tuple[dict, float]:
    """Appelle graph.invoke avec retry sur 429. Retourne (state, latency_s)."""
    for attempt, backoff in enumerate([0] + _RATE_LIMIT_BACKOFFS, start=1):
        if backoff:
            print(f"  [429] Rate limit. Attente {backoff}s (tentative {attempt}/{len(_RATE_LIMIT_BACKOFFS)+1})...")
            time.sleep(backoff)
        try:
            t0 = time.perf_counter()
            state = graph.invoke(inp)
            return state, round(time.perf_counter() - t0, 3)
        except openai.RateLimitError:
            if attempt > len(_RATE_LIMIT_BACKOFFS):
                raise
    raise RuntimeError("Unreachable")


def _run_sequential(golden: list[dict], graph, ws: gspread.Worksheet, jsonl_path: Path) -> list[dict]:
    """Une question à la fois. Sheet + JSONL mis à jour après chaque question."""
    rows  = []
    total = len(golden)

    for i, item in enumerate(golden, 1):
        q = item["question"]

        state, latency_s = _invoke_with_retry(graph, _graph_input(q))

        answer_text, final_apis, metrics = _extract_result(state)
        row = {
            "id":            item.get("id", ""),
            "question":      q,
            "output":        answer_text,
            "latency_s":     latency_s,
            "expected_apis": json.dumps(item.get("expected_apis", []), ensure_ascii=False),
            **metrics,
        }
        rows.append(row)
        append_row(ws, row)
        append_jsonl(_to_log_row(item, state, latency_s), jsonl_path)
        print(f"[{i}/{total}] {item.get('id','')} -> {final_apis}  ({latency_s}s)")

        if i < total:
            time.sleep(settings.batch_wait_s)

    return rows


def _run_parallel(golden: list[dict], graph, batch_size: int, ws: gspread.Worksheet, jsonl_path: Path) -> list[dict]:
    """Questions par chunks parallèles. Sheet + JSONL mis à jour après chaque chunk."""
    rows         = []
    chunks       = [golden[i : i + batch_size] for i in range(0, len(golden), batch_size)]
    total_chunks = len(chunks)

    for ci, chunk in enumerate(chunks, 1):
        inputs = [_graph_input(item["question"]) for item in chunk]

        for attempt, backoff in enumerate([0] + _RATE_LIMIT_BACKOFFS, start=1):
            if backoff:
                print(f"  [429] Rate limit chunk {ci}. Attente {backoff}s...")
                time.sleep(backoff)
            try:
                t0 = time.perf_counter()
                states = graph.batch(inputs)
                chunk_latency = round(time.perf_counter() - t0, 3)
                break
            except openai.RateLimitError:
                if attempt > len(_RATE_LIMIT_BACKOFFS):
                    raise

        for item, state in zip(chunk, states):
            answer_text, final_apis, metrics = _extract_result(state)
            row = {
                "id":            item.get("id", ""),
                "question":      item["question"],
                "output":        answer_text,
                "latency_s":     chunk_latency,
                "expected_apis": json.dumps(item.get("expected_apis", []), ensure_ascii=False),
                **metrics,
            }
            rows.append(row)
            append_row(ws, row)
            append_jsonl(_to_log_row(item, state, chunk_latency), jsonl_path)
            print(f"[chunk {ci}/{total_chunks}] {item.get('id','')} -> {final_apis}")

        print(f"  chunk {ci} termine en {chunk_latency}s ({len(chunk)} questions)")

        if ci < total_chunks:
            time.sleep(settings.batch_wait_s)

    return rows


def run(
    limit: int | None = None,
    worksheet: str | None = None,
    parallel: bool = False,
    batch_size: int = 5,
    skip: int = 0,
) -> list[dict]:
    worksheet = worksheet or settings.sheet_worksheet
    golden    = load_golden()
    if limit:
        golden = golden[:limit]

    if skip:
        golden = golden[skip:]
        print(f"[REPRISE] Saut des {skip} premieres questions -> depart a Q{skip+1:04d}")

    graph      = build_graph(use_memory=False)
    jsonl_path = log_path()
    ws         = get_worksheet(worksheet) if skip else init_sheet(worksheet, rows_estimate=len(golden) + skip)

    mode_label = f"parallele (batch_size={batch_size})" if parallel else "sequentiel"
    print(f"Mode : {mode_label} | {len(golden)} questions | retrieval: {settings.retrieval_mode}")
    resume_note = " (reprise, pas de reset)" if skip else ""
    print(f"Sheet: onglet '{worksheet}' - mise a jour en temps reel{resume_note}")
    print(f"Log  : {jsonl_path}")

    rows = []
    try:
        if parallel:
            rows = _run_parallel(golden, graph, batch_size, ws, jsonl_path)
        else:
            rows = _run_sequential(golden, graph, ws, jsonl_path)
    finally:
        print(f"\n[OK] {len(rows)} lignes dans '{worksheet}' | Log : {jsonl_path}")

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Batch d'evaluation -> Google Sheet + log JSONL")
    p.add_argument("--limit",      type=int,  default=None,  help="N premieres questions (smoke test)")
    p.add_argument("--worksheet",  type=str,  default=None,  help="Nom de l'onglet cible")
    p.add_argument("--parallel",   action="store_true",      help="Execution parallele par chunks")
    p.add_argument("--batch-size", type=int,  default=5,     help="Taille des chunks en mode parallele (defaut: 5)")
    p.add_argument("--skip",       type=int,  default=0,     help="Sauter les N premieres questions (reprise apres crash)")
    args = p.parse_args()
    run(
        limit=args.limit,
        worksheet=args.worksheet,
        parallel=args.parallel,
        batch_size=args.batch_size,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
