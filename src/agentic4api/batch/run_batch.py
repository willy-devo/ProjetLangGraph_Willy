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
    agentic4api-batch                              # sequentiel, tout le golden
    agentic4api-batch --limit 5                    # smoke test 5 questions
    agentic4api-batch --parallel --batch-size 3    # 3 questions en parallele
    agentic4api-batch --worksheet run_gemini35flash
    agentic4api-batch --resume                     # reprend depuis le dernier JSONL (auto)
    agentic4api-batch --resume-from logs/resultats_2026_06_24_14h13.jsonl
    agentic4api-batch --skip 59                    # reprend a partir de la question 60

Robustesse :
    - Retry automatique sur 429 (rate limit) : backoff 60s / 120s / 300s / 600s
    - JSONL écrit après CHAQUE question → résultats partiels préservés si crash
    - Sheet mis à jour en temps réel après chaque question
    - --skip N      : reprend depuis la question N sans effacer le Sheet existant
    - --resume      : auto-detecte le skip depuis logs/resultats_*.jsonl (somme tous les fichiers)
    - --resume-from : reprend depuis un JSONL specifique (chemin relatif ou absolu)
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
from agentic4api.batch.logger import append_error_jsonl, append_jsonl, error_log_path, log_path, serialize_messages
from agentic4api.batch.sheet_writer import append_row, get_worksheet, init_sheet
from agentic4api.config.settings import settings
from agentic4api.graph.build import build_graph

_RATE_LIMIT_BACKOFFS = [60, 120, 300, 600]  # secondes d'attente entre les retries
_HISTORY_TRUNCATE_WORDS = 40               # mots max par message dans history_summary


def _count_jsonl_lines(path: Path) -> int:
    """Compte les lignes d'un fichier JSONL (= nombre de questions traitees)."""
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _detect_resume() -> tuple[int, list[Path]]:
    """Lit tous les JSONL resultats_*.jsonl dans logs/, deduplique par ID de question,
    et retourne (nb_questions_uniques_traitees, liste_fichiers).
    Retourne (0, []) si aucun fichier existe.

    Deduplique car un run complet (--skip 0) peut chevaucher un run partiel anterieur.
    """
    log_dir = Path("logs")
    if not log_dir.exists():
        return 0, []
    jsonl_files = sorted(log_dir.glob("resultats_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not jsonl_files:
        return 0, []
    seen_ids: set[str] = set()
    for p in jsonl_files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        qid = row.get("id", "")
                        if qid:
                            seen_ids.add(qid)
                    except json.JSONDecodeError:
                        pass
    return len(seen_ids), jsonl_files


def _history_summary(messages: list) -> str:
    """_HISTORY_TRUNCATE_WORDS premiers mots par message, un par ligne."""
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
                words = " ".join(content.split()[:_HISTORY_TRUNCATE_WORDS])
                lines.append(f"Human : {words}")

        elif mtype == "AIMessage":
            content = msg.content or ""
            m = re.search(r"SEARCH:\s*(.+)", content)
            if m:
                lines.append(f'AI->tool : search("{m.group(1).strip()}")')
            else:
                words = " ".join(content.split()[:_HISTORY_TRUNCATE_WORDS])
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
        "tokens_detail":    json.dumps(state.get("tokens_detail", {}),    ensure_ascii=False),
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
        "tokens_detail":    state.get("tokens_detail", {}),
        "messages":         serialize_messages(state.get("messages", [])),
    }


def _graph_input(question: str) -> dict:
    if settings.retrieval_mode == "rag":
        return {"question": question}
    return {"messages": [("human", question)]}


def _invoke_with_retry(
    graph,
    inp: dict,
    error_path: Path,
    question_id: str = "",
    question: str = "",
) -> tuple[dict, float]:
    """Appelle graph.invoke avec retry sur 429. Retourne (state, latency_s).
    Chaque tentative echouee est loggee dans error_path."""
    total_wait = 0
    for attempt, backoff in enumerate([0] + _RATE_LIMIT_BACKOFFS, start=1):
        if backoff:
            print(f"  [429] Rate limit. Attente {backoff}s (tentative {attempt}/{len(_RATE_LIMIT_BACKOFFS)+1})...")
            time.sleep(backoff)
            total_wait += backoff
        try:
            t0 = time.perf_counter()
            state = graph.invoke(inp)
            return state, round(time.perf_counter() - t0, 3)
        except openai.RateLimitError as e:
            fatal = attempt > len(_RATE_LIMIT_BACKOFFS)
            append_error_jsonl(
                error_path,
                question_id=question_id,
                question=question,
                attempt=attempt,
                backoff_s=backoff,
                error_type="RateLimitError",
                error_message=str(e),
                fatal=fatal,
                total_wait_s=total_wait,
            )
            if fatal:
                raise
    raise RuntimeError("Unreachable")


def _run_sequential(golden: list[dict], graph, ws: gspread.Worksheet, jsonl_path: Path, error_path: Path) -> list[dict]:
    """Une question à la fois. Sheet + JSONL mis à jour après chaque question."""
    rows  = []
    total = len(golden)

    for i, item in enumerate(golden, 1):
        q = item["question"]

        state, latency_s = _invoke_with_retry(
            graph, _graph_input(q), error_path,
            question_id=item.get("id", ""), question=q,
        )

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


def _run_parallel(golden: list[dict], graph, batch_size: int, ws: gspread.Worksheet, jsonl_path: Path, error_path: Path) -> list[dict]:
    """Questions par chunks parallèles. Sheet + JSONL mis à jour après chaque chunk."""
    rows         = []
    chunks       = [golden[i : i + batch_size] for i in range(0, len(golden), batch_size)]
    total_chunks = len(chunks)

    for ci, chunk in enumerate(chunks, 1):
        inputs     = [_graph_input(item["question"]) for item in chunk]
        chunk_ids  = ", ".join(item.get("id", "?") for item in chunk)
        total_wait = 0

        for attempt, backoff in enumerate([0] + _RATE_LIMIT_BACKOFFS, start=1):
            if backoff:
                print(f"  [429] Rate limit chunk {ci}. Attente {backoff}s...")
                time.sleep(backoff)
                total_wait += backoff
            try:
                t0 = time.perf_counter()
                states = graph.batch(inputs)
                chunk_latency = round(time.perf_counter() - t0, 3)
                break
            except openai.RateLimitError as e:
                fatal = attempt > len(_RATE_LIMIT_BACKOFFS)
                append_error_jsonl(
                    error_path,
                    question_id=chunk_ids,
                    question=f"chunk {ci}",
                    attempt=attempt,
                    backoff_s=backoff,
                    error_type="RateLimitError",
                    error_message=str(e),
                    fatal=fatal,
                    total_wait_s=total_wait,
                )
                if fatal:
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
    resume: bool = False,
    resume_from: str | None = None,
) -> list[dict]:
    worksheet = worksheet or settings.sheet_worksheet
    golden    = load_golden()
    if limit:
        golden = golden[:limit]

    if resume or resume_from:
        if resume_from:
            path = Path(resume_from)
            skip = _count_jsonl_lines(path)
            print(f"[REPRISE] Lu {path.name} -> {skip} questions deja traitees")
        else:
            skip, paths = _detect_resume()
            if paths:
                files_str = ", ".join(p.name for p in paths)
                print(f"[REPRISE] {len(paths)} fichier(s) JSONL : {files_str}")
                print(f"[REPRISE] Total deja traite : {skip} questions")
            else:
                print("[REPRISE] Aucun JSONL trouve dans logs/ -> run complet")

    if skip:
        golden = golden[skip:]
        print(f"[REPRISE] Depart a Q{skip+1:04d} ({len(golden)} questions restantes)")

    graph      = build_graph(use_memory=False)
    jsonl_path = log_path()
    error_path = error_log_path()
    ws         = get_worksheet(worksheet) if skip else init_sheet(worksheet, rows_estimate=len(golden) + skip)

    mode_label = f"parallele (batch_size={batch_size})" if parallel else "sequentiel"
    print(f"Mode : {mode_label} | {len(golden)} questions | retrieval: {settings.retrieval_mode}")
    resume_note = " (reprise, pas de reset)" if skip else ""
    print(f"Sheet: onglet '{worksheet}' - mise a jour en temps reel{resume_note}")
    print(f"Log  : {jsonl_path}")
    print(f"Errors: {error_path}")

    rows = []
    try:
        if parallel:
            rows = _run_parallel(golden, graph, batch_size, ws, jsonl_path, error_path)
        else:
            rows = _run_sequential(golden, graph, ws, jsonl_path, error_path)
    finally:
        print(f"\n[OK] {len(rows)} lignes dans '{worksheet}' | Log : {jsonl_path}")

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Batch d'evaluation -> Google Sheet + log JSONL")
    p.add_argument("--limit",      type=int,  default=None,  help="N premieres questions (smoke test)")
    p.add_argument("--worksheet",  type=str,  default=None,  help="Nom de l'onglet cible")
    p.add_argument("--parallel",   action="store_true",      help="Execution parallele par chunks")
    p.add_argument("--batch-size", type=int,  default=5,     help="Taille des chunks en mode parallele (defaut: 5)")
    p.add_argument("--skip",        type=int,  default=0,    help="Sauter les N premieres questions (reprise apres crash)")
    p.add_argument("--resume",      action="store_true",     help="Reprendre depuis le dernier JSONL (auto-detecte dans logs/)")
    p.add_argument("--resume-from", type=str,  default=None, help="Reprendre depuis un JSONL specifique (chemin)")
    args = p.parse_args()
    run(
        limit=args.limit,
        worksheet=args.worksheet,
        parallel=args.parallel,
        batch_size=args.batch_size,
        skip=args.skip,
        resume=args.resume,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
