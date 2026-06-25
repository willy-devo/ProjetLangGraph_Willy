"""
Logger JSONL pour les runs batch.

Deux fichiers par run, dans logs/ :
  resultats_YYYY_MM_DD_HHhMM.jsonl  — une ligne par question reussie
  errors_YYYY_MM_DD_HHhMM.jsonl     — une ligne par erreur (429, crash, etc.)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def log_path() -> Path:
    """logs/resultats_2026_06_24_14h30.jsonl"""
    now  = datetime.now()
    name = now.strftime("resultats_%Y_%m_%d_%Hh%M.jsonl")
    path = Path("logs")
    path.mkdir(exist_ok=True)
    return path / name


def error_log_path() -> Path:
    """logs/errors_2026_06_24_14h30.jsonl — meme horodatage que log_path()."""
    now  = datetime.now()
    name = now.strftime("errors_%Y_%m_%d_%Hh%M.jsonl")
    path = Path("logs")
    path.mkdir(exist_ok=True)
    return path / name


def append_error_jsonl(
    error_path: Path,
    *,
    question_id: str,
    question: str,
    attempt: int,
    backoff_s: int,
    error_type: str,
    error_message: str,
    fatal: bool,
    total_wait_s: int = 0,
) -> None:
    """Ajoute une ligne d'erreur au fichier errors_*.jsonl."""
    row = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "question_id":  question_id,
        "question":     question,
        "attempt":      attempt,
        "backoff_s":    backoff_s,
        "error_type":   error_type,
        "error_message": error_message,
        "fatal":        fatal,
        "total_wait_s": total_wait_s,
    }
    with open(error_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def serialize_messages(messages: list) -> list[dict]:
    """Convertit les objets LangChain en dicts JSON-sérialisables."""
    out = []
    for msg in messages or []:
        d: dict = {"type": type(msg).__name__, "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            d["tool_calls"] = msg.tool_calls
        out.append(d)
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def append_jsonl(row: dict, path: Path) -> None:
    """Ajoute une ligne au fichier JSONL existant (mode append)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
