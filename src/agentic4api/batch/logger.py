"""
Logger JSONL pour les runs batch.

Écrit un fichier logs/resultats_YYYY_MM_DD_HHhMM.jsonl où chaque ligne
est une question complète avec son historique de messages sérialisé.
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
