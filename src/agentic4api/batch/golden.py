"""
Chargement du golden dataset depuis data/golden_dataset.json.

Le golden n'est PAS lu par l'agent : il sert d'entrée au batch (run_batch) et à
l'éval Colab. Chaque entrée a la forme :
    {"id", "question", "expected_apis", "category", "domain",
     "difficulty", "register", "phrasing"}
"""

from __future__ import annotations

import json
from pathlib import Path

# data/ est à la racine du projet.
# parents[3] : golden.py → batch → agentic4api → src → (racine du projet)
_DATA = Path(__file__).resolve().parents[3] / "data" / "golden_dataset.json"


def load_golden(path: str | Path | None = None) -> list[dict]:
    """
    Charge et renvoie la liste des questions du golden dataset.

    Lève FileNotFoundError avec un message explicite si le fichier est introuvable
    (cas courant : lancé depuis le mauvais dossier, ou `data/` non copié dans l'image
    Docker — vérifie le COPY data ./data du Dockerfile).
    """
    p = Path(path) if path else _DATA
    if not p.exists():
        raise FileNotFoundError(
            f"Golden dataset introuvable : {p}\n"
            f"Vérifie que data/golden_dataset.json existe et, en conteneur, "
            f"que le Dockerfile contient bien `COPY data ./data`."
        )

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Le golden doit être une LISTE de questions, reçu : {type(data).__name__}."
        )
    return data
