"""
Écriture des résultats dans le Google Sheet — UN seul write batch.

Deux modes d'authentification selon GOOGLE_AUTH_MODE dans le .env :

  "oauth2" (défaut — mode PERSO / local)
    → credentials.json  : fichier téléchargé depuis GCP Console (OAuth2 Desktop app)
    → token.json        : généré automatiquement au 1er lancement (navigateur s'ouvre une fois)
    → Utiliser pour : développement local, tests, Colab
    → Le Sheet doit être accessible par TON compte Google (pas besoin de partage spécial)

  "service_account" (mode ENTREPRISE / prod)
    → GOOGLE_SA_JSON = chemin vers service_account.json  (local)
    → GOOGLE_SA_JSON = contenu JSON brut                 (Cloud Run / Secret Manager)
    → Utiliser pour : Cloud Run, Docker, CI/CD
    → Le Sheet DOIT être partagé avec l'email du service account

Colonnes écrites :
  id, chatInput, question, output, retries,
  latency_s, tokens_in, tokens_out, tokens_total
"""

from __future__ import annotations

import json
import os

import gspread
from google.oauth2.service_account import Credentials

from agentic4api.config.settings import settings

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HEADERS = [
    "id", "chatInput", "question", "output", "final_apis",
    "latency_s",
    "llm_call_count", "tool_call_count", "tool_call_inputs",
    "tokens_in", "tokens_out", "tokens_think", "tokens_total",
    "retrieved_slugs",
]


def _client() -> gspread.Client:
    mode = settings.google_auth_mode.lower()

    if mode == "oauth2":
        # gspread gère tout : ouvre le navigateur si token absent, rafraîchit si expiré.
        return gspread.oauth(
            credentials_filename=settings.google_credentials_json,
            authorized_user_filename="token.json",
        )

    if mode == "service_account":
        sa_value = os.environ.get("GOOGLE_SA_JSON", settings.google_sa_json) or ""
        if sa_value.strip().startswith("{"):
            # Cloud Run : contenu JSON brut injecté via Secret Manager
            creds = Credentials.from_service_account_info(
                json.loads(sa_value), scopes=_SCOPES
            )
        else:
            # Local : chemin vers le fichier
            creds = Credentials.from_service_account_file(sa_value, scopes=_SCOPES)
        return gspread.authorize(creds)

    raise ValueError(
        f"GOOGLE_AUTH_MODE invalide : '{mode}'. Valeurs acceptées : 'oauth2', 'service_account'."
    )


def write_results(rows: list[dict], worksheet_name: str = "results") -> None:
    """
    `rows` : liste de dicts ayant (au moins) les clés de HEADERS.
    Crée l'onglet s'il n'existe pas, puis écrit headers + données en UN seul update.
    """
    gc = _client()
    sh = gc.open_by_key(settings.sheet_id)

    try:
        ws = sh.worksheet(worksheet_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=len(rows) + 10, cols=len(HEADERS))

    matrix = [HEADERS]
    for r in rows:
        matrix.append([r.get(h, "") for h in HEADERS])

    # Un seul appel réseau → pas de rate-limit.
    ws.update(matrix, value_input_option="RAW")
