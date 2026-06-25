"""
Écriture des résultats dans le Google Sheet.

Deux modes :
  write_results()     — écrit toutes les lignes en un seul appel (batch final)
  init_sheet()        — crée/vide l'onglet, écrit les headers, retourne le Worksheet
  append_row()        — ajoute UNE ligne en temps réel (appel après chaque question)

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
  id, question, output, final_apis,
  latency_s, llm_call_count, tool_call_count, tool_call_inputs,
  tokens_in, tokens_out, tokens_think, tokens_total,
  retrieved_slugs, history_summary
"""

from __future__ import annotations

import json
import os

import gspread
from google.oauth2.service_account import Credentials

from agentic4api.config.settings import settings

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HEADERS = [
    "id", "question", "output", "final_apis", "expected_apis",
    "latency_s", "tokens_in", "tokens_out", "tokens_think", "tokens_total", "tokens_detail",
    "llm_call_count", "tool_call_count", "tool_call_inputs",
    "retrieved_slugs", "history_summary",
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


def init_sheet(worksheet_name: str, rows_estimate: int = 500) -> gspread.Worksheet:
    """Crée ou vide l'onglet, écrit les headers. Retourne le Worksheet pour les appends suivants."""
    gc = _client()
    sh = gc.open_by_key(settings.sheet_id)

    try:
        ws = sh.worksheet(worksheet_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=rows_estimate + 10, cols=len(HEADERS))

    ws.update([HEADERS], value_input_option="RAW")
    return ws


def get_worksheet(worksheet_name: str) -> gspread.Worksheet:
    """Récupère le worksheet sans le vider — pour reprise après crash."""
    gc = _client()
    sh = gc.open_by_key(settings.sheet_id)
    try:
        return sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=510, cols=len(HEADERS))
        ws.update([HEADERS], value_input_option="RAW")
        return ws


def append_row(ws: gspread.Worksheet, row: dict) -> None:
    """Ajoute une ligne en temps réel — un appel réseau par question."""
    ws.append_row([row.get(h, "") for h in HEADERS], value_input_option="RAW")


def write_results(rows: list[dict], worksheet_name: str = "results") -> None:
    """Écrit toutes les lignes en un seul appel batch (mode non-temps-réel)."""
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

    ws.update(matrix, value_input_option="RAW")
