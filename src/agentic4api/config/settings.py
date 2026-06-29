"""
Configuration centralisée — UN seul endroit pour tous les hyperparamètres.

Pourquoi : ton principe d'isolation des variables. Pour comparer deux architectures
proprement, on change UN paramètre ici, on logge la config exacte du run, et on
n'a pas à chasser des constantes éparpillées dans le code.

Charge automatiquement le `.env` à l'import.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Routes Kong (remplacent GOOGLE_API_KEY) ---
    # Kong proxifie vers Gemini et expose une API OpenAI-compatible.
    kong_chat_url: str = ""      # .env — ex. https://host/ai-api/v1/chat/gemini
    kong_embed_url: str = ""     # .env — ex. https://host/ai-api/v1/embeddings
    kong_api_key: str = "dummy"  # .env — laisser "dummy" si Kong n'exige pas de clé
    kong_verify_ssl: bool = False  # False sur réseau interne avec certificat auto-signé

    # --- Autres secrets ---
    pinecone_api_key: str = ""   # .env
    pinecone_index: str = "gemb2-apiparsing-raw-v2s"
    # Host direct (optionnel mais recommandé) : évite un appel API de résolution.
    # Visible dans la console Pinecone → ton index → "Host".
    # Ex. https://agentic4api-xxxx.svc.aped-xxxx.pinecone.io
    pinecone_host: str = ""      # .env
    sheet_id: str = ""           # .env
    sheet_worksheet: str         # .env — nom de l'onglet cible, obligatoire, pas de défaut

    # --- Auth Google Sheets ---
    # "oauth2"          → perso/local : ouvre le navigateur une fois, sauvegarde token.json
    # "service_account" → entreprise  : service_account.json ou JSON brut dans GOOGLE_SA_JSON
    google_auth_mode: str = "oauth2"
    google_credentials_json: str = "./credentials.json"   # OAuth2 : téléchargé depuis GCP Console
    google_sa_json: str = "./service_account.json"        # .env en mode service_account (Cloud Run)

    # --- Modèles ---
    # Utilise le nom exact renvoyé par Kong dans "model" (visible dans la réponse JSON).
    gemini_model: str = "gemini-3.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"

    # --- Retrieval ---
    top_k: int = 20           # N8N utilisait 20 ; 5 donnait trop peu de candidats à Gemini
    # "rag"      : retrieve toujours appelé UNE fois avant le LLM (déterministe)
    # "agentic"  : le LLM décide lui-même quand/combien de fois appeler Pinecone (outil)
    retrieval_mode: str = "agentic"

    # --- Génération ---
    # thinking_budget / thinking_level supprimés : Kong (mode OpenAI-compat) ne les expose pas.
    # tokens_think sera toujours 0 dans le Sheet — c'est attendu.
    max_output_tokens: int = 4096
    temperature: float = 0.0

    # --- Agent ---
    max_retries: int = 5      # N8N utilisait MAX_RETRY = 5

    # --- Agent : mode d'appel d'outil ---
    # "text"       : le LLM écrit "SEARCH: <requête>" dans son texte (compatible Kong/thought_signature)
    # "bind_tools" : OpenAI structured tool calls — nécessite que Kong transmette la thought_signature
    tool_mode: str = "text"

    # --- Batch ---
    batch_wait_s: float = 2.0  # pause entre chaque question (rate limiting, identique au Wait N8N)

    # --- Debug ---
    debug_mode: bool = True   # DEBUG_MODE=true dans .env pour activer les prints de debug

    @property
    def kong_embed_base_url(self) -> str:
        """Base URL pour OpenAIEmbeddings : retire le segment /embeddings final.
        Ex. https://host/ai-api/v1/embeddings → https://host/ai-api/v1
        """
        url = self.kong_embed_url.rstrip("/")
        return url[: -len("/embeddings")] if url.endswith("/embeddings") else url


settings = Settings()
