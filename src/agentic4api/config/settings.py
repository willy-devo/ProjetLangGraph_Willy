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
    kong_chat_url: str = ""      # ex. https://host/ai-api/v1/chat/gemini
    kong_embed_url: str = ""     # ex. https://host/ai-api/v1/embeddings
    kong_api_key: str = "dummy"  # laisser "dummy" si Kong n'exige pas de clé
    kong_verify_ssl: bool = False  # False sur réseau interne avec certificat auto-signé

    # --- Autres secrets ---
    pinecone_api_key: str = ""
    pinecone_index: str = "agentic4api"
    # Host direct (optionnel mais recommandé) : évite un appel API de résolution.
    # Visible dans la console Pinecone → ton index → "Host".
    # Ex. https://agentic4api-xxxx.svc.aped-xxxx.pinecone.io
    pinecone_host: str = ""
    sheet_id: str = ""
    sheet_worksheet: str              # nom de l'onglet cible — obligatoire, pas de défaut

    # --- Auth Google Sheets ---
    # "oauth2"          → perso/local : ouvre le navigateur une fois, sauvegarde token.json
    # "service_account" → entreprise  : service_account.json ou JSON brut dans GOOGLE_SA_JSON
    google_auth_mode: str = "oauth2"
    google_credentials_json: str = "./credentials.json"   # OAuth2 : téléchargé depuis GCP Console
    google_sa_json: str = "./service_account.json"        # Service account : clé privée du robot

    # --- Modèles ---
    # Utilise le nom exact renvoyé par Kong dans "model" (visible dans la réponse JSON).
    gemini_model: str = "gemini-3.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"

    # --- Retrieval ---
    top_k: int = 5

    # --- Génération ---
    # thinking_budget / thinking_level supprimés : Kong (mode OpenAI-compat) ne les expose pas.
    # tokens_think sera toujours 0 dans le Sheet — c'est attendu.
    max_output_tokens: int = 4096
    temperature: float = 0.0

    # --- Agent ---
    max_retries: int = 2

    @property
    def kong_embed_base_url(self) -> str:
        """Base URL pour OpenAIEmbeddings : retire le segment /embeddings final.
        Ex. https://host/ai-api/v1/embeddings → https://host/ai-api/v1
        """
        url = self.kong_embed_url.rstrip("/")
        return url[: -len("/embeddings")] if url.endswith("/embeddings") else url


settings = Settings()
