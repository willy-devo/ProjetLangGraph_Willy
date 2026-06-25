# agentic4api

Agent LangGraph de découverte sémantique d'APIs internes (Pinecone + Gemini via Kong).
Partagé entre un **chat** (Chainlit) et un **batch d'évaluation** qui écrit dans Google Sheet.

---

## Installation

```powershell
# 1. Créer et activer l'environnement conda
conda create -n agentic4api python=3.11 -y
conda activate agentic4api

# 2. Installer le projet et ses dépendances
pip install -e ".[dev]"

# 3. Copier et remplir le fichier de config
cp .env.example .env
```

---

## Lancer le batch d'évaluation

```powershell
# Run complet (464 questions)
python -m agentic4api.batch.run_batch

# Smoke test (5 questions, vérifie que tout fonctionne)
python -m agentic4api.batch.run_batch --limit 5

# Reprendre automatiquement après un crash (lit logs/*.jsonl, déduplique par ID)
python -m agentic4api.batch.run_batch --resume

# Reprendre depuis un JSONL précis
python -m agentic4api.batch.run_batch --resume-from logs\resultats_2026_06_24_14h13.jsonl

# Sauter les N premières questions manuellement
python -m agentic4api.batch.run_batch --skip 59

# Écrire dans un onglet Sheet spécifique
python -m agentic4api.batch.run_batch --worksheet run_v2

# Mode parallèle (chunks de 3 questions simultanées)
python -m agentic4api.batch.run_batch --parallel --batch-size 3
```

### Arguments CLI

| Argument | Type | Défaut | Description |
|---|---|---|---|
| `--limit N` | int | — | Ne traiter que les N premières questions |
| `--worksheet NOM` | str | `SHEET_WORKSHEET` (.env) | Onglet Google Sheet cible |
| `--skip N` | int | 0 | Sauter les N premières questions |
| `--resume` | flag | — | Auto-détecte le skip depuis `logs/resultats_*.jsonl` |
| `--resume-from CHEMIN` | str | — | Reprendre depuis un JSONL spécifique |
| `--parallel` | flag | — | Mode parallèle via `graph.batch()` |
| `--batch-size N` | int | 5 | Taille des chunks en mode parallèle |

---

## Lancer le chat

```powershell
chainlit run src/agentic4api/chat/app.py -w
```

---

## Lancer les tests

```powershell
# Tous les tests (sans réseau, ~2s)
python -m pytest

# Juste les tests du batch
python -m pytest tests/test_batch.py -v
```

---

## Configuration

### Variables `.env`

| Variable | Obligatoire | Description |
|---|---|---|
| `KONG_CHAT_URL` | ✅ | URL Kong → endpoint chat Gemini (OpenAI-compat) |
| `KONG_EMBED_URL` | ✅ | URL Kong → endpoint embeddings |
| `KONG_API_KEY` | — | Clé Kong (`dummy` si pas d'auth) |
| `PINECONE_API_KEY` | ✅ | Clé API Pinecone |
| `PINECONE_INDEX` | ✅ | Nom de l'index Pinecone |
| `PINECONE_HOST` | ✅ | Host direct Pinecone (évite un appel de résolution) |
| `SHEET_ID` | ✅ | ID du Google Sheet (dans l'URL) |
| `SHEET_WORKSHEET` | ✅ | Nom de l'onglet cible par défaut |
| `GOOGLE_AUTH_MODE` | — | `oauth2` (local) ou `service_account` (prod) |
| `GOOGLE_CREDENTIALS_JSON` | mode oauth2 | Chemin vers `credentials.json` |
| `GOOGLE_SA_JSON` | mode service_account | Chemin ou JSON brut du service account |
| `GEMINI_MODEL` | — | Nom du modèle tel que renvoyé par Kong |
| `GEMINI_EMBED_MODEL` | — | Modèle d'embedding |
| `TOP_K` | — | Nombre de résultats Pinecone (défaut: 20) |
| `MAX_OUTPUT_TOKENS` | — | Limite tokens en sortie (défaut: 4096) |
| `MAX_RETRIES` | — | Tentatives max par question (défaut: 5) |
| `RETRIEVAL_MODE` | — | `agentic` (LLM décide) ou `rag` (1 appel fixe) |
| `TOOL_MODE` | — | `text` (SEARCH: dans le texte) ou `bind_tools` |
| `BATCH_WAIT_S` | — | Pause entre questions en secondes (défaut: 2.0) |

> **Auth Google Sheets :**
> - `oauth2` : ouvre le navigateur une fois, sauvegarde `token.json`. Pour le développement local.
> - `service_account` : utilise un fichier JSON de compte de service. Pour Cloud Run / CI.

### Hyperparamètres clés

| Paramètre | Impact |
|---|---|
| `BATCH_WAIT_S` | Contrôle le débit → évite les 429. Valeur conseillée : `15.0` (~15 000 TPM) |
| `TOP_K` | Nombre d'APIs candidates envoyées au LLM. `20` = ~2 000 tokens contexte Pinecone |
| `RETRIEVAL_MODE` | `agentic` = fidèle à N8N, plus de tokens. `rag` = déterministe, rapide |
| `MAX_RETRIES` | Retry sur 429 avec backoff : 60s / 120s / 300s / 600s |

---

## Arborescence

```
agentic4api/
├── pyproject.toml              # Dépendances + entrée CLI agentic4api-batch
├── .env                        # Secrets (jamais commité)
├── .env.example                # Template à copier
├── Dockerfile / Dockerfile.batch
│
├── data/
│   ├── golden_dataset.json     # 464 questions (id, question, expected_apis, category…)
│   └── api-catalogue-500/      # ~500 fichiers JSON — une API par fichier
│
├── logs/                       # Créé automatiquement au premier run
│   ├── resultats_YYYY_MM_DD_HHhMM.jsonl   # Une ligne par question réussie
│   └── errors_YYYY_MM_DD_HHhMM.jsonl      # Une ligne par erreur 429 / crash
│
├── scripts/
│   └── index_pinecone.py       # (Ré)indexation du catalogue → Pinecone
│
├── src/agentic4api/
│   ├── config/
│   │   └── settings.py         # Charge .env — point unique pour tous les hyperparamètres
│   │
│   ├── graph/                  # Cœur : orchestration LangGraph (partagée chat + batch)
│   │   ├── state.py            # AgentState (TypedDict) + reducers (tokens, slugs…)
│   │   ├── nodes.py            # Nœuds : agent_node, tools_node, should_continue
│   │   ├── prompts.py          # System prompts (SYSTEM_PROMPT, SYSTEM_PROMPT_BT)
│   │   ├── retriever.py        # Wrapper Pinecone : embed + query → candidats + scores
│   │   ├── tools.py            # search_apis_tool (mode bind_tools)
│   │   ├── transports.py       # KongChatTransport / AsyncKongChatTransport (httpx)
│   │   └── build.py            # Assemble et compile le StateGraph → expose `graph`
│   │
│   ├── chat/
│   │   ├── app.py              # Chainlit : graph.astream() → UI conversationnelle
│   │   └── server.py           # (optionnel) Endpoint FastAPI
│   │
│   └── batch/
│       ├── golden.py           # load_golden() → lit data/golden_dataset.json
│       ├── logger.py           # append_jsonl(), append_error_jsonl(), log_path()
│       ├── sheet_writer.py     # init_sheet(), append_row(), get_worksheet()
│       └── run_batch.py        # CLI principal — voir section "Lancer le batch"
│
└── tests/
    ├── conftest.py
    ├── test_batch.py           # Vérifie les colonnes Sheet (sans réseau)
    └── test_graph.py           # Smoke test graph.invoke()
```

---

## Colonnes Google Sheet

| Colonne | Description |
|---|---|
| `id` | Identifiant de la question (ex: Q0001) |
| `question` | Texte de la question |
| `output` | Réponse complète du LLM |
| `final_apis` | APIs recommandées extraites (`RECOMMANDED_APIS: [...]`) |
| `expected_apis` | APIs attendues (golden dataset — non transmises au LLM) |
| `latency_s` | Durée totale de l'appel graphe en secondes |
| `tokens_in` | Total tokens en entrée (tous appels LLM cumulés) |
| `tokens_out` | Total tokens en sortie |
| `tokens_think` | Total thinking tokens |
| `tokens_total` | Total tous types |
| `tokens_detail` | Détail par appel LLM : `{"tokens_in": [895, 3139], ...}` |
| `llm_call_count` | Nombre d'appels LLM (= tours du ReAct loop) |
| `tool_call_count` | Nombre d'appels Pinecone |
| `tool_call_inputs` | Requêtes envoyées à Pinecone |
| `retrieved_slugs` | Slugs récupérés avec leur fréquence |
| `history_summary` | Résumé tronqué de la conversation (40 mots/message) |

---

## Robustesse batch

- **Retry 429** : backoff automatique 60s → 120s → 300s → 600s. Chaque tentative loggée dans `logs/errors_*.jsonl`.
- **Sauvegarde incrémentale** : JSONL + Sheet mis à jour après **chaque** question. Un crash ne perd que la question en cours.
- **Reprise** : `--resume` détecte automatiquement les questions déjà traitées en lisant tous les `logs/resultats_*.jsonl` et en déduplicant par ID.

---

## Architecture

```
graph/build.py  →  expose `graph`
                      ├── chat/app.py      (Chainlit)
                      └── batch/run_batch.py  (éval)
```

Le même graphe est utilisé en production (chat) et en évaluation (batch) — ce qu'on évalue est exactement ce qui tourne en prod.
