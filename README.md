agentic4api/
│
├── pyproject.toml              # Dépendances + package importable (pip install -e .).
│                               #   langgraph, langchain-google-genai, langchain-pinecone,
│                               #   chainlit, gspread, google-auth, pandas. PAS de qdrant.
├── .env                        # Secrets (jamais commité) : GOOGLE_API_KEY, PINECONE_API_KEY,
│                               #   PINECONE_INDEX, SHEET_ID, GOOGLE_SA_JSON (service account).
├── .env.example                # Mêmes clés, valeurs vides — celui-ci est commité (doc).
├── .gitignore                  # Ignore .env, __pycache__, service account JSON.
├── README.md                   # Comment lancer le chat + l'éval.
├── Dockerfile                  # Image de l'agent/chat pour déploiement (Cloud Run).
├── .dockerignore
│
├── data/
│   └── golden_dataset.json     # Tes 464 questions (id, question, expected_apis, category,
│                               #   domain, difficulty, register, phrasing). PAS lu par l'agent,
│                               #   sert d'entrée à l'éval Colab + au batch.
│
├── src/
│   └── agentic4api/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py     # Charge .env. UN seul endroit pour : model string Gemini Flash,
│       │                       #   modèle d'embedding Gemini, topK Pinecone, nom d'index, SHEET_ID.
│       │
│       ├── graph/              # ── LE CŒUR : orchestration LangGraph partagée chat ↔ batch ──
│       │   ├── __init__.py
│       │   ├── state.py        # State (TypedDict) : question, retries, candidates, scores,
│       │   │                   #   final_apis + champs de mesure (latency, tokens_in/out/think).
│       │   ├── retriever.py    # Wrapper Pinecone : embed la question (Gemini Embedding),
│       │   │                   #   query l'index, renvoie candidats + SCORES bruts (pour threshold).
│       │   ├── prompts.py      # System prompts (noms génériques foo-bar-api, jamais le golden).
│       │   ├── nodes.py        # Les nœuds : guard (question corrompue), retrieve, answer
│       │   │                   #   (Gemini Flash décide/formate RECOMMANDED_APIS), + capture
│       │   │                   #   tokens & latence par question dans le State.
│       │   └── build.py        # Assemble le StateGraph + boucle de retry + MemorySaver, compile,
│       │                       #   EXPOSE `graph` — le point d'import unique (chat ET batch).
│       │
│       ├── chat/              # ── EXPOSER LE CHAT (ce que tu demandes) ──
│       │   ├── __init__.py
│       │   ├── app.py          # Chainlit : importe `graph`, graph.astream() → UI conversationnelle
│       │   │                   #   en streaming. C'est CE fichier que tu lances/déploies pour
│       │   │                   #   discuter et envoyer des requêtes plus tard.
│       │   └── server.py       # (optionnel) endpoint HTTP/FastAPI si tu veux appeler le graphe
│       │                       #   par API programmatique plutôt que par l'UI chat.
│       │
│       └── batch/             # ── GÉNÈRE LES RÉPONSES → ÉCRIT DANS LE SHEET ──
│           ├── __init__.py
│           ├── golden.py       # load_golden() → lit data/golden_dataset.json.
│           ├── sheet_writer.py # Auth service account + écriture batch dans le Google Sheet.
│           │                   #   Colonnes : id, chatInput, question, output, retries, latency_s,
│           │                   #   tokens_in, tokens_out, tokens_think, tokens_total. UN seul write.
│           └── run_batch.py    # CLI : graph.batch(golden) → collecte réponses + métriques
│                               #   d'exécution par question → sheet_writer. (Remplace le webhook n8n.)
│
├── scripts/
│   └── index_pinecone.py       # (Ré)indexation du catalogue OpenAPI → vecteurs Pinecone.
│
└── tests/
    ├── __init__.py
    ├── conftest.py             # Fixtures : sample golden, mock retriever.
    ├── test_graph.py           # Smoke test : graph.invoke() sur 2-3 questions.
    └── test_batch.py           # Vérifie que les colonnes (retries/latency/tokens) sont remplies.


# agentic4api

Découverte sémantique d'API : un agent LangGraph (Pinecone + Gemini) partagé entre
un **chat** (Chainlit) et un **batch d'évaluation** qui écrit dans Google Sheet.
L'évaluation des métriques (MRR, nDCG, Recall…) reste dans ton notebook Colab.

## Installation

```bash
pip install -e .
cp .env.example .env   # puis remplis les valeurs
```

## Configuration (.env)

- `GOOGLE_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX`
- `SHEET_ID` + `GOOGLE_SA_JSON` (service account). **Partage le Sheet avec l'email
  du service account**, sinon écriture impossible.
- `GEMINI_MODEL` : le model string EXACT (à recopier depuis ton n8n).

## Lancer

Chat (local) :
```bash
chainlit run src/agentic4api/chat/app.py -w
```

Batch → Sheet :
```bash
agentic4api-batch --limit 5            # smoke test
agentic4api-batch --worksheet run_v1   # run complet
```

Tests (sans réseau) :
```bash
pytest
```

## Architecture

`graph/build.py` expose `graph`. `chat/app.py` et `batch/run_batch.py` l'importent
tous deux → l'éval teste exactement l'orchestration de la prod.

Colonnes écrites dans le Sheet : `id, chatInput, question, output, retries,
latency_s, tokens_in, tokens_out, tokens_think, tokens_total`.

## Points à vérifier (cherche `⚠ VÉRIFIER` dans le code)

1. **Model string + thinking** : 2.5 → `thinking_budget`, 3.x → `thinking_level`.
2. **Format de sortie** : l'agent doit produire `RECOMMANDED_APIS: [...]`.
3. **Champs metadata Pinecone** : `slug` / `text` dans `retriever.py` doivent
   matcher ce qu'écrit `scripts/index_pinecone.py`.
