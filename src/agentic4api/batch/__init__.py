# ─────────────────────────────────────────────────────────────────────────────
# EMPLACEMENT DE CE FICHIER : src/agentic4api/batch/__init__.py
# (dans le sous-dossier batch/, à côté de golden.py, run_batch.py, sheet_writer.py)
# ─────────────────────────────────────────────────────────────────────────────
"""
batch — génération des réponses du golden dataset → écriture Google Sheet (éval).

Remplace le webhook n8n : exécute le graphe sur les 464 questions, mesure
latence/tokens/retries par question, et écrit tout dans le Sheet en un seul write.
L'évaluation des métriques (MRR, nDCG…) se fait ensuite côté Colab à partir du Sheet.

Sous-modules :
  - golden.py       : load_golden() → lit data/golden_dataset.json.
  - sheet_writer.py : auth service account + écriture batch dans le Google Sheet.
  - run_batch.py    : CLI (point d'entrée `agentic4api-batch`).
"""
