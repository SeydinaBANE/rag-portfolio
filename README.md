# KnowledgeForge

> Plateforme RAG (Retrieval-Augmented Generation) pour le Q&A intelligent sur documents techniques.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+pgvector-4169E1?logo=postgresql&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.17-0194E2?logo=mlflow&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-86%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Vue d'ensemble

KnowledgeForge ingère des documents (PDF, URL, JSONL), les découpe, les vectorise via Cohere, les stocke dans PostgreSQL+pgvector, puis répond à des questions en langage naturel en s'appuyant sur les passages les plus pertinents. Chaque appel LLM est tracé dans MLflow pour le suivi des métriques et la détection de drift.

```
Documents (PDF / URL / JSONL)
        │
        ▼
  Pipeline ETL ──────────────────────────────────────────────────▶ PostgreSQL
  loader → clean → chunk → embed (Cohere)                          (pgvector)
                                                                        │
  Question ──▶ Embedder ──▶ Retriever (cosine ANN, IVFFlat) ◀────────┘
                                  │
                                  ▼
                          Prompt Template (versionné)
                                  │
                                  ▼
                       OpenRouter LLM ──▶ Réponse + Sources citées
                                  │
                                  ▼
                        MLflow Tracker (latence, métriques, artifacts)
```

---

## Stack technique

| Couche | Technologie |
|---|---|
| API | FastAPI 0.115, Pydantic v2, uvicorn |
| RAG | LangChain LCEL, Cohere embed-multilingual-v3.0 |
| LLM | OpenRouter (nvidia/nemotron-super-49b, via API) |
| Vector DB | PostgreSQL 16 + pgvector, index IVFFlat cosine |
| ORM | SQLAlchemy async + asyncpg |
| MLOps | MLflow 2.17 (tracking, registry, évaluation, drift) |
| Migrations | Alembic |
| Tests | pytest-asyncio, testcontainers, coverage 86% |
| Lint / Types | ruff, mypy strict |
| Infra | Docker Compose (postgres, mlflow, api) |

---

## Démarrage rapide

### Prérequis

- Docker & Docker Compose
- Python 3.12
- Clé API [OpenRouter](https://openrouter.ai) et [Cohere](https://cohere.com)

### Installation

```bash
# 1. Cloner le repo
git clone https://github.com/SeydinaBANE/rag-portfolio.git
cd rag-portfolio

# 2. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env : renseigner OPENROUTER_API_KEY, COHERE_API_KEY, POSTGRES_PASSWORD

# 3. Démarrer les services (postgres + mlflow + api)
docker compose -f docker/docker-compose.yml --env-file .env up -d

# 4. Ingérer des documents de démo
python scripts/seed_data.py

# 5. Poser une question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does pgvector handle approximate nearest neighbor search?"}'
```

Interfaces disponibles :
- **API docs** → http://localhost:8000/docs
- **MLflow UI** → http://localhost:5001

---

## Variables d'environnement

| Variable | Description | Exemple |
|---|---|---|
| `OPENROUTER_API_KEY` | Clé API OpenRouter | `sk-or-...` |
| `COHERE_API_KEY` | Clé API Cohere (embeddings) | `...` |
| `POSTGRES_PASSWORD` | Mot de passe PostgreSQL | `changeme` |
| `DATABASE_URL` | URL de connexion asyncpg | `postgresql+asyncpg://kg_user:...@localhost:5432/knowledgeforge` |
| `MLFLOW_TRACKING_URI` | URI du serveur MLflow | `http://localhost:5001` |
| `CHAT_MODEL` | Modèle LLM via OpenRouter | `nvidia/nemotron-super-49b-v1:free` |
| `CHUNK_SIZE` | Taille des chunks (tokens) | `512` |
| `TOP_K_RETRIEVAL` | Nombre de chunks récupérés | `5` |

---

## API

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Statut des services (DB, MLflow) |
| `POST` | `/api/v1/query` | Question → réponse RAG + sources citées |
| `POST` | `/api/v1/ingest` | Ingestion de documents (PDF / URL / JSONL) |
| `GET` | `/api/v1/ingest/status/{job_id}` | Statut d'un job d'ingestion |
| `GET` | `/api/v1/documents` | Liste des documents ingérés |
| `DELETE` | `/api/v1/documents/{id}` | Supprimer un document et ses chunks |
| `POST` | `/api/v1/feedback` | Feedback utilisateur sur une réponse |
| `GET` | `/api/v1/metrics` | Métriques Prometheus |

**Exemple de requête :**

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "sources": [
      {"uri": "https://arxiv.org/pdf/2307.06435", "type": "pdf"},
      {"uri": "https://docs.python.org/3/", "type": "url"}
    ]
  }'
```

**Exemple de réponse `/query` :**

```json
{
  "answer": "pgvector uses IVFFlat indexing to partition the embedding space...",
  "query_log_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "latency_ms": 842,
  "sources": [
    {
      "document_title": "pgvector README",
      "source_uri": "https://github.com/pgvector/pgvector",
      "content_snippet": "IVFFlat divides vectors into lists and searches...",
      "similarity_score": 0.9231
    }
  ]
}
```

---

## Structure du projet

```
.
├── src/
│   ├── api/
│   │   ├── routers/        # Endpoints FastAPI (query, ingest, documents, feedback, health)
│   │   └── schemas/        # Schémas Pydantic I/O
│   ├── pipeline/
│   │   ├── loaders/        # Chargeurs PDF, URL, JSONL
│   │   └── transformers/   # Nettoyage, chunking, enrichissement metadata
│   ├── rag/
│   │   ├── chain.py        # Chaîne LCEL LangChain
│   │   ├── embedder.py     # Cohere embeddings
│   │   └── retriever.py    # Recherche ANN pgvector
│   ├── mlops/
│   │   ├── tracker.py      # Wrapper MLflow par appel LLM
│   │   ├── evaluator.py    # LLM-as-judge (faithfulness, relevancy)
│   │   └── drift_monitor.py
│   ├── db/
│   │   ├── models.py       # ORM SQLAlchemy (Document, Chunk, QueryLog)
│   │   └── repositories/   # Accès données async
│   └── config.py           # Source de vérité (Pydantic Settings)
├── tests/                  # pytest-asyncio + testcontainers
├── alembic/                # Migrations DB
├── docker/                 # Dockerfiles + docker-compose.yml
├── scripts/
│   ├── seed_data.py        # Ingestion de démo
│   └── run_eval.py         # Évaluation MLOps
└── data/eval/
    └── golden_qa.jsonl     # Dataset d'évaluation
```

---

## Tests

```bash
# Lancer la suite complète avec coverage
pytest tests/ --cov=src --cov-report=term-missing

# Lint et types
ruff check src/ tests/
mypy src/
```

Coverage actuelle : **86%** — tests unitaires (mocks LLM/embedder) + tests d'intégration (vraie DB via testcontainers).

---

## MLOps

Chaque appel LLM est wrappé dans un run MLflow avec :

| Métrique | Description |
|---|---|
| `latency_ms` | Temps de réponse end-to-end |
| `avg_answer_relevancy` | Pertinence de la réponse (LLM-as-judge) |
| `avg_faithfulness` | Fidélité aux sources récupérées |
| `avg_context_precision` | Précision du contexte fourni |
| `p95_latency_ms` | Latence au 95e percentile |

```bash
# Lancer une évaluation complète sur le dataset golden Q&A
python scripts/run_eval.py
```

Les résultats sont visibles dans MLflow UI → http://localhost:5001 sous l'expérience `knowledgeforge-rag`.

---

## Développement local (sans Docker)

```bash
# Installer les dépendances
pip install -r requirements.txt -r requirements-dev.txt

# PostgreSQL requis en local avec l'extension pgvector
# puis :
alembic upgrade head
uvicorn src.api.main:app --reload --port 8000
```

---

## Compétences démontrées

- **Data Engineering** — pipeline ETL modulaire, loaders multi-format, chunking configurable
- **RAG** — retrieval cosine ANN (IVFFlat), chaîne LCEL, prompt versionné
- **MLOps** — MLflow tracking, model registry, évaluation automatique, drift monitoring
- **API REST** — FastAPI, Pydantic v2, background tasks, pagination, injection de dépendances
- **Vector DB** — PostgreSQL + pgvector, index IVFFlat, similarité cosine
- **Tests** — pytest-asyncio, testcontainers (vraie DB), mocks LLM, coverage ≥ 75%
- **Qualité** — mypy strict, ruff, pre-commit hooks

---

## Licence

MIT — voir [LICENSE](LICENSE).
