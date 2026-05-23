# KnowledgeForge

Plateforme de Q&A intelligente sur documents techniques propulsée par RAG (Retrieval-Augmented Generation).

**Stack :** Python 3.12 · FastAPI · PostgreSQL + pgvector · OpenRouter · MLflow · LangChain · Docker · GitHub Actions

## Architecture

```
Documents (PDF / URL / JSONL)
        │
        ▼
  Pipeline ETL ──────────────────────────────────────────────────▶ PostgreSQL
  (loader → clean → chunk → embed)                                  (pgvector)
                                                                        │
  User Question ──▶ Embedder ──▶ Retriever (ANN search) ◀────────────┘
                                      │
                                      ▼
                              Prompt Template (versioned)
                                      │
                                      ▼
                              OpenRouter LLM ──▶ Answer + Sources
                                      │
                                      ▼
                              MLflow Tracker (metrics + versioning)
```

## Démarrage rapide

```bash
# 1. Cloner et configurer l'environnement
cp .env.example .env
# Remplir OPENROUTER_API_KEY et POSTGRES_PASSWORD dans .env

# 2. Démarrer les services
docker compose -f docker/docker-compose.yml up -d

# 3. Ingérer des documents de démo
python scripts/seed_data.py

# 4. Poser une question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does pgvector handle approximate nearest neighbor search?"}'

# 5. Ouvrir le dashboard MLflow
open http://localhost:5000
```

## API

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Statut des services |
| POST | `/api/v1/query` | Question → réponse RAG avec sources |
| POST | `/api/v1/ingest` | Ingestion de nouveaux documents |
| GET | `/api/v1/ingest/status/{job_id}` | Statut d'un job d'ingestion |
| GET | `/api/v1/documents` | Liste des documents ingérés |
| DELETE | `/api/v1/documents/{id}` | Supprimer un document |
| POST | `/api/v1/feedback` | Feedback sur une réponse |
| GET | `/api/v1/metrics` | Métriques Prometheus |

Documentation interactive : `http://localhost:8000/docs`

## Compétences démontrées

- **Data Engineering** : pipeline ETL modulaire (loaders PDF/URL/JSONL, nettoyage, chunking, ingestion)
- **RAG** : retrieval hybride pgvector + reranking MMR, chaîne LCEL LangChain
- **MLOps** : MLflow tracking (paramètres, métriques, artifacts), model registry, évaluation automatique, drift monitoring
- **API REST** : FastAPI avec dépendances injectées, Pydantic v2, pagination, background tasks
- **Vector DB** : PostgreSQL + pgvector, index IVFFlat, recherche cosine ANN
- **Prompt Engineering** : prompts versionnés, comparaison via MLflow, LLM-as-judge
- **CI/CD** : GitHub Actions (lint + mypy + tests + coverage gate + Docker build)
- **Tests** : pytest-asyncio, testcontainers, mocks LLM, coverage ≥ 75%

## Tests

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Évaluation MLOps

```bash
python scripts/run_eval.py
```

Lance une évaluation sur `data/eval/golden_qa.jsonl` et logue dans MLflow :
- `avg_answer_relevancy` (LLM-as-judge)
- `avg_faithfulness`
- `avg_context_precision`
- `p95_latency_ms`
