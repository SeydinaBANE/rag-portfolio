# KnowledgeForge — TODO

## Phase 1 — Foundation ✅
- [x] Structure de répertoires
- [x] CLAUDE.md, TODO.md, README.md
- [x] `pyproject.toml` avec dépendances et config outils
- [x] `requirements.txt` / `requirements-dev.txt`
- [x] `.env.example`
- [x] `src/config.py` — Pydantic Settings
- [x] `src/db/connection.py` — moteur SQLAlchemy async
- [x] `src/db/models.py` — ORM (Document, Chunk, QueryLog, Feedback, EvalRun)
- [x] `alembic/` — configuration + migration initiale
- [x] `docker/docker-compose.yml` — postgres+pgvector, mlflow, api
- [x] `docker/Dockerfile` — multi-stage
- [ ] **ACTION REQUISE** : `cp .env.example .env` et remplir les valeurs

## Phase 2 — Pipeline ETL ✅
- [x] `src/pipeline/loaders/pdf_loader.py` — PyMuPDF
- [x] `src/pipeline/loaders/url_loader.py` — httpx + BeautifulSoup
- [x] `src/pipeline/loaders/jsonl_loader.py`
- [x] `src/pipeline/transformers/cleaner.py` — strip HTML, normalisation unicode
- [x] `src/pipeline/transformers/chunker.py` — RecursiveCharacterTextSplitter
- [x] `src/pipeline/transformers/metadata_enricher.py`
- [x] `src/rag/embedder.py` — client embedding OpenRouter
- [x] `src/pipeline/ingestion_pipeline.py` — orchestrateur complet
- [x] `scripts/seed_data.py` — ingestion de 3 docs de démo
- [ ] **À VÉRIFIER** : `python scripts/seed_data.py` → rows dans `chunks` avec embeddings

## Phase 3 — RAG Core ✅
- [x] `src/rag/prompt_templates.py` — prompts versionnés v1/v2
- [x] `src/db/repositories/chunk_repo.py` — CRUD + recherche vectorielle
- [x] `src/db/repositories/query_log_repo.py`
- [x] `src/rag/retriever.py` — pgvector similarity search
- [x] `src/rag/chain.py` — chaîne LCEL complète + MLflow
- [x] `data/eval/golden_qa.jsonl` — 20 questions de référence
- [ ] **À VÉRIFIER** : smoke test REPL question → réponse avec sources

## Phase 4 — API REST ✅
- [x] `src/api/main.py` — app factory, lifespan, CORS
- [x] `src/api/dependencies.py` — injection DB
- [x] `src/api/schemas/` — QueryRequest/Response, IngestRequest, FeedbackRequest, DocumentOut
- [x] `src/api/routers/health.py` — GET /health, GET /metrics
- [x] `src/api/routers/query.py` — POST /query
- [x] `src/api/routers/ingest.py` — POST /ingest, GET /ingest/status/{id}
- [x] `src/api/routers/feedback.py` — POST /feedback
- [x] `src/api/routers/documents.py` — GET /documents, DELETE /documents/{id}
- [ ] **À VÉRIFIER** : `uvicorn src.api.main:app --reload` → Swagger UI sur /docs

## Phase 5 — MLOps ✅
- [x] `src/mlops/tracker.py` — wrapper MLflow
- [x] `src/rag/chain.py` — instrumenté avec MLflow
- [x] `src/mlops/evaluator.py` — eval sur golden_qa.jsonl (LLM-as-judge)
- [x] `src/mlops/drift_monitor.py` — alerte si dégradation qualité
- [x] `scripts/run_eval.py` — lancement évaluation + rapport MLflow
- [ ] **À VÉRIFIER** : MLflow UI localhost:5000 avec runs et métriques

## Phase 6 — Tests ✅
- [x] `tests/conftest.py` — fixtures async
- [x] `tests/unit/test_cleaner.py`
- [x] `tests/unit/test_chunker.py`
- [x] `tests/unit/test_prompt_templates.py`
- [x] `tests/unit/test_embedder.py` (mocké)
- [x] `tests/integration/test_ingestion_pipeline.py`
- [x] `tests/api/test_health.py`
- [x] `tests/api/test_query_endpoint.py`
- [x] `tests/api/test_feedback_endpoint.py`
- [ ] `tests/integration/test_retriever.py` — à compléter
- [ ] **À VÉRIFIER** : `pytest tests/ --cov=src` → coverage ≥ 75%

## Phase 7 — CI/CD + Polish ✅
- [x] `.github/workflows/ci.yml` — lint + type-check + tests
- [x] `.github/workflows/cd.yml` — build + push Docker
- [x] `README.md` — architecture, quickstart, API docs
- [x] `.gitignore`
- [ ] Prometheus metrics middleware (optionnel — à ajouter si souhaité)
- [ ] Badge coverage dans README (après push GitHub)

## Vérification finale end-to-end
- [ ] `docker compose up -d` sans erreur
- [ ] `python scripts/seed_data.py` ingère des docs
- [ ] `curl POST /api/v1/query` retourne une réponse avec sources
- [ ] MLflow UI montre les runs avec métriques
- [ ] `python scripts/run_eval.py` log avg_answer_relevancy
- [ ] `pytest --cov=src` ≥ 75% coverage
- [ ] GitHub Actions CI passe
