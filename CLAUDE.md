# KnowledgeForge — Instructions Claude

## Projet
Système RAG portfolio (Python 3.12, FastAPI, PostgreSQL + pgvector, OpenRouter, MLflow).
Répertoire racine : `/Users/baneseydina/Desktop/projet-1`

## Conventions de code
- Python 3.12, async/await partout (asyncpg + SQLAlchemy async)
- Pydantic v2 pour tous les modèles de données et la config
- `ruff` pour le lint et le format (ligne max 100 chars)
- `mypy` strict sur `src/` (pas de `Any` implicite)
- Pas de commentaires sauf si logique non-évidente
- Imports organisés : stdlib → third-party → local (ruff isort)

## Architecture clé
- **`src/config.py`** : unique source de vérité pour la config (Pydantic Settings)
- **`src/db/models.py`** : ORM SQLAlchemy — modifier ici, puis `alembic revision --autogenerate`
- **`src/pipeline/ingestion_pipeline.py`** : orchestrateur ETL (loader → clean → chunk → embed → store)
- **`src/rag/chain.py`** : chaîne LCEL LangChain (retriever → prompt → LLM)
- **`src/mlops/tracker.py`** : chaque appel LLM doit être wrappé dans un MLflow run

## LLM / Embeddings
- Provider : OpenRouter (`https://openrouter.ai/api/v1`)
- Utiliser `langchain_openai.ChatOpenAI` avec `openai_api_base=OPENROUTER_BASE_URL`
- Embedding model par défaut : `openai/text-embedding-3-small` via OpenRouter
- Ne jamais hardcoder les noms de modèles — toujours via `config.py`

## Base de données
- PostgreSQL 16 + pgvector via Docker
- Migrations via Alembic uniquement (jamais `CREATE TABLE` manuel)
- Connexion async via `asyncpg` driver
- Index IVFFlat sur `chunks.embedding` pour la recherche ANN

## Tests
- `testcontainers[postgres]` pour les tests d'intégration (vraie DB)
- Mock LLM/embedder dans les tests unitaires avec `unittest.mock`
- Coverage cible : ≥ 75%
- Lancer : `pytest tests/ --cov=src`

## Docker
- `docker compose up -d` démarre : postgres+pgvector, mlflow, api
- Après `up` : `alembic upgrade head` est lancé automatiquement (via command dans compose)
- Variables sensibles dans `.env` (copier `.env.example`)

## Variables d'environnement requises
```
OPENROUTER_API_KEY=sk-or-...
POSTGRES_PASSWORD=...
DATABASE_URL=postgresql+asyncpg://kg_user:PASSWORD@localhost:5432/knowledgeforge
MLFLOW_TRACKING_URI=http://localhost:5001
```

## Commandes fréquentes
```bash
# Démarrer l'environnement (--env-file requis car compose est dans docker/)
docker compose -f docker/docker-compose.yml --env-file .env up -d

# Migrations
alembic upgrade head
alembic revision --autogenerate -m "description"

# Dev
uvicorn src.api.main:app --reload --port 8000

# Tests
pytest tests/ --cov=src --cov-report=term-missing

# Lint
ruff check src/ tests/
ruff format src/ tests/
mypy src/

# Ingestion de démo
python scripts/seed_data.py

# Évaluation MLOps
python scripts/run_eval.py
```
