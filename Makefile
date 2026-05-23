COMPOSE = docker compose -f docker/docker-compose.yml --env-file .env

.PHONY: help install up down restart logs migrate seed eval dev test lint format typecheck clean

help:
	@echo ""
	@echo "  KnowledgeForge — commandes disponibles"
	@echo ""
	@echo "  Infrastructure"
	@echo "    make up          Démarrer tous les services Docker"
	@echo "    make down        Arrêter les services"
	@echo "    make restart     Redémarrer les services"
	@echo "    make logs        Afficher les logs en temps réel"
	@echo ""
	@echo "  Base de données"
	@echo "    make migrate     Appliquer les migrations Alembic"
	@echo ""
	@echo "  Développement"
	@echo "    make install     Installer les dépendances (venv)"
	@echo "    make dev         Lancer l'API en mode reload"
	@echo "    make seed        Ingérer les documents de démo"
	@echo "    make eval        Lancer l'évaluation MLOps"
	@echo ""
	@echo "  Qualité"
	@echo "    make test        Lancer tous les tests"
	@echo "    make lint        Vérifier le style (ruff)"
	@echo "    make format      Formater le code (ruff)"
	@echo "    make typecheck   Vérifier les types (mypy)"
	@echo "    make check       lint + typecheck + test"
	@echo ""
	@echo "  Nettoyage"
	@echo "    make clean       Supprimer les fichiers temporaires"

# ── Infrastructure ──────────────────────────────────────────────────────────

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart

logs:
	$(COMPOSE) logs -f

# ── Base de données ──────────────────────────────────────────────────────────

migrate:
	alembic upgrade head

# ── Développement ────────────────────────────────────────────────────────────

install:
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements-dev.txt && \
	pre-commit install
	@echo "✅ Environnement prêt — activez avec : source .venv/bin/activate"

dev:
	uvicorn src.api.main:app --reload --port 8000

seed:
	python scripts/seed_data.py

eval:
	python scripts/run_eval.py

# ── Qualité ──────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ tests/api/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

check: lint typecheck test

# ── Nettoyage ────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -f test_mlflow.db mlflow_test.db mlflow_cd.db
