from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import documents, feedback, health, ingest, query
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
    except Exception:
        pass  # MLflow may still be starting; will connect on first request
    yield


app = FastAPI(
    title="KnowledgeForge",
    description="RAG-powered document Q&A platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PREFIX = "/api/v1"
app.include_router(health.router, prefix=PREFIX)
app.include_router(query.router, prefix=PREFIX)
app.include_router(ingest.router, prefix=PREFIX)
app.include_router(feedback.router, prefix=PREFIX)
app.include_router(documents.router, prefix=PREFIX)
