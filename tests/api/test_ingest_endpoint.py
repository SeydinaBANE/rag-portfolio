import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")

from src.pipeline.ingestion_pipeline import IngestionResult


@pytest.mark.asyncio
async def test_ingest_returns_202_with_job_id():
    from src.api.dependencies import get_session
    from src.api.main import app

    mock_session = AsyncMock()

    async def override_session():
        yield mock_session

    fake_result = IngestionResult(
        job_id="some-id",
        documents_ingested=1,
        chunks_created=5,
        status="completed",
    )

    app.dependency_overrides[get_session] = override_session

    with patch("src.api.routers.ingest.ingest", new=AsyncMock(return_value=fake_result)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ingest",
                json={"sources": [{"uri": "https://example.com", "type": "url"}]},
            )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "processing"


@pytest.mark.asyncio
async def test_ingest_status_not_found():
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/ingest/status/nonexistent-job-id")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_ingest_validates_source_type():
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ingest",
            json={"sources": [{"uri": "https://example.com", "type": "unknown"}]},
        )

    assert response.status_code == 422
