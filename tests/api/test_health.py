import os
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://kg_user:testpassword@localhost:5432/knowledgeforge_test"
)
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")


@pytest.mark.asyncio
async def test_health_endpoint():
    from src.api.dependencies import get_session
    from src.api.main import app

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()

    async def override_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

    app.dependency_overrides.clear()
