import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://kg_user:testpassword@localhost:5432/knowledgeforge_test")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")


@pytest.mark.asyncio
async def test_feedback_invalid_rating():
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/feedback",
            json={"query_log_id": str(uuid.uuid4()), "rating": 5},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_feedback_not_found_log():
    from src.api.main import app
    from src.api.dependencies import get_session

    mock_session = AsyncMock()

    async def override():
        yield mock_session

    app.dependency_overrides[get_session] = override

    with patch("src.api.routers.feedback.get_log_by_id", new=AsyncMock(return_value=None)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/feedback",
                json={"query_log_id": str(uuid.uuid4()), "rating": 1},
            )
    assert response.status_code == 404
    app.dependency_overrides.clear()
