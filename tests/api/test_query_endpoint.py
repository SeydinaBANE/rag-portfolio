import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://kg_user:testpassword@localhost:5432/knowledgeforge_test")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")

from src.rag.chain import RAGResponse
from src.rag.retriever import RetrievedChunk


@pytest.mark.asyncio
async def test_query_endpoint_returns_answer():
    from src.api.main import app
    from src.api.dependencies import get_session

    mock_session = AsyncMock()

    async def override_session():
        yield mock_session

    mock_chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Python is a versatile programming language.",
        similarity_score=0.92,
        source_uri="https://example.com",
        document_title="Python Docs",
        metadata={},
    )
    fake_response = RAGResponse(
        answer="Python is a versatile language.",
        query_log_id=uuid.uuid4(),
        sources=[mock_chunk],
        latency_ms=120,
    )

    app.dependency_overrides[get_session] = override_session

    with patch("src.api.routers.query.run_rag", new=AsyncMock(return_value=fake_response)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"question": "What is Python?"},
            )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["latency_ms"] == 120

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_query_endpoint_validates_short_question():
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"question": "hi"})

    assert response.status_code == 422
