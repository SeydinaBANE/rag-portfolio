import os
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")

from src.db.models import Document


def _make_document(**kwargs) -> Document:
    doc = MagicMock(spec=Document)
    doc.id = kwargs.get("id", uuid.uuid4())
    doc.source_uri = kwargs.get("source_uri", "https://example.com")
    doc.doc_type = kwargs.get("doc_type", "url")
    doc.title = kwargs.get("title", "Example")
    doc.ingested_at = kwargs.get("ingested_at", datetime.now(UTC))
    doc.chunk_count = kwargs.get("chunk_count", 10)
    return doc


@pytest.mark.asyncio
async def test_list_documents_returns_paginated_list():
    from src.api.dependencies import get_session
    from src.api.main import app

    doc = _make_document()

    mock_session = AsyncMock()

    count_result = MagicMock()
    count_result.scalar_one.return_value = 1

    docs_result = MagicMock()
    docs_result.scalars.return_value = [doc]

    mock_session.execute.side_effect = [count_result, docs_result]

    async def override_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/documents")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["page"] == 1


@pytest.mark.asyncio
async def test_delete_document_returns_204():
    from src.api.dependencies import get_session
    from src.api.main import app

    doc_id = uuid.uuid4()
    doc = _make_document(id=doc_id)

    mock_session = AsyncMock()
    find_result = MagicMock()
    find_result.scalar_one_or_none.return_value = doc
    mock_session.execute.return_value = find_result

    async def override_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete(f"/api/v1/documents/{doc_id}")

    app.dependency_overrides.clear()

    assert response.status_code == 204


@pytest.mark.asyncio
async def test_delete_document_returns_404_when_not_found():
    from src.api.dependencies import get_session
    from src.api.main import app

    mock_session = AsyncMock()
    find_result = MagicMock()
    find_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = find_result

    async def override_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete(f"/api/v1/documents/{uuid.uuid4()}")

    app.dependency_overrides.clear()

    assert response.status_code == 404
