import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")


@pytest.mark.asyncio
async def test_retrieve_returns_retrieved_chunks():
    from src.rag.retriever import RetrievedChunk, retrieve

    fake_embedding = [0.1] * 1024

    mock_doc = MagicMock()
    mock_doc.source_uri = "https://example.com"
    mock_doc.title = "Example Doc"

    mock_chunk = MagicMock()
    mock_chunk.id = uuid.uuid4()
    mock_chunk.document_id = uuid.uuid4()
    mock_chunk.content = "Some content"
    mock_chunk.document = mock_doc
    mock_chunk.metadata_ = {"key": "value"}

    mock_session = AsyncMock()

    with (
        patch("src.rag.retriever.embed_query", new=AsyncMock(return_value=fake_embedding)),
        patch("src.rag.retriever.search_similar", new=AsyncMock(return_value=[(mock_chunk, 0.85)])),
    ):
        result = await retrieve(mock_session, "What is Python?")

    assert len(result) == 1
    assert isinstance(result[0], RetrievedChunk)
    assert result[0].similarity_score == 0.85
    assert result[0].content == "Some content"
    assert result[0].source_uri == "https://example.com"
    assert result[0].document_title == "Example Doc"


@pytest.mark.asyncio
async def test_retrieve_respects_top_k_override():
    from src.rag.retriever import retrieve

    mock_search = AsyncMock(return_value=[])
    mock_session = AsyncMock()

    with (
        patch("src.rag.retriever.embed_query", new=AsyncMock(return_value=[0.0] * 1024)),
        patch("src.rag.retriever.search_similar", new=mock_search),
    ):
        await retrieve(mock_session, "test", top_k=7)

    mock_search.assert_called_once_with(mock_session, [0.0] * 1024, top_k=7)


@pytest.mark.asyncio
async def test_retrieve_handles_missing_document():
    from src.rag.retriever import retrieve

    mock_chunk = MagicMock()
    mock_chunk.id = uuid.uuid4()
    mock_chunk.document_id = uuid.uuid4()
    mock_chunk.content = "Orphan chunk"
    mock_chunk.document = None
    mock_chunk.metadata_ = {}

    mock_session = AsyncMock()

    with (
        patch("src.rag.retriever.embed_query", new=AsyncMock(return_value=[0.1] * 1024)),
        patch("src.rag.retriever.search_similar", new=AsyncMock(return_value=[(mock_chunk, 0.5)])),
    ):
        result = await retrieve(mock_session, "test")

    assert result[0].source_uri == ""
    assert result[0].document_title is None
