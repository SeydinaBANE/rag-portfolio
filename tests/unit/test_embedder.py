import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-cohere-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")


@pytest.mark.asyncio
async def test_embed_texts_returns_list():
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1] * 1024, [0.2] * 1024]

    with patch("src.rag.embedder.get_cohere_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=mock_response)
        mock_get.return_value = mock_client

        from src.rag.embedder import embed_texts
        result = await embed_texts(["text1", "text2"])

    assert len(result) == 2
    assert len(result[0]) == 1024


@pytest.mark.asyncio
async def test_embed_query_returns_vector():
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1] * 1024]

    with patch("src.rag.embedder.get_cohere_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=mock_response)
        mock_get.return_value = mock_client

        from src.rag.embedder import embed_query
        result = await embed_query("test question")

    assert isinstance(result, list)
    assert len(result) == 1024
