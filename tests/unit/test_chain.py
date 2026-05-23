import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")

from src.rag.chain import RAGResponse, _format_context
from src.rag.retriever import RetrievedChunk


def _make_chunk(**kwargs) -> RetrievedChunk:
    return RetrievedChunk(
        **{
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "content": "Test content about Python.",
            "similarity_score": 0.9,
            "source_uri": "https://example.com",
            "document_title": "Python Docs",
            "metadata": {},
            **kwargs,
        }
    )


def test_format_context_uses_document_title():
    chunk = _make_chunk(content="Hello world", document_title="My Doc")
    result = _format_context([chunk])
    assert "[1] Source: My Doc" in result
    assert "Hello world" in result


def test_format_context_falls_back_to_source_uri():
    chunk = _make_chunk(document_title=None, source_uri="https://fallback.com")
    result = _format_context([chunk])
    assert "https://fallback.com" in result


def test_format_context_separates_multiple_chunks():
    chunks = [_make_chunk(document_title=f"Doc{i}", content=f"Content {i}") for i in range(3)]
    result = _format_context(chunks)
    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result
    assert "---" in result


def test_format_context_empty_returns_empty_string():
    assert _format_context([]) == ""


@pytest.mark.asyncio
async def test_run_rag_returns_rag_response():
    from src.db.models import QueryLog
    from src.rag.chain import run_rag

    fake_chunk = _make_chunk()
    fake_log = MagicMock(spec=QueryLog)
    fake_log.id = uuid.uuid4()

    mock_session = AsyncMock()

    with (
        patch("src.rag.chain.retrieve", new=AsyncMock(return_value=[fake_chunk])),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new=AsyncMock(return_value="test answer"),
        ),
        patch("asyncio.to_thread", new=AsyncMock(return_value=None)),
        patch("src.rag.chain.insert_query_log", new=AsyncMock(return_value=fake_log)),
    ):
        result = await run_rag(mock_session, "What is Python?")

    assert isinstance(result, RAGResponse)
    assert result.answer == "test answer"
    assert result.sources == [fake_chunk]
    assert isinstance(result.latency_ms, int)
    assert result.query_log_id == fake_log.id


@pytest.mark.asyncio
async def test_run_rag_passes_top_k_to_retrieve():
    from src.db.models import QueryLog
    from src.rag.chain import run_rag

    mock_retrieve = AsyncMock(return_value=[])
    fake_log = MagicMock(spec=QueryLog)
    fake_log.id = uuid.uuid4()
    mock_session = AsyncMock()

    with (
        patch("src.rag.chain.retrieve", new=mock_retrieve),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new=AsyncMock(return_value="ok"),
        ),
        patch("asyncio.to_thread", new=AsyncMock(return_value=None)),
        patch("src.rag.chain.insert_query_log", new=AsyncMock(return_value=fake_log)),
    ):
        await run_rag(mock_session, "test question", top_k=3)

    mock_retrieve.assert_called_once_with(mock_session, "test question", top_k=3)


@pytest.mark.asyncio
async def test_run_rag_uses_custom_prompt_version():
    from src.db.models import QueryLog
    from src.rag.chain import run_rag

    fake_log = MagicMock(spec=QueryLog)
    fake_log.id = uuid.uuid4()
    mock_session = AsyncMock()
    mock_insert = AsyncMock(return_value=fake_log)

    with (
        patch("src.rag.chain.retrieve", new=AsyncMock(return_value=[])),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new=AsyncMock(return_value="v2 answer"),
        ),
        patch("asyncio.to_thread", new=AsyncMock(return_value=None)),
        patch("src.rag.chain.insert_query_log", new=mock_insert),
    ):
        await run_rag(mock_session, "test?", prompt_version="v2")

    call_args = mock_insert.call_args
    log_obj = call_args[0][1]
    assert log_obj.prompt_version == "v2"
