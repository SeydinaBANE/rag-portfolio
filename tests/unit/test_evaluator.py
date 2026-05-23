import json
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")

from src.rag.chain import RAGResponse
from src.rag.retriever import RetrievedChunk


def _make_qa_file(tmp_path: Path, n: int = 3) -> Path:
    p = tmp_path / "golden_qa.jsonl"
    lines = [
        json.dumps({"question": f"Question {i}?", "reference": f"Answer {i}."}) for i in range(n)
    ]
    p.write_text("\n".join(lines))
    return p


def _make_rag_response() -> RAGResponse:
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Relevant content.",
        similarity_score=0.88,
        source_uri="https://docs.example.com",
        document_title="Example Doc",
        metadata={},
    )
    return RAGResponse(
        answer="This is the answer.",
        query_log_id=uuid.uuid4(),
        sources=[chunk],
        latency_ms=200,
    )


@pytest.mark.asyncio
async def test_run_evaluation_returns_metrics(tmp_path: Path):
    from src.mlops.evaluator import run_evaluation

    qa_file = _make_qa_file(tmp_path, n=2)
    mock_session = AsyncMock()

    with (
        patch("src.mlops.evaluator.run_rag", new=AsyncMock(return_value=_make_rag_response())),
        patch("src.mlops.evaluator._judge_relevancy", new=AsyncMock(return_value=0.8)),
        patch("asyncio.to_thread", new=AsyncMock(return_value=None)),
    ):
        result = await run_evaluation(mock_session, golden_path=str(qa_file))

    assert "metrics" in result
    assert "results" in result
    assert result["metrics"]["num_questions"] == 2
    assert result["metrics"]["avg_answer_relevancy"] == pytest.approx(0.8)
    assert len(result["results"]) == 2


@pytest.mark.asyncio
async def test_run_evaluation_raises_when_file_missing():
    from src.mlops.evaluator import run_evaluation

    mock_session = AsyncMock()
    with pytest.raises(FileNotFoundError):
        await run_evaluation(mock_session, golden_path="/nonexistent/path.jsonl")


@pytest.mark.asyncio
async def test_run_evaluation_p95_latency(tmp_path: Path):
    from src.mlops.evaluator import run_evaluation

    qa_file = _make_qa_file(tmp_path, n=3)
    mock_session = AsyncMock()

    calls = [100, 200, 300]
    responses = [
        RAGResponse(
            answer="ok",
            query_log_id=uuid.uuid4(),
            sources=[],
            latency_ms=ms,
        )
        for ms in calls
    ]
    mock_rag = AsyncMock(side_effect=responses)

    with (
        patch("src.mlops.evaluator.run_rag", new=mock_rag),
        patch("src.mlops.evaluator._judge_relevancy", new=AsyncMock(return_value=0.5)),
        patch("asyncio.to_thread", new=AsyncMock(return_value=None)),
    ):
        result = await run_evaluation(mock_session, golden_path=str(qa_file))

    assert result["metrics"]["avg_latency_ms"] == pytest.approx(200.0)
