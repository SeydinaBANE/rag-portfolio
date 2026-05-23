import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://kg_user:testpassword@localhost:5432/knowledgeforge_test")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")

from src.pipeline.ingestion_pipeline import SourceSpec, ingest


@pytest.mark.asyncio
async def test_ingest_jsonl(session, tmp_path):
    jsonl_file = tmp_path / "test.jsonl"
    jsonl_file.write_text('{"text": "Python is a programming language."}\n' * 5)

    fake_embeddings = [[0.1] * 1536] * 10  # enough for any chunk count

    with patch("src.pipeline.ingestion_pipeline.embed_texts", new=AsyncMock(return_value=fake_embeddings)):
        result = await ingest(session, [SourceSpec(uri=str(jsonl_file), type="jsonl")])

    assert result.documents_ingested == 1
    assert result.chunks_created > 0
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_ingest_duplicate_skipped(session, tmp_path):
    jsonl_file = tmp_path / "dup.jsonl"
    jsonl_file.write_text('{"text": "Duplicate content test."}\n' * 3)
    fake_embeddings = [[0.1] * 1536] * 10

    with patch("src.pipeline.ingestion_pipeline.embed_texts", new=AsyncMock(return_value=fake_embeddings)):
        await ingest(session, [SourceSpec(uri=str(jsonl_file), type="jsonl")])
        result = await ingest(session, [SourceSpec(uri=str(jsonl_file), type="jsonl")])

    assert result.documents_ingested == 0
    assert "Already ingested" in result.errors[0]
