import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")

from src.pipeline.transformers.metadata_enricher import enrich


def test_enrich_adds_required_fields():
    result = enrich(
        base_metadata={"file_name": "doc.pdf"},
        source_uri="https://example.com",
        doc_type="pdf",
        chunk_index=2,
        total_chunks=10,
    )
    assert result["source_uri"] == "https://example.com"
    assert result["doc_type"] == "pdf"
    assert result["chunk_index"] == 2
    assert result["total_chunks"] == 10
    assert "ingested_at" in result


def test_enrich_preserves_base_metadata():
    result = enrich(
        base_metadata={"page_count": 5, "file_name": "test.pdf"},
        source_uri="s3://bucket/file.pdf",
        doc_type="pdf",
        chunk_index=0,
        total_chunks=3,
    )
    assert result["page_count"] == 5
    assert result["file_name"] == "test.pdf"


def test_enrich_ingested_at_is_iso_format():
    result = enrich({}, "uri", "url", 0, 1)
    from datetime import datetime

    datetime.fromisoformat(result["ingested_at"])
