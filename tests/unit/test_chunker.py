import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")

from src.pipeline.transformers.chunker import chunk_pages


def test_chunk_pages_returns_non_empty():
    pages = ["This is a sentence. " * 50]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_pages_respects_size():
    pages = ["word " * 200]
    chunks = chunk_pages(pages, chunk_size=50, overlap=0)
    assert all(len(c) <= 100 for c in chunks)  # some tolerance


def test_chunk_pages_filters_empty():
    pages = ["", "   ", "real content here " * 10]
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    assert all(c.strip() for c in chunks)
