import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")


# ── JSONL loader ──────────────────────────────────────────────────────────────


def test_load_jsonl_reads_all_lines(tmp_path: Path):
    from src.pipeline.loaders.jsonl_loader import load_jsonl

    data = [
        {"text": "First document text.", "title": "Doc 1"},
        {"text": "Second document text.", "title": "Doc 2"},
    ]
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in data))

    result = load_jsonl(str(p))
    assert result.doc_type == "jsonl"
    assert len(result.pages) == 2
    assert result.pages[0] == "First document text."
    assert result.pages[1] == "Second document text."


def test_load_jsonl_skips_empty_lines(tmp_path: Path):
    from src.pipeline.loaders.jsonl_loader import load_jsonl

    p = tmp_path / "data.jsonl"
    p.write_text('{"text": "hello"}\n\n{"text": "world"}\n')

    result = load_jsonl(str(p))
    assert len(result.pages) == 2


def test_load_jsonl_uses_custom_field(tmp_path: Path):
    from src.pipeline.loaders.jsonl_loader import load_jsonl

    p = tmp_path / "data.jsonl"
    p.write_text('{"content": "custom field"}\n')

    result = load_jsonl(str(p), text_field="content")
    assert result.pages[0] == "custom field"


def test_load_jsonl_metadata_has_record_count(tmp_path: Path):
    from src.pipeline.loaders.jsonl_loader import load_jsonl

    p = tmp_path / "data.jsonl"
    p.write_text('{"text": "a"}\n{"text": "b"}\n')

    result = load_jsonl(str(p))
    assert result.metadata["record_count"] == 2


# ── PDF loader ────────────────────────────────────────────────────────────────


def test_load_pdf_returns_raw_document():
    from src.pipeline.loaders.pdf_loader import load_pdf

    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page content here."

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.metadata = {"title": "Test PDF"}
    mock_doc.__len__ = MagicMock(return_value=1)

    with patch("src.pipeline.loaders.pdf_loader.fitz.open", return_value=mock_doc):
        result = load_pdf("/fake/path/document.pdf")

    assert result.doc_type == "pdf"
    assert result.title == "Test PDF"
    assert "Page content here." in result.pages
    assert result.metadata["page_count"] == 1


def test_load_pdf_uses_stem_when_no_title():
    from src.pipeline.loaders.pdf_loader import load_pdf

    mock_page = MagicMock()
    mock_page.get_text.return_value = "Content."

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.metadata = {}
    mock_doc.__len__ = MagicMock(return_value=1)

    with patch("src.pipeline.loaders.pdf_loader.fitz.open", return_value=mock_doc):
        result = load_pdf("/fake/path/myreport.pdf")

    assert result.title == "myreport"


# ── URL loader ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_url_strips_script_and_nav():
    from src.pipeline.loaders.url_loader import load_url

    html = """<html><head><title>Test Page</title></head><body>
    <nav>Navigation</nav>
    <script>alert('hello');</script>
    <p>Main content here.</p>
    <footer>Footer text</footer>
    </body></html>"""

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("src.pipeline.loaders.url_loader.httpx.AsyncClient", return_value=mock_client):
        result = await load_url("https://example.com")

    assert result.doc_type == "url"
    assert result.title == "Test Page"
    assert "Navigation" not in result.pages[0]
    assert "alert" not in result.pages[0]
    assert "Footer text" not in result.pages[0]
    assert "Main content here." in result.pages[0]


@pytest.mark.asyncio
async def test_load_url_sets_source_uri():
    from src.pipeline.loaders.url_loader import load_url

    html = "<html><head><title>T</title></head><body><p>content</p></body></html>"

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("src.pipeline.loaders.url_loader.httpx.AsyncClient", return_value=mock_client):
        result = await load_url("https://mysite.com/page")

    assert result.source_uri == "https://mysite.com/page"
