from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class RawDocument:
    source_uri: str
    doc_type: str
    title: str | None
    pages: list[str]
    metadata: dict


def load_pdf(path: str) -> RawDocument:
    p = Path(path)
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    title = doc.metadata.get("title") or p.stem
    doc.close()
    return RawDocument(
        source_uri=str(p.resolve()),
        doc_type="pdf",
        title=title,
        pages=pages,
        metadata={"page_count": len(pages), "file_name": p.name},
    )
