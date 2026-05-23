import json
from pathlib import Path

from src.pipeline.loaders.pdf_loader import RawDocument


def load_jsonl(path: str, text_field: str = "text", title_field: str = "title") -> RawDocument:
    p = Path(path)
    pages: list[str] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pages.append(record.get(text_field, ""))

    return RawDocument(
        source_uri=str(p.resolve()),
        doc_type="jsonl",
        title=p.stem,
        pages=pages,
        metadata={"file_name": p.name, "record_count": len(pages)},
    )
