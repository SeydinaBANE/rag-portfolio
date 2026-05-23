import uuid
from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: uuid.UUID
    source_uri: str
    doc_type: str
    title: str | None
    ingested_at: datetime
    chunk_count: int


class DocumentListResponse(BaseModel):
    items: list[DocumentOut]
    total: int
    page: int
    page_size: int
