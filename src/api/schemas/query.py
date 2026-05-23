import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    model: str | None = None
    prompt_version: str = "v1"


class SourceChunk(BaseModel):
    chunk_id: uuid.UUID
    document_title: str | None
    source_uri: str
    content_snippet: str
    similarity_score: float


class QueryResponse(BaseModel):
    answer: str
    query_log_id: uuid.UUID
    sources: list[SourceChunk]
    latency_ms: int
