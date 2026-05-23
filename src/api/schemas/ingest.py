from pydantic import BaseModel, Field


class SourceSpec(BaseModel):
    uri: str
    type: str = Field(..., pattern="^(pdf|url|jsonl)$")
    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=500)


class IngestRequest(BaseModel):
    sources: list[SourceSpec] = Field(..., min_length=1)


class IngestStatus(BaseModel):
    job_id: str
    status: str
    documents_ingested: int
    chunks_created: int
    errors: list[str]
