import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Chunk, Document
from src.db.repositories.chunk_repo import get_document_by_uri, insert_chunks
from src.pipeline.loaders.jsonl_loader import load_jsonl
from src.pipeline.loaders.pdf_loader import load_pdf
from src.pipeline.loaders.url_loader import load_url
from src.pipeline.transformers.chunker import chunk_pages
from src.pipeline.transformers.cleaner import clean_pages
from src.pipeline.transformers.metadata_enricher import enrich
from src.rag.embedder import embed_texts


@dataclass
class SourceSpec:
    uri: str
    type: str  # "pdf" | "url" | "jsonl"
    chunk_size: int | None = None
    chunk_overlap: int | None = None


@dataclass
class IngestionResult:
    job_id: str
    documents_ingested: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)
    status: str = "completed"


async def ingest(
    session: AsyncSession,
    sources: list[SourceSpec],
    job_id: str | None = None,
) -> IngestionResult:
    result = IngestionResult(job_id=job_id or str(uuid.uuid4()))

    for spec in sources:
        try:
            # Load
            if spec.type == "pdf":
                raw = load_pdf(spec.uri)
            elif spec.type == "url":
                raw = await load_url(spec.uri)
            elif spec.type == "jsonl":
                raw = load_jsonl(spec.uri)
            else:
                result.errors.append(f"Unknown type '{spec.type}' for {spec.uri}")
                continue

            # Check for existing document (idempotent)
            existing = await get_document_by_uri(session, spec.uri)
            if existing:
                result.errors.append(f"Already ingested: {spec.uri}")
                continue

            # Clean + chunk
            cleaned_pages = clean_pages(raw.pages)
            text_chunks = chunk_pages(cleaned_pages, spec.chunk_size, spec.chunk_overlap)
            if not text_chunks:
                result.errors.append(f"No content extracted from {spec.uri}")
                continue

            # Embed
            embeddings = await embed_texts(text_chunks)

            # Persist document
            doc = Document(
                source_uri=spec.uri,
                doc_type=spec.type,
                title=raw.title,
                chunk_count=len(text_chunks),
                metadata_=raw.metadata,
            )
            session.add(doc)
            await session.flush()  # get doc.id

            # Persist chunks
            chunks = [
                Chunk(
                    document_id=doc.id,
                    content=text_chunks[i],
                    embedding=embeddings[i],
                    chunk_index=i,
                    token_count=len(text_chunks[i].split()),
                    metadata_=enrich(raw.metadata, spec.uri, spec.type, i, len(text_chunks)),
                )
                for i in range(len(text_chunks))
            ]
            await insert_chunks(session, chunks)
            await session.commit()

            result.documents_ingested += 1
            result.chunks_created += len(chunks)

        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"{spec.uri}: {exc}")
            await session.rollback()

    if result.errors and result.documents_ingested == 0:
        result.status = "failed"
    elif result.errors:
        result.status = "partial"

    return result
