from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db.repositories.chunk_repo import search_similar
from src.rag.embedder import embed_query


@dataclass
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    similarity_score: float
    source_uri: str
    document_title: str | None
    metadata: dict


async def retrieve(
    session: AsyncSession,
    question: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    k = top_k or settings.top_k_retrieval
    question_embedding = await embed_query(question)
    results = await search_similar(session, question_embedding, top_k=k)

    chunks: list[RetrievedChunk] = []
    for chunk, score in results:
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                similarity_score=score,
                source_uri=chunk.document.source_uri if chunk.document else "",
                document_title=chunk.document.title if chunk.document else None,
                metadata=chunk.metadata_,
            )
        )
    return chunks
