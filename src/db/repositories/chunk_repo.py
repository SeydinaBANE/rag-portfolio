import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Chunk, Document


async def insert_chunks(session: AsyncSession, chunks: list[Chunk]) -> None:
    session.add_all(chunks)
    await session.flush()


async def search_similar(
    session: AsyncSession, embedding: list[float], top_k: int = 5
) -> list[tuple[Chunk, float]]:
    """Return (chunk, cosine_similarity) pairs ordered by similarity desc."""
    vector_literal = f"[{','.join(str(v) for v in embedding)}]"
    stmt = (
        select(
            Chunk,
            text(f"1 - (embedding <=> '{vector_literal}'::vector) AS similarity"),
        )
        .where(Chunk.embedding.is_not(None))
        .order_by(text(f"embedding <=> '{vector_literal}'::vector"))
        .limit(top_k)
    )
    result = await session.execute(stmt)
    return [(row[0], float(row[1])) for row in result]


async def get_document_by_uri(session: AsyncSession, source_uri: str) -> Document | None:
    result = await session.execute(select(Document).where(Document.source_uri == source_uri))
    return result.scalar_one_or_none()


async def get_chunk_by_id(session: AsyncSession, chunk_id: uuid.UUID) -> Chunk | None:
    result = await session.execute(select(Chunk).where(Chunk.id == chunk_id))
    return result.scalar_one_or_none()
