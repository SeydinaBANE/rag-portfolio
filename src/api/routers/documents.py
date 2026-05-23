import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_session
from src.api.schemas.documents import DocumentListResponse, DocumentOut
from src.db.models import Document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    session: AsyncSession = Depends(get_session),
) -> DocumentListResponse:
    offset = (page - 1) * page_size
    total_result = await session.execute(select(func.count()).select_from(Document))
    total = total_result.scalar_one()

    result = await session.execute(
        select(Document).order_by(Document.ingested_at.desc()).offset(offset).limit(page_size)
    )
    docs = list(result.scalars())
    return DocumentListResponse(
        items=[
            DocumentOut(
                id=d.id,
                source_uri=d.source_uri,
                doc_type=d.doc_type,
                title=d.title,
                ingested_at=d.ingested_at,
                chunk_count=d.chunk_count,
            )
            for d in docs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    result = await session.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    await session.delete(doc)
    await session.commit()
