from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_session
from src.api.schemas.query import QueryRequest, QueryResponse, SourceChunk
from src.rag.chain import run_rag

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
) -> QueryResponse:
    response = await run_rag(
        session=session,
        question=body.question,
        top_k=body.top_k,
        model=body.model,
        prompt_version=body.prompt_version,
    )
    sources = [
        SourceChunk(
            chunk_id=c.chunk_id,
            document_title=c.document_title,
            source_uri=c.source_uri,
            content_snippet=c.content[:300],
            similarity_score=round(c.similarity_score, 4),
        )
        for c in response.sources
    ]
    return QueryResponse(
        answer=response.answer,
        query_log_id=response.query_log_id,
        sources=sources,
        latency_ms=response.latency_ms,
    )
