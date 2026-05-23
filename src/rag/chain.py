import time
from dataclasses import dataclass
from uuid import UUID

import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db.models import QueryLog
from src.db.repositories.query_log_repo import insert_query_log
from src.rag.prompt_templates import DEFAULT_VERSION, get_prompt
from src.rag.retriever import RetrievedChunk, retrieve


@dataclass
class RAGResponse:
    answer: str
    query_log_id: UUID
    sources: list[RetrievedChunk]
    latency_ms: int


def _get_llm(model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.chat_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0,
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.document_title or c.source_uri
        parts.append(f"[{i}] Source: {source}\n{c.content}")
    return "\n\n---\n\n".join(parts)


async def run_rag(
    session: AsyncSession,
    question: str,
    top_k: int | None = None,
    model: str | None = None,
    prompt_version: str = DEFAULT_VERSION,
) -> RAGResponse:
    model_name = model or settings.chat_model
    start = time.perf_counter()

    with mlflow.start_run(nested=True) as run:
        mlflow.log_params({
            "model_name": model_name,
            "prompt_version": prompt_version,
            "top_k": top_k or settings.top_k_retrieval,
            "chunk_size": settings.chunk_size,
            "embedding_model": settings.embedding_model,
        })

        chunks = await retrieve(session, question, top_k=top_k)
        context = _format_context(chunks)
        prompt = get_prompt(prompt_version)
        llm = _get_llm(model_name)
        chain = prompt | llm | StrOutputParser()

        answer = await chain.ainvoke({"context": context, "question": question})
        latency_ms = int((time.perf_counter() - start) * 1000)

        scores = [c.similarity_score for c in chunks]
        mlflow.log_metrics({
            "latency_ms": latency_ms,
            "retrieval_score_mean": sum(scores) / len(scores) if scores else 0.0,
            "retrieval_score_max": max(scores) if scores else 0.0,
            "num_chunks_retrieved": len(chunks),
        })

        log = QueryLog(
            question=question,
            answer=answer,
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
            retrieval_scores=scores,
            latency_ms=latency_ms,
            model_name=model_name,
            prompt_version=prompt_version,
            mlflow_run_id=run.info.run_id,
        )
        log = await insert_query_log(session, log)
        await session.commit()

    return RAGResponse(
        answer=answer,
        query_log_id=log.id,
        sources=chunks,
        latency_ms=latency_ms,
    )
