import asyncio
import os
import time
from dataclasses import dataclass
from uuid import UUID

import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db.models import QueryLog
from src.db.repositories.query_log_repo import insert_query_log
from src.mlops.tracker import setup_mlflow
from src.rag.prompt_templates import DEFAULT_VERSION, get_prompt
from src.rag.retriever import RetrievedChunk, retrieve

# langchain-openai reads OPENAI_API_KEY for the Bearer token
os.environ["OPENAI_API_KEY"] = settings.openrouter_api_key


@dataclass
class RAGResponse:
    answer: str
    query_log_id: UUID
    sources: list[RetrievedChunk]
    latency_ms: int


def _get_llm(model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=SecretStr(settings.openrouter_api_key),
        base_url=settings.openrouter_base_url,
        temperature=0,
        default_headers={"HTTP-Referer": "https://github.com/SeydinaBANE/rag-portfolio"},
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.document_title or c.source_uri
        parts.append(f"[{i}] Source: {source}\n{c.content}")
    return "\n\n---\n\n".join(parts)


def _log_to_mlflow(
    model_name: str,
    prompt_version: str,
    top_k: int,
    latency_ms: int,
    scores: list[float],
    run_id_holder: list[str],
) -> None:
    """Synchronous MLflow logging — run via asyncio.to_thread."""
    setup_mlflow()
    with mlflow.start_run(nested=True) as run:
        run_id_holder.append(run.info.run_id)
        mlflow.log_params(
            {
                "model_name": model_name,
                "prompt_version": prompt_version,
                "top_k": top_k,
                "chunk_size": settings.chunk_size,
                "embedding_model": settings.embedding_model,
            }
        )
        mlflow.log_metrics(
            {
                "latency_ms": latency_ms,
                "retrieval_score_mean": sum(scores) / len(scores) if scores else 0.0,
                "retrieval_score_max": max(scores) if scores else 0.0,
                "num_chunks_retrieved": len(scores),
            }
        )


async def run_rag(
    session: AsyncSession,
    question: str,
    top_k: int | None = None,
    model: str | None = None,
    prompt_version: str = DEFAULT_VERSION,
) -> RAGResponse:
    model_name = model or settings.chat_model
    k = top_k or settings.top_k_retrieval
    start = time.perf_counter()

    # Async: retrieve + generate
    chunks = await retrieve(session, question, top_k=k)
    context = _format_context(chunks)
    prompt = get_prompt(prompt_version)
    llm = _get_llm(model_name)
    chain = prompt | llm | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "question": question})
    latency_ms = int((time.perf_counter() - start) * 1000)

    scores = [c.similarity_score for c in chunks]

    # Sync: MLflow logging in a thread (avoids greenlet conflict with asyncpg)
    run_id_holder: list[str] = []
    await asyncio.to_thread(
        _log_to_mlflow, model_name, prompt_version, k, latency_ms, scores, run_id_holder
    )
    mlflow_run_id = run_id_holder[0] if run_id_holder else None

    # Async: persist query log to DB
    log = QueryLog(
        question=question,
        answer=answer,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        retrieval_scores=scores,
        latency_ms=latency_ms,
        model_name=model_name,
        prompt_version=prompt_version,
        mlflow_run_id=mlflow_run_id,
    )
    log = await insert_query_log(session, log)
    await session.commit()

    return RAGResponse(
        answer=answer,
        query_log_id=log.id,
        sources=chunks,
        latency_ms=latency_ms,
    )
