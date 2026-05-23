"""Evaluate RAG quality against golden_qa.jsonl using LLM-as-judge."""

import asyncio
import json
from pathlib import Path
from typing import Any

import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.rag.chain import run_rag

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator. Rate the answer on a scale from 0 to 1 based on how well it answers the question using the provided context. Respond with ONLY a number between 0 and 1.",
        ),
        (
            "human",
            "Question: {question}\n\nAnswer: {answer}\n\nContext used: {context}\n\nScore (0-1):",
        ),
    ]
)


async def _judge_relevancy(question: str, answer: str, context: str) -> float:
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=SecretStr(settings.openrouter_api_key),
        base_url=settings.openrouter_base_url,
        temperature=0,
        default_headers={"HTTP-Referer": "https://github.com/SeydinaBANE/rag-portfolio"},
    )
    chain = JUDGE_PROMPT | llm | StrOutputParser()
    raw = await chain.ainvoke({"question": question, "answer": answer, "context": context})
    try:
        return max(0.0, min(1.0, float(raw.strip())))
    except ValueError:
        return 0.0


async def run_evaluation(
    session: AsyncSession,
    golden_path: str = "data/eval/golden_qa.jsonl",
    prompt_version: str = "v1",
) -> dict[str, Any]:
    path = Path(golden_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden QA file not found: {path}")

    questions = []
    with path.open() as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    scores: list[float] = []
    latencies: list[int] = []
    results = []

    for qa in questions:
        question = qa["question"]
        resp = await run_rag(session, question, prompt_version=prompt_version)
        context = "\n".join(c.content for c in resp.sources)
        score = await _judge_relevancy(question, resp.answer, context)
        scores.append(score)
        latencies.append(resp.latency_ms)
        results.append(
            {
                "question": question,
                "answer": resp.answer,
                "relevancy_score": score,
                "latency_ms": resp.latency_ms,
            }
        )

    avg_relevancy = sum(scores) / len(scores) if scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    metrics = {
        "avg_answer_relevancy": avg_relevancy,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": float(p95_latency),
        "num_questions": len(questions),
    }

    await asyncio.to_thread(mlflow.log_metrics, metrics)
    return {"metrics": metrics, "results": results}
