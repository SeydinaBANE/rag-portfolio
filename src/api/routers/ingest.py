import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_session
from src.api.schemas.ingest import IngestRequest, IngestStatus
from src.pipeline.ingestion_pipeline import SourceSpec, ingest

router = APIRouter(prefix="/ingest", tags=["ingest"])

# In-memory store for demo purposes (use Redis/DB for production)
_jobs: dict[str, dict[str, Any]] = {}


async def _run_ingest(job_id: str, sources: list[SourceSpec], session: AsyncSession) -> None:
    _jobs[job_id] = {
        "status": "processing",
        "documents_ingested": 0,
        "chunks_created": 0,
        "errors": [],
    }
    result = await ingest(session, sources, job_id)
    _jobs[job_id] = {
        "status": result.status,
        "documents_ingested": result.documents_ingested,
        "chunks_created": result.chunks_created,
        "errors": result.errors,
    }


@router.post("", response_model=IngestStatus, status_code=202)
async def start_ingest(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> IngestStatus:
    job_id = str(uuid.uuid4())
    specs = [
        SourceSpec(uri=s.uri, type=s.type, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
        for s in body.sources
    ]
    background_tasks.add_task(_run_ingest, job_id, specs, session)
    _jobs[job_id] = {
        "status": "processing",
        "documents_ingested": 0,
        "chunks_created": 0,
        "errors": [],
    }
    return IngestStatus(
        job_id=job_id, status="processing", documents_ingested=0, chunks_created=0, errors=[]
    )


@router.get("/status/{job_id}", response_model=IngestStatus)
async def ingest_status(job_id: str) -> IngestStatus:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return IngestStatus(job_id=job_id, **job)
