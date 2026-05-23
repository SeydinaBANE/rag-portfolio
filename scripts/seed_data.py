"""Ingest sample documents for demo purposes."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import AsyncSessionLocal
from src.pipeline.ingestion_pipeline import SourceSpec, ingest

DEMO_SOURCES = [
    SourceSpec(uri="https://fastapi.tiangolo.com/", type="url"),
    SourceSpec(uri="https://docs.pydantic.dev/latest/", type="url"),
    SourceSpec(uri="https://python.langchain.com/docs/introduction/", type="url"),
]


async def main() -> None:
    print("Starting demo data ingestion...")
    async with AsyncSessionLocal() as session:
        result = await ingest(session, DEMO_SOURCES)

    print(f"✅ Ingested {result.documents_ingested} documents, {result.chunks_created} chunks")
    if result.errors:
        print("⚠️  Errors:")
        for e in result.errors:
            print(f"  - {e}")


if __name__ == "__main__":
    asyncio.run(main())
