import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import QueryLog


async def insert_query_log(session: AsyncSession, log: QueryLog) -> QueryLog:
    session.add(log)
    await session.flush()
    await session.refresh(log)
    return log


async def get_recent_logs(session: AsyncSession, limit: int = 100) -> list[QueryLog]:
    result = await session.execute(
        select(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit)
    )
    return list(result.scalars())


async def get_log_by_id(session: AsyncSession, log_id: uuid.UUID) -> QueryLog | None:
    result = await session.execute(select(QueryLog).where(QueryLog.id == log_id))
    return result.scalar_one_or_none()
