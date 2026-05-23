from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_session
from src.api.schemas.feedback import FeedbackRequest, FeedbackResponse
from src.db.models import Feedback
from src.db.repositories.query_log_repo import get_log_by_id

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    body: FeedbackRequest,
    session: AsyncSession = Depends(get_session),
) -> FeedbackResponse:
    if body.rating not in (1, -1):
        raise HTTPException(status_code=422, detail="rating must be 1 or -1")

    log = await get_log_by_id(session, body.query_log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Query log not found")

    feedback = Feedback(
        query_log_id=body.query_log_id,
        rating=body.rating,
        comment=body.comment,
    )
    session.add(feedback)
    await session.commit()
    await session.refresh(feedback)
    return FeedbackResponse(id=feedback.id, created_at=feedback.created_at)
