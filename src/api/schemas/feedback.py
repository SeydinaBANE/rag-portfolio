import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    query_log_id: uuid.UUID
    rating: int = Field(..., description="1 = positive, -1 = negative")
    comment: str | None = None

    @classmethod
    def validate_rating(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError("rating must be 1 or -1")
        return v


class FeedbackResponse(BaseModel):
    id: uuid.UUID
    created_at: datetime
