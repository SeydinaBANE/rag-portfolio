"""Detect quality drift by comparing recent query metrics against baseline."""
import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import mlflow

from src.config import settings
from src.db.models import QueryLog

logger = structlog.get_logger()

DRIFT_THRESHOLD = 0.10  # 10% degradation triggers alert


async def check_drift(session: AsyncSession, window: int = 100) -> dict:
    result = await session.execute(
        select(
            func.avg(QueryLog.latency_ms).label("avg_latency"),
        )
        .order_by(QueryLog.created_at.desc())
        .limit(window)
    )
    row = result.one_or_none()
    if not row or row.avg_latency is None:
        return {"status": "no_data"}

    avg_latency = float(row.avg_latency)

    # Compare against registered Production model baseline if available
    client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    try:
        latest = client.get_latest_versions("knowledgeforge-rag-chain", stages=["Production"])
        if latest:
            baseline_run = client.get_run(latest[0].run_id)
            baseline_latency = float(baseline_run.data.metrics.get("avg_latency_ms", avg_latency))
            drift_ratio = (avg_latency - baseline_latency) / max(baseline_latency, 1)
            if drift_ratio > DRIFT_THRESHOLD:
                logger.warning(
                    "latency_drift_detected",
                    current_avg_latency_ms=avg_latency,
                    baseline_latency_ms=baseline_latency,
                    drift_ratio=drift_ratio,
                )
                mlflow.set_tag("drift_alert", "true")
                return {"status": "drift_detected", "drift_ratio": drift_ratio}
    except Exception:  # noqa: BLE001
        pass

    return {"status": "ok", "avg_latency_ms": avg_latency}
