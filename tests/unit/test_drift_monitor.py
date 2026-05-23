import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")


def _make_session(avg_latency):
    mock_row = MagicMock()
    mock_row.avg_latency = avg_latency
    mock_result = MagicMock()
    mock_result.one_or_none.return_value = mock_row
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    return mock_session


@pytest.mark.asyncio
async def test_check_drift_returns_no_data_when_empty():
    from src.mlops.drift_monitor import check_drift

    mock_result = MagicMock()
    mock_result.one_or_none.return_value = None
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result

    result = await check_drift(mock_session)
    assert result == {"status": "no_data"}


@pytest.mark.asyncio
async def test_check_drift_returns_ok_when_no_production_model():
    from src.mlops.drift_monitor import check_drift

    mock_session = _make_session(avg_latency=500.0)

    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = []

    with patch("src.mlops.drift_monitor.mlflow.tracking.MlflowClient", return_value=mock_client):
        result = await check_drift(mock_session)

    assert result["status"] == "ok"
    assert result["avg_latency_ms"] == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_check_drift_detects_latency_degradation():
    from src.mlops.drift_monitor import check_drift

    mock_session = _make_session(avg_latency=1200.0)

    mock_run = MagicMock()
    mock_run.run_id = "baseline-run-id"
    mock_run.data.metrics = {"avg_latency_ms": 1000.0}

    mock_version = MagicMock()
    mock_version.run_id = "baseline-run-id"

    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [mock_version]
    mock_client.get_run.return_value = mock_run

    with (
        patch("src.mlops.drift_monitor.mlflow.tracking.MlflowClient", return_value=mock_client),
        patch("src.mlops.drift_monitor.mlflow.set_tag"),
    ):
        result = await check_drift(mock_session)

    assert result["status"] == "drift_detected"
    assert result["drift_ratio"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_check_drift_handles_mlflow_exception():
    from src.mlops.drift_monitor import check_drift

    mock_session = _make_session(avg_latency=400.0)

    mock_client = MagicMock()
    mock_client.get_latest_versions.side_effect = Exception("MLflow unavailable")

    with patch("src.mlops.drift_monitor.mlflow.tracking.MlflowClient", return_value=mock_client):
        result = await check_drift(mock_session)

    assert result["status"] == "ok"
    assert result["avg_latency_ms"] == pytest.approx(400.0)
