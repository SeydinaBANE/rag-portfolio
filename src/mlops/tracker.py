from collections.abc import Generator
from contextlib import contextmanager

import mlflow
from src.config import settings


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


@contextmanager
def rag_run(run_name: str | None = None) -> Generator[mlflow.ActiveRun, None, None]:
    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        yield run
