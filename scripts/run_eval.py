"""Run RAG evaluation against golden QA pairs and log results to MLflow."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from src.config import settings
from src.db.connection import AsyncSessionLocal
from src.mlops.evaluator import run_evaluation
from src.mlops.tracker import setup_mlflow


async def main() -> None:
    setup_mlflow()
    print(f"Running evaluation — experiment: {settings.mlflow_experiment_name}")

    with mlflow.start_run(run_name="eval-run"):
        async with AsyncSessionLocal() as session:
            result = await run_evaluation(session)

    metrics = result["metrics"]
    print("\n📊 Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Write results to file for CI badge
    out_path = Path("data/eval/last_eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")

    if metrics.get("avg_answer_relevancy", 0) < 0.60:
        print("❌ Quality gate failed: avg_answer_relevancy < 0.60")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
