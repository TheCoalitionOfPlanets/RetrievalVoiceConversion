"""Run preprocess -> index -> train in one go.

    uv run python scripts/run_pipeline.py --raw data/raw
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.pipeline_service import PipelineService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Full RVC pipeline")

    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument("--train-config", default="config/train.yaml")

    args = parser.parse_args()

    PipelineService(
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        device=args.device,
    ).run_all(args.raw)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
