"""Slice the raw dataset and extract HuBERT + F0 features.

    uv run python scripts/preprocess.py --raw data/raw
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.pipeline_service import PipelineService  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("preprocess")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess an RVC dataset")

    parser.add_argument("--raw", default="data/raw", help="Raw audio root")
    parser.add_argument("--processed", default=None, help="Output directory")
    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument("--train-config", default="config/train.yaml")
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    pipeline = PipelineService(
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        device=args.device,
    )

    count = pipeline.preprocess(args.raw, args.processed)

    if count == 0:
        logger.error("No segments produced — check that --raw contains audio")
        return 1

    logger.info(f"Done: {count} segments")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
