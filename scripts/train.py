"""Train the RVC synthesizer.

    uv run python scripts/train.py

Training resumes automatically from the newest checkpoint in
config/train.yaml -> checkpoint.dir, so Ctrl-C is safe.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.training_service import TrainingService  # noqa: E402
from src.utils.config import load_model_config  # noqa: E402
from src.utils.config import load_train_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("train")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an RVC model")

    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument("--train-config", default="config/train.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true")

    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    train_config = load_train_config(args.train_config)

    if args.epochs is not None:
        train_config.training.epochs = args.epochs

    if args.batch_size is not None:
        train_config.training.batch_size = args.batch_size

    if args.no_fp16:
        train_config.training.fp16 = False

    TrainingService(model_config, train_config, device=args.device).run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
