"""Build the FAISS retrieval index from extracted features.

    uv run python scripts/build_index.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.pipeline_service import PipelineService  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("build_index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RVC FAISS index")

    parser.add_argument("--processed", default=None)
    parser.add_argument("--index", default=None, help="Output .index path")
    parser.add_argument("--features", default=None, help="Output .npy bank path")
    parser.add_argument("--max-vectors", type=int, default=400_000)
    parser.add_argument("--model-config", default="config/model.yaml")

    args = parser.parse_args()

    pipeline = PipelineService(model_config_path=args.model_config, device="cpu")

    total = pipeline.build_index(
        processed_dir=args.processed,
        index_path=args.index,
        features_path=args.features,
        max_vectors=args.max_vectors,
    )

    logger.info(f"Index built with {total} vectors")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
