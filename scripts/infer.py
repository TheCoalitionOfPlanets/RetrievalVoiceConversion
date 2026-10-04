"""Convert audio with a trained RVC model.

    uv run python scripts/infer.py --input song.wav --output out.wav --pitch 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.conversion_service import ConversionService  # noqa: E402
from src.utils.config import load_inference_config  # noqa: E402
from src.utils.config import load_model_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("infer")


def main() -> int:
    parser = argparse.ArgumentParser(description="RVC inference")

    parser.add_argument("--input", required=True, help="Audio file or directory")
    parser.add_argument("--output", required=True, help="Output file or directory")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument(
        "--pitch", type=int, default=0, help="Transpose in semitones"
    )
    parser.add_argument("--index-rate", type=float, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--f0-method", default=None, choices=[None, "harvest", "dio", "crepe"])
    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument("--inference-config", default="config/inference.yaml")
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    inference_config = load_inference_config(args.inference_config)

    if args.checkpoint:
        inference_config.checkpoint = args.checkpoint

    if args.index:
        inference_config.index.path = args.index

    if args.f0_method:
        inference_config.conversion.f0_method = args.f0_method

    if args.device:
        inference_config.device = args.device

    service = ConversionService(
        load_model_config(args.model_config), inference_config
    )

    if Path(args.input).is_dir():
        results = service.convert_directory(
            args.input,
            args.output,
            speaker_id=args.speaker_id,
            pitch_shift=args.pitch,
        )
        logger.info(f"Converted {len(results)} files")
        return 0

    result = service.convert_file(
        args.input,
        args.output,
        speaker_id=args.speaker_id,
        pitch_shift=args.pitch,
        index_rate=args.index_rate,
    )

    logger.info(
        f"Wrote {result.output_path} "
        f"(RTF {result.realtime_factor:.2f})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
