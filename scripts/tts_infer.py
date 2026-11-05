"""Synthesize text with TTS, then convert it through a trained RVC model.

    uv run python scripts/tts_infer.py --text "Hello there" --output out.wav
"""

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.conversion_service import ConversionService  # noqa: E402
from src.utils.config import load_inference_config  # noqa: E402
from src.utils.config import load_model_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("tts_infer")

TTS_VOICES = [
    "en-US-AndrewNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
]


def synthesize(text: str, voice: str) -> str:
    import edge_tts

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
    import os

    os.close(fd)

    async def run() -> None:
        await edge_tts.Communicate(text, voice).save(path)

    asyncio.run(run())

    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="TTS + RVC inference")

    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--voice", default=TTS_VOICES[0], choices=TTS_VOICES, help="TTS source voice"
    )
    parser.add_argument("--output", required=True, help="Output audio file")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--pitch", type=int, default=0, help="Transpose in semitones")
    parser.add_argument("--index-rate", type=float, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--f0-method", default=None, choices=[None, "harvest", "dio", "crepe"])
    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument("--inference-config", default="config/inference.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--keep-tts", action="store_true", help="Also save the raw TTS audio"
    )

    args = parser.parse_args()

    logger.info(f"Synthesizing with {args.voice}: {args.text!r}")
    tts_path = synthesize(args.text, args.voice)

    if args.keep_tts:
        tts_output = str(Path(args.output).with_suffix("")) + ".tts.wav"
        import shutil

        shutil.copyfile(tts_path, tts_output)
        logger.info(f"Raw TTS audio: {tts_output}")

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

    result = service.convert_file(
        tts_path,
        args.output,
        speaker_id=args.speaker_id,
        pitch_shift=args.pitch,
        index_rate=args.index_rate,
    )

    logger.info(f"Wrote {result.output_path} (RTF {result.realtime_factor:.2f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
