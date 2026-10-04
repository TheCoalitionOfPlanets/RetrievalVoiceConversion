"""End-to-end inference pipeline: audio in, converted audio out."""

from pathlib import Path

from src.inference.converter import VoiceConverter
from src.inference.pipeline import BasePipeline
from src.models.conversion import ConversionRequest
from src.models.conversion import ConversionResult
from src.utils.audio import AudioLoader
from src.utils.logger import get_logger
from src.utils.settings import InferenceConfig
from src.utils.settings import ModelConfig

logger = get_logger(__name__)


class VoiceConversionPipeline(BasePipeline):

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
    ):
        self.converter = VoiceConverter(model_config, inference_config)

    def load(self) -> "VoiceConversionPipeline":
        self.converter.load()

        return self

    def run(self, request: ConversionRequest) -> ConversionResult:
        return self.converter.convert(request)

    def run_batch(
        self,
        input_dir: str,
        output_dir: str,
        speaker_id: int = 0,
        pitch_shift: int = 0,
    ) -> list[ConversionResult]:
        files = AudioLoader.discover(input_dir)

        if not files:
            logger.warning(f"No audio found under {input_dir}")
            return []

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        results = []

        for index, path in enumerate(files, start=1):
            logger.info(f"[{index}/{len(files)}] {path.name}")

            results.append(
                self.run(
                    ConversionRequest(
                        input_audio_path=str(path),
                        output_audio_path=str(out_root / f"{path.stem}.wav"),
                        speaker_id=speaker_id,
                        pitch_shift=pitch_shift,
                    )
                )
            )

        return results
