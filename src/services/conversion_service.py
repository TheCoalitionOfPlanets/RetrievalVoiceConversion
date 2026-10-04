"""Inference-facing service wrapping the conversion pipeline."""

from src.inference.voice_conversion_pipeline import VoiceConversionPipeline
from src.models.conversion import ConversionRequest
from src.models.conversion import ConversionResult
from src.utils.config import load_inference_config
from src.utils.config import load_model_config
from src.utils.logger import get_logger
from src.utils.settings import InferenceConfig
from src.utils.settings import ModelConfig

logger = get_logger(__name__)


class ConversionService:

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        inference_config: InferenceConfig | None = None,
    ):
        self.inference_config = inference_config or load_inference_config()
        self.model_config = model_config or load_model_config(
            self.inference_config.model_config_path
        )

        self.pipeline = VoiceConversionPipeline(
            self.model_config, self.inference_config
        )
        self._loaded = False

    def load(self) -> "ConversionService":
        if not self._loaded:
            self.pipeline.load()
            self._loaded = True

        return self

    def convert(self, request: ConversionRequest) -> ConversionResult:
        self.load()

        return self.pipeline.run(request)

    def convert_file(
        self,
        input_path: str,
        output_path: str,
        speaker_id: int = 0,
        pitch_shift: int = 0,
        index_rate: float | None = None,
    ) -> ConversionResult:
        return self.convert(
            ConversionRequest(
                input_audio_path=input_path,
                output_audio_path=output_path,
                speaker_id=speaker_id,
                pitch_shift=pitch_shift,
                index_rate=index_rate,
            )
        )

    def convert_directory(
        self,
        input_dir: str,
        output_dir: str,
        speaker_id: int = 0,
        pitch_shift: int = 0,
    ) -> list[ConversionResult]:
        self.load()

        return self.pipeline.run_batch(
            input_dir, output_dir, speaker_id=speaker_id, pitch_shift=pitch_shift
        )
