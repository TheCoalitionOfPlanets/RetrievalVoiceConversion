"""Lazy, cached access to the heavy models so nothing loads twice."""

from src.feature_extraction.extractor import FeatureExtractor
from src.feature_extraction.hubert import HubertExtractor
from src.feature_extraction.rmvpe import PitchExtractor
from src.retrieval.retrieval_service import RetrievalService
from src.utils.config import load_model_config
from src.utils.logger import get_logger
from src.utils.settings import IndexConfig
from src.utils.settings import ModelConfig

logger = get_logger(__name__)


class ModelRegistry:

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        device: str = "cpu",
        fp16: bool = False,
    ):
        self.config = model_config or load_model_config()
        self.device = device
        self.fp16 = fp16

        self._hubert: HubertExtractor | None = None
        self._pitch: PitchExtractor | None = None
        self._feature_extractor: FeatureExtractor | None = None
        self._retrieval: RetrievalService | None = None

    def get_hubert(self) -> HubertExtractor:
        if self._hubert is None:
            self._hubert = HubertExtractor(
                self.config.hubert, device=self.device, fp16=self.fp16
            ).load()

        return self._hubert

    def get_pitch_extractor(self) -> PitchExtractor:
        if self._pitch is None:
            self._pitch = PitchExtractor(self.config.pitch).load()

        return self._pitch

    def get_feature_extractor(self) -> FeatureExtractor:
        if self._feature_extractor is None:
            extractor = FeatureExtractor(
                self.config, device=self.device, fp16=self.fp16
            )

            # Share any already-warm sub-models instead of reloading them.
            if self._hubert is not None:
                extractor.hubert = self._hubert

            if self._pitch is not None:
                extractor.pitch = self._pitch

            self._feature_extractor = extractor.load()

        return self._feature_extractor

    def get_retrieval(self, index_config: IndexConfig | None = None):
        if self._retrieval is None:
            self._retrieval = RetrievalService(index_config).load()

        return self._retrieval

    def unload(self) -> None:
        self._hubert = None
        self._pitch = None
        self._feature_extractor = None
        self._retrieval = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Registry unloaded")
