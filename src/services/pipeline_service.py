"""Top-level orchestrator over the preprocess / index / train / infer stages."""

from pathlib import Path

from src.retrieval.faiss_index import FaissIndex
from src.services.conversion_service import ConversionService
from src.services.feature_service import FeatureService
from src.services.model_registry import ModelRegistry
from src.services.training_service import TrainingService
from src.utils.config import load_inference_config
from src.utils.config import load_model_config
from src.utils.config import load_train_config
from src.utils.device import describe_device
from src.utils.device import resolve_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineService:

    def __init__(
        self,
        model_config_path: str = "config/model.yaml",
        train_config_path: str = "config/train.yaml",
        inference_config_path: str = "config/inference.yaml",
        device: str = "auto",
    ):
        self.logger = get_logger(self.__class__.__name__)

        self.model_config = load_model_config(model_config_path)
        self.train_config = load_train_config(train_config_path)
        self.inference_config = load_inference_config(inference_config_path)

        self.device = resolve_device(device)
        self.device_spec = device

        self.registry = ModelRegistry(
            self.model_config,
            device=str(self.device),
            fp16=self.device.type == "cuda",
        )

    def initialize(self) -> "PipelineService":
        self.logger.info(f"Device: {describe_device(self.device)}")
        self.logger.info("Loading feature extractor")

        self.registry.get_feature_extractor()

        self.logger.info("Pipeline initialized")

        return self

    # ------------------------------------------------------------- stages

    def preprocess(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str | None = None,
    ) -> int:
        processed_dir = processed_dir or self.train_config.data.processed_dir

        service = FeatureService(
            self.model_config,
            device=str(self.device),
            fp16=self.device.type == "cuda",
        )

        return service.preprocess(raw_dir, processed_dir)

    def build_index(
        self,
        processed_dir: str | None = None,
        index_path: str | None = None,
        features_path: str | None = None,
        max_vectors: int = 400_000,
    ) -> int:
        """Collects every extracted content frame into one searchable bank."""

        import numpy as np

        processed_dir = processed_dir or self.train_config.data.processed_dir
        index_path = index_path or self.inference_config.index.path
        features_path = features_path or self.inference_config.index.features

        feature_dir = Path(processed_dir) / "features"
        files = sorted(feature_dir.glob("*.npy"))

        if not files:
            raise FileNotFoundError(
                f"No features in {feature_dir}; run preprocess first"
            )

        stacked = np.concatenate([np.load(f) for f in files], axis=0)

        if stacked.shape[0] > max_vectors:
            # Uniform stride keeps phonetic coverage better than a head slice.
            step = stacked.shape[0] // max_vectors + 1
            self.logger.info(
                f"Subsampling {stacked.shape[0]} -> ~{max_vectors} vectors "
                f"(stride {step})"
            )
            stacked = stacked[::step]

        index = FaissIndex().build(stacked)
        index.save(index_path, features_path)

        return index.ntotal

    def train(self) -> None:
        TrainingService(
            self.model_config, self.train_config, device=self.device_spec
        ).run()

    def convert(
        self,
        input_path: str,
        output_path: str,
        speaker_id: int = 0,
        pitch_shift: int = 0,
    ):
        service = ConversionService(self.model_config, self.inference_config)

        return service.convert_file(
            input_path,
            output_path,
            speaker_id=speaker_id,
            pitch_shift=pitch_shift,
        )

    def run_all(self, raw_dir: str = "data/raw") -> None:
        """Full path from a folder of audio to a trained, indexed model."""

        self.initialize()

        self.logger.info("Stage 1/3: preprocess")
        self.preprocess(raw_dir)

        self.logger.info("Stage 2/3: build retrieval index")
        self.build_index()

        self.logger.info("Stage 3/3: train")
        self.train()
