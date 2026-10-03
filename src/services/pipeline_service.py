from src.services.model_registry import ModelRegistry
from src.utils.logger import get_logger


class PipelineService:

    def __init__(self):

        self.logger = get_logger(
            self.__class__.__name__
        )

        self.registry = ModelRegistry()

    def initialize(self):

        self.logger.info(
            "Loading Feature Extractor"
        )

        self.registry.get_feature_extractor()

        self.logger.info(
            "Pipeline Initialized"
        )
