"""Thin wiring between configs and the Trainer."""

from src.training.trainer import Trainer
from src.utils.config import load_model_config
from src.utils.config import load_train_config
from src.utils.logger import get_logger
from src.utils.settings import ModelConfig
from src.utils.settings import TrainConfig

logger = get_logger(__name__)


class TrainingService:

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        train_config: TrainConfig | None = None,
        device: str = "auto",
    ):
        self.model_config = model_config or load_model_config()
        self.train_config = train_config or load_train_config()
        self.device = device

    @classmethod
    def from_paths(
        cls,
        model_config_path: str = "config/model.yaml",
        train_config_path: str = "config/train.yaml",
        device: str = "auto",
    ) -> "TrainingService":
        return cls(
            model_config=load_model_config(model_config_path),
            train_config=load_train_config(train_config_path),
            device=device,
        )

    def run(self) -> None:
        trainer = Trainer(self.model_config, self.train_config, device=self.device)

        trainer.train()
