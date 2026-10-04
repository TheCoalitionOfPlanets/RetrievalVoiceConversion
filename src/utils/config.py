from pathlib import Path

import yaml

from src.utils.settings import InferenceConfig
from src.utils.settings import ModelConfig
from src.utils.settings import TrainConfig


def load_yaml(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_model_config(path: str | Path = "config/model.yaml") -> ModelConfig:
    return ModelConfig.from_dict(load_yaml(path))


def load_train_config(path: str | Path = "config/train.yaml") -> TrainConfig:
    return TrainConfig.from_dict(load_yaml(path))


def load_inference_config(
    path: str | Path = "config/inference.yaml",
) -> InferenceConfig:
    return InferenceConfig.from_dict(load_yaml(path))
