from src.utils.config import load_yaml
from src.utils.settings import HubertConfig


config = load_yaml("config/model.yaml")

hubert = HubertConfig(**config["hubert"])

print(hubert)
print(hubert.model_name)