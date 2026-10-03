from dataclasses import dataclass


@dataclass
class HubertConfig:
    model_name: str


@dataclass
class RMVPEConfig:
    sample_rate: int


@dataclass
class VITSConfig:
    hidden_size: int