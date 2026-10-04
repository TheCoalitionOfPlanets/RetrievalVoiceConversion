from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


def _section(raw: dict | None, key: str) -> dict:
    if not raw:
        return {}
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


@dataclass
class AudioConfig:
    sample_rate: int = 32000
    hubert_sample_rate: int = 16000
    filter_length: int = 1024
    hop_length: int = 320
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "AudioConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class HubertConfig:
    model_name: str = "lengyue233/content-vec-best"
    output_layer: int = 12
    embedding_dim: int = 768

    @classmethod
    def from_dict(cls, raw: dict) -> "HubertConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class PitchConfig:
    method: str = "harvest"
    sample_rate: int = 16000
    hop_length: int = 160
    f0_min: float = 50.0
    f0_max: float = 1100.0
    f0_bin: int = 256

    @classmethod
    def from_dict(cls, raw: dict) -> "PitchConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class GeneratorConfig:
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.0
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: list[int] = field(default_factory=lambda: [10, 8, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 4, 4])
    spk_embed_dim: int = 109
    gin_channels: int = 256

    @classmethod
    def from_dict(cls, raw: dict) -> "GeneratorConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class DiscriminatorConfig:
    periods: list[int] = field(default_factory=lambda: [2, 3, 5, 7, 11, 17])
    use_spectral_norm: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "DiscriminatorConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class ModelConfig:
    """Everything needed to rebuild the synthesizer from a checkpoint."""

    sample_rate: int = 32000
    audio: AudioConfig = field(default_factory=AudioConfig)
    hubert: HubertConfig = field(default_factory=HubertConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        return cls(
            sample_rate=raw.get("sample_rate", 32000),
            audio=AudioConfig.from_dict(_section(raw, "audio")),
            hubert=HubertConfig.from_dict(_section(raw, "hubert")),
            pitch=PitchConfig.from_dict(_section(raw, "pitch")),
            generator=GeneratorConfig.from_dict(_section(raw, "generator")),
            discriminator=DiscriminatorConfig.from_dict(
                _section(raw, "discriminator")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class DataConfig:
    processed_dir: str = "data/processed"
    filelist: str = "data/processed/filelist.txt"
    min_frames: int = 60
    max_frames: int = 900

    @classmethod
    def from_dict(cls, raw: dict) -> "DataConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class OptimConfig:
    epochs: int = 300
    batch_size: int = 4
    segment_size: int = 12800
    learning_rate: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.8, 0.99])
    eps: float = 1e-9
    lr_decay: float = 0.999875
    fp16: bool = True
    grad_accum_steps: int = 1
    max_grad_norm: float | None = None
    num_workers: int = 2

    @classmethod
    def from_dict(cls, raw: dict) -> "OptimConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class LossConfig:
    c_mel: float = 45.0
    c_kl: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict) -> "LossConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class CheckpointConfig:
    dir: str = "data/checkpoints"
    save_every_epoch: int = 5
    keep_last: int = 3
    pretrained_g: str | None = None
    pretrained_d: str | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "CheckpointConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class LoggingConfig:
    log_dir: str = "data/logs"
    log_every_steps: int = 50
    eval_every_epoch: int = 10

    @classmethod
    def from_dict(cls, raw: dict) -> "LoggingConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class TrainConfig:
    seed: int = 1234
    data: DataConfig = field(default_factory=DataConfig)
    training: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, raw: dict) -> "TrainConfig":
        return cls(
            seed=raw.get("seed", 1234),
            data=DataConfig.from_dict(_section(raw, "data")),
            training=OptimConfig.from_dict(_section(raw, "training")),
            loss=LossConfig.from_dict(_section(raw, "loss")),
            checkpoint=CheckpointConfig.from_dict(_section(raw, "checkpoint")),
            logging=LoggingConfig.from_dict(_section(raw, "logging")),
        )


@dataclass
class IndexConfig:
    path: str = "data/indexes/speaker.index"
    features: str = "data/indexes/speaker_features.npy"
    rate: float = 0.75
    k: int = 8

    @classmethod
    def from_dict(cls, raw: dict) -> "IndexConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class ConversionConfig:
    speaker_id: int = 0
    pitch_shift: int = 0
    f0_method: str = "harvest"
    protect: float = 0.33
    rms_mix_rate: float = 0.25
    output_sample_rate: int = 32000
    chunk_seconds: float = 30.0
    crossfade_seconds: float = 0.1

    @classmethod
    def from_dict(cls, raw: dict) -> "ConversionConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})


@dataclass
class InferenceConfig:
    device: str = "auto"
    checkpoint: str = "data/checkpoints/G_latest.pth"
    model_config_path: str = "config/model.yaml"
    index: IndexConfig = field(default_factory=IndexConfig)
    conversion: ConversionConfig = field(default_factory=ConversionConfig)

    @classmethod
    def from_dict(cls, raw: dict) -> "InferenceConfig":
        model = _section(raw, "model")
        return cls(
            device=raw.get("device", "auto"),
            checkpoint=model.get("checkpoint", "data/checkpoints/G_latest.pth"),
            model_config_path=model.get("config", "config/model.yaml"),
            index=IndexConfig.from_dict(_section(raw, "index")),
            conversion=ConversionConfig.from_dict(_section(raw, "conversion")),
        )
