"""Logging, checkpointing and LR scheduling hooks for the trainer."""

from pathlib import Path

import torch

from src.utils.checkpoint import prune_checkpoints
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import get_logger
from src.utils.settings import CheckpointConfig
from src.utils.settings import LoggingConfig

logger = get_logger(__name__)


class TensorBoardLogger:

    def __init__(self, config: LoggingConfig):
        self.config = config
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            Path(config.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=config.log_dir)

            logger.info(f"TensorBoard logging -> {config.log_dir}")
        except ImportError:
            logger.warning("tensorboard not installed; scalar logging disabled")

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        if self.writer is None:
            return

        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)

    def log_audio(
        self, tag: str, audio: torch.Tensor, step: int, sample_rate: int
    ) -> None:
        if self.writer is None:
            return

        self.writer.add_audio(tag, audio, step, sample_rate)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class CheckpointManager:

    def __init__(self, config: CheckpointConfig, model_config: dict | None = None):
        self.config = config
        self.model_config = model_config

        Path(config.dir).mkdir(parents=True, exist_ok=True)

    def should_save(self, epoch: int) -> bool:
        return epoch % self.config.save_every_epoch == 0

    def save(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        learning_rate: float,
        epoch: int,
        step: int,
    ) -> None:
        directory = Path(self.config.dir)

        save_checkpoint(
            generator,
            optim_g,
            learning_rate,
            epoch,
            step,
            directory / f"G_{epoch}.pth",
            config=self.model_config,
        )
        save_checkpoint(
            discriminator,
            optim_d,
            learning_rate,
            epoch,
            step,
            directory / f"D_{epoch}.pth",
        )

        # A stable path for inference, so configs never need editing.
        save_checkpoint(
            generator,
            None,
            learning_rate,
            epoch,
            step,
            directory / "G_latest.pth",
            config=self.model_config,
        )

        prune_checkpoints(directory, "G", self.config.keep_last)
        prune_checkpoints(directory, "D", self.config.keep_last)


def build_schedulers(
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    lr_decay: float,
    last_epoch: int = -1,
):
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=lr_decay, last_epoch=last_epoch
    )

    return scheduler_g, scheduler_d
