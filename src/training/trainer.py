"""The adversarial training loop for the RVC synthesizer."""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.feature_extraction.spectrogram import mel_spectrogram_torch
from src.feature_extraction.spectrogram import spec_to_mel_torch
from src.models.vits.commons import clip_grad_value
from src.models.vits.commons import slice_segments
from src.models.vits.discriminator import MultiPeriodDiscriminator
from src.models.vits.model import SynthesizerTrnMs768NSFsid
from src.training.callbacks import CheckpointManager
from src.training.callbacks import TensorBoardLogger
from src.training.callbacks import build_schedulers
from src.training.dataset import DistributedBucketSampler
from src.training.dataset import VoiceConversionCollate
from src.training.dataset import VoiceConversionDataset
from src.training.losses import discriminator_loss
from src.training.losses import feature_loss
from src.training.losses import generator_loss
from src.training.losses import kl_loss
from src.utils.checkpoint import latest_checkpoint
from src.utils.checkpoint import load_checkpoint
from src.utils.device import describe_device
from src.utils.device import resolve_device
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.settings import ModelConfig
from src.utils.settings import TrainConfig

logger = get_logger(__name__)


class Trainer:

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        device: str = "auto",
    ):
        self.model_config = model_config
        self.config = train_config
        self.device = resolve_device(device)

        set_seed(train_config.seed)

        self.epoch = 1
        self.global_step = 0

        self._build_data()
        self._build_models()
        self._build_optimizers()

        self.tb = TensorBoardLogger(train_config.logging)
        self.checkpoints = CheckpointManager(
            train_config.checkpoint, model_config=model_config.to_dict()
        )

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=train_config.training.fp16 and self.device.type == "cuda"
        )

    # ---------------------------------------------------------------- setup

    def _build_data(self) -> None:
        cfg = self.config

        self.dataset = VoiceConversionDataset(
            cfg.data.filelist,
            self.model_config.audio,
            min_frames=cfg.data.min_frames,
            max_frames=cfg.data.max_frames,
        )

        sampler = DistributedBucketSampler(
            self.dataset, cfg.training.batch_size, shuffle=True
        )
        self.sampler = sampler

        self.loader = DataLoader(
            self.dataset,
            num_workers=cfg.training.num_workers,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
            collate_fn=VoiceConversionCollate(),
            batch_sampler=sampler,
            persistent_workers=cfg.training.num_workers > 0,
        )

    def _build_models(self) -> None:
        self.net_g = SynthesizerTrnMs768NSFsid.from_config(
            self.model_config, self.config.training.segment_size
        ).to(self.device)

        self.net_d = MultiPeriodDiscriminator(
            periods=self.model_config.discriminator.periods,
            use_spectral_norm=self.model_config.discriminator.use_spectral_norm,
        ).to(self.device)

        params_g = sum(p.numel() for p in self.net_g.parameters()) / 1e6
        params_d = sum(p.numel() for p in self.net_d.parameters()) / 1e6

        logger.info(f"Device: {describe_device(self.device)}")
        logger.info(f"Generator {params_g:.1f}M params, discriminator {params_d:.1f}M")

    def _build_optimizers(self) -> None:
        cfg = self.config.training

        self.optim_g = torch.optim.AdamW(
            self.net_g.parameters(),
            cfg.learning_rate,
            betas=tuple(cfg.betas),
            eps=cfg.eps,
        )
        self.optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            cfg.learning_rate,
            betas=tuple(cfg.betas),
            eps=cfg.eps,
        )

        self._resume()

        self.scheduler_g, self.scheduler_d = build_schedulers(
            self.optim_g, self.optim_d, cfg.lr_decay, last_epoch=self.epoch - 2
        )

    def _resume(self) -> None:
        """Prefer a run in progress; fall back to the configured pretrained pair."""

        ckpt = self.config.checkpoint

        g_path = latest_checkpoint(ckpt.dir, "G")
        d_path = latest_checkpoint(ckpt.dir, "D")

        if g_path and d_path:
            epoch, step, _ = load_checkpoint(g_path, self.net_g, self.optim_g)
            load_checkpoint(d_path, self.net_d, self.optim_d)

            self.epoch = epoch + 1
            self.global_step = step

            logger.info(f"Resuming from epoch {self.epoch}")
            return

        if ckpt.pretrained_g and Path(ckpt.pretrained_g).exists():
            load_checkpoint(ckpt.pretrained_g, self.net_g)
            logger.info(f"Initialised G from pretrained {ckpt.pretrained_g}")

        if ckpt.pretrained_d and Path(ckpt.pretrained_d).exists():
            load_checkpoint(ckpt.pretrained_d, self.net_d)
            logger.info(f"Initialised D from pretrained {ckpt.pretrained_d}")

    # ------------------------------------------------------------- training

    def train(self) -> None:
        total_epochs = self.config.training.epochs

        logger.info(
            f"Training epochs {self.epoch}..{total_epochs}, "
            f"{len(self.sampler)} steps/epoch, "
            f"batch {self.config.training.batch_size}, "
            f"fp16={self.scaler.is_enabled()}"
        )

        try:
            for epoch in range(self.epoch, total_epochs + 1):
                self.epoch = epoch
                self.sampler.set_epoch(epoch)

                self._train_epoch(epoch)

                self.scheduler_g.step()
                self.scheduler_d.step()

                if self.checkpoints.should_save(epoch):
                    self._save(epoch)
        except KeyboardInterrupt:
            logger.warning("Interrupted — saving before exit")
            self._save(self.epoch)
        finally:
            self.tb.close()

        logger.info("Training complete")

    def _save(self, epoch: int) -> None:
        self.checkpoints.save(
            self.net_g,
            self.net_d,
            self.optim_g,
            self.optim_d,
            self.optim_g.param_groups[0]["lr"],
            epoch,
            self.global_step,
        )

    def _train_epoch(self, epoch: int) -> None:
        self.net_g.train()
        self.net_d.train()

        audio = self.model_config.audio
        cfg = self.config.training
        started = time.time()

        for batch_index, batch in enumerate(self.loader):
            (
                content,
                content_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wav,
                wav_lengths,
                sid,
            ) = [t.to(self.device, non_blocking=True) for t in batch]

            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                (
                    y_hat,
                    ids_slice,
                    _,
                    z_mask,
                    (_, z_p, m_p, logs_p, _, logs_q),
                ) = self.net_g(
                    content,
                    content_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    sid,
                )

                mel = spec_to_mel_torch(
                    spec.float(),
                    audio.filter_length,
                    audio.n_mel_channels,
                    audio.sample_rate,
                    audio.mel_fmin,
                    audio.mel_fmax,
                )
                y_mel = slice_segments(
                    mel, ids_slice, cfg.segment_size // audio.hop_length
                )

                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    audio.filter_length,
                    audio.n_mel_channels,
                    audio.sample_rate,
                    audio.hop_length,
                    audio.win_length,
                    audio.mel_fmin,
                    audio.mel_fmax,
                )

                y = slice_segments(
                    wav, ids_slice * audio.hop_length, cfg.segment_size
                )

            # ---- discriminator
            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())

                with torch.amp.autocast("cuda", enabled=False):
                    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            self.optim_d.zero_grad(set_to_none=True)
            self.scaler.scale(loss_disc).backward()
            self.scaler.unscale_(self.optim_d)
            grad_norm_d = clip_grad_value(
                self.net_d.parameters(), cfg.max_grad_norm
            )
            self.scaler.step(self.optim_d)

            # ---- generator
            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                _, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)

                with torch.amp.autocast("cuda", enabled=False):
                    loss_mel = (
                        torch.nn.functional.l1_loss(y_mel, y_hat_mel)
                        * self.config.loss.c_mel
                    )
                    loss_kl = (
                        kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                        * self.config.loss.c_kl
                    )
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, _ = generator_loss(y_d_hat_g)

                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            self.optim_g.zero_grad(set_to_none=True)
            self.scaler.scale(loss_gen_all).backward()
            self.scaler.unscale_(self.optim_g)
            grad_norm_g = clip_grad_value(
                self.net_g.parameters(), cfg.max_grad_norm
            )
            self.scaler.step(self.optim_g)
            self.scaler.update()

            self.global_step += 1

            if self.global_step % self.config.logging.log_every_steps == 0:
                lr = self.optim_g.param_groups[0]["lr"]
                elapsed = time.time() - started

                logger.info(
                    f"epoch {epoch} | step {self.global_step} "
                    f"({batch_index + 1}/{len(self.sampler)}) | "
                    f"g {float(loss_gen_all):.3f} d {float(loss_disc):.3f} "
                    f"mel {float(loss_mel):.3f} kl {float(loss_kl):.3f} "
                    f"fm {float(loss_fm):.3f} | {elapsed:.0f}s"
                )

                self.tb.log_scalars(
                    {
                        "loss/g/total": float(loss_gen_all),
                        "loss/d/total": float(loss_disc),
                        "loss/g/mel": float(loss_mel),
                        "loss/g/kl": float(loss_kl),
                        "loss/g/fm": float(loss_fm),
                        "loss/g/adv": float(loss_gen),
                        "grad/g": grad_norm_g,
                        "grad/d": grad_norm_d,
                        "learning_rate": lr,
                    },
                    self.global_step,
                )
