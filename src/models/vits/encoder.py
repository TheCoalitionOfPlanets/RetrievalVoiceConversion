"""Content (prior) encoder and spectrogram (posterior) encoder."""

import math

import torch
from torch import nn

from src.models.vits import attentions
from src.models.vits.commons import sequence_mask
from src.models.vits.modules import WN


class TextEncoder(nn.Module):
    """Maps HuBERT content frames (+ coarse pitch) to the prior distribution."""

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        embedding_dim: int = 768,
        f0_bin: int = 256,
        use_f0: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.use_f0 = use_f0

        self.emb_phone = nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        if use_f0:
            self.emb_pitch = nn.Embedding(f0_bin, hidden_channels)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor | None,
        lengths: torch.Tensor,
        skip_head: int | None = None,
    ):
        """phone: (B, T, embedding_dim); pitch: (B, T) coarse indices."""

        x = self.emb_phone(phone)

        if self.use_f0 and pitch is not None:
            x = x + self.emb_pitch(pitch)

        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)

        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)

        if skip_head is not None:
            x = x[:, :, skip_head:]
            x_mask = x_mask[:, :, skip_head:]

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return m, logs, x_mask


class PosteriorEncoder(nn.Module):
    """Encodes the linear spectrogram into the latent posterior q(z|x)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: torch.Tensor | None = None,
    ):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask

    def remove_weight_norm(self) -> None:
        self.enc.remove_weight_norm()
