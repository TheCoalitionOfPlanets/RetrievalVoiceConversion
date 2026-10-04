"""SynthesizerTrnMs768NSFsid — the RVC v2 generator.

Training path:   spec -> posterior q(z|spec) -> flow -> compared against the
                 prior p(z|content,f0); a random slice of z drives the decoder.
Inference path:  content + f0 -> prior -> inverse flow -> decoder -> waveform.
"""

import torch
from torch import nn

from src.models.vits.commons import rand_slice_segments
from src.models.vits.commons import slice_segments
from src.models.vits.decoder import GeneratorNSF
from src.models.vits.encoder import PosteriorEncoder
from src.models.vits.encoder import TextEncoder
from src.models.vits.flow import ResidualCouplingBlock
from src.utils.settings import ModelConfig


class SynthesizerTrnMs768NSFsid(nn.Module):

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        embedding_dim: int = 768,
        f0_bin: int = 256,
    ):
        super().__init__()

        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.sr = sr

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            embedding_dim=embedding_dim,
            f0_bin=f0_bin,
            use_f0=True,
        )

        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            harmonic_num=0,
        )

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )

        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )

        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        segment_size: int,
    ) -> "SynthesizerTrnMs768NSFsid":
        gen = config.generator
        audio = config.audio

        return cls(
            spec_channels=audio.filter_length // 2 + 1,
            segment_size=segment_size // audio.hop_length,
            inter_channels=gen.inter_channels,
            hidden_channels=gen.hidden_channels,
            filter_channels=gen.filter_channels,
            n_heads=gen.n_heads,
            n_layers=gen.n_layers,
            kernel_size=gen.kernel_size,
            p_dropout=gen.p_dropout,
            resblock=str(gen.resblock),
            resblock_kernel_sizes=gen.resblock_kernel_sizes,
            resblock_dilation_sizes=gen.resblock_dilation_sizes,
            upsample_rates=gen.upsample_rates,
            upsample_initial_channel=gen.upsample_initial_channel,
            upsample_kernel_sizes=gen.upsample_kernel_sizes,
            spk_embed_dim=gen.spk_embed_dim,
            gin_channels=gen.gin_channels,
            sr=audio.sample_rate,
            embedding_dim=config.hubert.embedding_dim,
            f0_bin=config.pitch.f0_bin,
        )

    def remove_weight_norm(self) -> None:
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: torch.Tensor,
    ):
        """Training step. Returns (audio_slice, ids_slice, masks, latents)."""

        g = self.emb_g(ds).unsqueeze(-1)

        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)

        pitchf_slice = slice_segments(
            pitchf.unsqueeze(1), ids_slice, self.segment_size
        ).squeeze(1)

        o = self.dec(z_slice, pitchf_slice, g=g)

        return (
            o,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )

    @torch.no_grad()
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        noise_scale: float = 0.66,
    ):
        """Content + pitch -> waveform."""

        g = self.emb_g(sid).unsqueeze(-1)

        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)

        o = self.dec(z * x_mask, nsff0, g=g)

        return o, x_mask, (z, z_p, m_p, logs_p)
