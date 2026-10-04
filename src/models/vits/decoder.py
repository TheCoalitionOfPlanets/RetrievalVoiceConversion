"""NSF-HiFiGAN generator: harmonic source excitation + HiFi-GAN upsampling."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import ConvTranspose1d
from torch.nn import functional as F

from src.models.vits import modules
from src.models.vits.commons import init_weights
from src.models.vits.commons import remove_weight_norm
from src.models.vits.commons import weight_norm

LRELU_SLOPE = 0.1


class SineGen(nn.Module):
    """Builds a harmonic sine bank from an upsampled F0 contour."""

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        return (f0 > self.voiced_threshold).float()

    def forward(self, f0: torch.Tensor, upp: int):
        """f0: (B, T_frames, 1) in Hz -> sine waves at (B, T_samples, dim)."""

        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)

            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]

            for idx in range(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            rad_values = (f0_buf / self.sampling_rate) % 1
            rand_ini = torch.rand(
                f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device
            )
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

            tmp_over_one = torch.cumsum(rad_values, 1)
            tmp_over_one *= upp
            tmp_over_one = F.interpolate(
                tmp_over_one.transpose(2, 1),
                scale_factor=float(upp),
                mode="linear",
                align_corners=True,
            ).transpose(2, 1)

            rad_values = F.interpolate(
                rad_values.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)

            tmp_over_one %= 1
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = ((tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0).float() * -1.0

            sine_waves = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
            )
            sine_waves = sine_waves * self.sine_amp

            uv = self._f02uv(f0)
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise

        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """Collapses the harmonic bank into a single excitation channel."""

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0.0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, upp: int):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        return sine_merge, None, None


class GeneratorNSF(nn.Module):
    """HiFi-GAN generator whose upsampling path is injected with an F0 source."""

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        sr: int,
        harmonic_num: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = int(np.prod(upsample_rates))

        self.f0_upsamp = nn.Upsample(scale_factor=self.upp)
        self.m_source = SourceModuleHnNSF(sampling_rate=sr, harmonic_num=harmonic_num)

        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_cls = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))

            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        c_cur,
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

            # Downsample the raw excitation to whatever rate this stage runs at.
            if i + 1 < len(upsample_rates):
                stride_f0 = int(np.prod(upsample_rates[i + 1 :]))
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()

        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))

            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, k, tuple(d)))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        g: torch.Tensor | None = None,
        n_res: int | None = None,
    ) -> torch.Tensor:
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)

        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            x_source = self.noise_convs[i](har_source)

            # Transposed conv and strided conv can disagree by a frame or two.
            if x_source.shape[-1] != x.shape[-1]:
                x_source = F.pad(x_source, (0, x.shape[-1] - x_source.shape[-1]))
                x_source = x_source[:, :, : x.shape[-1]]

            x = x + x_source

            xs = None

            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                xs = out if xs is None else xs + out

            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)

        return torch.tanh(x)

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)

        for layer in self.resblocks:
            layer.remove_weight_norm()
