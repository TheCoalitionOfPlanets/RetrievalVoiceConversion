"""Multi-period + multi-scale discriminator (HiFi-GAN)."""

import torch
from torch import nn
from torch.nn import AvgPool1d
from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import functional as F

from src.models.vits.commons import get_padding
from src.models.vits.commons import spectral_norm
from src.models.vits.commons import weight_norm

LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    """Reshapes the waveform into a 2-D grid of `period` to catch periodicity."""

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1, 32, (kernel_size, 1), (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32, 128, (kernel_size, 1), (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128, 512, (kernel_size, 1), (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512, 1024, (kernel_size, 1), (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []

        b, c, t = x.shape

        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    """Plain 1-D waveform discriminator."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        fmap = []

        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):

    def __init__(
        self,
        periods: list[int] | None = None,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        periods = periods or [2, 3, 5, 7, 11, 17]

        discriminators = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discriminators += [
            DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods
        ]

        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
