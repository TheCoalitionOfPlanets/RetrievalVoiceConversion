"""WaveNet residual stack, HiFi-GAN ResBlocks and coupling layers."""

import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import functional as F

from src.models.vits.commons import fused_add_tanh_sigmoid_multiply
from src.models.vits.commons import get_padding
from src.models.vits.commons import init_weights
from src.models.vits.commons import remove_weight_norm
from src.models.vits.commons import weight_norm

LRELU_SLOPE = 0.1


class WN(nn.Module):
    """Gated WaveNet residual stack, optionally conditioned on a global vector."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)

            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.in_layers.append(weight_norm(in_layer, name="weight"))

            res_skip_channels = (
                2 * hidden_channels if i < n_layers - 1 else hidden_channels
            )
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            self.res_skip_layers.append(weight_norm(res_skip_layer, name="weight"))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_channels_tensor = self.hidden_channels

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if g is not None:
                offset = i * 2 * self.hidden_channels
                g_l = g[:, offset : offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = self.drop(
                fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            )

            res_skip_acts = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self) -> None:
        if self.gin_channels != 0:
            remove_weight_norm(self.cond_layer)

        for layer in self.in_layers:
            remove_weight_norm(layer)

        for layer in self.res_skip_layers:
            remove_weight_norm(layer)


class ResBlock1(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in dilation
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)

            if x_mask is not None:
                xt = xt * x_mask

            xt = F.leaky_relu(c1(xt), LRELU_SLOPE)

            if x_mask is not None:
                xt = xt * x_mask

            x = c2(xt) + x

        return x * x_mask if x_mask is not None else x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)

        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3),
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)

            if x_mask is not None:
                xt = xt * x_mask

            x = c(xt) + x

        return x * x_mask if x_mask is not None else x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_weight_norm(layer)


class Flip(nn.Module):

    def forward(self, x: torch.Tensor, *args, reverse: bool = False, **kwargs):
        x = torch.flip(x, [1])

        if not reverse:
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet

        return x


class ResidualCouplingLayer(nn.Module):
    """Affine (mean-only) coupling layer backed by a WaveNet stack."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()
        assert channels % 2 == 0

        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1
        )

        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = (m + x1 * torch.exp(logs)) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet

        x1 = ((x1 - m) * torch.exp(-logs)) * x_mask

        return torch.cat([x0, x1], 1)

    def remove_weight_norm(self) -> None:
        self.enc.remove_weight_norm()
