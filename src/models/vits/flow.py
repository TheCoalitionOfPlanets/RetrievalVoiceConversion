"""Normalizing flow bridging the prior and posterior latents."""

import torch
from torch import nn

from src.models.vits.modules import Flip
from src.models.vits.modules import ResidualCouplingLayer


class ResidualCouplingBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.n_flows = n_flows

        self.flows = nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)

        return x

    def remove_weight_norm(self) -> None:
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()
