"""Relative-position multi-head attention encoder used by the content encoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.vits.commons import convert_pad_shape


class LayerNorm(nn.Module):

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)

        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int | None = 10,
        heads_share: bool = True,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.proximal_bias = proximal_bias
        self.k_channels = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5

            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, _ = self.attention(q, k, v, mask=attn_mask)

        return self.conv_o(x)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, d, t_s = key.size()
        t_t = query.size(2)

        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores = scores + self._relative_position_to_absolute_position(rel_logits)

        if self.proximal_bias:
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        p_attn = self.drop(F.softmax(scores, dim=-1))
        output = torch.matmul(p_attn, value)

        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)

        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(self, x, y):
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, relative_embeddings, length: int):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1

        if pad_length > 0:
            relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )

        return relative_embeddings[:, slice_start:slice_end]

    def _relative_position_to_absolute_position(self, x: torch.Tensor):
        batch, heads, length, _ = x.size()

        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(
            x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]

    def _absolute_position_to_relative_position(self, x: torch.Tensor):
        batch, heads, length, _ = x.size()

        x = F.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))

        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

    def _attention_bias_proximal(self, length: int):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)

        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str | None = None,
        causal: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = self._causal_padding if causal else self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self.padding(x * x_mask))

        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)

        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))

        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x

        return F.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [self.kernel_size - 1, 0]])
        )

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x

        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2

        return F.pad(x, convert_pad_shape([[0, 0], [0, 0], [pad_l, pad_r]]))


class Encoder(nn.Module):
    """Stack of self-attention + FFN blocks with relative positional bias."""

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ):
        super().__init__()
        self.n_layers = n_layers

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        for i in range(self.n_layers):
            y = self.drop(self.attn_layers[i](x, x, attn_mask))
            x = self.norm_layers_1[i](x + y)

            y = self.drop(self.ffn_layers[i](x, x_mask))
            x = self.norm_layers_2[i](x + y)

        return x * x_mask
