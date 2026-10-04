"""Shared tensor helpers and weight-norm compatibility shims."""

import math

import torch
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm as _remove_weight_norm
from torch.nn.utils import spectral_norm as _spectral_norm
from torch.nn.utils import weight_norm as _weight_norm

weight_norm = _weight_norm
spectral_norm = _spectral_norm


def remove_weight_norm(module: torch.nn.Module) -> torch.nn.Module:
    """Strip weight norm, tolerating modules that never had it."""

    try:
        return _remove_weight_norm(module)
    except (ValueError, RuntimeError):
        return module


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max())

    x = torch.arange(max_length, dtype=length.dtype, device=length.device)

    return x.unsqueeze(0) < length.unsqueeze(1)


def slice_segments(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int = 4,
) -> torch.Tensor:
    """Slice (B, C, T) at per-batch start indices."""

    out = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        start = int(ids_str[i])
        out[i] = x[i, :, start : start + segment_size]

    return out


def slice_segments2(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int = 4,
) -> torch.Tensor:
    """Same as slice_segments for 2-D (B, T) tensors."""

    out = torch.zeros_like(x[:, :segment_size])

    for i in range(x.size(0)):
        start = int(ids_str[i])
        out[i] = x[i, start : start + segment_size]

    return out


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor | None = None,
    segment_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, t = x.size()

    if x_lengths is None:
        x_lengths = torch.full((b,), t, device=x.device)

    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=1)
    ids_str = (torch.rand(b, device=x.device) * ids_str_max).long()

    return slice_segments(x, ids_str, segment_size), ids_str


def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: int,
) -> torch.Tensor:
    """Gated activation. Not torch.jit.script'd: unsupported on Python 3.14+."""

    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])

    return t_act * s_act


def convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    layer = pad_shape[::-1]

    return [item for sublist in layer for item in sublist]


def kl_divergence(
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    m_q: torch.Tensor,
    logs_q: torch.Tensor,
) -> torch.Tensor:
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )

    return kl


def clip_grad_value(
    parameters,
    clip_value: float | None,
    norm_type: float = 2.0,
) -> float:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    total_norm = 0.0

    for p in parameters:
        total_norm += float(p.grad.data.norm(norm_type)) ** norm_type

        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)

    return total_norm ** (1.0 / norm_type)
