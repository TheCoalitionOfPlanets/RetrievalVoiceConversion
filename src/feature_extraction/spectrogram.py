"""Linear / mel spectrogram transforms with cached filter banks."""

import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

_mel_basis: dict[str, torch.Tensor] = {}
_hann_window: dict[str, torch.Tensor] = {}


def dynamic_range_compression_torch(
    x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5
) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    return dynamic_range_compression_torch(magnitudes)


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    """Linear magnitude spectrogram, shape (B, n_fft // 2 + 1, T)."""

    key = f"{win_size}_{y.dtype}_{y.device}"

    if key not in _hann_window:
        _hann_window[key] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window[key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float | None,
) -> torch.Tensor:
    key = f"{fmax}_{spec.dtype}_{spec.device}"

    if key not in _mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        _mel_basis[key] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    return spectral_normalize_torch(torch.matmul(_mel_basis[key], spec))


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float | None,
    center: bool = False,
) -> torch.Tensor:
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)

    return spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
