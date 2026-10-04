from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.models.audio import AudioData
from src.utils.exceptions import AudioLoadError

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac")


class AudioLoader:

    @staticmethod
    def load(path: str, sample_rate: int = 16000) -> AudioData:

        if not Path(path).exists():
            raise AudioLoadError(f"File not found: {path}")

        try:
            waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
        except Exception as exc:
            raise AudioLoadError(f"Could not decode {path}: {exc}") from exc

        if waveform.size == 0:
            raise AudioLoadError(f"Empty audio: {path}")

        return AudioData(
            waveform=waveform.astype(np.float32),
            sample_rate=sr,
        )

    @staticmethod
    def save(audio: AudioData, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio.waveform, audio.sample_rate)

    @staticmethod
    def discover(root: str) -> list[Path]:
        """Every decodable audio file under a directory tree, sorted."""

        base = Path(root)

        if not base.exists():
            raise AudioLoadError(f"Directory not found: {root}")

        files = [
            p
            for p in sorted(base.rglob("*"))
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
        ]

        return files


def normalize_peak(waveform: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.abs(waveform).max())

    if peak < 1e-6:
        return waveform

    return waveform * (target / peak)


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform

    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


def trim_silence(
    waveform: np.ndarray,
    top_db: float = 35.0,
) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)

    return trimmed if trimmed.size else waveform


def slice_audio(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = 3.0,
    overlap_seconds: float = 0.3,
    min_seconds: float = 0.5,
) -> list[np.ndarray]:
    """Cut a long recording into overlapping training-sized chunks."""

    chunk = int(chunk_seconds * sample_rate)
    hop = max(1, chunk - int(overlap_seconds * sample_rate))
    minimum = int(min_seconds * sample_rate)

    if waveform.size <= chunk:
        return [waveform] if waveform.size >= minimum else []

    pieces: list[np.ndarray] = []

    for start in range(0, waveform.size, hop):
        piece = waveform[start : start + chunk]

        if piece.size < minimum:
            break

        pieces.append(piece)

        if start + chunk >= waveform.size:
            break

    return pieces


def split_on_silence(
    waveform: np.ndarray,
    sample_rate: int,
    top_db: float = 35.0,
    min_seconds: float = 0.5,
    max_seconds: float = 4.0,
) -> list[np.ndarray]:
    """Prefer cutting at silences; fall back to fixed chunks when too long."""

    intervals = librosa.effects.split(waveform, top_db=top_db)
    limit = int(max_seconds * sample_rate)
    minimum = int(min_seconds * sample_rate)

    pieces: list[np.ndarray] = []

    for start, end in intervals:
        segment = waveform[start:end]

        if segment.size < minimum:
            continue

        if segment.size <= limit:
            pieces.append(segment)
            continue

        pieces.extend(
            slice_audio(
                segment,
                sample_rate,
                chunk_seconds=max_seconds,
                overlap_seconds=0.0,
                min_seconds=min_seconds,
            )
        )

    return pieces


def rms_match(
    source: np.ndarray,
    target: np.ndarray,
    rate: float,
    frame_length: int = 1024,
) -> np.ndarray:
    """Blend the target's loudness envelope toward the source's.

    rate == 0 keeps the source envelope entirely, rate == 1 keeps the model's.
    """

    if rate >= 1.0:
        return target

    import torch
    import torch.nn.functional as F

    src_rms = librosa.feature.rms(
        y=source, frame_length=frame_length, hop_length=frame_length // 2
    )
    tgt_rms = librosa.feature.rms(
        y=target, frame_length=frame_length, hop_length=frame_length // 2
    )

    def _stretch(values: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(values).float().unsqueeze(0)
        return F.interpolate(
            tensor, size=target.shape[0], mode="linear", align_corners=True
        ).squeeze()

    src = _stretch(src_rms)
    tgt = torch.maximum(_stretch(tgt_rms), torch.zeros(1) + 1e-6)

    gain = (src / tgt) ** (1.0 - rate)

    return (torch.from_numpy(target).float() * gain).numpy()
