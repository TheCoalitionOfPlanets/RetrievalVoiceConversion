"""F0 (pitch) extraction and quantisation.

The name is kept for continuity with the RVC lineage, but the default engine is
pyworld `harvest` — it needs no extra model weights and is the most robust of
the classical estimators. `dio` is the fast option; `crepe` is available when
the optional torchcrepe extra is installed.
"""

import numpy as np

from src.models.audio import AudioData
from src.models.pitch import PitchFeatures
from src.utils.exceptions import FeatureExtractionError
from src.utils.logger import get_logger
from src.utils.settings import PitchConfig

logger = get_logger(__name__)

SUPPORTED_METHODS = ("harvest", "dio", "crepe")


class PitchExtractor:
    """Produces a frame-aligned F0 contour plus its 256-bin coarse encoding."""

    def __init__(self, config: PitchConfig | None = None):
        self.config = config or PitchConfig()

        if self.config.method not in SUPPORTED_METHODS:
            raise FeatureExtractionError(
                f"Unknown f0 method '{self.config.method}'; "
                f"expected one of {SUPPORTED_METHODS}"
            )

        # Mel-scale bin edges, matching the RVC coarse-pitch convention.
        self.f0_mel_min = 1127.0 * np.log(1.0 + self.config.f0_min / 700.0)
        self.f0_mel_max = 1127.0 * np.log(1.0 + self.config.f0_max / 700.0)

    def load(self) -> "PitchExtractor":
        return self

    def extract(
        self,
        audio: AudioData,
        num_frames: int | None = None,
        pitch_shift: int = 0,
        method: str | None = None,
    ) -> PitchFeatures:
        """Returns F0 in Hz resampled to `num_frames`, plus coarse indices."""

        method = method or self.config.method

        if audio.sample_rate != self.config.sample_rate:
            raise FeatureExtractionError(
                f"Pitch extraction expects {self.config.sample_rate} Hz, "
                f"got {audio.sample_rate}"
            )

        waveform = audio.waveform.astype(np.double)

        if method == "harvest":
            f0 = self._harvest(waveform)
        elif method == "dio":
            f0 = self._dio(waveform)
        else:
            f0 = self._crepe(audio)

        if pitch_shift:
            f0 = f0 * (2.0 ** (pitch_shift / 12.0))

        if num_frames is not None:
            f0 = self._resize(f0, num_frames)

        return PitchFeatures(
            f0=f0.astype(np.float32),
            f0_coarse=self.to_coarse(f0),
        )

    def _harvest(self, waveform: np.ndarray) -> np.ndarray:
        import pyworld

        frame_period = 1000.0 * self.config.hop_length / self.config.sample_rate

        f0, t = pyworld.harvest(
            waveform,
            fs=self.config.sample_rate,
            f0_floor=self.config.f0_min,
            f0_ceil=self.config.f0_max,
            frame_period=frame_period,
        )

        return pyworld.stonemask(waveform, f0, t, self.config.sample_rate)

    def _dio(self, waveform: np.ndarray) -> np.ndarray:
        import pyworld

        frame_period = 1000.0 * self.config.hop_length / self.config.sample_rate

        f0, t = pyworld.dio(
            waveform,
            fs=self.config.sample_rate,
            f0_floor=self.config.f0_min,
            f0_ceil=self.config.f0_max,
            frame_period=frame_period,
        )

        return pyworld.stonemask(waveform, f0, t, self.config.sample_rate)

    def _crepe(self, audio: AudioData) -> np.ndarray:
        try:
            import torch
            import torchcrepe
        except ImportError as exc:
            raise FeatureExtractionError(
                "f0_method='crepe' needs the optional extra: "
                "uv sync --extra crepe"
            ) from exc

        waveform = torch.from_numpy(audio.waveform).float().unsqueeze(0)

        pitch = torchcrepe.predict(
            waveform,
            audio.sample_rate,
            hop_length=self.config.hop_length,
            fmin=self.config.f0_min,
            fmax=self.config.f0_max,
            model="full",
            batch_size=512,
            device="cpu",
            return_periodicity=False,
        )

        return pitch.squeeze(0).numpy().astype(np.double)

    @staticmethod
    def _resize(f0: np.ndarray, num_frames: int) -> np.ndarray:
        """Nearest-neighbour resample; linear would invent pitch in unvoiced gaps."""

        if f0.size == num_frames:
            return f0

        if f0.size == 0:
            return np.zeros(num_frames, dtype=np.double)

        idx = np.clip(
            np.round(np.linspace(0, f0.size - 1, num_frames)).astype(int),
            0,
            f0.size - 1,
        )

        return f0[idx]

    def to_coarse(self, f0: np.ndarray) -> np.ndarray:
        """Quantise Hz into [1, f0_bin - 1]; 0 is reserved for unvoiced."""

        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        bins = self.config.f0_bin

        scaled = (f0_mel - self.f0_mel_min) * (bins - 2) / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1.0

        scaled[f0_mel <= 0] = 1.0
        scaled[scaled > bins - 1] = bins - 1

        coarse = np.rint(scaled).astype(np.int64)
        coarse[f0 < 1e-3] = 0

        if coarse.size and (coarse.max() > bins - 1 or coarse.min() < 0):
            raise FeatureExtractionError(
                f"Coarse f0 out of range: [{coarse.min()}, {coarse.max()}]"
            )

        return coarse


# Backwards-compatible alias for the original module name.
RMVPEExtractor = PitchExtractor
