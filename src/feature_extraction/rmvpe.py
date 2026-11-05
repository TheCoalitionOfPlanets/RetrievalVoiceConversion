"""F0 (pitch) extraction and quantisation.

Three engines: `rmvpe` (neural, DeepUnet+BiGRU — matches upstream RVC's
pitch accuracy, needs the rmvpe.pt checkpoint and is best on a GPU), and the
classical pyworld estimators `harvest` (robust, no extra weights) and `dio`
(fast). `crepe` is available when the optional torchcrepe extra is installed.
"""

import numpy as np
import torch

from src.models.audio import AudioData
from src.models.pitch import PitchFeatures
from src.utils.exceptions import FeatureExtractionError
from src.utils.logger import get_logger
from src.utils.settings import PitchConfig

logger = get_logger(__name__)

SUPPORTED_METHODS = ("harvest", "dio", "crepe", "rmvpe")


class PitchExtractor:
    """Produces a frame-aligned F0 contour plus its 256-bin coarse encoding."""

    def __init__(
        self,
        config: PitchConfig | None = None,
        device: torch.device | str = "cpu",
        fp16: bool = False,
    ):
        self.config = config or PitchConfig()
        self.device = torch.device(device)
        self.fp16 = fp16
        self._rmvpe_model = None

        if self.config.method not in SUPPORTED_METHODS:
            raise FeatureExtractionError(
                f"Unknown f0 method '{self.config.method}'; "
                f"expected one of {SUPPORTED_METHODS}"
            )

        # Mel-scale bin edges, matching the RVC coarse-pitch convention.
        self.f0_mel_min = 1127.0 * np.log(1.0 + self.config.f0_min / 700.0)
        self.f0_mel_max = 1127.0 * np.log(1.0 + self.config.f0_max / 700.0)

    def load(self) -> "PitchExtractor":
        if self.config.method == "rmvpe":
            self._load_rmvpe()

        return self

    def _load_rmvpe(self):
        if self._rmvpe_model is not None:
            return

        from pathlib import Path

        from src.feature_extraction.rmvpe_model import RMVPEModel

        model_path = Path(self.config.rmvpe_model_path)
        if not model_path.is_file():
            raise FeatureExtractionError(
                f"RMVPE checkpoint not found: {model_path}. "
                "Copy rmvpe.pt there, or use --f0-method harvest."
            )

        logger.info(f"Loading RMVPE: {model_path}")
        self._rmvpe_model = RMVPEModel(str(model_path), self.device, fp16=self.fp16)

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
        elif method == "rmvpe":
            f0 = self._rmvpe(audio)
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

    def _rmvpe(self, audio: AudioData) -> np.ndarray:
        self._load_rmvpe()

        f0 = self._rmvpe_model.infer_from_audio(
            audio.waveform.astype(np.float32), thred=0.03
        )

        return f0.astype(np.double)

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

        # stonemask refines harvest's estimate and can land just under f0_min,
        # which underflows the first bin to 0 or -1. Clamp both ends.
        scaled[f0_mel <= 0] = 1.0
        scaled[scaled < 1.0] = 1.0
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
