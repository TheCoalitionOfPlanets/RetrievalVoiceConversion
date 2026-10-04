from dataclasses import dataclass

import numpy as np


@dataclass
class ConversionRequest:

    input_audio_path: str

    output_audio_path: str

    speaker_id: int = 0

    pitch_shift: int = 0

    index_rate: float | None = None

    protect: float | None = None

    f0_method: str | None = None


@dataclass
class ConversionResult:

    waveform: np.ndarray

    sample_rate: int

    output_path: str

    duration: float

    elapsed_seconds: float

    @property
    def realtime_factor(self) -> float:
        if self.duration <= 0:
            return 0.0
        return self.elapsed_seconds / self.duration
