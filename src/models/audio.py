from dataclasses import dataclass

import numpy as np


@dataclass
class AudioData:
    waveform: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return float(self.waveform.shape[-1]) / self.sample_rate

    @property
    def num_samples(self) -> int:
        return int(self.waveform.shape[-1])
