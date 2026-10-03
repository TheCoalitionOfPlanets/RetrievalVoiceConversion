from dataclasses import dataclass
import numpy as np


@dataclass
class AudioData:
    waveform: np.ndarray
    sample_rate: int