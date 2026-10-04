from dataclasses import dataclass

import numpy as np


@dataclass
class PitchFeatures:
    """Continuous F0 in Hz plus its quantised (coarse) index form."""

    f0: np.ndarray
    f0_coarse: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.f0.shape[0])

    @property
    def voiced_ratio(self) -> float:
        if self.f0.size == 0:
            return 0.0
        return float((self.f0 > 0).mean())
