from dataclasses import dataclass
import numpy as np


@dataclass
class PitchFeatures:
    f0: np.ndarray