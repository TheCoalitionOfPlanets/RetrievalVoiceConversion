from dataclasses import dataclass
import numpy as np


@dataclass
class HubertFeatures:
    embeddings: np.ndarray