from dataclasses import dataclass

import numpy as np


@dataclass
class HubertFeatures:
    """Content embeddings, shaped (frames, embedding_dim)."""

    embeddings: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[-1])
