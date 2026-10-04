"""Blends extracted content features with retrieved target-speaker features."""

import numpy as np

from src.retrieval.faiss_index import FaissIndex
from src.utils.logger import get_logger
from src.utils.settings import IndexConfig

logger = get_logger(__name__)


class RetrievalService:

    def __init__(self, config: IndexConfig | None = None):
        self.config = config or IndexConfig()
        self.index: FaissIndex | None = None

    def load(self) -> "RetrievalService":
        from pathlib import Path

        if not Path(self.config.path).exists():
            logger.warning(
                f"No index at {self.config.path}; retrieval disabled "
                f"(conversion still runs, timbre will be weaker)"
            )
            return self

        self.index = FaissIndex.load(self.config.path, self.config.features)

        return self

    @property
    def is_available(self) -> bool:
        return self.index is not None

    def retrieve(
        self,
        embeddings: np.ndarray,
        rate: float | None = None,
        k: int | None = None,
    ) -> np.ndarray:
        """embeddings: (T, D). Returns the same shape, blended toward the index.

        rate == 0 is a pass-through; rate == 1 fully replaces each frame with
        its retrieved neighbours.
        """

        rate = self.config.rate if rate is None else rate
        k = k or self.config.k

        if self.index is None or rate <= 0.0:
            return embeddings

        retrieved = self.index.search(embeddings, k=k)

        return (rate * retrieved + (1.0 - rate) * embeddings).astype(np.float32)
