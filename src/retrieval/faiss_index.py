"""FAISS index over the training speaker's HuBERT features.

This is what makes RVC *retrieval*-based: at inference each content frame is
replaced by a blend of its nearest neighbours from the target speaker's own
feature bank, which pulls timbre toward the target and suppresses artefacts the
decoder would otherwise invent for unfamiliar content.
"""

from pathlib import Path

import numpy as np

from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RetrievalError(
            "faiss is required for retrieval; install with: uv sync"
        ) from exc

    return faiss


class FaissIndex:
    """Build / save / load / search an IVF-Flat index of content features."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.index = None
        self.features: np.ndarray | None = None

    @staticmethod
    def choose_nlist(num_vectors: int) -> int:
        """FAISS wants >= ~39 training points per centroid to behave."""

        return max(1, min(int(16 * np.sqrt(num_vectors)), num_vectors // 39))

    def build(self, features: np.ndarray, nlist: int | None = None) -> "FaissIndex":
        faiss = _require_faiss()

        if features.ndim != 2:
            raise RetrievalError(f"Expected (N, D) features, got {features.shape}")

        features = np.ascontiguousarray(features.astype(np.float32))
        n, dim = features.shape

        if n < 2:
            raise RetrievalError(f"Need at least 2 feature vectors, got {n}")

        self.dim = dim
        nlist = nlist or self.choose_nlist(n)

        logger.info(f"Building IVF index: {n} vectors, dim {dim}, nlist {nlist}")

        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

        index.train(features)
        index.add(features)
        index.nprobe = max(1, nlist // 8)

        self.index = index
        self.features = features

        return self

    def save(self, index_path: str | Path, features_path: str | Path | None = None):
        faiss = _require_faiss()

        if self.index is None:
            raise RetrievalError("Nothing to save; build the index first")

        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        logger.info(f"Wrote index -> {index_path}")

        if features_path is not None and self.features is not None:
            features_path = Path(features_path)
            features_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(features_path, self.features)
            logger.info(f"Wrote feature bank -> {features_path}")

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        features_path: str | Path | None = None,
    ) -> "FaissIndex":
        faiss = _require_faiss()

        index_path = Path(index_path)

        if not index_path.exists():
            raise RetrievalError(f"Index not found: {index_path}")

        instance = cls()
        instance.index = faiss.read_index(str(index_path))
        instance.dim = instance.index.d

        if features_path is not None and Path(features_path).exists():
            instance.features = np.load(features_path).astype(np.float32)
        else:
            # Reconstruct the bank from the index so blending still works.
            try:
                instance.features = instance.index.reconstruct_n(
                    0, instance.index.ntotal
                ).astype(np.float32)
            except RuntimeError as exc:
                raise RetrievalError(
                    f"Index cannot reconstruct vectors and no feature bank was "
                    f"supplied: {exc}"
                ) from exc

        logger.info(
            f"Loaded index {index_path.name} "
            f"({instance.index.ntotal} vectors, dim {instance.dim})"
        )

        return instance

    def search(self, queries: np.ndarray, k: int = 8) -> np.ndarray:
        """Inverse-distance-weighted mean of the k nearest stored vectors."""

        if self.index is None or self.features is None:
            raise RetrievalError("Index not loaded")

        queries = np.ascontiguousarray(queries.astype(np.float32))

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(queries, k)

        # Guard against FAISS's -1 padding when fewer than k neighbours exist.
        indices = np.where(indices < 0, 0, indices)

        weights = 1.0 / np.maximum(distances, 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)

        neighbours = self.features[indices]

        return np.sum(neighbours * weights[..., None], axis=1).astype(np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0
