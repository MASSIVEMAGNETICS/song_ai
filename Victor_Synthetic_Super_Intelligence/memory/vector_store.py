"""Vector Store — similarity-search store backed by dense float vectors."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class VectorStore:
    """An in-memory nearest-neighbour store for embedding vectors.

    Vectors are stored as plain Python lists of floats so the module
    works without NumPy or a dedicated vector database.  For production
    workloads this class can be subclassed to delegate to FAISS, Annoy,
    or a hosted vector DB.

    Args:
        dimension: Expected dimensionality of every stored vector.  If
            ``None``, the dimension is inferred from the first insertion.
    """

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension: int | None = dimension
        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        logger.info("VectorStore initialised (dimension=%s)", dimension)

    # ------------------------------------------------------------------
    # Insertion / deletion
    # ------------------------------------------------------------------

    def add(
        self,
        key: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or overwrite a vector entry.

        Args:
            key: Unique identifier for this entry.
            vector: Dense float vector.
            metadata: Arbitrary annotations stored alongside the vector.

        Raises:
            ValueError: If *vector* length does not match *dimension*.
        """
        if self.dimension is None:
            self.dimension = len(vector)
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector length {len(vector)} does not match store dimension {self.dimension}."
            )
        self._vectors[key] = vector
        self._metadata[key] = metadata or {}
        logger.debug("Added vector for key '%s'", key)

    def remove(self, key: str) -> bool:
        """Delete a vector entry by key.

        Returns:
            ``True`` if the key existed, ``False`` otherwise.
        """
        if key in self._vectors:
            del self._vectors[key]
            del self._metadata[key]
            return True
        return False

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return the *top_k* most similar entries by cosine similarity.

        Args:
            query_vector: Query vector of the same dimension as the store.
            top_k: Number of results to return.

        Returns:
            List of ``(key, similarity_score, metadata)`` tuples ordered by
            descending similarity.
        """
        if not self._vectors:
            return []

        query_norm = math.sqrt(sum(x * x for x in query_vector))
        if query_norm == 0.0:
            return []

        scores: list[tuple[str, float]] = []
        for key, vec in self._vectors.items():
            vec_norm = math.sqrt(sum(x * x for x in vec))
            if vec_norm == 0.0:
                score = 0.0
            else:
                dot = sum(a * b for a, b in zip(query_vector, vec))
                score = dot / (query_norm * vec_norm)
            scores.append((key, score))

        scores.sort(key=lambda t: t[1], reverse=True)
        return [
            (key, score, self._metadata[key])
            for key, score in scores[:top_k]
        ]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, key: object) -> bool:
        return key in self._vectors
