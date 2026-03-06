"""Vector Store — thread-safe similarity-search store backed by dense float vectors.

Stores embedding vectors in memory and supports efficient top-k cosine
similarity retrieval.  For production-scale workloads the class can be
subclassed to delegate to FAISS, Annoy, Weaviate, Pinecone, or any
hosted vector database.

Example::

    from Victor_Synthetic_Super_Intelligence.memory.vector_store import VectorStore

    vs = VectorStore(dimension=4)
    vs.add("doc:1", [0.1, 0.2, 0.3, 0.4], metadata={"text": "hello"})
    vs.add("doc:2", [0.9, 0.8, 0.7, 0.6])

    results = vs.query([0.1, 0.2, 0.3, 0.4], top_k=2)
    # [("doc:1", 1.0, {...}), ("doc:2", 0.97, {})]

    stats = vs.stats()
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

from ..exceptions import VectorDimensionError

logger = logging.getLogger(__name__)


class VectorStore:
    """Thread-safe in-memory nearest-neighbour store for embedding vectors.

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
        self._lock = threading.RLock()
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
            VectorDimensionError: If *vector* length does not match the
                store's dimension.
            ValueError: If *vector* is empty.
        """
        if not vector:
            raise ValueError("Cannot add an empty vector.")
        with self._lock:
            if self.dimension is None:
                self.dimension = len(vector)
            if len(vector) != self.dimension:
                raise VectorDimensionError(
                    f"Vector length {len(vector)} does not match "
                    f"store dimension {self.dimension}."
                )
            self._vectors[key] = list(vector)
            self._metadata[key] = metadata or {}
        logger.debug("Added vector for key '%s'", key)

    def remove(self, key: str) -> bool:
        """Delete a vector entry by key.

        Args:
            key: The key to remove.

        Returns:
            ``True`` if the key existed and was removed, ``False`` otherwise.
        """
        with self._lock:
            if key in self._vectors:
                del self._vectors[key]
                del self._metadata[key]
                return True
        return False

    def clear(self) -> int:
        """Remove all vectors from the store.

        The inferred dimension is *not* reset so the store remains
        consistent if new vectors are added later.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            count = len(self._vectors)
            self._vectors.clear()
            self._metadata.clear()
        logger.info("VectorStore cleared (%d entries removed)", count)
        return count

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
            descending similarity.  Returns an empty list if the store is
            empty or the query vector is all-zeros.
        """
        with self._lock:
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
                (key, round(score, 6), self._metadata[key])
                for key, score in scores[:top_k]
            ]

    def get(self, key: str) -> list[float] | None:
        """Retrieve a specific vector by key.

        Args:
            key: The key to look up.

        Returns:
            The stored float vector or ``None`` if not found.
        """
        with self._lock:
            vec = self._vectors.get(key)
            return list(vec) if vec is not None else None

    def stats(self) -> dict[str, Any]:
        """Return operational statistics for this store.

        Returns:
            Dict with keys ``total_vectors`` and ``dimension``.
        """
        with self._lock:
            return {
                "total_vectors": len(self._vectors),
                "dimension": self.dimension,
            }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._vectors)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._vectors
