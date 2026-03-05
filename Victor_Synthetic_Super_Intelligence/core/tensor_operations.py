"""Tensor Operations — lightweight numeric utilities for the cognition stack.

This module intentionally avoids a hard dependency on any deep-learning
framework so that the codebase can be imported without GPU drivers or large
framework installations.  Heavier backends (PyTorch, JAX, etc.) can be
plugged in by sub-classing :class:`TensorOperations` and overriding the
relevant methods.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


class TensorOperations:
    """Provides core tensor / numeric operations used across the system.

    All methods operate on plain Python lists (1-D or 2-D) so that the
    module works without third-party dependencies.  Subclasses may override
    individual methods to use NumPy, PyTorch, or JAX.
    """

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode(self, value: Any) -> list[float]:
        """Encode an arbitrary *value* into a flat list of floats.

        Supported types
        ---------------
        * ``str`` — character-level unicode ordinals, L2-normalised
        * ``int`` / ``float`` — wrapped in a single-element list
        * ``list`` / ``tuple`` — each element cast to ``float``
        * Everything else — ``[0.0]`` fallback

        Args:
            value: The value to encode.

        Returns:
            A list of floats representing the encoded value.
        """
        if isinstance(value, str):
            raw = [float(ord(c)) for c in value] if value else [0.0]
        elif isinstance(value, (int, float)):
            raw = [float(value)]
        elif isinstance(value, (list, tuple)):
            raw = [float(x) for x in value]
        else:
            raw = [0.0]
        return self.normalize(raw)

    # ------------------------------------------------------------------
    # Basic linear algebra
    # ------------------------------------------------------------------

    def normalize(self, vector: Sequence[float]) -> list[float]:
        """L2-normalise *vector*.

        Args:
            vector: Input 1-D sequence of floats.

        Returns:
            Unit-length vector (or the zero vector if the norm is zero).
        """
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return list(vector)
        return [x / norm for x in vector]

    def dot_product(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Compute the dot product of two equal-length vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Scalar dot product.

        Raises:
            ValueError: If ``a`` and ``b`` have different lengths.
        """
        if len(a) != len(b):
            raise ValueError(
                f"Vectors must have the same length, got {len(a)} and {len(b)}."
            )
        return sum(x * y for x, y in zip(a, b))

    def cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Float in ``[-1, 1]``.  Returns ``0.0`` if either vector is zero.
        """
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return self.dot_product(a, b) / (norm_a * norm_b)

    def softmax(self, logits: Sequence[float]) -> list[float]:
        """Numerically stable softmax.

        Args:
            logits: Unnormalised log-probabilities.

        Returns:
            Probability distribution over the same indices.
        """
        max_val = max(logits)
        exps = [math.exp(x - max_val) for x in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def matmul(
        self,
        matrix_a: Sequence[Sequence[float]],
        matrix_b: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """Multiply two 2-D matrices (row-major lists of lists).

        Args:
            matrix_a: Left-hand matrix of shape ``(m, k)``.
            matrix_b: Right-hand matrix of shape ``(k, n)``.

        Returns:
            Result matrix of shape ``(m, n)``.

        Raises:
            ValueError: If inner dimensions do not match.
        """
        rows_a, cols_a = len(matrix_a), len(matrix_a[0])
        rows_b, cols_b = len(matrix_b), len(matrix_b[0])
        if cols_a != rows_b:
            raise ValueError(
                f"Incompatible matrix dimensions: ({rows_a}, {cols_a}) x ({rows_b}, {cols_b})."
            )
        result = [[0.0] * cols_b for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                result[i][j] = sum(matrix_a[i][k] * matrix_b[k][j] for k in range(cols_a))
        return result
