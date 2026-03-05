"""Reasoning Loop — iterative inference and self-reflection mechanism."""

from __future__ import annotations

import logging
from typing import Any

from .tensor_operations import TensorOperations

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 5
_CONVERGENCE_THRESHOLD = 1e-4


class ReasoningLoop:
    """Executes an iterative reasoning process until convergence or step limit.

    The loop applies a lightweight update rule at each step, comparing the
    current state to the previous one.  When the L2 distance between
    consecutive states falls below *convergence_threshold* the loop
    terminates early.

    Args:
        tensor_ops: Shared :class:`~core.tensor_operations.TensorOperations`
            instance.
        max_iterations: Maximum number of reasoning iterations.
        convergence_threshold: Early-stop criterion on state delta.
    """

    def __init__(
        self,
        tensor_ops: TensorOperations | None = None,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        convergence_threshold: float = _CONVERGENCE_THRESHOLD,
    ) -> None:
        self.tensor_ops = tensor_ops or TensorOperations()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        initial_state: list[float],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the reasoning loop.

        Args:
            initial_state: Encoded representation to reason over.
            context: Optional key/value context injected at each step.

        Returns:
            A dict containing:
            * ``"result"`` — final state vector after reasoning.
            * ``"iterations"`` — number of steps executed.
            * ``"converged"`` — whether early convergence was triggered.
        """
        context = context or {}
        state = list(initial_state)
        iterations = 0
        converged = False

        logger.debug("ReasoningLoop starting with state length %d", len(state))

        for step in range(self.max_iterations):
            iterations += 1
            new_state = self._step(state, step=step, context=context)

            delta = self._l2_distance(state, new_state)
            logger.debug("Step %d — delta=%.6f", step, delta)

            state = new_state
            if delta < self.convergence_threshold:
                converged = True
                logger.debug("Converged at step %d", step)
                break

        return {
            "result": state,
            "iterations": iterations,
            "converged": converged,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step(
        self,
        state: list[float],
        step: int,
        context: dict[str, Any],
    ) -> list[float]:
        """Apply a single reasoning update to *state*.

        The default implementation performs a lightweight normalised weighted
        average that gradually dampens extreme values, simulating an attention
        mechanism without external dependencies.

        Args:
            state: Current state vector.
            step: Current step index (0-based).
            context: Contextual key/value pairs.

        Returns:
            Updated state vector of the same length.
        """
        decay = 1.0 / (1.0 + step)
        updated = [x * (1.0 - decay) + decay * 0.5 for x in state]
        return self.tensor_ops.normalize(updated)

    @staticmethod
    def _l2_distance(a: list[float], b: list[float]) -> float:
        """Euclidean distance between two equal-length vectors."""
        if len(a) != len(b):
            return float("inf")
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
