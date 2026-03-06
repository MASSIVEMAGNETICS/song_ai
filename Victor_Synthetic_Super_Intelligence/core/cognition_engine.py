"""Cognition Engine — orchestrates high-level reasoning and perception.

The :class:`CognitionEngine` is the central processing unit of Victor SSI.
It integrates perception (stimulus encoding), iterative reasoning, and
optional performance instrumentation.

Example::

    from Victor_Synthetic_Super_Intelligence.core.cognition_engine import CognitionEngine

    engine = CognitionEngine(config={"max_iterations": 10})
    result = engine.process("Hello, world!")
    print(result["iterations"], result["converged"])
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .tensor_operations import TensorOperations
from .reasoning_loop import ReasoningLoop

logger = logging.getLogger(__name__)


class CognitionEngine:
    """Central cognition engine that integrates perception, reasoning, and action.

    The engine coordinates the :class:`ReasoningLoop` and
    :class:`TensorOperations` components to produce a coherent response or
    action for a given input stimulus.

    Performance timings are collected for every ``process()`` call and
    exposed via :attr:`last_timing`.

    Attributes:
        config (dict): Runtime configuration dictionary.
        tensor_ops (TensorOperations): Shared tensor utilities.
        reasoning_loop (ReasoningLoop): The active reasoning loop.
        last_timing (dict[str, float]): Timing breakdown (seconds) of the
            most recent :meth:`process` call.  Keys: ``perceive``,
            ``reason``, ``total``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.tensor_ops = TensorOperations()
        self.reasoning_loop = ReasoningLoop(
            tensor_ops=self.tensor_ops,
            max_iterations=self.config.get("max_iterations", 5),
            convergence_threshold=self.config.get("convergence_threshold", 1e-4),
        )
        self.last_timing: dict[str, float] = {}
        logger.info("CognitionEngine initialised with config: %s", self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def perceive(self, stimulus: Any) -> list[float]:
        """Convert a raw stimulus into an internal vector representation.

        The encoding is delegated to :meth:`TensorOperations.encode`.
        Supported types include strings, numbers, and lists of floats.

        Args:
            stimulus: Any raw input (text, numeric data, dict, …).

        Returns:
            A normalised list of floats ready for reasoning.
        """
        logger.debug("Perceiving stimulus (type=%s): %r", type(stimulus).__name__, stimulus)
        return self.tensor_ops.encode(stimulus)

    def reason(
        self,
        representation: list[float],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the reasoning loop over an internal representation.

        Args:
            representation: Internal vector produced by :meth:`perceive`.
            context: Optional contextual information (memory snapshots, etc.).

        Returns:
            The result dict from :meth:`ReasoningLoop.run`, containing keys
            ``result``, ``iterations``, and ``converged``.
        """
        return self.reasoning_loop.run(representation, context=context or {})

    def process(
        self,
        stimulus: Any,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """End-to-end pipeline: perceive → reason → return result.

        Timing breakdowns for each stage are stored in :attr:`last_timing`
        after every call.

        Args:
            stimulus: Raw input stimulus.
            context: Optional contextual dictionary.

        Returns:
            Final processed output dict with ``result``, ``iterations``,
            ``converged``, and ``timing`` keys.
        """
        t0 = time.perf_counter()
        representation = self.perceive(stimulus)
        t1 = time.perf_counter()

        result = self.reason(representation, context=context)
        t2 = time.perf_counter()

        self.last_timing = {
            "perceive": round(t1 - t0, 6),
            "reason": round(t2 - t1, 6),
            "total": round(t2 - t0, 6),
        }
        result["timing"] = self.last_timing
        return result
