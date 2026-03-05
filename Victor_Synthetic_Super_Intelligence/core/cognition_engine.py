"""Cognition Engine — orchestrates high-level reasoning and perception."""

from __future__ import annotations

import logging
from typing import Any

from .tensor_operations import TensorOperations
from .reasoning_loop import ReasoningLoop

logger = logging.getLogger(__name__)


class CognitionEngine:
    """Central cognition engine that integrates perception, reasoning, and action.

    The engine coordinates the :class:`ReasoningLoop` and
    :class:`TensorOperations` components to produce a coherent response or
    action for a given input stimulus.

    Attributes:
        config (dict): Runtime configuration dictionary.
        tensor_ops (TensorOperations): Shared tensor utilities.
        reasoning_loop (ReasoningLoop): The active reasoning loop.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.tensor_ops = TensorOperations()
        self.reasoning_loop = ReasoningLoop(tensor_ops=self.tensor_ops)
        logger.info("CognitionEngine initialised with config: %s", self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def perceive(self, stimulus: Any) -> Any:
        """Convert a raw stimulus into an internal representation.

        Args:
            stimulus: Any raw input (text, numeric data, dict, …).

        Returns:
            A normalised internal representation ready for reasoning.
        """
        logger.debug("Perceiving stimulus: %s", stimulus)
        return self.tensor_ops.encode(stimulus)

    def reason(self, representation: Any, context: dict[str, Any] | None = None) -> Any:
        """Run the reasoning loop over an internal representation.

        Args:
            representation: Internal representation produced by :meth:`perceive`.
            context: Optional contextual information (memory snapshots, etc.).

        Returns:
            The result of the reasoning process.
        """
        return self.reasoning_loop.run(representation, context=context or {})

    def process(self, stimulus: Any, context: dict[str, Any] | None = None) -> Any:
        """End-to-end pipeline: perceive → reason → return result.

        Args:
            stimulus: Raw input stimulus.
            context: Optional contextual dictionary.

        Returns:
            Final processed output.
        """
        representation = self.perceive(stimulus)
        return self.reason(representation, context=context)
