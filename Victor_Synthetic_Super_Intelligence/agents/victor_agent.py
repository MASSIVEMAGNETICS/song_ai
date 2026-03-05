"""Victor Agent — the primary autonomous reasoning agent."""

from __future__ import annotations

import logging
from typing import Any

from ..core.cognition_engine import CognitionEngine
from ..memory.long_term_memory import LongTermMemory
from ..memory.episodic_memory import EpisodicMemory
from ..memory.vector_store import VectorStore
from .task_executor import TaskExecutor

logger = logging.getLogger(__name__)


class VictorAgent:
    """The main agent that perceives, reasons, remembers, and acts.

    ``VictorAgent`` wires together the :class:`~core.CognitionEngine`,
    memory subsystems, and :class:`TaskExecutor` into a single
    conversational / task-oriented agent loop.

    Args:
        config: Optional configuration dictionary.  Recognised keys:

            ``max_ltm_entries``
                Maximum entries for long-term memory (default: 10 000).

            ``episodic_capacity``
                Number of episodes to retain (default: 1 000).

            ``vector_dim``
                Expected embedding dimension (default: ``None`` — inferred).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.cognition = CognitionEngine(config=self.config)
        self.long_term_memory = LongTermMemory(
            max_entries=self.config.get("max_ltm_entries", 10_000)
        )
        self.episodic_memory = EpisodicMemory(
            capacity=self.config.get("episodic_capacity", 1_000)
        )
        self.vector_store = VectorStore(
            dimension=self.config.get("vector_dim")
        )
        self.task_executor = TaskExecutor(agent=self)
        logger.info("VictorAgent initialised")

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def respond(self, stimulus: Any) -> Any:
        """Process a *stimulus* and return a response.

        The method:

        1. Retrieves recent context from episodic memory.
        2. Runs the full cognition pipeline.
        3. Records the episode.
        4. Returns the final result.

        Args:
            stimulus: Any raw input to process.

        Returns:
            The agent's response (dict with reasoning metadata).
        """
        context = self._build_context()
        result = self.cognition.process(stimulus, context=context)
        self.episodic_memory.record(stimulus=stimulus, response=result)
        self._update_vector_store(stimulus, result)
        logger.debug("VictorAgent.respond result: %s", result)
        return result

    def remember(self, key: str, value: Any) -> None:
        """Explicitly store a fact in long-term memory.

        Args:
            key: Unique identifier.
            value: Data to store.
        """
        self.long_term_memory.store(key, value)

    def recall(self, key: str) -> Any:
        """Retrieve a fact from long-term memory.

        Args:
            key: Identifier to look up.

        Returns:
            Stored value or ``None``.
        """
        return self.long_term_memory.retrieve(key)

    def execute_task(self, task: dict[str, Any]) -> Any:
        """Delegate a structured task to the :class:`TaskExecutor`.

        Args:
            task: Task specification dict.  Must contain at minimum a
                ``"type"`` key.

        Returns:
            Task result.
        """
        return self.task_executor.execute(task)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self) -> dict[str, Any]:
        """Assemble a context dict from recent episodes."""
        recent = self.episodic_memory.recent(n=5)
        return {"recent_episodes": [ep.to_dict() for ep in recent]}

    def _update_vector_store(self, stimulus: Any, result: dict[str, Any]) -> None:
        """Index the result vector in the vector store."""
        vector = result.get("result")
        if isinstance(vector, list) and vector:
            key = f"ep_{len(self.episodic_memory)}"
            try:
                self.vector_store.add(key, vector)
            except ValueError:
                pass
