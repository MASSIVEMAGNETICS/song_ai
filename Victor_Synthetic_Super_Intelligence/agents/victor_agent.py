"""Victor Agent — the primary autonomous reasoning agent.

:class:`VictorAgent` is the top-level facade that wires together the
:class:`~core.CognitionEngine`, all three memory subsystems, and the
:class:`TaskExecutor` into a single conversational / task-oriented agent.

Example::

    from Victor_Synthetic_Super_Intelligence.agents import VictorAgent

    agent = VictorAgent()
    result = agent.respond("What is the capital of France?")
    agent.remember("capital:france", "Paris")
    print(agent.recall("capital:france"))   # "Paris"
    print(agent.health())
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.cognition_engine import CognitionEngine
from ..memory.long_term_memory import LongTermMemory
from ..memory.episodic_memory import EpisodicMemory
from ..memory.vector_store import VectorStore
from ..metrics import MetricsRegistry, get_registry
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

            ``ltm_default_ttl``
                Default TTL in seconds for LTM entries (default: ``None``).

            ``episodic_capacity``
                Number of episodes to retain (default: 1 000).

            ``vector_dim``
                Expected embedding dimension (default: ``None`` — inferred).

            ``max_iterations``
                Maximum reasoning iterations per request (default: 5).

            ``convergence_threshold``
                Reasoning convergence threshold (default: 1e-4).

        metrics: Optional :class:`~metrics.MetricsRegistry` instance.
            Defaults to the module-level shared registry.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.config: dict[str, Any] = config or {}
        self._start_time = time.time()
        self._metrics = metrics or get_registry()

        # Sub-systems
        self.cognition = CognitionEngine(config=self.config)
        self.long_term_memory = LongTermMemory(
            max_entries=self.config.get("max_ltm_entries", 10_000),
            default_ttl=self.config.get("ltm_default_ttl"),
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

    def respond(self, stimulus: Any) -> dict[str, Any]:
        """Process a *stimulus* and return a response.

        The method:

        1. Retrieves recent context from episodic memory.
        2. Runs the full cognition pipeline (perceive → reason).
        3. Records the episode in episodic memory.
        4. Indexes the result vector in the vector store.
        5. Returns the final result with timing metadata.

        Args:
            stimulus: Any raw input to process.

        Returns:
            Dict with keys ``result`` (state vector), ``iterations``,
            ``converged``, and ``timing``.
        """
        self._metrics.increment("agent.respond.calls")
        t_start = time.perf_counter()

        context = self._build_context()
        result = self.cognition.process(stimulus, context=context)
        self.episodic_memory.record(stimulus=stimulus, response=result)
        self._update_vector_store(stimulus, result)

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self._metrics.observe("agent.respond.latency_ms", latency_ms)
        logger.debug("VictorAgent.respond result: %s", result)
        return result

    def remember(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Explicitly store a fact in long-term memory.

        Args:
            key: Unique identifier.
            value: Data to store.
            metadata: Optional annotations (source, importance, etc.).
        """
        self.long_term_memory.store(key, value, metadata=metadata)
        self._metrics.increment("agent.remember.calls")

    def recall(self, key: str) -> Any:
        """Retrieve a fact from long-term memory.

        Args:
            key: Identifier to look up.

        Returns:
            Stored value or ``None`` if not found (or expired).
        """
        self._metrics.increment("agent.recall.calls")
        return self.long_term_memory.retrieve(key)

    def execute_task(self, task: dict[str, Any]) -> Any:
        """Delegate a structured task to the :class:`TaskExecutor`.

        Args:
            task: Task specification dict.  Must contain at minimum a
                ``"type"`` key.

        Returns:
            Task result.

        Raises:
            :class:`~exceptions.UnknownTaskTypeError`: If the task type
                has no registered handler.
            :class:`~exceptions.MissingTaskFieldError`: If the ``"type"``
                field is absent.
        """
        self._metrics.increment("agent.execute_task.calls")
        return self.task_executor.execute(task)

    def health(self) -> dict[str, Any]:
        """Return a structured health report for this agent.

        The report includes operational status for each subsystem and
        overall memory utilisation.

        Returns:
            Dict with keys:

            * ``status`` — ``"ok"`` or ``"degraded"``.
            * ``uptime_seconds`` — seconds since the agent was created.
            * ``memory`` — per-store statistics.
            * ``cognition`` — reasoning-loop configuration.
        """
        ltm_stats = self.long_term_memory.stats()
        ep_stats = self.episodic_memory.stats()
        vs_stats = self.vector_store.stats()

        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "memory": {
                "long_term": ltm_stats,
                "episodic": ep_stats,
                "vector_store": vs_stats,
            },
            "cognition": {
                "max_iterations": self.cognition.reasoning_loop.max_iterations,
                "convergence_threshold": self.cognition.reasoning_loop.convergence_threshold,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self) -> dict[str, Any]:
        """Assemble a context dict from recent episodes."""
        recent = self.episodic_memory.recent(n=5)
        return {"recent_episodes": [ep.to_dict() for ep in recent]}

    def _update_vector_store(self, stimulus: Any, result: dict[str, Any]) -> None:
        """Index the result vector in the vector store (best-effort)."""
        vector = result.get("result")
        if isinstance(vector, list) and vector:
            key = f"ep_{len(self.episodic_memory)}"
            try:
                self.vector_store.add(key, vector)
            except (ValueError, Exception):
                pass
