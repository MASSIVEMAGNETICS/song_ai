"""Task Executor — interprets and executes structured task specifications.

Provides a registry-based dispatcher for structured tasks.  New task types
can be registered by decorating methods with ``@_register("type_name")``.

Example::

    executor = TaskExecutor(agent=agent)

    # Built-in tasks
    executor.execute({"type": "remember", "key": "greeting", "value": "Hello"})
    greeting = executor.execute({"type": "recall", "key": "greeting"})

    # Custom task
    @_register("ping")
    def _handle_ping(self, task):
        return "pong"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from ..exceptions import MissingTaskFieldError, UnknownTaskTypeError

if TYPE_CHECKING:
    from .victor_agent import VictorAgent

logger = logging.getLogger(__name__)

# Registry that maps task type names to handler callables.
_TASK_REGISTRY: dict[str, Callable[["TaskExecutor", dict[str, Any]], Any]] = {}


def _register(task_type: str):
    """Class-method decorator that registers a handler in the task registry.

    Args:
        task_type: The string identifier for this task type.

    Returns:
        Decorator function.
    """
    def decorator(fn):
        _TASK_REGISTRY[task_type] = fn
        return fn
    return decorator


class TaskExecutor:
    """Executes structured tasks on behalf of :class:`~agents.VictorAgent`.

    Tasks are dispatched via a simple registry pattern.  New task types can
    be added by decorating methods with ``@_register("task_type_name")``.

    Args:
        agent: The owning :class:`~agents.VictorAgent` instance.
    """

    def __init__(self, agent: "VictorAgent") -> None:
        self.agent = agent

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def execute(self, task: dict[str, Any]) -> Any:
        """Execute a task specification.

        Args:
            task: A dictionary that **must** contain a ``"type"`` key.
                Additional keys are task-type specific.

        Returns:
            Task result.

        Raises:
            :class:`~exceptions.MissingTaskFieldError`: If ``"type"`` is
                missing from *task*.
            :class:`~exceptions.UnknownTaskTypeError`: If the task type
                has no registered handler.
        """
        if "type" not in task:
            raise MissingTaskFieldError("type")
        task_type = task["type"]
        handler = _TASK_REGISTRY.get(task_type)
        if handler is None:
            raise UnknownTaskTypeError(task_type, list(_TASK_REGISTRY.keys()))
        logger.info("Executing task type '%s'", task_type)
        return handler(self, task)

    @classmethod
    def registered_types(cls) -> list[str]:
        """Return a sorted list of all registered task type names.

        Returns:
            Sorted list of task type identifier strings.
        """
        return sorted(_TASK_REGISTRY.keys())

    # ------------------------------------------------------------------
    # Built-in task handlers
    # ------------------------------------------------------------------

    @_register("recall")
    def _handle_recall(self, task: dict[str, Any]) -> Any:
        """Retrieve a value from long-term memory.

        Task keys
        ---------
        ``key`` : str
            Memory key to look up.
        """
        key = task.get("key", "")
        return self.agent.recall(key)

    @_register("remember")
    def _handle_remember(self, task: dict[str, Any]) -> None:
        """Store a value in long-term memory.

        Task keys
        ---------
        ``key`` : str
            Memory key.
        ``value`` : Any
            Value to store.
        ``metadata`` : dict, optional
            Annotations for the entry.
        """
        self.agent.remember(
            task.get("key", ""),
            task.get("value"),
            metadata=task.get("metadata"),
        )

    @_register("respond")
    def _handle_respond(self, task: dict[str, Any]) -> Any:
        """Run the agent's full perception-reasoning pipeline.

        Task keys
        ---------
        ``stimulus`` : Any
            Input to process.
        """
        return self.agent.respond(task.get("stimulus", ""))

    @_register("search_memory")
    def _handle_search_memory(self, task: dict[str, Any]) -> list[tuple[str, Any]]:
        """Search long-term memory by key substring.

        Task keys
        ---------
        ``query`` : str
            Substring to match against stored keys.
        """
        query = task.get("query", "")
        return self.agent.long_term_memory.search(query)

    @_register("vector_query")
    def _handle_vector_query(
        self, task: dict[str, Any]
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Query the vector store for similar entries.

        Task keys
        ---------
        ``vector`` : list[float]
            Query vector.
        ``top_k`` : int
            Number of results (default: 5).
        """
        vector = task.get("vector", [])
        top_k = task.get("top_k", 5)
        return self.agent.vector_store.query(vector, top_k=top_k)

    @_register("health")
    def _handle_health(self, task: dict[str, Any]) -> dict[str, Any]:
        """Return the agent health report.

        Task keys
        ---------
        (none)
        """
        return self.agent.health()

    @_register("memory_stats")
    def _handle_memory_stats(self, task: dict[str, Any]) -> dict[str, Any]:
        """Return memory utilisation statistics.

        Task keys
        ---------
        (none)
        """
        return {
            "long_term": self.agent.long_term_memory.stats(),
            "episodic": self.agent.episodic_memory.stats(),
            "vector_store": self.agent.vector_store.stats(),
        }
