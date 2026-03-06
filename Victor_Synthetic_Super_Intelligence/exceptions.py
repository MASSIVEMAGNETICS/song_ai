"""Victor SSI — custom exception hierarchy.

All public exceptions raised by the Victor Synthetic Super Intelligence
package are defined here so that callers can catch them at the appropriate
level of granularity.

Example::

    from Victor_Synthetic_Super_Intelligence.exceptions import VictorError, MemoryError

    try:
        agent.recall("missing_key")
    except MemoryError as exc:
        logger.warning("Memory lookup failed: %s", exc)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class VictorError(Exception):
    """Base class for all Victor SSI exceptions.

    All package-specific exceptions inherit from this class, allowing
    callers to catch any Victor SSI error with a single ``except`` clause::

        try:
            ...
        except VictorError:
            ...
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationError(VictorError):
    """Raised when a configuration value is missing or invalid."""


# ---------------------------------------------------------------------------
# Memory subsystem
# ---------------------------------------------------------------------------


class MemoryError(VictorError):
    """Base class for memory-subsystem errors."""


class MemoryCapacityError(MemoryError):
    """Raised when a memory store has reached its maximum capacity and
    automatic eviction has been disabled."""


class MemoryKeyError(MemoryError):
    """Raised when a required memory key is not found."""


class VectorDimensionError(MemoryError):
    """Raised when a vector does not match the expected dimensionality of
    the store."""


# ---------------------------------------------------------------------------
# Cognition / reasoning
# ---------------------------------------------------------------------------


class CognitionError(VictorError):
    """Base class for errors in the cognition engine or reasoning loop."""


class EncodingError(CognitionError):
    """Raised when an input value cannot be encoded into a vector."""


class ReasoningError(CognitionError):
    """Raised when the reasoning loop encounters an unrecoverable state."""


# ---------------------------------------------------------------------------
# Agent / task execution
# ---------------------------------------------------------------------------


class AgentError(VictorError):
    """Base class for agent-layer errors."""


class TaskError(AgentError):
    """Base class for task-execution errors."""


class UnknownTaskTypeError(TaskError):
    """Raised when a task type has no registered handler."""

    def __init__(self, task_type: str, registered: list[str]) -> None:
        self.task_type = task_type
        self.registered = registered
        super().__init__(
            f"Unknown task type '{task_type}'. "
            f"Registered types: {registered}"
        )


class MissingTaskFieldError(TaskError):
    """Raised when a required field is absent from a task specification."""

    def __init__(self, field: str, task_type: str | None = None) -> None:
        self.field = field
        self.task_type = task_type
        msg = f"Required field '{field}' is missing"
        if task_type:
            msg += f" for task type '{task_type}'"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Interface layer
# ---------------------------------------------------------------------------


class InterfaceError(VictorError):
    """Base class for interface (API/CLI) errors."""


class RateLimitExceededError(InterfaceError):
    """Raised when a client has exceeded the configured rate limit."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        super().__init__(
            f"Rate limit of {limit} requests per {window_seconds}s exceeded."
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainingError(VictorError):
    """Base class for training-pipeline errors."""


class DatasetError(TrainingError):
    """Raised when a dataset cannot be loaded or is malformed."""


__all__ = [
    "VictorError",
    "ConfigurationError",
    "MemoryError",
    "MemoryCapacityError",
    "MemoryKeyError",
    "VectorDimensionError",
    "CognitionError",
    "EncodingError",
    "ReasoningError",
    "AgentError",
    "TaskError",
    "UnknownTaskTypeError",
    "MissingTaskFieldError",
    "InterfaceError",
    "RateLimitExceededError",
    "TrainingError",
    "DatasetError",
]
