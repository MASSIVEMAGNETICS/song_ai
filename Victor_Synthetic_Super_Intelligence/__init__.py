"""Victor Synthetic Super Intelligence — top-level package.

Victor SSI is a modular, framework-agnostic synthetic intelligence system
built on pure Python.  It provides:

* A full **cognition pipeline** (perception → iterative reasoning → action)
* Three **memory subsystems**: long-term key/value store, episodic circular
  buffer, and a cosine-similarity vector store
* A registry-based **task executor** for structured task dispatch
* A production-ready **HTTP REST API** with CORS, rate limiting, and
  security headers
* An interactive **CLI** for human-in-the-loop sessions
* A **metrics registry** for operational observability
* A **config loader** that reads YAML and respects environment variables

Quick start::

    from Victor_Synthetic_Super_Intelligence import VictorAgent

    agent = VictorAgent()
    result = agent.respond("Hello, Victor!")
    print(result)

    agent.remember("capital:france", "Paris")
    print(agent.recall("capital:france"))

    health = agent.health()
    print(health)

See the project README for full documentation and API reference.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Victor SSI"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Convenience re-exports — the most commonly used public symbols are
# importable directly from the top-level package.
# ---------------------------------------------------------------------------

from .agents.victor_agent import VictorAgent
from .agents.task_executor import TaskExecutor
from .core.cognition_engine import CognitionEngine
from .core.reasoning_loop import ReasoningLoop
from .core.tensor_operations import TensorOperations
from .memory.long_term_memory import LongTermMemory
from .memory.episodic_memory import EpisodicMemory, Episode
from .memory.vector_store import VectorStore
from .interfaces.api_server import APIServer
from .interfaces.cli_interface import CLIInterface
from .training.dataset_loader import DatasetLoader
from .training.training_pipeline import TrainingPipeline
from .config_loader import load_config
from .metrics import MetricsRegistry, get_registry
from .exceptions import (
    VictorError,
    ConfigurationError,
    MemoryError,
    MemoryCapacityError,
    MemoryKeyError,
    VectorDimensionError,
    CognitionError,
    EncodingError,
    ReasoningError,
    AgentError,
    TaskError,
    UnknownTaskTypeError,
    MissingTaskFieldError,
    InterfaceError,
    RateLimitExceededError,
    TrainingError,
    DatasetError,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",
    # Agents
    "VictorAgent",
    "TaskExecutor",
    # Core
    "CognitionEngine",
    "ReasoningLoop",
    "TensorOperations",
    # Memory
    "LongTermMemory",
    "EpisodicMemory",
    "Episode",
    "VectorStore",
    # Interfaces
    "APIServer",
    "CLIInterface",
    # Training
    "DatasetLoader",
    "TrainingPipeline",
    # Utilities
    "load_config",
    "MetricsRegistry",
    "get_registry",
    # Exceptions
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
