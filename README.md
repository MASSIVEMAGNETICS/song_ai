# song_ai

**Victor Synthetic Super Intelligence** — enterprise-grade, modular AI agent framework.

See [`Victor_Synthetic_Super_Intelligence/README.md`](Victor_Synthetic_Super_Intelligence/README.md) for full documentation.




Victor Synthetic Super Intelligence
Enterprise-grade, modular, framework-agnostic AI agent framework built on pure Python.

Python 3.10+ License: MIT Zero Dependencies

Table of Contents
Overview
Key Features
Architecture
Project Structure
Installation
Quick Start
Configuration
API Reference
VictorAgent
CognitionEngine
Memory Subsystem
TaskExecutor
Metrics
Config Loader
Exceptions
HTTP REST API
CLI Interface
Training Pipeline
Deployment
Extending Victor SSI
Testing
Design Principles
Changelog
Overview
Victor SSI is a modular, enterprise-ready synthetic intelligence framework that orchestrates:

Perception — converts raw inputs (text, numbers, data) into internal vector representations
Iterative Cognition — a convergence-based reasoning loop with configurable depth
Three-tier Memory — long-term key/value store, episodic circular buffer, and cosine-similarity vector database
Task Execution — a type-safe, registry-based dispatcher for structured agent tasks
Observability — built-in metrics collection (counters, gauges, histograms) exposed via /metrics
HTTP REST API — production-ready server with CORS, rate limiting, security headers, and request IDs
Interactive CLI — a feature-rich REPL for human-in-the-loop sessions
The entire stack runs on the Python standard library — no runtime dependencies are required.

Key Features
Feature	Details
Zero mandatory dependencies	Runs on Python 3.10+ stdlib only
Thread-safe memory	All memory stores use threading.RLock
TTL expiry	Long-term memory supports per-entry and default TTLs
Rate limiting	Sliding-window per-IP rate limiter on the API server
CORS & security headers	X-Content-Type-Options, X-Frame-Options, Cache-Control, CORS
Request tracing	Every HTTP response includes a X-Request-ID / request_id field
Structured metrics	Counters, gauges, histograms via MetricsRegistry
Config as code	YAML config + env-variable overrides (VICTOR_<SECTION>_<KEY>)
Custom exceptions	Full exception hierarchy (VictorError → domain-specific subclasses)
146 unit tests	Full coverage of all modules, including integration tests for the HTTP API
Type annotations	from __future__ import annotations throughout, ready for mypy
Pluggable backends	Sub-class any component to replace with GPU tensors, FAISS, Redis, etc.
Architecture
┌────────────────────────────────────────────────────────────────────┐
│                        VictorAgent                                 │
│  ┌────────────────┐  ┌──────────────────────────────────────────┐  │
│  │ CognitionEngine│  │            Memory Subsystem              │  │
│  │  ┌───────────┐ │  │  ┌─────────────┐  ┌──────────────────┐  │  │
│  │  │ perceive()│ │  │  │LongTermMemory│  │  EpisodicMemory  │  │  │
│  │  │  encode() │ │  │  │  key/value  │  │ circular buffer  │  │  │
│  │  └─────┬─────┘ │  │  │  TTL/FIFO   │  │  Episode objects │  │  │
│  │        │       │  │  └─────────────┘  └──────────────────┘  │  │
│  │  ┌─────▼─────┐ │  │  ┌──────────────────────────────────┐   │  │
│  │  │reasoning_ │ │  │  │         VectorStore              │   │  │
│  │  │  loop()   │ │  │  │  cosine-similarity top-k search  │   │  │
│  │  │convergence│ │  │  └──────────────────────────────────┘   │  │
│  │  └───────────┘ │  └──────────────────────────────────────────┘  │
│  └────────────────┘                                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      TaskExecutor                              │ │
│  │  recall · remember · respond · search_memory · vector_query   │ │
│  │  health · memory_stats  (+ your custom handlers)              │ │
│  └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
         ▲                              ▲
         │                              │
   ┌─────┴──────┐               ┌───────┴──────┐
   │ HTTP API   │               │  CLI (REPL)  │
   │ /health    │               │  remember    │
   │ /version   │               │  recall      │
   │ /metrics   │               │  forget      │
   │ /respond   │               │  search      │
   │ /task      │               │  recent      │
   │ /memory/*  │               │  health      │
   └────────────┘               │  metrics     │
                                └──────────────┘
Project Structure
Victor_Synthetic_Super_Intelligence/
│
├── __init__.py                # Public API re-exports, __version__
├── exceptions.py              # Full exception hierarchy
├── config_loader.py           # YAML config loader with env-var overrides
├── metrics.py                 # Thread-safe metrics registry
│
├── core/
│   ├── cognition_engine.py    # Perceive → reason → action pipeline (with timing)
│   ├── reasoning_loop.py      # Iterative inference with convergence detection
│   └── tensor_operations.py   # Pure-Python vector/tensor utilities
│
├── memory/
│   ├── long_term_memory.py    # Thread-safe key/value store (TTL, FIFO eviction)
│   ├── episodic_memory.py     # Thread-safe circular buffer of agent experiences
│   └── vector_store.py        # Thread-safe cosine-similarity vector database
│
├── agents/
│   ├── victor_agent.py        # Top-level agent facade with metrics & health
│   └── task_executor.py       # Registry-based structured task dispatcher
│
├── interfaces/
│   ├── api_server.py          # Production HTTP server (CORS, rate-limit, security)
│   └── cli_interface.py       # Feature-rich interactive REPL
│
├── training/
│   ├── dataset_loader.py      # JSON / JSONL / CSV / TXT dataset loading
│   └── training_pipeline.py   # Framework-agnostic training orchestration
│
└── configs/
    └── model_config.yaml      # Default configuration (YAML)
Installation
From source
git clone https://github.com/MASSIVEMAGNETICS/song_ai.git
cd song_ai
pip install -e .
With optional PyYAML support
pip install -e ".[yaml]"
Development install (includes test tools)
pip install -e ".[dev]"
Note: The package works without any additional dependencies. PyYAML is optional — a built-in minimal YAML parser handles model_config.yaml automatically when PyYAML is not installed.

Quick Start
Python API
from Victor_Synthetic_Super_Intelligence import VictorAgent

# Create an agent with default configuration
agent = VictorAgent()

# Process a stimulus through the full cognition pipeline
result = agent.respond("Hello, Victor!")
print(result)
# {
#   "result": [0.408, 0.408, ...],
#   "iterations": 3,
#   "converged": True,
#   "timing": {"perceive": 0.00012, "reason": 0.00034, "total": 0.00046}
# }

# Long-term memory
agent.remember("capital:france", "Paris")
print(agent.recall("capital:france"))   # "Paris"

# Health report
health = agent.health()
print(health["status"])     # "ok"
print(health["memory"])     # per-store utilisation statistics

# Structured task execution
result = agent.execute_task({"type": "search_memory", "query": "capital"})
CLI
# Start the interactive REPL
python -m Victor_Synthetic_Super_Intelligence.interfaces.cli_interface

# Or use the installed entry point
victor-cli
Inside the REPL:

victor> Hello, Victor!
victor> remember greeting Hello
victor> recall greeting
victor> search greeting
victor> recent 5
victor> health
victor> metrics
victor> forget greeting
victor> help
victor> exit
HTTP API
# Start the server
python -m Victor_Synthetic_Super_Intelligence.interfaces.api_server --port 8080

# Or use the installed entry point
victor-api --port 8080
# Health check
curl http://localhost:8080/health

# Version
curl http://localhost:8080/version

# Process a stimulus
curl -X POST http://localhost:8080/respond \
     -H "Content-Type: application/json" \
     -d '{"stimulus": "Hello"}'

# Execute a task
curl -X POST http://localhost:8080/task \
     -H "Content-Type: application/json" \
     -d '{"type": "remember", "key": "name", "value": "Victor"}'

# Memory operations
curl -X POST http://localhost:8080/memory/remember \
     -H "Content-Type: application/json" \
     -d '{"key": "greeting", "value": "Hello", "metadata": {"source": "user"}}'

curl -X POST http://localhost:8080/memory/recall \
     -H "Content-Type: application/json" \
     -d '{"key": "greeting"}'

# Operational metrics
curl http://localhost:8080/metrics

# Memory statistics
curl http://localhost:8080/memory/stats
Configuration
YAML file
Edit Victor_Synthetic_Super_Intelligence/configs/model_config.yaml:

cognition:
  max_iterations: 10          # Reasoning depth (default: 5)
  convergence_threshold: 1e-5 # Tighter convergence (default: 1e-4)

memory:
  long_term:
    max_entries: 50000        # Larger LTM
    default_ttl: 3600         # 1-hour TTL for all entries

interfaces:
  api:
    port: 9000
    rate_limit: 120           # 120 req/min per client
    cors_origin: "https://myapp.example.com"
Environment variables
Any VICTOR_<SECTION>_<KEY>=<value> variable overrides the corresponding YAML value:

export VICTOR_COGNITION_MAX_ITERATIONS=10
export VICTOR_INTERFACES_API_PORT=9000
export VICTOR_MEMORY_LONG_TERM_MAX_ENTRIES=100000
Programmatic overrides
from Victor_Synthetic_Super_Intelligence import load_config, VictorAgent

cfg = load_config(overrides={
    "cognition": {"max_iterations": 10},
    "memory": {"long_term": {"max_entries": 50_000}},
})

agent = VictorAgent(config={
    "max_iterations": cfg["cognition"]["max_iterations"],
    "max_ltm_entries": cfg["memory"]["long_term"]["max_entries"],
    "ltm_default_ttl": 3600,
})
API Reference
VictorAgent
The top-level agent facade.

VictorAgent(config=None, metrics=None)
Method	Signature	Description
respond	(stimulus: Any) → dict	Full perception-reasoning pipeline
remember	(key: str, value: Any, metadata=None) → None	Store in LTM
recall	(key: str) → Any	Retrieve from LTM
execute_task	(task: dict) → Any	Dispatch a structured task
health	() → dict	Agent health report
Config keys:

Key	Default	Description
max_ltm_entries	10_000	LTM capacity
ltm_default_ttl	None	Default TTL (seconds) for LTM entries
episodic_capacity	1_000	Episodic memory capacity
vector_dim	None	Vector store dimensionality (inferred)
max_iterations	5	Reasoning loop depth
convergence_threshold	1e-4	Reasoning convergence threshold
CognitionEngine
CognitionEngine(config=None)
Method	Description
perceive(stimulus)	Encode input to float vector
reason(representation, context)	Run reasoning loop
process(stimulus, context)	Full pipeline (perceive + reason)
last_timing	Dict of {"perceive", "reason", "total"} timings in seconds
Memory Subsystem
LongTermMemory
LongTermMemory(max_entries=10_000, default_ttl=None)
Method	Description
store(key, value, metadata, ttl)	Persist a value
retrieve(key)	Fetch by key (returns None if expired)
retrieve_with_metadata(key)	Fetch full entry dict
delete(key)	Remove an entry
search(query)	Substring search over keys
clear()	Remove all entries, returns count
purge_expired()	Explicitly remove expired entries
stats()	{total_entries, max_entries, utilisation_pct, default_ttl}
EpisodicMemory
EpisodicMemory(capacity=1_000)
Method	Description
record(stimulus, response, metadata)	Store an episode
recent(n)	Return last N episodes (oldest first)
search(query)	Substring match on str(stimulus)
clear()	Discard all episodes
stats()	{total_episodes, capacity, utilisation_pct}
VectorStore
VectorStore(dimension=None)
Method	Description
add(key, vector, metadata)	Insert/overwrite a vector
remove(key)	Delete by key
query(query_vector, top_k)	Top-k cosine-similarity search
get(key)	Retrieve a specific vector
clear()	Remove all vectors
stats()	{total_vectors, dimension}
TaskExecutor
Built-in task types:

Type	Required fields	Description
recall	key	Retrieve from LTM
remember	key, value	Store in LTM
respond	stimulus	Full agent pipeline
search_memory	query	Substring search in LTM
vector_query	vector, top_k	Vector similarity search
health	—	Agent health report
memory_stats	—	All memory utilisation stats
Custom task types can be added by decorating methods with @_register("my_type") inside a subclass of TaskExecutor.

Metrics
from Victor_Synthetic_Super_Intelligence import get_registry

metrics = get_registry()

metrics.increment("my.counter")
metrics.increment("my.counter", amount=5.0)
metrics.gauge_set("active_connections", 42)
metrics.observe("response_latency_ms", 12.5)

snapshot = metrics.snapshot()
# {
#   "uptime_seconds": 3600.0,
#   "counters": {"my.counter": 6.0, ...},
#   "gauges": {"active_connections": 42.0, ...},
#   "histograms": {
#     "response_latency_ms": {
#       "count": 1, "sum": 12.5, "mean": 12.5, "min": 12.5, "max": 12.5
#     }
#   }
# }
Config Loader
from Victor_Synthetic_Super_Intelligence import load_config

# Load defaults
cfg = load_config()

# With overrides
cfg = load_config(overrides={"cognition": {"max_iterations": 10}})

# Custom path
cfg = load_config(path="/etc/victor/config.yaml")
Environment variables matching VICTOR_<SECTION>_<KEY> are automatically applied between the file and caller overrides.

Exceptions
All exceptions inherit from VictorError:

VictorError
├── ConfigurationError
├── MemoryError
│   ├── MemoryCapacityError
│   ├── MemoryKeyError
│   └── VectorDimensionError
├── CognitionError
│   ├── EncodingError
│   └── ReasoningError
├── AgentError
│   └── TaskError
│       ├── UnknownTaskTypeError
│       └── MissingTaskFieldError
├── InterfaceError
│   └── RateLimitExceededError
└── TrainingError
    └── DatasetError
HTTP REST API
Endpoints
GET /health
Returns agent health report including memory utilisation.

Response:

{
  "request_id": "uuid4",
  "status": "ok",
  "uptime_seconds": 3600.0,
  "memory": {
    "long_term": {"total_entries": 42, "max_entries": 10000, "utilisation_pct": 0.42},
    "episodic":  {"total_episodes": 17, "capacity": 1000, "utilisation_pct": 1.7},
    "vector_store": {"total_vectors": 17, "dimension": 5}
  },
  "cognition": {"max_iterations": 5, "convergence_threshold": 0.0001}
}
GET /version
{"request_id": "...", "version": "1.0.0", "name": "Victor SSI"}
GET /metrics
Returns the full metrics snapshot (counters, gauges, histograms, uptime).

GET /memory/stats
Returns per-store statistics (long_term, episodic, vector_store).

POST /respond
Run the full agent cognition pipeline.

Request: {"stimulus": "any input"}

Response:

{
  "request_id": "...",
  "result": [0.408, 0.408, ...],
  "iterations": 3,
  "converged": true,
  "timing": {"perceive": 0.00012, "reason": 0.00034, "total": 0.00046}
}
POST /task
Execute a structured task.

Request: Any task dict with a "type" field.

Response: {"request_id": "...", "result": <task result>}

Error (400): {"request_id": "...", "error": "description"}

POST /memory/remember
Request: {"key": "...", "value": ..., "metadata": {...}}

POST /memory/recall
Request: {"key": "..."} Response: {"request_id": "...", "key": "...", "value": ...}

Security Headers
Every response includes:

Header	Value
X-Content-Type-Options	nosniff
X-Frame-Options	DENY
Cache-Control	no-store
X-Request-ID	UUID4 for this request
Access-Control-Allow-Origin	Configurable (default: *)
Rate Limiting
The server enforces a sliding-window per-IP rate limit (default: 60 requests per 60 seconds). Exceeded clients receive HTTP 429 with a Retry-After-style body:

{"error": "rate limit exceeded", "limit": 60, "window_seconds": 60}
Configure via CLI flags or config:

victor-api --rate-limit 120 --rate-window 60
CLI Interface
╔══════════════════════════════════════════════════════╗
║   Victor Synthetic Super Intelligence v1.0.0         ║
║   Type 'help' for commands, 'exit' to quit           ║
╚══════════════════════════════════════════════════════╝

Available commands
──────────────────
  <text>                   Send a stimulus and receive a response.
  remember <key> <val>     Store a value in long-term memory.
  recall <key>             Retrieve a value from long-term memory.
  forget <key>             Delete a key from long-term memory.
  search <query>           Search long-term memory keys by substring.
  recent [n]               Show n most recent episodes (default: 5).
  health                   Display agent health and memory statistics.
  metrics                  Display runtime metrics snapshot.
  version                  Print version information.
  clear                    Clear the terminal screen.
  help                     Show this help text.
  exit / quit              Exit the CLI.
# Start with debug logging
victor-cli --log-level DEBUG
Training Pipeline
The training module is framework-agnostic — you supply the train_step and optional eval_step callables.

from Victor_Synthetic_Super_Intelligence.training.training_pipeline import TrainingPipeline

def my_train_step(batch):
    # batch is a list of samples from your dataset
    # return a scalar loss value
    return 0.0

pipeline = TrainingPipeline(
    config={
        "epochs": 5,
        "batch_size": 64,
        "checkpoint_dir": "checkpoints",
        "log_interval": 5,
    },
    train_step=my_train_step,
)

history = pipeline.run("data/train.jsonl", eval_path="data/eval.jsonl")
for record in history:
    print(record)
# {"epoch": 1, "train_loss": 0.0, "duration_s": 0.0012}
Supported dataset formats
Extension	Format
.jsonl	JSON Lines — one JSON object per line
.json	Single top-level JSON array
.csv	CSV with header row
.txt	Plain text — one sample per line
Deployment
As a standalone API server
victor-api \
  --host 0.0.0.0 \
  --port 8080 \
  --rate-limit 100 \
  --rate-window 60 \
  --cors-origin "https://my-frontend.example.com" \
  --log-level INFO
As a Python library (embedded)
from Victor_Synthetic_Super_Intelligence import VictorAgent, APIServer
import threading

agent = VictorAgent(config={"max_ltm_entries": 100_000})
server = APIServer(agent=agent, host="127.0.0.1", port=8080)

thread = threading.Thread(target=server.start, daemon=True)
thread.start()

# Use agent directly in the same process
result = agent.respond("concurrent usage is safe")
Environment variable configuration
export VICTOR_COGNITION_MAX_ITERATIONS=10
export VICTOR_MEMORY_LONG_TERM_MAX_ENTRIES=100000
export VICTOR_INTERFACES_API_PORT=9000
victor-api
Extending Victor SSI
Custom tensor backend (e.g. NumPy)
import numpy as np
from Victor_Synthetic_Super_Intelligence.core.tensor_operations import TensorOperations

class NumPyTensorOps(TensorOperations):
    def normalize(self, vector):
        arr = np.array(vector, dtype=np.float64)
        norm = np.linalg.norm(arr)
        return (arr / norm).tolist() if norm > 0 else arr.tolist()

    def dot_product(self, a, b):
        return float(np.dot(a, b))
Custom memory backend (e.g. Redis)
from Victor_Synthetic_Super_Intelligence.memory.long_term_memory import LongTermMemory

class RedisLongTermMemory(LongTermMemory):
    def __init__(self, redis_client, **kwargs):
        super().__init__(**kwargs)
        self._redis = redis_client

    def store(self, key, value, **kwargs):
        import json
        self._redis.set(key, json.dumps(value))

    def retrieve(self, key):
        import json
        raw = self._redis.get(key)
        return json.loads(raw) if raw else None
Custom task type
from Victor_Synthetic_Super_Intelligence.agents.task_executor import TaskExecutor, _register

class MyExecutor(TaskExecutor):
    @_register("ping")
    def _handle_ping(self, task):
        return {"pong": True, "ts": time.time()}
Testing
# Install test dependencies
pip install -e ".[dev]"

# Run all 146 tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=Victor_Synthetic_Super_Intelligence --cov-report=term-missing

# Run a specific module
python -m pytest tests/test_memory.py -v
python -m pytest tests/test_api_server.py -v
Design Principles
Zero mandatory heavy dependencies — the entire stack runs on the Python standard library. GPU-backed tensors or vector databases can be wired in by sub-classing the relevant modules.

Thread-safe by default — all memory stores use threading.RLock to support concurrent agent instances within a single process.

Observable — all components log via Python's logging module (LOG_LEVEL=DEBUG to trace every step) and publish operational metrics via the MetricsRegistry.

Modular — each sub-package is independently usable; you can use just the memory stores, just the cognition engine, or the full agent stack.

Enterprise-ready interfaces — the HTTP API includes security headers, CORS, per-IP rate limiting, request tracing, and structured JSON errors from day one.

Fail loudly and precisely — the full exception hierarchy ensures callers can catch errors at the right level of granularity without resorting to bare except Exception.

Pluggable at every layer — tensor operations, vector storage, memory persistence, and task handlers are all designed for easy replacement via subclassing.

Changelog
v1.0.0 (2026-03-06)
Upgrade & Enterprise-Grade Changes:

Thread safety — all three memory classes (LongTermMemory, EpisodicMemory, VectorStore) now use threading.RLock for concurrent access
Custom exception hierarchy — exceptions.py introduces 16 typed exceptions under VictorError
Metrics registry — metrics.py provides thread-safe counters, gauges, and histograms; exposed via GET /metrics
Config loader — config_loader.py loads YAML with deep-merge, environment-variable overrides (VICTOR_*), and fallback for missing PyYAML
TTL support — LongTermMemory gains default_ttl, per-entry TTL, purge_expired(), and clear()
VectorStore enhancements — empty-vector validation, clear(), get(), stats(), and VectorDimensionError instead of bare ValueError
EpisodicMemory enhancements — thread-safe iteration, clear(), stats(), typed Iterator[Episode]
CognitionEngine timing — process() now includes timing dict in result; last_timing attribute tracks last call
VictorAgent — adds health(), per-operation metrics, metadata parameter on remember()
TaskExecutor — health and memory_stats built-in task types; MissingTaskFieldError / UnknownTaskTypeError instead of bare exceptions; registered_types() class method
API server — CORS pre-flight, security headers, per-IP sliding-window rate limiter, request IDs, /version, /metrics, /memory/stats, /memory/remember, /memory/recall endpoints
CLI — forget, search, health, metrics, version, clear commands
Package — __init__.py exposes all public symbols; version bumped to 1.0.0; pyproject.toml added
Tests — 146 unit + integration tests covering all modules
v0.1.0 (initial)
Initial release: CognitionEngine, ReasoningLoop, TensorOperations, LongTermMemory, EpisodicMemory, VectorStore, VictorAgent, TaskExecutor, APIServer (basic), CLIInterface (basic), DatasetLo
