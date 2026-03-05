# Victor Synthetic Super Intelligence

A modular, framework-agnostic synthetic intelligence system built in Python.

## Project Structure

```
Victor_Synthetic_Super_Intelligence/
│
├── core/
│   ├── cognition_engine.py    # Orchestrates perception → reasoning → action
│   ├── tensor_operations.py   # Numeric/vector utilities (no heavy deps)
│   ├── reasoning_loop.py      # Iterative reasoning with convergence check
│
├── memory/
│   ├── long_term_memory.py    # Persistent key/value knowledge store
│   ├── episodic_memory.py     # Ordered circular buffer of past experiences
│   ├── vector_store.py        # Cosine-similarity nearest-neighbour store
│
├── agents/
│   ├── victor_agent.py        # Primary autonomous agent (wires everything)
│   ├── task_executor.py       # Structured task dispatch registry
│
├── interfaces/
│   ├── api_server.py          # Lightweight HTTP REST server
│   ├── cli_interface.py       # Interactive REPL command-line interface
│
├── training/
│   ├── dataset_loader.py      # JSON / JSONL / CSV / TXT dataset loading
│   ├── training_pipeline.py   # Framework-agnostic training orchestration
│
└── configs/
    └── model_config.yaml      # Default model & runtime configuration
```

## Quick Start

### Python API

```python
from Victor_Synthetic_Super_Intelligence.agents import VictorAgent

agent = VictorAgent()
result = agent.respond("Hello, Victor!")
print(result)
```

### CLI

```bash
python -m Victor_Synthetic_Super_Intelligence.interfaces.cli_interface
```

### HTTP API

```bash
python -m Victor_Synthetic_Super_Intelligence.interfaces.api_server --port 8080
```

Then:

```bash
curl -X POST http://localhost:8080/respond \
     -H "Content-Type: application/json" \
     -d '{"stimulus": "Hello"}'
```

## Configuration

Edit `configs/model_config.yaml` to adjust model parameters, memory
capacities, training settings, and interface options.

## Design Principles

* **No mandatory heavy dependencies** — the entire stack runs on the Python
  standard library.  GPU-backed tensors or vector databases can be wired in
  by sub-classing the relevant modules.
* **Modular** — each sub-package is independently usable.
* **Observable** — all components log via Python's `logging` module; set
  `LOG_LEVEL=DEBUG` to trace every step.
