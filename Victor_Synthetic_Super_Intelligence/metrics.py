"""Metrics — lightweight, thread-safe counters and gauges for observability.

Provides a simple metrics registry used by the API server, agent, and
other components to expose operational data via the ``/metrics`` endpoint.

All counter and timing methods are thread-safe.

Example::

    from Victor_Synthetic_Super_Intelligence.metrics import get_registry

    metrics = get_registry()
    metrics.increment("requests_total", labels={"endpoint": "/respond"})
    metrics.timing("response_latency_ms", 42.3, labels={"endpoint": "/respond"})

    snapshot = metrics.snapshot()
    print(snapshot)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class _Counter:
    """A thread-safe monotonically increasing counter."""

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value: float = 0.0
        self._lock = threading.Lock()

    def increment(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class _Gauge:
    """A thread-safe gauge that can go up or down."""

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def increment(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def decrement(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class _Histogram:
    """A thread-safe histogram for tracking distributions (e.g. latency)."""

    __slots__ = ("_count", "_sum", "_min", "_max", "_lock")

    def __init__(self) -> None:
        self._count: int = 0
        self._sum: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "count": self._count,
                "sum": round(self._sum, 4),
                "mean": round(self._sum / self._count, 4) if self._count else 0.0,
                "min": round(self._min, 4) if self._count else 0.0,
                "max": round(self._max, 4) if self._count else 0.0,
            }


class MetricsRegistry:
    """Central registry of all application metrics.

    Supports three metric types:

    * **counters** — monotonically increasing totals.
    * **gauges** — values that go up and down.
    * **histograms** — distributions (count, sum, mean, min, max).

    All public methods are thread-safe.

    Args:
        name: Optional registry name (useful when multiple registries
            coexist in tests).
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._counters: dict[str, _Counter] = defaultdict(_Counter)
        self._gauges: dict[str, _Gauge] = defaultdict(_Gauge)
        self._histograms: dict[str, _Histogram] = defaultdict(_Histogram)

    # ------------------------------------------------------------------
    # Counter API
    # ------------------------------------------------------------------

    def increment(self, name: str, amount: float = 1.0) -> None:
        """Increment a named counter.

        Args:
            name: Metric name (e.g. ``"requests_total"``).
            amount: Amount to add (default: 1.0).
        """
        with self._lock:
            self._counters[name].increment(amount)

    def counter(self, name: str) -> float:
        """Return the current value of a counter.

        Args:
            name: Counter name.

        Returns:
            Current counter value (0.0 if never incremented).
        """
        with self._lock:
            return self._counters[name].value

    # ------------------------------------------------------------------
    # Gauge API
    # ------------------------------------------------------------------

    def gauge_set(self, name: str, value: float) -> None:
        """Set a gauge to an absolute value.

        Args:
            name: Gauge name.
            value: New value.
        """
        with self._lock:
            self._gauges[name].set(value)

    def gauge_inc(self, name: str, amount: float = 1.0) -> None:
        """Increment a gauge.

        Args:
            name: Gauge name.
            amount: Amount to add (default: 1.0).
        """
        with self._lock:
            self._gauges[name].increment(amount)

    def gauge_dec(self, name: str, amount: float = 1.0) -> None:
        """Decrement a gauge.

        Args:
            name: Gauge name.
            amount: Amount to subtract (default: 1.0).
        """
        with self._lock:
            self._gauges[name].decrement(amount)

    def gauge(self, name: str) -> float:
        """Return the current value of a gauge.

        Args:
            name: Gauge name.

        Returns:
            Current gauge value (0.0 if never set).
        """
        with self._lock:
            return self._gauges[name].value

    # ------------------------------------------------------------------
    # Histogram API
    # ------------------------------------------------------------------

    def observe(self, name: str, value: float) -> None:
        """Record an observation in a histogram.

        Args:
            name: Histogram name (e.g. ``"response_latency_ms"``).
            value: The observed value.
        """
        with self._lock:
            self._histograms[name].observe(value)

    def timing(self, name: str, value_ms: float) -> None:
        """Convenience method: record a latency value in milliseconds.

        Args:
            name: Histogram name.
            value_ms: Latency in milliseconds.
        """
        self.observe(name, value_ms)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of all metrics.

        Returns:
            A dict with keys ``"counters"``, ``"gauges"``,
            ``"histograms"``, and ``"uptime_seconds"``.
        """
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "counters": {k: v.value for k, v in self._counters.items()},
                "gauges": {k: v.value for k, v in self._gauges.items()},
                "histograms": {k: v.snapshot() for k, v in self._histograms.items()},
            }

    def reset(self) -> None:
        """Clear all metrics and reset the start time.

        Primarily useful in tests to avoid state leakage between test
        cases.
        """
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.time()


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY = MetricsRegistry(name="victor_ssi")


def get_registry() -> MetricsRegistry:
    """Return the default module-level :class:`MetricsRegistry`.

    All components share this registry by default.  Tests that need
    isolation should create their own :class:`MetricsRegistry` instance.

    Returns:
        The default :class:`MetricsRegistry`.
    """
    return _DEFAULT_REGISTRY


__all__ = ["MetricsRegistry", "get_registry"]
