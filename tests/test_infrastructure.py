"""Tests for exceptions, metrics, and config_loader modules."""

import os
import time
import threading
import unittest

from Victor_Synthetic_Super_Intelligence.exceptions import (
    VictorError,
    MemoryError,
    MemoryCapacityError,
    VectorDimensionError,
    UnknownTaskTypeError,
    MissingTaskFieldError,
    RateLimitExceededError,
)
from Victor_Synthetic_Super_Intelligence.metrics import MetricsRegistry, get_registry
from Victor_Synthetic_Super_Intelligence.config_loader import load_config, _deep_merge


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------

class TestExceptions(unittest.TestCase):

    def test_memory_error_is_victor_error(self):
        self.assertTrue(issubclass(MemoryError, VictorError))

    def test_memory_capacity_error_is_memory_error(self):
        self.assertTrue(issubclass(MemoryCapacityError, MemoryError))

    def test_vector_dimension_error_is_memory_error(self):
        self.assertTrue(issubclass(VectorDimensionError, MemoryError))

    def test_unknown_task_type_error_message(self):
        exc = UnknownTaskTypeError("foo", ["bar", "baz"])
        self.assertIn("foo", str(exc))
        self.assertIn("bar", str(exc))
        self.assertEqual(exc.task_type, "foo")
        self.assertEqual(exc.registered, ["bar", "baz"])

    def test_missing_task_field_error_message(self):
        exc = MissingTaskFieldError("key", task_type="remember")
        self.assertIn("key", str(exc))
        self.assertIn("remember", str(exc))

    def test_rate_limit_exceeded_error(self):
        exc = RateLimitExceededError(limit=100, window_seconds=60)
        self.assertIn("100", str(exc))
        self.assertIn("60", str(exc))


# ---------------------------------------------------------------------------
# MetricsRegistry tests
# ---------------------------------------------------------------------------

class TestMetricsRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = MetricsRegistry(name="test")

    def test_increment_counter(self):
        self.reg.increment("hits")
        self.assertEqual(self.reg.counter("hits"), 1.0)

    def test_increment_by_amount(self):
        self.reg.increment("bytes", 512.0)
        self.assertEqual(self.reg.counter("bytes"), 512.0)

    def test_counter_starts_at_zero(self):
        self.assertEqual(self.reg.counter("new_counter"), 0.0)

    def test_gauge_set(self):
        self.reg.gauge_set("connections", 42.0)
        self.assertEqual(self.reg.gauge("connections"), 42.0)

    def test_gauge_increment_decrement(self):
        self.reg.gauge_inc("workers", 3.0)
        self.reg.gauge_dec("workers", 1.0)
        self.assertEqual(self.reg.gauge("workers"), 2.0)

    def test_observe_histogram(self):
        self.reg.observe("latency_ms", 10.0)
        self.reg.observe("latency_ms", 20.0)
        snap = self.reg.snapshot()
        hist = snap["histograms"]["latency_ms"]
        self.assertEqual(hist["count"], 2)
        self.assertAlmostEqual(hist["sum"], 30.0, places=4)
        self.assertAlmostEqual(hist["mean"], 15.0, places=4)
        self.assertAlmostEqual(hist["min"], 10.0, places=4)
        self.assertAlmostEqual(hist["max"], 20.0, places=4)

    def test_timing_alias(self):
        self.reg.timing("resp_ms", 5.5)
        snap = self.reg.snapshot()
        self.assertIn("resp_ms", snap["histograms"])

    def test_snapshot_has_uptime(self):
        snap = self.reg.snapshot()
        self.assertGreaterEqual(snap["uptime_seconds"], 0.0)

    def test_reset(self):
        self.reg.increment("x")
        self.reg.reset()
        self.assertEqual(self.reg.counter("x"), 0.0)

    def test_thread_safety(self):
        errors = []

        def worker():
            try:
                for _ in range(1000):
                    self.reg.increment("thread_counter")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(self.reg.counter("thread_counter"), 10_000.0)

    def test_get_registry_returns_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        self.assertIs(r1, r2)


# ---------------------------------------------------------------------------
# Config loader tests
# ---------------------------------------------------------------------------

class TestDeepMerge(unittest.TestCase):

    def test_shallow_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 99)

    def test_nested_merge(self):
        base = {"x": {"y": 1, "z": 2}}
        override = {"x": {"z": 99}}
        result = _deep_merge(base, override)
        self.assertEqual(result["x"]["y"], 1)
        self.assertEqual(result["x"]["z"], 99)

    def test_non_destructive(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        _deep_merge(base, override)
        self.assertNotIn("c", base["a"])


class TestLoadConfig(unittest.TestCase):

    def test_load_default_config(self):
        cfg = load_config()
        self.assertIn("model", cfg)
        self.assertIn("cognition", cfg)
        self.assertIn("memory", cfg)

    def test_overrides_applied(self):
        cfg = load_config(overrides={"cognition": {"max_iterations": 99}})
        self.assertEqual(cfg["cognition"]["max_iterations"], 99)

    def test_missing_file_returns_empty_config(self):
        cfg = load_config(path="/nonexistent/path/config.yaml")
        self.assertIsInstance(cfg, dict)

    def test_env_override(self):
        os.environ["VICTOR_COGNITION_MAX_ITERATIONS"] = "42"
        try:
            cfg = load_config()
            self.assertEqual(cfg["cognition"]["max_iterations"], 42)
        finally:
            del os.environ["VICTOR_COGNITION_MAX_ITERATIONS"]


if __name__ == "__main__":
    unittest.main()
