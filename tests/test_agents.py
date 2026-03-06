"""Tests for agents.victor_agent and agents.task_executor."""

import unittest

from Victor_Synthetic_Super_Intelligence.agents.victor_agent import VictorAgent
from Victor_Synthetic_Super_Intelligence.agents.task_executor import TaskExecutor
from Victor_Synthetic_Super_Intelligence.exceptions import (
    MissingTaskFieldError,
    UnknownTaskTypeError,
)
from Victor_Synthetic_Super_Intelligence.metrics import MetricsRegistry


class TestVictorAgent(unittest.TestCase):

    def setUp(self):
        self.metrics = MetricsRegistry(name="test")
        self.agent = VictorAgent(metrics=self.metrics)

    def test_respond_returns_dict(self):
        result = self.agent.respond("Hello")
        self.assertIsInstance(result, dict)

    def test_respond_result_keys(self):
        result = self.agent.respond("test")
        self.assertIn("result", result)
        self.assertIn("iterations", result)
        self.assertIn("converged", result)

    def test_remember_and_recall(self):
        self.agent.remember("capital:uk", "London")
        self.assertEqual(self.agent.recall("capital:uk"), "London")

    def test_recall_missing_returns_none(self):
        self.assertIsNone(self.agent.recall("nonexistent_key"))

    def test_respond_increments_metric(self):
        self.agent.respond("metric test")
        self.assertGreater(self.metrics.counter("agent.respond.calls"), 0)

    def test_respond_records_latency(self):
        self.agent.respond("latency test")
        hist = self.metrics.snapshot()["histograms"]
        self.assertIn("agent.respond.latency_ms", hist)

    def test_remember_increments_metric(self):
        self.agent.remember("k", "v")
        self.assertGreater(self.metrics.counter("agent.remember.calls"), 0)

    def test_recall_increments_metric(self):
        self.agent.recall("k")
        self.assertGreater(self.metrics.counter("agent.recall.calls"), 0)

    def test_respond_stores_episode(self):
        initial_len = len(self.agent.episodic_memory)
        self.agent.respond("episode test")
        self.assertEqual(len(self.agent.episodic_memory), initial_len + 1)

    def test_health_returns_ok(self):
        health = self.agent.health()
        self.assertEqual(health["status"], "ok")
        self.assertIn("memory", health)
        self.assertIn("cognition", health)
        self.assertGreaterEqual(health["uptime_seconds"], 0.0)

    def test_health_memory_keys(self):
        health = self.agent.health()
        self.assertIn("long_term", health["memory"])
        self.assertIn("episodic", health["memory"])
        self.assertIn("vector_store", health["memory"])

    def test_execute_task_recall(self):
        self.agent.remember("foo", "bar")
        result = self.agent.execute_task({"type": "recall", "key": "foo"})
        self.assertEqual(result, "bar")

    def test_execute_task_remember(self):
        self.agent.execute_task({"type": "remember", "key": "x", "value": 99})
        self.assertEqual(self.agent.recall("x"), 99)

    def test_execute_task_respond(self):
        result = self.agent.execute_task({"type": "respond", "stimulus": "ping"})
        self.assertIn("result", result)

    def test_execute_task_missing_type_raises(self):
        with self.assertRaises(MissingTaskFieldError):
            self.agent.execute_task({"key": "value"})

    def test_execute_task_unknown_type_raises(self):
        with self.assertRaises(UnknownTaskTypeError):
            self.agent.execute_task({"type": "nonexistent_task"})

    def test_execute_task_health(self):
        result = self.agent.execute_task({"type": "health"})
        self.assertEqual(result["status"], "ok")

    def test_execute_task_memory_stats(self):
        result = self.agent.execute_task({"type": "memory_stats"})
        self.assertIn("long_term", result)
        self.assertIn("episodic", result)
        self.assertIn("vector_store", result)

    def test_execute_task_search_memory(self):
        self.agent.remember("test:search:1", "val1")
        self.agent.remember("test:search:2", "val2")
        results = self.agent.execute_task({"type": "search_memory", "query": "test:search"})
        self.assertEqual(len(results), 2)

    def test_config_applied(self):
        agent = VictorAgent(config={"max_ltm_entries": 500, "episodic_capacity": 50})
        self.assertEqual(agent.long_term_memory.max_entries, 500)
        self.assertEqual(agent.episodic_memory.capacity, 50)


class TestTaskExecutor(unittest.TestCase):

    def setUp(self):
        self.agent = VictorAgent()
        self.executor = self.agent.task_executor

    def test_registered_types_includes_builtins(self):
        types = TaskExecutor.registered_types()
        for expected in ("recall", "remember", "respond", "search_memory", "vector_query", "health", "memory_stats"):
            self.assertIn(expected, types)

    def test_unknown_task_error_has_type_info(self):
        with self.assertRaises(UnknownTaskTypeError) as ctx:
            self.executor.execute({"type": "nope"})
        self.assertEqual(ctx.exception.task_type, "nope")
        self.assertIsInstance(ctx.exception.registered, list)

    def test_missing_type_error_message(self):
        with self.assertRaises(MissingTaskFieldError) as ctx:
            self.executor.execute({})
        self.assertIn("type", str(ctx.exception))

    def test_vector_query_task(self):
        self.agent.vector_store.add("v1", [1.0, 0.0, 0.0])
        results = self.executor.execute({"type": "vector_query", "vector": [1.0, 0.0, 0.0], "top_k": 1})
        self.assertEqual(results[0][0], "v1")


if __name__ == "__main__":
    unittest.main()
