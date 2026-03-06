"""Tests for core.reasoning_loop and core.cognition_engine."""

import unittest

from Victor_Synthetic_Super_Intelligence.core.tensor_operations import TensorOperations
from Victor_Synthetic_Super_Intelligence.core.reasoning_loop import ReasoningLoop
from Victor_Synthetic_Super_Intelligence.core.cognition_engine import CognitionEngine


class TestReasoningLoop(unittest.TestCase):

    def setUp(self):
        self.ops = TensorOperations()
        self.loop = ReasoningLoop(tensor_ops=self.ops, max_iterations=10)

    def test_run_returns_expected_keys(self):
        state = [0.6, 0.8]
        result = self.loop.run(state)
        self.assertIn("result", result)
        self.assertIn("iterations", result)
        self.assertIn("converged", result)

    def test_run_result_is_list_of_floats(self):
        result = self.loop.run([1.0, 0.0, 0.0])
        self.assertIsInstance(result["result"], list)
        self.assertTrue(all(isinstance(x, float) for x in result["result"]))

    def test_iterations_within_bounds(self):
        result = self.loop.run([0.5, 0.5])
        self.assertGreaterEqual(result["iterations"], 1)
        self.assertLessEqual(result["iterations"], 10)

    def test_converged_flag_type(self):
        result = self.loop.run([0.5, 0.5])
        self.assertIsInstance(result["converged"], bool)

    def test_l2_distance_equal_vectors(self):
        self.assertEqual(ReasoningLoop._l2_distance([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_l2_distance_mismatched_lengths(self):
        self.assertEqual(ReasoningLoop._l2_distance([1.0], [1.0, 2.0]), float("inf"))

    def test_l2_distance_known_value(self):
        dist = ReasoningLoop._l2_distance([0.0, 0.0], [3.0, 4.0])
        self.assertAlmostEqual(dist, 5.0, places=6)

    def test_convergence_threshold_respected(self):
        loop = ReasoningLoop(
            tensor_ops=self.ops,
            max_iterations=100,
            convergence_threshold=100.0,  # always converge immediately
        )
        result = loop.run([0.5, 0.5])
        self.assertEqual(result["iterations"], 1)
        self.assertTrue(result["converged"])

    def test_run_with_context(self):
        ctx = {"recent_episodes": []}
        result = self.loop.run([0.5, 0.5], context=ctx)
        self.assertIn("result", result)


class TestCognitionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = CognitionEngine(config={"max_iterations": 5})

    def test_perceive_string(self):
        rep = self.engine.perceive("Hello")
        self.assertIsInstance(rep, list)
        self.assertTrue(len(rep) > 0)

    def test_perceive_returns_normalised_vector(self):
        import math
        rep = self.engine.perceive("test")
        norm = math.sqrt(sum(x * x for x in rep))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_process_returns_dict(self):
        result = self.engine.process("Hello, world!")
        self.assertIsInstance(result, dict)

    def test_process_result_keys(self):
        result = self.engine.process("test")
        self.assertIn("result", result)
        self.assertIn("iterations", result)
        self.assertIn("converged", result)
        self.assertIn("timing", result)

    def test_process_timing_keys(self):
        result = self.engine.process("timing test")
        timing = result["timing"]
        self.assertIn("perceive", timing)
        self.assertIn("reason", timing)
        self.assertIn("total", timing)
        self.assertGreaterEqual(timing["total"], 0.0)

    def test_last_timing_updated(self):
        self.engine.process("first")
        first_timing = self.engine.last_timing.copy()
        self.engine.process("second")
        # last_timing should be overwritten
        self.assertIsNotNone(self.engine.last_timing)
        self.assertGreater(len(self.engine.last_timing), 0)

    def test_process_with_context(self):
        ctx = {"recent_episodes": []}
        result = self.engine.process("ctx stimulus", context=ctx)
        self.assertIn("result", result)

    def test_reason_method(self):
        rep = self.engine.perceive("abc")
        result = self.engine.reason(rep)
        self.assertIn("result", result)


if __name__ == "__main__":
    unittest.main()
