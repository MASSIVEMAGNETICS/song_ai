"""Tests for memory subsystem modules."""

import time
import threading
import unittest

from Victor_Synthetic_Super_Intelligence.memory.long_term_memory import LongTermMemory
from Victor_Synthetic_Super_Intelligence.memory.episodic_memory import EpisodicMemory, Episode
from Victor_Synthetic_Super_Intelligence.memory.vector_store import VectorStore
from Victor_Synthetic_Super_Intelligence.exceptions import VectorDimensionError


# ---------------------------------------------------------------------------
# LongTermMemory tests
# ---------------------------------------------------------------------------

class TestLongTermMemory(unittest.TestCase):

    def test_store_and_retrieve(self):
        ltm = LongTermMemory()
        ltm.store("k1", "hello")
        self.assertEqual(ltm.retrieve("k1"), "hello")

    def test_retrieve_missing_returns_none(self):
        ltm = LongTermMemory()
        self.assertIsNone(ltm.retrieve("no_such_key"))

    def test_delete_existing_key(self):
        ltm = LongTermMemory()
        ltm.store("k1", 42)
        self.assertTrue(ltm.delete("k1"))
        self.assertIsNone(ltm.retrieve("k1"))

    def test_delete_missing_key_returns_false(self):
        ltm = LongTermMemory()
        self.assertFalse(ltm.delete("ghost"))

    def test_search_by_substring(self):
        ltm = LongTermMemory()
        ltm.store("user:1:name", "Alice")
        ltm.store("user:2:name", "Bob")
        ltm.store("config:theme", "dark")
        results = ltm.search("user:")
        keys = [r[0] for r in results]
        self.assertIn("user:1:name", keys)
        self.assertIn("user:2:name", keys)
        self.assertNotIn("config:theme", keys)

    def test_fifo_eviction_when_full(self):
        ltm = LongTermMemory(max_entries=3)
        ltm.store("k1", 1)
        time.sleep(0.01)
        ltm.store("k2", 2)
        time.sleep(0.01)
        ltm.store("k3", 3)
        time.sleep(0.01)
        ltm.store("k4", 4)  # triggers eviction of k1
        self.assertIsNone(ltm.retrieve("k1"))
        self.assertEqual(ltm.retrieve("k4"), 4)

    def test_ttl_expiry(self):
        ltm = LongTermMemory(default_ttl=0.05)  # 50 ms
        ltm.store("temp", "value")
        self.assertEqual(ltm.retrieve("temp"), "value")
        time.sleep(0.1)
        self.assertIsNone(ltm.retrieve("temp"))

    def test_clear(self):
        ltm = LongTermMemory()
        ltm.store("a", 1)
        ltm.store("b", 2)
        count = ltm.clear()
        self.assertEqual(count, 2)
        self.assertEqual(len(ltm), 0)

    def test_stats(self):
        ltm = LongTermMemory(max_entries=100)
        ltm.store("x", 1)
        stats = ltm.stats()
        self.assertEqual(stats["total_entries"], 1)
        self.assertEqual(stats["max_entries"], 100)
        self.assertGreater(stats["utilisation_pct"], 0)

    def test_contains(self):
        ltm = LongTermMemory()
        ltm.store("present", True)
        self.assertIn("present", ltm)
        self.assertNotIn("absent", ltm)

    def test_thread_safety(self):
        """Multiple threads writing and reading should not corrupt state."""
        ltm = LongTermMemory(max_entries=10_000)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 100):
                    ltm.store(f"key:{i}", i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertGreater(len(ltm), 0)

    def test_retrieve_with_metadata(self):
        ltm = LongTermMemory()
        ltm.store("meta_key", "meta_val", metadata={"source": "test"})
        entry = ltm.retrieve_with_metadata("meta_key")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["value"], "meta_val")
        self.assertEqual(entry["metadata"]["source"], "test")


# ---------------------------------------------------------------------------
# EpisodicMemory tests
# ---------------------------------------------------------------------------

class TestEpisodicMemory(unittest.TestCase):

    def test_record_and_recent(self):
        em = EpisodicMemory(capacity=100)
        em.record("stim1", {"result": [0.1]})
        em.record("stim2", {"result": [0.2]})
        recent = em.recent(n=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0].stimulus, "stim1")
        self.assertEqual(recent[1].stimulus, "stim2")

    def test_recent_returns_at_most_n(self):
        em = EpisodicMemory(capacity=100)
        for i in range(10):
            em.record(f"s{i}", {})
        recent = em.recent(n=3)
        self.assertEqual(len(recent), 3)

    def test_capacity_circular_buffer(self):
        em = EpisodicMemory(capacity=3)
        for i in range(5):
            em.record(f"s{i}", {})
        self.assertEqual(len(em), 3)

    def test_search_by_stimulus(self):
        em = EpisodicMemory()
        em.record("hello world", {})
        em.record("goodbye", {})
        results = em.search("hello")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].stimulus, "hello world")

    def test_clear(self):
        em = EpisodicMemory()
        em.record("x", {})
        em.record("y", {})
        count = em.clear()
        self.assertEqual(count, 2)
        self.assertEqual(len(em), 0)

    def test_stats(self):
        em = EpisodicMemory(capacity=50)
        em.record("a", {})
        stats = em.stats()
        self.assertEqual(stats["total_episodes"], 1)
        self.assertEqual(stats["capacity"], 50)
        self.assertAlmostEqual(stats["utilisation_pct"], 2.0, places=1)

    def test_episode_to_dict(self):
        ep = Episode("stim", {"result": [1.0]}, metadata={"tag": "test"})
        d = ep.to_dict()
        self.assertEqual(d["stimulus"], "stim")
        self.assertEqual(d["metadata"]["tag"], "test")
        self.assertIn("timestamp", d)


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

class TestVectorStore(unittest.TestCase):

    def test_add_and_query(self):
        vs = VectorStore(dimension=3)
        vs.add("a", [1.0, 0.0, 0.0])
        vs.add("b", [0.0, 1.0, 0.0])
        results = vs.query([1.0, 0.0, 0.0], top_k=1)
        self.assertEqual(results[0][0], "a")
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_dimension_inferred(self):
        vs = VectorStore()
        vs.add("x", [1.0, 2.0, 3.0])
        self.assertEqual(vs.dimension, 3)

    def test_dimension_mismatch_raises(self):
        vs = VectorStore(dimension=2)
        with self.assertRaises(VectorDimensionError):
            vs.add("bad", [1.0, 2.0, 3.0])

    def test_empty_vector_raises(self):
        vs = VectorStore()
        with self.assertRaises(ValueError):
            vs.add("empty", [])

    def test_remove(self):
        vs = VectorStore(dimension=2)
        vs.add("k", [1.0, 0.0])
        self.assertTrue(vs.remove("k"))
        self.assertFalse(vs.remove("k"))
        self.assertNotIn("k", vs)

    def test_query_empty_store(self):
        vs = VectorStore()
        results = vs.query([1.0, 0.0], top_k=5)
        self.assertEqual(results, [])

    def test_query_zero_vector(self):
        vs = VectorStore(dimension=2)
        vs.add("x", [1.0, 0.0])
        results = vs.query([0.0, 0.0], top_k=5)
        self.assertEqual(results, [])

    def test_clear(self):
        vs = VectorStore(dimension=2)
        vs.add("a", [1.0, 0.0])
        vs.add("b", [0.0, 1.0])
        count = vs.clear()
        self.assertEqual(count, 2)
        self.assertEqual(len(vs), 0)

    def test_stats(self):
        vs = VectorStore(dimension=2)
        vs.add("x", [1.0, 0.0])
        stats = vs.stats()
        self.assertEqual(stats["total_vectors"], 1)
        self.assertEqual(stats["dimension"], 2)

    def test_thread_safety(self):
        vs = VectorStore(dimension=3)
        errors = []

        def adder(offset):
            try:
                for i in range(50):
                    vs.add(f"vec:{offset}:{i}", [float(i), float(offset), 0.0])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=adder, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_get_specific_vector(self):
        vs = VectorStore(dimension=2)
        vs.add("v1", [1.0, 0.0])
        result = vs.get("v1")
        self.assertEqual(result, [1.0, 0.0])
        self.assertIsNone(vs.get("missing"))


if __name__ == "__main__":
    unittest.main()
