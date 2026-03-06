"""Tests for core.tensor_operations module."""

import math
import unittest

from Victor_Synthetic_Super_Intelligence.core.tensor_operations import TensorOperations


class TestTensorOperationsEncode(unittest.TestCase):
    def setUp(self):
        self.ops = TensorOperations()

    def test_encode_string_returns_list_of_floats(self):
        result = self.ops.encode("AB")
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(x, float) for x in result))

    def test_encode_string_is_unit_length(self):
        result = self.ops.encode("Hello")
        norm = math.sqrt(sum(x * x for x in result))
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_encode_empty_string_returns_single_zero(self):
        result = self.ops.encode("")
        self.assertEqual(result, [0.0])

    def test_encode_integer(self):
        result = self.ops.encode(5)
        self.assertEqual(result, [1.0])  # normalised single-element

    def test_encode_float(self):
        result = self.ops.encode(0.0)
        self.assertEqual(result, [0.0])

    def test_encode_list(self):
        result = self.ops.encode([3.0, 4.0])
        self.assertAlmostEqual(result[0], 0.6, places=6)
        self.assertAlmostEqual(result[1], 0.8, places=6)

    def test_encode_unknown_type_returns_zero_vector(self):
        result = self.ops.encode(object())
        self.assertEqual(result, [0.0])


class TestTensorOperationsNormalize(unittest.TestCase):
    def setUp(self):
        self.ops = TensorOperations()

    def test_normalize_unit_vector(self):
        v = [3.0, 4.0]
        result = self.ops.normalize(v)
        self.assertAlmostEqual(result[0], 0.6, places=6)
        self.assertAlmostEqual(result[1], 0.8, places=6)

    def test_normalize_zero_vector(self):
        v = [0.0, 0.0, 0.0]
        result = self.ops.normalize(v)
        self.assertEqual(result, [0.0, 0.0, 0.0])


class TestTensorOperationsDotProduct(unittest.TestCase):
    def setUp(self):
        self.ops = TensorOperations()

    def test_dot_product_basic(self):
        result = self.ops.dot_product([1.0, 2.0], [3.0, 4.0])
        self.assertEqual(result, 11.0)

    def test_dot_product_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            self.ops.dot_product([1.0], [1.0, 2.0])


class TestTensorOperationsCosineSimilarity(unittest.TestCase):
    def setUp(self):
        self.ops = TensorOperations()

    def test_identical_vectors(self):
        v = [1.0, 0.0]
        self.assertAlmostEqual(self.ops.cosine_similarity(v, v), 1.0, places=6)

    def test_orthogonal_vectors(self):
        a, b = [1.0, 0.0], [0.0, 1.0]
        self.assertAlmostEqual(self.ops.cosine_similarity(a, b), 0.0, places=6)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(self.ops.cosine_similarity([0.0], [1.0]), 0.0)


class TestTensorOperationsSoftmax(unittest.TestCase):
    def setUp(self):
        self.ops = TensorOperations()

    def test_softmax_sums_to_one(self):
        result = self.ops.softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(result), 1.0, places=6)

    def test_softmax_max_gets_highest_probability(self):
        result = self.ops.softmax([0.0, 0.0, 10.0])
        self.assertGreater(result[2], 0.99)


class TestTensorOperationsMatmul(unittest.TestCase):
    def setUp(self):
        self.ops = TensorOperations()

    def test_matmul_basic(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        result = self.ops.matmul(a, b)
        self.assertEqual(result, [[19.0, 22.0], [43.0, 50.0]])

    def test_matmul_incompatible_dimensions(self):
        with self.assertRaises(ValueError):
            self.ops.matmul([[1.0, 2.0]], [[1.0], [2.0], [3.0]])


if __name__ == "__main__":
    unittest.main()
