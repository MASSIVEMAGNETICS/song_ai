"""Tests for interfaces.api_server."""

import io
import json
import threading
import time
import unittest
import urllib.request
import urllib.error

from Victor_Synthetic_Super_Intelligence.agents.victor_agent import VictorAgent
from Victor_Synthetic_Super_Intelligence.interfaces.api_server import APIServer, _RateLimiter


class TestRateLimiter(unittest.TestCase):

    def test_allows_within_limit(self):
        rl = _RateLimiter(limit=5, window_seconds=10)
        for _ in range(5):
            self.assertTrue(rl.is_allowed("1.2.3.4"))

    def test_blocks_over_limit(self):
        rl = _RateLimiter(limit=3, window_seconds=10)
        for _ in range(3):
            rl.is_allowed("1.2.3.4")
        self.assertFalse(rl.is_allowed("1.2.3.4"))

    def test_different_ips_are_independent(self):
        rl = _RateLimiter(limit=1, window_seconds=10)
        rl.is_allowed("1.1.1.1")
        self.assertFalse(rl.is_allowed("1.1.1.1"))
        self.assertTrue(rl.is_allowed("2.2.2.2"))

    def test_window_expiry(self):
        rl = _RateLimiter(limit=1, window_seconds=0.05)
        rl.is_allowed("host")
        self.assertFalse(rl.is_allowed("host"))
        time.sleep(0.1)
        self.assertTrue(rl.is_allowed("host"))


class TestAPIServerIntegration(unittest.TestCase):
    """Integration tests that start a real server on a free port."""

    @classmethod
    def setUpClass(cls):
        cls.agent = VictorAgent()
        cls.server = APIServer(
            agent=cls.agent,
            host="127.0.0.1",
            port=0,   # let OS pick a free port
            rate_limit=1000,
            rate_window=60,
        )

        # Patch HTTPServer to record the actual port
        from http.server import HTTPServer as _HTTPServer
        import socket

        handler = type(
            "_BoundHandler",
            (cls.server.__class__.__mro__[0],),  # dummy — we build it manually
            {},
        )

        # Build the server directly so we can read the port
        from Victor_Synthetic_Super_Intelligence.interfaces.api_server import _RequestHandler
        BoundHandler = type(
            "_BoundHandler",
            (_RequestHandler,),
            {
                "agent": cls.agent,
                "rate_limiter": cls.server._rate_limiter,
                "cors_origin": cls.server.cors_origin,
            },
        )
        cls._http_server = _HTTPServer(("127.0.0.1", 0), BoundHandler)
        cls.port = cls._http_server.server_address[1]

        cls._thread = threading.Thread(
            target=cls._http_server.serve_forever, daemon=True
        )
        cls._thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls._http_server.shutdown()

    def _get(self, path):
        req = urllib.request.Request(f"{self.base}{path}")
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())

    def _post(self, path, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health_endpoint(self):
        status, body = self._get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_version_endpoint(self):
        status, body = self._get("/version")
        self.assertEqual(status, 200)
        self.assertIn("version", body)

    def test_metrics_endpoint(self):
        status, body = self._get("/metrics")
        self.assertEqual(status, 200)
        self.assertIn("counters", body)

    def test_memory_stats_endpoint(self):
        status, body = self._get("/memory/stats")
        self.assertEqual(status, 200)
        self.assertIn("long_term", body)

    def test_not_found_get(self):
        try:
            status, body = self._get("/nonexistent")
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = json.loads(exc.read())
        self.assertEqual(status, 404)

    def test_respond_endpoint(self):
        status, body = self._post("/respond", {"stimulus": "Hello"})
        self.assertEqual(status, 200)
        self.assertIn("result", body)

    def test_task_endpoint_remember_recall(self):
        self._post("/task", {"type": "remember", "key": "srv_key", "value": "srv_val"})
        status, body = self._post("/task", {"type": "recall", "key": "srv_key"})
        self.assertEqual(status, 200)
        self.assertEqual(body["result"], "srv_val")

    def test_task_endpoint_unknown_type(self):
        status, body = self._post("/task", {"type": "invalid_task"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_task_endpoint_missing_type(self):
        status, body = self._post("/task", {"key": "value"})
        self.assertEqual(status, 400)

    def test_memory_remember_endpoint(self):
        status, body = self._post("/memory/remember", {"key": "api_key", "value": "api_val"})
        self.assertEqual(status, 200)
        self.assertEqual(body["key"], "api_key")

    def test_memory_remember_missing_key(self):
        status, body = self._post("/memory/remember", {"value": "no_key"})
        self.assertEqual(status, 400)

    def test_memory_recall_endpoint(self):
        self.agent.remember("recall_test", "recall_value")
        status, body = self._post("/memory/recall", {"key": "recall_test"})
        self.assertEqual(status, 200)
        self.assertEqual(body["value"], "recall_value")

    def test_response_has_request_id(self):
        status, body = self._get("/health")
        self.assertIn("request_id", body)

    def test_cors_headers_present(self):
        req = urllib.request.Request(f"{self.base}/health")
        with urllib.request.urlopen(req) as resp:
            self.assertIsNotNone(resp.headers.get("Access-Control-Allow-Origin"))


if __name__ == "__main__":
    unittest.main()
