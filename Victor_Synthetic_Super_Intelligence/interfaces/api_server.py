"""API Server — production-ready HTTP interface for VictorAgent.

Provides a thread-safe, enterprise-grade HTTP REST server with:

* Security headers (X-Content-Type-Options, X-Frame-Options, …)
* CORS support (configurable allowed origins)
* Per-IP rate limiting (configurable window and limit)
* Structured JSON responses with request IDs
* Endpoints: ``/health``, ``/version``, ``/metrics``, ``/respond``,
  ``/task``, ``/memory/stats``
* Graceful error handling with appropriate HTTP status codes

Example::

    server = APIServer(host="127.0.0.1", port=9000)
    server.start()   # blocking

    # Or run from CLI:
    # python -m Victor_Synthetic_Super_Intelligence.interfaces.api_server --port 9000
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from ..agents.victor_agent import VictorAgent
from ..exceptions import (
    MissingTaskFieldError,
    RateLimitExceededError,
    UnknownTaskTypeError,
    VictorError,
)
from .. import __version__

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 8080
_DEFAULT_RATE_LIMIT = 60          # requests per window
_DEFAULT_RATE_WINDOW = 60         # seconds
_CORS_ALLOWED_ORIGINS = "*"


class _RateLimiter:
    """Simple sliding-window per-IP rate limiter.

    Args:
        limit: Maximum requests allowed within the window.
        window_seconds: Length of the rolling window in seconds.
    """

    def __init__(self, limit: int = _DEFAULT_RATE_LIMIT, window_seconds: int = _DEFAULT_RATE_WINDOW) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        """Return ``True`` if the client is within the rate limit.

        Args:
            client_ip: Identifier for the client (IP address string).

        Returns:
            ``True`` if the request should be allowed, ``False`` if the
            client has exceeded the rate limit.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            dq = self._requests[client_ip]
            # Evict timestamps outside the window
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.limit:
                return False
            dq.append(now)
            return True


class _RequestHandler(BaseHTTPRequestHandler):
    """Handles incoming HTTP requests and delegates to the agent.

    Class attributes injected by :class:`APIServer`:
        agent: The :class:`~agents.VictorAgent` instance.
        rate_limiter: The :class:`_RateLimiter` instance.
        cors_origin: CORS ``Access-Control-Allow-Origin`` header value.
    """

    agent: VictorAgent
    rate_limiter: _RateLimiter
    cors_origin: str

    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        logger.debug(fmt, *args)

    # ------------------------------------------------------------------
    # HTTP method handlers
    # ------------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle pre-flight CORS requests."""
        self._send_cors_preflight()

    def do_GET(self) -> None:  # noqa: N802
        request_id = self._new_request_id()
        client_ip = self.client_address[0]

        if not self._check_rate_limit(client_ip, request_id):
            return

        if self.path == "/health":
            health = self.agent.health()
            self._respond(200, health, request_id=request_id)
        elif self.path == "/version":
            self._respond(200, {"version": __version__, "name": "Victor SSI"}, request_id=request_id)
        elif self.path == "/metrics":
            from ..metrics import get_registry
            snapshot = get_registry().snapshot()
            self._respond(200, snapshot, request_id=request_id)
        elif self.path == "/memory/stats":
            stats = {
                "long_term": self.agent.long_term_memory.stats(),
                "episodic": self.agent.episodic_memory.stats(),
                "vector_store": self.agent.vector_store.stats(),
            }
            self._respond(200, stats, request_id=request_id)
        else:
            self._respond(404, {"error": "not found", "path": self.path}, request_id=request_id)

    def do_POST(self) -> None:  # noqa: N802
        request_id = self._new_request_id()
        client_ip = self.client_address[0]

        if not self._check_rate_limit(client_ip, request_id):
            return

        body = self._read_body()

        if self.path == "/respond":
            stimulus = body.get("stimulus", "")
            try:
                result = self.agent.respond(stimulus)
                self._respond(200, result, request_id=request_id)
            except VictorError as exc:
                logger.warning("Agent error on /respond: %s", exc)
                self._respond(500, {"error": str(exc)}, request_id=request_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error on /respond")
                self._respond(500, {"error": "internal server error"}, request_id=request_id)

        elif self.path == "/task":
            try:
                result = self.agent.execute_task(body)
                self._respond(200, {"result": result}, request_id=request_id)
            except MissingTaskFieldError as exc:
                self._respond(400, {"error": str(exc)}, request_id=request_id)
            except UnknownTaskTypeError as exc:
                self._respond(400, {"error": str(exc), "registered_types": exc.registered}, request_id=request_id)
            except VictorError as exc:
                self._respond(500, {"error": str(exc)}, request_id=request_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error on /task")
                self._respond(500, {"error": "internal server error"}, request_id=request_id)

        elif self.path == "/memory/remember":
            key = body.get("key")
            value = body.get("value")
            if not key:
                self._respond(400, {"error": "missing required field 'key'"}, request_id=request_id)
                return
            self.agent.remember(key, value, metadata=body.get("metadata"))
            self._respond(200, {"status": "stored", "key": key}, request_id=request_id)

        elif self.path == "/memory/recall":
            key = body.get("key")
            if not key:
                self._respond(400, {"error": "missing required field 'key'"}, request_id=request_id)
                return
            value = self.agent.recall(key)
            self._respond(200, {"key": key, "value": value}, request_id=request_id)

        else:
            self._respond(404, {"error": "not found", "path": self.path}, request_id=request_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_rate_limit(self, client_ip: str, request_id: str) -> bool:
        """Return ``True`` if the request is within the rate limit, else
        respond with HTTP 429 and return ``False``."""
        if not self.rate_limiter.is_allowed(client_ip):
            self._respond(
                429,
                {
                    "error": "rate limit exceeded",
                    "limit": self.rate_limiter.limit,
                    "window_seconds": self.rate_limiter.window_seconds,
                },
                request_id=request_id,
            )
            logger.warning("Rate limit exceeded for %s", client_ip)
            return False
        return True

    def _read_body(self) -> dict[str, Any]:
        """Read and JSON-decode the request body."""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _respond(
        self,
        status: int,
        payload: Any,
        request_id: str | None = None,
    ) -> None:
        """Send a JSON response with security and CORS headers."""
        if isinstance(payload, dict) and request_id:
            payload = {"request_id": request_id, **payload}
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        # Content headers
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Security headers
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Request-ID", request_id or "")
        self.send_header("Cache-Control", "no-store")
        # CORS headers
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_preflight(self) -> None:
        """Respond to an OPTIONS pre-flight request."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    @staticmethod
    def _new_request_id() -> str:
        """Generate a unique request ID (UUID4)."""
        return str(uuid.uuid4())


class APIServer:
    """Production-ready HTTP server wrapper around :class:`~agents.VictorAgent`.

    Args:
        agent: The agent instance to expose.  A default agent is created
            if none is supplied.
        host: Bind address (default: ``"0.0.0.0"``).
        port: Port number (default: ``8080``).
        rate_limit: Maximum requests per client per ``rate_window`` seconds.
        rate_window: Rolling window duration for rate limiting, in seconds.
        cors_origin: Value for the ``Access-Control-Allow-Origin`` response
            header (default: ``"*"``).
    """

    def __init__(
        self,
        agent: VictorAgent | None = None,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        rate_limit: int = _DEFAULT_RATE_LIMIT,
        rate_window: int = _DEFAULT_RATE_WINDOW,
        cors_origin: str = _CORS_ALLOWED_ORIGINS,
    ) -> None:
        self.agent = agent or VictorAgent()
        self.host = host
        self.port = port
        self.cors_origin = cors_origin
        self._rate_limiter = _RateLimiter(limit=rate_limit, window_seconds=rate_window)
        self._server: HTTPServer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the HTTP server (blocking).

        The server handles one request at a time on the calling thread.
        For concurrent workloads consider running multiple instances behind
        a load balancer.
        """
        handler = type(
            "_BoundHandler",
            (_RequestHandler,),
            {
                "agent": self.agent,
                "rate_limiter": self._rate_limiter,
                "cors_origin": self.cors_origin,
            },
        )
        self._server = HTTPServer((self.host, self.port), handler)
        logger.info("APIServer listening on %s:%d", self.host, self.port)
        self._server.serve_forever()

    def stop(self) -> None:
        """Shut down the server gracefully if it is running."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            logger.info("APIServer stopped")


def main() -> None:
    """Entry point for running the API server from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Victor SSI — Production HTTP API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=_DEFAULT_HOST, help="Bind address")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help="Port number")
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=_DEFAULT_RATE_LIMIT,
        dest="rate_limit",
        help="Max requests per client per rate window",
    )
    parser.add_argument(
        "--rate-window",
        type=int,
        default=_DEFAULT_RATE_WINDOW,
        dest="rate_window",
        help="Rate-limit window duration in seconds",
    )
    parser.add_argument(
        "--cors-origin",
        default=_CORS_ALLOWED_ORIGINS,
        dest="cors_origin",
        help="CORS Access-Control-Allow-Origin value",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        dest="log_level",
        help="Logging verbosity",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    server = APIServer(
        host=args.host,
        port=args.port,
        rate_limit=args.rate_limit,
        rate_window=args.rate_window,
        cors_origin=args.cors_origin,
    )
    server.start()


if __name__ == "__main__":
    main()
