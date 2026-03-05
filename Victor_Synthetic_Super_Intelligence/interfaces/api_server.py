"""API Server — lightweight HTTP interface for VictorAgent."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from ..agents.victor_agent import VictorAgent

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 8080


class _RequestHandler(BaseHTTPRequestHandler):
    """Handles incoming HTTP requests and delegates to the agent."""

    agent: VictorAgent  # injected by APIServer

    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        logger.debug(fmt, *args)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/respond":
            body = self._read_body()
            stimulus = body.get("stimulus", "")
            result = self.agent.respond(stimulus)
            self._respond(200, result)
        elif self.path == "/task":
            body = self._read_body()
            try:
                result = self.agent.execute_task(body)
                self._respond(200, {"result": result})
            except (KeyError, NotImplementedError) as exc:
                self._respond(400, {"error": str(exc)})
        else:
            self._respond(404, {"error": "not found"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _respond(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class APIServer:
    """HTTP server wrapper around :class:`~agents.VictorAgent`.

    Args:
        agent: The agent instance to expose.
        host: Bind address (default: ``"0.0.0.0"``).
        port: Port number (default: ``8080``).
    """

    def __init__(
        self,
        agent: VictorAgent | None = None,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
    ) -> None:
        self.agent = agent or VictorAgent()
        self.host = host
        self.port = port
        self._server: HTTPServer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the HTTP server (blocking)."""
        handler = type(
            "_BoundHandler",
            (_RequestHandler,),
            {"agent": self.agent},
        )
        self._server = HTTPServer((self.host, self.port), handler)
        logger.info("APIServer listening on %s:%d", self.host, self.port)
        self._server.serve_forever()

    def stop(self) -> None:
        """Shut down the server if it is running."""
        if self._server is not None:
            self._server.shutdown()
            logger.info("APIServer stopped")


def main() -> None:
    """Entry point for running the API server from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Victor SSI API Server")
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = parser.parse_args()

    server = APIServer(host=args.host, port=args.port)
    server.start()


if __name__ == "__main__":
    main()
