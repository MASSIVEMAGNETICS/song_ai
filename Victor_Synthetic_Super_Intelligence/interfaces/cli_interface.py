"""CLI Interface — interactive command-line front-end for VictorAgent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from ..agents.victor_agent import VictorAgent

logger = logging.getLogger(__name__)

_BANNER = """\
╔══════════════════════════════════════════════╗
║   Victor Synthetic Super Intelligence v0.1   ║
║   Type 'help' for commands, 'exit' to quit   ║
╚══════════════════════════════════════════════╝
"""

_HELP_TEXT = """\
Available commands
──────────────────
  <text>                 Send a stimulus and receive a response.
  remember <key> <val>   Store a value in long-term memory.
  recall <key>           Retrieve a value from long-term memory.
  recent [n]             Show n most recent episodes (default: 5).
  help                   Show this help text.
  exit / quit            Exit the CLI.
"""


class CLIInterface:
    """Read-eval-print loop (REPL) interface for :class:`~agents.VictorAgent`.

    Args:
        agent: Optional pre-configured agent.  A default agent is created
            if none is supplied.
        stream_in: Input stream (default: ``sys.stdin``).
        stream_out: Output stream (default: ``sys.stdout``).
    """

    def __init__(
        self,
        agent: VictorAgent | None = None,
        stream_in=None,
        stream_out=None,
    ) -> None:
        self.agent = agent or VictorAgent()
        self.stream_in = stream_in or sys.stdin
        self.stream_out = stream_out or sys.stdout

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive REPL loop."""
        self._print(_BANNER)
        while True:
            try:
                line = self._input("victor> ")
            except (EOFError, KeyboardInterrupt):
                self._print("\nGoodbye.")
                break

            line = line.strip()
            if not line:
                continue

            should_exit = self._dispatch(line)
            if should_exit:
                break

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, line: str) -> bool:
        """Process one line of input.

        Returns:
            ``True`` if the user wants to exit.
        """
        parts = line.split(maxsplit=2)
        command = parts[0].lower()

        if command in ("exit", "quit"):
            self._print("Goodbye.")
            return True

        if command == "help":
            self._print(_HELP_TEXT)
            return False

        if command == "remember" and len(parts) >= 3:
            self.agent.remember(parts[1], parts[2])
            self._print(f"Stored '{parts[1]}'.")
            return False

        if command == "recall" and len(parts) >= 2:
            value = self.agent.recall(parts[1])
            self._print(f"{parts[1]}: {value!r}")
            return False

        if command == "recent":
            n = int(parts[1]) if len(parts) >= 2 else 5
            episodes = self.agent.episodic_memory.recent(n=n)
            for ep in episodes:
                self._print(json.dumps(ep.to_dict(), default=str))
            return False

        # Default: treat the entire line as a stimulus.
        result = self.agent.respond(line)
        self._print(self._format_result(result))
        return False

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _input(self, prompt: str = "") -> str:
        self.stream_out.write(prompt)
        self.stream_out.flush()
        return self.stream_in.readline().rstrip("\n")

    def _print(self, text: str) -> None:
        self.stream_out.write(text + "\n")
        self.stream_out.flush()

    @staticmethod
    def _format_result(result: Any) -> str:
        if isinstance(result, dict):
            return json.dumps(result, default=str, indent=2)
        return str(result)


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Victor SSI — CLI Interface")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging verbosity.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    CLIInterface().run()


if __name__ == "__main__":
    main()
