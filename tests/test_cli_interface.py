"""Tests for interfaces.cli_interface."""

import io
import json
import unittest

from Victor_Synthetic_Super_Intelligence.agents.victor_agent import VictorAgent
from Victor_Synthetic_Super_Intelligence.interfaces.cli_interface import CLIInterface


def _make_cli(commands: list[str]) -> tuple[CLIInterface, io.StringIO]:
    """Create a CLI with pre-loaded commands and capture output."""
    stdin = io.StringIO("\n".join(commands) + "\n")
    stdout = io.StringIO()
    agent = VictorAgent()
    cli = CLIInterface(agent=agent, stream_in=stdin, stream_out=stdout)
    return cli, stdout


class TestCLIInterface(unittest.TestCase):

    def test_exit_command(self):
        cli, out = _make_cli(["exit"])
        cli.run()
        self.assertIn("Goodbye", out.getvalue())

    def test_quit_command(self):
        cli, out = _make_cli(["quit"])
        cli.run()
        self.assertIn("Goodbye", out.getvalue())

    def test_help_command(self):
        cli, out = _make_cli(["help", "exit"])
        cli.run()
        output = out.getvalue()
        self.assertIn("remember", output)
        self.assertIn("recall", output)

    def test_remember_and_recall(self):
        cli, out = _make_cli([
            "remember greeting Hello",
            "recall greeting",
            "exit",
        ])
        cli.run()
        output = out.getvalue()
        self.assertIn("Stored 'greeting'", output)
        self.assertIn("Hello", output)

    def test_forget_command(self):
        cli, out = _make_cli([
            "remember temp_key temp_value",
            "forget temp_key",
            "recall temp_key",
            "exit",
        ])
        cli.run()
        output = out.getvalue()
        self.assertIn("Deleted 'temp_key'", output)

    def test_forget_missing_key(self):
        cli, out = _make_cli(["forget nonexistent", "exit"])
        cli.run()
        self.assertIn("not found", out.getvalue())

    def test_recent_command_no_episodes(self):
        cli, out = _make_cli(["recent", "exit"])
        cli.run()
        output = out.getvalue()
        self.assertIn("No episodes", output)

    def test_recent_command_with_episodes(self):
        cli, out = _make_cli(["Hello world", "recent 1", "exit"])
        cli.run()
        output = out.getvalue()
        self.assertIn("Hello world", output)

    def test_search_command(self):
        cli, out = _make_cli([
            "remember search:key1 value1",
            "remember search:key2 value2",
            "search search:",
            "exit",
        ])
        cli.run()
        output = out.getvalue()
        self.assertIn("search:key1", output)
        self.assertIn("search:key2", output)

    def test_search_no_results(self):
        cli, out = _make_cli(["search zzz_no_match", "exit"])
        cli.run()
        self.assertIn("No keys matching", out.getvalue())

    def test_health_command(self):
        cli, out = _make_cli(["health", "exit"])
        cli.run()
        output = out.getvalue()
        self.assertIn("ok", output)

    def test_version_command(self):
        cli, out = _make_cli(["version", "exit"])
        cli.run()
        self.assertIn("Victor SSI", out.getvalue())

    def test_metrics_command(self):
        cli, out = _make_cli(["metrics", "exit"])
        cli.run()
        self.assertIn("counters", out.getvalue())

    def test_stimulus_produces_response(self):
        cli, out = _make_cli(["Hello Victor", "exit"])
        cli.run()
        output = out.getvalue()
        # The result dict should be JSON-formatted
        self.assertIn("result", output)

    def test_empty_lines_ignored(self):
        cli, out = _make_cli(["", "   ", "exit"])
        cli.run()
        self.assertIn("Goodbye", out.getvalue())


if __name__ == "__main__":
    unittest.main()
