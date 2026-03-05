"""Episodic Memory — sequential record of agent experiences."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque

logger = logging.getLogger(__name__)


class Episode:
    """A single recorded experience.

    Attributes:
        stimulus: The raw input that triggered the episode.
        response: The agent's response or action.
        timestamp: Unix timestamp of the episode.
        metadata: Arbitrary additional annotations.
    """

    def __init__(
        self,
        stimulus: Any,
        response: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.stimulus = stimulus
        self.response = response
        self.timestamp: float = time.time()
        self.metadata: dict[str, Any] = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialise the episode to a plain dictionary."""
        return {
            "stimulus": self.stimulus,
            "response": self.response,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"Episode(stimulus={self.stimulus!r}, "
            f"response={self.response!r}, "
            f"timestamp={self.timestamp:.3f})"
        )


class EpisodicMemory:
    """Ordered circular buffer that stores recent agent experiences.

    Args:
        capacity: Maximum number of episodes to retain before the oldest
            is overwritten.
    """

    def __init__(self, capacity: int = 1_000) -> None:
        self.capacity = capacity
        self._episodes: Deque[Episode] = deque(maxlen=capacity)
        logger.info("EpisodicMemory initialised (capacity=%d)", capacity)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        stimulus: Any,
        response: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Record a new episode.

        Args:
            stimulus: Input that produced the episode.
            response: Agent output / action.
            metadata: Optional annotation dict.

        Returns:
            The newly created :class:`Episode`.
        """
        episode = Episode(stimulus, response, metadata=metadata)
        self._episodes.append(episode)
        logger.debug("Recorded episode: %s", episode)
        return episode

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def recent(self, n: int = 10) -> list[Episode]:
        """Return the *n* most recent episodes (newest last).

        Args:
            n: Number of episodes to return.

        Returns:
            List of :class:`Episode` objects.
        """
        episodes = list(self._episodes)
        return episodes[-n:]

    def search(self, query: str) -> list[Episode]:
        """Return all episodes whose stimulus string contains *query*.

        Args:
            query: Substring to match.

        Returns:
            Matching episodes in chronological order.
        """
        results = [
            ep for ep in self._episodes
            if query in str(ep.stimulus)
        ]
        return results

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self):
        return iter(self._episodes)
