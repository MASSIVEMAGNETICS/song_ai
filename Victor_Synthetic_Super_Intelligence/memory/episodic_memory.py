"""Episodic Memory — sequential record of agent experiences.

This module provides a thread-safe circular buffer that stores recent
agent experiences (:class:`Episode` objects) and supports retrieval by
recency or stimulus-content search.

Example::

    from Victor_Synthetic_Super_Intelligence.memory.episodic_memory import EpisodicMemory

    em = EpisodicMemory(capacity=500)
    ep = em.record(stimulus="Hello", response={"result": [0.1, 0.2]})
    recent = em.recent(n=10)
    matches = em.search("Hello")
    stats = em.stats()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Iterator

logger = logging.getLogger(__name__)


class Episode:
    """A single recorded experience.

    Attributes:
        stimulus: The raw input that triggered the episode.
        response: The agent's response or action.
        timestamp: Unix timestamp of the episode.
        metadata: Arbitrary additional annotations.
    """

    __slots__ = ("stimulus", "response", "timestamp", "metadata")

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
        """Serialise the episode to a plain dictionary.

        Returns:
            Dict with keys ``stimulus``, ``response``, ``timestamp``,
            and ``metadata``.
        """
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
    """Thread-safe ordered circular buffer that stores recent agent experiences.

    When the buffer is full the oldest episode is silently discarded to
    make room for the newest one.

    Args:
        capacity: Maximum number of episodes to retain before the oldest
            is overwritten.
    """

    def __init__(self, capacity: int = 1_000) -> None:
        self.capacity = capacity
        self._episodes: Deque[Episode] = deque(maxlen=capacity)
        self._lock = threading.RLock()
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
        with self._lock:
            self._episodes.append(episode)
        logger.debug("Recorded episode: %s", episode)
        return episode

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def recent(self, n: int = 10) -> list[Episode]:
        """Return the *n* most recent episodes (oldest first).

        Args:
            n: Maximum number of episodes to return.  If the buffer holds
               fewer than *n* episodes all are returned.

        Returns:
            List of :class:`Episode` objects in chronological order.
        """
        with self._lock:
            episodes = list(self._episodes)
        return episodes[-n:] if n < len(episodes) else episodes

    def search(self, query: str) -> list[Episode]:
        """Return all episodes whose stimulus string contains *query*.

        The comparison is case-sensitive and substring-based.

        Args:
            query: Substring to match against ``str(episode.stimulus)``.

        Returns:
            Matching episodes in chronological order (oldest first).
        """
        with self._lock:
            return [
                ep for ep in self._episodes
                if query in str(ep.stimulus)
            ]

    def clear(self) -> int:
        """Discard all recorded episodes.

        Returns:
            The number of episodes that were removed.
        """
        with self._lock:
            count = len(self._episodes)
            self._episodes.clear()
        logger.info("EpisodicMemory cleared (%d episodes removed)", count)
        return count

    def stats(self) -> dict[str, Any]:
        """Return operational statistics for this memory.

        Returns:
            Dict with keys ``total_episodes``, ``capacity``, and
            ``utilisation_pct``.
        """
        with self._lock:
            total = len(self._episodes)
        return {
            "total_episodes": total,
            "capacity": self.capacity,
            "utilisation_pct": round(100.0 * total / self.capacity, 2) if self.capacity else 0.0,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._episodes)

    def __iter__(self) -> Iterator[Episode]:
        with self._lock:
            return iter(list(self._episodes))
