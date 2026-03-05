"""Long-Term Memory — persistent key/value knowledge store."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class LongTermMemory:
    """An in-memory, dictionary-backed store for persistent knowledge.

    In production this class is intended to be subclassed to replace the
    internal ``_store`` with a database backend (e.g. SQLite, Redis).

    Args:
        max_entries: Maximum number of entries to retain.  Oldest entries
            are evicted when the limit is exceeded (FIFO).
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self._store: dict[str, dict[str, Any]] = {}
        logger.info("LongTermMemory initialised (max_entries=%d)", max_entries)

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Persist *value* under *key*.

        Args:
            key: Unique string identifier for the memory entry.
            value: Arbitrary data to store.
            metadata: Optional metadata attached to the entry (e.g. source,
                importance score).
        """
        if len(self._store) >= self.max_entries:
            self._evict()
        self._store[key] = {
            "value": value,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        logger.debug("Stored key '%s'", key)

    def retrieve(self, key: str) -> Any:
        """Return the value stored under *key*, or ``None`` if not found.

        Args:
            key: The key to look up.

        Returns:
            Stored value or ``None``.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        return entry["value"]

    def delete(self, key: str) -> bool:
        """Remove a memory entry by key.

        Args:
            key: The key to remove.

        Returns:
            ``True`` if the key existed and was removed, ``False`` otherwise.
        """
        if key in self._store:
            del self._store[key]
            logger.debug("Deleted key '%s'", key)
            return True
        return False

    def search(self, query: str) -> list[tuple[str, Any]]:
        """Naive substring search over stored keys.

        Args:
            query: Substring to match against stored keys.

        Returns:
            List of ``(key, value)`` pairs whose keys contain *query*.
        """
        return [
            (k, v["value"])
            for k, v in self._store.items()
            if query in k
        ]

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """Remove the oldest entry to make room for a new one."""
        oldest_key = min(self._store, key=lambda k: self._store[k]["timestamp"])
        del self._store[oldest_key]
        logger.debug("Evicted oldest key '%s'", oldest_key)
