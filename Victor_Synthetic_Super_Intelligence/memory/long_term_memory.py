"""Long-Term Memory — persistent key/value knowledge store.

This module provides a thread-safe, in-memory knowledge store with FIFO
eviction, optional TTL expiry, and bulk-operation helpers.  It is designed
to be sub-classed for database-backed production deployments.

Example::

    ltm = LongTermMemory(max_entries=5_000, default_ttl=3600)
    ltm.store("user:42:name", "Alice", metadata={"source": "profile"})
    ltm.store("user:42:role", "admin")

    name = ltm.retrieve("user:42:name")   # "Alice"
    ltm.delete("user:42:role")
    results = ltm.search("user:42")       # [("user:42:name", "Alice")]
    stats = ltm.stats()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Sentinel used to distinguish "caller did not pass ttl" from "caller passed None".
_SENTINEL = object()


class LongTermMemory:
    """A thread-safe, dictionary-backed store for persistent knowledge.

    In production this class is intended to be subclassed to replace the
    internal ``_store`` with a database backend (e.g. SQLite, Redis,
    PostgreSQL).

    Args:
        max_entries: Maximum number of entries to retain.  Oldest entries
            are evicted when the limit is exceeded (FIFO based on insertion
            timestamp).
        default_ttl: Optional time-to-live in seconds applied to every new
            entry.  Expired entries are lazily removed on access.  Pass
            ``None`` (default) to disable TTL.
    """

    def __init__(
        self,
        max_entries: int = 10_000,
        default_ttl: float | None = None,
    ) -> None:
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        logger.info(
            "LongTermMemory initialised (max_entries=%d, default_ttl=%s)",
            max_entries,
            default_ttl,
        )

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
        ttl: Any = _SENTINEL,
    ) -> None:
        """Persist *value* under *key*.

        Args:
            key: Unique string identifier for the memory entry.
            value: Arbitrary data to store.
            metadata: Optional metadata attached to the entry (e.g. source,
                importance score).
            ttl: Time-to-live in seconds for this specific entry.  If
                omitted, ``default_ttl`` is used.  Pass ``0`` or a negative
                number to store without expiry regardless of the default.
        """
        effective_ttl: float | None
        if ttl is _SENTINEL:
            effective_ttl = self.default_ttl
        else:
            effective_ttl = ttl

        with self._lock:
            if len(self._store) >= self.max_entries and key not in self._store:
                self._evict()
            now = time.time()
            self._store[key] = {
                "value": value,
                "timestamp": now,
                "expires_at": (now + effective_ttl) if effective_ttl and effective_ttl > 0 else None,
                "metadata": metadata or {},
            }
        logger.debug("Stored key '%s'", key)

    def retrieve(self, key: str) -> Any:
        """Return the value stored under *key*, or ``None`` if not found or
        expired.

        Args:
            key: The key to look up.

        Returns:
            Stored value or ``None``.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._store[key]
                logger.debug("Key '%s' expired on retrieval", key)
                return None
            return entry["value"]

    def retrieve_with_metadata(self, key: str) -> dict[str, Any] | None:
        """Return the full entry dict including metadata and timestamps.

        Args:
            key: The key to look up.

        Returns:
            Entry dict (``value``, ``timestamp``, ``expires_at``,
            ``metadata``) or ``None`` if not found / expired.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._store[key]
                return None
            return dict(entry)

    def delete(self, key: str) -> bool:
        """Remove a memory entry by key.

        Args:
            key: The key to remove.

        Returns:
            ``True`` if the key existed and was removed, ``False`` otherwise.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                logger.debug("Deleted key '%s'", key)
                return True
        return False

    def search(self, query: str) -> list[tuple[str, Any]]:
        """Case-sensitive substring search over stored keys.

        Expired entries are skipped and lazily removed during the search.

        Args:
            query: Substring to match against stored keys.

        Returns:
            List of ``(key, value)`` pairs whose keys contain *query*,
            ordered by insertion time (oldest first).
        """
        results: list[tuple[str, Any]] = []
        expired_keys: list[str] = []
        with self._lock:
            for k, v in self._store.items():
                if self._is_expired(v):
                    expired_keys.append(k)
                    continue
                if query in k:
                    results.append((k, v["value"]))
            for k in expired_keys:
                del self._store[k]
        return results

    def clear(self) -> int:
        """Remove all entries from the store.

        Returns:
            The number of entries that were cleared.
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
        logger.info("LongTermMemory cleared (%d entries removed)", count)
        return count

    def purge_expired(self) -> int:
        """Explicitly remove all expired entries.

        Returns:
            The number of entries removed.
        """
        expired: list[str] = []
        with self._lock:
            for k, v in list(self._store.items()):
                if self._is_expired(v):
                    expired.append(k)
            for k in expired:
                del self._store[k]
        if expired:
            logger.debug("Purged %d expired entries", len(expired))
        return len(expired)

    def stats(self) -> dict[str, Any]:
        """Return operational statistics for this store.

        Returns:
            Dict with keys ``total_entries``, ``max_entries``,
            ``utilisation_pct``, and ``default_ttl``.
        """
        with self._lock:
            total = len(self._store)
        return {
            "total_entries": total,
            "max_entries": self.max_entries,
            "utilisation_pct": round(100.0 * total / self.max_entries, 2) if self.max_entries else 0.0,
            "default_ttl": self.default_ttl,
        }

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._store.keys()))

    def __contains__(self, key: object) -> bool:
        with self._lock:
            entry = self._store.get(key)  # type: ignore[arg-type]
            if entry is None:
                return False
            if self._is_expired(entry):
                del self._store[key]  # type: ignore[arg-type]
                return False
            return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_expired(entry: dict[str, Any]) -> bool:
        """Return ``True`` if the entry has passed its expiry time."""
        expires_at = entry.get("expires_at")
        return expires_at is not None and time.time() > expires_at

    def _evict(self) -> None:
        """Remove the oldest non-expired entry to make room for a new one.

        If all entries are expired they are all removed.
        """
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k]["timestamp"])
        del self._store[oldest_key]
        logger.debug("Evicted oldest key '%s'", oldest_key)
