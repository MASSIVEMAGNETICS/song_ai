"""Dataset Loader — utilities for loading and preprocessing training data."""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Generator, Iterator

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads training datasets from various file formats.

    Supported formats
    -----------------
    * JSON Lines (``.jsonl``) — one JSON object per line.
    * JSON (``.json``) — a single top-level list.
    * CSV (``.csv``) — standard comma-separated values.
    * Plain text (``.txt``) — one sample per line.

    Args:
        batch_size: Number of samples per batch when iterating.
        shuffle: Whether to shuffle data on load (not yet implemented for
            streaming sources).
    """

    def __init__(self, batch_size: int = 32, shuffle: bool = False) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        logger.info(
            "DatasetLoader initialised (batch_size=%d, shuffle=%s)",
            batch_size,
            shuffle,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> list[Any]:
        """Load an entire dataset into memory.

        Args:
            path: Path to the dataset file.

        Returns:
            List of parsed samples.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file extension is not supported.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        loaders = {
            ".jsonl": self._load_jsonl,
            ".json": self._load_json,
            ".csv": self._load_csv,
            ".txt": self._load_txt,
        }
        loader = loaders.get(ext)
        if loader is None:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {list(loaders.keys())}"
            )
        data = loader(path)
        logger.info("Loaded %d samples from '%s'", len(data), path)
        return data

    def stream(self, path: str) -> Iterator[Any]:
        """Lazily yield individual samples from *path*.

        Args:
            path: Path to the dataset file.

        Yields:
            Individual parsed samples.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".jsonl":
            yield from self._stream_jsonl(path)
        elif ext == ".txt":
            yield from self._stream_txt(path)
        else:
            yield from self.load(path)

    def batches(self, data: list[Any]) -> Generator[list[Any], None, None]:
        """Yield fixed-size batches from a pre-loaded list.

        Args:
            data: List of samples.

        Yields:
            Sub-lists of length up to ``batch_size``.
        """
        for i in range(0, len(data), self.batch_size):
            yield data[i : i + self.batch_size]

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: str) -> list[Any]:
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _load_json(path: str) -> list[Any]:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a top-level list.")
        return data

    @staticmethod
    def _load_csv(path: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
        return rows

    @staticmethod
    def _load_txt(path: str) -> list[str]:
        with open(path, encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh if line.strip()]

    @staticmethod
    def _stream_jsonl(path: str) -> Iterator[Any]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _stream_txt(path: str) -> Iterator[str]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                if stripped:
                    yield stripped
