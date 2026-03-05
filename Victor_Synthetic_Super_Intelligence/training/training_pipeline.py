"""Training Pipeline — orchestrates data loading, optimization, and checkpointing."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

from .dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Coordinates end-to-end training runs.

    The pipeline is deliberately framework-agnostic.  Heavy lifting is
    delegated to the caller-supplied *train_step* function, which receives
    a batch of samples and returns a scalar loss value.

    Args:
        config: Training configuration dict.  Recognised keys:

            ``epochs`` : int
                Number of passes over the dataset (default: 1).

            ``batch_size`` : int
                Samples per training batch (default: 32).

            ``checkpoint_dir`` : str
                Directory for saving checkpoints (default: ``"checkpoints"``).

            ``log_interval`` : int
                Log progress every N batches (default: 10).

        train_step: A callable ``(batch) -> float`` executed on every batch.
            Defaults to a no-op that returns ``0.0``.

        eval_step: An optional callable ``(batch) -> float`` for validation.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        train_step: Callable[[list[Any]], float] | None = None,
        eval_step: Callable[[list[Any]], float] | None = None,
    ) -> None:
        self.config: dict[str, Any] = config or {}
        self.epochs: int = self.config.get("epochs", 1)
        self.batch_size: int = self.config.get("batch_size", 32)
        self.checkpoint_dir: str = self.config.get("checkpoint_dir", "checkpoints")
        self.log_interval: int = self.config.get("log_interval", 10)
        self.train_step: Callable[[list[Any]], float] = train_step or (lambda batch: 0.0)
        self.eval_step: Callable[[list[Any]], float] | None = eval_step
        self.loader = DatasetLoader(batch_size=self.batch_size)
        self.history: list[dict[str, Any]] = []
        logger.info("TrainingPipeline initialised: epochs=%d, batch_size=%d", self.epochs, self.batch_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        train_path: str,
        eval_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the full training run.

        Args:
            train_path: Path to the training dataset file.
            eval_path: Optional path to an evaluation dataset file.

        Returns:
            Training history: a list of per-epoch dicts containing
            ``epoch``, ``train_loss``, ``eval_loss`` (if applicable),
            and ``duration_s``.
        """
        train_data = self.loader.load(train_path)
        eval_data = self.loader.load(eval_path) if eval_path else None

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            train_loss = self._train_epoch(train_data, epoch=epoch)
            eval_loss = self._eval_epoch(eval_data) if eval_data is not None else None
            duration = time.time() - epoch_start

            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "duration_s": round(duration, 4),
            }
            if eval_loss is not None:
                record["eval_loss"] = eval_loss

            self.history.append(record)
            logger.info("Epoch %d/%d — %s", epoch, self.epochs, record)
            self._save_checkpoint(epoch, record)

        return self.history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, data: list[Any], epoch: int) -> float:
        """Iterate over all training batches for one epoch."""
        total_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(self.loader.batches(data)):
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            if (batch_idx + 1) % self.log_interval == 0:
                logger.debug(
                    "Epoch %d — batch %d — loss=%.6f",
                    epoch,
                    batch_idx + 1,
                    loss,
                )
        return total_loss / num_batches if num_batches else 0.0

    def _eval_epoch(self, data: list[Any]) -> float:
        """Run evaluation over all batches and return mean loss."""
        if self.eval_step is None:
            return 0.0
        total_loss = 0.0
        num_batches = 0
        for batch in self.loader.batches(data):
            total_loss += self.eval_step(batch)
            num_batches += 1
        return total_loss / num_batches if num_batches else 0.0

    def _save_checkpoint(self, epoch: int, record: dict[str, Any]) -> None:
        """Persist a JSON checkpoint for the given epoch."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.json"
        )
        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        logger.debug("Saved checkpoint: %s", checkpoint_path)
