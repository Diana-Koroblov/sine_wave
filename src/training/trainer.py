import time
from copy import deepcopy

import numpy as np
import psutil
import torch
from torch import nn, optim

import src.shared.config as config


class ModelTrainer:
    """Generic trainer for denoising models with train and validation loops."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = config.LEARNING_RATE,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_validation_loss = float("inf")
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.best_epoch: int | None = None

    def _current_ram_mb(self) -> float:
        """Return the current resident memory size in megabytes."""
        rss_bytes = psutil.Process().memory_info().rss
        return rss_bytes / (1024 * 1024)

    def _iter_batches(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ):
        indices = np.arange(len(inputs))
        if shuffle:
            indices = np.random.permutation(indices)

        for start_index in range(0, len(indices), batch_size):
            batch_indices = indices[start_index : start_index + batch_size]
            batch_inputs = torch.as_tensor(
                inputs[batch_indices], dtype=torch.float32, device=self.device
            )
            batch_targets = torch.as_tensor(
                targets[batch_indices],
                dtype=torch.float32,
                device=self.device,
            )
            yield batch_inputs, batch_targets

    def train_epoch(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        batch_size: int = config.BATCH_SIZE,
    ) -> float:
        """Run one optimization epoch and return the average MSE loss."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_inputs, batch_targets in self._iter_batches(
            train_inputs,
            train_targets,
            batch_size,
            shuffle=True,
        ):
            self.optimizer.zero_grad()
            predictions = self.model(batch_inputs)
            loss = self.criterion(predictions, batch_targets)
            loss.backward()
            self.optimizer.step()

            batch_size_actual = batch_inputs.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

        return total_loss / total_samples

    @torch.no_grad()
    def validate_epoch(
        self,
        validation_inputs: np.ndarray,
        validation_targets: np.ndarray,
        batch_size: int = config.BATCH_SIZE,
    ) -> float:
        """Evaluate one validation epoch and return the average MSE loss."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for batch_inputs, batch_targets in self._iter_batches(
            validation_inputs,
            validation_targets,
            batch_size,
            shuffle=False,
        ):
            predictions = self.model(batch_inputs)
            loss = self.criterion(predictions, batch_targets)

            batch_size_actual = batch_inputs.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

        return total_loss / total_samples

    def fit(
        self,
        dataset_splits: dict[str, dict[str, np.ndarray]],
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
    ) -> dict[str, list[float] | float | int | None]:
        """Train across epochs, validate after each epoch, and retain the best weights."""
        train_losses: list[float] = []
        validation_losses: list[float] = []
        start_time = time.perf_counter()
        peak_ram_mb = self._current_ram_mb()

        for epoch_index in range(epochs):
            train_loss = self.train_epoch(
                dataset_splits["train"]["inputs"],
                dataset_splits["train"]["targets"],
                batch_size=batch_size,
            )
            validation_loss = self.validate_epoch(
                dataset_splits["validation"]["inputs"],
                dataset_splits["validation"]["targets"],
                batch_size=batch_size,
            )
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            peak_ram_mb = max(peak_ram_mb, self._current_ram_mb())

            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                self.best_model_state = deepcopy(self.model.state_dict())
                self.best_epoch = epoch_index + 1

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        total_time = time.perf_counter() - start_time

        return {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "best_validation_loss": self.best_validation_loss,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
            "peak_ram_mb": peak_ram_mb,
        }
