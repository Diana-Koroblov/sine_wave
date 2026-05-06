from collections import OrderedDict

import numpy as np
import torch

import src.shared.config as config
from src.training.trainer import ModelTrainer


def make_dataset(size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple learnable mapping from 14 features to 10 outputs."""
    inputs = np.random.default_rng(0).normal(size=(size, config.INPUT_SIZE)).astype(np.float32)
    targets = inputs[:, config.NUM_FREQUENCIES :].copy()
    return inputs, targets


def test_train_epoch_updates_model_parameters():
    """Verify one training epoch performs optimization with Adam and MSE."""
    torch.manual_seed(0)
    inputs, targets = make_dataset()
    model = torch.nn.Linear(config.INPUT_SIZE, config.OUTPUT_SIZE)
    trainer = ModelTrainer(model, learning_rate=0.05)
    initial_weight = model.weight.detach().clone()

    loss = trainer.train_epoch(inputs, targets, batch_size=8)

    assert loss >= 0.0
    assert not torch.allclose(initial_weight, model.weight.detach())
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert isinstance(trainer.criterion, torch.nn.MSELoss)


def test_fit_runs_validation_and_restores_best_weights(monkeypatch):
    """Verify validation is tracked each epoch and the best weights are restored."""
    model = torch.nn.Linear(config.INPUT_SIZE, config.OUTPUT_SIZE, bias=False)
    trainer = ModelTrainer(model)
    dataset_splits = {
        "train": {
            "inputs": np.zeros((4, config.INPUT_SIZE), dtype=np.float32),
            "targets": np.zeros((4, config.OUTPUT_SIZE), dtype=np.float32),
        },
        "validation": {
            "inputs": np.zeros((4, config.INPUT_SIZE), dtype=np.float32),
            "targets": np.zeros((4, config.OUTPUT_SIZE), dtype=np.float32),
        },
    }
    validation_losses = iter([0.8, 0.2, 0.5])
    saved_states: dict[int, OrderedDict[str, torch.Tensor]] = {}
    epoch_counter = {"value": 0}

    def fake_train_epoch(*args, **kwargs):
        epoch_counter["value"] += 1
        with torch.no_grad():
            model.weight.fill_(float(epoch_counter["value"]))
        saved_states[epoch_counter["value"]] = OrderedDict(
            (name, tensor.detach().clone()) for name, tensor in model.state_dict().items()
        )
        return float(epoch_counter["value"])

    def fake_validate_epoch(*args, **kwargs):
        return next(validation_losses)

    monkeypatch.setattr(trainer, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "validate_epoch", fake_validate_epoch)

    history = trainer.fit(dataset_splits, epochs=3, batch_size=2)

    assert history["train_losses"] == [1.0, 2.0, 3.0]
    assert history["validation_losses"] == [0.8, 0.2, 0.5]
    assert history["best_validation_loss"] == 0.2
    assert history["best_epoch"] == 2
    assert trainer.best_model_state is not None
    assert torch.equal(model.weight.detach(), saved_states[2]["weight"])


def test_trainer_loss_decreases():
    """Verify the trainer can reduce validation loss on a learnable linear mapping."""
    torch.manual_seed(0)
    inputs, targets = make_dataset(size=96)
    model = torch.nn.Linear(config.INPUT_SIZE, config.OUTPUT_SIZE)
    trainer = ModelTrainer(model, learning_rate=0.05)

    initial_loss = trainer.validate_epoch(inputs, targets, batch_size=16)
    for _ in range(12):
        trainer.train_epoch(inputs, targets, batch_size=16)
    final_loss = trainer.validate_epoch(inputs, targets, batch_size=16)

    assert final_loss < initial_loss


def test_trainer_resource_tracking(monkeypatch):
    """Verify fit reports non-zero time and peak RAM metrics."""
    model = torch.nn.Linear(config.INPUT_SIZE, config.OUTPUT_SIZE)
    trainer = ModelTrainer(model)
    dataset_splits = {
        "train": {
            "inputs": np.zeros((4, config.INPUT_SIZE), dtype=np.float32),
            "targets": np.zeros((4, config.OUTPUT_SIZE), dtype=np.float32),
        },
        "validation": {
            "inputs": np.zeros((4, config.INPUT_SIZE), dtype=np.float32),
            "targets": np.zeros((4, config.OUTPUT_SIZE), dtype=np.float32),
        },
    }
    time_points = iter([10.0, 13.5])
    memory_points = iter([128.0, 140.0, 136.0])

    monkeypatch.setattr("src.training.trainer.time.perf_counter", lambda: next(time_points))
    monkeypatch.setattr(trainer, "_current_ram_mb", lambda: next(memory_points))
    monkeypatch.setattr(trainer, "train_epoch", lambda *args, **kwargs: 0.4)
    monkeypatch.setattr(trainer, "validate_epoch", lambda *args, **kwargs: 0.2)

    history = trainer.fit(dataset_splits, epochs=2, batch_size=2)

    assert history["total_time"] == 3.5
    assert history["peak_ram_mb"] == 140.0
