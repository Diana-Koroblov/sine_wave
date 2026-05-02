import numpy as np
import pytest
import torch

import src.shared.config as config
from src.sdk.interface import SignalDenoiserSDK


class DummyModel(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, config.NUM_FREQUENCIES :] * self.scale


def test_evaluate_on_test_set_returns_metrics_and_summary_table(monkeypatch):
    """Verify test-set evaluation scores all three models and formats a markdown summary."""
    sdk = SignalDenoiserSDK()
    base_window = np.arange(config.WINDOW_SIZE, dtype=np.float32)
    test_inputs = np.tile(
        np.concatenate([np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), base_window]),
        (3, 1),
    )
    test_targets = np.tile(base_window, (3, 1))
    empty_split = {
        "inputs": np.zeros((1, config.INPUT_SIZE), dtype=np.float32),
        "targets": np.zeros((1, config.OUTPUT_SIZE), dtype=np.float32),
    }
    sdk.dataset_splits = {
        "train": empty_split,
        "validation": empty_split,
        "test": {"inputs": test_inputs, "targets": test_targets},
    }
    scales = {"FC": 1.0, "RNN": 0.5, "LSTM": 0.0}
    run_calls = []

    def fake_run_training(model_type: str):
        run_calls.append(model_type)
        sdk.trained_models[model_type] = DummyModel(scales[model_type])
        return {"model_type": model_type}

    monkeypatch.setattr(sdk, "run_training", fake_run_training)

    evaluation = sdk.evaluate_on_test_set()

    assert run_calls == ["FC", "RNN", "LSTM"]
    assert evaluation["metrics"]["FC"]["mse"] == pytest.approx(0.0)
    assert evaluation["metrics"]["FC"]["mae"] == pytest.approx(0.0)
    assert evaluation["metrics"]["FC"]["pearson_correlation"] == pytest.approx(1.0)
    assert evaluation["metrics"]["RNN"]["mse"] > 0.0
    assert evaluation["metrics"]["LSTM"]["mae"] > evaluation["metrics"]["RNN"]["mae"]
    assert "| Model | MSE | MAE | Pearson Correlation |" in evaluation["summary_table"]
    assert "| FC |" in evaluation["summary_table"]
    assert "| RNN |" in evaluation["summary_table"]
    assert "| LSTM |" in evaluation["summary_table"]


def test_run_training_rejects_unknown_model_type():
    """Verify the SDK surfaces unsupported model names clearly."""
    sdk = SignalDenoiserSDK()

    with pytest.raises(ValueError, match="Unsupported model_type"):
        sdk.run_training("cnn")