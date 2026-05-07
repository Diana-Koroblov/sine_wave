from pathlib import Path

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
        return x[:, config.SIGNAL_START_INDEX :] * self.scale


def test_evaluate_on_test_set_returns_metrics_and_summary_table(monkeypatch, tmp_path: Path):
    """Verify test-set evaluation scores all three models and formats a markdown summary."""
    sdk = SignalDenoiserSDK()
    monkeypatch.setattr(config, "ASSETS_PATH", str(tmp_path))
    base_window = np.arange(config.WINDOW_SIZE, dtype=np.float32)
    test_inputs = np.tile(
        np.concatenate(
            [
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([config.NOISE_ALPHA], dtype=np.float32),
                base_window,
            ]
        ),
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

    def fake_run_training(
        model_type: str,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
    ):
        run_calls.append(model_type)
        assert epochs == config.EPOCHS
        assert batch_size == config.BATCH_SIZE
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
    assert len(evaluation["frequency_mse"]["FC"]) == config.NUM_FREQUENCIES
    assert Path(evaluation["artifacts"]["frequency_mse_comparison"]).exists()
    assert "| Model | MSE | MAE | Pearson Correlation |" in evaluation["summary_table"]
    assert "| FC |" in evaluation["summary_table"]
    assert "| RNN |" in evaluation["summary_table"]
    assert "| LSTM |" in evaluation["summary_table"]


def test_run_training_rejects_unknown_model_type():
    """Verify the SDK surfaces unsupported model names clearly."""
    sdk = SignalDenoiserSDK()

    with pytest.raises(ValueError, match="Unsupported model_type"):
        sdk.run_training("cnn")


def test_generate_report_returns_markdown_with_frequency_section(monkeypatch, tmp_path: Path):
    """Verify generate_report returns a non-empty markdown summary with exported artifacts."""
    sdk = SignalDenoiserSDK()
    monkeypatch.setattr(config, "ASSETS_PATH", str(tmp_path))

    def fake_evaluate_on_test_set(*args, **kwargs):
        return {
            "summary_table": "| Model | MSE |\n| --- | ---: |\n| FC | 0.1 |",
            "frequency_mse": {
                "FC": [0.1, 0.2, 0.3, 0.4],
                "RNN": [0.2, 0.3, 0.4, 0.5],
                "LSTM": [0.05, 0.1, 0.15, 0.2],
            },
            "artifacts": {
                "frequency_mse_comparison": str(tmp_path / "frequency_mse_comparison.png")
            },
        }

    monkeypatch.setattr(sdk, "evaluate_on_test_set", fake_evaluate_on_test_set)

    report = sdk.generate_report()

    assert "# Signal Denoising Report" in report
    assert "## Per-Frequency MSE" in report
    assert "FC: 25Hz=0.100000" in report
