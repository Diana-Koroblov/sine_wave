from pathlib import Path

import numpy as np
import pytest
import torch

import src.sdk.sensitivity as sensitivity_module
import src.shared.config as config
from src.sdk.interface import SignalDenoiserSDK
from src.utils.visuals import Visualizer


def test_visualizer_exports_reconstruction_and_loss_curves(tmp_path: Path):
    """Verify plotting helpers save non-empty PNG files without crashing."""
    visualizer = Visualizer(tmp_path)
    # Input vector: 4 elements for OHE + 1 sigma + WINDOW_SIZE samples
    noisy_input = np.zeros(config.INPUT_SIZE, dtype=np.float32)
    noisy_input[0] = 1.0  # Freq Class 0
    noisy_input[config.SIGMA_INDEX] = config.NOISE_ALPHA
    pure_signal = np.linspace(0.0, 1.0, config.WINDOW_SIZE, dtype=np.float32)
    reconstructed_signal = pure_signal + 0.05

    reconstruction_path = visualizer.plot_reconstruction(
        noisy_input, pure_signal, reconstructed_signal, model_name="FC"
    )
    loss_curves_path = visualizer.plot_loss_curves(
        {"FC": {"train_losses": [0.8, 0.5], "validation_losses": [0.9, 0.6]}}
    )

    assert reconstruction_path.exists()
    assert reconstruction_path.suffix == ".png"
    assert reconstruction_path.stat().st_size > 0
    assert loss_curves_path.exists()
    assert loss_curves_path.stat().st_size > 0


def test_visualizer_exports_sensitivity_curve(tmp_path: Path):
    """Verify the noise sweep plot is exported successfully."""
    visualizer = Visualizer(tmp_path)

    output_path = visualizer.plot_sensitivity_curve(
        [0.1, 0.2],
        {
            "FC": [{"mse": 0.1}, {"mse": 0.2}],
            "RNN": [{"mse": 0.15}, {"mse": 0.25}],
        },
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_visualizer_exports_frequency_mse_comparison(tmp_path: Path):
    """Verify the per-frequency MSE comparison chart is exported successfully."""
    visualizer = Visualizer(tmp_path)

    output_path = visualizer.plot_frequency_mse_comparison(
        config.FREQUENCIES,
        {"FC": [0.1, 0.2, 0.3, 0.4], "LSTM": [0.08, 0.18, 0.2, 0.25]},
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


class EchoWindowModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, config.SIGNAL_START_INDEX :]


def test_run_sensitivity_analysis_collects_metrics_and_exports_graphs(monkeypatch, tmp_path: Path):
    """Verify the noise sweep collects metrics per level and restores config afterward."""
    sdk = SignalDenoiserSDK()
    base_window = np.arange(config.WINDOW_SIZE, dtype=np.float32)
    empty_split = {
        "inputs": np.zeros((2, config.INPUT_SIZE), dtype=np.float32),
        "targets": np.zeros((2, config.OUTPUT_SIZE), dtype=np.float32),
    }
    prepared_levels = []
    original_alpha = config.NOISE_ALPHA
    original_beta = config.NOISE_BETA

    def fake_prepare_data(dataset_size=config.DATASET_SIZE):
        prepared_levels.append((config.NOISE_ALPHA, config.NOISE_BETA))
        assert dataset_size == config.SENSITIVITY_DATASET_SIZE
        test_inputs = np.tile(
            np.concatenate(
                [
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.array([config.NOISE_ALPHA], dtype=np.float32),
                    base_window,
                ]
            ),
            (2, 1),
        )
        sdk.dataset_splits = {
            "train": empty_split,
            "validation": empty_split,
            "test": {"inputs": test_inputs, "targets": np.tile(base_window, (2, 1))},
        }
        return sdk.dataset_splits

    def fake_evaluate_on_test_set(
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        export_artifacts=True,
    ):
        assert epochs == config.SENSITIVITY_EPOCHS
        assert batch_size == config.SENSITIVITY_BATCH_SIZE
        assert export_artifacts is False
        for model_type in ("FC", "RNN", "LSTM"):
            sdk.trained_models[model_type] = EchoWindowModel()
            sdk.training_runs[model_type] = {
                "train_losses": [0.4, 0.2],
                "validation_losses": [0.5, 0.3],
            }
        return {
            "metrics": {
                "FC": {"mse": 0.1, "mae": 0.05, "pearson_correlation": 0.9},
                "RNN": {"mse": 0.2, "mae": 0.1, "pearson_correlation": 0.8},
                "LSTM": {"mse": 0.3, "mae": 0.15, "pearson_correlation": 0.7},
            }
        }

    monkeypatch.setattr(sdk, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(sdk, "evaluate_on_test_set", fake_evaluate_on_test_set)
    monkeypatch.setattr(sensitivity_module.config, "ASSETS_PATH", str(tmp_path))

    # Note: Using 0.1 and 0.4 ensures we don't trigger the 0.5/0.9 robust visualization logic,
    # keeping the test focused on the standard artifact dictionary structure.
    result = sdk.run_sensitivity_analysis(noise_levels=[0.1, 0.4])

    assert prepared_levels == [(0.1, 0.1), (0.4, 0.4)]
    assert result["noise_levels"] == [0.1, 0.4]
    assert len(result["metrics"]["FC"]) == 2
    assert Path(result["artifacts"]["sensitivity_mse"]).exists()
    assert Path(result["artifacts"]["loss_curves"]).exists()
    assert config.NOISE_ALPHA == pytest.approx(original_alpha)
    assert config.NOISE_BETA == pytest.approx(original_beta)
