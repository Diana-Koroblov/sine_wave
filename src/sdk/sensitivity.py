import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import src.shared.config as config
from src.utils.visuals import Visualizer

if TYPE_CHECKING:
    from src.sdk.interface import SignalDenoiserSDK


MODEL_TYPES = ("FC", "RNN", "LSTM")
logger = logging.getLogger(__name__)


def _build_reconstruction_snapshot(
    sdk: "SignalDenoiserSDK", metrics: dict[str, dict[str, float]], noise_level: float
) -> dict[str, Any]:
    best_model_type = min(metrics, key=lambda model_type: metrics[model_type]["mse"])
    test_inputs = sdk.dataset_splits["test"]["inputs"][:1]
    pure_signal = sdk.dataset_splits["test"]["targets"][0]
    # We now pass the FULL input vector including OHE
    noisy_input = test_inputs[0]
    model = sdk.trained_models[best_model_type]
    with torch.no_grad():
        input_tensor = torch.as_tensor(test_inputs, dtype=torch.float32)
        reconstructed_signal = model(input_tensor).cpu().numpy()[0]
    return {
        "model_type": best_model_type,
        "noise_level": noise_level,
        "noisy_input": noisy_input,
        "pure_signal": pure_signal,
        "reconstructed_signal": reconstructed_signal,
    }


def run_sensitivity_analysis(
    sdk: "SignalDenoiserSDK", noise_levels: list[float] | None = None
) -> dict[str, Any]:
    levels = noise_levels or [round(float(level), 1) for level in np.arange(0.1, 1.0, 0.1)]
    visualizer = Visualizer(config.ASSETS_PATH)
    metrics_by_model = {model_type: [] for model_type in MODEL_TYPES}
    latest_histories: dict[str, dict[str, list[float]]] = {}
    original_alpha = config.NOISE_ALPHA
    original_beta = config.NOISE_BETA

    # Track artifact paths
    artifacts = {}

    try:
        for level in levels:
            config.NOISE_ALPHA = float(level)
            config.NOISE_BETA = float(level)
            sdk.dataset_splits = {}
            sdk.trained_models = {}
            sdk.training_runs = {}
            sdk.prepare_data(dataset_size=config.SENSITIVITY_DATASET_SIZE)
            evaluation = sdk.evaluate_on_test_set(
                epochs=config.SENSITIVITY_EPOCHS,
                batch_size=config.SENSITIVITY_BATCH_SIZE,
                export_artifacts=False,
            )
            for model_type, metrics in evaluation["metrics"].items():
                metrics_by_model[model_type].append({"noise_level": float(level), **metrics})
            latest_histories = {
                model_type: {
                    "train_losses": training_run["train_losses"],
                    "validation_losses": training_run["validation_losses"],
                }
                for model_type, training_run in sdk.training_runs.items()
            }
            _build_reconstruction_snapshot(sdk, evaluation["metrics"], float(level))

            # Special case: Robust visualizations at 0.5 and 0.9 noise for ALL frequencies
            if level in (0.5, 0.9):
                for model_type in MODEL_TYPES:
                    model = sdk.trained_models[model_type]
                    test_inputs = sdk.dataset_splits["test"]["inputs"]
                    test_targets = sdk.dataset_splits["test"]["targets"]

                    for freq_idx in range(config.NUM_FREQUENCIES):
                        # Find the first example of this frequency class
                        found_idx = -1
                        for i, x_in in enumerate(test_inputs):
                            if np.argmax(x_in[: config.NUM_FREQUENCIES]) == freq_idx:
                                found_idx = i
                                break

                        if found_idx != -1:
                            x_in = test_inputs[found_idx]
                            y_true = test_targets[found_idx]
                            with torch.no_grad():
                                t_in = torch.as_tensor(x_in[None, :], dtype=torch.float32)
                                y_pred = model(t_in).cpu().numpy()[0]

                            n_str = str(level).replace(".", "_")
                            fname = (
                                f"reconstruction_{model_type.lower()}_"
                                f"freq{freq_idx}_noise{n_str}.png"
                            )
                            visualizer.plot_reconstruction(
                                x_in, y_true, y_pred, model_name=model_type, filename=fname
                            )

        # Cleanup old ambiguous assets
        for old_file in ["reconstruction_fc_noise_0_9.png", "reconstruction_lstm_noise_0_9.png"]:
            old_path = Path(config.ASSETS_PATH) / old_file
            if old_path.exists():
                old_path.unlink()
                logger.info("Deleted ambiguous asset: %s", old_path)

    finally:
        config.NOISE_ALPHA = original_alpha
        config.NOISE_BETA = original_beta

    artifacts.update(
        {
            "sensitivity_mse": str(
                visualizer.plot_sensitivity_curve(levels, metrics_by_model, metric="mse")
            ),
            "loss_curves": str(visualizer.plot_loss_curves(latest_histories)),
        }
    )

    return {
        "noise_levels": levels,
        "metrics": metrics_by_model,
        "artifacts": artifacts,
    }
