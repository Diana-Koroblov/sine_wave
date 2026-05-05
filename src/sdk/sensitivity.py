from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import src.shared.config as config
from src.utils.visuals import Visualizer

if TYPE_CHECKING:
    from src.sdk.interface import SignalDenoiserSDK


MODEL_TYPES = ("FC", "RNN", "LSTM")


def _build_reconstruction_snapshot(
    sdk: "SignalDenoiserSDK", metrics: dict[str, dict[str, float]], noise_level: float
) -> dict[str, Any]:
    best_model_type = min(metrics, key=lambda model_type: metrics[model_type]["mse"])
    test_inputs = sdk.dataset_splits["test"]["inputs"][:1]
    pure_signal = sdk.dataset_splits["test"]["targets"][0]
    noisy_signal = test_inputs[0, config.NUM_FREQUENCIES :]
    model = sdk.trained_models[best_model_type]
    with torch.no_grad():
        reconstructed_signal = model(torch.as_tensor(test_inputs, dtype=torch.float32)).cpu().numpy()[0]
    return {
        "model_type": best_model_type,
        "noise_level": noise_level,
        "noisy_signal": noisy_signal,
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
    snapshot: dict[str, Any] | None = None
    original_alpha = config.NOISE_ALPHA
    original_beta = config.NOISE_BETA

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
            snapshot = _build_reconstruction_snapshot(sdk, evaluation["metrics"], float(level))
    finally:
        config.NOISE_ALPHA = original_alpha
        config.NOISE_BETA = original_beta

    artifacts = {
        "sensitivity_mse": str(
            visualizer.plot_sensitivity_curve(levels, metrics_by_model, metric="mse")
        ),
        "loss_curves": str(visualizer.plot_loss_curves(latest_histories)),
    }
    if snapshot is not None:
        artifacts["reconstruction"] = str(
            visualizer.plot_reconstruction(
                snapshot["noisy_signal"],
                snapshot["pure_signal"],
                snapshot["reconstructed_signal"],
                filename=(
                    f"reconstruction_{snapshot['model_type'].lower()}_"
                    f"noise_{str(snapshot['noise_level']).replace('.', '_')}.png"
                ),
            )
        )

    return {
        "noise_levels": levels,
        "metrics": metrics_by_model,
        "artifacts": artifacts,
    }