from typing import TYPE_CHECKING, Any

import src.shared.config as config
from src.sdk.evaluation import evaluate_frequency_mse, evaluate_model, format_metrics_table
from src.utils.visuals import Visualizer

if TYPE_CHECKING:
    from src.sdk.interface import SignalDenoiserSDK


def evaluate_sdk_on_test_set(
    sdk: "SignalDenoiserSDK",
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    export_artifacts: bool = True,
) -> dict[str, Any]:
    """Train missing models, score the unseen test split, and optionally export figures."""
    if not sdk.dataset_splits:
        sdk.prepare_data()

    test_split = sdk.dataset_splits["test"]
    metrics_by_model = {}
    frequency_mse_by_model = {}
    for model_type in ("FC", "RNN", "LSTM"):
        if model_type not in sdk.trained_models:
            sdk.run_training(model_type, epochs=epochs, batch_size=batch_size)
        metrics_by_model[model_type] = evaluate_model(
            sdk.trained_models[model_type],
            test_split["inputs"],
            test_split["targets"],
        )
        frequency_mse_by_model[model_type] = evaluate_frequency_mse(
            sdk.trained_models[model_type],
            test_split["inputs"],
            test_split["targets"],
        )

    artifacts = {}
    if export_artifacts:
        artifacts["frequency_mse_comparison"] = str(
            Visualizer(config.ASSETS_PATH).plot_frequency_mse_comparison(
                config.FREQUENCIES,
                frequency_mse_by_model,
            )
        )

    return {
        "metrics": metrics_by_model,
        "frequency_mse": frequency_mse_by_model,
        "artifacts": artifacts,
        "summary_table": format_metrics_table(metrics_by_model),
    }


def generate_sdk_report(sdk: "SignalDenoiserSDK") -> str:
    """Generate a concise markdown report from the current test-set evaluation."""
    evaluation = sdk.evaluate_on_test_set()
    lines = [
        "# Signal Denoising Report",
        "",
        "## Overall Metrics",
        evaluation["summary_table"],
        "",
        "## Per-Frequency MSE",
    ]
    for model_type in ("FC", "RNN", "LSTM"):
        mse_values = evaluation["frequency_mse"][model_type]
        formatted_values = ", ".join(
            f"{frequency}Hz={mse:.6f}"
            for frequency, mse in zip(config.FREQUENCIES, mse_values, strict=True)
        )
        lines.append(f"- {model_type}: {formatted_values}")
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Frequency comparison: {evaluation['artifacts']['frequency_mse_comparison']}",
        ]
    )
    return "\n".join(lines)
